/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Apr. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the main algorithm of iterative updates of viewpoint pose in
 *                   FlyCo.
 * Copyright    :    Copyright (c) 2026 Chen Feng and Guiyong Zheng.
 * License      :    PolyForm Noncommercial License 1.0.0
 *                   <https://polyformproject.org/licenses/noncommercial/1.0.0/>
 *
 *                   This software is released for noncommercial research and
 *                   educational use only. You may use, modify, and distribute
 *                   this software for noncommercial purposes, subject to the
 *                   terms of the PolyForm Noncommercial License 1.0.0.
 *
 *                   Commercial use, including use in commercial products,
 *                   commercial services, paid consulting, or internal business
 *                   operations, is prohibited without prior written permission
 *                   from the copyright holders.
 *
 *                   This software is provided "as is", without warranty of any
 *                   kind, express or implied.
 * Project      :    FlyCo: Foundation Model-Empowered Drones for Autonomous 3D Structure Scanning in Open-World Environments
 * Website      :    https://hkust-aerial-robotics.github.io/FC-Planner/
 *⭐⭐⭐*****************************************************************⭐⭐⭐*/

#include "viewpoint_manager/viewpoint_manager.h"

namespace flyco
{
  ViewpointManager::ViewpointManager()
  {
  }
  ViewpointManager::~ViewpointManager()
  {
  }

  void ViewpointManager::init(ros::NodeHandle &nh)
  {
    nh_ = nh;
    nh.param("viewpoint_manager/visible_range", visible_range, -1.0);
    nh.param("viewpoint_manager/viewpoints_distance", dist_vp, -1.0);
    nh.param("viewpoint_manager/fov_h", fov_h, -1.0);
    nh.param("viewpoint_manager/fov_w", fov_w, -1.0);
    nh.param("viewpoint_manager/pitch_upper", pitch_upper, -1.0);
    nh.param("viewpoint_manager/pitch_lower", pitch_lower, -1.0);
    nh.param("viewpoint_manager/zGround", zFlag, false);
    nh.param("viewpoint_manager/GroundPos", GroundZPos, -1.0);
    nh.param("viewpoint_manager/safeHeight", safeHeight, -1.0);
    nh.param("viewpoint_manager/safe_radius", safe_radius_vp, -1.0);
    nh.param("viewpoint_manager/attitude_type", attitude, string("null"));
    nh.param("viewpoint_manager/max_iter_num", max_iter_num, 0);
    nh.param("viewpoint_manager/pose_update", pose_update, true);
    nh.param("viewpoint_manager/vis_inf", res, -1.0);
    nh.param("hctraj/drone_radius", drone_radius_, -1.0);
    nh.param("viewpoint_manager/shrink_factor", shrink_factor_, 1.0);
    nh.param("viewpoint_manager/save_lower", save_lower_, 5);
    bev_seg_ = false;

    bev_sub_ = nh.subscribe("/prediction/bev_box", 1, &ViewpointManager::bevCallback, this);

    // Attitude settings
    if (attitude == "yaw")
    {
      pitch_upper = 0.0;
      pitch_lower = 0.0;
    }
    fov_base = min(fov_h, fov_w) * M_PI / 360.0;

    percep_utils_.reset(new PerceptionUtils);
    percep_utils_->setShrinkFactor(this->shrink_factor_);
    percep_utils_->init(nh);
    HCMap.reset(new SDFMap);

    ROS_INFO("\033[33m[ViewpointManager] Initialized!\033[34m");
  }

  void ViewpointManager::reset()
  {
    TS.reset();
    VPI.reset();
    MCI.reset();
    VPP.reset();
    if (this->scene_ != nullptr)
    {
      rtcReleaseScene(this->scene_);
      this->scene_ = nullptr;
    }

    map_cloud_kdtree_ = pcl::KdTreeFLANN<pcl::PointXYZ>();
  }

  void ViewpointManager::setMapPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input_map, SDFMap::Ptr& hcmap)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr new_model(new pcl::PointCloud<pcl::PointXYZ>);
    *new_model = *input_map;
    this->HCMap = hcmap;
    map_cloud_kdtree_.setInputCloud(new_model);
    raycaster_.reset(new RayCaster);
    raycaster_->setParams(HCMap->hcmp_->resolution_, HCMap->hcmp_->map_origin_);

    TS.map_flag_ = true;
    ROS_WARN("[ViewpointManager] Input map successfully!! size           = %ld.", new_model->points.size());
  }

  void ViewpointManager::setModel(pcl::PointCloud<pcl::PointXYZ>::Ptr input_model)
  {
    if (input_model->points.size() == 0)
    {
      ROS_ERROR("[ViewpointManager] Input model is empty!!");
      return;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr new_model(new pcl::PointCloud<pcl::PointXYZ>);
    *new_model = *input_model;
    model_ = new_model;

    MCI.cover_state.resize(model_->points.size(), false);
    MCI.cover_contrib.resize(model_->points.size(), 0);
    MCI.contrib_id.resize(model_->points.size(), -1);

    TS.model_flag_ = true;
    ROS_WARN("[ViewpointManager] Input model successfully!! size         = %ld.", model_->points.size());
  }

  void ViewpointManager::setNormals(map<Eigen::Vector3d, Eigen::Vector3d, Vector3dCompare0> &pt_normal_pairs)
  {
    if (pt_normal_pairs.size() == 0)
    {
      ROS_ERROR("[ViewpointManager] Input normal Fail!! Input normal is empty!!!");
      return;
    }
    TS.normal_flag_ = true;
    MCI.pt_normal_pairs = pt_normal_pairs;
    ROS_WARN("[ViewpointManager] Input model normals successfully!! size = %ld.", MCI.pt_normal_pairs.size());
  }

  void ViewpointManager::setInitViewpoints(pcl::PointCloud<pcl::PointNormal>::Ptr input_normal_vps)
  {
    if (input_normal_vps->points.size() == 0)
    {
      ROS_ERROR("[ViewpointManager] Input viewpoints Fail!! Input viewpoint is empty!!!");
      return;
    }
    VPI.vps_voxcount.resize(input_normal_vps->points.size(), 0);
    VPI.vps_contri.resize(input_normal_vps->points.size(), 0);
    VPI.vps_pose.resize(input_normal_vps->points.size(), 5);
    for (int i = 0; i < (int)input_normal_vps->points.size(); i++)
    {
      pcl::PointNormal nor_vp = input_normal_vps->points[i];
      Eigen::Vector3d pos(nor_vp.x, nor_vp.y, nor_vp.z);
      Eigen::Vector3d dir(nor_vp.normal_x, nor_vp.normal_y, nor_vp.normal_z);
      VPI.vps_pose.row(i) = getPose(i, pos, dir);
    }
    TS.viewpoints_flag_ = true;
    ROS_WARN("[ViewpointManager] Input viewpoints successfully!! size    = %ld.", VPI.vps_pose.rows());
  }

  void ViewpointManager::setAdp(adaptive_utils::Ptr &adp)
  {
    this->adp_ = adp;
    adp_flag = true;
  }

  void ViewpointManager::setScene(Eigen::MatrixXd& mesh_V, Eigen::MatrixXi& mesh_F)
  { 
    visibility_st::mesh2scene(mesh_V, mesh_F, this->scene_);

    return;
  }

  void ViewpointManager::updateViewpoints()
  {
    if (!TS.viewpoints_flag_)
    {
      ROS_WARN("[ViewpointManager] No input viewpoints!! Use normals generate viewpoints!!");
      pcl::PointCloud<pcl::PointNormal>::Ptr vps(new pcl::PointCloud<pcl::PointNormal>);
      for (int i = 0; i < (int)model_->points.size(); i++)
      {
        pcl::PointXYZ pt, vp;
        pt = model_->points[i];
        Eigen::Vector3d pt_vec = Eigen::Vector3d(pt.x, pt.y, pt.z);
        if (MCI.pt_normal_pairs.find(pt_vec) == MCI.pt_normal_pairs.end())
          continue;
        auto normal_dir = MCI.pt_normal_pairs.find(pt_vec)->second;

        vp.x = pt.x + dist_vp * normal_dir(0);
        vp.y = pt.y + dist_vp * normal_dir(1);
        vp.z = pt.z + dist_vp * normal_dir(2);
        if (!viewpointSafetyCheck(vp))
          continue;
        pcl::PointNormal vp_fov_normal;
        vp_fov_normal.x = vp.x;
        vp_fov_normal.y = vp.y;
        vp_fov_normal.z = vp.z;
        vp_fov_normal.normal_x = -normal_dir(0);
        vp_fov_normal.normal_y = -normal_dir(1);
        vp_fov_normal.normal_z = -normal_dir(2);
        vps->points.push_back(vp_fov_normal);
      }
      setInitViewpoints(vps);
    }

    if (!TS.isReady())
      return;

    vector<int> temp_vps_voxcount;
    /* viewpoints pruning */
    MCI.cover_state.clear();
    MCI.cover_state.resize(model_->points.size(), false);
    MCI.cover_contrib.clear();
    MCI.cover_contrib.resize(model_->points.size(), 0);
    MCI.contrib_id.clear();
    MCI.contrib_id.resize(model_->points.size(), -1);
    temp_vps_voxcount = VPI.vps_voxcount;
    VPI.vps_voxcount.clear();
    VPI.vps_voxcount.resize(model_->points.size(), 0);
    viewpointsPrune(VPI.vps_pose, temp_vps_voxcount);

    // compute visible voxel count
    int sum_seen = 0;
    for (auto final_vp : VPP.final_vps_)
    {
      sum_seen += final_vp.vox_count;
    }

    /* process uncovered area */
    vector<int> uncoveredID;
    for (int i = 0; i < (int)MCI.cover_state.size(); ++i)
    {
      if (MCI.cover_state[i] == false)
        uncoveredID.push_back(i);
    }

    /* generate viewpoints library for uncovered area */
    auto lib_t1 = std::chrono::high_resolution_clock::now();
    unordered_map<int, vector<pcl::PointNormal>> uncVpsLib;
    int unc_vps_size = 0;
    for (int i = 0; i < (int)uncoveredID.size(); ++i)
    {
      int p = uncoveredID[i];

      Eigen::Vector3d pt_vec(model_->points[p].x, model_->points[p].y, model_->points[p].z);
      if (MCI.pt_normal_pairs.find(pt_vec) == MCI.pt_normal_pairs.end())
        continue;
      Eigen::Vector3d normal_dir = MCI.pt_normal_pairs.find(pt_vec)->second;
      pcl::Normal negiNormal;
      negiNormal.normal_x = normal_dir(0);
      negiNormal.normal_y = normal_dir(1);
      negiNormal.normal_z = normal_dir(2);
      vector<pcl::PointNormal> vpNeighs_2 = uncNeighVps(model_->points[p], negiNormal);
      uncVpsLib[p] = vpNeighs_2;
      unc_vps_size += vpNeighs_2.size();
    }
    auto lib_t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> lib_ms = lib_t2 - lib_t1;
    ROS_INFO("\033[36m[ViewpointManager] primitive viewpoints library size = %d, generation time = %lf ms.\033[32m", unc_vps_size, (double)lib_ms.count());

    ROS_INFO("\033[36m[ViewpointManager] ----------------------------------------------------------\033[32m");
    ROS_INFO("\033[36m[ViewpointManager] number of updated viewpoints                        = %d.\033[32m", (int)VPP.final_vps_.size());
    ROS_INFO("\033[36m[ViewpointManager] number of total voxels                              = %d.\033[32m", (int)MCI.cover_state.size());
    ROS_INFO("\033[36m[ViewpointManager] number of seen voxels before iter updating          = %d.\033[32m", sum_seen);
    ROS_INFO("\033[36m[ViewpointManager] number of uncovered voxels before iter updating     = %d.\033[32m", (int)uncoveredID.size());
    ROS_INFO("\033[36m[ViewpointManager] ----------------------------------------------------------\033[32m");

    int unc_iter = 0, last_unc_num = -1;
    int iter_max = max_iter_num;
    while ((int)uncoveredID.size() > 0 && unc_iter < iter_max && last_unc_num != (int)uncoveredID.size())
    {
      last_unc_num = (int)uncoveredID.size();

      /* Generate viewpoints library (UncoveredVps) for uncovered area */
      pcl::PointCloud<pcl::PointNormal>::Ptr UncoveredVps(new pcl::PointCloud<pcl::PointNormal>);
      for (auto p : uncoveredID)
      {
        for (auto uvp : uncVpsLib[p])
          UncoveredVps->points.push_back(uvp);
      }

      /* Randomly select 100 viewpoints on UncoveredVps */
      int candiNum = 100;
      if ((int)UncoveredVps->points.size() > candiNum)
      {
        pcl::RandomSample<pcl::PointNormal> rs;
        rs.setInputCloud(UncoveredVps);
        rs.setSample(candiNum);
        rs.filter(*UncoveredVps);
      }
      vector<Eigen::Vector3d> tempVps;
      vector<Eigen::Vector3d> tempVpsDir;
      for (int i = 0; i < (int)UncoveredVps->points.size(); i++)
      {
        Eigen::Vector3d vp_position;
        vp_position(0) = UncoveredVps->points[i].x;
        vp_position(1) = UncoveredVps->points[i].y;
        vp_position(2) = UncoveredVps->points[i].z;

        Eigen::Vector3d posi_, dir_;
        posi_ << UncoveredVps->points[i].x, UncoveredVps->points[i].y, UncoveredVps->points[i].z;
        dir_ << UncoveredVps->points[i].normal_x, UncoveredVps->points[i].normal_y, UncoveredVps->points[i].normal_z;
        tempVpsDir.push_back(dir_);
        tempVps.push_back(vp_position);
      }

      /* Viewpoints prune start */
      vector<bool> tempCoverState = MCI.cover_state;
      vector<int> tempCoverContrib = MCI.cover_contrib;
      vector<int> tempContribId = MCI.contrib_id;

      int oriVpNum = VPI.vps_pose.rows();
      VPI.last_vps_num = oriVpNum;

      Eigen::MatrixXd pose_set_unc;
      pose_set_unc.resize(tempVps.size(), 5);
      vector<int> vps_count(tempVps.size(), 0);
      vector<int> vps_contri(tempVps.size(), 0);
      VPI.vps_voxcount.insert(VPI.vps_voxcount.end(), vps_count.begin(), vps_count.end());
      VPI.vps_contri.insert(VPI.vps_contri.end(), vps_contri.begin(), vps_contri.end());
      vector<int> before_vps_voxcount = VPI.vps_voxcount;

      for (int i = 0; i < (int)tempVps.size(); ++i)
      {
        int vp_id = oriVpNum + i;
        pose_set_unc.row(i) = getPose(vp_id, tempVps[i], tempVpsDir[i]);
      }

      Eigen::MatrixXd oriVps = VPI.vps_pose;
      if ((int)oriVps.rows() > 0)
      {
        VPI.vps_pose.resize(oriVps.rows() + tempVps.size(), 5);
        VPI.vps_pose.topRows(oriVps.rows()) = oriVps;
        VPI.vps_pose.bottomRows(tempVps.size()) = pose_set_unc;
      }
      else
        VPI.vps_pose = pose_set_unc;

      MCI.cover_state = tempCoverState;
      MCI.cover_contrib = tempCoverContrib;
      MCI.contrib_id = tempContribId;
      temp_vps_voxcount = VPI.vps_voxcount;
      VPI.vps_voxcount = before_vps_voxcount;
      viewpointsPrune(VPI.vps_pose, temp_vps_voxcount);
      ROS_INFO("\033[36m[ViewpointManager] Iter_%d -> number of updated viewpoints              = %d.\033[32m", unc_iter, (int)VPP.final_vps_.size());

      /* Viewpoints prune end and Then update uncovered area for next iter */
      uncoveredID.clear();
      for (int i = 0; i < (int)MCI.cover_state.size(); ++i)
        if (MCI.cover_state[i] == false)
          uncoveredID.push_back(i);

      sum_seen = 0;
      for (auto final_vp : VPP.final_vps_)
      {
        sum_seen += final_vp.vox_count;
      }
      ROS_INFO("\033[36m[ViewpointManager] Iter_%d -> number of total voxels                    = %d.\033[32m", unc_iter, (int)MCI.cover_state.size());
      ROS_INFO("\033[36m[ViewpointManager] Iter_%d -> number of seen voxels after updating      = %d.\033[32m", unc_iter, sum_seen);
      ROS_INFO("\033[36m[ViewpointManager] Iter_%d -> number of uncovered voxels after updating = %d.\033[32m", unc_iter, (int)uncoveredID.size());
      ROS_INFO("\033[36m[ViewpointManager] ----------------------------------------------------------\033[32m");

      unc_iter++;
    }
  }

  Eigen::VectorXd ViewpointManager::getPose(int vp_id, const Eigen::Vector3d &pos, const Eigen::Vector3d &dir)
  {
    vector<int> index_set;
    Eigen::Vector3d viewpoint, visible_candidate;
    Eigen::Vector3i idx, idx_rev;
    viewpoint = pos;
    Eigen::Vector2d init_py;
    Eigen::Vector3d fov_sample_ray = dir;
    fov_sample_ray.normalize();
    Eigen::Vector3d fov_yaw_ray = fov_sample_ray;
    fov_yaw_ray(2) = 0.0;
    fov_yaw_ray.normalize();
    init_py = PitchYaw(fov_sample_ray);

    if (!pose_update)
    {
      Eigen::VectorXd pose_res(5);
      pose_res << viewpoint(0), viewpoint(1), viewpoint(2), init_py(0), init_py(1);
      updateViewpointInfo(pose_res, vp_id);
      return pose_res;
    }

    /* find visible voxel candidates by normal search */
    Eigen::Vector3d pt_vec;
    percep_utils_->setPose_PY(viewpoint, init_py(0), init_py(1));
    for (int i = 0; i < (int)model_->points.size(); ++i)
    {
      pt_vec << model_->points[i].x, model_->points[i].y, model_->points[i].z;
      if (percep_utils_->insideShrinkFOV(pt_vec) == true) index_set.push_back(i);
    }

    Eigen::Vector3d ray_dir, ray_dir_yaw;
    double ray_length;
    Eigen::Vector2d pitchyaw;

    double avg_pitch = 0.0, avg_yaw = 0.0;
    double max_avg_yaw = -1.0;
    Eigen::Vector3d max_yaw_ray;
    int free_num = 0;

    vector<double> pitch_list, yaw_list;
    unordered_map<int, Eigen::Vector3d> free_vox;
    
    Eigen::VectorXd pose_result;
    for (int i = 0; i < (int)index_set.size(); ++i)
    {
      visible_candidate << model_->points[index_set[i]].x, model_->points[index_set[i]].y, model_->points[index_set[i]].z;

      Eigen::Vector3d ray_dir = (viewpoint - visible_candidate).normalized();
      Eigen::Vector3d inflate_pt = visible_candidate + res * ray_dir;

      if (!visibility_st::rayTracing(viewpoint, inflate_pt, scene_)) continue;
      
      ray_dir = (visible_candidate - viewpoint).normalized();
      ray_length = (visible_candidate - viewpoint).norm();
      
      if (ray_length <= visible_range)
      {
        pitchyaw = PitchYaw(ray_dir);
        warpAngle(pitchyaw(0));
        warpAngle(pitchyaw(1));
        avg_pitch += pitchyaw(0);

        pitch_list.push_back(pitchyaw(0));

        double yaw_vec = std::min(std::max(fov_sample_ray.dot(ray_dir), -1.0), 1.0);
        double yawoff = acos(yaw_vec);
        if (yawoff > max_avg_yaw)
        {
          max_yaw_ray = ray_dir;
          max_avg_yaw = yawoff;
        }

        ray_dir_yaw = visible_candidate - viewpoint;
        ray_dir_yaw(2) = 0.0;
        ray_dir_yaw.normalize();

        if (ray_dir_yaw.cross(fov_yaw_ray)[2] < 0)
          yawoff = -yawoff;
        
        yaw_list.push_back(yawoff);

        free_num++;
        free_vox[index_set[i]] = visible_candidate;
      }
    }

    if (free_num == 0)
    {
      avg_pitch = init_py(0);
      avg_yaw = init_py(1);
      pose_result.resize(5);
      pose_result << viewpoint(0), viewpoint(1), viewpoint(2), avg_pitch, avg_yaw;
      percep_utils_->setPose_PY(viewpoint, avg_pitch, avg_yaw);

      return pose_result;
    }
    /* only adjust pitch */
    avg_pitch = std::accumulate(pitch_list.begin(), pitch_list.end(), 0.0) / free_num;
    avg_yaw = std::accumulate(yaw_list.begin(), yaw_list.end(), 0.0) / free_num;
    
    avg_yaw = init_py(1) + avg_yaw;
    warpAngle(avg_pitch);
    warpAngle(avg_yaw);

    if (avg_pitch > pitch_upper * M_PI / 180.0)
      avg_pitch = pitch_upper * M_PI / 180.0;
    if (avg_pitch < pitch_lower * M_PI / 180.0)
      avg_pitch = pitch_lower * M_PI / 180.0;

    map<int, Eigen::Vector3d> valid_vox;
    int contribute_voxel_num = 0;
    bool inside_flag = false;

    pose_result.resize(5);
    pose_result << viewpoint(0), viewpoint(1), viewpoint(2), avg_pitch, avg_yaw;
    percep_utils_->setPose_PY(viewpoint, avg_pitch, avg_yaw);

    for (const auto &id_vox : free_vox)
    {
      inside_flag = percep_utils_->insideShrinkFOV(id_vox.second);
      if (inside_flag == true)
      {
        Eigen::Vector3d visible_candidate = id_vox.second;
        Eigen::Vector3d ray_dir, inflate_pt;
        ray_dir = (viewpoint - visible_candidate).normalized();
        inflate_pt = visible_candidate + res * ray_dir;
        bool visible = visibility_st::rayTracing(viewpoint, inflate_pt, scene_);
        if (visible == true)
        {
          contribute_voxel_num++;
          valid_vox[id_vox.first] = id_vox.second;
        }
      }
    }

    if (contribute_voxel_num > 0)
    {
      VPI.vps_contri[vp_id] = contribute_voxel_num;
      for (const auto &val_id_vox : valid_vox)
      {
        if (MCI.cover_state[val_id_vox.first] == false)
        {
          MCI.cover_state[val_id_vox.first] = true;
          MCI.cover_contrib[val_id_vox.first] = contribute_voxel_num;
          MCI.contrib_id[val_id_vox.first] = vp_id;
          VPI.vps_voxcount[vp_id] += 1;
        }
        else
        {
          if (contribute_voxel_num > MCI.cover_contrib[val_id_vox.first])
          {
            int before_vp_id = MCI.contrib_id[val_id_vox.first];
            if (before_vp_id >= VPI.last_vps_num)
            {
              MCI.cover_contrib[val_id_vox.first] = contribute_voxel_num;
              MCI.contrib_id[val_id_vox.first] = vp_id;

              VPI.vps_voxcount[before_vp_id] -= 1;
              VPI.vps_voxcount[vp_id] += 1;
            }
          }
        }
      }
    }

    return pose_result;
  }

  void ViewpointManager::updateViewpointInfo(Eigen::VectorXd &pose, const int &vp_id)
  {
    int contribute_voxel_num = 0;

    vector<int> valid_vox;
    Eigen::Vector3d position_ = pose.head(3);
    double pitch_, yaw_;
    pitch_ = pose(3);
    yaw_ = pose(4);

    percep_utils_->setPose_PY(position_, pitch_, yaw_);
    Eigen::Vector3d pt_vec;
    Eigen::Vector3i idx, idx_rev;
    for (int i = 0; i < (int)model_->points.size(); ++i)
    {
      pt_vec(0) = model_->points[i].x;
      pt_vec(1) = model_->points[i].y;
      pt_vec(2) = model_->points[i].z;
      if (percep_utils_->insideShrinkFOV(pt_vec))
      {
        // // Filter out non-surface obstacles
        // Vector3d pos1 = pt_vec;
        // Vector3d pos2 = position_;
        // Vector3d free_pose = pt_vec;
        // raycaster_->input(pos1, pos2);
        // int count = 0;
        // while (raycaster_->nextId(idx))
        // {
        //   if (HCMap->occCheck(idx)) // is free
        //   {
        //     count++;
        //     if (count == 2)
        //       break;
        //   }
        // }
        // HCMap->indexToPos_hc(idx, free_pose);

        // // Bidirectional Ray Casting
        // bool vis_flag = true, vis_flag_rev = true;
        // raycaster_->biInput(position_, free_pose);
        // while (raycaster_->biNextId(idx, idx_rev))
        // {
        //   vis_flag = HCMap->occCheck(idx);
        //   vis_flag_rev = HCMap->occCheck(idx_rev);
        //   if (vis_flag == false || vis_flag_rev == false)
        //     break;
        // }
        // vis_flag = HCMap->occCheck(idx);
        // if (vis_flag == true && vis_flag_rev == true)

        Eigen::Vector3d visible_candidate = pt_vec;
        Eigen::Vector3d ray_dir, inflate_pt;
        ray_dir = (position_ - visible_candidate).normalized();
        inflate_pt = visible_candidate + res * ray_dir;
        bool visible = visibility_st::rayTracing(position_, inflate_pt, scene_);
        if (visible == true)
        {
          contribute_voxel_num++;
          valid_vox.push_back(i);
        }
      }
    }

    if (contribute_voxel_num == 0)
      return;

    VPI.vps_contri[vp_id] = contribute_voxel_num;
    for (const auto &val_id_vox : valid_vox)
    {
      if (MCI.cover_state[val_id_vox] == false)
      {
        MCI.cover_state[val_id_vox] = true;
        MCI.cover_contrib[val_id_vox] = contribute_voxel_num;
        MCI.contrib_id[val_id_vox] = vp_id;
        VPI.vps_voxcount[vp_id] += 1;
      }
      else
      {
        if (contribute_voxel_num > MCI.cover_contrib[val_id_vox])
        {
          int before_vp_id = MCI.contrib_id[val_id_vox];
          if (before_vp_id >= VPI.last_vps_num)
          {
            MCI.cover_contrib[val_id_vox] = contribute_voxel_num;
            MCI.contrib_id[val_id_vox] = vp_id;

            VPI.vps_voxcount[before_vp_id] -= 1;
            VPI.vps_voxcount[vp_id] += 1;
          }
        }
      }
    }
  }

  void ViewpointManager::viewpointsPrune(Eigen::MatrixXd vps_pose, vector<int> vps_voxcount)
  {
    VPP.inverse_idx_.clear();
    VPP.idx_viewpoints_.clear();
    VPP.idx_ctrl_voxels_.clear();
    VPP.idx_live_state_.clear();
    VPP.idx_query_state_.clear();
    vector<int> prune_order_;

    pcl::PointXYZ vp_pt_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr vp_cloud_(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = VPI.last_vps_num; i < (int)vps_pose.rows(); ++i)
    {
      pcl::PointXYZ pos_pt;
      pos_pt.x = vps_pose(i, 0);
      pos_pt.y = vps_pose(i, 1);
      pos_pt.z = vps_pose(i, 2);
      double pitch_pt = vps_pose(i, 3);
      warpAngle(pitch_pt);
      if (vps_voxcount[i] > 0 && viewpointSafetyCheck(pos_pt) && pitch_pt >= pitch_lower * M_PI / 180.0 && pitch_pt <= pitch_upper * M_PI / 180.0)
      {
        Eigen::VectorXd qualified_vp_ = vps_pose.row(i);
        int index_ = i;
        vp_pt_.x = qualified_vp_(0);
        vp_pt_.y = qualified_vp_(1);
        vp_pt_.z = qualified_vp_(2);

        Eigen::Vector3d tmp_pos(vp_pt_.x, vp_pt_.y, vp_pt_.z);
        vp_cloud_->points.push_back(vp_pt_);

        VPP.inverse_idx_[tmp_pos] = index_;
        VPP.idx_viewpoints_[index_] = qualified_vp_;
        VPP.idx_ctrl_voxels_[index_] = vps_voxcount[i];
        VPP.idx_live_state_[index_] = true; // all initialized as active
        VPP.idx_query_state_[index_] = false;
      }
    }

    if (vp_cloud_->points.size() == 0)
    {
      ROS_ERROR("No need prune viewpoints!");
      return;
    }

    /* sort vps according to ctrl voxels */
    vector<pair<int, int>> ctrlVector(VPP.idx_ctrl_voxels_.begin(), VPP.idx_ctrl_voxels_.end());
    sort(ctrlVector.begin(), ctrlVector.end(), [](const auto &a, const auto &b)
         { return a.second > b.second; });
    for (const auto &p : ctrlVector)
      prune_order_.push_back(p.first);

    /* construct viewpoints kdtree */
    pcl::KdTreeFLANN<pcl::PointXYZ> vps_tree_;
    vps_tree_.setInputCloud(vp_cloud_);

    /* Gravitation-like model */
    double bubble_radius = 0.55 * dist_vp * tan(fov_base);
    for (auto prune_idx : prune_order_)
    {
      Eigen::Vector3d prune_position = VPP.idx_viewpoints_.find(prune_idx)->second.head(3);
      bool psv = nearCheck(prune_position, VPP.idx_ctrl_voxels_.find(prune_idx)->second);
      if (psv == false)
      {
        VPP.idx_live_state_.find(prune_idx)->second = false;
        continue;
      }

      vector<int> index_set;
      vector<float> radius_set;
      vp_pt_.x = VPP.idx_viewpoints_.find(prune_idx)->second(0);
      vp_pt_.y = VPP.idx_viewpoints_.find(prune_idx)->second(1);
      vp_pt_.z = VPP.idx_viewpoints_.find(prune_idx)->second(2);
      vps_tree_.radiusSearch(vp_pt_, bubble_radius, index_set, radius_set);

      if (VPP.idx_live_state_.find(prune_idx)->second == true && (int)index_set.size() > 1)
      {
        updatePoseGravitation(vp_cloud_, index_set, prune_idx);
      }
    }

    for (auto &pair : VPP.idx_viewpoints_)
    {
      if (VPP.idx_live_state_.find(pair.first)->second == true)
        updateViewpointInfo(pair.second, pair.first);
    }

    SingleViewpoint struct_vp;
    for (auto &pair : VPP.idx_viewpoints_)
    {
      if (VPP.idx_live_state_.find(pair.first)->second == true && VPI.vps_voxcount[pair.first] > this->save_lower_)
      {
        struct_vp.vp_id = pair.first;
        struct_vp.pose = pair.second;
        struct_vp.vox_count = VPI.vps_voxcount[pair.first];
        VPP.final_vps_.push_back(struct_vp);
      }
    }
  }

  void ViewpointManager::updatePoseGravitation(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, vector<int> &inner_ids, const int &cur_idx)
  {
    VPP.idx_query_state_.find(cur_idx)->second = true;

    double pitch_bound = 0.3 * fov_h * M_PI / 180.0;
    double yaw_bound = 0.3 * fov_w * M_PI / 180.0;

    Eigen::VectorXd cur_pose = VPP.idx_viewpoints_.find(cur_idx)->second;
    Eigen::Vector3d cur_position = cur_pose.head(3);
    double cur_pitch = cur_pose(3);
    double cur_yaw = cur_pose(4);
    int cur_vox = VPP.idx_ctrl_voxels_.find(cur_idx)->second;

    int index_finder;
    Eigen::Vector3d vp_finder;
    Eigen::VectorXd inner_pose;
    int inner_vox;
    double diff_yaw, diff_pitch;
    bool update_flag = false;
    vector<int> updated_indexer;

    for (auto id_ : inner_ids)
    {
      update_flag = false;
      vp_finder(0) = cloud->points[id_].x;
      vp_finder(1) = cloud->points[id_].y;
      vp_finder(2) = cloud->points[id_].z;
      index_finder = VPP.inverse_idx_.find(vp_finder)->second;
      inner_pose = VPP.idx_viewpoints_.find(index_finder)->second;
      inner_vox = VPP.idx_ctrl_voxels_.find(index_finder)->second;
      if (inner_vox > cur_vox)
        continue;
      else
      {
        diff_yaw = fabs(cur_pose(4) - inner_pose(4));
        warpAngle(diff_yaw);
        diff_pitch = fabs(cur_pose(3) - inner_pose(3));
        warpAngle(diff_pitch);
        if (fabs(diff_pitch) < pitch_bound && fabs(diff_yaw) < yaw_bound)
          update_flag = true;

        if (update_flag == true && VPP.idx_query_state_.find(index_finder)->second == false)
        {
          VPP.idx_live_state_.find(index_finder)->second = false;
          updated_indexer.push_back(index_finder);
        }
      }
    }
    /* gravitation-like model */
    Eigen::Vector3d updated_position = cur_position;
    double updated_pitch = cur_pitch;
    double updated_yaw = cur_yaw;
    if ((int)updated_indexer.size() > 0)
    {
      for (auto ui_ : updated_indexer)
      {
        // Position model
        double grav_factor = (double)VPP.idx_ctrl_voxels_.find(ui_)->second / (double)cur_vox;
        Eigen::Vector3d up_position = VPP.idx_viewpoints_.find(ui_)->second.head(3);
        Eigen::Vector3d pull_vec = grav_factor * (up_position - cur_position);
        updated_position = updated_position + pull_vec;
        if (attitude == "all")
        {
          // Pitch model
          double up_pitch = VPP.idx_viewpoints_.find(ui_)->second(3);
          double pitch_diff = abs(up_pitch - cur_pitch) > M_PI ? (2 * M_PI - abs(up_pitch - cur_pitch)) : abs(up_pitch - cur_pitch);
          int cal_flag_pitch = up_pitch - cur_pitch > 0 ? 1 : -1;
          int cal_dir_pitch = abs(up_pitch - cur_pitch) > M_PI ? -1 : 1;
          double pull_pitch = grav_factor * cal_flag_pitch * cal_dir_pitch * pitch_diff;
          updated_pitch = updated_pitch + pull_pitch;

          warpAngle(updated_pitch);
          updated_pitch = updated_pitch > pitch_upper * M_PI / 180.0 ? pitch_upper * M_PI / 180.0 : updated_pitch;
          updated_pitch = updated_pitch < pitch_lower * M_PI / 180.0 ? pitch_lower * M_PI / 180.0 : updated_pitch;
        }
        // Yaw model
        double up_yaw = VPP.idx_viewpoints_.find(ui_)->second(4);
        double yaw_diff = abs(up_yaw - cur_yaw) > M_PI ? (2 * M_PI - abs(up_yaw - cur_yaw)) : abs(up_yaw - cur_yaw);
        int cal_flag_yaw = up_yaw - cur_yaw > 0 ? 1 : -1;
        int cal_dir_yaw = abs(up_yaw - cur_yaw) > M_PI ? -1 : 1;
        double pull_yaw = grav_factor * cal_flag_yaw * cal_dir_yaw * yaw_diff;
        updated_yaw = updated_yaw + pull_yaw;
        warpAngle(updated_yaw);
      }
    }
    if (zFlag == true && updated_position(2) < GroundZPos + safeHeight)
      updated_position(2) = GroundZPos + safeHeight + 0.1;

    pcl::PointXYZ searchPoint;
    searchPoint.x = updated_position(0);
    searchPoint.y = updated_position(1);
    searchPoint.z = updated_position(2);
    if (viewpointSafetyCheck(searchPoint) == true)
    {
      VPP.idx_viewpoints_.find(cur_idx)->second(0) = updated_position(0);
      VPP.idx_viewpoints_.find(cur_idx)->second(1) = updated_position(1);
      VPP.idx_viewpoints_.find(cur_idx)->second(2) = updated_position(2);
    }
    else
    {
      VPP.idx_viewpoints_.find(cur_idx)->second(0) = cur_position(0);
      VPP.idx_viewpoints_.find(cur_idx)->second(1) = cur_position(1);
      VPP.idx_viewpoints_.find(cur_idx)->second(2) = cur_position(2);
    }

    if (!pose_update)
    {
      VPP.idx_viewpoints_.find(cur_idx)->second(3) = updated_pitch;
      VPP.idx_viewpoints_.find(cur_idx)->second(4) = updated_yaw;
    }
    else
    {
      // Further update of pitch and yaw
      vector<int> inner_idx;
      percep_utils_->setPose_PY(updated_position, updated_pitch, updated_yaw);
      for (int i = 0; i < (int)model_->points.size(); ++i)
      {
        Eigen::Vector3d pt_vec;
        pt_vec << model_->points[i].x, model_->points[i].y, model_->points[i].z;
        if (percep_utils_->insideShrinkFOV(pt_vec) == true)
          inner_idx.push_back(i);
      }
      // bool vis_flag = true, vis_flag_rev = true;
      Eigen::Vector3d visible_candidate;
      Eigen::Vector3i idx, idx_rev;
      Eigen::Vector3d ray_dir, ray_dir_yaw;
      double avg_pitch = updated_pitch, avg_yaw = updated_yaw;
      if ((int)inner_idx.size() > 0)
      {
        for (int i = 0; i < (int)inner_idx.size(); ++i)
        {
          // vis_flag = true;
          // vis_flag_rev = true;
          visible_candidate(0) = model_->points[inner_idx[i]].x;
          visible_candidate(1) = model_->points[inner_idx[i]].y;
          visible_candidate(2) = model_->points[inner_idx[i]].z;

          // raycaster_->biInput(updated_position, visible_candidate);
          // raycaster_->biNextId(idx, idx_rev);
          // while (raycaster_->biNextId(idx, idx_rev))
          // {
          //   vis_flag = HCMap->occCheck(idx);
          //   vis_flag_rev = HCMap->occCheck(idx_rev);
          //   if (vis_flag == false || vis_flag_rev == false)
          //     break;
          // }
          // vis_flag = HCMap->occCheck(idx);
          // if (vis_flag == true && vis_flag_rev == true)

          Eigen::Vector3d ray_dir, inflate_pt;
          ray_dir = (updated_position - visible_candidate).normalized();
          inflate_pt = visible_candidate + res * ray_dir;
          bool visible = visibility_st::rayTracing(updated_position, inflate_pt, scene_);
          if (visible == true)

          {
            ray_dir = (visible_candidate - updated_position).normalized();
            ray_dir_yaw = visible_candidate - updated_position;
            ray_dir_yaw(2) = 0.0;
            ray_dir_yaw.normalized();
            Eigen::Vector2d pitchyaw = PitchYaw(ray_dir);
            warpAngle(pitchyaw(0));
            warpAngle(pitchyaw(1));

            double pitch_off = abs(pitchyaw(0) - updated_pitch) > M_PI ? (2 * M_PI - abs(pitchyaw(0) - updated_pitch)) : abs(pitchyaw(0) - updated_pitch);
            int flag_pitch = pitchyaw(0) - updated_pitch > 0 ? 1 : -1;
            int dir_pitch = abs(pitchyaw(0) - updated_pitch) > M_PI ? -1 : 1;
            avg_pitch = avg_pitch + flag_pitch * dir_pitch * pitch_off;

            double yaw_off = abs(pitchyaw(1) - updated_yaw) > M_PI ? (2 * M_PI - abs(pitchyaw(1) - updated_yaw)) : abs(pitchyaw(1) - updated_yaw);
            int flag_yaw = pitchyaw(1) - updated_yaw > 0 ? 1 : -1;
            int dir_yaw = abs(pitchyaw(1) - updated_yaw) > M_PI ? -1 : 1;
            avg_yaw = avg_yaw + flag_yaw * dir_yaw * yaw_off;
          }
        }
      }
      updated_pitch = avg_pitch;
      updated_yaw = avg_yaw;
      if (updated_pitch > pitch_upper * M_PI / 180.0)
        updated_pitch = pitch_upper * M_PI / 180.0;
      if (updated_pitch < pitch_lower * M_PI / 180.0)
        updated_pitch = pitch_lower * M_PI / 180.0;
      warpAngle(updated_pitch);
      warpAngle(updated_yaw);
      VPP.idx_viewpoints_.find(cur_idx)->second(3) = updated_pitch;
      VPP.idx_viewpoints_.find(cur_idx)->second(4) = updated_yaw;
    }
  }

  bool ViewpointManager::viewpointSafetyCheck(pcl::PointXYZ &vp)
  {
    if (zFlag && vp.z < GroundZPos + safeHeight)
      return false;

    Eigen::Vector3d test(vp.x, vp.y, vp.z);
    bool inside_test = visibility_st::inlierCheck(test, this->scene_); 
    if (inside_test)
      return false;

    double check_range = this->drone_radius_ + this->res;
    bool real_safe = HCMap->getSafety(test, check_range, check_range, 0.5*check_range);
    if (!real_safe)
      return false;

    Eigen::Vector3d safe_check_pt = test;
    if (!HCMap->safety_check(safe_check_pt))
      return false;

    return true;
  }

  bool ViewpointManager::nearCheck(Eigen::Vector3d &cur_position, int &vox_num)
  {
    if (!cur_position.allFinite())
    {
      return false;
    }
    bool preserve_ = true;
    int knn = 1;
    pcl::PointXYZ search_point;

    search_point.x = cur_position(0);
    search_point.y = cur_position(1);
    search_point.z = cur_position(2);
    vector<int> nearest_id(knn);
    vector<float> nn_squared_distance(knn);
    try
    {
      map_cloud_kdtree_.nearestKSearch(search_point, knn, nearest_id, nn_squared_distance);
    }
    catch (const std::exception &e)
    {
      std::cerr << "Exception during KD tree query: " << e.what() << std::endl;
      return false;
    }

    double dist = sqrt(nn_squared_distance[0]);
    if (dist < safe_radius_vp || vox_num <= 1)
      preserve_ = false;

    return preserve_;
  }

  vector<pcl::PointNormal> ViewpointManager::uncNeighVps(pcl::PointXYZ oript, pcl::Normal normal)
  {
    double max_pitch_range = 90.0; // random max pitch angle
    // double max_yaw_range = 15.0;   // random max yaw angle
    double max_yaw_range = 4.0;   // ! random max yaw angle -> concerthall
    if (attitude == "yaw")
        max_pitch_range = 0;
    vector<pcl::PointNormal> unc_vps;

    Eigen::Vector3d oriNormal;
    oriNormal << normal.normal_x, normal.normal_y, normal.normal_z;
    Eigen::Vector2d oriPY = PitchYaw(oriNormal);

    // Initialize random generators and distributions
    default_random_engine gen(std::chrono::system_clock::now().time_since_epoch().count());
    uniform_real_distribution<double> dist_p(0.0, max_pitch_range * M_PI / 180.0);
    uniform_real_distribution<double> dist_y(0.0, max_yaw_range * M_PI / 180.0);

    // Generate random pitch and yaw angles
    vector<double> rand_p(4), rand_y(4);
    for (int i = 0; i < 4; ++i) 
    {
      rand_p[i] = dist_p(gen);
      rand_y[i] = dist_y(gen);
    }

    // Generate 8 viewpoint vectors
    vector<Eigen::Vector2d> py(8);
    py[0] << oriPY(0) + rand_p[0], oriPY(1) + rand_y[0];
    py[1] << oriPY(0) - rand_p[1], oriPY(1) - rand_y[1];
    py[2] << oriPY(0) + rand_p[2], oriPY(1) + rand_y[2];
    py[3] << oriPY(0) - rand_p[3], oriPY(1) - rand_y[3];
    py[4] << oriPY(0) + rand_p[0], oriPY(1) - rand_y[0];
    py[5] << oriPY(0) - rand_p[1], oriPY(1) + rand_y[1];
    py[6] << oriPY(0) + rand_p[2], oriPY(1) - rand_y[2];
    py[7] << oriPY(0) - rand_p[3], oriPY(1) + rand_y[3];

    // Convert pitch-yaw to vectors
    vector<Eigen::Vector3d> vec(8);
    for (int i = 0; i < 8; ++i) 
    {
      warpAngle(py[i](0));
      warpAngle(py[i](1));
      py[i][0] = py[i][0] > pitch_upper * M_PI / 180.0 ? pitch_upper * M_PI / 180.0 : py[i][0];
      py[i][0] = py[i][0] < pitch_lower * M_PI / 180.0 ? pitch_lower * M_PI / 180.0 : py[i][0];

      vec[i] = pyToVec(py[i]);
    }

    // Create and check viewpoints
    vector<double> sample_dist = {0.25*dist_vp, 0.5*dist_vp, 0.75*dist_vp, dist_vp};
    for (int i = 0; i < 8; ++i) 
    {
      for (int j = 0; j < (int)sample_dist.size(); ++j) 
      {
        pcl::PointNormal vp;

        double qual_dist_vp = sample_dist[j];
        if (abs(vec[i](2)) != 0)
        {
          double inner_dist_vp = (-1.0-oript.z) / vec[i](2);
          if (inner_dist_vp > 0 && inner_dist_vp < qual_dist_vp) qual_dist_vp = inner_dist_vp - 1.0;
        }

        vp.x = oript.x + qual_dist_vp * vec[i](0);
        vp.y = oript.y + qual_dist_vp * vec[i](1);
        vp.z = oript.z + qual_dist_vp * vec[i](2);
        vp.normal_x = -vec[i](0);
        vp.normal_y = -vec[i](1);
        vp.normal_z = -vec[i](2);

        pcl::PointXYZ vp_xyz;
        vp_xyz.x = vp.x;
        vp_xyz.y = vp.y;
        vp_xyz.z = vp.z;

        if (viewpointSafetyCheck(vp_xyz)) 
        {
          unc_vps.push_back(vp);
        }
      }
    }

    return unc_vps;
  }

  void ViewpointManager::getUpdatedViewpoints(vector<SingleViewpoint> &viewpoints)
  {
    for (auto x : VPP.final_vps_)
    {
      Eigen::Vector3d vp_pos(x.pose(0), x.pose(1), x.pose(2));
      bool inflate_safe = HCMap->safety_check(vp_pos);
      if (!inflate_safe)
      {
        ROS_ERROR("viewpoint inside the inflation!!! ------------------------------------------------------------------------------");
      }

    }
    
    viewpoints = VPP.final_vps_;
  }

  void ViewpointManager::getUncoveredModelCloud(pcl::PointCloud<pcl::PointXYZ> &cloud)
  {
    cloud.clear();
    for (int i = 0; i < (int)MCI.cover_state.size(); ++i)
      if (MCI.cover_state[i] == false)
        cloud.push_back(model_->points[i]);
  }

  bool ViewpointManager::rayTracing(Eigen::Vector3d& pointA, Eigen::Vector3d& pointB, RTCScene& scene)
  {
    if (!scene) 
    {
      std::cerr << "Error: RTCSceneTy pointer is null." << std::endl;
      exit(1);
    }
    
    Eigen::Vector3d direction_d = (pointB - pointA).normalized();
    Eigen::Vector3f direction = direction_d.cast<float>();
    double length = (pointB - pointA).norm();

    RTCRayHit rayhit;
    RTCRay ray;
    ray.org_x = pointA(0);
    ray.org_y = pointA(1);
    ray.org_z = pointA(2);
    ray.dir_x = direction(0);
    ray.dir_y = direction(1);
    ray.dir_z = direction(2);
    ray.tnear = 0;
    ray.tfar = length;
    ray.time = 0.0f;
    ray.mask = -1;
    ray.id = 0;
    ray.flags = 0;
    rayhit.ray = ray;
    RTCHit hit;
    hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit = hit;

    RTCIntersectContext context;
    rtcInitIntersectContext(&context);
    rtcIntersect1(scene, &context, &rayhit);

    if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) 
      return true;
    else
      return false;
  }
}