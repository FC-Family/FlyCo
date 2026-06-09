/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Jun. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the main algorithm of hierarchical coverage planning in FlyCo.
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

#include "hierarchical_coverage_planner/fcplanner_pp.h"
namespace flyco
{

const double MAX = 1e6;

void FCPlanner_PP::init(ros::NodeHandle& nh)
{
  auto hcm_t1 = chrono::high_resolution_clock::now();

  // * Module Initialization
  skeleton_operator.reset(new sk_decomp);
  raycaster_.reset(new RayCaster);
  percep_utils_.reset(new PerceptionUtils);
  viewpoint_manager_.reset(new ViewpointManager);
  solver_.reset(new HCSolver);

  skeleton_operator->init(nh);
  percep_utils_->init(nh);
  viewpoint_manager_->init(nh);
  solver_->init(nh);

  // * Visualization
  vis_utils_.reset(new PlanningVisualization(nh));
  skeleton_operator->visFlag = false;
  visFlag = false;
  vis_timer_ = nh.createTimer(ros::Duration(0.5), &FCPlanner_PP::vis_Callback, this);

  // * ROS Service
  exec_traj_sub_ = nh.subscribe("/traj_server/exec_traj_waypts", 1, &FCPlanner_PP::execTrajCallback, this);
  bev_sub_ = nh.subscribe("/prediction/bev_box", 1, &FCPlanner_PP::bevCallback, this);

  // * Param Initialization
  this->nh_ = nh;
  nh.param("hcplanner/model_downsample_size", model_ds_size, -1.0);
  nh.param("hcplanner/cx", cx, -1.0);
  nh.param("hcplanner/cy", cy, -1.0);
  nh.param("hcplanner/fx", fx, -1.0);
  nh.param("hcplanner/fy", fy, -1.0);
  nh.param("hcplanner/vmax_", vmax_, -1.0);
  nh.param("hcplanner/amean_", amean_, -1.0);
  nh.param("hcplanner/exec_mode", mode_, string("null"));
  nh.param("hcplanner/safe_inner_dist_coeff", safe_inner_coefficient_, -1.0);
  nh.param("hcplanner/JointRadius", JointCoeff, -1.0);
  nh.param("hcplanner/localRange", local_range_, -1.0);
  nh.param("hcplanner/overlap_length", overlap_last_, -1.0);
  nh.param("hcplanner/global_consistency", global_consistency_, false);
  nh.param("hcplanner/localrefine_time", refine_time_upper_, -1.0);
  nh.param("hcplanner/suspension_rate", suspension_rate_, -1.0);
  nh.param("hcplanner/mesh_sample_size", sample_size_, -1);
  nh.param("hcplanner/sub_consistent", sub_consistent_, false);
  nh.param("hcplanner/open_chamfer", open_chamfer_, false);
  nh.param("hcplanner/max_cvx_range", max_cvx_range_, -1.0);
  nh.param("hcplanner/tsp_horizon", tsp_horizon_, 25.0);
  nh.param("hcplanner/match_num", match_num_, -1);
  nh.param("viewpoint_manager/viewpoints_distance", dist_vp, -1.0);
  nh.param("viewpoint_manager/fov_h", fov_h, -1.0);
  nh.param("viewpoint_manager/fov_w", fov_w, -1.0);
  nh.param("viewpoint_manager/GroundPos", GroundZPos, -1.0);
  nh.param("viewpoint_manager/TopPos", TopZPos, -1.0);
  nh.param("viewpoint_manager/zGround", zFlag, false);
  nh.param("viewpoint_manager/safeHeight", safeHeight, -1.0);
  nh.param("viewpoint_manager/safe_radius", safe_radius_vp, -1.0);
  nh.param("hctraj/drone_radius", drone_radius_, -1.0);
  nh.param("viewpoint_manager/init_size", vpm_init_size_, -1);
  nh.param("viewpoint_manager/save_lower", save_visible_lower_, 5);
  nh.param("viewpoint_manager/vis_inf", vis_inflation_, 0.5);
  nh.param("hcplanner/gs_size", ground_ds_size_, 5.0);
  nh.param("perception_utils/top_angle", v_fov_, 0.0);
  nh.param("hcplanner/g_flight_h", ground_flight_height_, 1.0);
  nh.param("hcplanner/g_sample_dist", g_vps_sample_dist_, 5.0);
  nh.param("flyco_planner/enable_inherit", en_inherit_, false);
  bev_seg_ = false;
  match_open_ = 0;

  fov_base = min(fov_h, fov_w) * M_PI / 360.0;
  corridorProgress = 2.0 * dist_vp * tan(fov_base);
  dist_upper_ = 0.5 * corridorProgress > 0.25 * local_range_ ? 0.25 * local_range_ : 0.5 * corridorProgress;
  mesh = skeleton_operator->input_mesh;

  PR.last_time_next_sub_inliers = {};

  auto hcm_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> hcm_ms = hcm_t2 - hcm_t1;
  double hcm_time = (double)hcm_ms.count(); 
  ROS_INFO("\033[32m[Planner] Initialization time = %lf ms.\033[32m", hcm_time);

  ROS_INFO("\033[35m[GlobalPlanner] Initialized! \033[32m");
}

void FCPlanner_PP::setInput(Eigen::MatrixXd& mesh_V, Eigen::MatrixXi& mesh_F)
{
  auto hcm_t1 = chrono::high_resolution_clock::now();

  // * Space Decomp
  skeleton_operator->set_mesh(mesh_V, mesh_F);

  mesh_V_ = mesh_V;
  mesh_F_ = mesh_F;
  visibility_st::mesh2scene(mesh_V, mesh_F, this->scene_);

  // * Evaluation
  occ_model.reset(new pcl::PointCloud<pcl::PointXYZ>);

  auto hcm_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> hcm_ms = hcm_t2 - hcm_t1;
  double hcm_time = (double)hcm_ms.count(); 
  ROS_INFO("\033[32m[Planner] set input time = %lf ms.\033[32m", hcm_time);
}

void FCPlanner_PP::setMap(shared_ptr<SDFMap> map)
{
  this->HCMap = map;
}

void FCPlanner_PP::reset()
{
  this->solver_.reset(new HCSolver);
  this->solver_->init(this->nh_);
  this->skeleton_operator.reset(new sk_decomp);
  this->skeleton_operator->init(this->nh_);
  this->skeleton_operator->visFlag = true;
  this->PR.clear();
  if (this->scene_ != nullptr)
  {
    rtcReleaseScene(this->scene_);
    this->scene_ = nullptr;
  }
}

bool FCPlanner_PP::plan(Eigen::VectorXd& current_pose, Eigen::Vector3d& cur_vel, vector<Eigen::VectorXd>& last_global_path, vector<bool>& last_global_indi, bool v1)
{
  auto plan_t1 = chrono::high_resolution_clock::now();

  PR.last_path_ = last_global_path;
  PR.last_wp_indicators_ = last_global_indi;

  this->start_pose_ = current_pose;
  this->start_vel_ = cur_vel;
  current_pos_ = current_pose.head(3);

  auto t1 = chrono::high_resolution_clock::now();

  // * Space Decomposition
  skeleton_operator->setScene(this->scene_);
  skeleton_operator->main();
  visibility_st::setInliers(skeleton_operator->P.dense_inliers);

  // * Evaluation
  if (v1 == true)
  {
    Fullmodel.reset(new pcl::PointCloud<pcl::PointXYZ>);
    visibleFullmodel.reset(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*skeleton_operator->adp_utils_->sampled_cloud, *Fullmodel);
  }
  pcl::copyPointCloud(*skeleton_operator->adp_utils_->sampled_cloud, *occ_model);

  // * Mapping & Solver & BiRC
  setMSB();

  // * Update Info
  HCMap->setInternal(skeleton_operator->P.ori_pts_, skeleton_operator->P.pt_inlier_pairs);
  solver_->setMap(HCMap);
  mapResults();
  uniformSampling();

  // * Update Current Uncovered Area
  updateUncoveredArea();
  if (this->suspension_ == true)
    return false;

  this->updated_pos_ = PR.prior_path.back().head(3);
  this->updated_pose_ = PR.prior_path.back();
  solver_->setStart(this->updated_pos_);

  reviseNormals();

  auto t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> ms = t2 - t1;
  double time = (double)ms.count();
  ROS_INFO("\033[33m[Planner] space info time = %lf ms.\033[32m", time);

  // * Viewpoint Generation
  viewpointGeneration();

  // * Hierarchical Coverage Planning
  if (this->viewpointNum > 0)
  {
    decompPlanning();
  }
  else
  {
    PR.FullPath_ = PR.prior_path;
    PR.waypoints_indicators_ = PR.prior_wp_indicators_;
  }

  // * Get Current HCMap
  PR.cur_hc_map_.reset(new pcl::PointCloud<pcl::PointXYZ>);

  auto plan_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> plan_ms = plan_t2 - plan_t1;
  plan_time = (double)plan_ms.count();
  ROS_INFO("\033[33m[Planner] Path Planning latency = %lf ms. \033[32m", plan_time);

  ROS_INFO("\033[35m[Planner] --- <Planner finished> --- \033[35m");

  // ! Coverage Evaluation
  if (v1 == true)
  {
    pathCoverageEval();
  }

  return true;
}

void FCPlanner_PP::viewpointGeneration()
{
  auto vpg_t1 = chrono::high_resolution_clock::now();

  pcl::PointCloud<pcl::PointXYZ>::Ptr basic_vps(new pcl::PointCloud<pcl::PointXYZ>);
  map<pcl::PointXYZ, int, pclCompareFunc> vp_search_sub_;
  all_safe_normal_vps.reset(new pcl::PointCloud<pcl::PointNormal>);
  vector<int> origin_sub_ids;

  pcl::PointNormal pt_;
  Eigen::Vector3d pt_vec, normal_dir, vp_vec;
  pcl::PointXYZ vp;
  pcl::PointNormal vp_fov_normal;
  Eigen::Vector3d safe_check_pt;
  for (int i=0; i<(int)skeleton_operator->P.sub_space_ptnr.size(); ++i)
  {
    int sub_id = i;
    pcl::PointCloud<pcl::PointNormal>::Ptr seg_vps(new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr safe_vps_in_seg(new pcl::PointCloud<pcl::PointXYZ>);

    for (int j=0; j<(int)skeleton_operator->P.sub_space_ptnr[i]->points.size(); ++j)
    {
      if (PR.uncovered_surface_[i][j] == false)
      {
        continue;
      }

      pt_ = skeleton_operator->P.sub_space_ptnr[i]->points[j];
      pt_vec << pt_.x, pt_.y, pt_.z;
      PR.pt_sub_pairs[pt_vec] = sub_id;
      // ! /* normal-based sampling */
      if (PR.pt_normal_pairs.find(pt_vec) != PR.pt_normal_pairs.end())
        normal_dir = PR.pt_normal_pairs.find(pt_vec)->second;
      else
        continue;
    
      double qual_dist_vp = dist_vp;
      if (abs(normal_dir(2)) != 0)
      {
        double inner_dist_vp = (-1.0-pt_.z) / normal_dir(2);
        if (inner_dist_vp > 0 && inner_dist_vp < qual_dist_vp) qual_dist_vp = inner_dist_vp - 1.0;
      }

      vp.x = pt_.x + qual_dist_vp * normal_dir(0);
      vp.y = pt_.y + qual_dist_vp * normal_dir(1);
      vp.z = pt_.z + qual_dist_vp * normal_dir(2);

      Eigen::Vector3d vp_det(vp.x, vp.y, vp.z);
      Eigen::Vector3f vp_dir(normal_dir(0), normal_dir(1), normal_dir(2));
      bool det = visibility_st::inlierCheck(vp_det, this->scene_);
      if (det == true)
        continue;

      if (zFlag == true)
      {
        if (pt_.z > 0.0)
        {
          if (vp.z > GroundZPos + safeHeight)
          {
            vp_vec(0) = vp.x; vp_vec(1) = vp.y; vp_vec(2) = vp.z;
            basic_vps->points.push_back(vp);
            vp_search_sub_[vp] = sub_id;

            vp_fov_normal.x = vp.x;
            vp_fov_normal.y = vp.y;
            vp_fov_normal.z = vp.z;
            vp_fov_normal.normal_x = -normal_dir(0);
            vp_fov_normal.normal_y = -normal_dir(1);
            vp_fov_normal.normal_z = -normal_dir(2);

            seg_vps->points.push_back(vp_fov_normal);
          }
          else
          {
            vp_vec(0) = vp.x; vp_vec(1) = vp.y; vp_vec(2) = GroundZPos + safeHeight;
            pcl::PointXYZ tmp_vp = vp;
            tmp_vp.z = vp_vec(2);
            basic_vps->points.push_back(tmp_vp);
            vp_search_sub_[tmp_vp] = sub_id;

            vp_fov_normal.x = vp_vec(0);
            vp_fov_normal.y = vp_vec(1);
            vp_fov_normal.z = vp_vec(2);
            Eigen::Vector3d groundsite(pt_.x, pt_.y, pt_.z);
            Eigen::Vector3d new_dir = (groundsite - vp_vec).normalized();
            double z_new = sqrt(pow(new_dir(2), 2) * (pow(normal_dir(0), 2) + pow(normal_dir(1), 2)) / (pow(new_dir(0), 2) + pow(new_dir(1), 2)));
            z_new = new_dir(2) > 0 ? z_new : -z_new;
            vp_fov_normal.normal_x = -normal_dir(0);
            vp_fov_normal.normal_y = -normal_dir(1);
            vp_fov_normal.normal_z = z_new;

            Eigen::Vector3f check_dir(normal_dir(0), normal_dir(1), -z_new);
            bool inside = visibility_st::inlierCheck(vp_vec, this->scene_);
          
            if (inside == false)
              seg_vps->points.push_back(vp_fov_normal);
          }
        }
      }
      if (zFlag == false)
      {
        vp_vec(0) = vp.x; vp_vec(1) = vp.y; vp_vec(2) = vp.z;
        basic_vps->points.push_back(vp);
        vp_search_sub_[vp] = sub_id;

        vp_fov_normal.x = vp.x;
        vp_fov_normal.y = vp.y;
        vp_fov_normal.z = vp.z;
        vp_fov_normal.normal_x = -normal_dir(0);
        vp_fov_normal.normal_y = -normal_dir(1);
        vp_fov_normal.normal_z = -normal_dir(2);

        seg_vps->points.push_back(vp_fov_normal);
      }
    }

    // * Safety Check
    pcl::PointXYZ safe_vp;
    Eigen::Vector3d filter_vp_, filter_dir_;
    for (auto vp : seg_vps->points)
    {
      bool safe_flag = true;

      Eigen::Vector3d vp_position(vp.x, vp.y, vp.z);
      Eigen::Vector3d vp_orientation(vp.normal_x, vp.normal_y, vp.normal_z);
      bool inside = visibility_st::inlierCheck(vp_position, this->scene_);
      if (inside == true) continue;

      safe_check_pt = vp_position;
      safe_flag = HCMap->safety_check(safe_check_pt);

      if (safe_flag == true)
      {
        filter_vp_(0) = vp.x; filter_vp_(1) = vp.y; filter_vp_(2) = vp.z;
        filter_dir_(0) = vp.normal_x;
        filter_dir_(1) = vp.normal_y;
        filter_dir_(2) = vp.normal_z;

        safe_vp.x = vp.x; safe_vp.y = vp.y; safe_vp.z = vp.z;
        safe_vps_in_seg->points.push_back(safe_vp);
        all_safe_normal_vps->points.push_back(vp);
        origin_sub_ids.push_back(sub_id);
      }
    }

    // * Determine Direction
    Eigen::MatrixXd pose_set;
    Eigen::VectorXd pose;
    Eigen::Vector3d sampled_vp;

    pose_set.resize(safe_vps_in_seg->points.size(), 5);

    PR.sub_vps_inflate[sub_id] = safe_vps_in_seg;
    PR.sub_vps_pose[sub_id] = pose_set;
  }

  // ? tips: small 'vpm_init_size_' for faster computation
  int upper_init = this->vpm_init_size_;
  if (upper_init > 0 && (int)all_safe_normal_vps->points.size() > upper_init)
  {
    const int n = (int)all_safe_normal_vps->points.size();
    const int k = upper_init;

    pcl::PointCloud<pcl::PointNormal>::Ptr fps_vps(new pcl::PointCloud<pcl::PointNormal>);
    fps_vps->points.reserve(k);

    // Deterministic farthest point sampling seed: candidate closest to centroid.
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const auto& p : all_safe_normal_vps->points)
      centroid += Eigen::Vector3d(p.x, p.y, p.z);
    centroid /= (double)n;

    int first_id = 0;
    double best_centroid_dist = std::numeric_limits<double>::max();
    for (int i=0; i<n; ++i)
    {
      const auto& p = all_safe_normal_vps->points[i];
      Eigen::Vector3d pos(p.x, p.y, p.z);
      double dist2 = (pos - centroid).squaredNorm();
      if (dist2 < best_centroid_dist)
      {
        best_centroid_dist = dist2;
        first_id = i;
      }
    }

    vector<char> selected(n, 0);
    vector<double> min_dist2(n, std::numeric_limits<double>::max());

    int cur_id = first_id;
    for (int iter=0; iter<k; ++iter)
    {
      selected[cur_id] = 1;
      fps_vps->points.push_back(all_safe_normal_vps->points[cur_id]);

      const auto& cur = all_safe_normal_vps->points[cur_id];
      Eigen::Vector3d cur_pos(cur.x, cur.y, cur.z);

      int next_id = -1;
      double max_min_dist2 = -1.0;
      for (int i=0; i<n; ++i)
      {
        if (selected[i]) continue;

        const auto& p = all_safe_normal_vps->points[i];
        Eigen::Vector3d pos(p.x, p.y, p.z);
        double dist2 = (pos - cur_pos).squaredNorm();
        if (dist2 < min_dist2[i]) min_dist2[i] = dist2;

        if (min_dist2[i] > max_min_dist2)
        {
          max_min_dist2 = min_dist2[i];
          next_id = i;
        }
      }

      if (next_id < 0) break;
      cur_id = next_id;
    }

    fps_vps->width = fps_vps->points.size();
    fps_vps->height = 1;
    fps_vps->is_dense = true;
    all_safe_normal_vps = fps_vps;
  }

  PR.vps_set_.clear();

  vector<SingleViewpoint> updated_vps;
  viewpoint_manager_->reset();
  viewpoint_manager_->setScene(mesh_V_, mesh_F_);
  viewpoint_manager_->setMapPointCloud(occ_model, HCMap);
  viewpoint_manager_->setModel(planning_target_surf_); // <change it as -> current uncovered area = prediction - inspected>
  viewpoint_manager_->setNormals(PR.pt_normal_pairs);
  viewpoint_manager_->setInitViewpoints(all_safe_normal_vps);
  if ((int)all_safe_normal_vps->points.size() > 0)
  {
    viewpoint_manager_->updateViewpoints();
    viewpoint_manager_->getUpdatedViewpoints(updated_vps);
    viewpoint_manager_->getUncoveredModelCloud(uncovered_area); 

    pcl::KdTreeFLANN<pcl::PointXYZ> basic_vps_tree_;
    basic_vps_tree_.setInputCloud(basic_vps);
    for (auto updated_vp : updated_vps)
    {
      Viewpoint final_vp;
      final_vp.vp_id = updated_vp.vp_id;
      final_vp.pose = updated_vp.pose;
      final_vp.vox_count = updated_vp.vox_count;

      Eigen::Vector3d position = updated_vp.pose.head(3);
      bool inside = visibility_st::inlierCheck(position, this->scene_);
      if (inside == true)
        continue;

      if (this->zFlag == true)
      {
        if (position(2) < GroundZPos + safeHeight)
          continue;
      }

      pcl::PointXYZ pt;
      pt.x = updated_vp.pose(0); pt.y = updated_vp.pose(1); pt.z = updated_vp.pose(2);
      vector<int> nearest;
      vector<float> k_sqr_distances;
      basic_vps_tree_.nearestKSearch(pt, 1, nearest, k_sqr_distances);
      if (vp_search_sub_.find(basic_vps->points[nearest[0]]) != vp_search_sub_.end())
        final_vp.sub_id = vp_search_sub_.find(basic_vps->points[nearest[0]])->second;
      else
        ROS_ERROR("Viewpoint not found in sub-space!");

      PR.vps_set_.push_back(final_vp);
    }

    // * Minimal Viewpoints Set -> All Active Viewpoints
    activeViewpoints();
  }

  // * Viewpoints Clustering
  decompViewpoints(PR.final_sub_vps_pairs);

  ROS_INFO("\033[36m[ViewpointManager] viewpoints number = %d. \033[32m", this->viewpointNum);

  auto vpg_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> vpg_ms = vpg_t2 - vpg_t1;
  double vpg_time = (double)vpg_ms.count();
  ROS_INFO("\033[33m[Planner] viewpoint generation time = %lf ms. \033[32m", vpg_time);
}

void FCPlanner_PP::decompPlanning()
{
  auto t1 = chrono::high_resolution_clock::now();

  // * step 1: Find the end direction of prior path
  if ((int)PR.prior_path.size() > 1)
  {
    Eigen::Vector3d last_dir = (PR.prior_path.back().head(3) - PR.prior_path[PR.prior_path.size()-2].head(3)).normalized();
    this->last_vel_ = this->vmax_ * last_dir;
  }
  else
    this->last_vel_ = this->start_vel_;

  // * step 2: Find the next subspace
  vector<vector<int>> reorder_decomp_vps;
  vector<int> reused_sub_idx;

  if (PR.LAST_reused_viewpoints_.empty())
  {
    int next_sub_idx = -1;
    double time_cost = 1e6;
    for (int i=0; i<(int)PR.decomp_vps_.size(); ++i)
      for (int j=0; j<(int)PR.decomp_vps_[i].size(); ++j)
      {
        Eigen::Vector3d cur_pos = PR.total_viewpoints_[PR.decomp_vps_[i][j]].head(3);
        double cost = path_tools::estPathTime(this->last_vel_, this->updated_pos_, cur_pos, this->vmax_, this->amean_);
        next_sub_idx = cost < time_cost? i : next_sub_idx;
        time_cost = cost < time_cost? cost : time_cost;
      }
  
    if (next_sub_idx != -1) reused_sub_idx.push_back(next_sub_idx);
  }
  else
  {
    int first_sub_idx = PR.decomp_vps_.size() - PR.LAST_reused_viewpoints_.size();
    for (int i=0; i<(int)PR.LAST_reused_viewpoints_.size(); ++i)
      reused_sub_idx.push_back(first_sub_idx+i);
  }

  for (int i=0; i<(int)reused_sub_idx.size(); ++i)
    reorder_decomp_vps.push_back(PR.decomp_vps_[reused_sub_idx[i]]);

  unordered_set<int> reused_sub_idx_set(reused_sub_idx.begin(), reused_sub_idx.end());
  for (int i=0; i<(int)PR.decomp_vps_.size(); ++i)
  {
    if (reused_sub_idx_set.find(i) == reused_sub_idx_set.end()) reorder_decomp_vps.push_back(PR.decomp_vps_[i]);
  }

  PR.decomp_vps_.clear();
  PR.decomp_vps_ = reorder_decomp_vps;

  // * step 3: Find global sequence consistent with the last time
  int updated_reused_sets = (int)reused_sub_idx.size();

  ROS_ERROR("match num prior size: %d", (int)PR.prior_path.size());
  ROS_ERROR("effective reused viewpoints: %d", this->reused_viewpoints_num_);

  int match_k = 0;
  if ((int)PR.prior_path.size() == 1 && this->reused_viewpoints_num_ == 0)
  {
    ROS_WARN("open chamfer matching...");
    PR.prior_path = PR.prior_backup_;
    PR.prior_wp_indicators_ = PR.prior_indi_backup_;
  }

  if (this->ground_priority_) this->match_open_ = 0;
  else this->match_open_ += 1;

  if (this->match_open_ > 1) match_k = this->match_num_;
  else match_k = 0;

  auto t1_step3 = chrono::high_resolution_clock::now();

  this->solver_->decompGlobalSeq(PR.total_viewpoints_, PR.decomp_vps_, PR.LAST_total_viewpoints_, PR.LAST_decomp_vps_, PR.LAST_decomp_global_seq_, PR.LAST_reused_sets_, updated_reused_sets, match_k, PR.decomp_global_seq_, PR.decomp_global_seq_path_);

  auto t2_step3 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> ms_step3 = t2_step3 - t1_step3;
  double time_step3 = (double)ms_step3.count();
  ROS_INFO("\033[33m[Planner] decomp global seq time = %lf ms.\033[32m", time_step3);

  PR.decomp_global_seq_path_.insert(PR.decomp_global_seq_path_.begin(), this->updated_pos_);

  // * step 4: Subspace path planning
  vector<Eigen::Vector3d> seq_path_wo_up = {PR.decomp_global_seq_path_.begin()+1, PR.decomp_global_seq_path_.end()};

  auto t1_step4 = chrono::high_resolution_clock::now();

  PR.decomp_full_path_.clear();
  PR.decomp_full_indicators_.clear();
  this->solver_->decompGlobalPath(this->updated_pose_, this->last_vel_, PR.prior_remaining_, PR.total_viewpoints_, PR.decomp_vps_, PR.decomp_global_seq_, seq_path_wo_up, updated_reused_sets, PR.decomp_full_path_, PR.decomp_full_indicators_);

  auto t2_step4 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> ms_step4 = t2_step4 - t1_step4;
  double time_step4 = (double)ms_step4.count();
  ROS_INFO("\033[33m[Planner] decomp global path time = %lf ms.\033[32m", time_step4);

  // * step 5: Merge prior path
  if ((int)PR.prior_path.size() > 1)
  {
    vector<Eigen::VectorXd> updated_full_path;
    vector<bool> updated_indicators;
    updated_full_path.insert(updated_full_path.end(), PR.prior_path.begin(), PR.prior_path.end()-1);
    updated_indicators.insert(updated_indicators.end(), PR.prior_wp_indicators_.begin(), PR.prior_wp_indicators_.end()-1);
    updated_full_path.insert(updated_full_path.end(), PR.decomp_full_path_.begin(), PR.decomp_full_path_.end());
    updated_indicators.insert(updated_indicators.end(), PR.decomp_full_indicators_.begin(), PR.decomp_full_indicators_.end());

    PR.decomp_full_path_.clear();
    PR.decomp_full_path_ = updated_full_path;
    PR.decomp_full_indicators_.clear();
    PR.decomp_full_indicators_ = updated_indicators;
  }

  // * step 6: Interpolate the path
  this->decompInterpolatePath(dist_upper_, PR.decomp_full_path_, PR.decomp_full_indicators_);
  vector<Eigen::VectorXd> safe_full_path;
  vector<bool> safe_indicators;
  for (int i=0; i<(int)PR.decomp_full_path_.size(); ++i)
  {
    Eigen::Vector3d pos = PR.decomp_full_path_[i].head(3);
    if (this->HCMap->getInflateOccupancy(pos) == 0)
    {
      safe_full_path.push_back(PR.decomp_full_path_[i]);
      safe_indicators.push_back(PR.decomp_full_indicators_[i]);
    }
  }
  PR.decomp_full_path_ = safe_full_path;
  PR.decomp_full_indicators_ = safe_indicators;

  // * step 7: Update LAST (history) info
  vector<vector<Eigen::VectorXd>> reused_viewpoints;
  vector<Eigen::VectorXd> remaining_viewpoints;
  vector<vector<vector<Eigen::Vector3d>>> reused_observations;
  vector<vector<Eigen::Vector3d>> remaining_observations;
  findReusedViewpoints(this->tsp_horizon_, reused_viewpoints, remaining_viewpoints, reused_observations, remaining_observations);

  PR.LAST_decomp_vps_.clear();
  PR.LAST_total_viewpoints_.clear();
  PR.LAST_decomp_global_seq_.clear();
  PR.LAST_reused_viewpoints_.clear();
  PR.LAST_reused_sets_ = 0;
  PR.LAST_remaining_viewpoints_.clear();
  PR.LAST_reused_observations_.clear();
  PR.LAST_remaining_observations_.clear();
  PR.LAST_decomp_vps_ = PR.decomp_vps_;
  PR.LAST_total_viewpoints_ = PR.total_viewpoints_;
  PR.LAST_decomp_global_seq_ = PR.decomp_global_seq_;
  PR.LAST_remaining_viewpoints_ = remaining_viewpoints;
  PR.LAST_remaining_observations_ = remaining_observations;
  if (!this->ground_priority_)
  {
    PR.LAST_reused_viewpoints_ = reused_viewpoints;
    PR.LAST_reused_sets_ = (int)PR.LAST_reused_viewpoints_.size();
    PR.LAST_reused_observations_ = reused_observations;
  }

  PR.FullPath_.clear();
  PR.waypoints_indicators_.clear();

  PR.FullPath_ = PR.decomp_full_path_;
  PR.waypoints_indicators_ = PR.decomp_full_indicators_;

  auto t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> ms = t2 - t1;
  double time = (double)ms.count();
  ROS_INFO("\033[33m[Planner] hierarchical decomp planning time = %lf ms.\033[32m", time);

  return;
}

// ! /* --------------------- Tools --------------------- */

void FCPlanner_PP::vis_Callback(const ros::TimerEvent& e)
{
  if (visFlag == false)
  {

    if (PR.model != nullptr && uncovered_surf_ != nullptr)
    {
      pcl::PointCloud<pcl::PointXYZ> pcloud;
      pcloud = *PR.model;
    
      vis_utils_->publishSurface(pcloud);

      pcl::PointCloud<pcl::PointXYZ> covered_cloud;
      covered_cloud = *uncovered_surf_;

      vis_utils_->publishUncovered(covered_cloud);
    }

    if ((int)uncovered_area.size() != 0)
    {
      vis_utils_->publishGlobalUncovered(uncovered_area);
    }

    if ((int)this->cur_exec_traj_.size() != 0)
    {
      vector<Eigen::VectorXd> exec_part = this->cur_exec_traj_;
      vis_utils_->publishExecPart(exec_part);
    }

    if (!PR.sub_centroids_.empty() && !PR.global_seq_.empty() && !PR.global_boundary_id_.empty() && !PR.final_sub_vps_pairs.empty())
    {
      vis_utils_->publishGlobalSeq(current_pos_, PR.sub_centroids_, PR.global_seq_);
      vis_utils_->publishGlobalBoundary(current_pos_, PR.global_boundary_id_, PR.final_sub_vps_pairs, PR.global_seq_);
    }

    if (!PR.connectJoints.empty() && !PR.innerVps.empty())
      vis_utils_->publishJointSphere(PR.connectJoints, PR.searchRange, PR.innerVps);

    if ((int)PR.prior_path.size() > 1)
      vis_utils_->publishBeforeConsistencyPath(PR.prior_path);

    if ((int)PR.FullPath_.size() > 0)
    {
      // * updated viewpoints in each sub-space
      map<int, vector<vector<Eigen::Vector3d>>> sub_list1, sub_list2;
      map<int, vector<double>> sub_yaws;
      vector<Eigen::Vector3d> l1_, l2_;
      Eigen::Vector3d pos_;
      double pitch_, yaw_;
      int s_id;
      for (auto &sub_vps : PR.final_sub_vps_pairs)
      {
        s_id = sub_vps.first;
        for (int i = 0; i < (int)sub_vps.second.size(); ++i)
        {

          pos_(0) = sub_vps.second[i](0);
          pos_(1) = sub_vps.second[i](1);
          pos_(2) = sub_vps.second[i](2);
          pitch_ = sub_vps.second[i](3);
          yaw_ = sub_vps.second[i](4);
          percep_utils_->setPose_PY(pos_, pitch_, yaw_);
          percep_utils_->getFOV_PY(l1_, l2_);

          sub_list1[s_id].push_back(l1_);
          sub_list2[s_id].push_back(l2_);
          sub_yaws[s_id].push_back(yaw_);
        }
      }

      vis_utils_->publishFinalFOV(sub_list1, sub_list2, sub_yaws);
      vis_utils_->publishHCOPPPath(PR.FullPath_);
    }

    // if (all_safe_normal_vps != nullptr)
    // {
    //   pcl::PointCloud<pcl::PointXYZ> pts;
    //   pcl::PointCloud<pcl::Normal> normals;
    //   for (auto pt : all_safe_normal_vps->points)
    //   {
    //     pcl::PointXYZ p;
    //     p.x = pt.x;
    //     p.y = pt.y;
    //     p.z = pt.z;
    //     pts.push_back(p);
    //     pcl::Normal n;
    //     n.normal_x = pt.normal_x;
    //     n.normal_y = pt.normal_y;
    //     n.normal_z = pt.normal_z;
    //     normals.push_back(n);
    //   }

    //   vis_utils_->publishRevisedNormal(pts, normals);
    // }

    if (!PR.local_paths_viewpts_.empty())
      vis_utils_->publishLocalPath(PR.local_paths_viewpts_);
    if ((int)PR.next_sub_inliers.size() > 0 && (int)PR.this_time_last_4_selection.size() > 0)
      vis_utils_->publishGlobalNextSubConsistency(PR.this_time_last_4_selection, PR.next_sub_inliers);
    if (!PR.total_viewpoints_.empty())
      vis_utils_->publishGlobalVPVisGraph(PR.total_viewpoints_, PR.vp_vis_graph_, PR.decomp_vps_);
    if (!PR.decomp_global_seq_path_.empty())
      vis_utils_->publishDecompGlobalPath(PR.decomp_global_seq_path_);

    if (PR.safe_g_vps_ != nullptr)
    {
      pcl::PointCloud<pcl::PointXYZ> safe_pts = *PR.safe_g_vps_;
      vis_utils_->publishInternal(safe_pts);
    }

    if (!PR.sampled_g_vps_set_.empty())
    {
      pcl::PointCloud<pcl::PointXYZ> sampled_pts;
      for (auto pt : PR.sampled_g_vps_set_)
      {
        pcl::PointXYZ p;
        p.x = pt(0);
        p.y = pt(1);
        p.z = pt(2);
        sampled_pts.push_back(p);
      }
      vis_utils_->publishOccupied(sampled_pts);
    }
  }

  if (mode_ == "debug")
  {
    if ((int)HCMap->hcmd_->internal_cloud.points.size() > 0)
      vis_utils_->publishInternal(HCMap->hcmd_->internal_cloud);
    if ((int)HCMap->hcmd_->occ_cloud.points.size() > 0)
      vis_utils_->publishOccupied(HCMap->hcmd_->occ_cloud);
  }

  if (visFlag == true)
  {
    // * coverage path visualization
    vector<Eigen::VectorXd> vis_full_path(PR.FullPath_.begin() + 1, PR.FullPath_.end());
  
    // * publish visualization
    vis_utils_->publishInitVps(all_safe_normal_vps);
    vis_utils_->publishJointSphere(PR.connectJoints, PR.searchRange, PR.innerVps);
  }
}

void FCPlanner_PP::execTrajCallback(const quadrotor_msgs::EigenVectorArrayConstPtr& msg)
{
  const quadrotor_msgs::EigenVectorArray &traj_msg = *msg;

  this->cur_exec_traj_.clear();
  for (const auto& array : traj_msg.vectors)
  {
    Eigen::VectorXd vec(array.data.size());
    for (size_t i = 0; i < array.data.size(); ++i)
    {
      vec[i] = array.data[i];
    }
    this->cur_exec_traj_.push_back(vec);
  }

  return;
}

void FCPlanner_PP::bevCallback(const quadrotor_msgs::BEVBoxConstPtr& msg)
{
  const quadrotor_msgs::BEVBox &bev_msg = *msg;

  this->bev_seg_ = bev_msg.en_bev_box == 1 ? true : false;

  if (!this->bev_seg_) return;

  this->bev_min_x_ = bev_msg.min_x;
  this->bev_max_x_ = bev_msg.max_x;
  this->bev_min_y_ = bev_msg.min_y;
  this->bev_max_y_ = bev_msg.max_y;

  return;
}

void FCPlanner_PP::setMSB()
{
  auto sc_t1 = chrono::high_resolution_clock::now();

  HCMap->resetHCMap(occ_model);
  if (this->bev_seg_) HCMap->resetHCVirtualBound(this->bev_min_x_, this->bev_max_x_, this->bev_min_y_, this->bev_max_y_);

  // * Set RayCaster
  solver_->setRayCaster(HCMap->hcmp_->resolution_, HCMap->hcmp_->map_origin_);
  solver_->setRayCasterReal(HCMap->mp_->resolution_, HCMap->mp_->map_origin_);
  raycaster_->setParams(HCMap->mp_->resolution_, HCMap->mp_->map_origin_);

  auto sc_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> sc_ms = sc_t2 - sc_t1;
  double sc_time = (double)sc_ms.count();
  ROS_INFO("\033[33m[Planner] set module time = %lf ms.\033[32m", sc_time);

  return;
}

void FCPlanner_PP::mapResults()
{
  if (mode_ == "debug")
  {
    pcl::PointXYZ ipt;
    for (int x = HCMap->hcmp_->box_min_(0) /* + 1 */; x < HCMap->hcmp_->box_max_(0); ++x)
      for (int y = HCMap->hcmp_->box_min_(1) /* + 1 */; y < HCMap->hcmp_->box_max_(1); ++y)
        for (int z = HCMap->hcmp_->box_min_(2) /* + 1 */; z < HCMap->hcmp_->box_max_(2); ++z)
        {
          if (!HCMap->isInMap_hc(Eigen::Vector3i(x, y, z))) continue;

          if (HCMap->hcmd_->occupancy_inflate_buffer_hc_[HCMap->toAddress_hc(x, y, z)] == 1)
          {
            Eigen::Vector3d pos;
            HCMap->indexToPos_hc(Eigen::Vector3i(x, y, z), pos);
            ipt.x = pos(0); ipt.y = pos(1); ipt.z = pos(2);
            HCMap->hcmd_->internal_cloud.push_back(ipt);
          }
        }
  
    cout << "Internal points: " << HCMap->hcmd_->internal_cloud.size() << endl;
  
    auto t1 = chrono::high_resolution_clock::now();
    pcl::PointXYZ occpt;
    for (int x = HCMap->hcmp_->box_min_(0) /* + 1 */; x < HCMap->hcmp_->box_max_(0); ++x)
      for (int y = HCMap->hcmp_->box_min_(1) /* + 1 */; y < HCMap->hcmp_->box_max_(1); ++y)
        for (int z = HCMap->hcmp_->box_min_(2) /* + 1 */; z < HCMap->hcmp_->box_max_(2); ++z)
        {
          if (!HCMap->isInMap_hc(Eigen::Vector3i(x, y, z))) continue;

          if (HCMap->hcmd_->occupancy_buffer_hc_[HCMap->toAddress_hc(x, y, z)] == 1)
          {
            Eigen::Vector3d pos;
            HCMap->indexToPos_hc(Eigen::Vector3i(x, y, z), pos);
            occpt.x = pos(0); occpt.y = pos(1); occpt.z = pos(2);
            HCMap->hcmd_->occ_cloud.push_back(occpt);
          }
        }
    auto t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms = t2 - t1;
    double time = (double)ms.count();
    ROS_INFO("\033[33m[Planner] Occupied points time = %lf ms.\033[32m", time);

    cout << "Occupied points: " << HCMap->hcmd_->occ_cloud.size() << endl;
  }
}

void FCPlanner_PP::uniformSampling()
{
  auto sc_t1 = chrono::high_resolution_clock::now();

  PR.ori_model.reset(new pcl::PointCloud<pcl::PointXYZ>);
  PR.model.reset(new pcl::PointCloud<pcl::PointXYZ>);
  PR.uncovered_surface_.clear();

  pcl::PointCloud<pcl::PointNormal>::Ptr model_w_nr(new pcl::PointCloud<pcl::PointNormal>);
  pcl::copyPointCloud(*skeleton_operator->P.ori_pts_, *PR.ori_model);
  PR.uncovered_surface_.reserve((int)skeleton_operator->P.sub_space_ptnr.size());

  for (int i = 0; i < (int)skeleton_operator->P.sub_space_ptnr.size(); ++i)
  {
    auto& sub_space_cloud = skeleton_operator->P.sub_space_ptnr[i];
    if (zFlag == true)
    {
      pcl::PassThrough<pcl::PointNormal> pass;
      pass.setInputCloud(sub_space_cloud);
      pass.setFilterFieldName("z");
      pass.setFilterLimits(GroundZPos, TopZPos);
      pcl::PointCloud<pcl::PointNormal>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointNormal>);
      pass.filter(*filtered_cloud);
      sub_space_cloud = filtered_cloud;
    }

    pcl::VoxelGrid<pcl::PointNormal> ds;
    ds.setInputCloud(sub_space_cloud);
    ds.setLeafSize(model_ds_size, model_ds_size, model_ds_size);
    pcl::PointCloud<pcl::PointNormal>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointNormal>);
    ds.filter(*downsampled_cloud);
    sub_space_cloud = downsampled_cloud;

    model_w_nr->insert(model_w_nr->end(), sub_space_cloud->begin(), sub_space_cloud->end());

    vector<bool> sub_uncovered(downsampled_cloud->points.size(), true);
    PR.uncovered_surface_.push_back(sub_uncovered);
  }

  pcl::copyPointCloud(*model_w_nr, *PR.model);

  auto sc_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> sc_ms = sc_t2 - sc_t1;
  double sc_time = (double)sc_ms.count();
  ROS_INFO("\033[33m[Planner] uniform sampling time = %lf ms.\033[32m", sc_time);

  return;
}

void FCPlanner_PP::reviseNormals()
{
  auto sc_t1 = chrono::high_resolution_clock::now();

  for (int i=0; i<(int)skeleton_operator->P.sub_space_ptnr.size(); ++i)
  {
    for (int j=0; j<(int)skeleton_operator->P.sub_space_ptnr[i]->points.size(); ++j)
    {
      if (PR.uncovered_surface_[i][j] == false) continue;
    
      pcl::PointNormal ptnr = skeleton_operator->P.sub_space_ptnr[i]->points[j];

      Eigen::Vector3d pt_vect(ptnr.x, ptnr.y, ptnr.z);
      Eigen::Vector3d nr_vect(ptnr.normal_x, ptnr.normal_y, ptnr.normal_z);
      nr_vect.normalize();
      Eigen::Vector3d nr_vect_rev = -nr_vect;
    
      Eigen::Vector3d candidate_a = pt_vect + this->vis_inflation_ * nr_vect;
      Eigen::Vector3d candidate_b = pt_vect + this->vis_inflation_ * nr_vect_rev;

      bool inside_a = visibility_st::inlierCheck(candidate_a, this->scene_);
      if (inside_a == false)
      {
        PR.pt_normal_pairs[pt_vect] = nr_vect;
        continue;
      }
    
      bool inside_b = visibility_st::inlierCheck(candidate_b, this->scene_);
      if (inside_b == false)
      {
        PR.pt_normal_pairs[pt_vect] = nr_vect_rev;
        continue;
      }
    }
  }

  auto sc_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> sc_ms = sc_t2 - sc_t1;
  double sc_time = (double)sc_ms.count();
  ROS_INFO("\033[33m[Planner] normal revision time = %lf ms.\033[32m", sc_time);

  return;
}

void FCPlanner_PP::activeViewpoints()
{
  valid_viewpoints.clear();
  good_vps_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

  Eigen::VectorXd valid_vp;
  int sub_space_id = -1;
  map<int, vector<Eigen::VectorXd>> init_vp_pairs;
  for (auto vp : PR.vps_set_)
  {
    sub_space_id = vp.sub_id;
    valid_vp = vp.pose;
    init_vp_pairs[sub_space_id].push_back(valid_vp);
  }

  // * Subspace Merging
  int minimum_vp = 0;
  vector<int> valid_cvx, bad_cvx;
  for (const auto &p : init_vp_pairs)
  {
    int sub_id = p.first;
    if ((int)p.second.size() > minimum_vp)
    {
      valid_cvx.push_back(sub_id);
    }
    else
      bad_cvx.push_back(sub_id);
  }

  vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> new_sub;
  vector<pcl::PointCloud<pcl::PointNormal>::Ptr> new_sub_ptnr;
  vector<Eigen::VectorXd> empty_vps = {};
  vector<vector<Eigen::VectorXd>> empty_waypts = {};
  int new_id = 0;
  for (auto gid:valid_cvx)
  {
    new_sub.push_back(skeleton_operator->P.sub_space_scale[gid]);
    new_sub_ptnr.push_back(skeleton_operator->P.sub_space_ptnr[gid]);
    PR.final_sub_vps_pairs[new_id] = init_vp_pairs[gid];
    valid_viewpoints.insert(valid_viewpoints.end(), init_vp_pairs[gid].begin(), init_vp_pairs[gid].end());
  
    // * Initialize full viewpoints and waypoints
    PR.local_paths_viewpts_[new_id] = empty_vps;
    PR.local_paths_waypts_[new_id] = empty_waypts;

    new_id++;
  }

  return;
}

/* uniform path interpolation */
void FCPlanner_PP::interFullPath(double dist_bound)
{
  vector<Eigen::VectorXd> temp_full_path;
  vector<int> waypt_ids;

  temp_full_path.push_back(PR.FullPath_[0]);
  int temp_id = 1;
  for (int i = 1; i < (int)PR.FullPath_.size(); ++i)
  {
    Eigen::Vector3d lastPos = PR.FullPath_[i - 1].head(3);
    Eigen::Vector3d currPos = PR.FullPath_[i].head(3);
    double dist = (lastPos - currPos).norm();
    if (dist > dist_bound)
    {
      int piece = ceil(dist / dist_bound);
      double inter_dist = dist / (double)piece;
      vector<Eigen::Vector3d> inter_waypts;
      vector<Eigen::VectorXd> updated_waypts;
      for (int j = 0; j < piece + 1; ++j)
      {
        Eigen::Vector3d interPos = lastPos + (double)j * inter_dist * (currPos - lastPos).normalized();
        inter_waypts.push_back(interPos);
      }
      solver_->AngleInterpolation(PR.FullPath_[i - 1], PR.FullPath_[i], inter_waypts, updated_waypts);

      for (auto x : updated_waypts)
      {
        Eigen::Vector3d updatedPos = x.head(3);
        if ((updatedPos - lastPos).norm() > 1e-3 && (updatedPos - currPos).norm() > 1e-3)
        {
          temp_full_path.push_back(x);
          waypt_ids.push_back(temp_id);
          temp_id++;
        }
      }
    }

    temp_full_path.push_back(PR.FullPath_[i]);
    if (PR.waypoints_indicators_[i] == true)
      waypt_ids.push_back(temp_id);
    temp_id++;
  }

  PR.FullPath_.clear();
  PR.FullPath_ = temp_full_path;
  PR.waypoints_indicators_.clear();
  PR.waypoints_indicators_.resize(PR.FullPath_.size(), false);
  for (int i = 0; i < (int)waypt_ids.size(); ++i)
    PR.waypoints_indicators_[waypt_ids[i]] = true;
}

void FCPlanner_PP::cameraModelProjection(Eigen::VectorXd &pose, Eigen::Vector3d &point, double &xRes, double &yRes, Eigen::Vector2d &leftdown, double &distance, Eigen::Vector2i &inCam)
{
  Eigen::Vector3d pos_ = pose.head(3);
  distance = (point - pos_).norm();
  double pitch = pose(3), yaw = pose(4);
  Eigen::Matrix3d R_y, R_p;
  R_y << cos(yaw), -sin(yaw), 0.0, sin(yaw), cos(yaw), 0.0, 0.0, 0.0, 1.0;
  R_p << cos(pitch), 0.0, -sin(pitch), 0.0, 1.0, 0.0, sin(pitch), 0.0, cos(pitch);
  Eigen::Vector3d pointCam = R_p.inverse() * R_y.inverse() * (point - pos_);

  double camX = fx * pointCam(1) / (1000.0 * pointCam(0)) - leftdown(0);
  double camY = fy * pointCam(2) / (1000.0 * pointCam(0)) - leftdown(1);

  inCam(0) = floor(camX / xRes);
  inCam(1) = floor(camY / yRes);
}

void FCPlanner_PP::pathCoverageEval()
{
  auto cove_t1 = chrono::high_resolution_clock::now();

  vector<Eigen::VectorXd> HCOPPPath(PR.FullPath_.begin() + 1, PR.FullPath_.end());
  vector<bool> CoverState;
  CoverState.resize(Fullmodel->points.size(), false);
  for (auto v : HCOPPPath)
  {
    vector<int> tempCids;
    PinHoleCamera(v, tempCids, Fullmodel);
    for (auto x : tempCids)
      CoverState[x] = true;
  }
  int numTrue = 0;
  for (int s = 0; s < (int)CoverState.size(); ++s)
  {
    if (CoverState[s] == true)
    {
      numTrue++;
      visibleFullmodel->points.push_back(Fullmodel->points[s]);
    }
  }
  coverage = (double)numTrue / (double)Fullmodel->points.size();

  auto cove_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> cove_ms = cove_t2 - cove_t1;
  double cove_time = (double)cove_ms.count();
  ROS_INFO("\033[36m[CoverageAnalyzer] FC-Planner-v2 path coverage evaluation time = %lf s. \033[32m", cove_time / 1000.0);
}

void FCPlanner_PP::updateUncoveredArea()
{
  auto uncov_t1 = chrono::high_resolution_clock::now(); 

  // * If bev segmentation is enabled by human's prompt
  if (this->bev_seg_)
  {
    for (int i=0; i<(int)skeleton_operator->P.sub_space_ptnr.size(); ++i)
    {
      for (int j=0; j<(int)skeleton_operator->P.sub_space_ptnr[i]->points.size(); ++j)
      {
        pcl::PointNormal pt = skeleton_operator->P.sub_space_ptnr[i]->points[j];
        if (pt.x > this->bev_max_x_ || pt.x < this->bev_min_x_ || pt.y > this->bev_max_y_ || pt.y < this->bev_min_y_)
          PR.uncovered_surface_[i][j] = false;
      }
    }
  }

  // * Process executed trajectory
  this->cur_exec_traj_.push_back(this->start_pose_);
  for (int i=0; i<(int)skeleton_operator->P.sub_space_ptnr.size(); ++i)
  {
    for (int j=0; j<(int)skeleton_operator->P.sub_space_ptnr[i]->points.size(); ++j)
    {
      if (PR.uncovered_surface_[i][j] == true)
      {
        pcl::PointNormal pt = skeleton_operator->P.sub_space_ptnr[i]->points[j];

        for (int k=0; k<(int)this->cur_exec_traj_.size(); ++k)
        {
          Eigen::Vector3d vp_pos = this->cur_exec_traj_[k].head(3);
          Eigen::Vector3d pt_pos(pt.x, pt.y, pt.z);

          this->percep_utils_->setPose_PY(vp_pos, this->cur_exec_traj_[k](3), this->cur_exec_traj_[k](4));
          if (this->percep_utils_->insideFOV(pt_pos) == false)
            continue;

          bool visible = visibility_st::visibilityCheck(vp_pos, pt_pos, this->vis_inflation_, this->scene_);

          if (visible == true)
          {
            PR.uncovered_surface_[i][j] = false;
            break;
          }
        }
      }
    }
  }

  uncovered_surf_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  PR.uncovered_model_local_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  for (int i=0; i<(int)skeleton_operator->P.sub_space_ptnr.size(); ++i)
    for (int j=0; j<(int)skeleton_operator->P.sub_space_ptnr[i]->points.size(); ++j)
    {
      if (PR.uncovered_surface_[i][j] == true)
      {
        pcl::PointXYZ pt;
        pt.x = skeleton_operator->P.sub_space_ptnr[i]->points[j].x;
        pt.y = skeleton_operator->P.sub_space_ptnr[i]->points[j].y;
        pt.z = skeleton_operator->P.sub_space_ptnr[i]->points[j].z;
        uncovered_surf_->points.push_back(pt);
        PR.uncovered_model_local_->points.push_back(pt);
      }
    }

  double uncovered_rate = (1.0 - (double)uncovered_surf_->points.size()/ (double)PR.model->points.size())*100.0;
  stringstream stream;
  stream << fixed << setprecision(2) << uncovered_rate;
  string uncovered_rate_str = stream.str();

  ROS_INFO("\033[36m[CoverageAnalyzer] current coverage rate = %s %%, uncovered points : %d, total points: %d.\033[32m", uncovered_rate_str.c_str(), (int)uncovered_surf_->points.size(), (int)PR.model->points.size());

  // task suspension condition
  if (uncovered_rate > this->suspension_rate_)
  {
    this->suspension_ = true;
    return;
  }

  vector<Eigen::VectorXd> reverse_exec_traj = {this->start_pose_};
  reverse_exec_traj.insert(reverse_exec_traj.end(), this->cur_exec_traj_.rbegin(), this->cur_exec_traj_.rend());

  // * Process last time planning's viewpoints with overlaps
  PR.prior_path = {this->start_pose_};
  PR.prior_wp_indicators_ = {false};
  if (!PR.last_path_.empty() && this->global_consistency_ == true)
  { 
    double mean_interval = 0.0;
    if ((int)PR.last_path_.size() > 1)
    {
      for (int i=0; i<(int)PR.last_path_.size()-1; ++i)
        mean_interval += (PR.last_path_[i+1].head(3) - PR.last_path_[i].head(3)).norm();
      mean_interval /= (double)(PR.last_path_.size()-1);
    }
  
    // select the nearest point within an annulus
    vector<Eigen::VectorXd> candidates;
    vector<int> candidate_ids;
    double lower_range = mean_interval;
    for (int i=0; i<(int)PR.last_path_.size(); ++i)
    {
      double dist = (PR.last_path_[i].head(3) - this->current_pos_).norm();
    
      if (dist < this->overlap_last_ && dist > lower_range)
      {
        candidates.push_back(PR.last_path_[i]);
        candidate_ids.push_back(i);
      }
    }
  
    if (!candidates.empty())
    {
      double cost = 1e5, cur_cost = 0.0;
      int nearest_id = -1;

      if (this->en_inherit_)
      {
        string filename = ros::package::getPath("flyco_planner_manager") + "/inherit_data/inherit_point.yaml";

        YAML::Node point_config = YAML::LoadFile(filename);
        for (const auto& point : point_config["velocity"])
        {
          this->start_vel_(0) = point[0].as<double>();
          this->start_vel_(1) = point[1].as<double>();
          this->start_vel_(2) = point[2].as<double>();
        }

        this->en_inherit_ = false;
      }

      for (int i=0; i<(int)candidates.size(); ++i)
      {
        Eigen::Vector3d pos = candidates[i].head(3);
        cur_cost = path_tools::estPathTime(this->start_vel_, this->current_pos_, pos, this->vmax_, this->amean_);

        if (cur_cost < cost)
        {
          cost = cur_cost;
          nearest_id = i;
        }
      }

      bool need_check = false;
      double dist2start = 0.0;
      double safe_dist = this->drone_radius_ + 2*HCMap->hcmp_->resolution_;

      PR.prior_path.push_back(PR.last_path_[candidate_ids[nearest_id]]);
      PR.prior_wp_indicators_.push_back(PR.last_wp_indicators_[candidate_ids[nearest_id]]);
      double l_prior = (PR.prior_path.back().head(3) - PR.prior_path.front().head(3)).norm();
      for (int i=candidate_ids[nearest_id]+1; i<(int)PR.last_path_.size(); ++i)
      {
        need_check = false;
        for (auto p : reverse_exec_traj)
        {
          dist2start = (PR.last_path_[i].head(3) - p.head(3)).norm();
          if (dist2start < this->local_range_)
          {
            need_check = true;
            break;
          }
        }
        if (need_check)
        {
          bool update_suc = this->updateRealPose(PR.last_path_[i], safe_dist);
          if (!update_suc) continue;
        }

        double temp_length = (PR.last_path_[i].head(3) - PR.prior_path.back().head(3)).norm();
        l_prior += temp_length;

        if (this->ground_priority_ && l_prior > 5.0) break;

        PR.prior_wp_indicators_.push_back(PR.last_wp_indicators_[i]);
        PR.prior_path.push_back(PR.last_path_[i]);
      }
    }
  }

  // * Process uncovered area update
  double inflation = this->drone_radius_ + HCMap->hcmp_->resolution_;

  if ((int)PR.prior_path.size() > 1)
  {
    vector<int> effective_ids = {0};
    vector<Eigen::VectorXd> safe_prior_path = {PR.prior_path[0]};
    Eigen::Vector3d min_p, max_p;

    for (int i=1; i<(int)PR.prior_path.size(); ++i)
    {
      Eigen::Vector3d vp_pos = PR.prior_path[i].head(3);
      double pitch = PR.prior_path[i](3);
      double yaw = PR.prior_path[i](3);
      this->percep_utils_->setPose_PY(vp_pos, pitch, yaw);

      bool inside_mesh = visibility_st::inlierCheck(vp_pos, this->scene_);
      if (inside_mesh) continue;

      bool prior_safe = this->HCMap->getSafety(vp_pos, inflation, inflation, 0.5*inflation);
      if (prior_safe == false) continue;

      safe_prior_path.push_back(PR.prior_path[i]);

      int cover_num = 0;
      // ? check uncovered points
      for (int j=0; j<(int)skeleton_operator->P.sub_space_ptnr.size(); ++j)
        for (int k=0; k<(int)skeleton_operator->P.sub_space_ptnr[j]->points.size(); ++k)
        {
          if (PR.uncovered_surface_[j][k] == true)
          {
            pcl::PointNormal pt = skeleton_operator->P.sub_space_ptnr[j]->points[k];
            Eigen::Vector3d pt_pos(pt.x, pt.y, pt.z);
            if (this->percep_utils_->insideShrinkFOV(pt_pos) == false)
              continue;
          
            bool visible = visibility_st::visibilityCheck(vp_pos, pt_pos, this->vis_inflation_, this->scene_);
            if (visible == true)
            {
              PR.uncovered_surface_[j][k] = false;
              cover_num++;
            }
          }
        }
    
      if (cover_num > this->save_visible_lower_)
        effective_ids.push_back(i);
    }

    // ? update prior path
    PR.prior_backup_.clear();
    PR.prior_backup_ = safe_prior_path;
    PR.prior_indi_backup_.clear();
    PR.prior_indi_backup_ = vector<bool>((int)PR.prior_backup_.size(), false);

    vector<Eigen::VectorXd> updated_prior_path = {PR.prior_path[effective_ids.front()]};
    vector<bool> updated_wp_indicators = {PR.prior_wp_indicators_[effective_ids.front()]};

    double prior_length = 0.0;
    PR.prior_remaining_.clear(); 

    for (int i=0; i<(int)effective_ids.size()-1; ++i)
    {
      if (prior_length < this->overlap_last_)
      {
        int cur_id = effective_ids[i];
        int next_id = effective_ids[i+1];

        vector<Eigen::VectorXd> updated_inter_path;
        vector<bool> updated_inter_indi;
        solver_->findRealBridgePath(PR.prior_path[cur_id], PR.prior_path[next_id], cost_cn_, updated_inter_path, updated_inter_indi);

        if ((int)updated_inter_path.size() > 0)
        {
          updated_prior_path.insert(updated_prior_path.end(), updated_inter_path.begin(), updated_inter_path.end());
          updated_wp_indicators.insert(updated_wp_indicators.end(), updated_inter_indi.begin(), updated_inter_indi.end());
        }

        updated_prior_path.push_back(PR.prior_path[effective_ids[i+1]]);
        updated_wp_indicators.push_back(PR.prior_wp_indicators_[effective_ids[i+1]]);

        prior_length = 0.0;
        if (!this->ground_priority_)
        {
          for (int i=0; i<(int)updated_prior_path.size()-1; ++i)
            prior_length += (updated_prior_path[i+1].head(3) - updated_prior_path[i].head(3)).norm();
        }
      }
      else
      {
        if (!this->ground_priority_) PR.prior_remaining_.push_back(PR.prior_path[effective_ids[i+1]]);
      }
    }

    PR.prior_path.clear();
    PR.prior_wp_indicators_.clear();
    PR.prior_path = updated_prior_path;
    PR.prior_wp_indicators_ = updated_wp_indicators;
  }

  // * Update ground priority set
  if (this->ground_priority_) 
  {
    this->findGroundPrioritySet();
    this->updateGroundPrioritySet();
  }

  // * Process LAST reused viewpoints
  this->reused_viewpoints_num_ = 0;
  if (!PR.LAST_reused_viewpoints_.empty())
  { 
    vector<vector<Eigen::VectorXd>> updated_LAST_reused_viewpoints;
    Eigen::Vector3d min_corner, max_corner;
  
    bool need_check = false;
    double dist2start = 0.0;

    for (int i=0; i<(int)PR.LAST_reused_viewpoints_.size(); ++i)
    {
      vector<Eigen::VectorXd> updated_vp_set;
      for (int j=0; j<(int)PR.LAST_reused_viewpoints_[i].size(); ++j)
      {
        Eigen::Vector3d vp_pos = PR.LAST_reused_viewpoints_[i][j].head(3);
        double pitch = PR.LAST_reused_viewpoints_[i][j](3);
        double yaw = PR.LAST_reused_viewpoints_[i][j](4);
      
        // check current safety of this viewpoint
        bool inside_mesh = visibility_st::inlierCheck(vp_pos, this->scene_);
        if (inside_mesh) continue;

        bool hc_safe = this->HCMap->getHCSafety(vp_pos, inflation, inflation, 0.5*inflation);
        if (hc_safe == false) continue;

        bool reuse_safe = this->HCMap->getSafety(vp_pos, inflation, inflation, 0.5*inflation);
        if (reuse_safe == false) continue;

        need_check = false;
        for (auto p : reverse_exec_traj)
        {
          dist2start = (vp_pos - p.head(3)).norm();
          if (dist2start < this->local_range_)
          {
            need_check = true;
            break;
          }
        }
        if (need_check)
        {
          if (this->HCMap->getInflateOccupancy(vp_pos) == 1 || this->HCMap->getOccupancy(vp_pos) == SDFMap::OCCUPANCY::UNKNOWN) continue;
        }

        int see_count = 0;
        vector<Eigen::Vector3d> last_observations = PR.LAST_reused_observations_[i][j];
        vector<Eigen::Vector3d> cur_observations;
        for (int n=0; n<(int)skeleton_operator->P.sub_space_ptnr.size(); ++n)
        {
          for (int m=0; m<(int)skeleton_operator->P.sub_space_ptnr[n]->points.size(); ++m)
          {
            // if (PR.uncovered_surface_[n][m] == false) continue;

            pcl::PointNormal pt = skeleton_operator->P.sub_space_ptnr[n]->points[m];
            Eigen::Vector3d pt_pos(pt.x, pt.y, pt.z);

            this->percep_utils_->setPose_PY(vp_pos, pitch, yaw);
            if (this->percep_utils_->insideShrinkFOV(pt_pos) == false) continue;

            bool visible = visibility_st::visibilityCheck(vp_pos, pt_pos, this->vis_inflation_, this->scene_);
            if (visible == true) 
            {
              if (PR.uncovered_surface_[n][m] == true)
              {
                PR.uncovered_surface_[n][m] = false;
                see_count++;
              }
              cur_observations.push_back(pt_pos);
            }
          }
        }

        // ? add chamfer distance constraint
        double chamfer_distance_reused = solver_tools::computeBidirectionalChamfer(last_observations, cur_observations);
        if (chamfer_distance_reused > 3*this->model_ds_size) continue;
      
        if (see_count > this->save_visible_lower_) 
        {
          updated_vp_set.push_back(PR.LAST_reused_viewpoints_[i][j]);
          this->reused_viewpoints_num_++;
        }
      } 
      if (!updated_vp_set.empty()) updated_LAST_reused_viewpoints.push_back(updated_vp_set);
    }

    PR.LAST_reused_viewpoints_.clear();
    PR.LAST_reused_viewpoints_ = updated_LAST_reused_viewpoints;
  }

  // * Process LAST remaining viewpoints
  vector<Eigen::VectorXd> updated_LAST_remaining_viewpoints;
  if (!PR.LAST_remaining_viewpoints_.empty())
  {
    Eigen::Vector3d min_corner, max_corner;
    bool need_check = false;
    double dist2start = 0.0;
    for (int i=0; i<(int)PR.LAST_remaining_viewpoints_.size(); ++i)
    {
      Eigen::Vector3d vp_pos = PR.LAST_remaining_viewpoints_[i].head(3);
      double pitch = PR.LAST_remaining_viewpoints_[i](3);
      double yaw = PR.LAST_remaining_viewpoints_[i](4);

      // check current safety of this viewpoint
      bool inside_mesh = visibility_st::inlierCheck(vp_pos, this->scene_);
      if (inside_mesh) continue;

      bool hc_safe = this->HCMap->getHCSafety(vp_pos, inflation, inflation, 0.5*inflation);
      if (hc_safe == false) continue;

      bool reuse_safe = this->HCMap->getSafety(vp_pos, inflation, inflation, 0.5*inflation);
      if (reuse_safe == false) continue;

      need_check = false;
      for (auto p : reverse_exec_traj)
      {
        dist2start = (vp_pos - p.head(3)).norm();
        if (dist2start < this->local_range_)
        {
          need_check = true;
          break;
        }
      }
      if (need_check)
      {
        if (this->HCMap->getInflateOccupancy(vp_pos) == 1 || this->HCMap->getOccupancy(vp_pos) == SDFMap::OCCUPANCY::UNKNOWN) continue;
      }

      int see_count = 0;
      vector<Eigen::Vector3d> last_observations = PR.LAST_remaining_observations_[i];
      vector<Eigen::Vector3d> cur_observations;
      for (int n=0; n<(int)skeleton_operator->P.sub_space_ptnr.size(); ++n)
      {
        for (int m=0; m<(int)skeleton_operator->P.sub_space_ptnr[n]->points.size(); ++m)
        {
          pcl::PointNormal pt = skeleton_operator->P.sub_space_ptnr[n]->points[m];
          Eigen::Vector3d pt_pos(pt.x, pt.y, pt.z);

          this->percep_utils_->setPose_PY(vp_pos, pitch, yaw);
          if (this->percep_utils_->insideShrinkFOV(pt_pos) == false) continue;

          bool visible = visibility_st::visibilityCheck(vp_pos, pt_pos, this->vis_inflation_, this->scene_);
          if (visible == true) 
          {
            if (PR.uncovered_surface_[n][m] == true)
            {
              PR.uncovered_surface_[n][m] = false;
              see_count++;
            }
            cur_observations.push_back(pt_pos);
          }
        }
      }

      // ? add chamfer distance constraint
      double chamfer_distance = solver_tools::computeBidirectionalChamfer(last_observations, cur_observations);
      if (chamfer_distance > 3*this->model_ds_size) continue;

      if (see_count > this->save_visible_lower_) 
      {
        updated_LAST_remaining_viewpoints.push_back(PR.LAST_remaining_viewpoints_[i]);
      }
    }

    PR.LAST_remaining_viewpoints_.clear();
    PR.LAST_remaining_viewpoints_ = updated_LAST_remaining_viewpoints;
  }

  // * Update uncovered surface for generating remaining viewpoints
  planning_target_surf_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  for (int i=0; i<(int)skeleton_operator->P.sub_space_ptnr.size(); ++i)
    for (int j=0; j<(int)skeleton_operator->P.sub_space_ptnr[i]->points.size(); ++j)
    {
      if (PR.uncovered_surface_[i][j] == true)
      {
        pcl::PointXYZ pt;
        pt.x = skeleton_operator->P.sub_space_ptnr[i]->points[j].x;
        pt.y = skeleton_operator->P.sub_space_ptnr[i]->points[j].y;
        pt.z = skeleton_operator->P.sub_space_ptnr[i]->points[j].z;
        planning_target_surf_->points.push_back(pt);
      }
    }

  auto uncov_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> uncov_ms = uncov_t2 - uncov_t1;
  double uncov_time = (double)uncov_ms.count();
  ROS_INFO("\033[33m[Planner] uncovered area update time = %lf ms.\033[32m", uncov_time);

  return;
}

void FCPlanner_PP::findReusedViewpoints(const double& range, vector<vector<Eigen::VectorXd>>& reused_viewpoints, vector<Eigen::VectorXd>& remaining_viewpoints, vector<vector<vector<Eigen::Vector3d>>>& reused_observations, vector<vector<Eigen::Vector3d>>& remaining_observations)
{
  vector<Eigen::Vector3d> included_points = {this->updated_pos_};
  double traversal_dist = 0.0;

  // * Find the first set
  vector<Eigen::VectorXd> first_cluster_vps;
  Eigen::Vector3d first_centroid = Eigen::Vector3d::Zero();
  for (int i=0; i<(int)PR.decomp_vps_[PR.decomp_global_seq_.front()].size(); ++i)
  {
    first_centroid += PR.total_viewpoints_[PR.decomp_vps_[PR.decomp_global_seq_.front()][i]].head(3);
    first_cluster_vps.push_back(PR.total_viewpoints_[PR.decomp_vps_[PR.decomp_global_seq_.front()][i]]);
  }
  first_centroid /= (double)PR.decomp_vps_[PR.decomp_global_seq_.front()].size();
  traversal_dist = (first_centroid - included_points.back()).norm();
  reused_viewpoints.push_back(first_cluster_vps);
  included_points.push_back(first_centroid);

  vector<vector<Eigen::Vector3d>> set_obs(first_cluster_vps.size());
  for (int l=0; l<(int)first_cluster_vps.size(); ++l)
  {
    vector<Eigen::Vector3d> observations;
    Eigen::Vector3d vp_pos = first_cluster_vps[l].head(3);
    double pitch = first_cluster_vps[l](3), yaw = first_cluster_vps[l](4);

    for (int n=0; n<(int)skeleton_operator->P.sub_space_ptnr.size(); ++n)
    {
      for (int m=0; m<(int)skeleton_operator->P.sub_space_ptnr[n]->points.size(); ++m)
      {
        pcl::PointNormal pt = skeleton_operator->P.sub_space_ptnr[n]->points[m];
        Eigen::Vector3d pt_pos(pt.x, pt.y, pt.z);

        this->percep_utils_->setPose_PY(vp_pos, pitch, yaw);
        if (this->percep_utils_->insideShrinkFOV(pt_pos) == false) continue;
        bool visible = visibility_st::visibilityCheck(vp_pos, pt_pos, this->vis_inflation_, this->scene_);
        if (visible == true) observations.push_back(pt_pos);
      }
    }

    set_obs[l] = observations;
  }

  reused_observations.push_back(set_obs);

  // * Find the remaining sets
  for (int i=1; i<(int)PR.decomp_global_seq_.size(); ++i)
  {
    vector<Eigen::VectorXd> cur_cluster_vps;
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (int j=0; j<(int)PR.decomp_vps_[PR.decomp_global_seq_[i]].size(); ++j)
    {
      Eigen::Vector3d vp = PR.total_viewpoints_[PR.decomp_vps_[PR.decomp_global_seq_[i]][j]].head(3);
      centroid += vp;
      cur_cluster_vps.push_back(PR.total_viewpoints_[PR.decomp_vps_[PR.decomp_global_seq_[i]][j]]);
    }
    centroid /= (double)PR.decomp_vps_[PR.decomp_global_seq_[i]].size();
  
    traversal_dist += (centroid - included_points.back()).norm();

    if (traversal_dist < range && !this->ground_priority_)
    {
      reused_viewpoints.push_back(cur_cluster_vps);
      included_points.push_back(centroid);

      vector<vector<Eigen::Vector3d>> set_obs(cur_cluster_vps.size());
      for (int l=0; l<(int)cur_cluster_vps.size(); ++l)
      {
        vector<Eigen::Vector3d> observations;
        Eigen::Vector3d vp_pos = cur_cluster_vps[l].head(3);
        double pitch = cur_cluster_vps[l](3), yaw = cur_cluster_vps[l](4);

        for (int n=0; n<(int)skeleton_operator->P.sub_space_ptnr.size(); ++n)
        {
          for (int m=0; m<(int)skeleton_operator->P.sub_space_ptnr[n]->points.size(); ++m)
          {
            pcl::PointNormal pt = skeleton_operator->P.sub_space_ptnr[n]->points[m];
            Eigen::Vector3d pt_pos(pt.x, pt.y, pt.z);

            this->percep_utils_->setPose_PY(vp_pos, pitch, yaw);
            if (this->percep_utils_->insideShrinkFOV(pt_pos) == false) continue;
            bool visible = visibility_st::visibilityCheck(vp_pos, pt_pos, this->vis_inflation_, this->scene_);
            if (visible == true) observations.push_back(pt_pos);
          }
        }

        set_obs[l] = observations;
      }

      reused_observations.push_back(set_obs);
    }
    else
    {
      remaining_viewpoints.insert(remaining_viewpoints.end(), cur_cluster_vps.begin(), cur_cluster_vps.end());
      for (int k=0; k<(int)cur_cluster_vps.size(); ++k)
      {
        vector<Eigen::Vector3d> observations;
        Eigen::Vector3d vp_pos = cur_cluster_vps[k].head(3);
        double pitch = cur_cluster_vps[k](3), yaw = cur_cluster_vps[k](4);

        for (int n=0; n<(int)skeleton_operator->P.sub_space_ptnr.size(); ++n)
        {
          for (int m=0; m<(int)skeleton_operator->P.sub_space_ptnr[n]->points.size(); ++m)
          {
            pcl::PointNormal pt = skeleton_operator->P.sub_space_ptnr[n]->points[m];
            Eigen::Vector3d pt_pos(pt.x, pt.y, pt.z);

            this->percep_utils_->setPose_PY(vp_pos, pitch, yaw);
            if (this->percep_utils_->insideShrinkFOV(pt_pos) == false) continue;
            bool visible = visibility_st::visibilityCheck(vp_pos, pt_pos, this->vis_inflation_, this->scene_);
            if (visible == true) observations.push_back(pt_pos);
          }
        }

        remaining_observations.push_back(observations);
      }
    }
  }

  return;
}

void FCPlanner_PP::decompInterpolatePath(double dist_bound, vector<Eigen::VectorXd>& path, vector<bool>& indicators)
{
  vector<Eigen::VectorXd> temp_full_path;
  vector<bool> temp_indicators;

  temp_full_path.push_back(path[0]);
  temp_indicators.push_back(indicators[0]);
  for (int i = 1; i < (int)path.size(); ++i)
  {
    Eigen::Vector3d lastPos = path[i-1].head(3);
    Eigen::Vector3d currPos = path[i].head(3);
    double dist = (lastPos - currPos).norm();
    if (dist > dist_bound)
    {
      int piece = ceil(dist / dist_bound);
      double inter_dist = dist / (double)piece;
      vector<Eigen::Vector3d> inter_waypts;
      vector<Eigen::VectorXd> updated_waypts;
      for (int j = 0; j < piece + 1; ++j)
      {
        Eigen::Vector3d interPos = lastPos + (double)j * inter_dist * (currPos - lastPos).normalized();
        inter_waypts.push_back(interPos);
      }
      this->solver_->AngleInterpolation(path[i-1], path[i], inter_waypts, updated_waypts);

      for (auto x : updated_waypts)
      {
        Eigen::Vector3d updatedPos = x.head(3);
        if ((updatedPos - lastPos).norm() > 1e-3 && (updatedPos - currPos).norm() > 1e-3)
        {
          temp_full_path.push_back(x);
          temp_indicators.push_back(true);
        }
      }
    }

    temp_full_path.push_back(path[i]);
    temp_indicators.push_back(indicators[i]);
  }

  path.clear();
  path = temp_full_path;
  indicators.clear();
  indicators = temp_indicators;

  return;
}

void FCPlanner_PP::decompViewpoints(map<int, vector<Eigen::VectorXd>>& sub_vps)
{
  auto t1 = chrono::high_resolution_clock::now();

  PR.total_viewpoints_.clear();
  vector<int> viewpoint_sub_ids;
  size_t estimated_vp_num = PR.LAST_remaining_viewpoints_.size();
  for (const auto& pair : sub_vps)
    estimated_vp_num += pair.second.size();
  PR.total_viewpoints_.reserve(estimated_vp_num);
  viewpoint_sub_ids.reserve(estimated_vp_num);

  map<int, Eigen::Vector3d> subspace_centroids;
  map<int, int> subspace_counts;
  for (const auto& pair:sub_vps)
  {
    if (subspace_centroids.find(pair.first) == subspace_centroids.end())
    {
      subspace_centroids[pair.first] = Eigen::Vector3d::Zero();
      subspace_counts[pair.first] = 0;
    }

    for (const auto& vp : pair.second)
    {
      PR.total_viewpoints_.push_back(vp);
      viewpoint_sub_ids.push_back(pair.first);
      subspace_centroids[pair.first] += vp.head(3);
      subspace_counts[pair.first]++;
    }
  }

  for (auto& pair : subspace_centroids)
  {
    int cnt = subspace_counts[pair.first];
    if (cnt > 0) pair.second /= (double)cnt;
  }

  auto nearest_subspace_id = [&](const Eigen::Vector3d& pos) -> int
  {
    int best_sub_id = -1;
    double best_dist = MAX;
    for (const auto& pair : subspace_centroids)
    {
      double dist = (pos - pair.second).norm();
      if (dist < best_dist)
      {
        best_dist = dist;
        best_sub_id = pair.first;
      }
    }

    return best_sub_id;
  };

  // * add effective LAST remaining viewpoints
  if (!PR.LAST_remaining_viewpoints_.empty())
  {
    for (const auto& vp : PR.LAST_remaining_viewpoints_)
    {
      PR.total_viewpoints_.push_back(vp);
      viewpoint_sub_ids.push_back(nearest_subspace_id(vp.head(3)));
    }
  }

  // * construct visibility graph
  int vp_size = PR.total_viewpoints_.size();
  PR.vp_vis_graph_.resize(vp_size, vp_size);
  PR.vp_vis_graph_.setZero();

  for (int i=0; i<vp_size; ++i)
    for (int j=0; j<vp_size; ++j)
    {
      if (i>=j)
      {
        if (i==j) PR.vp_vis_graph_(i,j) = 1;
        else
        {
          Eigen::Vector3d p_i = PR.total_viewpoints_[i].head(3);
          Eigen::Vector3d p_j = PR.total_viewpoints_[j].head(3);
          bool visible = visibility_st::rayTracing(p_i, p_j, this->scene_);

          bool safe = true;
          Eigen::Vector3i idx;
          raycaster_->input(p_i, p_j);
          while (raycaster_->nextId(idx)) 
          {
            if (HCMap->getOccupancy(idx) == SDFMap::OCCUPANCY::OCCUPIED || HCMap->getInflateOccupancy(idx) == 1) 
            {
              safe = false;
              break;
            }
          
            if (safe == false)
              break;
          }

          bool double_see = false;
          if (visible && safe) double_see = true;

          PR.vp_vis_graph_(i,j) = double_see == true? 1 : 0;
          PR.vp_vis_graph_(j,i) = double_see == true? 1 : 0;
        }
      }
    }

  // * decomp viewpoints
  vector<Eigen::Vector3d> vps_pos;
  for (const auto& vp : PR.total_viewpoints_)
    vps_pos.push_back(vp.head(3));

  PR.decomp_vps_.clear();

  if (vp_size > 0)
  {
    // Keep the visibility graph as the hard convexity/safety constraint, while
    // using skeleton subspace ids only as a stable decomposition guide.  Each
    // local convex decomposition receives a submatrix of PR.vp_vis_graph_, so a
    // returned set is still required to be mutually visible and range-bounded.
    map<int, vector<int>> skeleton_bins;
    for (int i=0; i<vp_size; ++i)
    {
      int sub_id = i < (int)viewpoint_sub_ids.size() ? viewpoint_sub_ids[i] : -1;
      skeleton_bins[sub_id].push_back(i);
    }

    auto viewpoint_index_less = [&](const int a, const int b) -> bool
    {
      const Eigen::Vector3d& pa = vps_pos[a];
      const Eigen::Vector3d& pb = vps_pos[b];
      if (pa(0) != pb(0)) return pa(0) < pb(0);
      if (pa(1) != pb(1)) return pa(1) < pb(1);
      if (pa(2) != pb(2)) return pa(2) < pb(2);
      return a < b;
    };

    int guided_bin_num = 0;
    for (auto& bin_pair : skeleton_bins)
    {
      vector<int> global_indices = bin_pair.second;
      if (global_indices.empty()) continue;

      guided_bin_num++;
      stable_sort(global_indices.begin(), global_indices.end(), viewpoint_index_less);

      int local_num = (int)global_indices.size();
      vector<Eigen::Vector3d> local_points;
      local_points.reserve(local_num);
      for (auto idx : global_indices)
        local_points.push_back(vps_pos[idx]);

      Eigen::MatrixXi local_visibility;
      local_visibility.resize(local_num, local_num);
      for (int i=0; i<local_num; ++i)
        for (int j=0; j<local_num; ++j)
          local_visibility(i,j) = PR.vp_vis_graph_(global_indices[i], global_indices[j]);

      vector<vector<int>> local_decomp_vps = cvx_decomp::decompCvxSizeUni(local_visibility, local_points, this->max_cvx_range_);
      for (const auto& local_set : local_decomp_vps)
      {
        vector<int> global_set;
        global_set.reserve(local_set.size());
        for (auto local_idx : local_set)
        {
          if (local_idx >= 0 && local_idx < local_num)
            global_set.push_back(global_indices[local_idx]);
        }
        if (!global_set.empty()) PR.decomp_vps_.push_back(global_set);
      }
    }

    // Defensive fallback: preserve the original behavior if the guided path is
    // ever given an empty/invalid bin set.
    if (PR.decomp_vps_.empty())
      PR.decomp_vps_ = cvx_decomp::decompCvxSizeUni(PR.vp_vis_graph_, vps_pos, this->max_cvx_range_);
  
    vector<Eigen::Vector3d> decomp_vps_centroids;
    for (auto set : PR.decomp_vps_)
    {
      Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
      for (auto idx : set)
        centroid += vps_pos[idx];
      centroid /= (double)set.size();
      decomp_vps_centroids.push_back(centroid);
    }

    vector<vector<int>> updated_decomp_vps;
    for (auto set : PR.decomp_vps_)
    {
      if ((int)set.size() == 1)
      {
        Eigen::Vector3d vp_pos = vps_pos[set[0]];

        int connect_nbs = 0;
        for (auto centroid : decomp_vps_centroids)
        {
          double dist = (vp_pos - centroid).norm();
          if (dist < 2*this->max_cvx_range_) connect_nbs++;
        }

        // if (connect_nbs > 1) updated_decomp_vps.push_back(set);
        updated_decomp_vps.push_back(set);
      }
      else
      {
        updated_decomp_vps.push_back(set);
      }
    }
    PR.decomp_vps_ = updated_decomp_vps;

    // Temporal stabilization is intentionally set-level only: it reorders the
    // current convex sets to follow the previous global sequence when centroids
    // still match spatially, but it never changes any set membership and never
    // relaxes PR.vp_vis_graph_.
    struct OrderedDecompSet
    {
      vector<int> set;
      Eigen::Vector3d centroid;
      int sub_id;
      int last_rank;
      double last_dist;
      int original_id;
    };

    struct LastDecompSet
    {
      Eigen::Vector3d centroid;
      int rank;
    };

    auto set_centroid = [&](const vector<int>& set, const vector<Eigen::Vector3d>& positions) -> Eigen::Vector3d
    {
      Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
      if (set.empty()) return centroid;

      int valid_num = 0;
      for (auto idx : set)
      {
        if (idx >= 0 && idx < (int)positions.size())
        {
          centroid += positions[idx];
          valid_num++;
        }
      }
      if (valid_num > 0) centroid /= (double)valid_num;

      return centroid;
    };

    auto set_major_sub_id = [&](const vector<int>& set) -> int
    {
      map<int, int> sub_counter;
      for (auto idx : set)
      {
        if (idx >= 0 && idx < (int)viewpoint_sub_ids.size())
          sub_counter[viewpoint_sub_ids[idx]]++;
      }

      int best_sub_id = -1;
      int best_count = -1;
      for (const auto& pair : sub_counter)
      {
        if (pair.second > best_count)
        {
          best_count = pair.second;
          best_sub_id = pair.first;
        }
      }

      return best_sub_id;
    };

    vector<OrderedDecompSet> ordered_sets;
    ordered_sets.reserve(PR.decomp_vps_.size());
    for (int i=0; i<(int)PR.decomp_vps_.size(); ++i)
    {
      OrderedDecompSet info;
      info.set = PR.decomp_vps_[i];
      info.centroid = set_centroid(info.set, vps_pos);
      info.sub_id = set_major_sub_id(info.set);
      info.last_rank = std::numeric_limits<int>::max();
      info.last_dist = MAX;
      info.original_id = i;
      ordered_sets.push_back(info);
    }

    if (!ordered_sets.empty() && !PR.LAST_decomp_vps_.empty() && !PR.LAST_total_viewpoints_.empty())
    {
      vector<Eigen::Vector3d> last_vps_pos;
      last_vps_pos.reserve(PR.LAST_total_viewpoints_.size());
      for (const auto& vp : PR.LAST_total_viewpoints_)
        last_vps_pos.push_back(vp.head(3));

      vector<int> last_rank(PR.LAST_decomp_vps_.size(), std::numeric_limits<int>::max());
      for (int rank=0; rank<(int)PR.LAST_decomp_global_seq_.size(); ++rank)
      {
        int set_id = PR.LAST_decomp_global_seq_[rank];
        if (set_id >= 0 && set_id < (int)last_rank.size() && last_rank[set_id] == std::numeric_limits<int>::max())
          last_rank[set_id] = rank;
      }
      int tail_rank = (int)PR.LAST_decomp_global_seq_.size();
      for (int i=0; i<(int)last_rank.size(); ++i)
      {
        if (last_rank[i] == std::numeric_limits<int>::max())
          last_rank[i] = tail_rank++;
      }

      vector<LastDecompSet> last_sets;
      last_sets.reserve(PR.LAST_decomp_vps_.size());
      for (int i=0; i<(int)PR.LAST_decomp_vps_.size(); ++i)
      {
        if (PR.LAST_decomp_vps_[i].empty()) continue;
        LastDecompSet info;
        info.centroid = set_centroid(PR.LAST_decomp_vps_[i], last_vps_pos);
        info.rank = last_rank[i];
        last_sets.push_back(info);
      }

      struct MatchCandidate
      {
        int cur_id;
        int last_id;
        double dist;
      };

      vector<MatchCandidate> candidates;
      const double match_dist_limit = 2.5 * this->max_cvx_range_;
      for (int i=0; i<(int)ordered_sets.size(); ++i)
        for (int j=0; j<(int)last_sets.size(); ++j)
        {
          double dist = (ordered_sets[i].centroid - last_sets[j].centroid).norm();
          if (dist < match_dist_limit)
            candidates.push_back({i, j, dist});
        }

      stable_sort(candidates.begin(), candidates.end(), [](const MatchCandidate& a, const MatchCandidate& b)
      {
        if (a.dist != b.dist) return a.dist < b.dist;
        if (a.cur_id != b.cur_id) return a.cur_id < b.cur_id;
        return a.last_id < b.last_id;
      });

      vector<bool> cur_used(ordered_sets.size(), false);
      vector<bool> last_used(last_sets.size(), false);
      for (const auto& candidate : candidates)
      {
        if (cur_used[candidate.cur_id] || last_used[candidate.last_id]) continue;
        ordered_sets[candidate.cur_id].last_rank = last_sets[candidate.last_id].rank;
        ordered_sets[candidate.cur_id].last_dist = candidate.dist;
        cur_used[candidate.cur_id] = true;
        last_used[candidate.last_id] = true;
      }
    }

    stable_sort(ordered_sets.begin(), ordered_sets.end(), [](const OrderedDecompSet& a, const OrderedDecompSet& b)
    {
      if (a.last_rank != b.last_rank) return a.last_rank < b.last_rank;
      if (a.sub_id != b.sub_id) return a.sub_id < b.sub_id;
      if (a.centroid(0) != b.centroid(0)) return a.centroid(0) < b.centroid(0);
      if (a.centroid(1) != b.centroid(1)) return a.centroid(1) < b.centroid(1);
      if (a.centroid(2) != b.centroid(2)) return a.centroid(2) < b.centroid(2);
      return a.original_id < b.original_id;
    });

    PR.decomp_vps_.clear();
    for (const auto& info : ordered_sets)
      PR.decomp_vps_.push_back(info.set);

    ROS_INFO("\033[36m[ViewpointManager] skeleton-guided visibility decomp: bins = %d, sets = %d.\033[32m", guided_bin_num, (int)PR.decomp_vps_.size());
  }

  if (!PR.LAST_reused_viewpoints_.empty())
  {
    int cur_vp_num = PR.total_viewpoints_.size();
    for (int i=0; i<(int)PR.LAST_reused_viewpoints_.size(); ++i)
    {
      PR.total_viewpoints_.insert(PR.total_viewpoints_.end(), PR.LAST_reused_viewpoints_[i].begin(), PR.LAST_reused_viewpoints_[i].end());
      vector<int> reused_set;
      for (int j=0; j<(int)PR.LAST_reused_viewpoints_[i].size(); ++j)
        reused_set.push_back(cur_vp_num+j);
      PR.decomp_vps_.push_back(reused_set);
    
      cur_vp_num = PR.total_viewpoints_.size();
    }
  }

  int cur_effective_num = 0;
  for (auto x : PR.decomp_vps_)
    cur_effective_num += (int)x.size();

  ROS_INFO("\033[36m[ViewpointManager] cur effective vps = %d, total vps = %d.\033[32m", cur_effective_num, (int)PR.total_viewpoints_.size());

  this->viewpointNum = cur_effective_num;

  auto t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> ms = t2 - t1;
  double time = (double)ms.count();
  ROS_INFO("\033[33m[Planner] viewpoint visibility decomposition time = %lf ms.\033[32m", time);

  return;
}

void FCPlanner_PP::findGroundPrioritySet()
{
  auto t1 = chrono::high_resolution_clock::now();

  double lower = this->ground_flight_height_;
  double upper = lower + 2*model_ds_size;
  double lower_inf = lower + 0.5*this->drone_radius_;

  // * Preparation
  int ground_size = 0;
  map<Eigen::Vector3d, Eigen::Vector3d, Vector3dCompare0> potential_vp_rays;
  for (int i=0; i<(int)skeleton_operator->P.sub_space_ptnr.size(); ++i)
  {
    for (int j=0; j<(int)skeleton_operator->P.sub_space_ptnr[i]->points.size(); ++j)
    {
      pcl::PointNormal ptnr = skeleton_operator->P.sub_space_ptnr[i]->points[j];

      if (ptnr.z >= lower && ptnr.z <= upper)
      {
        ground_size++;
      
        Eigen::Vector3d pt_vect(ptnr.x, ptnr.y, ptnr.z);
        Eigen::Vector3d pt_ground(ptnr.x, ptnr.y, lower_inf);
        Eigen::Vector3d nr_vect(ptnr.normal_x, ptnr.normal_y, 0.0);
        nr_vect.normalize();
        Eigen::Vector3d nr_vect_rev = -nr_vect;

        Eigen::Vector3d candidate_a = pt_ground + this->vis_inflation_ * nr_vect;
        Eigen::Vector3d candidate_b = pt_ground + this->vis_inflation_ * nr_vect_rev;

        bool inside_a = visibility_st::inlierCheck(candidate_a, this->scene_);
        if (inside_a == false)
        {
          potential_vp_rays[pt_ground] = nr_vect;
          continue;
        }
      
        bool inside_b = visibility_st::inlierCheck(candidate_b, this->scene_);
        if (inside_b == false)
        {
          potential_vp_rays[pt_ground] = nr_vect_rev;
          continue;
        }
      }
    }
  }

  // * Inflate the ground points
  double inflation = this->drone_radius_ + HCMap->hcmp_->resolution_;
  double find_safe_threshold = inflation + 2*HCMap->hcmp_->resolution_;
  vector<Eigen::VectorXd> safe_ground_vps;
  PR.safe_g_vps_.reset(new pcl::PointCloud<pcl::PointXYZ>);

  for (const auto& pair : potential_vp_rays)
  {
    Eigen::Vector3d pt = pair.first;
    Eigen::Vector3d nr = pair.second;

    Eigen::Vector3d vp_pos = pt + this->g_vps_sample_dist_ * nr;
    Eigen::Vector3d vp_dir = -nr;
    Eigen::Vector2d py = path_tools::vecToPy(vp_dir);

    // * set pitch angle for ground coverage
    double cur_height = vp_pos(2) - this->GroundZPos;
    double vis_angle = atan(cur_height / this->g_vps_sample_dist_);
    double opt_pitch = v_fov_ - vis_angle;
    py(0) = opt_pitch;

    bool det = visibility_st::insideCheck(vp_pos, 0.0, 0.0, this->scene_, true);
    // bool det = visibility_st::inlierCheck(vp_pos, this->scene_);
    if (det)
    {
      Eigen::Vector3d p_start = pt + this->vis_inflation_ * nr;
      Eigen::Vector3d p_end = vp_pos;
      Eigen::Vector3d new_vp_pos;
      bool find_inter = visibility_st::getFirstIntersection(p_start, p_end, this->scene_, new_vp_pos);
      if (find_inter)
      {
        vp_pos = new_vp_pos + inflation * vp_dir;
        bool det_updated = visibility_st::insideCheck(vp_pos, 0.0, 0.0, scene_, true);
        // bool det_updated = visibility_st::inlierCheck(vp_pos, scene_);
        if (det_updated) continue;
      }
    }

    bool hc_safe = HCMap->safety_check(vp_pos);
    if (!hc_safe) continue;

    Eigen::VectorXd vp(5);
    vp << vp_pos, py;
    safe_ground_vps.push_back(vp);
    pcl::PointXYZ p;
    p.x = vp_pos(0); p.y = vp_pos(1); p.z = vp_pos(2);
    PR.safe_g_vps_->points.push_back(p);
  }

  // * Downsample the ground points
  struct ImprovedVoxelGrid : public pcl::VoxelGrid<pcl::PointXYZ> 
  {
    using pcl::VoxelGrid<pcl::PointXYZ>::VoxelGrid;

    std::vector<int> applyFilterIndices() 
    {
      std::unordered_map<Eigen::Vector3i, std::vector<int>, Vector3iHash> voxel_map;
      std::unordered_map<Eigen::Vector3i, Eigen::Vector4f, Vector3iHash> voxel_centroids;
      std::vector<int> result_indices;

      for (size_t i = 0; i < input_->points.size(); ++i) {
          const Eigen::Vector4f& point = input_->points[i].getVector4fMap();
          Eigen::Vector3i voxel_index = getVoxelIndex(point);

          voxel_map[voxel_index].push_back(static_cast<int>(i));

          if (voxel_centroids.find(voxel_index) == voxel_centroids.end()) {
              voxel_centroids[voxel_index] = point;
          } else {
              voxel_centroids[voxel_index] += point;
          }
      }

      for (const auto& pair : voxel_map) {
          Eigen::Vector3i voxel_index = pair.first;
          const std::vector<int>& indices = pair.second;

          Eigen::Vector4f centroid = voxel_centroids[voxel_index] / static_cast<float>(indices.size());

          float min_dist = std::numeric_limits<float>::max();
          int closest_index = -1;
          for (int idx : indices) {
              const Eigen::Vector4f& point = input_->points[idx].getVector4fMap();
              float dist = (point - centroid).squaredNorm();
              if (dist < min_dist) {
                  min_dist = dist;
                  closest_index = idx;
              }
          }

          if (closest_index != -1) {
              result_indices.push_back(closest_index);
          }
      }

      return result_indices;
    }

    Eigen::Vector3i getVoxelIndex(const Eigen::Vector4f& point) {
        return Eigen::Vector3i(
            static_cast<int>(std::floor((point[0] - min_b_[0]) * inverse_leaf_size_[0])),
            static_cast<int>(std::floor((point[1] - min_b_[1]) * inverse_leaf_size_[1])),
            static_cast<int>(std::floor((point[2] - min_b_[2]) * inverse_leaf_size_[2]))
        );
    }
  };

  ImprovedVoxelGrid custom_voxel_filter;
  custom_voxel_filter.setInputCloud(PR.safe_g_vps_);
  custom_voxel_filter.setLeafSize(this->ground_ds_size_, this->ground_ds_size_, this->ground_ds_size_);
  vector<int> downsampled_indices = custom_voxel_filter.applyFilterIndices();

  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_safe_ground(new pcl::PointCloud<pcl::PointXYZ>);
  map<pcl::PointXYZ, Eigen::VectorXd, PointXYZCompare> pt_vp_map;
  for (const auto& idx : downsampled_indices)
  {
    Eigen::VectorXd vp = safe_ground_vps[idx];
    Eigen::Vector3d vp_pos = vp.head(3);

    bool real_safe = true;
    if (this->HCMap->getInflateOccupancy(vp_pos) == 1) real_safe = false;
    bool find_new_safe = false;

    if (!real_safe) find_new_safe = this->findSafePose(vp, find_safe_threshold);
    else find_new_safe = true;

    Eigen::Vector3d new_vp_pos = vp.head(3);
    bool det_inside = visibility_st::insideCheck(new_vp_pos, 0.0, 0.0, this->scene_, true);
    // bool det_inside = visibility_st::inlierCheck(new_vp_pos, this->scene_);

    if (find_new_safe && !det_inside)
    {
      pcl::PointXYZ p;
      p.x = vp(0); p.y = vp(1); p.z = vp(2);
      downsampled_safe_ground->points.push_back(p);
      pt_vp_map[p] = vp;
    }
  }

  // * outliers removal
  pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
  ror.setInputCloud(downsampled_safe_ground);
  ror.setRadiusSearch(3*this->ground_ds_size_);
  ror.setMinNeighborsInRadius(1);
  ror.filter(*downsampled_safe_ground);

  PR.sampled_g_vps_set_.clear();
  PR.safe_g_vps_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  *PR.safe_g_vps_ = *downsampled_safe_ground; 
  for (int i=0; i<(int)PR.safe_g_vps_->points.size(); ++i)
  {
    pcl::PointXYZ p = PR.safe_g_vps_->points[i];
    if (pt_vp_map.find(p) != pt_vp_map.end())
    {
      Eigen::VectorXd vp = pt_vp_map[p];
      PR.sampled_g_vps_set_.push_back(vp);
    }
  }

  auto t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> ms = t2 - t1;
  double time = (double)ms.count();
  ROS_INFO("\033[33m[Planner] ground priority path time = %lf ms.\033[32m", time);

  return;
}

void FCPlanner_PP::updateGroundPrioritySet()
{
  auto t1 = chrono::high_resolution_clock::now();

  // * Update active ground viewpoints
  Eigen::VectorXd start = PR.prior_path.back();
  Eigen::Vector3d start_vel;
  if ((int)PR.prior_path.size() < 2) 
  {
    start_vel = this->start_vel_;
  } 
  else 
  {
    Eigen::Vector3d last_pos = PR.prior_path.back().head(3);
    Eigen::Vector3d second_last_pos = PR.prior_path[PR.prior_path.size() - 2].head(3);
    start_vel = this->vmax_ * (last_pos - second_last_pos).normalized();
  }

  vector<Eigen::VectorXd> updated_g_vps_set;
  double safe_dist = this->drone_radius_ + 2*HCMap->hcmp_->resolution_;

  vector<Eigen::VectorXd> reverse_exec_traj = {start};
  reverse_exec_traj.insert(reverse_exec_traj.end(), this->cur_exec_traj_.rbegin(), this->cur_exec_traj_.rend());

  bool need_check = false;
  double dist2start = 0.0;

  for (int i=0; i<(int)PR.sampled_g_vps_set_.size(); ++i)
  {
    need_check = false;
    for (auto p : reverse_exec_traj)
    {
      dist2start = (PR.sampled_g_vps_set_[i].head(3) - p.head(3)).norm();
      if (dist2start < this->local_range_)
      {
        need_check = true;
        break;
      }
    }

    if (need_check)
    {
      bool update_suc = this->updateRealPose(PR.sampled_g_vps_set_[i], safe_dist);
      if (!update_suc) continue;
    }

    Eigen::Vector3d vp_pos = PR.sampled_g_vps_set_[i].head(3);
    double pitch = PR.sampled_g_vps_set_[i](3), yaw = PR.sampled_g_vps_set_[i](4);
    this->percep_utils_->setPose_PY(vp_pos, pitch, yaw);

    bool near = false;
    for (auto x : this->cur_exec_traj_)
    {
      Eigen::Vector3d traj_pos = x.head(3);
      double dist = (traj_pos - vp_pos).norm();
      if (dist < ground_ds_size_)
      {
        near = true;
        break;
      }
    }

    if (near == true) continue;

    int see_num = 0;
    for (int j=0; j<(int)skeleton_operator->P.sub_space_ptnr.size(); ++j)
    {
      for (int k=0; k<(int)skeleton_operator->P.sub_space_ptnr[j]->points.size(); ++k)
      {
        if (PR.uncovered_surface_[j][k] == false) continue;

        pcl::PointNormal pt = skeleton_operator->P.sub_space_ptnr[j]->points[k];
        Eigen::Vector3d pt_pos(pt.x, pt.y, pt.z);
        if (this->percep_utils_->insideGSFOV(pt_pos) == false) continue;

        bool visible = visibility_st::visibilityCheck(vp_pos, pt_pos, this->vis_inflation_, this->scene_);
        if (visible == true) see_num++;
      }
    }

    if (see_num > this->save_visible_lower_ && dist2start > 1.0)
    {
      updated_g_vps_set.push_back(PR.sampled_g_vps_set_[i]);
    }
  }

  PR.sampled_g_vps_set_.clear();
  PR.sampled_g_vps_set_ = updated_g_vps_set;

  if (PR.sampled_g_vps_set_.empty())
  {
    this->ground_priority_ = false;
    return;
  }

  // * Find current shortest path
  int next_idx = -1;
  double min_cost = 1e6;
  for (int i=0; i<(int)PR.sampled_g_vps_set_.size(); ++i)
  {
    Eigen::VectorXd vp = PR.sampled_g_vps_set_[i];
    double dynamic_cost = path_tools::estPathTime(start_vel, start.head(3), vp.head(3), this->vmax_, this->amean_);
    if (dynamic_cost < min_cost)
    {
      min_cost = dynamic_cost;
      next_idx = i;
    }
  }

  vector<Eigen::VectorXd> pose_set;
  if (next_idx != -1)
  {
    Eigen::VectorXd next_vp = PR.sampled_g_vps_set_[next_idx];
    pose_set = {next_vp};
    for (int i=0; i<(int)PR.sampled_g_vps_set_.size(); ++i)
    {
      if (i == next_idx) continue;
      pose_set.push_back(PR.sampled_g_vps_set_[i]);
    }
  }
  else
  {
    this->ground_priority_ = false;
    return;
  }

  vector<Eigen::VectorXd> solved_path;
  vector<bool> solved_indicators;
  this->solver_->groundFindPath(start_vel, pose_set, solved_path, solved_indicators);

  PR.ground_priority_path_ = vector<Eigen::VectorXd>(solved_path.begin(), solved_path.end());
  PR.ground_priority_indicators_ = vector<bool>(solved_indicators.begin(), solved_indicators.end());

  // * Update prior path & uncovered state
  for (int j=0; j<(int)skeleton_operator->P.sub_space_ptnr.size(); ++j)
    for (int k=0; k<(int)skeleton_operator->P.sub_space_ptnr[j]->points.size(); ++k)
    {
      if (PR.uncovered_surface_[j][k] == true)
      {
        pcl::PointNormal pt = skeleton_operator->P.sub_space_ptnr[j]->points[k];
        Eigen::Vector3d pt_pos(pt.x, pt.y, pt.z);

        for (int i=0; i<(int)PR.ground_priority_path_.size(); ++i)
        {
          Eigen::Vector3d vp_pos = PR.ground_priority_path_[i].head(3);
          double pitch = PR.ground_priority_path_[i](3), yaw = PR.ground_priority_path_[i](4);
          this->percep_utils_->setPose_PY(vp_pos, pitch, yaw);

          if (this->percep_utils_->insideShrinkFOV(pt_pos) == true)
          {
            bool visible = visibility_st::visibilityCheck(vp_pos, pt_pos, this->vis_inflation_, this->scene_);
            if (visible == true)
            {
              PR.uncovered_surface_[j][k] = false;
              break;
            }
          }
        }
      }
    }

  PR.prior_path.insert(PR.prior_path.end(), PR.ground_priority_path_.begin(), PR.ground_priority_path_.end());
  PR.prior_wp_indicators_.insert(PR.prior_wp_indicators_.end(), PR.ground_priority_indicators_.begin(), PR.ground_priority_indicators_.end());

  auto t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> ms = t2 - t1;
  double time = (double)ms.count();
  ROS_INFO("\033[33m[Planner] update ground priority path time = %lf ms.\033[32m", time);

  return;
}

bool FCPlanner_PP::findSafePose(Eigen::VectorXd& pose, const double safe_dist)
{
  bool suc = false;

  double range = this->find_range_, step = this->HCMap->hcmp_->resolution_;

  int num_points_per_axis = static_cast<int>((2 * range) / step) + 1;
  int total_points = num_points_per_axis * num_points_per_axis * num_points_per_axis;

  Eigen::VectorXd offsets = Eigen::VectorXd::LinSpaced(num_points_per_axis, -range, range);
  Eigen::Matrix3Xd grid(3, total_points);
  int index = 0;
  for (int i = 0; i < num_points_per_axis; ++i)
    for (int j = 0; j < num_points_per_axis; ++j)
        for (int k = 0; k < num_points_per_axis; ++k) 
        {
            grid.col(index) << offsets(i), offsets(j), offsets(k);
            index++;
        }

  Eigen::Matrix3Xd sampled_points = grid.colwise() + pose.head(3);
  Eigen::VectorXd distances = (grid.colwise().squaredNorm()).array().sqrt();
  Eigen::VectorXi sorted_indices = Eigen::VectorXi::LinSpaced(total_points, 0, total_points - 1);
  std::sort(sorted_indices.data(), sorted_indices.data() + sorted_indices.size(),
              [&distances](int i, int j) {
                  return distances(i) < distances(j);
              });

  double z_min = this->GroundZPos + 2*this->drone_radius_;

  for (int i = 0; i < total_points; ++i)
  {
    Eigen::Vector3d sample_new_end = sampled_points.col(sorted_indices(i));
    bool safe = true;

    double z_sample = sample_new_end(2);

    if (!this->HCMap->isInMap(sample_new_end) || z_sample < z_min) safe = false;

    if (this->HCMap->getInflateOccupancy(sample_new_end) == 1) safe = false;

    if (safe == true)
    {
      pose.head(3) = sample_new_end;
      suc = true;
      break;
    }
  }

  return suc;
}

bool FCPlanner_PP::updateRealPose(Eigen::VectorXd& pose, const double safe_dist)
{
  Eigen::Vector3d position = pose.head(3);
  Eigen::Vector2d py(pose(3), pose(4));
  Eigen::Vector3d ray_dir = path_tools::pyToVec(py);

  if (this->HCMap->getInflateOccupancy(position) == 1 || this->HCMap->getOccupancy(position) == SDFMap::OCCUPANCY::UNKNOWN)
  {
    bool suc = false;

    double range = this->find_range_, step = this->HCMap->hcmp_->resolution_;
    Eigen::Matrix3Xd sampled_points;
    Eigen::VectorXi sorted_indices;

    bool is_unknown = this->HCMap->getOccupancy(position) == SDFMap::OCCUPANCY::UNKNOWN;

    // * If in unknown area, ray search
    if (is_unknown)
    {
      int sample_num = ceil(2*range/step);
      Eigen::Matrix3Xd ray_pts(3, 2*sample_num);
      for (int i=1; i<=sample_num; ++i)
      {
        Eigen::Vector3d cur_p = position + i*step*ray_dir;
        ray_pts.col(i-1) = cur_p;
      }
      for (int i=1; i<=sample_num; ++i)
      {
        Eigen::Vector3d cur_p = position - i*step*ray_dir;
        ray_pts.col(sample_num+i-1) = cur_p;
      }

      sampled_points = ray_pts;
    }

    // * Else, cube search
    else
    {
      int num_points_per_axis = static_cast<int>((2 * range) / step) + 1;
      int total_points = num_points_per_axis * num_points_per_axis * num_points_per_axis;

      Eigen::VectorXd offsets = Eigen::VectorXd::LinSpaced(num_points_per_axis, -range, range);
      Eigen::Matrix3Xd grid(3, total_points);
      int index = 0;
      for (int i = 0; i < num_points_per_axis; ++i)
        for (int j = 0; j < num_points_per_axis; ++j)
            for (int k = 0; k < num_points_per_axis; ++k) 
            {
                grid.col(index) << offsets(i), offsets(j), offsets(k);
                index++;
            }
    
      sampled_points = grid.colwise() + position;
      Eigen::VectorXd distances = (grid.colwise().squaredNorm()).array().sqrt();
      sorted_indices = Eigen::VectorXi::LinSpaced(total_points, 0, total_points - 1);
      std::sort(sorted_indices.data(), sorted_indices.data() + sorted_indices.size(),
                  [&distances](int i, int j) {
                      return distances(i) < distances(j);
                  });
    }

    double z_min = this->GroundZPos + 2*this->drone_radius_;

    for (int i = 0; i < (int)sampled_points.cols(); ++i)
    {
      Eigen::Vector3d sample_new_end;
      if (is_unknown) sample_new_end = sampled_points.col(i);
      else sample_new_end = sampled_points.col(sorted_indices(i));

      bool safe = true;

      double z_sample = sample_new_end(2);

      if (!this->HCMap->isInMap(sample_new_end) || z_sample < z_min) safe = false;

      if (!this->HCMap->getSafety(sample_new_end, safe_dist, safe_dist, 0.5*safe_dist) || this->HCMap->getInflateOccupancy(sample_new_end) == 1 || this->HCMap->getOccupancy(sample_new_end) == SDFMap::OCCUPANCY::UNKNOWN) safe = false;
    
      if (safe == true)
      {
        Eigen::Vector3d n2o = position - sample_new_end;
        Eigen::Vector3d ray = this->dist_vp * ray_dir;
        Eigen::Vector2d new_py;

        Eigen::Vector3d cross_product = n2o.cross(ray);
      
        if (cross_product.isZero(1e-3))
        {
          double dot_product = n2o.dot(ray);
          if (dot_product < 0)
          {
            new_py = py;
          }
          else
          {
            double original_height = this->dist_vp * tan(this->v_fov_);
            double cur_dist = this->dist_vp - n2o.norm();
            double rot_angle = atan(original_height / cur_dist) - this->v_fov_;
            new_py << py(0)-rot_angle, py(1);
          }
        }
        else
        {
          Eigen::Vector3d new_ray = (n2o + ray).normalized();
          new_py = path_tools::vecToPy(new_ray);
        }

        pose.head(3) = sample_new_end;
        pose(3) = new_py(0);
        pose(4) = new_py(1);
        suc = true;
        break;
      }
    }

    return suc;
  }
  else
  {
    return true;
  }
}

// ! /* --------------------- Eval --------------------- */

void FCPlanner_PP::PinHoleCamera(Eigen::VectorXd& vp, vector<int>& covered_id, pcl::PointCloud<pcl::PointXYZ>::Ptr& checkCloud)
{
  int xPixel = 2 * (int)cx, yPixel = 2 * (int)cy;
  double xRange = fx * tan(0.5 * fov_w * M_PI / 180.0) / 1000.0;
  double yRange = fy * tan(0.5 * fov_h * M_PI / 180.0) / 1000.0;
  Eigen::Vector2d ldc;
  ldc << -xRange, -yRange;
  double xRes = 2 * xRange / xPixel, yRes = 2 * yRange / yPixel;

  Eigen::Vector3d pt_vec;
  Eigen::Vector3i idx;
  Eigen::Vector2i camCod;
  double camDist;
  int xID, yID;

  vector<int> visible_id_vp;

  Eigen::MatrixXd ImageStateTable(xPixel, yPixel);
  ImageStateTable.setZero();
  Eigen::MatrixXi ImagePointTable(xPixel, yPixel);
  Eigen::MatrixXd ImageDistTable(xPixel, yPixel);
  ImageDistTable.setOnes();
  ImageDistTable = 1000.0 * ImageDistTable;

  Eigen::Vector3d position = vp.head(3);
  percep_utils_->setPose_PY(position, vp(3), vp(4));
  for (int i = 0; i < (int)checkCloud->points.size(); ++i)
  {
    pt_vec(0) = checkCloud->points[i].x;
    pt_vec(1) = checkCloud->points[i].y;
    pt_vec(2) = checkCloud->points[i].z;
    if (percep_utils_->insideFOV(pt_vec) == true)
    {
      cameraModelProjection(vp, pt_vec, xRes, yRes, ldc, camDist, camCod);
      xID = camCod(0);
      yID = camCod(1);
      if (xID < 0 || yID < 0)
        continue;
      if (ImageStateTable(xID, yID) == 0.0)
      {
        ImageStateTable(xID, yID) = 1.0;
        ImagePointTable(xID, yID) = i;
        ImageDistTable(xID, yID) = camDist;
      }
      else
      {
        if (camDist < ImageDistTable(xID, yID))
        {
          ImagePointTable(xID, yID) = i;
          ImageDistTable(xID, yID) = camDist;
        }
      }
    }
  }

  for (int r = 0; r < xPixel; ++r)
    for (int c = 0; c < yPixel; ++c)
    {
      if (ImageStateTable(r, c) != 0.0)
        visible_id_vp.push_back(ImagePointTable(r, c));
    }

  set<int> visSet(visible_id_vp.begin(), visible_id_vp.end());
  for (const auto &visid : visSet)
    covered_id.push_back(visid);
}

double FCPlanner_PP::trajCoverageEval(vector<Eigen::VectorXd>& poseSet)
{
  auto ce_t1 = chrono::high_resolution_clock::now();

  double coverageRate = 0.0;
  vector<bool> cover_state;
  cover_state.resize(Fullmodel->points.size(), false);
  int xPixel = 2 * (int)cx, yPixel = 2 * (int)cy;
  double xRange = fx * tan(0.5 * fov_w * M_PI / 180.0) / 1000.0;
  double yRange = fy * tan(0.5 * fov_h * M_PI / 180.0) / 1000.0;
  Eigen::Vector2d ldc;
  ldc << -xRange, -yRange;
  double xRes = 2 * xRange / xPixel, yRes = 2 * yRange / yPixel;

  Eigen::Vector3d pt_vec;
  Eigen::Vector3i idx;
  Eigen::Vector2i camCod;
  double camDist;
  int xID, yID;
  // bool visible_flag = false;
  vector<int> allVisible;
  for (auto vp : poseSet)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr visibleCloud(new pcl::PointCloud<pcl::PointXYZ>);
    vector<int> visible_id_vp;
    Eigen::MatrixXd ImageStateTable(xPixel, yPixel);
    ImageStateTable.setZero();
    Eigen::MatrixXi ImagePointTable(xPixel, yPixel);
    Eigen::MatrixXd ImageDistTable(xPixel, yPixel);
    ImageDistTable.setOnes();
    ImageDistTable = 1000.0 * ImageDistTable;
    Eigen::Vector3d position = vp.head(3);
    percep_utils_->setPose_PY(position, vp(3), vp(4));
    for (int i = 0; i < (int)Fullmodel->points.size(); ++i)
    {
      pt_vec(0) = Fullmodel->points[i].x;
      pt_vec(1) = Fullmodel->points[i].y;
      pt_vec(2) = Fullmodel->points[i].z;
      if (percep_utils_->insideFOV(pt_vec) == true)
      {
        // ! /* Pinhole Camera Model Evaluation */
        cameraModelProjection(vp, pt_vec, xRes, yRes, ldc, camDist, camCod);
        xID = camCod(0);
        yID = camCod(1);
        if (xID < 0 || yID < 0)
          continue;
        if (ImageStateTable(xID, yID) == 0.0)
        {
          ImageStateTable(xID, yID) = 1.0;
          ImagePointTable(xID, yID) = i;
          ImageDistTable(xID, yID) = camDist;
        }
        else
        {
          if (camDist < ImageDistTable(xID, yID))
          {
            cover_state[ImagePointTable(xID, yID)] = false;
            ImagePointTable(xID, yID) = i;
            ImageDistTable(xID, yID) = camDist;
          }
        }
      }
    }
    for (int r = 0; r < xPixel; ++r)
      for (int c = 0; c < yPixel; ++c)
      {
        if (ImageStateTable(r, c) != 0.0)
          visible_id_vp.push_back(ImagePointTable(r, c));
      }

    set<int> visSet(visible_id_vp.begin(), visible_id_vp.end());
    for (const auto &visid : visSet)
    {
      if (cover_state[visid] == false)
        visibleCloud->points.push_back(Fullmodel->points[visid]);
    }
    visible_group.push_back(visibleCloud);
    for (const auto &upid : visSet)
      cover_state[upid] = true;

    allVisible.insert(allVisible.end(), visible_id_vp.begin(), visible_id_vp.end());
  }
  set<int> uniqueSet(allVisible.begin(), allVisible.end());

  int truenum = 0;
  truenum = (int)uniqueSet.size();
  coverageRate = (double)truenum / (double)cover_state.size();

  auto ce_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> ce_ms = ce_t2 - ce_t1;
  double ce_time = (double)ce_ms.count();
  ROS_INFO("\033[36m[CoverageAnalyzer] FC-Planner-v2 trajectory coverage evaluation time = %lf s.\033[32m", ce_time / 1000.0);

  return coverageRate;
}

} // namespace flyco
