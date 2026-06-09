/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Jul. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the header file of Local_Replan class, which implements the
 *                   Local Motion Replanning in FlyCo.
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

#ifndef _LOCAL_REPLAN_H_
#define _LOCAL_REPLAN_H_

#include "plan_env/sdf_map.h"
#include "plan_env/raycast.h"
#include "path_searching/astar2.h"
#include "active_perception/perception_utils.h"
#include "plan_utils/visibility_st.hpp"
#include "plan_utils/path_tools.hpp"
#include "gcopter/voxel_map.hpp"
#include "gcopter/geo_utils.hpp"
#include "gcopter/trajectory.hpp"
#include "hierarchical_coverage_planner/hctraj.h"
#include "quadrotor_msgs/EigenVectorArray.h"
#include <ros/ros.h>
#include <ros/package.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <chrono>

using namespace std;
using std::unique_ptr;
using std::shared_ptr;

class RayCaster;

namespace flyco
{

class SDFMap;
class PerceptionUtils;
class TrajGenerator;
class Astar;
class Local_Replan
{

struct Vector5dCompare
{
  bool operator()(const Eigen::VectorXd &v1, const Eigen::VectorXd &v2) const
  {
    if (v1(0) != v2(0))
      return v1(0) < v2(0);
    if (v1(1) != v2(1))
      return v1(1) < v2(1);
    if (v1(2) != v2(2))
      return v1(2) < v2(2);
    if (v1(3) != v2(3))
      return v1(3) < v2(3);
    return v1(4) < v2(4);
  }
};

public:
  Local_Replan(){
  }
  ~Local_Replan(){
  }
  /* Func */
  void init(ros::NodeHandle& nh);
  void reset();
  bool replan(bool& feasible, const Eigen::VectorXd& local_start, const Eigen::Vector3d vel, const Eigen::Vector3d acc, const Eigen::Vector3d pyd, const Eigen::Vector3d pyd_dot);
  /* Interface */
  void setMap(shared_ptr<SDFMap>& map);
  void setMesh(Eigen::MatrixXd& mesh_V, Eigen::MatrixXi& mesh_F);
  void setGlobalPath(vector<Eigen::VectorXd>& path, vector<bool>& way_indi);
  void setGlobalPriorPath(vector<Eigen::VectorXd>& path);
  void setGlobalHCMap(pcl::PointCloud<pcl::PointXYZ>::Ptr& hc);
  void setGlobalModel(pcl::PointCloud<pcl::PointXYZ>::Ptr& model);
  void setGlobalNextSub(vector<Eigen::Vector3d>& inliers, vector<Eigen::VectorXd>& vps);
  void getUsedGlobalRoute(vector<Eigen::VectorXd>& path, vector<Eigen::VectorXd>& prior);
  void getUsedGlobalNextSub(vector<Eigen::Vector3d>& inliers, vector<Eigen::VectorXd>& vps);
  void getLocalPath(vector<Eigen::VectorXd>& path);
  void getLocalUpdatedPath(vector<Eigen::VectorXd>& path);
  void getLocalRegion(pcl::PointCloud<pcl::PointXYZ>::Ptr& hc, Eigen::Vector3d& min_bound, Eigen::Vector3d& max_bound);
  void getLocalTraj(Trajectory<7>& pos_traj, Trajectory<7>& orientation_traj);
  void getReplanPathWithIndi(vector<Eigen::VectorXd>& path, vector<bool>& indi);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  /* Operations */
  bool localRange(const Eigen::VectorXd& pose);
  void localPathVisOpt();
  bool localTrajPlan();
  bool localFeasibility();
  /* Tools */
  bool safetyCheck(const Eigen::Vector3d& pos, const double& safe_dist);
  bool realSafetyCheck(const Eigen::Vector3d& pos, const double& safe_dist);
  bool lineCheck(const Eigen::Vector3d& start, const Eigen::Vector3d& end, const double safe_dist);
  void pathToOrientedPath(const vector<Eigen::Vector3d>& path, vector<Eigen::VectorXd>& after_path, Eigen::VectorXd start, Eigen::VectorXd& end);
  bool setFixedView(const double& left_node, const double& right_node, const double& cur_node, const double& upper_bound);
  bool overGap(const double& left_node, const double& right_node, const double upper_bound, double& output_gap);
  double angleDiff(const double& angle1, const double& angle2);
  bool findSafePose(Eigen::VectorXd& pose, const double safe_dist, const double find_range);
  bool findSafeAnchorPoints(Eigen::VectorXd& pose, const double safe_dist, const double find_range);
  void execTrajCallback(const quadrotor_msgs::EigenVectorArrayConstPtr& msg);
  /* Utils */
  shared_ptr<SDFMap> mapping_utils_ = nullptr;
  unique_ptr<PerceptionUtils> percep_utils_ = nullptr;
  unique_ptr<TrajGenerator> motion_planner_= nullptr;
  unique_ptr<Astar> astar_= nullptr;
  unique_ptr<RayCaster> raycaster_ = nullptr;
  /* Param */
  bool zFlag = false;
  double ground_z_ = 0.0;
  double safe_height_ = 0.0;
  double fov_h_ = 0.0, fov_w_ = 0.0;
  double fov_h_half_ = 0.0, fov_w_half_ = 0.0;
  double local_region_ = 0.0;
  double visible_dist_ = 0.0;
  double dist_vp_ = 0.0;
  double resolution_ = 0.0;
  double drone_radius_ = 0.0;
  double sfc_prog_ = 0.0;
  double dist_thresh_ = 0.0;
  double vp_inflate_dist_ = 0.0;
  double safe_band_ = 0.0;
  double vmax_ = 0.0, wmax_ = 0.0, amean_ = 0.0;
  double min_dist_ = 1.0;
  double find_range_ = 5.0;
  double safe_inflation_ = 0.0;
  double vis_inflation_ = 0.0;
  int save_visible_lower_ = 0;
  Eigen::Vector3d vel_, acc_, pyd_, pyd_dot_;
  /* Data */
  pcl::PointCloud<pcl::PointXYZ>::Ptr global_hc_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr global_model_;
  voxel_map::VoxelMap voxelMap;
  RTCScene scene_ = nullptr;
  vector<Eigen::VectorXd> global_path_;
  vector<bool> global_waypt_indicators_;
  vector<Eigen::VectorXd> global_prior_path_;
  vector<Eigen::Vector3d> global_next_sub_inliers_;
  vector<Eigen::VectorXd> global_next_sub_vps_;
  pcl::KdTreeFLANN<pcl::PointXYZ> global_tree_;
  int start_local_id_;
  Eigen::Vector3d min_bound_ = Eigen::Vector3d::Zero(), max_bound_ = Eigen::Vector3d::Zero();
  vector<Eigen::VectorXd> local_path_;
  vector<bool> local_waypt_indicators_;
  map<Eigen::VectorXd, bool, Vector5dCompare> waypts_finder_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_hc_, local_traj_occ_;
  vector<Eigen::VectorXd> local_replan_path_;
  vector<bool> local_replan_indi_;
  vector<Eigen::VectorXd> local_prior_order_; // sequentially store those viewpoints that don't need to be optimized
  vector<Eigen::VectorXd> local_updated_path_;
  // pos_traj_: [x, y, z] (world frame -> NWU)
  // orientation_traj_: [yaw (world frame -> NWU), pitch (gimbal frame -> NED), 0]
  Trajectory<7> pos_traj_, orientation_traj_;
  vector<Eigen::VectorXd> cur_exec_traj_;
  /* ROS Service */
  ros::Subscriber exec_traj_sub_;
};

inline bool Local_Replan::localRange(const Eigen::VectorXd& pose)
{ 
  if ((int)global_path_.size() < 2)
  {
    ROS_WARN("No waypoints in global path!");
    return false;
  }

  // * Find Next Waypoint in Global Path
  int nearest_id = -1;

  ROS_ERROR("[Local] global model size -> %d", (int)this->global_model_->points.size());
  vector<bool> cur_inspect_states(this->global_model_->points.size(), false);

  // * Exec Traj Check
  vector<Eigen::VectorXd> recent_exec_traj = {pose};
  double exec_length = 0.0;

  if ((int)this->cur_exec_traj_.size() > 1)
  {
    recent_exec_traj.push_back(this->cur_exec_traj_.back());
    for (int i=(int)this->cur_exec_traj_.size()-2; i>=0; --i)
    {
      exec_length += (recent_exec_traj.back().head(3) - this->cur_exec_traj_[i].head(3)).norm();
      if (exec_length > this->local_region_) break;
      recent_exec_traj.push_back(this->cur_exec_traj_[i]);
    }
  }
  else
  {
    recent_exec_traj = this->cur_exec_traj_;
  }

  for (int j=0; j<(int)this->global_model_->points.size(); ++j)
  {
    if (cur_inspect_states[j] == false)
    {
      pcl::PointXYZ pt = this->global_model_->points[j];
      for (int k=0; k<(int)recent_exec_traj.size(); ++k)
      {
        Eigen::Vector3d vp_pos = recent_exec_traj[k].head(3);
        double vp_pitch = recent_exec_traj[k](3), vp_yaw = recent_exec_traj[k](4);
        Eigen::Vector3d pt_pos(pt.x, pt.y, pt.z);

        this->percep_utils_->setPose_PY(vp_pos, vp_pitch, vp_yaw);
        if (this->percep_utils_->insideFOV(pt_pos) == false)
          continue;
        bool visible = visibility_st::visibilityCheck(vp_pos, pt_pos, this->vis_inflation_, this->scene_);
        if (visible == true)
        {
          cur_inspect_states[j] = true;
          break;
        }
      }
    }
  }

  // * Global Path Check
  int effective_id = -1;
  for (int i=1; i<(int)global_path_.size(); ++i)
  {
    Eigen::Vector3d pos = global_path_[i].head(3);
    double pitch = global_path_[i](3), yaw = global_path_[i](4);
    this->percep_utils_->setPose_PY(pos, pitch, yaw);

    int see_count = 0;
    for (int n=0; n<(int)this->global_model_->points.size(); ++n)
    {
      if (cur_inspect_states[n] == true) continue;
      
      Eigen::Vector3d pt_pos(this->global_model_->points[n].x, this->global_model_->points[n].y, this->global_model_->points[n].z);
      if (this->percep_utils_->insideShrinkFOV(pt_pos) == true) 
      {
        bool visible = visibility_st::visibilityCheck(pos, pt_pos, this->vis_inflation_, this->scene_);
        if (visible == true) see_count++;
      }
    }

    if (see_count > this->save_visible_lower_)
    {
      effective_id = i;
      break;
    }
  }

  double upper_interval = 0.4*this->local_region_;
  if (effective_id != -1)
  {
    double dist = (pose.head(3)-global_path_[effective_id].head(3)).norm();
    if (dist > upper_interval)
    {
      Eigen::Vector3d pos_i = pose.head(3);
      Eigen::Vector3d pos_j = global_path_[effective_id].head(3);

      bool safe = true;
      Eigen::Vector3i idx;
      this->raycaster_->input(pos_i, pos_j);
      while (this->raycaster_->nextId(idx)) 
      {
        if (this->mapping_utils_->getOccupancy(idx) == SDFMap::OCCUPANCY::OCCUPIED || this->mapping_utils_->getInflateOccupancy(idx) == 1) 
        {
          safe = false;
          break;
        }
      }

      if (!safe)
      {
        ROS_WARN("The next waypoint needs to be re-selected!");
        for (int i=effective_id-1; i>0; --i)
        {
          double dist = (pose.head(3)-global_path_[i].head(3)).norm();
          if (dist < upper_interval)
          {
            effective_id = i;
            break;
          }
        }
      }
    }
  }

  nearest_id = effective_id;

  if (nearest_id == -1)
  {
    if ((int)this->global_path_.size() >= 2)
    {
      pcl::PointXYZ searchPoint;
      searchPoint.x = pose(0);
      searchPoint.y = pose(1);
      searchPoint.z = pose(2);
      const size_t num_results = 2;
      vector<int> ret_indexes(num_results);
      vector<float> out_dists_sqr(num_results);

      vector<int> nearest;
      vector<float> k_sqr_distances;
      global_tree_.nearestKSearch(searchPoint, num_results, nearest, k_sqr_distances);
      
      nearest_id = nearest[1];
      if (nearest_id == 0) nearest_id = 1;
    }
    else
    {
      nearest_id = 0;
    }
  }

  start_local_id_ = nearest_id;

  // * Determine Local Path Maximal Size
  int local_size = 1+start_local_id_;
  double length = (global_path_[start_local_id_].head(3)-pose.head(3)).norm();
  min_bound_(0) = pose.head(3)(0) < global_path_[start_local_id_].head(3)(0) ? pose.head(3)(0) : global_path_[start_local_id_].head(3)(0);
  min_bound_(1) = pose.head(3)(1) < global_path_[start_local_id_].head(3)(1) ? pose.head(3)(1) : global_path_[start_local_id_].head(3)(1);
  min_bound_(2) = pose.head(3)(2) < global_path_[start_local_id_].head(3)(2) ? pose.head(3)(2) : global_path_[start_local_id_].head(3)(2);
  max_bound_(0) = pose.head(3)(0) > global_path_[start_local_id_].head(3)(0) ? pose.head(3)(0) : global_path_[start_local_id_].head(3)(0);
  max_bound_(1) = pose.head(3)(1) > global_path_[start_local_id_].head(3)(1) ? pose.head(3)(1) : global_path_[start_local_id_].head(3)(1);
  max_bound_(2) = pose.head(3)(2) > global_path_[start_local_id_].head(3)(2) ? pose.head(3)(2) : global_path_[start_local_id_].head(3)(2);
  
  if (local_size < (int)global_path_.size())
  {
    for (int i=start_local_id_; i<(int)global_path_.size()-1; ++i)
    {
      length += (global_path_[i+1].head(3)-global_path_[i].head(3)).norm();
      if (length < local_region_)
      {
        min_bound_(0) = min_bound_(0) < global_path_[i+1].head(3)(0) ? min_bound_(0) : global_path_[i+1].head(3)(0);
        min_bound_(1) = min_bound_(1) < global_path_[i+1].head(3)(1) ? min_bound_(1) : global_path_[i+1].head(3)(1);
        min_bound_(2) = min_bound_(2) < global_path_[i+1].head(3)(2) ? min_bound_(2) : global_path_[i+1].head(3)(2);
        max_bound_(0) = max_bound_(0) > global_path_[i+1].head(3)(0) ? max_bound_(0) : global_path_[i+1].head(3)(0);
        max_bound_(1) = max_bound_(1) > global_path_[i+1].head(3)(1) ? max_bound_(1) : global_path_[i+1].head(3)(1);
        max_bound_(2) = max_bound_(2) > global_path_[i+1].head(3)(2) ? max_bound_(2) : global_path_[i+1].head(3)(2);
        local_size++;
      }
      else
        break;
    }
  }

  double inflate = visible_dist_;
  min_bound_(0) -= inflate; min_bound_(1) -= inflate; min_bound_(2) -= inflate;
  max_bound_(0) += inflate; max_bound_(1) += inflate; max_bound_(2) += inflate;

  min_bound_(0) = this->mapping_utils_->mp_->map_min_boundary_(0) > min_bound_(0) ? this->mapping_utils_->mp_->map_min_boundary_(0) : min_bound_(0);
  min_bound_(1) = this->mapping_utils_->mp_->map_min_boundary_(1) > min_bound_(1) ? this->mapping_utils_->mp_->map_min_boundary_(1) : min_bound_(1);
  min_bound_(2) = this->mapping_utils_->mp_->map_min_boundary_(2)+1.0 > min_bound_(2) ? this->mapping_utils_->mp_->map_min_boundary_(2) : min_bound_(2);
  max_bound_(0) = this->mapping_utils_->mp_->map_max_boundary_(0) < max_bound_(0) ? this->mapping_utils_->mp_->map_max_boundary_(0) : max_bound_(0);
  max_bound_(1) = this->mapping_utils_->mp_->map_max_boundary_(1) < max_bound_(1) ? this->mapping_utils_->mp_->map_max_boundary_(1) : max_bound_(1);
  max_bound_(2) = this->mapping_utils_->mp_->map_max_boundary_(2) < max_bound_(2) ? this->mapping_utils_->mp_->map_max_boundary_(2) : max_bound_(2);

  // * Initial Local Path
  local_path_.clear();
  local_waypt_indicators_.clear();
  local_path_.push_back(pose);
  local_waypt_indicators_.push_back(false);
  for (int i=start_local_id_; i<local_size; ++i)
  {
    local_path_.push_back(global_path_[i]);
    if (global_waypt_indicators_[i] == true)
    {
      local_waypt_indicators_.push_back(true);
    }
    else
    {
      local_waypt_indicators_.push_back(false);
      waypts_finder_[global_path_[i]] = false;
    }
  }

  // * Find anchor point
  Eigen::VectorXd pt_start = local_path_[0], pt_next = local_path_[1];
  double safe_threshold = this->drone_radius_ + 2*this->resolution_;
  Eigen::Vector3d p1 = pt_start.head(3), p2 = pt_next.head(3);
  bool safe = true;
  Eigen::Vector3i idx;
  this->raycaster_->input(p1, p2);
  while (this->raycaster_->nextId(idx)) 
  {
    if (this->mapping_utils_->getOccupancy(idx) == SDFMap::OCCUPANCY::UNKNOWN || this->mapping_utils_->getInflateOccupancy(idx) == 1) 
    {
      safe = false;
      break;
    }
  }

  if (!safe)
  {
    ROS_WARN("Too Long & Insert Anchor Point!");
    Eigen::VectorXd anchor_pose(5);
    anchor_pose.head(3) = 0.5 * (p1 + p2);
    double pitch_interval = pt_next(3) - pt_start(3);
    double yaw_interval = pt_next(4) - pt_start(4);

    while (pitch_interval > M_PI)
      pitch_interval -= 2*M_PI;
    while (pitch_interval < -M_PI)
      pitch_interval += 2*M_PI;
    while (yaw_interval > M_PI)
      yaw_interval -= 2*M_PI;
    while (yaw_interval < -M_PI)
      yaw_interval += 2*M_PI;

    anchor_pose(3) = pt_start(3) + 0.5*pitch_interval;
    anchor_pose(4) = pt_start(4) + 0.5*yaw_interval;

    bool find_new = findSafeAnchorPoints(anchor_pose, safe_threshold, upper_interval);
    if (find_new)
    {
      local_path_.insert(local_path_.begin() + 1, anchor_pose);
      local_waypt_indicators_.insert(local_waypt_indicators_.begin() + 1, false);
      waypts_finder_[anchor_pose] = false;
    }
  }

  if ((int)local_path_.size() < 2)
  {
    ROS_WARN("Local path is too short!");
    return false;
  }

  return true;
}

inline void Local_Replan::localPathVisOpt()
{
  auto t1 = chrono::high_resolution_clock::now();

  visibility_st::setParams(percep_utils_->max_dist_, fov_w_half_, fov_h_half_, resolution_);                          

  // * Inflate FoV
  vp_inflate_dist_ = this->drone_radius_ + this->safe_inflation_;
  safe_band_ = 2*vp_inflate_dist_;

  ROS_ERROR("Local replanning start!");
  Eigen::VectorXd start = local_path_.front();
  Eigen::VectorXd end = local_path_.back();
  
  double find_safe_threshold = vp_inflate_dist_ + this->resolution_;

  local_replan_path_.clear();
  local_replan_indi_.clear();
  local_replan_path_.push_back(start);
  local_replan_indi_.push_back(false);

  for (int i=1; i<(int)local_path_.size(); ++i)
  {
    Eigen::VectorXd cur_point = local_path_[i];
    bool flag = realSafetyCheck(cur_point.head(3), vp_inflate_dist_);

    if (flag == false)
    {
      flag = this->findSafePose(cur_point, find_safe_threshold, this->find_range_);
      if (flag == true)
      {
        bool safe = true;
        Eigen::Vector3i idx;
        this->raycaster_->input(local_replan_path_.back().head(3), cur_point.head(3));
        while (this->raycaster_->nextId(idx)) 
        {
          if (this->mapping_utils_->getOccupancy(idx) == SDFMap::OCCUPANCY::OCCUPIED || this->mapping_utils_->getInflateOccupancy(idx) == 1) 
          {
            safe = false;
            break;
          }
          
          if (safe == false)
            break;
        }
        
        if (safe == false)
        {
          this->astar_->reset();
          this->astar_->setResolution(this->resolution_);
          int result = this->astar_->wholeSearch(local_replan_path_.back().head(3), cur_point.head(3));
          
          vector<Eigen::VectorXd> inter_vps;
          if (result == Astar::REACH_END)
          {
            vector<Eigen::Vector3d> path = this->astar_->getPath();
            vector<Eigen::VectorXd> vp_path;
            pathToOrientedPath(path, vp_path, start, end);
            if ((int)vp_path.size() > 2)
              copy(vp_path.begin() + 1, vp_path.end() - 1, back_inserter(inter_vps));
          }
          vector<bool> inter_waypts_indi(inter_vps.size(), true);
          local_replan_path_.insert(local_replan_path_.end(), inter_vps.begin(), inter_vps.end());
          local_replan_indi_.insert(local_replan_indi_.end(), inter_waypts_indi.begin(), inter_waypts_indi.end());
        }

        local_replan_path_.push_back(cur_point);
        local_replan_indi_.push_back(false);
      }
    }
    else
    {
      local_replan_path_.push_back(cur_point);
      local_replan_indi_.push_back(local_waypt_indicators_[i]);
    }
  }

  local_updated_path_.clear();
  local_updated_path_ = local_replan_path_;
  

  // * Interpolate Waypoints
  vector<Eigen::VectorXd> intered_path;
  vector<bool> intered_indi;

  for (int i=0; i<(int)local_replan_path_.size()-1; ++i)
  {
    intered_path.push_back(local_replan_path_[i]);
    intered_indi.push_back(local_replan_indi_[i]);
    
    vector<Eigen::VectorXd> piece;
    path_tools::pieceInterpolate(local_replan_path_[i], local_replan_path_[i+1], dist_thresh_, piece);
    vector<bool> piece_indi(piece.size(), true);
    
    intered_path.insert(intered_path.end(), piece.begin(), piece.end());
    intered_indi.insert(intered_indi.end(), piece_indi.begin(), piece_indi.end());
  }
  intered_path.push_back(local_replan_path_.back());
  intered_indi.push_back(false);

  local_replan_path_.clear();
  local_replan_indi_.clear();
  local_replan_path_ = intered_path;
  local_replan_indi_ = intered_indi;

  // * Truncate Replanned Path
  vector<Eigen::VectorXd> trunc_path_a; trunc_path_a.push_back(local_replan_path_.front());
  vector<bool> trunc_indi_a; trunc_indi_a.push_back(local_replan_indi_.front());
  vector<Eigen::VectorXd> trunc_path_b; trunc_path_b.push_back(local_replan_path_.front());
  vector<bool> trunc_indi_b; trunc_indi_b.push_back(local_replan_indi_.front());

  double cur_l = 0.0;
  for (int i=1; i<(int)local_replan_path_.size(); ++i)
  {
    double l = (local_replan_path_[i].head(3) - trunc_path_b.back().head(3)).norm();
    cur_l += l;
    if (cur_l < local_region_)
    {
      trunc_path_a.push_back(local_replan_path_[i]);
      trunc_indi_a.push_back(local_replan_indi_[i]);
      trunc_path_b.push_back(local_replan_path_[i]);
      trunc_indi_b.push_back(local_replan_indi_[i]);
    }
    else
    {
      trunc_path_b.push_back(local_replan_path_[i]);
      trunc_indi_b.push_back(local_replan_indi_[i]);
      if (trunc_indi_b.back() == false) break;
    }
  }

  double l_b = 0.0;
  for (int i=1; i<(int)trunc_path_b.size(); ++i)
    l_b += (trunc_path_b[i].head(3) - trunc_path_b[i-1].head(3)).norm();
  
  local_replan_path_.clear();
  local_replan_indi_.clear();

  if (l_b < 1.5*local_region_)
  {
    local_replan_path_ = trunc_path_b;
    local_replan_indi_ = trunc_indi_b;
  }
  else
  {
    local_replan_path_ = trunc_path_a;
    local_replan_indi_ = trunc_indi_a;
  }

  // * Global Rectification
  vector<Eigen::VectorXd> temp_path = {local_replan_path_.front()};
  vector<bool> temp_indi = {local_replan_indi_.front()};
  for (int i=1; i<(int)local_replan_path_.size(); ++i)
  {
    if (local_replan_indi_[i] == false)
    {
      if (temp_indi.back() == true)
      {
        if ((local_replan_path_[i].head(3) - temp_path.back().head(3)).norm() > min_dist_)
        {
          temp_path.push_back(local_replan_path_[i]);
          temp_indi.push_back(local_replan_indi_[i]);
        }
        else
        {
          temp_path.back() = local_replan_path_[i];
          temp_indi.back() = local_replan_indi_[i];
        }
      }
      else
      {
        temp_path.push_back(local_replan_path_[i]);
        temp_indi.push_back(local_replan_indi_[i]);
      }
    }
    else
    {
      if ((local_replan_path_[i].head(3) - temp_path.back().head(3)).norm() > min_dist_)
      {
        temp_path.push_back(local_replan_path_[i]);
        temp_indi.push_back(local_replan_indi_[i]);
      }
    }
  }
  temp_indi.front() = false;
  temp_indi.back() = false;

  local_replan_path_ = temp_path;
  local_replan_indi_ = temp_indi;

  // * Waypoint Specific Setting
  double angle_diff = 10.0*M_PI/180.0;

  if ((int)local_replan_path_.size() > 2)
  {
    double dist = (local_replan_path_[1].head(3) - local_replan_path_[0].head(3)).norm();
    double angle = 0.0;

    Eigen::Vector3d path_dir = (local_replan_path_[1].head(3) - local_replan_path_[0].head(3)).normalized();
    Eigen::Vector3d vel_dir = this->vel_.normalized();
    if (this->vel_.norm() < 1e-3)
      angle = 90.0 * M_PI / 180.0;
    else
      angle = abs(acos(path_dir.dot(vel_dir)));
    if (dist < 0.5*dist_thresh_ && angle < angle_diff)
    {
      local_replan_indi_[1] = true;
    }
  }

  auto t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> l_ms = t2 - t1;
  double l_time = (double)l_ms.count();
  ROS_INFO("\033[33m[FlyCo] local path optimization : %lf ms. \033[32m", l_time);
}

inline bool Local_Replan::localTrajPlan()
{
  local_replan_indi_.front() = false;
  local_replan_indi_.back() = false;

  // make sure enough optimization dimensions
  if ((int)local_replan_path_.size() < 4)
  {
    vector<Eigen::VectorXd> inter_replan_path;
    vector<bool> inter_replan_indi;
    if ((int)local_replan_path_.size() == 1)
    {
      inter_replan_path = local_replan_path_;
      inter_replan_indi = local_replan_indi_;
    }
    if ((int)local_replan_path_.size() == 2)
    {
      Eigen::VectorXd start = local_replan_path_.front(), end = local_replan_path_.back();
      double dist = (end.head(3) - start.head(3)).norm();
      double angle_pitch = this->angleDiff(start(3), end(3));
      double angle_yaw = this->angleDiff(start(4), end(4));
      double interval = dist / 3.0, pitch_interval = angle_pitch / 3.0, yaw_interval = angle_yaw / 3.0;
      Eigen::Vector3d dir = (end.head(3) - start.head(3)).normalized();

      inter_replan_path.push_back(start);
      inter_replan_indi.push_back(local_replan_indi_.front());
      for (int i=1; i<3; ++i)
      {
        Eigen::VectorXd inter(5);
        inter.head(3) = start.head(3) + i*interval*dir;
        inter(3) = start(3) + i*pitch_interval;
        inter(4) = start(4) + i*yaw_interval;
        inter_replan_path.push_back(inter);
        inter_replan_indi.push_back(true);
      }
      inter_replan_path.push_back(end);
      inter_replan_indi.push_back(local_replan_indi_.back());
    }
    if ((int)local_replan_path_.size() == 3)
    {
      inter_replan_path.push_back(local_replan_path_.front());
      inter_replan_indi.push_back(local_replan_indi_.front());
      for (int i=0; i<2; ++i)
      {
        Eigen::VectorXd inter(5);
        inter.head(3) = 0.5*(local_replan_path_[i].head(3) + local_replan_path_[i+1].head(3));
        double pitch_gap = this->angleDiff(local_replan_path_[i](3), local_replan_path_[i+1](3));
        double yaw_gap = this->angleDiff(local_replan_path_[i](4), local_replan_path_[i+1](4));
        inter(3) = local_replan_path_[i](3) + 0.5*pitch_gap;
        inter(4) = local_replan_path_[i](4) + 0.5*yaw_gap;
        inter_replan_path.push_back(inter);
        inter_replan_indi.push_back(true);

        inter_replan_path.push_back(local_replan_path_[i+1]);
        inter_replan_indi.push_back(local_replan_indi_[i+1]);
      }
    }

    local_replan_path_.clear();
    local_replan_path_ = inter_replan_path;
    local_replan_indi_.clear();
    local_replan_indi_ = inter_replan_indi;
  }

  if ((int)local_replan_path_.size() < 2)
    return false;
  
  double angle_upper = M_PI/6.0;
  double factor = 1.5;
  
  // pre-process replan path
  vector<int> view_idx;
  for (int i=0; i<(int)local_replan_indi_.size(); ++i)
  {
    if (local_replan_indi_[i] == false)
      view_idx.push_back(i);
  }

  for (int i=0; i<(int)view_idx.size()-1; ++i)
  {
    int left_idx = view_idx[i], right_idx = view_idx[i+1];
    double left_pitch = local_replan_path_[left_idx](3), left_yaw = local_replan_path_[left_idx](4);
    double right_pitch = local_replan_path_[right_idx](3), right_yaw = local_replan_path_[right_idx](4);
    
    if (right_idx - left_idx == 1) continue;

    for (int j=left_idx+1; j<right_idx; ++j)
    {
      double cur_pitch = local_replan_path_[j](3), cur_yaw = local_replan_path_[j](4);
      bool pitch_flag = this->setFixedView(left_pitch, right_pitch, cur_pitch, angle_upper);
      bool yaw_flag = this->setFixedView(left_yaw, right_yaw, cur_yaw, angle_upper);

      if (pitch_flag == true || yaw_flag == true)
        local_replan_indi_[j] = false;
    }
  }

  vector<Eigen::VectorXd> updated_replan_path;
  vector<bool> updated_replan_indi;

  vector<int> updated_view_idx;
  for (int i=0; i<(int)local_replan_indi_.size(); ++i)
  {
    if (local_replan_indi_[i] == false)
      updated_view_idx.push_back(i);
  }

  for (int i=0; i<(int)updated_view_idx.size()-1; ++i)
  {
    int left_idx = updated_view_idx[i], right_idx = updated_view_idx[i+1];
    double left_pitch = local_replan_path_[left_idx](3), left_yaw = local_replan_path_[left_idx](4);
    double right_pitch = local_replan_path_[right_idx](3), right_yaw = local_replan_path_[right_idx](4);
    double pitch_gap = 0.0, yaw_gap = 0.0;
    bool over_pitch_flag = this->overGap(left_pitch, right_pitch, factor*angle_upper, pitch_gap);
    bool over_yaw_flag = this->overGap(left_yaw, right_yaw, factor*angle_upper, yaw_gap);

    if (over_pitch_flag == false && over_yaw_flag == false)
    {
      for (int j=left_idx; j<right_idx; ++j)
      {
        updated_replan_path.push_back(local_replan_path_[j]);
        updated_replan_indi.push_back(local_replan_indi_[j]);
      }
    }
    else
    {
      if (right_idx - left_idx == 1)
      {
        Eigen::VectorXd inter(5);
        inter.head(3) = 0.5*(local_replan_path_[left_idx].head(3) + local_replan_path_[right_idx].head(3));
        inter(3) = left_pitch + 0.5*pitch_gap;
        inter(4) = left_yaw + 0.5*yaw_gap;

        updated_replan_path.push_back(local_replan_path_[left_idx]);
        updated_replan_path.push_back(inter);
        updated_replan_indi.push_back(local_replan_indi_[left_idx]);
        updated_replan_indi.push_back(false);
      }
      else
      {
        for (int j=left_idx; j<right_idx; ++j)
        {
          updated_replan_path.push_back(local_replan_path_[j]);
          updated_replan_indi.push_back(false);
        }
      }
    }
  }
  updated_replan_path.push_back(local_replan_path_.back());
  updated_replan_indi.push_back(local_replan_indi_.back());
  local_replan_path_.clear();
  local_replan_path_ = updated_replan_path;
  local_replan_indi_.clear();
  local_replan_indi_ = updated_replan_indi;

  // pose
  Eigen::Vector3d now_posi = local_replan_path_.front().head(3);
  vector<Eigen::Vector3d> wps;
  vector<bool> wps_waypt_indicators;
  vector<double> pitchs, yaws;

  wps.resize((int)local_replan_path_.size()-1);
  wps_waypt_indicators.resize((int)local_replan_path_.size()-1);
  pitchs.resize((int)local_replan_path_.size());
  yaws.resize((int)local_replan_path_.size());

  for (int i=0; i<(int)local_replan_path_.size(); ++i)
  {
    if (i > 0)
    {
      wps[i-1] = local_replan_path_[i].head(3);
      wps_waypt_indicators[i-1] = local_replan_indi_[i];
    }  
    pitchs[i] = local_replan_path_[i](3);
    yaws[i] = local_replan_path_[i](4);
  }

  // ? Select better path
  vector<Eigen::Vector3d> updated_wps;
  vector<bool> updated_wps_waypt_indicators;
  vector<double> updated_pitchs = {pitchs.front()}, updated_yaws = {yaws.front()};

  bool next_waypt = true;
  for (int i=0; i<(int)wps.size(); ++i)
  {
    double dist = (wps[i] - now_posi).norm();
    if (dist > dist_thresh_)
    {
      updated_wps.push_back(wps[i]);
      // if (next_waypt)
      // {
      //   updated_wps_waypt_indicators.push_back(wps_waypt_indicators[i]);
      //   next_waypt = false;
      // }
      // else
      // {
      //   updated_wps_waypt_indicators.push_back(false);
      // }

      updated_wps_waypt_indicators.push_back(wps_waypt_indicators[i]);
      updated_pitchs.push_back(pitchs[i+1]);
      updated_yaws.push_back(yaws[i+1]);
    }
  }

  wps = updated_wps;
  wps_waypt_indicators = updated_wps_waypt_indicators;
  pitchs = updated_pitchs;
  yaws = updated_yaws;

  if ((int)wps.size() < 2)
  {
    double path_length = (now_posi - wps.front()).norm();
    if (path_length < 2*dist_thresh_)
    {
      ROS_ERROR("Local replan path is too short!");
      return false;
    }
    else
    {
      Eigen::Vector3d mid_pos = 0.5*(now_posi + wps.front());
      wps.insert(wps.begin(), mid_pos);
      wps_waypt_indicators.insert(wps_waypt_indicators.begin(), true);
      double mid_pitch = pitchs.front() + 0.5*this->angleDiff(pitchs.front(), pitchs.back());
      double mid_yaw = yaws.front() + 0.5*this->angleDiff(yaws.front(), yaws.back());
      pitchs.insert(pitchs.begin()+1, mid_pitch);
      yaws.insert(yaws.begin()+1, mid_yaw);
    }
  }

  // ? buffer zone for traj optimization
  double buffer_length = (now_posi-wps.front()).norm();

  if (buffer_length < 2*dist_thresh_ && wps_waypt_indicators.front() == false && (int)wps.size() > 1)
    wps_waypt_indicators.front() = true;
  
  wps_waypt_indicators.back() = false;

  double length = 0.0;
  for (int i=0; i<(int)local_replan_path_.size()-1; ++i)
    length += (local_replan_path_[i+1].head(3) - local_replan_path_[i].head(3)).norm();
  int waypts = count(wps_waypt_indicators.begin(), wps_waypt_indicators.end(), true);

  // statistic
  ROS_INFO("\033[31m[PathAnalyzer] full path size = %d. \033[32m", (int)wps.size()+1);
  ROS_INFO("\033[31m[PathAnalyzer] all waypoints quantity = %d. \033[32m", waypts);
  ROS_INFO("\033[31m[PathAnalyzer] path length = %lf m.\033[32m", length);

  // * Local Motion Planning 
  motion_planner_->HCTraj(now_posi, wps, wps_waypt_indicators, pitchs, yaws, voxelMap, sfc_prog_, 2.0, this->vel_, this->acc_, this->pyd_, this->pyd_dot_);
  if (!motion_planner_->traj_opt_suc_)
  {
    ROS_ERROR("Local trajectory optimization failed!");
    return false;
  }

  this->pos_traj_.clear();
  this->orientation_traj_.clear();
  this->pos_traj_ = this->motion_planner_->minco_traj;
  this->orientation_traj_ = this->motion_planner_->minco_orientation_traj; // p: [θ, ψ, 0]

  ROS_INFO("\033[35m[Traj] --- <Trajectory Generation finished> --- \033[35m");

  return true;
}

inline bool Local_Replan::localFeasibility()
{
  bool feasible = true;

  double max_vel = this->pos_traj_.getMaxVelRate();
  if (max_vel > 2.0*this->vmax_)
    feasible = false;

  double yd = this->orientation_traj_.getMaxYawd();
  if (yd > 2.5*this->wmax_)
    feasible = false;

  if (feasible == false)
  {
    ROS_ERROR("Local traj is not feasible!");
  }

  return feasible;
}

// ! --------------------------- Tools ---------------------------

inline bool Local_Replan::safetyCheck(const Eigen::Vector3d& pos, const double& safe_dist)
{
  if (zFlag == true)
  {
    if (pos(2) < ground_z_ + safe_height_)
    {
      return false;
    }
  }
  
  bool f1 = this->mapping_utils_->getSafety(pos, safe_dist, safe_dist, 0.5*safe_dist);
  bool f2 = this->mapping_utils_->getHCSafety(pos, safe_dist, safe_dist, 0.5*safe_dist);

  if (f1 == true && f2 == true)
    return true;
  else
    return false;
}

inline bool Local_Replan::realSafetyCheck(const Eigen::Vector3d& pos, const double& safe_dist)
{
  if (zFlag == true)
  {
    if (pos(2) < ground_z_ + safe_height_)
    {
      return false;
    }
  }

  if (!this->mapping_utils_->getSafety(pos, safe_dist, safe_dist, 0.5*safe_dist)|| !this->mapping_utils_->isInMap(pos) || this->mapping_utils_->getInflateOccupancy(pos) == 1 || this->mapping_utils_->getOccupancy(pos) == SDFMap::OCCUPANCY::UNKNOWN)
    return false;

  return true;
}

inline bool Local_Replan::lineCheck(const Eigen::Vector3d& start, const Eigen::Vector3d& end, const double safe_dist)
{
  double step = 0.5*this->resolution_;
  int check_num = ceil((end-start).norm()/step);
  step = (end-start).norm()/check_num;

  bool safe = true;
  for (int i=1; i<check_num-1; ++i)
  {
    Eigen::Vector3d check_pos = start + i*step*(end-start).normalized();
    safe = safetyCheck(check_pos, safe_dist);
    if (safe == false)
      break;
  }

  return safe;
}

inline void Local_Replan::pathToOrientedPath(const vector<Eigen::Vector3d>& path, vector<Eigen::VectorXd>& after_path, Eigen::VectorXd start, Eigen::VectorXd& end)
{
  after_path.clear();

  if ((int)path.size() <= 2)
  {
    after_path = { start, end };
    return;
  }

  vector<Eigen::Vector3d> inter(path.begin() + 1, path.end() - 1);
  vector<Eigen::VectorXd> ori_ip_path;
  path_tools::orienInterpolate(start, end, inter, ori_ip_path);

  after_path.push_back(start);
  after_path.insert(after_path.end(), ori_ip_path.begin(), ori_ip_path.end());
  after_path.push_back(end);

  return;
}

inline bool Local_Replan::setFixedView(const double& left_node, const double& right_node, const double& cur_node, const double& upper_bound)
{
  double left_gap = cur_node - left_node;
  while (left_gap > M_PI)
    left_gap -= 2*M_PI;
  while (left_gap < -M_PI)
    left_gap += 2*M_PI;
  
  double right_gap = cur_node - right_node;
  while (right_gap > M_PI)
    right_gap -= 2*M_PI;
  while (right_gap < -M_PI)
    right_gap += 2*M_PI;
  
  if (abs(left_gap) > upper_bound && abs(right_gap) > upper_bound)
    return true;
  else
    return false;
}

inline bool Local_Replan::overGap(const double& left_node, const double& right_node, const double upper_bound, double& output_gap)
{
  double gap = right_node - left_node;
  while (gap > M_PI)
    gap -= 2*M_PI;
  while (gap < -M_PI)
    gap += 2*M_PI;

  output_gap = gap; 

  if (abs(gap) > upper_bound)
    return true;
  else
    return false;
}

inline double Local_Replan::angleDiff(const double& angle1, const double& angle2)
{
  double gap = angle2 - angle1;
  while (gap > M_PI)
    gap -= 2*M_PI;
  while (gap < -M_PI)
    gap += 2*M_PI;

  return gap;
}

inline bool Local_Replan::findSafePose(Eigen::VectorXd& pose, const double safe_dist, const double find_range)
{
  bool suc = false;

  double range = find_range, step = this->resolution_;
  Eigen::Matrix3Xd sampled_points;
  Eigen::VectorXi sorted_indices;

  Eigen::Vector3d position = pose.head(3);
  Eigen::Vector2d py(pose(3), pose(4));
  Eigen::Vector3d ray_dir = path_tools::pyToVec(py);

  bool is_unknown = this->mapping_utils_->getOccupancy(position) == SDFMap::OCCUPANCY::UNKNOWN;

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
    
    sampled_points = grid.colwise() + pose.head(3);
    Eigen::VectorXd distances = (grid.colwise().squaredNorm()).array().sqrt();
    sorted_indices = Eigen::VectorXi::LinSpaced(total_points, 0, total_points - 1);
    std::sort(sorted_indices.data(), sorted_indices.data() + sorted_indices.size(),
                [&distances](int i, int j) {
                    return distances(i) < distances(j);
                });
  }
  
  double z_min = ground_z_ + 2*drone_radius_;

  for (int i = 0; i < (int)sampled_points.cols(); ++i)
  {
    Eigen::Vector3d sample_new_end;
    if (is_unknown) sample_new_end = sampled_points.col(i);
    else sample_new_end = sampled_points.col(sorted_indices(i));

    bool safe = true;

    double z_sample = sample_new_end(2);

    if (!this->mapping_utils_->isInMap(sample_new_end) || z_sample < z_min) safe = false;

    if (!this->mapping_utils_->getSafety(sample_new_end, safe_dist, safe_dist, 0.5*safe_dist) || this->mapping_utils_->getInflateOccupancy(sample_new_end) == 1 || this->mapping_utils_->getOccupancy(sample_new_end) == SDFMap::OCCUPANCY::UNKNOWN) safe = false;
    
    if (safe == true)
    {
      Eigen::Vector3d n2o = position - sample_new_end;
      Eigen::Vector3d ray = this->dist_vp_ * ray_dir;
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
          double original_height = this->dist_vp_ * tan(this->fov_h_half_);
          double cur_dist = this->dist_vp_ - n2o.norm();
          double rot_angle = atan(original_height / cur_dist) - this->fov_h_half_;
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

inline bool Local_Replan::findSafeAnchorPoints(Eigen::VectorXd& pose, const double safe_dist, const double find_range)
{
  bool suc = false;

  double range = find_range, step = this->resolution_;
  Eigen::Matrix3Xd sampled_points;
  Eigen::VectorXi sorted_indices;

  Eigen::Vector3d position = pose.head(3);
  Eigen::Vector2d py(pose(3), pose(4));
  Eigen::Vector3d ray_dir = path_tools::pyToVec(py);

  bool is_unknown = this->mapping_utils_->getOccupancy(position) == SDFMap::OCCUPANCY::UNKNOWN;

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
    
    sampled_points = grid.colwise() + pose.head(3);
    Eigen::VectorXd distances = (grid.colwise().squaredNorm()).array().sqrt();
    sorted_indices = Eigen::VectorXi::LinSpaced(total_points, 0, total_points - 1);
    std::sort(sorted_indices.data(), sorted_indices.data() + sorted_indices.size(),
                [&distances](int i, int j) {
                    return distances(i) < distances(j);
                });
  }
  
  double z_min = ground_z_ + 2*drone_radius_;

  for (int i = 0; i < (int)sampled_points.cols(); ++i)
  {
    Eigen::Vector3d sample_new_end;
    if (is_unknown) sample_new_end = sampled_points.col(i);
    else sample_new_end = sampled_points.col(sorted_indices(i));

    bool safe = true;

    double z_sample = sample_new_end(2);

    if (visibility_st::insideCheck(sample_new_end, pose(3), pose(4), this->scene_, false)) safe = false;

    if (!this->mapping_utils_->isInMap(sample_new_end) || z_sample < z_min) safe = false;

    if (!this->mapping_utils_->getSafety(sample_new_end, safe_dist, safe_dist, 0.5*safe_dist) || this->mapping_utils_->getInflateOccupancy(sample_new_end) == 1 || this->mapping_utils_->getOccupancy(sample_new_end) == SDFMap::OCCUPANCY::UNKNOWN) safe = false;
    
    if (safe == true)
    {
      Eigen::Vector3d n2o = position - sample_new_end;
      Eigen::Vector3d ray = this->dist_vp_ * ray_dir;
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
          double original_height = this->dist_vp_ * tan(this->fov_h_half_);
          double cur_dist = this->dist_vp_ - n2o.norm();
          double rot_angle = atan(original_height / cur_dist) - this->fov_h_half_;
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

inline void Local_Replan::execTrajCallback(const quadrotor_msgs::EigenVectorArrayConstPtr& msg)
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

} // namespace flyco

#endif