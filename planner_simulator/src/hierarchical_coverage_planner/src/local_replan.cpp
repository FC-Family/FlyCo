/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Jul. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the main file of Local Motion Replanning in
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

#include "hierarchical_coverage_planner/local_replan.h"

namespace flyco
{

void Local_Replan::init(ros::NodeHandle& nh)
{
  // * Module Initialization
  this->percep_utils_.reset(new PerceptionUtils);
  this->motion_planner_.reset(new TrajGenerator);
  this->percep_utils_->init(nh);
  this->motion_planner_->init(nh);
  this->astar_.reset(new Astar);
  this->astar_->init_hc(nh);
  this->raycaster_.reset(new RayCaster);

  // * Params Initialization
  nh.param("viewpoint_manager/zGround", zFlag, false);
  nh.param("viewpoint_manager/GroundPos", ground_z_, -1.0);
  nh.param("viewpoint_manager/safeHeight", safe_height_, -1.0);
  nh.param("viewpoint_manager/fov_h", fov_h_, -1.0);
  nh.param("viewpoint_manager/fov_w", fov_w_, -1.0);
  nh.param("perception_utils/top_angle", fov_h_half_, -1.0);
  nh.param("perception_utils/left_angle", fov_w_half_, -1.0);
  nh.param("hcplanner/localRange", local_region_, -1.0);
  nh.param("viewpoint_manager/visible_range", visible_dist_, -1.0);
  nh.param("viewpoint_manager/viewpoints_distance", dist_vp_, -1.0);
  nh.param("hcmap/resolution", resolution_, -1.0);
  nh.param("hctraj/drone_radius", drone_radius_, -1.0);
  nh.param("hctraj/vmax_", vmax_, -1.0);
  nh.param("hctraj/ydmax_", wmax_, -1.0);
  nh.param("hcplanner/amean_", amean_, -1.0);
  nh.param("sdf_map/obstacles_inflation", safe_inflation_, 0.3);
  nh.param("viewpoint_manager/vis_inf", vis_inflation_, 0.5);
  nh.param("viewpoint_manager/save_lower", save_visible_lower_, 5);

  sfc_prog_ = 15.0;
  dist_thresh_ = 0.8;
  min_dist_ = dist_thresh_;

  // * ROS Service
  this->exec_traj_sub_ = nh.subscribe("/traj_server/exec_traj_waypts", 1, &Local_Replan::execTrajCallback, this);

  ROS_INFO("\033[35m[LocalReplanner] Initialized! \033[32m");
}

void Local_Replan::reset()
{ 
  if (this->scene_ != nullptr)
  {
    rtcReleaseScene(this->scene_);
    this->scene_ = nullptr;
  }
  if (this->motion_planner_ != nullptr)
    this->motion_planner_->reset();
  
  this->global_path_.clear();
  this->global_waypt_indicators_.clear();
  this->global_prior_path_.clear();
  this->global_tree_ = pcl::KdTreeFLANN<pcl::PointXYZ>();
  this->local_path_.clear();
  this->local_waypt_indicators_.clear();
  this->waypts_finder_.clear();
  this->global_hc_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  this->global_model_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  this->local_hc_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  this->local_traj_occ_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  this->local_replan_path_.clear();
  this->local_replan_indi_.clear();
  this->local_prior_order_.clear();
  this->local_updated_path_.clear();
  this->pos_traj_.clear();
  this->orientation_traj_.clear();

  return;
}

bool Local_Replan::replan(bool& feasible, const Eigen::VectorXd& local_start, const Eigen::Vector3d vel, const Eigen::Vector3d acc, const Eigen::Vector3d pyd, const Eigen::Vector3d pyd_dot)
{
  this->motion_planner_->visFlag = true;

  // * Dynamic Parameters
  this->vel_ = vel;
  this->acc_ = acc;
  this->pyd_ = pyd;
  this->pyd_dot_ = pyd_dot;

  // * Local Trajectory Planning
  auto t1 = chrono::high_resolution_clock::now();
  bool find_range = localRange(local_start);
  if (find_range == false)
  {
    return false;
  }
  auto t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> l_ms = t2 - t1;
  double l_time = (double)l_ms.count();
  ROS_INFO("\033[32m[LocalReplanner] Local Range Time: %f ms \033[32m", l_time);

  localPathVisOpt();
  bool traj_gen = localTrajPlan();
  if (traj_gen == false)
  {
    return false;
  }

  bool feasi = localFeasibility();
  feasible = feasi;

  return feasi;
}

void Local_Replan::setMap(shared_ptr<SDFMap>& map)
{
  this->mapping_utils_ = map;
  this->astar_->setMap(this->mapping_utils_);
  this->raycaster_->setParams(this->mapping_utils_->mp_->resolution_, this->mapping_utils_->mp_->map_origin_);
  this->motion_planner_->setMap(this->mapping_utils_);

  return;
}

void Local_Replan::setMesh(Eigen::MatrixXd& mesh_V, Eigen::MatrixXi& mesh_F)
{
  visibility_st::mesh2scene(mesh_V, mesh_F, this->scene_);

  return;
}

void Local_Replan::setGlobalPath(vector<Eigen::VectorXd>& path, vector<bool>& way_indi)
{
  this->global_path_ = path;
  this->global_waypt_indicators_ = way_indi;
  pcl::PointCloud<pcl::PointXYZ>::Ptr global_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i=0; i<this->global_path_.size(); ++i)
  {
    pcl::PointXYZ pt;
    pt.x = this->global_path_[i](0);
    pt.y = this->global_path_[i](1);
    pt.z = this->global_path_[i](2);
    global_cloud->points.push_back(pt);
  }
  this->global_tree_.setInputCloud(global_cloud);

  return;
}

void Local_Replan::setGlobalPriorPath(vector<Eigen::VectorXd>& path)
{
  this->global_prior_path_ = path;

  return;
}

void Local_Replan::setGlobalHCMap(pcl::PointCloud<pcl::PointXYZ>::Ptr& hc)
{
  this->global_hc_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  *this->global_hc_ = *hc;

  return;
}

void Local_Replan::setGlobalModel(pcl::PointCloud<pcl::PointXYZ>::Ptr& model)
{
  this->global_model_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  *this->global_model_ = *model;

  return;
}

void Local_Replan::setGlobalNextSub(vector<Eigen::Vector3d>& inliers, vector<Eigen::VectorXd>& vps)
{
  this->global_next_sub_inliers_ = inliers;
  this->global_next_sub_vps_ = vps;

  return;
}

void Local_Replan::getUsedGlobalRoute(vector<Eigen::VectorXd>& path, vector<Eigen::VectorXd>& prior)
{
  path.clear();
  prior.clear();
  path = this->global_path_;
  prior = this->global_prior_path_;

  return;
}

void Local_Replan::getUsedGlobalNextSub(vector<Eigen::Vector3d>& inliers, vector<Eigen::VectorXd>& vps)
{
  inliers.clear();
  vps.clear();
  inliers = this->global_next_sub_inliers_;
  vps = this->global_next_sub_vps_;

  return;
}

void Local_Replan::getLocalPath(vector<Eigen::VectorXd>& path)
{
  path.clear();
  path = this->local_path_;

  return;
}

void Local_Replan::getLocalUpdatedPath(vector<Eigen::VectorXd>& path)
{
  path.clear();
  path = this->local_updated_path_;

  return;
}

void Local_Replan::getLocalRegion(pcl::PointCloud<pcl::PointXYZ>::Ptr& hc, Eigen::Vector3d& min_bound, Eigen::Vector3d& max_bound)
{
  hc.reset(new pcl::PointCloud<pcl::PointXYZ>);
  *hc = *this->local_hc_;
  min_bound = this->min_bound_;
  max_bound = this->max_bound_;

  return;
}

void Local_Replan::getLocalTraj(Trajectory<7>& pos_traj, Trajectory<7>& orientation_traj)
{
  pos_traj.clear();
  pos_traj = this->pos_traj_;
  orientation_traj.clear();
  orientation_traj = this->orientation_traj_;

  return;
}

void Local_Replan::getReplanPathWithIndi(vector<Eigen::VectorXd>& path, vector<bool>& indi)
{
  path.clear();
  indi.clear();
  path = this->local_replan_path_;
  indi = this->local_replan_indi_;

  return;
}

} // namespace flyco