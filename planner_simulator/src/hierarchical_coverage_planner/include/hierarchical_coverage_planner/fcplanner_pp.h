/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Jun. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the header file of FCPlanner_PP class, which implements
 *                   hierarchical coverage planning in FlyCo.
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

#ifndef _FCPLANNER_PP_H_
#define _FCPLANNER_PP_H_

#include "skeleton_decomp/adaptive_utils.h"
#include "skeleton_decomp/sk_decomp.h"
#include "active_perception/perception_utils.h"
#include "viewpoint_manager/viewpoint_manager.h"
#include "hierarchical_coverage_planner/hcsolver.h"
#include "plan_env/raycast.h"
#include "plan_env/sdf_map.h"
#include "vis_utils/planning_visualization.h"
#include "quadrotor_msgs/EigenVectorArray.h"
#include "quadrotor_msgs/BEVBox.h"
#include "plan_utils/visibility_st.hpp"
#include "plan_utils/path_tools.hpp"
#include "plan_utils/cvx_decomp.hpp"
#include "plan_utils/solver_tools.hpp"
#include <ros/ros.h>
#include <ros/package.h>
#include <iomanip>
#include <sstream>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <yaml-cpp/yaml.h>

using namespace std;
using std::unique_ptr;
using std::shared_ptr;
typedef pcl::PointXYZ PointT;

class RayCaster;

namespace flyco
{

class adaptive_utils;
class sk_decomp;
class PerceptionUtils;
class ViewpointManager;
class HCSolver;
class SDFMap;
class PlanningVisualization;

class FCPlanner_PP
{
struct VectorHash {
    size_t operator()(const std::vector<double>& v) const {
        std::hash<double> hasher;
        size_t seed = 0;
        for (double i : v) {
            seed ^= hasher(i) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }
        return seed;
    }
};

struct VectorXdCompare
{
  bool operator()(const Eigen::VectorXd &v1, const Eigen::VectorXd &v2) const
  {
    for (int i = 0; i < v1.size(); ++i)
    {
      if (v1(i) != v2(i))
        return v1(i) < v2(i);
    }
    return false;
  }
};

struct Vector3iHash 
{
    std::size_t operator()(const Eigen::Vector3i& key) const {
        return std::hash<int>()(key.x()) ^ std::hash<int>()(key.y()) ^ std::hash<int>()(key.z());
    }
};

struct PointXYZCompare {
    bool operator()(const pcl::PointXYZ& p1, const pcl::PointXYZ& p2) const {
        if (p1.x != p2.x) return p1.x < p2.x;
        if (p1.y != p2.y) return p1.y < p2.y;
        return p1.z < p2.z;
    }
};

struct Viewpoint
{
  int sub_id;
  int vp_id;
  Eigen::VectorXd pose;
  int vox_count;
};

struct PlanningDataWrapper
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr ori_model;
  pcl::PointCloud<pcl::PointXYZ>::Ptr model;
  pcl::PointCloud<pcl::PointXYZ>::Ptr uncovered_model_local_;
  map<Eigen::Vector3d, Eigen::Vector3d, Vector3dCompare0> pt_normal_pairs;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cur_hc_map_;
  // * Uncovered Surface
  vector<vector<bool>> uncovered_surface_;
  // * Topology Graph
  vector<Eigen::Vector3d> topoNodes;
  vector<vector<int>> graph_nodes_idx;
  vector<Eigen::Vector3d> graph_nodes;
  map<vector<int>, int> realEdges;
  Eigen::MatrixXd topoAdjMat;
  map<vector<int>, vector<Eigen::Vector3d>> ctrlPts;
  map<int, Eigen::Vector3d> nearestPts;
  // * Viewpoint
  map<Eigen::Vector3d, int, Vector3dCompare0> pt_sub_pairs;
  map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> sub_vps_inflate;
  map<int, Eigen::MatrixXd> sub_vps_pose;
  vector<Viewpoint> vps_set_;
  map<int, vector<Eigen::VectorXd>> final_sub_vps_pairs;
  // * Planning
  vector<Eigen::VectorXd> last_path_;
  vector<bool> last_wp_indicators_;
  vector<Eigen::VectorXd> prior_path;
  vector<Eigen::VectorXd> prior_remaining_;
  vector<bool> prior_wp_indicators_;
  vector<int> global_seq_;
  vector<Eigen::Vector3d> sub_centroids_;
  map<int, vector<int>> global_boundary_id_;
  map<int, vector<Eigen::VectorXd>> local_paths_viewpts_; // sub-space: [viewpoint, viewpoint, ...]
  map<int, vector<vector<Eigen::VectorXd>>> local_paths_waypts_; // sub-space: [waypts, waypts, ...]
  vector<Eigen::VectorXd> full_viewpoints_;
  vector<vector<Eigen::VectorXd>> full_waypoints_;
  unordered_map<vector<double>, bool, VectorHash> viewpoints_detector_;
  unordered_map<vector<double>, int, VectorHash> FullPathId_;
  vector<Eigen::VectorXd> FullPath_;
  vector<Eigen::Vector3d> connectJoints;
  double searchRange;
  vector<vector<Eigen::Vector3d>> innerVps;
  vector<bool> waypoints_indicators_;
  map<Eigen::VectorXd, bool, VectorXdCompare> waypt_indi_;
  // * Sub Consistent Planning
  vector<Eigen::Vector3d> next_sub_inliers;
  vector<Eigen::VectorXd> next_sub_vps;
  vector<Eigen::Vector3d> last_time_next_sub_inliers;
  vector<Eigen::Vector3d> this_time_last_4_selection;

  // * Decomp Planning

  // ? History: last time buffer
  vector<Eigen::VectorXd> LAST_total_viewpoints_;
  vector<vector<int>> LAST_decomp_vps_;
  vector<int> LAST_decomp_global_seq_;
  vector<vector<Eigen::VectorXd>> LAST_reused_viewpoints_;
  vector<Eigen::VectorXd> LAST_remaining_viewpoints_;
  vector<vector<vector<Eigen::Vector3d>>> LAST_reused_observations_;
  vector<vector<Eigen::Vector3d>> LAST_remaining_observations_;
  int LAST_reused_sets_ = 1;

  // ? Viewpoint Cluster
  vector<Eigen::VectorXd> total_viewpoints_;
  Eigen::MatrixXi vp_vis_graph_;
  vector<vector<int>> decomp_vps_;
  // ? Planning Data
  vector<int> decomp_global_seq_;
  vector<Eigen::Vector3d> decomp_global_seq_path_;
  vector<Eigen::VectorXd> decomp_full_path_;
  vector<bool> decomp_full_indicators_;

  vector<Eigen::VectorXd> prior_backup_;
  vector<bool> prior_indi_backup_;

  // * Ground Priority Flight
  pcl::PointCloud<pcl::PointXYZ>::Ptr safe_g_vps_;
  vector<Eigen::VectorXd> sampled_g_vps_set_;
  vector<Eigen::VectorXd> ground_priority_path_;
  vector<bool> ground_priority_indicators_;

  void clear()
  {
    ori_model.reset(new pcl::PointCloud<pcl::PointXYZ>);
    model.reset(new pcl::PointCloud<pcl::PointXYZ>);
    uncovered_model_local_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    cur_hc_map_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    pt_normal_pairs.clear();
    uncovered_surface_.clear();
    topoNodes.clear();
    graph_nodes_idx.clear();
    graph_nodes.clear();
    realEdges.clear();
    topoAdjMat = Eigen::MatrixXd();
    ctrlPts.clear();
    nearestPts.clear();
    pt_sub_pairs.clear();
    sub_vps_inflate.clear();
    sub_vps_pose.clear();
    vps_set_.clear();
    final_sub_vps_pairs.clear();
    global_seq_.clear();
    sub_centroids_.clear();
    global_boundary_id_.clear();
    local_paths_viewpts_.clear();
    local_paths_waypts_.clear();
    full_viewpoints_.clear();
    full_waypoints_.clear();
    viewpoints_detector_.clear();
    FullPathId_.clear();
    FullPath_.clear();
    connectJoints.clear();
    searchRange = 0.0;
    innerVps.clear();
    waypoints_indicators_.clear();
    last_path_.clear();
    last_wp_indicators_.clear();
    prior_path.clear();
    prior_remaining_.clear();
    prior_wp_indicators_.clear();
    waypt_indi_.clear();
    next_sub_inliers.clear();
    next_sub_vps.clear();
    this_time_last_4_selection.clear();

    total_viewpoints_.clear();
    decomp_vps_.clear();
    decomp_global_seq_.clear();
    decomp_global_seq_path_.clear();
    decomp_full_path_.clear();
    decomp_full_indicators_.clear();

    prior_backup_.clear();
    prior_indi_backup_.clear();

    safe_g_vps_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    sampled_g_vps_set_.clear();
    ground_priority_path_.clear();
    ground_priority_indicators_.clear();
  }
};

public:
  FCPlanner_PP(){
  }
  ~FCPlanner_PP(){
  }
  /* Func */
  void init(ros::NodeHandle& nh);
  void setInput(Eigen::MatrixXd& mesh_V, Eigen::MatrixXi& mesh_F);
  void setMap(shared_ptr<SDFMap> map);
  void reset();
  bool plan(Eigen::VectorXd& current_pose, Eigen::Vector3d& cur_vel, vector<Eigen::VectorXd>& last_global_path, vector<bool>& last_global_indi, bool v1);
  /* Param */
  double corridorProgress = 0.0;
  bool visFlag = false;
  ros::NodeHandle nh_ = ros::NodeHandle();
  /* Data */
  PlanningDataWrapper PR;
  RTCScene scene_ = nullptr;
  /* Utils */
  shared_ptr<SDFMap> HCMap = nullptr;
  unique_ptr<sk_decomp> skeleton_operator = nullptr;
  double min_z_voxelMap = 0.0;
  Eigen::Vector3i xyz;
  Eigen::Vector3d offset;
  /* Statistic */
  double plan_time = 0.0, coverage = 0.0;
  int viewpointNum = 0;
  vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> visible_group;
  /* Evaluation */
  pcl::PointCloud<pcl::PointXYZ>::Ptr Fullmodel, visibleFullmodel;
  void PinHoleCamera(Eigen::VectorXd& vp, vector<int>& covered_id, pcl::PointCloud<pcl::PointXYZ>::Ptr& checkCloud);
  double trajCoverageEval(vector<Eigen::VectorXd>& poseSet);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  /* Func */
  void viewpointGeneration();
  /* Param */
  double dist_vp = 0.0, model_ds_size = 0.0, safe_radius_vp = 0.0, safe_inner_coefficient_ = 0.0, JointCoeff = 0.0, cost_cn_ = 0.0;
  double fov_h = 0.0, fov_w = 0.0, fov_base = 0.0;
  double cx = 0.0, cy = 0.0, fx = 0.0, fy = 0.0;
  double vmax_ = 0.0, amean_ = 0.0;
  double GroundZPos = 0.0, TopZPos = 0.0, safeHeight = 0.0;
  double local_range_ = 0.0, dist_upper_ = 0.0;
  double refine_time_upper_ = 0.0;
  double suspension_rate_ = 0.0;
  double max_cvx_range_ = 0.0;
  double tsp_horizon_ = 0.0;
  int sample_size_ = 0;
  int match_num_ = 0;
  string mode_ = "";
  bool zFlag, global_consistency_, suspension_ = false;
  Eigen::Vector3d current_pos_, updated_pos_, remain_pos_;
  Eigen::VectorXd start_pose_, updated_pose_, remain_pose_;
  Eigen::Vector3d start_vel_, last_vel_;
  bool sub_consistent_ = false, open_chamfer_ = false;
  double overlap_last_ = 0.0;
  double drone_radius_ = 0.0;
  int vpm_init_size_ = 0;
  int save_visible_lower_ = 0;
  double vis_inflation_ = 0.0;

  bool ground_priority_ = true;
  double ground_ds_size_ = 0.0;
  double find_range_ = 5.0;
  double v_fov_ = 0.0;
  double ground_flight_height_ = 0.0;
  double g_vps_sample_dist_ = 0.0;

  bool bev_seg_ = false;
  double bev_min_x_ = 0.0, bev_max_x_ = 0.0, bev_min_y_ = 0.0, bev_max_y_ = 0.0;

  bool en_inherit_ = false;
  /* Data */
  string mesh;
  pcl::PointCloud<pcl::PointXYZ>::Ptr occ_model, good_vps_cloud;
  pcl::PointCloud<pcl::PointXYZ>::Ptr uncovered_surf_, planning_target_surf_;
  pcl::PointCloud<pcl::PointXYZ> uncovered_area;
  pcl::PointCloud<pcl::PointNormal>::Ptr all_safe_normal_vps;
  map<pcl::PointXYZ, int, pclCompareFunc> good_vps_sub;
  vector<Eigen::VectorXd> valid_viewpoints;
  pcl::KdTreeFLANN<pcl::PointXYZ> good_vps_tree;
  Eigen::MatrixXd mesh_V_;
  Eigen::MatrixXi mesh_F_;
  vector<Eigen::VectorXd> cur_exec_traj_;
  vector<Eigen::VectorXd> before_consistency_trunc_path_;
  double bridge_cost_A_ = 0.0, bridge_cost_B_ = 0.0;
  int reused_viewpoints_num_ = 0;
  int match_open_ = 0;
  /* Utils */
  unique_ptr<RayCaster> raycaster_ = nullptr;
  unique_ptr<PerceptionUtils> percep_utils_ = nullptr;
  unique_ptr<ViewpointManager> viewpoint_manager_ = nullptr;
  unique_ptr<HCSolver> solver_ = nullptr;
  shared_ptr<PlanningVisualization> vis_utils_ = nullptr;
  /* Tools */
  void vis_Callback(const ros::TimerEvent& e);
  void execTrajCallback(const quadrotor_msgs::EigenVectorArrayConstPtr& msg);
  void bevCallback(const quadrotor_msgs::BEVBoxConstPtr& msg);
  void setMSB();
  void mapResults();
  void uniformSampling();
  void reviseNormals();
  void activeViewpoints();
  void interFullPath(double dist_bound);
  void cameraModelProjection(Eigen::VectorXd& pose, Eigen::Vector3d& point, double& xRes, double& yRes, Eigen::Vector2d& leftdown, double& distance, Eigen::Vector2i& inCam);
  void pathCoverageEval();
  void updateUncoveredArea();

  void findReusedViewpoints(const double& range, vector<vector<Eigen::VectorXd>>& reused_viewpoints, vector<Eigen::VectorXd>& remaining_viewpoints, vector<vector<vector<Eigen::Vector3d>>>& reused_observations, vector<vector<Eigen::Vector3d>>& remaining_observations);
  void decompInterpolatePath(double dist_bound, vector<Eigen::VectorXd>& path, vector<bool>& indicators);
  void decompViewpoints(map<int, vector<Eigen::VectorXd>>& sub_vps);
  void decompPlanning();

  void findGroundPrioritySet();
  void updateGroundPrioritySet();
  bool findSafePose(Eigen::VectorXd& pose, const double safe_dist);
  bool updateRealPose(Eigen::VectorXd& pose, const double safe_dist);
  /* ROS Service */
  ros::Timer vis_timer_;
  ros::Subscriber exec_traj_sub_, bev_sub_;
};

} // namespace flyco

#endif