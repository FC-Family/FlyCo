/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    May. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the header file of sk_decomp class, which implements a new
 *                   robust and adaptive skeleton extraction & decomposition in FlyCo.
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

#ifndef _SK_DECOMP_H_
#define _SK_DECOMP_H_

#include "skeleton_decomp/Extra_Del.h"
#include "skeleton_decomp/datawrapper.h"
#include "skeleton_decomp/adaptive_utils.h"
#include "vis_utils/planning_visualization.h"
#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/filters/random_sample.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <chrono>
#include <random>
#include <atomic>
#include <queue>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

using namespace std;
using std::unique_ptr;
using std::shared_ptr;

namespace flyco
{

struct Vector3dCompareFunc 
{
  bool operator()(const Eigen::Vector3d& a, const Eigen::Vector3d& b) const 
  {
    for (int i = 0; i < 3; ++i) 
    {
      if (std::abs(a[i] - b[i]) > 1e-6) 
      {
        return a[i] < b[i];
      }
    }
    return false;
  }
};

struct pclCompareFunc 
{
  bool operator()(const pcl::PointXYZ& a, const pcl::PointXYZ& b) const 
  {
    for (int i = 0; i < 3; ++i) 
    {
      if (std::abs(a.data[i] - b.data[i]) > 1e-6) 
      {
        return a.data[i] < b.data[i];
      }
    }
    return false;
  }

};

struct hlvis_graph_node
{
  int subspace_idx;
  Eigen::Vector3d subspace_center;
  vector<int> base_nodes; 
  vector<int> boundary_nodes;
  vector<int> connected_subspaces;
  unordered_map<int, Eigen::Vector2i> edges; // key: subspace index, value: <self_node, neighbor_node>.
};

struct data_wrapper
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr pts_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr vmap_pts_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr ori_pts_;
  pcl::PointCloud<pcl::Normal>::Ptr normals_;
  pcl::PointCloud<pcl::Normal>::Ptr ori_nrs_;
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals;
  Eigen::MatrixXd pts_mat;
  Eigen::MatrixXd nrs_mat;
  double* datas;
  double scale;
  Eigen::Vector3d center;
  // ! FPS
  vector<int> selected_idx_; // FPS selected indices
  pcl::PointCloud<pcl::PointXYZ>::Ptr selected_pts_;
  pcl::PointCloud<pcl::Normal>::Ptr selected_normals_;
  Eigen::MatrixXd selected_pts_mat;
  Eigen::MatrixXd selected_nrs_mat;
  map<int, int> selected_idx_map_; // selected_idx_map_[i] = j, i is the index in the original point cloud, j is the index in the selected point cloud.
  map<int, int> selected_idx_map_inv_; // selected_idx_map_inv_[j] = i, j is the index in the selected point cloud, i is the index in the original point cloud.
  // ! skeletal points
  vector<vector<int>> surf_neighs;
  vector<Eigen::Vector3d> inliers;        // strict voxelized inliers that passed inside/raytracing checks
  vector<Eigen::Vector3d> decomp_points;  // active decomposition nodes; strict + relaxed points in dcROSA branch-subspace mode
  Eigen::MatrixXd strict_inlier_candidates;   // raw candidates that pass strict inside/raytracing check
  Eigen::MatrixXd relaxed_inlier_candidates;  // raw candidates near the mesh surface but not strict
  Eigen::MatrixXd rejected_inlier_candidates; // raw candidates that are neither strict nor near-surface relaxed
  vector<Eigen::Vector3d> branch_decomp_points; // strict + relaxed candidates assigned to dcROSA branches for space decomposition
  Eigen::MatrixXi strict_visibility_graph;
  Eigen::MatrixXi visibility_graph;
  vector<vector<int>> subspace_sets; // final node groups used for space allocation; dcROSA branch groups when branch-subspace mode is active
  vector<Eigen::Vector3d> subspace_centers;
  unordered_map<int, int> node_subspace_map;
  unordered_map<int, Eigen::Vector3d> subspace_maindir;
  unordered_map<int, vector<int>> subspace_merge_sets;
  unordered_map<int, bool> subspace_states;
  vector<hlvis_graph_node> hl_vis_sets;
  Eigen::MatrixXi hl_vis_graph;
  Eigen::MatrixXd dir_sim_graph;
  Eigen::MatrixXd center_dir_cst_graph;
  Eigen::MatrixXd local_dir_cst_graph;
  map<vector<int>, vector<int>> hl_vis_neighs; 
  vector<vector<Eigen::Vector3d>> subspace_skeleton_paths; // representative path for each subspace
  Eigen::MatrixXd dcrosa_skelver; // dcROSA-derived topology vertices in world frame
  Eigen::MatrixXi dcrosa_skeladj; // dcROSA-derived topology adjacency
  vector<vector<Eigen::Vector3d>> dcrosa_skeleton_paths; // branch paths extracted from dcROSA topology
  // ! space allocation
  unordered_map<int, int> inlier_sub; // inlier idx -> sub_space idx
  map<pcl::PointXYZ, int, pclCompareFunc> tg_pt_idx;
  unordered_map<int, Eigen::Vector3d> pt_inlier_pairs; // point idx -> inlier pos
  vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> sub_space_scale;
  vector<pcl::PointCloud<pcl::PointNormal>::Ptr> sub_space_ptnr;
  pcl::PointCloud<pcl::PointXYZ>::Ptr dense_inliers;
  pcl::PointCloud<pcl::PointXYZ>::Ptr strict_dense_inliers;

  void clear()
  {
    pts_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    vmap_pts_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    ori_pts_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    normals_.reset(new pcl::PointCloud<pcl::Normal>);
    ori_nrs_.reset(new pcl::PointCloud<pcl::Normal>);
    cloud_with_normals.reset(new pcl::PointCloud<pcl::PointNormal>);
    pts_mat.resize(0, 0);
    nrs_mat.resize(0, 0);
    datas = nullptr;
    scale = 0.0;
    center = Eigen::Vector3d::Zero();
    selected_idx_.clear();
    selected_pts_.reset();
    selected_normals_.reset(new pcl::PointCloud<pcl::Normal>);
    selected_pts_mat.resize(0, 0);
    selected_nrs_mat.resize(0, 0);
    selected_idx_map_.clear();
    selected_idx_map_inv_.clear();
    surf_neighs.clear();
    inliers.clear();
    decomp_points.clear();
    strict_inlier_candidates.resize(0, 0);
    relaxed_inlier_candidates.resize(0, 0);
    rejected_inlier_candidates.resize(0, 0);
    branch_decomp_points.clear();
    strict_visibility_graph.resize(0, 0);
    visibility_graph.resize(0, 0);
    subspace_sets.clear();
    subspace_centers.clear();
    node_subspace_map.clear();
    subspace_maindir.clear();
    subspace_merge_sets.clear();
    subspace_states.clear();
    hl_vis_sets.clear();
    hl_vis_graph.resize(0, 0);
    dir_sim_graph.resize(0, 0);
    center_dir_cst_graph.resize(0, 0);
    local_dir_cst_graph.resize(0, 0);
    hl_vis_neighs.clear();
    subspace_skeleton_paths.clear();
    dcrosa_skelver.resize(0, 0);
    dcrosa_skeladj.resize(0, 0);
    dcrosa_skeleton_paths.clear();
    inlier_sub.clear();
    tg_pt_idx.clear();
    pt_inlier_pairs.clear();
    dense_inliers.reset(new pcl::PointCloud<pcl::PointXYZ>);
    strict_dense_inliers.reset(new pcl::PointCloud<pcl::PointXYZ>);
    sub_space_scale.clear();
    sub_space_ptnr.clear();
  }
};

class Extra_Del;
class DataWrapper;
class PlanningVisualization;

class sk_decomp
{
public:
  sk_decomp(){
  }
  ~sk_decomp(){
    stopVisualization();
  }

  void init(ros::NodeHandle& nh);
  /* Func */
  void main();
  void set_mesh(Eigen::MatrixXd& mesh_V, Eigen::MatrixXi& mesh_F);
  void setScene(RTCScene input_scene);
  void subspace_representative(vector<int>& subspace_set, vector<Eigen::Vector3d>& path);
  void subspace_representative(vector<int>& subspace_set, vector<vector<Eigen::Vector3d>>& paths);
  bool gen_hlvis_edge(vector<int>& subspace_set_1, vector<int>& subspace_set_2, int& idx_1, int& idx_2);
  void stopVisualization();
  /* Param */
  std::atomic_bool visFlag{false};
  /* Data */
  data_wrapper P;
  string input_mesh = "";
  /* Utils */
  shared_ptr<adaptive_utils> adp_utils_ = nullptr;

private:
  /* Param */
  int calNum = 0, ne_KNN = 0, estNum = 0, selected_num = 0, k_KNN = 0, iter_ext = 0, iter_shr = 0;
  double pt_downsample_voxel_size = 0.0, delta = 0.0, epsilon = 0.0;
  double orientation_convergence_thresh_ = 1e-3;
  double relaxed_candidate_dist_ = -1.0;
  int hybrid_decomp_max_points_ = 50;
  double hybrid_component_radius_scale_ = 0.0;
  double hybrid_projection_weight_ = 1.0;
  int hybrid_min_component_size_ = 4;
  int hybrid_dcrosa_iter_ = 1;
  double hybrid_dcrosa_confidence_th_ = 0.5;
  double hybrid_dcrosa_neighbor_blend_ = 1.0;
  int hybrid_smooth_iter_ = 0;
  bool dcrosa_branch_subspace_ready_ = false;
  double dcrosa_topology_sample_radius_ = 0.0;
  double dcrosa_topology_effective_radius_ = 0.0;
  double dcrosa_guided_min_branch_len_scale_ = 1.0;
  double dcrosa_guided_confidence_ratio_ = 0.75;
  int dcrosa_guided_min_group_size_ = 3;
  int dcrosa_topology_min_support_ = 2;
  /* Data */
  int pcd_size_ = 0, inliers_size = 0;
  double norm_scale = 0.0, edge_min_dist = 0.0;
  Eigen::Vector4d centroid;
  Eigen::MatrixXd pset, vset, vvar;
  Eigen::MatrixXd pset_check;
  Eigen::MatrixXf pset_dir;
  vector<Eigen::Vector3d> pc_set;
  vector<int> pc_cor_idx;
  vector<vector<Eigen::Vector3d>> intersec_pts;
  /* Func */
  void set_input();
  void load_pcd();
  void process_data();
  void normalize();
  void inliers_compression();
  void inliers_extraction();
  void inliers_shrinking();
  void inliers_check();
  vector<vector<int>> hybrid_active_sample_sets(int idx, Eigen::Vector3d& p_cut, Eigen::Vector3d& v_cut);
  bool robust_drosa_point(const vector<vector<int>>& sample_sets, Eigen::Vector3d& point);
  bool estimate_drosa_from_indices(const vector<int>& indices, Eigen::Vector3d& point, double& confidence);
  void hybrid_dcrosa();
  void smooth_hybrid_drosa();
  void build_dcrosa_topology();
  void apply_dcrosa_guided_decomposition();
  void build_branch_subspace_graph();
  vector<vector<Eigen::Vector3d>> extract_dcrosa_branches(const Eigen::MatrixXd& skelver, const Eigen::MatrixXi& skeladj) const;
  double dcrosa_branch_length(const vector<Eigen::Vector3d>& path) const;
  int nearest_dcrosa_branch(const Eigen::Vector3d& point,
                            const vector<char>* valid_branch_mask = nullptr,
                            double* best_dist_sq = nullptr,
                            double* second_best_dist_sq = nullptr) const;
  Eigen::MatrixXi build_visibility_graph(const vector<Eigen::Vector3d>& points, RTCScene query_scene, double& min_edge_dist);
  const vector<Eigen::Vector3d>& active_decomp_points() const;
  const vector<vector<int>>& active_subspace_sets() const;
  void build_strict_internal_inliers();
  void fill_subspace_map(const vector<vector<int>>& subspace_sets, unordered_map<int, int>& node_subspace_map) const;
  vector<vector<int>> visibility_subspace_decomp(const Eigen::MatrixXi& graph, unordered_map<int, int>* node_subspace_map = nullptr);
  void build_subspace_skeleton_paths();
  void distribute_ori_cloud();
  void merge_fallback_subspaces();

  void high_level_vis_graph();
  /* Utils */
  shared_ptr<PlanningVisualization> vis_utils_;
  void inliers_initialize(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::Normal>::Ptr& normals);
  Eigen::Matrix3d create_orthonormal_frame(Eigen::Vector3d& v);
  Eigen::MatrixXd rosa_compute_active_samples(int& idx, Eigen::Vector3d& p_cut, Eigen::Vector3d& v_cut);
  void pcloud_isoncut(Eigen::Vector3d& p_cut, Eigen::Vector3d& v_cut, vector<int>& isoncut, double*& datas, int& size);
  void distance_query(DataWrapper& data, const vector<double>& Pp, const vector<double>& Np, double delta, vector<int>& isoncut);
  Eigen::Vector3d compute_symmetrynormal(Eigen::MatrixXd& local_normals);
  double symmnormal_variance(Eigen::Vector3d& symm_nor, Eigen::MatrixXd& local_normals);
  Eigen::Vector3d symmnormal_smooth(Eigen::MatrixXd& V, Eigen::MatrixXd& w);
  Eigen::Vector3d closest_projection_point(Eigen::MatrixXd& P, Eigen::MatrixXd& V);
  bool canJoinVisibilitySubspace(const Eigen::MatrixXi& graph, const vector<int>& subspace_set, int node);
  Eigen::Vector3d PCA(Eigen::MatrixXd& A);
  int findMostFrequentElement(const vector<int>& vec);
  void visualization(const ros::TimerEvent& e);
  bool visualizationDataReadyUnsafe() const;
  bool validVisGraphUnsafe() const;
  /* Timer */
  ros::Timer vis_timer_;
  std::atomic_bool vis_data_ready_{false};
  mutable std::mutex data_mutex_;
};

inline void sk_decomp::normalize()
{
  auto norm_t1 = chrono::high_resolution_clock::now();
  
  P.ori_pts_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  P.ori_nrs_.reset(new pcl::PointCloud<pcl::Normal>);
  pcl::copyPointCloud(*P.pts_, *P.ori_pts_);
  pcl::copyPointCloud(*P.normals_, *P.ori_nrs_);

  pcl::PointXYZ min;
  pcl::PointXYZ max;
  pcl::getMinMax3D(*P.pts_,min,max);
  double x_scale, y_scale, z_scale, max_scale;
  x_scale = max.x-min.x; y_scale = max.y-min.y; z_scale = max.z-min.z; 
  if (x_scale >= y_scale)
    max_scale = x_scale;
  else 
    max_scale = y_scale;
  if (max_scale < z_scale)
    max_scale = z_scale;

  norm_scale = max_scale;
  pcl::compute3DCentroid(*P.pts_, centroid);

  P.scale = norm_scale;
  P.center(0) = centroid(0);
  P.center(1) = centroid(1);
  P.center(2) = centroid(2);
  
  // normalize
  for (int i=0; i<(int)P.pts_->points.size(); ++i)
  {
    P.pts_->points[i].x = (P.pts_->points[i].x-centroid(0))/max_scale;
    P.pts_->points[i].y = (P.pts_->points[i].y-centroid(1))/max_scale;
    P.pts_->points[i].z = (P.pts_->points[i].z-centroid(2))/max_scale;
  } 

  // voxel grid downsample
  P.cloud_with_normals.reset(new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointNormal>::Ptr temp_all(new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointNormal>::Ptr temp_all_down(new pcl::PointCloud<pcl::PointNormal>);
  pcl::concatenateFields (*P.pts_, *P.normals_, *temp_all_down);
  pcl::concatenateFields (*P.pts_, *P.normals_, *P.cloud_with_normals);

  auto a_t1 = chrono::high_resolution_clock::now();
  int upper_size = estNum;
  int curPtSize = (int)temp_all_down->points.size();
  while (curPtSize > upper_size)
  {
    pcl::VoxelGrid<pcl::PointNormal> vgx;
    vgx.setInputCloud(temp_all_down);
    vgx.setLeafSize(pt_downsample_voxel_size, pt_downsample_voxel_size, pt_downsample_voxel_size);
    vgx.filter(*temp_all);
    
    curPtSize = (int)temp_all->points.size();
    if (curPtSize > upper_size)
    {
      pt_downsample_voxel_size += 0.002;
    }
  }

  pcl::VoxelGrid<pcl::PointNormal> vgf;
  vgf.setInputCloud(P.cloud_with_normals);
  vgf.setLeafSize(pt_downsample_voxel_size, pt_downsample_voxel_size, pt_downsample_voxel_size);
  vgf.filter(*P.cloud_with_normals);
  curPtSize = (int)P.cloud_with_normals->points.size();

  delta = pt_downsample_voxel_size; //* keep plane thickness and downsample size equal
  epsilon = 1.05*pt_downsample_voxel_size; //* adaptive param for clustering 
  ROS_INFO("Final downsample voxel size: %f, Cloud size: %d.", pt_downsample_voxel_size, (int)P.cloud_with_normals->points.size());
  
  // update pcd info
  pcd_size_ = P.cloud_with_normals->points.size();
  P.pts_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  P.normals_.reset(new pcl::PointCloud<pcl::Normal>);
  P.pts_mat.resize(pcd_size_, 3);
  P.nrs_mat.resize(pcd_size_, 3);

  pcl::PointXYZ pt;
  pcl::Normal normal;
  for (int i=0; i<pcd_size_; ++i)
  {
    pt.x = P.cloud_with_normals->points[i].x; 
    pt.y = P.cloud_with_normals->points[i].y; 
    pt.z = P.cloud_with_normals->points[i].z; 
    normal.normal_x = -P.cloud_with_normals->points[i].normal_x; 
    normal.normal_y = -P.cloud_with_normals->points[i].normal_y; 
    normal.normal_z = -P.cloud_with_normals->points[i].normal_z;
    P.pts_->points.push_back(pt);
    P.normals_->points.push_back(normal);
    P.pts_mat(i,0) = P.pts_->points[i].x; 
    P.pts_mat(i,1) = P.pts_->points[i].y; 
    P.pts_mat(i,2) = P.pts_->points[i].z;
    P.nrs_mat(i,0) = P.normals_->points[i].normal_x; 
    P.nrs_mat(i,1) = P.normals_->points[i].normal_y; 
    P.nrs_mat(i,2) = P.normals_->points[i].normal_z;
  }

  // FPS sampling for skeletal points
  P.selected_idx_ = adp_utils_->fps(P.cloud_with_normals, selected_num);

  P.selected_pts_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  P.selected_normals_.reset(new pcl::PointCloud<pcl::Normal>);
  P.selected_pts_mat.resize(selected_num, 3);
  P.selected_nrs_mat.resize(selected_num, 3);

  for (int i=0; i<selected_num; ++i)
  {
    int idx = P.selected_idx_[i];
    P.selected_idx_map_[i] = idx;
    P.selected_idx_map_inv_[idx] = i;

    pt.x = P.cloud_with_normals->points[idx].x; 
    pt.y = P.cloud_with_normals->points[idx].y; 
    pt.z = P.cloud_with_normals->points[idx].z;
    normal.normal_x = P.cloud_with_normals->points[idx].normal_x; 
    normal.normal_y = P.cloud_with_normals->points[idx].normal_y; 
    normal.normal_z = P.cloud_with_normals->points[idx].normal_z;
    P.selected_pts_->points.push_back(pt);
    P.selected_normals_->points.push_back(normal);
    P.selected_pts_mat(i,0) = P.selected_pts_->points[i].x;
    P.selected_pts_mat(i,1) = P.selected_pts_->points[i].y; 
    P.selected_pts_mat(i,2) = P.selected_pts_->points[i].z;
    P.selected_nrs_mat(i,0) = P.selected_normals_->points[i].normal_x; 
    P.selected_nrs_mat(i,1) = P.selected_normals_->points[i].normal_y; 
    P.selected_nrs_mat(i,2) = P.selected_normals_->points[i].normal_z;
  }

  auto norm_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> norm_ms = norm_t2 - norm_t1;
  double norm_time = (double)norm_ms.count();
  ROS_INFO("\033[32m[SSD] normalize time = %lf ms.\033[32m", norm_time);
}

class dfs_cluster
{
public:
  inline void dfs(const MatrixXi& matrix, int node, vector<bool>& visited, vector<int>& component)
  {
    visited[node] = true;
    component.push_back(node);

    for(int i = 0; i < matrix.cols(); ++i) 
    {
      if(matrix(node, i) == 1 && !visited[i] && i != node) {
        dfs(matrix, i, visited, component);
      }
    }
  }

  inline vector<vector<int>> find_directly_connected_components(const MatrixXi& matrix) {
    int n = matrix.rows();
    vector<bool> visited(n, false);
    vector<vector<int>> components;

    for(int i = 0; i < n; ++i) {
        if(!visited[i]) {
            vector<int> component;
            dfs(matrix, i, visited, component);
            if(!component.empty()) {
                components.push_back(component);
            }
        }
    }

    return components;
  }
};

}

#endif
