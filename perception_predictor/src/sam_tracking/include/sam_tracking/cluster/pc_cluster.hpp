/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Jun. 2026
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is part of the FlyCo Perception public ROS runtime.
 * Copyright    :    Copyright (c) 2026 Chen Feng and Guiyong Zheng.
 * License      :    PolyForm Noncommercial License 1.0.0
 *                   <https://polyformproject.org/licenses/noncommercial/1.0.0/>
 * Project      :    FlyCo: Foundation Model-Empowered Drones for Autonomous 3D Structure Scanning in Open-World Environments
 * Website      :    https://hkust-aerial-robotics.github.io/FC-Planner/
 *⭐⭐⭐******************************************************************⭐⭐⭐*/

/**
 * @file pc_cluster.hpp
 * @brief Header file for the PointCloudCluster class and related structures.
 */

#ifndef PC_CLUSTER_HPP_
#define PC_CLUSTER_HPP_

#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sam_tracking/cluster/dbscan.hpp"
#include "sam_tracking/cluster/pc_utils.hpp"

/**
 * @struct FilterRange
 * @brief Structure representing the range for filtering point cloud data.
 */
struct FilterRange {
  double min_x; /**< 最小 x 值 */
  double max_x; /**< 最大 x 值 */
  double min_y; /**< 最小 y 值 */
  double max_y; /**< 最大 y 值 */
  double min_z; /**< 最小 z 值 */
  double max_z; /**< 最大 z 值 */

  /**
   * @brief 默认构造函数，初始化所有范围为无效值。
   */
  FilterRange()
      : min_x(std::numeric_limits<double>::lowest()),
        max_x(std::numeric_limits<double>::max()),
        min_y(std::numeric_limits<double>::lowest()),
        max_y(std::numeric_limits<double>::max()),
        min_z(std::numeric_limits<double>::lowest()),
        max_z(std::numeric_limits<double>::max()) {}

  /**
   * @brief 参数化构造函数，允许直接指定过滤范围。
   * @param min_x 最小 x 值
   * @param max_x 最大 x 值
   * @param min_y 最小 y 值
   * @param max_y 最大 y 值
   * @param min_z 最小 z 值
   * @param max_z 最大 z 值
   */
  FilterRange(double min_x, double max_x, double min_y, double max_y,
              double min_z, double max_z)
      : min_x(min_x),
        max_x(max_x),
        min_y(min_y),
        max_y(max_y),
        min_z(min_z),
        max_z(max_z) {}
};

typedef std::pair<int, int> GridKey;

// 自定义哈希函数以支持 std::unordered_map
struct GridKeyHash {
  std::size_t operator()(const GridKey &key) const {
    return std::hash<int>()(key.first) ^ std::hash<int>()(key.second);
  }
};

/**
 * @class PointCloudCluster
 * @brief Class representing a point cloud cluster.
 */
class PointCloudCluster {
 public:
  /**
   * @brief Constructor for PointCloudCluster class.
   * @param min_height The minimum height of the cluster.
   * @param resolution The resolution for downsampling points.
   * @param cluster_tolerance The tolerance for clustering points.
   * @param min_cluster_size The minimum size of a cluster.
   * @param max_cluster_size The maximum size of a cluster.
   * @param max_points The maximum number of points.
   * @param epsilon The epsilon value for comparison.
   */
  PointCloudCluster(double min_height = 0.0, double resolution = 0.1,
                    double cluster_tolerance = 1, int min_cluster_size = 50,
                    int max_cluster_size = 85000, int max_points = 1024,
                    double epsilon = 0.5, double filter_grid_size = 1,
                    double grid_max_height = 5,
                    double grid_height_threshold = 1);

  /**
   * @brief Setter function for the minimum height of the cluster.
   * @param min_height The minimum height of the cluster.
   */

  void set_filter_grid_size(double filter_grid_size) {
    filter_grid_size_ = filter_grid_size;
  }

  void set_grid_max_height(double grid_max_height) {
    grid_max_height_ = grid_max_height;
  }

  void set_grid_height_threshold(double grid_height_threshold) {
    grid_height_threshold_ = grid_height_threshold;
  }

  void set_min_height(double min_height) { min_height_ = min_height; }

  void set_ground_filter_enabled(bool ground_filter_enabled) {
    ground_filter_mode_ = ground_filter_enabled ? "grid" : "off";
  }

  void set_ground_filter_mode(const std::string &ground_filter_mode) {
    ground_filter_mode_ = ground_filter_mode;
  }

  void set_ground_filter_margin(double ground_filter_margin) {
    ground_filter_margin_ = ground_filter_margin;
  }

  void set_resolution(double resolution) { resolution_ = resolution; }

  void set_max_points(int max_points) { max_points_ = max_points; }

  /**
   * @brief Setter function for the filter range.
   * @param min_x Pointer to the minimum x-coordinate value.
   * @param max_x Pointer to the maximum x-coordinate value.
   * @param min_y Pointer to the minimum y-coordinate value.
   * @param max_y Pointer to the maximum y-coordinate value.
   * @param min_z Pointer to the minimum z-coordinate value.
   * @param max_z Pointer to the maximum z-coordinate value.
   */
  void set_filter_range(double min_x, double max_x, double min_y, double max_y,
                        double min_z, double max_z);

  /**
   * @brief Get the debug point cloud.
   * @param all_pc The point cloud containing all points.
   * @param target_pc The target point cloud.
   * @return True if successful, false otherwise.
   */
  void set_cluster_params(double cluster_tolerance, int min_cluster_size,
                          int max_cluster_size) {
    cluster_tolerance_ = cluster_tolerance;
    min_cluster_size_ = min_cluster_size;
    max_cluster_size_ = max_cluster_size;
  }

  void set_cluster_algorithm(const std::string &cluster_algorithm) {
    cluster_algorithm_ = cluster_algorithm;
  }

  void set_dbscan_core_min_points(int dbscan_core_min_points) {
    dbscan_core_min_points_ = dbscan_core_min_points;
  }

  void set_selection_params(double temporal_weight, double min_score,
                            double max_centroid_jump) {
    selection_temporal_weight_ = temporal_weight;
    selection_min_score_ = min_score;
    selection_max_centroid_jump_ = max_centroid_jump;
  }

  void set_target_group_params(bool enabled, int max_aux_clusters,
                               int candidate_limit, int min_match_count,
                               double min_guide_coverage,
                               double min_cluster_precision,
                               double max_centroid_distance) {
    target_group_enabled_ = enabled;
    target_group_max_aux_clusters_ = max_aux_clusters;
    target_group_candidate_limit_ = candidate_limit;
    target_group_min_match_count_ = min_match_count;
    target_group_min_guide_coverage_ = min_guide_coverage;
    target_group_min_cluster_precision_ = min_cluster_precision;
    target_group_max_centroid_distance_ = max_centroid_distance;
  }

  void set_target_refine_params(const std::string &mode, double radius) {
    target_refine_mode_ = mode;
    target_refine_radius_ = radius;
  }

  /**
   * @brief Getter function for the filter range.
   * @return The filter range.
   */
  FilterRange get_filter_range() const;

  /**
   * @brief Get the debug point cloud.
   * @param all_pc The point cloud containing all points.
   * @param target_pc The target point cloud.
   * @return True if successful, false otherwise.
   */
  bool get_pc_debug(pcl::PointCloud<pcl::PointXYZ>::Ptr &all_pc,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr &target_pc);

  /**
   * @brief Get the debug colored point cloud.
   * @param rgb_pc The colored point cloud.
   * @return True if successful, false otherwise.
   */
  bool get_color_pc_debug(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &rgb_pc);

  /**
   * @brief Update the cluster based on the given point cloud.
   * @param cloud The point cloud to update the cluster with.
   */
  void update_cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

  /**
   * @brief Filter the point cloud based on height.
   * @param cloud The input point cloud.
   * @param filtered_cloud The filtered point cloud.
   * @param min_height The minimum height to filter.
   */
  void filter_height(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud,
                     double min_height = 0.0);

  /**
   * @brief Update the target point cloud.
   * @param cloud The point cloud to update the target with.
   */
  void update_target(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

  /**
   * @brief Get the cluster point cloud.
   * @param target_pc The target point cloud.
   * @param env_pc The environment point cloud.
   * @return True if successful, false otherwise.
   */
  bool get_cluster_pointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &target_pc,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr &env_pc,
                              bool incremental = false);

  void filter_height_and_ground(
      pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
      pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud, double min_height);

  bool load_pc_from_file(std::string all_point_path,
                         std::string raw_all_point_path,
                         std::string target_path);

  bool save_pc_to_file(std::string all_point_path,
                       std::string raw_all_point_path, std::string target_path);

 private:
  // Cluster Properties
  double min_height_;           /**< The minimum height of the cluster. */
  double resolution_;           /**< The resolution for downsample points. */
  double cluster_tolerance_;    /**< The tolerance for clustering points. */
  int min_cluster_size_;        /**< The minimum size of a cluster. */
  int max_cluster_size_;        /**< The maximum size of a cluster. */
  std::string ground_filter_mode_ = "off";
  double ground_filter_margin_ = 0.3;
  bool first_guide_ground_reference_ready_ = false;
  double first_guide_ground_reference_z_ = 0.0;
  pcl::PointCloud<pcl::PointXYZ>::Ptr first_guide_reference_cloud_;
  double filter_grid_size_ = 2; /**< The grid size for height filtering. */
  double grid_max_height_ = 5;  /**< The maximum height for height filtering. */
  double grid_height_threshold_ = 1; /**< The threshold for height filtering. */
  int max_points_;                   /**< The maximum number of points. */
  double epsilon_;                   /**< The epsilon value for comparison. */
  int points_sample_num_max; /**< The maximum number of sample points. */

  bool pub_dense = false;
  bool first_cluster = true;

  FilterRange filter_range_; /**< The range for filtering point cloud data. */

  // Point Clouds
  pcl::PointCloud<pcl::PointXYZ>::Ptr
      all_point_cloud_; /**< The point cloud containing all points. */

  pcl::PointCloud<pcl::PointXYZ>::Ptr
      raw_all_point_cloud_; /**< The point cloud containing all points. */

  pcl::PointCloud<pcl::PointXYZ>::Ptr
      raw_point_cloud_; /**< The point cloud containing newest points. */

  pcl::PointCloud<pcl::PointXYZ>::Ptr excluded_points_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr
      target_point_cloud_; /**< The target point cloud. */

  pcl::PointCloud<pcl::PointXYZ>::Ptr
      first_target_point_cloud_; /**< The first target point cloud. */

  pcl::PointCloud<pcl::PointXYZ>::Ptr last_best_cluster_cloud_;

  bool has_last_best_cluster_ = false;

  // Clustering Results
  pcl::PointIndices best_cluster_points_indices; /**< The indices of the best
                                                    cluster points. */
  std::vector<pcl::PointIndices>
      clusters; /**< The vector of cluster indices. */

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> incremental_cluster_clouds_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr incremental_unassigned_cloud_;

  std::vector<int> incremental_point_cluster_labels_;

  // Supporting Structures
  pcl::search::KdTree<pcl::PointXYZ>::Ptr
      kdtree_; /**< The KdTree for point search. */

  PointCloudUtils::Voxelization voxel;

  // Miscellaneous

  /**
   * @brief Filter the point cloud based on the axis range.
   * @param cloud The input point cloud.
   * @param filtered_cloud The filtered point cloud.
   * @param excluded_cloud The point cloud to store the excluded points.
   * @param range The range for filtering.
   */
  void filter_axis_range(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr &excluded_cloud,
                         const FilterRange &range);

  void filter_first_guide_reference(
      pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
      pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud,
      double min_height);

  bool initialize_first_guide_ground_reference(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

  double percentile(std::vector<double> values, double q) const;

  double guide_protect_radius() const;

  /**
   * @brief Scale the points in the point cloud.
   * @param cloud The input point cloud.
   * @param scale_pc The scaled point cloud.
   * @param gt_mean The ground truth mean.
   * @param pc_L_max The maximum length of the point cloud.
   */
  void scale_points(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr &scale_pc,
                    Eigen::Vector3d &gt_mean, double &pc_L_max);

  /**
   * @brief Scale the points to the unit cube.
   * @param cloud The input point cloud.
   */
  void scale_to_unit_cube(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

  /**
   * @brief Maintain the size of the point cloud.
   * @param cloud The input point cloud.
   * @param max_points The maximum number of points.
   * @param leaf_size The leaf size for downsampling.
   */
  void maintain_size(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                     int max_points = 1024, double leaf_size = 0.9);

  /**
   * @brief Maintain the size of the point cloud using voxelization.
   * @param cloud The input point cloud.
   * @param max_points The maximum number of points.
   * @param resolution The resolution for voxelization.
   */
  void maintain_voxel_size(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                           int max_points = 1024, double resolution = 0.9);

  /**
   * @brief Cluster the points based on distance.
   * @param cloud The input point cloud.
   * @param cluster_indices The indices of the clusters.
   * @param cluster_tolerance The tolerance for clustering points.
   * @param min_cluster_size The minimum size of a cluster.
   * @param max_cluster_size The maximum size of a cluster.
   */
  void dis_cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                   std::vector<pcl::PointIndices> &cluster_indices,
                   float cluster_tolerance = 1, int min_cluster_size = 50,
                   int max_cluster_size = 85000);

  /**
   * @brief Incremental distance clustering of points.
   * @param cloud The input point cloud.
   * @param cluster_indices The indices of the clusters.
   * @param tree The KdTree for point search.
   * @param cluster_tolerance The tolerance for clustering points.
   * @param min_cluster_size The minimum size of a cluster.
   * @param max_cluster_size The maximum size of a cluster.
   */
  void incremental_dis_cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                               std::vector<pcl::PointIndices> &cluster_indices,
                               pcl::search::KdTree<pcl::PointXYZ>::Ptr &tree,
                               float cluster_tolerance = 1,
                               int min_cluster_size = 500,
                               int max_cluster_size = 1000000);

  /**
   * @brief DBSCAN clustering of points.
   * @param cloud The input point cloud.
   * @param cluster_indices The indices of the clusters.
   */
  void dbscan_cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                      std::vector<pcl::PointIndices> &cluster_indices);

  void update_incremental_euclidean(
      pcl::PointCloud<pcl::PointXYZ>::Ptr &new_points);

  void update_incremental_dbscan(
      pcl::PointCloud<pcl::PointXYZ>::Ptr &new_points);

  void update_incremental_cluster(
      pcl::PointCloud<pcl::PointXYZ>::Ptr &new_points,
      bool promote_unassigned_with_dbscan);

  void rebuild_incremental_cluster_index();

  void merge_incremental_clusters(
      const std::vector<std::pair<int, int>> &merge_pairs);

  void downsample_incremental_clusters();

  float nearest_distance_squared_to_cluster(
      const pcl::PointXYZ &point,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster_cloud) const;

  /**
   * @brief Compute the bounding box of the point cloud.
   * @param cloud The input point cloud.
   * @param min_pt The minimum point of the bounding box.
   * @param max_pt The maximum point of the bounding box.
   */
  void compute_bounding_box(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                            pcl::PointXYZ &min_pt, pcl::PointXYZ &max_pt);

  void computeBoundingBox(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                          pcl::PointXYZ &min_pt, pcl::PointXYZ &max_pt);

  /**
   * @brief Check if two bounding boxes intersect.
   * @param min_pt1 The minimum point of the first bounding box.
   * @param max_pt1 The maximum point of the first bounding box.
   * @param min_pt2 The minimum point of the second bounding box.
   * @param max_pt2 The maximum point of the second bounding box.
   * @return True if the bounding boxes intersect, false otherwise.
   */
  bool isBoundingBoxIntersected(const pcl::PointXYZ &min_pt1,
                                const pcl::PointXYZ &max_pt1,
                                const pcl::PointXYZ &min_pt2,
                                const pcl::PointXYZ &max_pt2);

  /**
   * @brief Find the most likely cluster based on the target point cloud.
   * @param target_pc The target point cloud.
   * @param cluster_indices The indices of the clusters.
   * @param best_cluster_points_indices The indices of the best cluster points.
   * @param all_point_cloud The point cloud containing all points.
   */
  void find_most_likely_cluster(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &target_pc,
      const std::vector<pcl::PointIndices> &cluster_indices,
      pcl::PointIndices &best_cluster_points_indices,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &all_point_cloud);

  void refine_target_cluster_with_guide(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &target_pc,
      pcl::PointIndices &cluster_indices,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &all_point_cloud) const;

  /**
   * @brief Find the index of a point in the point cloud.
   * @param point The point to find the index of.
   * @param cloud The point cloud to search in.
   * @param epsilon The epsilon value for comparison.
   * @return The index of the point in the point cloud.
   */
  int find_point_index_in_cloud(
      const pcl::PointXYZ &point,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, float epsilon = 0.5);

  float membership_search_radius() const;

 public:
  const std::string &cluster_algorithm() const { return cluster_algorithm_; }

  int dbscan_core_min_points() const { return dbscan_core_min_points_; }

  double last_cluster_runtime_ms() const { return last_cluster_runtime_ms_; }

  std::size_t last_cluster_count() const { return last_cluster_count_; }

  std::size_t last_cluster_input_size() const { return last_cluster_input_size_; }

  std::size_t last_target_guide_size() const { return last_target_guide_size_; }

  std::size_t last_best_cluster_size() const { return last_best_cluster_size_; }

 private:
  std::string cluster_algorithm_ = "euclidean";
  int dbscan_core_min_points_ = 1;
  double last_cluster_runtime_ms_ = 0.0;
  std::size_t last_cluster_count_ = 0;
  std::size_t last_cluster_input_size_ = 0;
  std::size_t last_target_guide_size_ = 0;
  std::size_t last_best_cluster_size_ = 0;
  double selection_temporal_weight_ = 0.25;
  double selection_min_score_ = 0.05;
  double selection_max_centroid_jump_ = 8.0;
  bool target_group_enabled_ = true;
  int target_group_max_aux_clusters_ = 3;
  int target_group_candidate_limit_ = 12;
  int target_group_min_match_count_ = 20;
  double target_group_min_guide_coverage_ = 0.04;
  double target_group_min_cluster_precision_ = 0.08;
  double target_group_max_centroid_distance_ = 20.0;
  std::string target_refine_mode_ = "guide_radius";
  double target_refine_radius_ = 20.0;
};
#endif  // PC_CLUSTER_HPP_
