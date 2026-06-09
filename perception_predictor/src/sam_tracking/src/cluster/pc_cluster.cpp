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

#include "sam_tracking/cluster/pc_cluster.hpp"

#include <ros/ros.h>

#include <algorithm>
#include <limits>
#include <sstream>

#include "sam_tracking/cluster/pc_utils.hpp"

namespace {

std::string zStatsSummary(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  if (cloud == nullptr) {
    return "none";
  }
  if (cloud->empty()) {
    return "empty";
  }

  std::vector<float> z_values;
  z_values.reserve(cloud->size());
  for (const auto &point : cloud->points) {
    if (std::isfinite(point.z)) {
      z_values.push_back(point.z);
    }
  }
  if (z_values.empty()) {
    return "no_finite_z";
  }

  auto percentile = [&z_values](double q) {
    q = std::min(1.0, std::max(0.0, q));
    std::vector<float> sorted = z_values;
    std::sort(sorted.begin(), sorted.end());
    const double pos = q * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(pos));
    const auto upper = static_cast<std::size_t>(std::ceil(pos));
    if (lower == upper) {
      return static_cast<double>(sorted[lower]);
    }
    const double ratio = pos - static_cast<double>(lower);
    return static_cast<double>(sorted[lower]) * (1.0 - ratio) +
           static_cast<double>(sorted[upper]) * ratio;
  };

  const auto minmax =
      std::minmax_element(z_values.begin(), z_values.end());
  const auto min_it = minmax.first;
  const auto max_it = minmax.second;
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(3);
  oss << "n=" << z_values.size() << " z[min/p1/p5/med/p95/max]="
      << *min_it << "/" << percentile(0.01) << "/" << percentile(0.05) << "/"
      << percentile(0.50) << "/" << percentile(0.95) << "/" << *max_it;
  return oss.str();
}

}  // namespace

PointCloudCluster::PointCloudCluster(double min_height, double resolution,
                                     double cluster_tolerance,
                                     int min_cluster_size, int max_cluster_size,
                                     int max_points, double epsilon,
                                     double filter_grid_size,
                                     double grid_max_height,
                                     double grid_height_threshold)
    : min_height_(min_height),
      resolution_(resolution),
      cluster_tolerance_(cluster_tolerance),
      min_cluster_size_(min_cluster_size),
      max_cluster_size_(max_cluster_size),
      filter_grid_size_(filter_grid_size),
      grid_max_height_(grid_max_height),
      grid_height_threshold_(grid_height_threshold),
      max_points_(max_points),
      epsilon_(epsilon),
      points_sample_num_max(0),       // 明确初始化
      best_cluster_points_indices(),  // 默认构造
      clusters()                      // 默认构造
{
  all_point_cloud_ =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  raw_all_point_cloud_ =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  raw_point_cloud_ =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  target_point_cloud_ =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

  first_target_point_cloud_ =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  first_guide_reference_cloud_ =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

  excluded_points_ =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

  last_best_cluster_cloud_ =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

  incremental_unassigned_cloud_ =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

  kdtree_ = pcl::search::KdTree<pcl::PointXYZ>::Ptr(
      new pcl::search::KdTree<pcl::PointXYZ>());

  filter_range_ = FilterRange();
}

bool PointCloudCluster::get_pc_debug(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &all_pc,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &target_pc) {
  all_pc = all_point_cloud_;
  target_pc = target_point_cloud_;
  return true;
}

void PointCloudCluster::set_filter_range(double min_x, double max_x,
                                         double min_y, double max_y,
                                         double min_z, double max_z) {
  filter_range_ = {min_x, max_x, min_y, max_y, min_z, max_z};
}

void PointCloudCluster::filter_axis_range(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &excluded_cloud,  // 用于存储被排除的点
    const FilterRange &range) {
  if (cloud->empty()) {
    std::cerr << "Input cloud is empty!" << std::endl;
    return;
  }

  for (const auto &point : cloud->points) {
    // 判断点是否在指定范围内
    bool in_x_range = (point.x <= range.min_x || point.x >= range.max_x);
    bool in_y_range = (point.y <= range.min_y || point.y >= range.max_y);
    bool in_z_range = (point.z <= range.min_z || point.z >= range.max_z);

    // 如果点在指定范围内，则加入 excluded_cloud，否则加入 filtered_cloud
    if (in_x_range || in_y_range || in_z_range) {
      excluded_cloud->points.push_back(point);  // 存储被排除的点
    } else {
      filtered_cloud->points.push_back(point);  // 保留不在范围内的点
    }
  }

  // 更新点云属性
  filtered_cloud->width = filtered_cloud->points.size();
  filtered_cloud->height = 1;
  filtered_cloud->is_dense = true;

  excluded_cloud->width = excluded_cloud->points.size();
  excluded_cloud->height = 1;
  excluded_cloud->is_dense = true;
}

bool PointCloudCluster::get_color_pc_debug(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &rgb_pc) {
  std::mt19937 rng(std::random_device{}());         // 随机数生成器
  std::uniform_int_distribution<int> dist(0, 255);  // 颜色范围

  if (clusters.empty()) {
    return false;
  }

  for (size_t cluster_idx = 0; cluster_idx < clusters.size(); ++cluster_idx) {
    uint8_t r = dist(rng);  // 随机颜色
    uint8_t g = dist(rng);
    uint8_t b = dist(rng);

    for (int point_idx : clusters[cluster_idx].indices) {
      pcl::PointXYZRGB point;
      point.x = all_point_cloud_->points[point_idx].x;
      point.y = all_point_cloud_->points[point_idx].y;
      point.z = all_point_cloud_->points[point_idx].z;

      // 分配颜色
      point.r = r;
      point.g = g;
      point.b = b;

      rgb_pc->points.push_back(point);
    }
  }

  rgb_pc->width = rgb_pc->points.size();
  rgb_pc->height = 1;  // 单层点云
  rgb_pc->is_dense = true;
  return true;
}

bool PointCloudCluster::save_pc_to_file(std::string all_point_path,
                                        std::string raw_all_point_path,
                                        std::string target_path) {
  // 保存 all_point_cloud_ 到 all_point_path
  bool if_save = true;
  if (pcl::io::savePCDFileBinary(all_point_path, *all_point_cloud_) == -1) {
    std::cerr << "Failed to save all point cloud to " << all_point_path
              << std::endl;
    if_save = false;
  } else {
    std::cout << "Saved all point cloud to " << all_point_path << " with "
              << all_point_cloud_->size() << " points." << std::endl;
  }

  // 保存 raw_all_point_cloud_ 到 raw_all_point_path
  if (pcl::io::savePCDFileBinary(raw_all_point_path, *raw_all_point_cloud_) ==
      -1) {
    std::cerr << "Failed to save raw all point cloud to " << raw_all_point_path
              << std::endl;
    if_save = false;
  } else {
    std::cout << "Saved raw all point cloud to " << raw_all_point_path
              << " with " << raw_all_point_cloud_->size() << " points."
              << std::endl;
  }

  // 保存 target_point_cloud_ 到 target_path
  if (pcl::io::savePCDFileBinary(target_path, *target_point_cloud_) == -1) {
    std::cerr << "Failed to save target point cloud to " << target_path
              << std::endl;
    if_save = false;
  } else {
    std::cout << "Saved target point cloud to " << target_path << " with "
              << target_point_cloud_->size() << " points." << std::endl;
  }
  return if_save;
}

bool PointCloudCluster::load_pc_from_file(std::string all_point_path,
                                          std::string raw_all_point_path,
                                          std::string target_path) {
  // 创建用于临时存储点云的对象

  bool if_load = true;
  pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);

  // 加载 all_point_path 中的点云
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(all_point_path, *temp_cloud) == -1) {
    std::cerr << "Failed to load all point cloud from " << all_point_path
              << std::endl;
    if_load = false;
  } else {
    std::cout << "Loaded all point cloud from " << all_point_path << " with "
              << temp_cloud->size() << " points." << std::endl;
    all_point_cloud_->points = temp_cloud->points;  // 更新全局点云
  }

  // 加载 raw_all_point_path 中的点云
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(raw_all_point_path, *temp_cloud) ==
      -1) {
    std::cerr << "Failed to load raw all point cloud from "
              << raw_all_point_path << std::endl;
    if_load = false;
  } else {
    std::cout << "Loaded raw all point cloud from " << raw_all_point_path
              << " with " << temp_cloud->size() << " points." << std::endl;
    raw_all_point_cloud_->points = temp_cloud->points;  // 更新原始全局点云
  }

  // 加载 target_path 中的点云
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(target_path, *temp_cloud) == -1) {
    std::cerr << "Failed to load target point cloud from " << target_path
              << std::endl;
    if_load = false;
  } else {
    std::cout << "Loaded target point cloud from " << target_path << " with "
              << temp_cloud->size() << " points." << std::endl;
    target_point_cloud_->points = temp_cloud->points;  // 更新目标点云
  }

  // 确保点云大小符合设定，维护点云大小
  maintain_size(all_point_cloud_, max_points_, resolution_);
  maintain_size(raw_all_point_cloud_, max_points_, resolution_);
  maintain_size(target_point_cloud_, max_points_, resolution_);

  // 更新 KdTree，用于后续的聚类
  kdtree_->setInputCloud(all_point_cloud_);

  std::cout << "Point clouds loaded and initialized successfully." << std::endl;
  return if_load;
}

void PointCloudCluster::update_cluster(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  auto start_time = std::chrono::high_resolution_clock::now();

  // 创建点云指针用于存储中间结果
  pcl::PointCloud<pcl::PointXYZ>::Ptr height_filtered_pc(
      new pcl::PointCloud<pcl::PointXYZ>);  // 按高度过滤后的点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_pc(
      new pcl::PointCloud<pcl::PointXYZ>);  // 参与聚类的点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr excluded_pc(
      new pcl::PointCloud<pcl::PointXYZ>);  // 被排除的点云

  if (ground_filter_mode_ == "grid") {
    filter_height_and_ground(cloud, height_filtered_pc, min_height_);
  } else if (ground_filter_mode_ == "first_guide_reference") {
    filter_first_guide_reference(cloud, height_filtered_pc, min_height_);
  } else {
    filter_height(cloud, height_filtered_pc, min_height_);
  }

  raw_all_point_cloud_->points.insert(raw_all_point_cloud_->points.end(),
                                      cloud->points.begin(),
                                      cloud->points.end());

  raw_point_cloud_->points = cloud->points;

  // 按指定轴范围过滤
  filter_axis_range(height_filtered_pc, filtered_pc, excluded_pc,
                    filter_range_);

  // 保存被排除的点云到成员变量中，供后续使用
  excluded_points_ = excluded_pc;

  maintain_size(raw_all_point_cloud_, max_points_, 0.70);
  maintain_size(raw_point_cloud_, max_points_, 0.50);
  maintain_size(raw_point_cloud_, max_points_, 0.50);

  // 聚类
  if (cluster_algorithm_ == "euclidean") {
    all_point_cloud_->points.insert(all_point_cloud_->points.end(),
                                    filtered_pc->points.begin(),
                                    filtered_pc->points.end());
    maintain_size(all_point_cloud_, max_points_, resolution_);
    kdtree_->setInputCloud(all_point_cloud_);  // 重建 KdTree
    dis_cluster(all_point_cloud_, clusters, cluster_tolerance_,
                min_cluster_size_, max_cluster_size_);
  } else if (cluster_algorithm_ == "dbscan") {
    all_point_cloud_->points.insert(all_point_cloud_->points.end(),
                                    filtered_pc->points.begin(),
                                    filtered_pc->points.end());
    maintain_size(all_point_cloud_, max_points_, resolution_);
    kdtree_->setInputCloud(all_point_cloud_);  // 重建 KdTree
    dbscan_cluster(all_point_cloud_, clusters);
  } else if (cluster_algorithm_ == "incremental_euclidean") {
    update_incremental_euclidean(filtered_pc);
  } else if (cluster_algorithm_ == "incremental_dbscan") {
    update_incremental_dbscan(filtered_pc);
  } else {
    ROS_WARN_STREAM_THROTTLE(
        2.0, "Unknown cluster_algorithm='" << cluster_algorithm_
                                           << "', fallback to euclidean.");
    all_point_cloud_->points.insert(all_point_cloud_->points.end(),
                                    filtered_pc->points.begin(),
                                    filtered_pc->points.end());
    maintain_size(all_point_cloud_, max_points_, resolution_);
    kdtree_->setInputCloud(all_point_cloud_);  // 重建 KdTree
    dis_cluster(all_point_cloud_, clusters, cluster_tolerance_,
                min_cluster_size_, max_cluster_size_);
  }

  auto end_time1 = std::chrono::high_resolution_clock::now();

  // 如果存在目标点云，找到最可能的聚类
  if (!clusters.empty() && !target_point_cloud_->empty()) {
    find_most_likely_cluster(target_point_cloud_, clusters,
                             best_cluster_points_indices, all_point_cloud_);
  } else {
    best_cluster_points_indices.indices.clear();
  }

  auto end_time2 = std::chrono::high_resolution_clock::now();

  last_cluster_input_size_ = all_point_cloud_->size();
  last_cluster_count_ = clusters.size();
  last_target_guide_size_ = target_point_cloud_->size();
  last_best_cluster_size_ = best_cluster_points_indices.indices.size();
  last_cluster_runtime_ms_ =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          end_time2 - start_time)
          .count();

  (void)start_time;
  (void)end_time1;
  (void)end_time2;
}

float PointCloudCluster::nearest_distance_squared_to_cluster(
    const pcl::PointXYZ &point,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster_cloud) const {
  if (cluster_cloud == nullptr || cluster_cloud->empty()) {
    return std::numeric_limits<float>::max();
  }

  float best_distance_squared = std::numeric_limits<float>::max();
  for (const auto &candidate : cluster_cloud->points) {
    const float dx = point.x - candidate.x;
    const float dy = point.y - candidate.y;
    const float dz = point.z - candidate.z;
    const float distance_squared = dx * dx + dy * dy + dz * dz;
    if (distance_squared < best_distance_squared) {
      best_distance_squared = distance_squared;
    }
  }
  return best_distance_squared;
}

void PointCloudCluster::merge_incremental_clusters(
    const std::vector<std::pair<int, int>> &merge_pairs) {
  if (merge_pairs.empty() || incremental_cluster_clouds_.size() < 2) {
    return;
  }

  std::vector<int> parent(incremental_cluster_clouds_.size());
  for (std::size_t i = 0; i < parent.size(); ++i) {
    parent[i] = static_cast<int>(i);
  }

  const auto find_root = [&parent](int index) {
    int root = index;
    while (parent[root] != root) {
      root = parent[root];
    }
    while (parent[index] != index) {
      const int next = parent[index];
      parent[index] = root;
      index = next;
    }
    return root;
  };

  for (const auto &pair : merge_pairs) {
    if (pair.first < 0 || pair.second < 0 ||
        pair.first >= static_cast<int>(parent.size()) ||
        pair.second >= static_cast<int>(parent.size())) {
      continue;
    }
    const int root_a = find_root(pair.first);
    const int root_b = find_root(pair.second);
    if (root_a != root_b) {
      parent[root_b] = root_a;
    }
  }

  std::unordered_map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> merged_map;
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> merged_clusters;
  for (std::size_t i = 0; i < incremental_cluster_clouds_.size(); ++i) {
    const int root = find_root(static_cast<int>(i));
    if (merged_map.find(root) == merged_map.end()) {
      merged_map[root] =
          pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
      merged_clusters.push_back(merged_map[root]);
    }
    merged_map[root]->points.insert(merged_map[root]->points.end(),
                                    incremental_cluster_clouds_[i]->points.begin(),
                                    incremental_cluster_clouds_[i]->points.end());
  }
  incremental_cluster_clouds_ = merged_clusters;
}

void PointCloudCluster::downsample_incremental_clusters() {
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> compact_clusters;
  compact_clusters.reserve(incremental_cluster_clouds_.size());
  for (auto &cluster_cloud : incremental_cluster_clouds_) {
    if (cluster_cloud == nullptr || cluster_cloud->empty()) {
      continue;
    }
    maintain_size(cluster_cloud, max_points_, resolution_);
    compact_clusters.push_back(cluster_cloud);
  }
  incremental_cluster_clouds_ = compact_clusters;
  maintain_size(incremental_unassigned_cloud_, max_points_ * 2, resolution_);
}

void PointCloudCluster::rebuild_incremental_cluster_index() {
  all_point_cloud_->clear();
  clusters.clear();
  best_cluster_points_indices.indices.clear();
  incremental_point_cluster_labels_.clear();

  int point_offset = 0;
  for (std::size_t cluster_id = 0; cluster_id < incremental_cluster_clouds_.size();
       ++cluster_id) {
    const auto &cluster_cloud = incremental_cluster_clouds_[cluster_id];
    if (cluster_cloud == nullptr || cluster_cloud->empty()) {
      continue;
    }
    pcl::PointIndices cluster_indices;
    cluster_indices.indices.reserve(cluster_cloud->size());
    for (const auto &point : cluster_cloud->points) {
      all_point_cloud_->points.push_back(point);
      cluster_indices.indices.push_back(point_offset++);
      incremental_point_cluster_labels_.push_back(static_cast<int>(cluster_id));
    }
    if (!cluster_indices.indices.empty()) {
      clusters.push_back(cluster_indices);
    }
  }

  all_point_cloud_->width = all_point_cloud_->points.size();
  all_point_cloud_->height = 1;
  all_point_cloud_->is_dense = true;

  if (!all_point_cloud_->empty()) {
    kdtree_->setInputCloud(all_point_cloud_);
  }
}

void PointCloudCluster::update_incremental_euclidean(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &new_points) {
  update_incremental_cluster(new_points, false);
}

void PointCloudCluster::update_incremental_dbscan(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &new_points) {
  update_incremental_cluster(new_points, true);
}

void PointCloudCluster::update_incremental_cluster(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &new_points,
    bool promote_unassigned_with_dbscan) {
  if (new_points == nullptr) {
    return;
  }

  const float assignment_radius_squared =
      cluster_tolerance_ * cluster_tolerance_;
  std::vector<std::pair<int, int>> merge_pairs;
  const bool has_index = !all_point_cloud_->empty() &&
                         incremental_point_cluster_labels_.size() ==
                             all_point_cloud_->size();

  for (const auto &point : new_points->points) {
    int best_cluster_index = -1;
    float best_distance_squared = std::numeric_limits<float>::max();

    if (has_index) {
      std::vector<int> point_indices;
      std::vector<float> point_distances;
      if (kdtree_->radiusSearch(point, cluster_tolerance_, point_indices,
                                point_distances) > 0) {
        std::unordered_set<int> matching_cluster_set;
        for (std::size_t match_index = 0; match_index < point_indices.size();
             ++match_index) {
          const int point_index = point_indices[match_index];
          if (point_index < 0 ||
              point_index >=
                  static_cast<int>(incremental_point_cluster_labels_.size())) {
            continue;
          }
          const int cluster_index =
              incremental_point_cluster_labels_[point_index];
          if (cluster_index < 0 ||
              cluster_index >=
                  static_cast<int>(incremental_cluster_clouds_.size())) {
            continue;
          }
          matching_cluster_set.insert(cluster_index);
          if (point_distances[match_index] < best_distance_squared) {
            best_distance_squared = point_distances[match_index];
            best_cluster_index = cluster_index;
          }
        }

        for (const int cluster_index : matching_cluster_set) {
          if (cluster_index != best_cluster_index) {
            merge_pairs.emplace_back(best_cluster_index, cluster_index);
          }
        }
      }
    }

    if (best_cluster_index == -1 ||
        best_distance_squared > assignment_radius_squared) {
      incremental_unassigned_cloud_->points.push_back(point);
      continue;
    }

    incremental_cluster_clouds_[best_cluster_index]->points.push_back(point);
  }

  merge_incremental_clusters(merge_pairs);

  std::vector<pcl::PointIndices> promoted_cluster_indices;
  if (!incremental_unassigned_cloud_->empty()) {
    if (promote_unassigned_with_dbscan) {
      dbscan_cluster(incremental_unassigned_cloud_, promoted_cluster_indices);
    } else {
      dis_cluster(incremental_unassigned_cloud_, promoted_cluster_indices,
                  cluster_tolerance_, min_cluster_size_, max_cluster_size_);
    }
  }

  std::vector<bool> promoted_mask(incremental_unassigned_cloud_->size(), false);
  for (const auto &indices : promoted_cluster_indices) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr new_cluster(
        new pcl::PointCloud<pcl::PointXYZ>);
    new_cluster->points.reserve(indices.indices.size());
    for (const int point_index : indices.indices) {
      if (point_index < 0 ||
          point_index >= static_cast<int>(incremental_unassigned_cloud_->size())) {
        continue;
      }
      promoted_mask[point_index] = true;
      new_cluster->points.push_back(
          incremental_unassigned_cloud_->points[point_index]);
    }
    if (!new_cluster->empty()) {
      incremental_cluster_clouds_.push_back(new_cluster);
    }
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_unassigned(
      new pcl::PointCloud<pcl::PointXYZ>);
  remaining_unassigned->points.reserve(incremental_unassigned_cloud_->size());
  for (std::size_t point_index = 0;
       point_index < incremental_unassigned_cloud_->size(); ++point_index) {
    if (!promoted_mask[point_index]) {
      remaining_unassigned->points.push_back(
          incremental_unassigned_cloud_->points[point_index]);
    }
  }
  incremental_unassigned_cloud_ = remaining_unassigned;

  downsample_incremental_clusters();
  rebuild_incremental_cluster_index();
}

void PointCloudCluster::filter_height(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud, double min_height) {
  if (cloud->empty()) {
    std::cerr << "Input cloud is empty!" << std::endl;
    return;
  }

  for (const auto &point : cloud->points) {
    if (point.z >= min_height) {
      filtered_cloud->points.push_back(point);
    }
  }
}

double PointCloudCluster::percentile(std::vector<double> values,
                                     double q) const {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  q = std::min(1.0, std::max(0.0, q));
  std::sort(values.begin(), values.end());
  const double pos = q * static_cast<double>(values.size() - 1);
  const auto lower = static_cast<std::size_t>(std::floor(pos));
  const auto upper = static_cast<std::size_t>(std::ceil(pos));
  if (lower == upper) {
    return values[lower];
  }
  const double ratio = pos - static_cast<double>(lower);
  return values[lower] * (1.0 - ratio) + values[upper] * ratio;
}

double PointCloudCluster::guide_protect_radius() const {
  return std::max(1.5, std::max(resolution_ * 3.0, cluster_tolerance_));
}

bool PointCloudCluster::initialize_first_guide_ground_reference(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  if (first_guide_ground_reference_ready_) {
    return true;
  }
  if (cloud == nullptr || cloud->empty() ||
      first_target_point_cloud_ == nullptr ||
      first_target_point_cloud_->size() < 20) {
    return false;
  }

  std::vector<double> guide_x;
  std::vector<double> guide_y;
  std::vector<double> guide_z;
  guide_x.reserve(first_target_point_cloud_->size());
  guide_y.reserve(first_target_point_cloud_->size());
  guide_z.reserve(first_target_point_cloud_->size());
  for (const auto &point : first_target_point_cloud_->points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        !std::isfinite(point.z)) {
      continue;
    }
    guide_x.push_back(point.x);
    guide_y.push_back(point.y);
    guide_z.push_back(point.z);
  }
  if (guide_z.size() < 20) {
    return false;
  }

  const double x_low = percentile(guide_x, 0.05);
  const double x_high = percentile(guide_x, 0.95);
  const double y_low = percentile(guide_y, 0.05);
  const double y_high = percentile(guide_y, 0.95);
  const double z_low = percentile(guide_z, 0.05);
  const double z_high = percentile(guide_z, 0.95);
  if (!std::isfinite(x_low) || !std::isfinite(x_high) ||
      !std::isfinite(y_low) || !std::isfinite(y_high) ||
      !std::isfinite(z_low) || !std::isfinite(z_high) || x_low >= x_high ||
      y_low >= y_high || z_low > z_high) {
    return false;
  }

  first_guide_reference_cloud_->clear();
  for (const auto &point : first_target_point_cloud_->points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        !std::isfinite(point.z)) {
      continue;
    }
    if (point.x < x_low || point.x > x_high || point.y < y_low ||
        point.y > y_high || point.z < z_low || point.z > z_high) {
      continue;
    }
    first_guide_reference_cloud_->points.push_back(point);
  }
  if (first_guide_reference_cloud_->size() < 10) {
    first_guide_reference_cloud_->points = first_target_point_cloud_->points;
  }
  first_guide_reference_cloud_->width = first_guide_reference_cloud_->size();
  first_guide_reference_cloud_->height = 1;
  first_guide_reference_cloud_->is_dense = true;

  pcl::search::KdTree<pcl::PointXYZ> guide_tree;
  guide_tree.setInputCloud(first_guide_reference_cloud_);
  const double expand = std::max(cluster_tolerance_ * 2.0, resolution_ * 4.0);
  const double protect_radius = guide_protect_radius();
  const double protect_radius_sq = protect_radius * protect_radius;
  std::vector<double> candidate_z;
  candidate_z.reserve(cloud->size());

  for (const auto &point : cloud->points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        !std::isfinite(point.z) || point.z < min_height_) {
      continue;
    }
    if (point.x < x_low - expand || point.x > x_high + expand ||
        point.y < y_low - expand || point.y > y_high + expand) {
      continue;
    }

    std::vector<int> indices;
    std::vector<float> distances;
    if (guide_tree.nearestKSearch(point, 1, indices, distances) > 0 &&
        !distances.empty() && distances[0] <= protect_radius_sq) {
      continue;
    }
    candidate_z.push_back(point.z);
  }

  if (candidate_z.size() < 50) {
    ROS_WARN_STREAM_THROTTLE(
        5.0, "first_guide_reference ground filter has insufficient "
                 "non-guide ground candidates; falling back to height filter");
    return false;
  }

  first_guide_ground_reference_z_ = percentile(candidate_z, 0.10);
  if (!std::isfinite(first_guide_ground_reference_z_)) {
    return false;
  }
  first_guide_ground_reference_ready_ = true;
  ROS_INFO_STREAM("Initialized first-guide ground reference: z="
                  << first_guide_ground_reference_z_
                  << " candidates=" << candidate_z.size()
                  << " guide_points=" << first_guide_reference_cloud_->size());
  return true;
}

void PointCloudCluster::filter_first_guide_reference(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud, double min_height) {
  if (!initialize_first_guide_ground_reference(cloud)) {
    filter_height(cloud, filtered_cloud, min_height);
    return;
  }

  pcl::search::KdTree<pcl::PointXYZ> guide_tree;
  guide_tree.setInputCloud(first_guide_reference_cloud_);
  const double protect_radius = guide_protect_radius();
  const double protect_radius_sq = protect_radius * protect_radius;
  const double ground_threshold =
      first_guide_ground_reference_z_ + std::max(0.0, ground_filter_margin_);

  for (const auto &point : cloud->points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        !std::isfinite(point.z) || point.z < min_height) {
      continue;
    }

    bool guide_protected = false;
    std::vector<int> indices;
    std::vector<float> distances;
    if (guide_tree.nearestKSearch(point, 1, indices, distances) > 0 &&
        !distances.empty() && distances[0] <= protect_radius_sq) {
      guide_protected = true;
    }

    if (point.z <= ground_threshold && !guide_protected) {
      continue;
    }
    filtered_cloud->points.push_back(point);
  }
  filtered_cloud->width = filtered_cloud->points.size();
  filtered_cloud->height = 1;
  filtered_cloud->is_dense = true;
}

void PointCloudCluster::filter_height_and_ground(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud, double min_height) {
  if (cloud->empty()) {
    std::cerr << "Input cloud is empty!" << std::endl;
    return;
  }

  // 创建网格存储
  std::unordered_map<GridKey, std::vector<pcl::PointXYZ>, GridKeyHash> grid_map;

  // 遍历点云，将点分配到对应的网格中
  for (const auto &point : cloud->points) {
    int grid_x = static_cast<int>(std::floor(point.x / filter_grid_size_));
    int grid_y = static_cast<int>(std::floor(point.y / filter_grid_size_));
    GridKey grid_key(grid_x, grid_y);

    grid_map[grid_key].push_back(point);
  }

  // 遍历每个网格，找到最低点并过滤非地面点
  for (const auto &grid_entry : grid_map) {
    const auto &points = grid_entry.second;

    if (points.empty()) {
      continue;
    }

    // 找到网格中的最低点
    auto min_z_point =
        *std::min_element(points.begin(), points.end(),
                          [](const pcl::PointXYZ &a, const pcl::PointXYZ &b) {
                            return a.z < b.z;
                          });

    // 遍历网格中的点，剔除地面点，将非地面点加入到输出点云
    for (const auto &point : points) {
      if ((point.z > min_z_point.z + grid_height_threshold_ ||
           point.z > grid_max_height_) &&
          (point.z >= min_height)) {
        filtered_cloud->points.push_back(point);
      }
    }
  }
}

void PointCloudCluster::update_target(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr filter_pc(
      new pcl::PointCloud<pcl::PointXYZ>);
  // The target/guide cloud is already produced by the 2D mask projection path.
  // Re-running grid ground filtering here can erase sparse but valid guide
  // points, so only keep the coarse height gate for cluster association.
  filter_height(cloud, filter_pc, min_height_);
  ROS_INFO_STREAM_THROTTLE(
      1.0, "cluster guide input stats: raw{" << zStatsSummary(cloud)
                                             << "} filtered{"
                                             << zStatsSummary(filter_pc)
                                             << "} min_height=" << min_height_);

  // 将过滤后的点云添加到目标点云中
  target_point_cloud_->points.insert(target_point_cloud_->points.end(),
                                     filter_pc->points.begin(),
                                     filter_pc->points.end());

  if (first_target_point_cloud_->empty() && !filter_pc->empty()) {
    first_target_point_cloud_->points = filter_pc->points;
    maintain_size(first_target_point_cloud_, max_points_, resolution_);
  }

  // 维护目标点云的大小
  maintain_size(target_point_cloud_, max_points_, resolution_);

  // 如果 first_target_point_cloud_ 为空，则用当前 target_point_cloud_ 初始化它
  // 假设 first_target_point_cloud_ 是用来保存第一次的目标点云
  // if (first_target_point_cloud_->empty()) { // 修正: 变量名, empty()函数,
  // if条件括号
  //   // 将 target_point_cloud_ 的所有点复制到 first_target_point_cloud_
  //   first_target_point_cloud_->points.insert(first_target_point_cloud_->points.end(),
  //   // 修正: 目标容器的迭代器
  //                                    target_point_cloud_->points.begin(),
  //                                    target_point_cloud_->points.end());
  // }

  // std::cout << "target size " << target_point_cloud_->size() << std::endl;
}

// bool PointCloudCluster::get_cluster_pointcloud(
//     pcl::PointCloud<pcl::PointXYZ>::Ptr &target_pc,
//     pcl::PointCloud<pcl::PointXYZ>::Ptr &env_pc) {
//   // 检查输入和状态
//   if (clusters.empty() || all_point_cloud_->empty() ||
//       best_cluster_points_indices.indices.empty()) {
//     std::cout << "cluster error " << std::endl;
//     std::cout << "cluster size " << clusters.size() << std::endl;
//     return false;
//   }

//   // 使用 unordered_set 存储所有最佳聚类的点索引
//   std::unordered_set<int> best_cluster_indices(
//       best_cluster_points_indices.indices.begin(),
//       best_cluster_points_indices.indices.end());

//   // 遍历所有点，根据是否在最佳聚类中区分
//   for (size_t i = 0; i < all_point_cloud_->points.size(); ++i) {
//     if (best_cluster_indices.count(i)) {
//       target_pc->points.push_back(all_point_cloud_->points[i]);
//     } else {
//       env_pc->points.push_back(all_point_cloud_->points[i]);
//     }
//   }

//   return true;
// }

bool PointCloudCluster::get_cluster_pointcloud(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &target_pc,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &env_pc, bool incremental) {
  // 记录开始时间
  auto start_time = std::chrono::high_resolution_clock::now();
  pcl::PointCloud<pcl::PointXYZ>::Ptr temp_pc_point_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

  if (incremental) {
    if (first_cluster == true) {
      temp_pc_point_cloud->points = raw_all_point_cloud_->points;
    } else {
      temp_pc_point_cloud->points = raw_point_cloud_->points;
    }

  } else {
    temp_pc_point_cloud->points = raw_all_point_cloud_->points;
  }

  // 检查聚类结果和输入点云状态
  if (clusters.empty() || temp_pc_point_cloud->empty() ||
      best_cluster_points_indices.indices.empty()) {
    // std::cout << "Cluster error: no clusters or invalid input point clouds."
    //           << std::endl;
    // std::cout << "Cluster size: " << clusters.size() << std::endl;
    // std::cout<< " pc size "<< temp_pc_point_cloud->points.size()<<std::endl;
    // std::cout<< "best_cluster_points_indices
    // "<<best_cluster_points_indices.indices.empty()<<std::endl;

    // 如果没有聚类结果，将被排除的点直接划归 `env`
    if (!excluded_points_->empty()) {
      env_pc->points.insert(env_pc->points.end(),
                            excluded_points_->points.begin(),
                            excluded_points_->points.end());
    }
    return false;
  }

  // target_pc->points.insert(target_pc->points.end(),
  //                          target_point_cloud_->points.begin(),
  //                          target_point_cloud_->points.end());

  // 使用 unordered_set 存储最佳聚类的点索引（加速查找）
  std::unordered_set<int> best_cluster_indices(
      best_cluster_points_indices.indices.begin(),
      best_cluster_points_indices.indices.end());
  const float membership_radius = membership_search_radius();

  // 遍历未降采样点云，根据是否属于最佳聚类划分到 target_pc 或 env_pc
  for (size_t i = 0; i < temp_pc_point_cloud->points.size(); ++i) {
    // 在未降采样点云中，找到对应的点索引
    int index_in_cloud =
        find_point_index_in_cloud(temp_pc_point_cloud->points[i],
                                  all_point_cloud_, membership_radius);

    if (index_in_cloud != -1 && best_cluster_indices.count(index_in_cloud)) {
      target_pc->points.push_back(temp_pc_point_cloud->points[i]);
    } else {
      env_pc->points.push_back(temp_pc_point_cloud->points[i]);
    }
  }

  // 将被排除的点（如过滤掉 y 轴范围内的点）划归 `env`
  if (!excluded_points_->empty()) {
    env_pc->points.insert(env_pc->points.end(),
                          excluded_points_->points.begin(),
                          excluded_points_->points.end());
  }

  (void)start_time;

  first_cluster = false;

  return true;
}

void PointCloudCluster::scale_points(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &scale_pc, Eigen::Vector3d &gt_mean,
    double &pc_L_max) {
  // 获取点云的维度
  int num_points = cloud->points.size();
  scale_pc->resize(num_points);

  // 将点云数据转换为 Eigen::MatrixXd
  Eigen::MatrixXd points_matrix(num_points, 3);
  for (int i = 0; i < num_points; ++i) {
    points_matrix(i, 0) = cloud->points[i].x;
    points_matrix(i, 1) = cloud->points[i].y;
    points_matrix(i, 2) = cloud->points[i].z;
  }

  // 计算中心点
  gt_mean = (points_matrix.colwise().maxCoeff() +
             points_matrix.colwise().minCoeff()) /
            2.0;

  // 中心化点云
  Eigen::MatrixXd centered_pc = points_matrix.rowwise() - gt_mean.transpose();

  // 计算最大距离
  pc_L_max = centered_pc.rowwise().norm().maxCoeff();

  // 缩放点云
  Eigen::MatrixXd resized_pc = centered_pc / pc_L_max;

  // 将缩放后的点云数据转换回 PCL 点云
  for (int i = 0; i < num_points; ++i) {
    scale_pc->points[i].x = resized_pc(i, 0);
    scale_pc->points[i].y = resized_pc(i, 1);
    scale_pc->points[i].z = resized_pc(i, 2);
  }
}

void PointCloudCluster::scale_to_unit_cube(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  if (cloud->empty()) {
    std::cerr << "Input cloud is empty!" << std::endl;
    return;
  }

  // 计算每个维度的最小值和最大值
  float min_x = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::min();
  float min_y = std::numeric_limits<float>::max();
  float max_y = std::numeric_limits<float>::min();
  float min_z = std::numeric_limits<float>::max();
  float max_z = std::numeric_limits<float>::min();

  if (max_x == min_x || max_y == min_y || max_z == min_z) {
    std::cerr << "Cannot scale cloud: zero range in one or more dimensions!"
              << std::endl;
    return;
  }

  for (const auto &point : cloud->points) {
    if (point.x < min_x) min_x = point.x;
    if (point.x > max_x) max_x = point.x;
    if (point.y < min_y) min_y = point.y;
    if (point.y > max_y) max_y = point.y;
    if (point.z < min_z) min_z = point.z;
    if (point.z > max_z) max_z = point.z;
  }

  // 缩放点云到 0-1 之间
  for (auto &point : cloud->points) {
    point.x = (point.x - min_x) / (max_x - min_x);
    point.y = (point.y - min_y) / (max_y - min_y);
    point.z = (point.z - min_z) / (max_z - min_z);
  }
}

void PointCloudCluster::maintain_size(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, int max_points,
    double leaf_size) {
  // 维护点云的大小，如果超过 max_points, 则进行下采样。
  if (cloud->size() > static_cast<std::size_t>(max_points)) {
    // 下采样
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(leaf_size, leaf_size, leaf_size);
    sor.filter(*cloud);
  }
}

void PointCloudCluster::maintain_voxel_size(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, int max_points,
    double resolution) {
  // 维护点云的大小，如果超过 max_points, 则进行下采样。
  //   if (cloud->size() > max_points) {
  // 下采样
  PointCloudUtils::Voxelization voxel;
  voxel.input(cloud, resolution);
  pcl::PointCloud<pcl::PointXYZ>::Ptr ds_points(
      new pcl::PointCloud<pcl::PointXYZ>);
  voxel.getVoxelPoints(ds_points);
  *cloud = *ds_points;
  //   }
}

// void PointCloudCluster::maintain_size_voxel_centers(
//     pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, int max_points,
//     double resolution) {
//   if (cloud == nullptr || cloud->empty()) {
//     return;
//   }

//   if (cloud->size() > max_points) {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(
//         new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_voxel_centers(
//         new pcl::PointCloud<pcl::PointXYZ>);

//     pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_;
//     voxel_grid_.setInputCloud(cloud);
//     voxel_grid_.setLeafSize(resolution, resolution, resolution);

//     voxel_grid_.filter(*cloud_downsampled);

//     for (const auto &point : cloud_downsampled->points) {
//       Eigen::Vector3f min_bound, max_bound;
//       Eigen::Vector3i *voxel_index = new Eigen::Vector3i; // 修改这里

//       voxel_grid_.getGridCoordinates(point, voxel_index);

//       min_bound[0] = (*voxel_index)[0] * resolution +
//                      voxel_grid_.getMinBoxCoordinates()[0];
//       min_bound[1] = (*voxel_index)[1] * resolution +
//                      voxel_grid_.getMinBoxCoordinates()[1];
//       min_bound[2] = (*voxel_index)[2] * resolution +
//                      voxel_grid_.getMinBoxCoordinates()[2];

//       max_bound[0] = ((*voxel_index)[0] + 1) * resolution +
//                      voxel_grid_.getMinBoxCoordinates()[0];
//       max_bound[1] = ((*voxel_index)[1] + 1) * resolution +
//                      voxel_grid_.getMinBoxCoordinates()[1];
//       max_bound[2] = ((*voxel_index)[2] + 1) * resolution +
//                      voxel_grid_.getMinBoxCoordinates()[2];

//       pcl::PointXYZ center_point;
//       center_point.x = (min_bound[0] + max_bound[0]) / 2.0f;
//       center_point.y = (min_bound[1] + max_bound[1]) / 2.0f;
//       center_point.z = (min_bound[2] + max_bound[2]) / 2.0f;

//       cloud_voxel_centers->points.push_back(center_point);
//       delete voxel_index; // 释放内存
//     }
//     cloud_voxel_centers->width = cloud_voxel_centers->points.size();
//     cloud_voxel_centers->height = 1;

//     cloud->swap(*cloud_voxel_centers);
//   }
// }

void PointCloudCluster::dis_cluster(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    std::vector<pcl::PointIndices> &cluster_indices, float cluster_tolerance,
    int min_cluster_size, int max_cluster_size) {
  // 检查输入点云是否为空
  if (cloud->empty()) {
    std::cerr << "Input cloud is empty!" << std::endl;
    return;
  }
  cluster_indices.clear();

  // 创建 KdTree
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
      new pcl::search::KdTree<pcl::PointXYZ>());
  tree->setInputCloud(cloud);

  // 创建 EuclideanClusterExtraction 对象
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(cluster_tolerance);  // 设置距离阈值（欧几里得距离）
  ec.setMinClusterSize(min_cluster_size);  // 设置最小聚类点数（minPts）
  ec.setMaxClusterSize(max_cluster_size);  // 可选，设置最大聚类点数
  ec.setSearchMethod(tree);                // 设置搜索方法
  ec.setInputCloud(cloud);                 // 设置输入点云

  // 执行聚类
  ec.extract(cluster_indices);
}

void PointCloudCluster::incremental_dis_cluster(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    std::vector<pcl::PointIndices> &cluster_indices,
    pcl::search::KdTree<pcl::PointXYZ>::Ptr &tree, float cluster_tolerance,
    int min_cluster_size, int max_cluster_size) {
  if (cloud->empty()) {
    std::cerr << "Input cloud is empty!" << std::endl;
    return;
  }

  // 检查是否已经有聚类结果，如果没有则进行初始化
  (void)cluster_indices;

  // if (!tree) {
  //   tree.reset(new pcl::search::KdTree<pcl::PointXYZ>());
  //   tree->setInputCloud(cloud);
  // }

  std::vector<bool> is_processed(cloud->size(), false);
  for (const auto &cluster : cluster_indices) {
    for (const auto &idx : cluster.indices) {
      is_processed[idx] = true;
    }
  }

  std::vector<pcl::PointIndices> new_cluster_indices;
  for (size_t i = 0; i < cloud->size(); ++i) {
    if (is_processed[i]) continue;

    std::vector<int> point_idx_radius_search;
    std::vector<float> point_radius_squared_distance;
    if (tree->radiusSearch(cloud->points[i], cluster_tolerance,
                           point_idx_radius_search,
                           point_radius_squared_distance) > 0) {
      pcl::PointIndices cluster;
      for (int idx : point_idx_radius_search) {
        if (idx >= 0 && idx < static_cast<int>(cloud->size()) &&
            !is_processed[idx]) {
          cluster.indices.push_back(idx);
          is_processed[idx] = true;
        }
      }

      if (cluster.indices.size() >= static_cast<size_t>(min_cluster_size) &&
          cluster.indices.size() <= static_cast<size_t>(max_cluster_size)) {
        new_cluster_indices.push_back(cluster);
      }
    }
  }

  if (!new_cluster_indices.empty()) {
    cluster_indices.insert(cluster_indices.end(), new_cluster_indices.begin(),
                           new_cluster_indices.end());
  }
}

void PointCloudCluster::dbscan_cluster(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    std::vector<pcl::PointIndices> &cluster_indices) {
  // 检查输入点云是否为空
  if (cloud->empty()) {
    std::cerr << "Input cloud is empty!" << std::endl;
    return;
  }

  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
      new pcl::search::KdTree<pcl::PointXYZ>());
  tree->setInputCloud(cloud);

  DBSCANSimpleCluster<pcl::PointXYZ> ec;
  ec.setCorePointMinPts(dbscan_core_min_points_);
  ec.setClusterTolerance(cluster_tolerance_);
  ec.setMinClusterSize(min_cluster_size_);
  ec.setMaxClusterSize(max_cluster_size_);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);

  // 检查结果
  if (cluster_indices.empty()) {
    return;
  }
}

void PointCloudCluster::computeBoundingBox(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointXYZ &min_pt,
    pcl::PointXYZ &max_pt) {
  min_pt.x = min_pt.y = min_pt.z = std::numeric_limits<float>::max();
  max_pt.x = max_pt.y = max_pt.z = std::numeric_limits<float>::lowest();
  for (const auto &point : cloud->points) {
    min_pt.x = std::min(min_pt.x, point.x);
    min_pt.y = std::min(min_pt.y, point.y);
    min_pt.z = std::min(min_pt.z, point.z);
    max_pt.x = std::max(max_pt.x, point.x);
    max_pt.y = std::max(max_pt.y, point.y);
    max_pt.z = std::max(max_pt.z, point.z);
  }
}

// 判断两个边界框是否相交
bool PointCloudCluster::isBoundingBoxIntersected(const pcl::PointXYZ &min_pt1,
                                                 const pcl::PointXYZ &max_pt1,
                                                 const pcl::PointXYZ &min_pt2,
                                                 const pcl::PointXYZ &max_pt2) {
  return (min_pt1.x <= max_pt2.x && max_pt1.x >= min_pt2.x) &&
         (min_pt1.y <= max_pt2.y && max_pt1.y >= min_pt2.y) &&
         (min_pt1.z <= max_pt2.z && max_pt1.z >= min_pt2.z);
}

void PointCloudCluster::refine_target_cluster_with_guide(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &target_pc,
    pcl::PointIndices &cluster_indices,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &all_point_cloud) const {
  if (target_refine_mode_ == "off" || target_pc == nullptr ||
      target_pc->empty() || all_point_cloud == nullptr ||
      all_point_cloud->empty() || cluster_indices.indices.empty()) {
    return;
  }
  if (target_refine_mode_ != "guide_radius") {
    ROS_WARN_STREAM_THROTTLE(2.0, "Unknown target_refine_mode='"
                                      << target_refine_mode_
                                      << "', skipping target refinement");
    return;
  }

  std::vector<double> xs;
  std::vector<double> ys;
  std::vector<double> zs;
  xs.reserve(target_pc->size());
  ys.reserve(target_pc->size());
  zs.reserve(target_pc->size());
  for (const auto &point : target_pc->points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        !std::isfinite(point.z)) {
      continue;
    }
    xs.push_back(point.x);
    ys.push_back(point.y);
    zs.push_back(point.z);
  }
  if (zs.size() < 10) {
    return;
  }

  const double x_low = percentile(xs, 0.02);
  const double x_high = percentile(xs, 0.98);
  const double y_low = percentile(ys, 0.02);
  const double y_high = percentile(ys, 0.98);
  const double z_low = percentile(zs, 0.02);
  const double z_high = percentile(zs, 0.98);

  pcl::PointCloud<pcl::PointXYZ>::Ptr guide_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  guide_cloud->points.reserve(target_pc->size());
  for (const auto &point : target_pc->points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        !std::isfinite(point.z)) {
      continue;
    }
    if (point.x < x_low || point.x > x_high || point.y < y_low ||
        point.y > y_high || point.z < z_low || point.z > z_high) {
      continue;
    }
    guide_cloud->points.push_back(point);
  }
  if (guide_cloud->size() < 10) {
    guide_cloud->points = target_pc->points;
  }
  guide_cloud->width = guide_cloud->size();
  guide_cloud->height = 1;
  guide_cloud->is_dense = true;

  pcl::search::KdTree<pcl::PointXYZ> guide_tree;
  guide_tree.setInputCloud(guide_cloud);

  const double radius =
      target_refine_radius_ > 0.0
          ? target_refine_radius_
          : std::max(1.0, resolution_ * 2.5);
  const double radius_sq = radius * radius;
  pcl::PointIndices refined_indices;
  refined_indices.indices.reserve(cluster_indices.indices.size());

  for (int point_index : cluster_indices.indices) {
    if (point_index < 0 ||
        point_index >= static_cast<int>(all_point_cloud->size())) {
      continue;
    }
    std::vector<int> guide_indices;
    std::vector<float> guide_distances;
    if (guide_tree.nearestKSearch(all_point_cloud->points[point_index], 1,
                                  guide_indices, guide_distances) > 0 &&
        !guide_distances.empty() && guide_distances[0] <= radius_sq) {
      refined_indices.indices.push_back(point_index);
    }
  }

  if (refined_indices.indices.size() >=
      static_cast<std::size_t>(std::max(10, min_cluster_size_ / 10))) {
    ROS_INFO_STREAM_THROTTLE(
        1.0, "Guide-refined target cluster: before="
                 << cluster_indices.indices.size()
                 << " after=" << refined_indices.indices.size()
                 << " radius=" << radius
                 << " guide_points=" << guide_cloud->size());
    cluster_indices.indices.swap(refined_indices.indices);
  } else {
    ROS_WARN_STREAM_THROTTLE(
        2.0, "Guide refinement rejected: before="
                 << cluster_indices.indices.size()
                 << " after=" << refined_indices.indices.size()
                 << " radius=" << radius
                 << ". Keeping original cluster.");
  }
}

void PointCloudCluster::find_most_likely_cluster(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &target_pc,
    const std::vector<pcl::PointIndices> &cluster_indices,
    pcl::PointIndices &best_cluster_points_indices,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &all_point_cloud) {
  // 检查输入是否为空
  if (target_pc->empty() || cluster_indices.empty() ||
      all_point_cloud->empty()) {
    std::cerr << "Input clouds or cluster indices are empty!" << std::endl;
    return;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  // 计算目标点云的包围盒
  Eigen::Vector4f target_min, target_max;
  pcl::getMinMax3D(*target_pc, target_min, target_max);

  double best_score = 0.0;
  int best_cluster_index = -1;  // 最可能的类的索引
  pcl::PointCloud<pcl::PointXYZ>::Ptr best_cluster_cloud(
      new pcl::PointCloud<pcl::PointXYZ>());

  std::vector<pcl::PointIndices> temp_cluster_indices;
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> temp_cluster_clouds;

  const auto build_cluster_cloud = [&all_point_cloud](
                                       const pcl::PointIndices &cluster) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_pc(
        new pcl::PointCloud<pcl::PointXYZ>());
    cluster_pc->points.reserve(cluster.indices.size());
    for (int idx : cluster.indices) {
      if (idx >= 0 && idx < static_cast<int>(all_point_cloud->size())) {
        cluster_pc->points.push_back(all_point_cloud->points[idx]);
      }
    }
    return cluster_pc;
  };

  const auto centroid = [](const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
    Eigen::Vector3d center(0.0, 0.0, 0.0);
    if (cloud == nullptr || cloud->empty()) {
      return center;
    }
    for (const auto &point : cloud->points) {
      center.x() += point.x;
      center.y() += point.y;
      center.z() += point.z;
    }
    center /= static_cast<double>(cloud->size());
    return center;
  };

  // 遍历每个聚类
  for (size_t cluster_idx = 0; cluster_idx < cluster_indices.size();
       ++cluster_idx) {
    const auto &cluster = cluster_indices[cluster_idx];

    // 计算当前聚类的包围盒
    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_pc =
        build_cluster_cloud(cluster);
    if (cluster_pc->empty()) {
      continue;
    }

    Eigen::Vector4f cluster_min, cluster_max;
    pcl::getMinMax3D(*cluster_pc, cluster_min, cluster_max);

    // 判断包围盒是否有重叠，没有重叠则跳过
    if (target_min.x() > cluster_max.x() || target_max.x() < cluster_min.x() ||
        target_min.y() > cluster_max.y() || target_max.y() < cluster_min.y() ||
        target_min.z() > cluster_max.z() || target_max.z() < cluster_min.z()) {
      continue;
    }
    temp_cluster_indices.push_back(cluster);
    temp_cluster_clouds.push_back(cluster_pc);
  }

  if (temp_cluster_indices.empty() && has_last_best_cluster_) {
    const Eigen::Vector3d last_center = centroid(last_best_cluster_cloud_);
    for (const auto &cluster : cluster_indices) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_pc =
          build_cluster_cloud(cluster);
      if (cluster_pc->empty()) {
        continue;
      }
      const double center_distance = (centroid(cluster_pc) - last_center).norm();
      if (center_distance <= selection_max_centroid_jump_) {
        temp_cluster_indices.push_back(cluster);
        temp_cluster_clouds.push_back(cluster_pc);
      }
    }
  }

  // sort size
  std::vector<std::size_t> sorted_indices(temp_cluster_indices.size());
  for (std::size_t i = 0; i < sorted_indices.size(); ++i) {
    sorted_indices[i] = i;
  }
  std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&temp_cluster_indices](std::size_t a, std::size_t b) {
              return temp_cluster_indices[a].indices.size() >
                     temp_cluster_indices[b].indices.size();
            });

  std::vector<pcl::PointIndices> sorted_cluster_indices;
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> sorted_cluster_clouds;
  for (std::size_t i : sorted_indices) {
    sorted_cluster_indices.push_back(temp_cluster_indices[i]);
    sorted_cluster_clouds.push_back(temp_cluster_clouds[i]);
  }
  temp_cluster_indices = sorted_cluster_indices;
  temp_cluster_clouds = sorted_cluster_clouds;

  // only keep the 5 biggest cluster
  const std::size_t candidate_limit =
      target_group_enabled_
          ? static_cast<std::size_t>(
                std::max(5, target_group_candidate_limit_))
          : 5;
  if (temp_cluster_indices.size() > candidate_limit) {
    temp_cluster_indices.resize(candidate_limit);
    temp_cluster_clouds.resize(candidate_limit);
  }

  (void)start_time;
  if (temp_cluster_indices.empty()) {
    best_cluster_points_indices.indices.clear();
    return;
  }

  std::vector<int> candidate_match_counts(temp_cluster_indices.size(), 0);
  std::vector<double> candidate_guide_coverages(temp_cluster_indices.size(),
                                                0.0);
  std::vector<double> candidate_cluster_precisions(temp_cluster_indices.size(),
                                                   0.0);

  for (size_t cluster_idx = 0; cluster_idx < temp_cluster_indices.size();
       ++cluster_idx) {
    const auto &cluster = temp_cluster_indices[cluster_idx];

    // 创建一个哈希表加速查找
    std::unordered_set<int> cluster_set(cluster.indices.begin(),
                                        cluster.indices.end());

    // 统计目标点云中点与当前聚类的匹配数量
    int match_count = 0;
    for (const auto &point : target_pc->points) {
      int point_index = find_point_index_in_cloud(point, all_point_cloud,
                                                  membership_search_radius());

      if (point_index != -1 &&
          cluster_set.find(point_index) != cluster_set.end()) {
        ++match_count;
      }
    }

    if (match_count == 0) {
      continue;
    }

    const double guide_coverage =
        static_cast<double>(match_count) / target_pc->points.size();
    const double cluster_precision =
        static_cast<double>(match_count) / cluster.indices.size();
    candidate_match_counts[cluster_idx] = match_count;
    candidate_guide_coverages[cluster_idx] = guide_coverage;
    candidate_cluster_precisions[cluster_idx] = cluster_precision;
    const double guide_score = guide_coverage * 0.7 + cluster_precision * 0.3;

    double temporal_score = 0.0;
    if (has_last_best_cluster_ && last_best_cluster_cloud_ != nullptr &&
        !last_best_cluster_cloud_->empty()) {
      const double center_distance =
          (centroid(temp_cluster_clouds[cluster_idx]) -
           centroid(last_best_cluster_cloud_))
              .norm();
      temporal_score =
          std::max(0.0, 1.0 - center_distance / selection_max_centroid_jump_);
    }

    const double temporal_weight =
        has_last_best_cluster_
            ? std::min(std::max(selection_temporal_weight_, 0.0), 0.9)
            : 0.0;
    const double score =
        guide_score * (1.0 - temporal_weight) + temporal_score * temporal_weight;

    // Prefer clusters that explain the guide points without selecting a large
    // background component that merely overlaps the guide bbox. When a
    // previous stable cluster exists, prefer spatially consistent candidates.
    if (score > best_score) {
      best_score = score;
      best_cluster_index = static_cast<int>(cluster_idx);
      best_cluster_cloud = temp_cluster_clouds[cluster_idx];
    }
  }

  // 设置最可能的聚类
  if (best_cluster_index != -1 && best_score >= selection_min_score_) {
    best_cluster_points_indices = temp_cluster_indices[best_cluster_index];

    if (target_group_enabled_ && target_group_max_aux_clusters_ > 0) {
      std::unordered_set<int> merged_indices(
          best_cluster_points_indices.indices.begin(),
          best_cluster_points_indices.indices.end());
      const Eigen::Vector3d best_center = centroid(best_cluster_cloud);
      int aux_count = 0;

      for (std::size_t cluster_idx = 0;
           cluster_idx < temp_cluster_indices.size(); ++cluster_idx) {
        if (static_cast<int>(cluster_idx) == best_cluster_index) {
          continue;
        }
        if (aux_count >= target_group_max_aux_clusters_) {
          break;
        }
        if (candidate_match_counts[cluster_idx] <
            target_group_min_match_count_) {
          continue;
        }
        const bool guide_supported =
            candidate_guide_coverages[cluster_idx] >=
                target_group_min_guide_coverage_ ||
            candidate_cluster_precisions[cluster_idx] >=
                target_group_min_cluster_precision_;
        if (!guide_supported) {
          continue;
        }
        const double center_distance =
            (centroid(temp_cluster_clouds[cluster_idx]) - best_center).norm();
        if (center_distance > target_group_max_centroid_distance_) {
          continue;
        }

        for (int idx : temp_cluster_indices[cluster_idx].indices) {
          merged_indices.insert(idx);
        }
        ++aux_count;
      }

      best_cluster_points_indices.indices.assign(merged_indices.begin(),
                                                merged_indices.end());
      std::sort(best_cluster_points_indices.indices.begin(),
                best_cluster_points_indices.indices.end());
    }

    refine_target_cluster_with_guide(target_pc, best_cluster_points_indices,
                                     all_point_cloud);
    best_cluster_cloud = build_cluster_cloud(best_cluster_points_indices);
    *last_best_cluster_cloud_ = *best_cluster_cloud;
    has_last_best_cluster_ = true;
  } else {
    best_cluster_points_indices.indices.clear();
  }
}

// 辅助函数：查找某点在点云中的索引
int PointCloudCluster::find_point_index_in_cloud(
    const pcl::PointXYZ &point,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, float epsilon) {
  std::vector<int> point_indices;
  std::vector<float> point_distances;

  // 搜索最近点
  if (kdtree_->nearestKSearch(point, 1, point_indices, point_distances) > 0) {
    if (point_distances[0] < epsilon * epsilon) {
      return point_indices[0];
    }
  }
  return -1;  // 未找到
}

float PointCloudCluster::membership_search_radius() const {
  const double conservative_radius = std::max(0.05, resolution_ * 0.75);
  return static_cast<float>(std::min(epsilon_, conservative_radius));
}
