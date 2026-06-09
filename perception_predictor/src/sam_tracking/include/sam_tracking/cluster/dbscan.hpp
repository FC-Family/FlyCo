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

#ifndef DBSCAN_H
#define DBSCAN_H

#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#define UN_PROCESSED 0
#define PROCESSING 1
#define PROCESSED 2

inline bool comparePointClusters(const pcl::PointIndices &a,
                                 const pcl::PointIndices &b) {
  return (a.indices.size() < b.indices.size());
}

template <typename PointT>
class DBSCANSimpleCluster {
 public:
  typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
  typedef typename pcl::search::KdTree<PointT>::Ptr KdTreePtr;

  virtual void setInputCloud(PointCloudPtr cloud) { input_cloud_ = cloud; }

  void setSearchMethod(KdTreePtr tree) { search_method_ = tree; }

  void extract(std::vector<pcl::PointIndices> &cluster_indices) {
    cluster_indices.clear();
    if (!input_cloud_ || !search_method_ || input_cloud_->empty()) {
      return;
    }

    std::vector<int> nn_indices;
    std::vector<float> nn_distances;
    const std::size_t point_count = input_cloud_->points.size();
    std::vector<bool> processed(point_count, false);
    std::vector<bool> assigned(point_count, false);

    for (std::size_t i = 0; i < point_count; ++i) {
      if (processed[i]) {
        continue;
      }

      processed[i] = true;
      int nn_size = radiusSearch(static_cast<int>(i), eps_, nn_indices,
                                 nn_distances);
      if (nn_size < minPts_) {
        continue;
      }

      std::vector<int> seed_queue;
      seed_queue.reserve(nn_indices.size());
      seed_queue.push_back(i);
      assigned[i] = true;

      for (int j = 0; j < nn_size; ++j) {
        const int neighbor_index = nn_indices[j];
        if (neighbor_index < 0 ||
            neighbor_index >= static_cast<int>(point_count)) {
          continue;
        }
        if (!assigned[neighbor_index]) {
          seed_queue.push_back(neighbor_index);
          assigned[neighbor_index] = true;
        }
      }

      std::size_t sq_idx = 1;
      while (sq_idx < seed_queue.size()) {
        const int cloud_index = seed_queue[sq_idx++];
        if (cloud_index < 0 || cloud_index >= static_cast<int>(point_count)) {
          continue;
        }
        if (processed[cloud_index]) {
          continue;
        }
        processed[cloud_index] = true;

        nn_size = radiusSearch(cloud_index, eps_, nn_indices, nn_distances);
        if (nn_size >= minPts_) {
          for (int j = 0; j < nn_size; ++j) {
            const int neighbor_index = nn_indices[j];
            if (neighbor_index < 0 ||
                neighbor_index >= static_cast<int>(point_count)) {
              continue;
            }
            if (!assigned[neighbor_index]) {
              seed_queue.push_back(neighbor_index);
              assigned[neighbor_index] = true;
            }
          }
        }
      }

      if (seed_queue.size() >= static_cast<std::size_t>(min_pts_per_cluster_) &&
          seed_queue.size() <= static_cast<std::size_t>(max_pts_per_cluster_)) {
        pcl::PointIndices r;
        r.indices = seed_queue;
        std::sort(r.indices.begin(), r.indices.end());
        r.indices.erase(std::unique(r.indices.begin(), r.indices.end()),
                        r.indices.end());

        if (r.indices.size() >=
                static_cast<std::size_t>(min_pts_per_cluster_) &&
            r.indices.size() <=
                static_cast<std::size_t>(max_pts_per_cluster_)) {
          r.header = input_cloud_->header;
          cluster_indices.push_back(r);
        }
      }
    }

    std::sort(cluster_indices.begin(), cluster_indices.end(),
              [](const pcl::PointIndices &a, const pcl::PointIndices &b) {
                return a.indices.size() > b.indices.size();
              });
  }

  void setClusterTolerance(double tolerance) { eps_ = tolerance; }

  void setMinClusterSize(int min_cluster_size) {
    min_pts_per_cluster_ = min_cluster_size;
  }

  void setMaxClusterSize(int max_cluster_size) {
    max_pts_per_cluster_ = max_cluster_size;
  }

  void setCorePointMinPts(int core_point_min_pts) {
    minPts_ = core_point_min_pts;
  }

 protected:
  PointCloudPtr input_cloud_;
  double eps_{0.0};
  int minPts_{1};  // not including the point itself.
  int min_pts_per_cluster_{1};
  int max_pts_per_cluster_{std::numeric_limits<int>::max()};
  KdTreePtr search_method_;

  virtual int radiusSearch(int index, double radius,
                           std::vector<int> &k_indices,
                           std::vector<float> &k_sqr_distances) const {
    return this->search_method_->radiusSearch(index, radius, k_indices,
                                              k_sqr_distances);
  }
};  // class DBSCANCluster

#endif  // DBSCAN_H
