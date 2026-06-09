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

#ifndef PC_UTILS_HPP_
#define PC_UTILS_HPP_

#include <pcl/ModelCoefficients.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

namespace PointCloudUtils {

// inline void segmentGroundPlaneByRANSAC(
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
//     pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud,
//     pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground_cloud,
//     double distance_threshold = 0.1) {
//   // 平面分割对象
//   pcl::SACSegmentation<pcl::PointXYZ> seg;
//   pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
//   pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());

//   // 配置分割参数
//   seg.setOptimizeCoefficients(true);            // 优化模型系数
//   seg.setModelType(pcl::SACMODEL_PLANE);        // 平面模型
//   seg.setMethodType(pcl::SAC_RANSAC);           // RANSAC 算法
//   seg.setDistanceThreshold(distance_threshold); // 距离阈值
//   seg.setInputCloud(cloud);

//   // 分割地面平面
//   seg.segment(*inliers, *coefficients);
//   if (inliers->indices.empty()) {
//     std::cerr << "Could not estimate a planar model for the given dataset."
//               << std::endl;
//     return;
//   }

//   // 提取地面点
//   pcl::ExtractIndices<pcl::PointXYZ> extract;
//   extract.setInputCloud(cloud);
//   extract.setIndices(inliers);
//   extract.setNegative(false); // 提取地面点
//   extract.filter(*ground_cloud);

//   // 提取非地面点
//   extract.setNegative(true); // 提取非地面点
//   extract.filter(*non_ground_cloud);

//   std::cout << "Ground plane has " << ground_cloud->size() << " points."
//             << std::endl;
//   std::cout << "Non-ground has " << non_ground_cloud->size() << " points."
//             << std::endl;
// }

// inline void segmentGroundPlaneByHeight(
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
//     pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud,
//     pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground_cloud,
//     float min_height = -0.1, float max_height = 0.1) {
//   pcl::PassThrough<pcl::PointXYZ> pass;
//   pass.setInputCloud(cloud);
//   pass.setFilterFieldName("z");
//   pass.setFilterLimits(min_height, max_height);

//   // 提取地面点
//   pass.filter(*ground_cloud);

//   // 提取非地面点
//   pass.setNegative(true);
//   pass.filter(*non_ground_cloud);

//   std::cout << "Ground cloud has " << ground_cloud->size() << " points."
//             << std::endl;
//   std::cout << "Non-ground cloud has " << non_ground_cloud->size() << "
//   points."
//             << std::endl;
// }

class Voxelization {
 public:
  Voxelization() {}
  ~Voxelization() {}

  void input(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, double &resolution);
  void getVoxelPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr &ds_points);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  Eigen::Vector3d min_bound_, max_bound_, cloud_size_;
  Eigen::Vector3i voxel_num_, min_box_, max_box_;
  std::vector<char> occupancy_buffer_;

  double resolution_, resolution_inv_;

  void posToIndex(const Eigen::Vector3d &pos, Eigen::Vector3i &id);
  void indexToPos(const Eigen::Vector3i &id, Eigen::Vector3d &pos);
  int toAddress(const int &x, const int &y, const int &z);
  int toAddress(const Eigen::Vector3i &id);
};

inline void Voxelization::posToIndex(const Eigen::Vector3d &pos,
                                     Eigen::Vector3i &id) {
  for (int i = 0; i < 3; ++i)
    id(i) = floor((pos(i) - this->min_bound_(i)) * this->resolution_inv_);
}

inline void Voxelization::indexToPos(const Eigen::Vector3i &id,
                                     Eigen::Vector3d &pos) {
  for (int i = 0; i < 3; ++i)
    pos(i) = (id(i) + 0.5) * this->resolution_ + this->min_bound_(i);
}

inline int Voxelization::toAddress(const int &x, const int &y, const int &z) {
  return x * voxel_num_(1) * voxel_num_(2) + y * voxel_num_(2) + z;
}

inline int Voxelization::toAddress(const Eigen::Vector3i &id) {
  return toAddress(id[0], id[1], id[2]);
}

}  // namespace PointCloudUtils

#endif  // PC_UTILS_HPP_
