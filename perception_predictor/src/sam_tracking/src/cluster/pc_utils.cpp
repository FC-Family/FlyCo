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

#include "sam_tracking/cluster/pc_utils.hpp"

namespace PointCloudUtils {

void Voxelization::input(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                         double &resolution) {
  this->resolution_ = resolution;
  this->resolution_inv_ = 1 / resolution;

  pcl::PointXYZ min_pt, max_pt;
  pcl::getMinMax3D(*cloud, min_pt, max_pt);

  double inflation = 2 * resolution;
  this->min_bound_ = Eigen::Vector3d(min_pt.x - inflation, min_pt.y - inflation,
                                     min_pt.z - inflation);
  this->max_bound_ = Eigen::Vector3d(max_pt.x + inflation, max_pt.y + inflation,
                                     max_pt.z + inflation);
  this->cloud_size_ = this->max_bound_ - this->min_bound_;
  for (int i = 0; i < 3; ++i)
    this->voxel_num_(i) = ceil(this->cloud_size_(i) / resolution);
  this->posToIndex(this->min_bound_, this->min_box_);
  this->posToIndex(this->max_bound_, this->max_box_);

  int buffer_size =
      this->voxel_num_(0) * this->voxel_num_(1) * this->voxel_num_(2);
  this->occupancy_buffer_ = std::vector<char>(buffer_size, 0);

  for (auto pt : cloud->points) {
    Eigen::Vector3d pt_w;
    Eigen::Vector3i id;
    int adr;
    pt_w << pt.x, pt.y, pt.z;
    posToIndex(pt_w, id);
    adr = toAddress(id);
    this->occupancy_buffer_[adr] = 1;
  }

  return;
}

void Voxelization::getVoxelPoints(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &ds_points) {
  for (int x = this->min_box_(0); x < this->max_box_(0); ++x)
    for (int y = this->min_box_(1); y < this->max_box_(1); ++y)
      for (int z = this->min_box_(2); z < this->max_box_(2); ++z) {
        Eigen::Vector3i id(x, y, z);
        if (this->occupancy_buffer_[toAddress(id)] == 1) {
          Eigen::Vector3d pos;
          indexToPos(id, pos);
          pcl::PointXYZ pt;
          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = pos(2);
          ds_points->points.push_back(pt);
        }
      }

  return;
}

}  // namespace PointCloudUtils
