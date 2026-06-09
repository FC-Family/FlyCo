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

#ifndef PC_PROJECT_HPP_
#define PC_PROJECT_HPP_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

namespace PointCloudProject {

// 将点云从世界坐标系转换到相机坐标系
void transform_pointcloud_to_camera(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const Eigen::Matrix3d &R,
    const Eigen::Vector3d &T,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &camera_coords) {
  camera_coords->resize(cloud->points.size());

  for (size_t i = 0; i < cloud->points.size(); ++i) {
    Eigen::Vector3d world_point(cloud->points[i].x, cloud->points[i].y,
                                cloud->points[i].z);
    Eigen::Vector3d camera_point = R * world_point + T;

    camera_coords->points[i].x = camera_point.x();
    camera_coords->points[i].y = camera_point.y();
    camera_coords->points[i].z = camera_point.z();
  }
}

// 将点云从相机坐标系转换为像素坐标
void project_pointcloud_to_pixels(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &camera_coords,
    const Eigen::Matrix3d &K, std::vector<cv::Point2d> &pixel_coords) {
  pixel_coords.resize(camera_coords->points.size());

  for (size_t i = 0; i < camera_coords->points.size(); ++i) {
    const pcl::PointXYZ &point = camera_coords->points[i];
    if (point.z > 0) {  // 忽略无效点
      Eigen::Vector3d camera_point(point.x, point.y, point.z);
      Eigen::Vector3d pixel_point = K * camera_point;

      // 将齐次坐标转换为像素坐标
      pixel_coords[i] = cv::Point2d(pixel_point.x() / pixel_point.z(),
                                    pixel_point.y() / pixel_point.z());
    } else {
      pixel_coords[i] = cv::Point2d(-1, -1);  // 无效点
    }
  }
}

void filter_pixel_coords(const std::vector<cv::Point2d> &pixel_coords,
                         const cv::Mat &mask,
                         std::vector<cv::Point2d> &filtered_coords) {
  filtered_coords.clear();

  for (const auto &point : pixel_coords) {
    // 检查坐标是否在图像边界内
    if (point.x >= 0 && point.x < mask.cols && point.y >= 0 &&
        point.y < mask.rows) {
      // 检查该点是否在 mask 的有效区域内
      if (mask.at<uchar>(static_cast<int>(point.y),
                         static_cast<int>(point.x)) == 1) {
        filtered_coords.push_back(point);
      }
    }
  }
}

std::vector<cv::Point2d> get_points_in_mask(
    const std::vector<cv::Point2d> &pixel_coords, const cv::Mat &mask) {
  std::vector<cv::Point2d> filtered_coords;

  // 检查 mask 是否有效
  if (mask.empty() || mask.channels() != 1) {
    std::cerr << "Invalid mask image!" << std::endl;
    return filtered_coords;
  }

  // 遍历每个像素坐标并检查是否在 mask 中
  for (const auto &point : pixel_coords) {
    // 检查坐标是否在图像边界内
    if (point.x >= 0 && point.x < mask.cols && point.y >= 0 &&
        point.y < mask.rows) {
      // 检查该点是否在 mask 的有效区域内
      if (mask.at<uchar>(static_cast<int>(point.y),
                         static_cast<int>(point.x)) == 1) {
        filtered_coords.push_back(point);
      }
    }
  }

  return filtered_coords;
}

std::vector<cv::Point3d> convert_pixel_coords_to_world(
    const std::vector<cv::Point2d> &pixel_coords, const cv::Mat &depth_map,
    const cv::Mat &K, const cv::Mat &R, const cv::Mat &T) {
  std::vector<cv::Point3d> world_coords;

  for (const auto &pixel : pixel_coords) {
    // 检查 pixel 是否在深度图范围内
    if (pixel.x < 0 || pixel.x >= depth_map.cols || pixel.y < 0 ||
        pixel.y >= depth_map.rows) {
      continue;
    }

    // 获取深度值（假设深度图是单通道的 CV_32F 图像）
    float depth = depth_map.at<float>(static_cast<int>(pixel.y),
                                      static_cast<int>(pixel.x));
    if (depth <= 0) {
      continue;  // 跳过无效的深度值
    }

    // 将像素坐标转换为相机坐标系下的坐标
    double x_camera =
        (pixel.x - K.at<double>(0, 2)) / K.at<double>(0, 0) * depth;
    double y_camera =
        (pixel.y - K.at<double>(1, 2)) / K.at<double>(1, 1) * depth;
    double z_camera = depth;

    // 将相机坐标系下的坐标转换为世界坐标系
    cv::Mat camera_point =
        (cv::Mat_<double>(3, 1) << x_camera, y_camera, z_camera);
    cv::Mat world_point = R * camera_point + T;

    // 将结果保存到世界坐标向量中
    world_coords.emplace_back(world_point.at<double>(0, 0),
                              world_point.at<double>(1, 0),
                              world_point.at<double>(2, 0));
  }

  return world_coords;
}
}  // namespace PointCloudProject

#endif  // PC_PROJECT_HPP_
