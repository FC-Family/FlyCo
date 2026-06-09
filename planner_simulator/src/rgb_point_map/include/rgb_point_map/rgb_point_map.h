/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Apr. 2025
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the header file of RGBPointMap class, which implements the
 *                   RGB-D mapping of FlyCo.
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

#ifndef _RGB_POINT_MAP_H_
#define _RGB_POINT_MAP_H_

#include "plan_utils/visibility_st.hpp"
#include "quadrotor_msgs/Mesh.h"
#include <ros/ros.h>
#include <ros/package.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Joy.h>
#include <boost/circular_buffer.hpp>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <chrono>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <visualization_msgs/MarkerArray.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/tf.h>

#include <geometry_msgs/PoseStamped.h>
#include <igl/read_triangle_mesh.h>

#define BELIEF_TIMES 2
#define BELIEF_THRESHOLD 7.0

using namespace std;
using std::unique_ptr;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry, geometry_msgs::QuaternionStamped> SyncPolicyImagePoseGimbal;
typedef unique_ptr<message_filters::Synchronizer<SyncPolicyImagePoseGimbal>> SynchronizerImagePoseGimbal;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry> SyncPolicyCloudPose;
typedef unique_ptr<message_filters::Synchronizer<SyncPolicyCloudPose>> SynchronizerCloudPose;

struct VoxelIndex 
{
  int x, y, z;
  bool operator==(const VoxelIndex& other) const 
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct ChunkIndex 
{
  int x, y, z;
  bool operator==(const ChunkIndex& other) const 
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

namespace std 
{
  template <>
  struct hash<VoxelIndex> 
  {
    size_t operator()(const VoxelIndex& idx) const 
    {
      size_t h1 = std::hash<int>()(idx.x);
      size_t h2 = std::hash<int>()(idx.y);
      size_t h3 = std::hash<int>()(idx.z);
      return ((h1 ^ (h2 << 1)) >> 1) ^ (h3 << 1);
    }
  };
  template <>
  struct hash<ChunkIndex> 
  {
    size_t operator()(const ChunkIndex& idx) const 
    {
      size_t h1 = std::hash<int>()(idx.x);
      size_t h2 = std::hash<int>()(idx.y);
      size_t h3 = std::hash<int>()(idx.z);
      return ((h1 ^ (h2 << 1)) >> 1) ^ (h3 << 1);
    }
  };
}

struct VoxelAccumulator 
{
  float x = 0, y = 0, z = 0;
  int r = 0, g = 0, b = 0;
  int count = 0;

  bool has_pt = false;
  vector<Eigen::Vector3d> cam_dir_list;

  bool is_published = false;
  
  void add(const pcl::PointXYZRGB& pt, const Eigen::Vector3d& cam_pos, bool inherit) 
  {
    if (!inherit)
    {
      // * first point
      // if (count == 1) return;

      // * belief accumulation -> enough observations make the voxel valid
      if ((int)cam_dir_list.size() < BELIEF_TIMES)
      {
        Eigen::Vector3d pt_3d(pt.x, pt.y, pt.z);
        Eigen::Vector3d cam_dir = (pt_3d - cam_pos).normalized();

        if (cam_dir_list.empty()) cam_dir_list.push_back(cam_dir);
        else
        {
          bool add = true;
          for (auto vec : cam_dir_list)
          {
            double angle = acos(cam_dir.dot(vec));
            if (angle < BELIEF_THRESHOLD * M_PI / 180.0) add = false;
          }
          if (add) cam_dir_list.push_back(cam_dir);
        }
      }

      // * average points
      x += pt.x;
      y += pt.y;
      z += pt.z;
      r += static_cast<int>(pt.r);
      g += static_cast<int>(pt.g);
      b += static_cast<int>(pt.b);
      count++;
      if ((int)cam_dir_list.size() > BELIEF_TIMES - 1) has_pt = true;
    }
    else
    {
      // * average points
      x += pt.x;
      y += pt.y;
      z += pt.z;
      r += static_cast<int>(pt.r);
      g += static_cast<int>(pt.g);
      b += static_cast<int>(pt.b);
      count++;
      has_pt = true;
    }
  }
  pcl::PointXYZRGB average() const 
  {
    pcl::PointXYZRGB pt;
    if (!has_pt) return pcl::PointXYZRGB();

    pt.x = x / count;
    pt.y = y / count;
    pt.z = z / count;
    pt.r = static_cast<uint8_t>(r / count);
    pt.g = static_cast<uint8_t>(g / count);
    pt.b = static_cast<uint8_t>(b / count);
    return pt;
  }
  void published()
  {
    if(!has_pt) return;

    is_published = true;
    return;
  }
  bool getHasPt()
  {
    return has_pt;
  }
};

struct Chunk 
{
  unordered_map<VoxelIndex, VoxelAccumulator> voxels;
};

namespace flyco
{

class RGBPointMap
{
public:
  RGBPointMap(){
  };
  ~RGBPointMap(){
  };

  /* Func */
  void init(ros::NodeHandle& nh);
  void start();

private:
  /* Func */
  void rgbdCallback(const sensor_msgs::PointCloud2ConstPtr& msg, const nav_msgs::OdometryConstPtr& pose);
  void igoCallback(const sensor_msgs::ImageConstPtr& img, const nav_msgs::OdometryConstPtr& pose, const geometry_msgs::QuaternionStampedConstPtr& quat);
  void odomCallback(const nav_msgs::OdometryConstPtr& pose);
  void mappingCallback(const ros::TimerEvent& e);
  void publishMapCallback(const ros::TimerEvent& e);
  void publishCamCallback(const ros::TimerEvent& e);
  void joyCallback(const sensor_msgs::Joy::ConstPtr& Joy);
  void triggerCallback(const geometry_msgs::PoseStamped::ConstPtr& goal_msg);
  void meshCallback(const quadrotor_msgs::Mesh& mesh_msg);
  void saveMapCallback(const ros::TimerEvent& e);
  void insertPointCloudIncremental(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_in, const Eigen::Vector3d& cam_pos, bool inherit);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr exportVoxelCloud();
  void loadInheritMap();
  void processImg(const sensor_msgs::Image& img);
  void getFOV(const Eigen::Vector3d& pos, const double& pitch, const double& yaw, vector<Eigen::Vector3d>& list1, vector<Eigen::Vector3d>& list2);

  /* Param */
  bool start_mapping_ = false, en_inherit_ = false;
  bool save_map_ = false, already_save_ = false;
  float voxel_size_;
  int chunk_size_;
  float origin_x_, origin_y_, origin_z_;
  float map_size_x_, map_size_y_, map_size_z_;
  int voxel_num_x_, voxel_num_y_, voxel_num_z_;
  double fx_, fy_, cx_, cy_;
  int img_rows_, img_cols_;
  double fov_scale_, fov_horizontal_, fov_vertical_;
  double c2b_x_, c2b_y_, c2b_z_, c2b_roll_, c2b_pitch_, c2b_yaw_;
  Eigen::Matrix3d cam2body_R_matrix_;
  Eigen::Vector3d cam2body_t_vector_;

  /* Data */
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr total_rgb_cloud_ = nullptr, img_pixels_ = nullptr;
  vector<vector<Eigen::Vector3d>> img_3d_table_;
  vector<Eigen::Vector3d> cam_vertices1_, cam_vertices2_;
  boost::circular_buffer<sensor_msgs::PointCloud2> buffer_rgbd_;
  boost::circular_buffer<nav_msgs::Odometry> buffer_rgbd_pose_;
  boost::circular_buffer<sensor_msgs::Image> buffer_img_;
  boost::circular_buffer<nav_msgs::Odometry> buffer_pose_;
  boost::circular_buffer<geometry_msgs::QuaternionStamped> buffer_quat_;
  boost::circular_buffer<nav_msgs::Odometry> buffer_cur_pose_;

  /* Chunk Map */
  unordered_map<ChunkIndex, Chunk> chunk_map_;

  /* Predicted Mesh */
  string gt_address_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_cloud_ = nullptr;
  pcl::KdTreeFLANN<pcl::PointXYZ> mesh_tree_;

  /* ROS Service */
  ros::Publisher map_pub_, img_3d_pub_, fov_pub_;
  ros::Subscriber joy_sub_, odom_sub_, trigger_sub_, pred_mesh_sub_;
  ros::Timer mapping_timer_, publish_map_timer_, publish_cam_timer_, save_timer_;
  unique_ptr<message_filters::Subscriber<sensor_msgs::Image>> image_sub_;
  unique_ptr<message_filters::Subscriber<nav_msgs::Odometry>> pose_sub_;
  unique_ptr<message_filters::Subscriber<geometry_msgs::QuaternionStamped>> gimbal_sub_;
  SynchronizerImagePoseGimbal sync_image_pose_gimbal_;
  unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> rgbd_sub_;
  unique_ptr<message_filters::Subscriber<nav_msgs::Odometry>> rgbd_pose_sub_;
  SynchronizerCloudPose sync_rgbd_pose_;

};

}

#endif