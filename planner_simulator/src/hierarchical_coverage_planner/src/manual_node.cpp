/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Sept. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the main file of the manual mode of FlyCo.
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

#include "hierarchical_coverage_planner/manual_node.h"

namespace flyco
{

void Manual_Node::init(ros::NodeHandle& nh)
{
  nh.param("hcplanner/suspension_rate", this->suspension_target_, 99.0);
  nh.param("viewpoint_manager/GroundPos", this->ground_, -1.0);
  nh.param("hcmap/resolution", this->res_, 0.4);
  nh.param("hcplanner/mesh_sample_size", this->sample_size_, -1);
  nh.param("viewpoint_manager/vis_inf", vis_inflation_, 0.5);

  int num = 10;
  this->buffer_mesh_V_ = boost::circular_buffer<Eigen::MatrixXd>(num);
  this->buffer_mesh_F_ = boost::circular_buffer<Eigen::MatrixXi>(num);
  
  this->percep_utils_.reset(new PerceptionUtils);
  this->percep_utils_->init(nh);

  this->odom_sub_.reset(new message_filters::Subscriber<nav_msgs::Odometry>(nh, "/flyco_planner/body_pose", 10));
  this->gimbal_sub_.reset(new message_filters::Subscriber<geometry_msgs::QuaternionStamped>(nh, "/flyco_planner/gimbal_pose", 10));
  this->sync_odom_gimbal_.reset(new message_filters::Synchronizer<SyncPolicyPoseGimbal>(SyncPolicyPoseGimbal(20), *this->odom_sub_, *this->gimbal_sub_));
  this->sync_odom_gimbal_->registerCallback(boost::bind(&Manual_Node::execCallback, this, _1, _2));

  this->manual_traj_pub_ = nh.advertise<nav_msgs::Path>("/joy_ctrl/task_traj", 1, true);
  this->manual_vp_pub_ = nh.advertise<visualization_msgs::Marker>("/joy_ctrl/viewpoints", 1, true);
  this->manual_mesh_pub_ = nh.advertise<visualization_msgs::Marker>("/joy_ctrl/mesh", 1, true);
  this->manual_uncovered_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/joy_ctrl/uncovered", 1);

  this->manual_vis_timer_ = nh.createTimer(ros::Duration(0.05), &Manual_Node::manualVisCallback, this);
  this->manual_sta_timer_ = nh.createTimer(ros::Duration(0.1), &Manual_Node::statisticsCallback, this);
  this->mesh_sub_ = nh.subscribe("/flyco_planner/predicted_mesh", 1, &Manual_Node::meshCallback, this);
  this->joy_sub_ = nh.subscribe("/joy", 50, &Manual_Node::joyCallback, this);
  
  return;
}

} // namespace flyco