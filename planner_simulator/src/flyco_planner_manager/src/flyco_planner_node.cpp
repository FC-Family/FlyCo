/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Sept. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the execution file of mapping and planning part in FlyCo.
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

#include "flyco_planner_manager/flyco_planner_manager.h"
#include "flyco_planner_manager/logger.hpp"
#include "backward.hpp"
#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <boost/circular_buffer.hpp>
namespace backward
{
  backward::SignalHandling sh;
}

using namespace std;
using std::unique_ptr;
using namespace flyco;

/* Odom */
typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, geometry_msgs::QuaternionStamped> SyncPolicyPoseGimbal;
typedef unique_ptr<message_filters::Synchronizer<SyncPolicyPoseGimbal>> SynchronizerPoseGimbal;

unique_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub_ = nullptr;
unique_ptr<message_filters::Subscriber<geometry_msgs::QuaternionStamped>> gimbal_sub_ = nullptr;
SynchronizerPoseGimbal sync_odom_gimbal_;

/* Buffer */
boost::circular_buffer<nav_msgs::OdometryConstPtr> buffer_odom_;
boost::circular_buffer<geometry_msgs::QuaternionStampedConstPtr> buffer_quat_;

/* Param */
bool receive_odom_ = true, start_task_ = false, sim_flag_ = true;
double ext_x_ = 0.0, ext_y_ = 0.0, ext_z_ = 0.0;
Eigen::Vector3d cam2body_t_vector_;
double odom_fps_ = 1.0, signal_task_fps_ = 1.0;
Eigen::VectorXd start_pose_(5);

/* Indicator */
int indi_num_ = -1;
vector<Eigen::VectorXd> indicators_;

/* ROS Service */
unique_ptr<ros::Rate> signal_task_rate_ = nullptr;
ros::Publisher indicator_pub_;

/* Joy Trigger */
bool last_lb_pressed_ = false;
const int JOY_LB_BUTTON = 4;

/* Utils */
void indicatorCallback(const ros::TimerEvent& e)
{
  if (start_task_ == true) return;

  double scale = 1.0;

  visualization_msgs::MarkerArray mk_array;

  for (int i=0; i<indi_num_; ++i)
  {
    Eigen::VectorXd indi_pt = indicators_[i];

    Eigen::Matrix3d Rwb_y, Rwb_p;
    Rwb_y << cos(indi_pt(4)), -sin(indi_pt(4)), 0,
            sin(indi_pt(4)), cos(indi_pt(4)), 0,
            0, 0, 1;
    Rwb_p << cos(indi_pt(3)), 0.0, -sin(indi_pt(3)), 
            0, 1, 0,
            sin(indi_pt(3)), 0.0, cos(indi_pt(3));
    
    Eigen::Vector3d position(indi_pt(0), indi_pt(1), indi_pt(2));
    Eigen::Vector3d orientation = position + Rwb_y*Rwb_p*Eigen::Vector3d(scale, 0.0, 0.0);
    
    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.id = i;
    mk.ns = "selection_indicator";
    mk.type = visualization_msgs::Marker::ARROW;
    mk.pose.orientation.w = 1.0;
    mk.scale.x = 0.2;
    mk.scale.y = 0.4;
    mk.scale.z = 0.3;
    mk.color.r = 1.0;
    mk.color.g = 0.0;
    mk.color.b = 0.0;
    mk.color.a = 1.0;

    geometry_msgs::Point pt_;
    pt_.x = position(0);
    pt_.y = position(1);
    pt_.z = position(2);
    mk.points.push_back(pt_);

    pt_.x = orientation(0);
    pt_.y = orientation(1);
    pt_.z = orientation(2);
    mk.points.push_back(pt_);

    mk_array.markers.push_back(mk);

    visualization_msgs::Marker text_marker;
    text_marker.header.frame_id = "world";
    text_marker.header.stamp = ros::Time::now();
    text_marker.id = i;
    text_marker.ns = "selection_indicator_text";
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.text = to_string(i);
    text_marker.pose.position.x = position(0);
    text_marker.pose.position.y = position(1);
    text_marker.pose.position.z = position(2);
    text_marker.pose.orientation.w = 1.0;
    text_marker.scale.x = scale;
    text_marker.scale.y = scale;
    text_marker.scale.z = scale;
    text_marker.color.r = 0.0;
    text_marker.color.g = 0.0;
    text_marker.color.b = 0.0;
    text_marker.color.a = 1.0;

    mk_array.markers.push_back(text_marker);
  }
  
  indicator_pub_.publish(mk_array);

  return;
}

void odomCallback(const nav_msgs::OdometryConstPtr& pose, const geometry_msgs::QuaternionStampedConstPtr& quat)
{
  if (receive_odom_ == true)
  {
    buffer_odom_.push_back(pose);
    buffer_quat_.push_back(quat);
  }

  return;
}

void updatePoseCallback(const ros::TimerEvent& e)
{
  if (!receive_odom_) return;
  if (buffer_odom_.empty() || buffer_quat_.empty()) return;

  nav_msgs::OdometryConstPtr pose = buffer_odom_.back();
  geometry_msgs::QuaternionStampedConstPtr quat = buffer_quat_.back();

  // * Process Odometry
  tf::Quaternion tfQuat;
  tf::quaternionMsgToTF(quat->quaternion, tfQuat);
  tf::Matrix3x3 m(tfQuat);
  double roll_gimbal, pitch_gimbal, yaw_gimbal;
  m.getRPY(roll_gimbal, pitch_gimbal, yaw_gimbal);

  Eigen::Vector3d body2world_t_vector;
  Eigen::Matrix3d body2world_R_matrix;
  Eigen::Quaterniond body2world_R_quat;

  body2world_t_vector(0) = pose->pose.pose.position.x; // in world coordinate
  body2world_t_vector(1) = pose->pose.pose.position.y; // in world coordinate
  body2world_t_vector(2) = pose->pose.pose.position.z; // in world coordinate

  tf::Quaternion tfQuat_body;
  tf::quaternionMsgToTF(pose->pose.pose.orientation, tfQuat_body);
  tf::Matrix3x3 m_body(tfQuat_body);
  double roll_body, pitch_body, yaw_body;
  m_body.getRPY(roll_body, pitch_body, yaw_body);

  body2world_R_matrix = Eigen::AngleAxisd(yaw_body, Eigen::Vector3d::UnitZ()) *
                      Eigen::AngleAxisd(pitch_body, Eigen::Vector3d::UnitY()) *
                      Eigen::AngleAxisd(roll_body, Eigen::Vector3d::UnitX());
  body2world_R_quat = Eigen::Quaterniond(body2world_R_matrix);

  if (sim_flag_ == true)
  {
  // ! AirSim Simulation
  yaw_gimbal += yaw_body;
  pitch_gimbal += pitch_body;
  roll_gimbal = 0.0; // ? Fixed with Drone
  // ! ---------------------
  }

  Eigen::Vector3d camera_pos = body2world_t_vector;
  double pitch = pitch_gimbal;
  double yaw = yaw_body;

  // * Current Pose
  start_pose_ << camera_pos(0), camera_pos(1), camera_pos(2), pitch, yaw;

  return;
}

void selectStartFromJoystick()
{
  if (start_task_ == false)
  {
    start_task_ = true;
    ROS_INFO("\033[41;37m[FlyCo]\033[47;31m Successfully select the initialization from \n (%lf, %lf, %lf) in [World->NWU] frame with pitch %lf deg and yaw %lf deg! \033[0m", start_pose_(0), start_pose_(1), start_pose_(2), start_pose_(3)*180.0/M_PI, start_pose_(4)*180.0/M_PI);
  }

  return;
}

void joyTriggerCallback(const sensor_msgs::Joy::ConstPtr& Joy)
{
  if (!Joy) return;

  bool lb_pressed = Joy->buttons.size() > JOY_LB_BUTTON && Joy->buttons[JOY_LB_BUTTON] == true;

  if (lb_pressed && !last_lb_pressed_)
  {
    ROS_WARN("\033[43;37m[FlyCo Joy]\033[47;33m LB pressed: select current pose as start. \033[0m");
    selectStartFromJoystick();
  }

  last_lb_pressed_ = lb_pressed;

  return;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "flyco_planner");
  ros::NodeHandle nh("~");
  
  // * Initialization
  nh.param("flyco_planner/sim",         sim_flag_, true);
  nh.param("flyco_planner/odom_fps",    odom_fps_, 1.0);
  nh.param("flyco_planner/st_fps",      signal_task_fps_, 1.0);
  nh.param("flyco_planner/extrinsic_x", ext_x_, 0.0);
  nh.param("flyco_planner/extrinsic_y", ext_y_, 0.0);
  nh.param("flyco_planner/extrinsic_z", ext_z_, 0.0);
  cam2body_t_vector_ << ext_x_, ext_y_, ext_z_;
  double odom_time = 1.0/odom_fps_;

  nh.param("flyco_planner/indi_num", indi_num_, -1);
  for (int i=0; i<3; ++i)
  {
    Eigen::VectorXd indi_pt(5);
    nh.param("flyco_planner/indi_x_"+to_string(i),     indi_pt(0), 0.0);
    nh.param("flyco_planner/indi_y_"+to_string(i),     indi_pt(1), 0.0);
    nh.param("flyco_planner/indi_z_"+to_string(i),     indi_pt(2), 0.0);
    nh.param("flyco_planner/indi_pitch_"+to_string(i), indi_pt(3), 0.0);
    nh.param("flyco_planner/indi_yaw_"+to_string(i),   indi_pt(4), 0.0);
    indicators_.push_back(indi_pt);
  }

  int num = 20;
  buffer_odom_ = boost::circular_buffer<nav_msgs::OdometryConstPtr>(num);
  buffer_quat_ = boost::circular_buffer<geometry_msgs::QuaternionStampedConstPtr>(num);

  signal_task_rate_ = make_unique<ros::Rate>(ros::Rate(signal_task_fps_));
  odom_sub_.reset(new message_filters::Subscriber<nav_msgs::Odometry>(nh, "/flyco_planner/body_pose", 10));
  gimbal_sub_.reset(new message_filters::Subscriber<geometry_msgs::QuaternionStamped>(nh, "/flyco_planner/gimbal_pose", 10));
  sync_odom_gimbal_.reset(new message_filters::Synchronizer<SyncPolicyPoseGimbal>(SyncPolicyPoseGimbal(20), *odom_sub_, *gimbal_sub_));
  sync_odom_gimbal_->registerCallback(boost::bind(&odomCallback, _1, _2));
  indicator_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/flyco_planner/indicator", 1);
  ros::Timer update_pose_timer = nh.createTimer(ros::Duration(odom_time), updatePoseCallback);
  ros::Timer indicator_timer = nh.createTimer(ros::Duration(odom_time), indicatorCallback);

  flyco::logger::deviceInfo();

  // * Waiting User's Start Initialization
  ros::Subscriber joy_trigger_sub = nh.subscribe("/joy", 50, joyTriggerCallback);
  ROS_INFO("\033[42;37m[Indication]\033[47;32m Press joystick LB to select start. After activation, press joystick RB to trigger planning. \033[0m");
  while (ros::ok() && start_task_ == false)
  {
    ros::spinOnce();
    signal_task_rate_->sleep();
  }

  // * Task Starting
  update_pose_timer.stop();
  indicator_timer.stop();
  sync_odom_gimbal_.reset();
  odom_sub_.reset();
  gimbal_sub_.reset();

  FlyCoPlanManager flyco_planner_manager;
  flyco_planner_manager.setStart(start_pose_);
  flyco_planner_manager.init(nh);
  ROS_INFO("\033[41;37m[FlyCo]\033[47;31m WAITING your trigger...... \033[0m");
  flyco_planner_manager.run();

  ros::Duration(1.0).sleep();
  ros::spin();

  return 0;
}
