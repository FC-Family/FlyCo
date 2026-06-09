/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Sept. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the manual controller using joystick.
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

#include "airsim_ros_pkgs/GimbalAngleEulerCmd.h"
#include "active_perception/perception_utils.h"
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Joy.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <visualization_msgs/Marker.h>
#include <Eigen/Dense>
#include <thread>
#include <tf/tf.h>

using namespace std;
using std::unique_ptr;
using flyco::PerceptionUtils;

typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, geometry_msgs::QuaternionStamped> SyncPolicyPoseGimbal;
typedef unique_ptr<message_filters::Synchronizer<SyncPolicyPoseGimbal>> SynchronizerPoseGimbal;

unique_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub_;
unique_ptr<message_filters::Subscriber<geometry_msgs::QuaternionStamped>> gimbal_sub_;
SynchronizerPoseGimbal sync_odom_gimbal_;

/* ROS Service */
ros::Publisher pos_cmd_pub, gimbal_cmd_pub, exec_traj_pub, robot_pub, fov_pub;
quadrotor_msgs::PositionCommand cmd;
airsim_ros_pkgs::GimbalAngleEulerCmd gimbal_cmd;
sensor_msgs::Joy joy_;
Eigen::Quaterniond current_orientation;
unique_ptr<ros::Rate> rate = nullptr;
unique_ptr<PerceptionUtils> percep_utils_ = nullptr;

/* Param */
double ctrl_time_;
double transition_;
double kp_x_, kp_y_, kp_z_, kp_yaw_, kp_pitch_;
double v_max_, a_max_, ydot_max_;
double pitch_upper_, pitch_lower_;
double init_z_;
bool start_joy_ = false;

/* Data */
double current_body_x = 0.0, current_body_y = 0.0, current_body_z = 0.0;
double current_body_vel_x = 0.0, current_body_vel_y = 0.0, current_body_vel_z = 0.0;
double currennt_body_yaw = 0.0;
double current_gimbal_pitch = 0.0;

template <typename T>
T clamp(T v, T lo, T hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

vector<Eigen::Vector3d> exec_traj_;

/* Func */
double smoothStep(double edge0, double edge1, double x) 
{
    // Scale, bias and saturate x to 0..1 range
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    // Evaluate polynomial
    return x * x * (3 - 2 * x);
}

double mapAxis(double value, double kp, double a_max, double transition_point)
{
    double abs_value = std::abs(value);
    if (abs_value < transition_point) 
    {
        return kp * a_max * std::pow(value, 3);
    } 
    else if (abs_value < transition_point + 0.1) 
    {
        double t = smoothStep(transition_point, transition_point + 0.1, abs_value);
        double cubic = std::pow(transition_point, 3);
        double linear = transition_point;
        double interpolated = (1 - t) * cubic + t * linear;
        return kp * a_max * (value >= 0 ? interpolated : -interpolated);
    } 
    else 
    {
        return kp * a_max * value;
    }
}

void odomCallback(const nav_msgs::OdometryConstPtr& pose, const geometry_msgs::QuaternionStampedConstPtr& quat)
{
    current_body_x = pose->pose.pose.position.x;
    current_body_y = pose->pose.pose.position.y;
    current_body_z = pose->pose.pose.position.z;

    current_body_vel_x = pose->twist.twist.linear.x;
    current_body_vel_y = pose->twist.twist.linear.y;
    current_body_vel_z = pose->twist.twist.linear.z;
    
    current_orientation.w() = pose->pose.pose.orientation.w;
    current_orientation.x() = pose->pose.pose.orientation.x;
    current_orientation.y() = pose->pose.pose.orientation.y;
    current_orientation.z() = pose->pose.pose.orientation.z;

    tf::Quaternion tfQuat_body;
    tf::quaternionMsgToTF(pose->pose.pose.orientation, tfQuat_body);
    tf::Matrix3x3 m_body(tfQuat_body);
    double roll_body, pitch_body, yaw_body;
    m_body.getRPY(roll_body, pitch_body, yaw_body);

    currennt_body_yaw = yaw_body;

    tf::Quaternion tfQuat;
    tf::quaternionMsgToTF(quat->quaternion, tfQuat);
    tf::Matrix3x3 m(tfQuat);
    double roll_gimbal, pitch_gimbal, yaw_gimbal;
    m.getRPY(roll_gimbal, pitch_gimbal, yaw_gimbal);

    current_gimbal_pitch = pitch_gimbal;

    if (exec_traj_.empty())
    {
        Eigen::Vector3d cur_pos(pose->pose.pose.position.x, pose->pose.pose.position.y, pose->pose.pose.position.z);
        exec_traj_.push_back(cur_pos);
    }
    else
    {
        Eigen::Vector3d last_pos = exec_traj_.back();
        Eigen::Vector3d cur_pos(pose->pose.pose.position.x, pose->pose.pose.position.y, pose->pose.pose.position.z);
        double dist = (cur_pos - last_pos).norm();
        if (dist > 0.1)
            exec_traj_.push_back(cur_pos);
    }

    return;
}

void execVisCallback(const ros::TimerEvent& e)
{   
    Eigen::Vector3d cur_pos(current_body_x, current_body_y, current_body_z);
    
    // * robot
    visualization_msgs::Marker robot_vis;
    robot_vis.header.frame_id = "world";
    robot_vis.header.stamp = ros::Time::now();
    robot_vis.id = 0;
    robot_vis.ns = "joy_robot_vis";
    robot_vis.type = visualization_msgs::Marker::MESH_RESOURCE;
    robot_vis.color.r = 1.0;
    robot_vis.color.g = 0.0;
    robot_vis.color.b = 1.0;
    robot_vis.color.a = 1.0;
    robot_vis.scale.x = 1.0;
    robot_vis.scale.y = 1.0;
    robot_vis.scale.z = 1.0;
    robot_vis.mesh_resource = "package://vis_utils/DJI_M30T_S.dae";

    robot_vis.action = visualization_msgs::Marker::DELETE;
    robot_pub.publish(robot_vis);

    double yaw_drone = currennt_body_yaw - M_PI/2.0;

    robot_vis.pose.orientation.x = 0.0;
    robot_vis.pose.orientation.y = 0.0;
    robot_vis.pose.orientation.z = sin(0.5*yaw_drone);
    robot_vis.pose.orientation.w = cos(0.5*yaw_drone);
    robot_vis.pose.position.x = cur_pos(0);
    robot_vis.pose.position.y = cur_pos(1);
    robot_vis.pose.position.z = cur_pos(2);
    robot_vis.action = visualization_msgs::Marker::ADD;

    robot_pub.publish(robot_vis);
    
    // * FoV
    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.id = 0;
    mk.ns = "joy_fov_vis";
    mk.type = visualization_msgs::Marker::LINE_LIST;
    mk.color.r = 0.0;
    mk.color.g = 0.0;
    mk.color.b = 0.0;
    mk.color.a = 1.0;
    mk.scale.x = 0.08;
    mk.scale.y = 0.08;
    mk.scale.z = 0.08;
    mk.pose.orientation.w = 1.0;

    mk.action = visualization_msgs::Marker::DELETE;
    fov_pub.publish(mk);

    percep_utils_->setPose_PY(cur_pos, current_gimbal_pitch, currennt_body_yaw);
    vector<Eigen::Vector3d> list1, list2;
    percep_utils_->getFOV_PY(list1, list2);

    geometry_msgs::Point pt;
    for (int i = 0; i < int(list1.size()); ++i) 
    {
        pt.x = list1[i](0);
        pt.y = list1[i](1);
        pt.z = list1[i](2);
        mk.points.push_back(pt);

        pt.x = list2[i](0);
        pt.y = list2[i](1);
        pt.z = list2[i](2);
        mk.points.push_back(pt);
    }
    mk.action = visualization_msgs::Marker::ADD;
    fov_pub.publish(mk);

    // * exec trajectory
    if (exec_traj_.empty())
    return;
    
    nav_msgs::Path traj_msg_;
    traj_msg_.header.stamp = ros::Time::now();
    traj_msg_.header.frame_id = "world";

    for (const auto& point : exec_traj_)
    {
        geometry_msgs::PoseStamped pose;
        pose.pose.position.x = point(0);
        pose.pose.position.y = point(1);
        pose.pose.position.z = point(2);

        pose.pose.orientation.x = 0;
        pose.pose.orientation.y = 0;
        pose.pose.orientation.z = 0;
        pose.pose.orientation.w = 1;

        pose.header.stamp=ros::Time::now();;
        pose.header.frame_id="world";

        traj_msg_.poses.push_back(pose);
    }

    exec_traj_pub.publish(traj_msg_);

    return;
}

void timeCallback(const ros::TimerEvent& e)
{   
    ros::Time cur_time = ros::Time::now();
    
    if (!start_joy_)
        return;

    cmd.kx = { 7.0, 7.0, 8.0 };
    cmd.kv = { 4.4, 4.4, 5.0 };
    cmd.header.stamp = cur_time;
    cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
    cmd.trajectory_id = 0;

    cmd.jerk.x = 0.0;
    cmd.jerk.y = 0.0;
    cmd.jerk.z = 0.0;

    double delta_x = 0, delta_y = 0, delta_z = 0, delta_yaw = 0;
    double pitch_up = 0, pitch_down = 0, pitch_cmd = 0;
    double position_scaling_x = v_max_ * ctrl_time_, position_scaling_y = v_max_ * ctrl_time_, position_scaling_z = v_max_ * ctrl_time_;

    if (!joy_.axes.empty())
    {
        delta_x = mapAxis(joy_.axes[1], kp_x_, position_scaling_x, transition_);
        delta_y = mapAxis(joy_.axes[0], kp_y_, position_scaling_y, transition_);
        delta_z = mapAxis(joy_.axes[4], kp_z_, position_scaling_z, transition_);
        delta_yaw = kp_yaw_ * ydot_max_ * std::pow(joy_.axes[3], 2) * (joy_.axes[3] >= 0 ? 1 : -1);

        pitch_up = -kp_pitch_ * (joy_.axes[2] - 1.0);
        pitch_down = -kp_pitch_ * (joy_.axes[5] - 1.0);
        pitch_cmd = pitch_up - pitch_down;
    }

    // * Local to World Frame
    Eigen::Vector3d delta_body(delta_x, delta_y, delta_z);
    Eigen::Vector3d delta_world = current_orientation * delta_body;
    delta_x = delta_world.x();
    delta_y = delta_world.y();
    delta_z = delta_world.z();

    double new_x = cmd.position.x + delta_x;
    double new_y = cmd.position.y + delta_y;
    double new_z = cmd.position.z + delta_z;

    Eigen::Vector3d des_vel(delta_x / ctrl_time_, delta_y / ctrl_time_, delta_z / ctrl_time_);
    double des_vel_norm = des_vel.norm();
    des_vel = des_vel_norm > v_max_ ? des_vel / des_vel_norm * v_max_ : des_vel;

    cmd.velocity.x = des_vel(0);
    cmd.velocity.y = des_vel(1);
    cmd.velocity.z = des_vel(2);

    cmd.position.x = new_x;
    cmd.position.y = new_y;
    cmd.position.z = new_z;

    cmd.yaw += delta_yaw * ctrl_time_;

    cmd.acceleration.x = 0.0;
    cmd.acceleration.y = 0.0;
    cmd.acceleration.z = 0.0;
    
    cmd.yaw_dot = delta_yaw;

    gimbal_cmd.header.stamp = cur_time;
    gimbal_cmd.roll = 0.0;
    gimbal_cmd.pitch += pitch_cmd;
    gimbal_cmd.pitch = std::max(pitch_lower_, std::min(pitch_upper_, gimbal_cmd.pitch));
    gimbal_cmd.yaw = 0.0;

    gimbal_cmd_pub.publish(gimbal_cmd);
    pos_cmd_pub.publish(cmd);

    rate->sleep();
}

void joyCallback(const sensor_msgs::Joy::ConstPtr& Joy)
{
    if (Joy)
      joy_ = *Joy;
    
    if (joy_.buttons[0] == true)
    {
        start_joy_ = true;
        ROS_WARN("[Joy Node]: Start Joy Control......");

        // Initialize command
        cmd.position.x = current_body_x;
        cmd.position.y = current_body_y;
        cmd.position.z = current_body_z;
        cmd.velocity.x = 0.0;
        cmd.velocity.y = 0.0;
        cmd.velocity.z = 0.0;
        cmd.acceleration.x = 0.0;
        cmd.acceleration.y = 0.0;
        cmd.acceleration.z = 0.0;
        cmd.jerk.x = 0.0;
        cmd.jerk.y = 0.0;
        cmd.jerk.z = 0.0;
        cmd.yaw = currennt_body_yaw;
        cmd.yaw_dot = 0.0;

        gimbal_cmd.roll = 0.0;
        gimbal_cmd.pitch = current_gimbal_pitch*180.0/M_PI;
        gimbal_cmd.yaw = 0.0;
    }
    else if (joy_.buttons[1] == true)
    {
        start_joy_ = false;
        ROS_WARN("[Joy Node]: Stop Joy Control......");
    }

    return;
}

int main(int argc,char** argv) {
    ros::init(argc, argv, "joy_node");
    ros::NodeHandle n;
    ros::Subscriber sub, odom_sub;

    ROS_ERROR("[Joy Node]: Please trigger the 'LT' and 'RT' keys at the same time to unlock the device......");
    ROS_ERROR("[Joy Node]: Press 'A' button to start the joy control......");
    ROS_ERROR("[Joy Node]: Press 'B' button to stop the joy control......");

    n.param("/joy_cmd/time", ctrl_time_, -1.0);
    n.param("/joy_cmd/transition", transition_, 0.5);
    n.param("/joy_cmd/kp_x", kp_x_, -1.0);
    n.param("/joy_cmd/kp_y", kp_y_, -1.0);
    n.param("/joy_cmd/kp_z", kp_z_, -1.0);
    n.param("/joy_cmd/kp_yaw", kp_yaw_, -1.0);
    n.param("/joy_cmd/kp_pitch", kp_pitch_, -1.0);
    n.param("/joy_cmd/max_vel", v_max_, -1.0);
    n.param("/joy_cmd/max_acc", a_max_, -1.0);
    n.param("/joy_cmd/max_yd", ydot_max_, -1.0);
    n.param("/joy_cmd/pitch_upper", pitch_upper_, -1.0);
    n.param("/joy_cmd/pitch_lower", pitch_lower_, -1.0);
    n.param("/joy_cmd/init_z", init_z_, -1.0);

    double top_angle, left_angle, right_angle, max_dist, vis_dist;
    n.param("/joy_cmd/top_angle", top_angle, -1.0);
    n.param("/joy_cmd/left_angle", left_angle, -1.0);
    n.param("/joy_cmd/right_angle", right_angle, -1.0);
    n.param("/joy_cmd/max_dist", max_dist, -1.0);
    n.param("/joy_cmd/vis_dist", vis_dist, -1.0);
    ros::NodeHandle nh;
    nh.setParam("perception_utils/top_angle", top_angle);
    nh.setParam("perception_utils/left_angle", left_angle);
    nh.setParam("perception_utils/right_angle", right_angle);
    nh.setParam("perception_utils/max_dist", max_dist);
    nh.setParam("perception_utils/vis_dist", vis_dist);
    percep_utils_.reset(new PerceptionUtils);
    percep_utils_->init(nh);

    odom_sub_.reset(new message_filters::Subscriber<nav_msgs::Odometry>(n, "/airsim_node/drone_1/odom_local_enu", 10));
    gimbal_sub_.reset(new message_filters::Subscriber<geometry_msgs::QuaternionStamped>(n, "/gimbal_pose", 10));
    sync_odom_gimbal_.reset(new message_filters::Synchronizer<SyncPolicyPoseGimbal>(SyncPolicyPoseGimbal(20), *odom_sub_, *gimbal_sub_));
    sync_odom_gimbal_->registerCallback(boost::bind(&odomCallback, _1, _2));

    // * Initialize cmds
    cmd.kx = { 7.0, 7.0, 8.0 };
    cmd.kv = { 4.4, 4.4, 5.0 };
    cmd.header.stamp = ros::Time::now();
    cmd.header.frame_id = "world";
    cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
    cmd.trajectory_id = 0;

    gimbal_cmd.header.stamp = ros::Time::now();
    gimbal_cmd.camera_name = "front_center";
    gimbal_cmd.vehicle_name = "drone_1";

    double fps = 1.0 / ctrl_time_;
    rate = make_unique<ros::Rate>(ros::Rate(fps));

    pos_cmd_pub = n.advertise<quadrotor_msgs::PositionCommand>("/joy_ctrl/pos_cmd", 50);
    gimbal_cmd_pub = n.advertise<airsim_ros_pkgs::GimbalAngleEulerCmd>("/joy_ctrl/gimbal_cmd", 50);
    exec_traj_pub = n.advertise<nav_msgs::Path>("/joy_ctrl/exec_traj", 1, true);
    robot_pub = n.advertise<visualization_msgs::Marker>("/joy_ctrl/robot", 1);
    fov_pub = n.advertise<visualization_msgs::Marker>("/joy_ctrl/fov", 1);

    ros::Timer exec_timer_ = n.createTimer(ros::Duration(ctrl_time_), &timeCallback);
    ros::Timer exec_vis_timer_ = n.createTimer(ros::Duration(0.1), &execVisCallback);
    sub = n.subscribe<sensor_msgs::Joy>("joy",50, &joyCallback);
    
    ros::Duration(1.0).sleep();
    ros::spin();

    return 0;
}