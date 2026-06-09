/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Sept. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the header file of PX4 control input handling for FlyCo
 *                   simulation.
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

#ifndef __INPUT_H
#define __INPUT_H

#include <ros/ros.h>
#include <Eigen/Dense>

#include <sensor_msgs/Imu.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <mavros_msgs/RCIn.h>
#include <mavros_msgs/State.h>
#include <uav_utils/utils.h>
#include "PX4CtrlParam.h"

class RC_Data_t {
  public:
    double mode;
    double gear;
    double last_mode;
    double last_gear;
    bool have_init_last_mode{false};
    bool have_init_last_gear{false};

    mavros_msgs::RCIn msg;
    ros::Time rcv_stamp;

    bool is_command_mode;
    bool enter_command_mode;
    bool is_api_mode;
    bool enter_api_mode;    

    static constexpr double GEAR_SHIFT_VALUE = 0.75;
    static constexpr double API_MODE_THRESHOLD_VALUE = 0.75;

    RC_Data_t();
    void check_validity();
    void feed(mavros_msgs::RCInConstPtr pMsg);
};

class Odom_Data_t {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d p;
    Eigen::Vector3d v;
    Eigen::Vector3d w;
    Eigen::Quaterniond q;

    nav_msgs::Odometry msg;
    ros::Time rcv_stamp;
    bool odom_init;
    Odom_Data_t();
    void feed(nav_msgs::OdometryConstPtr pMsg);
};

class Imu_Data_t {
  public:
    Eigen::Quaterniond q;
    Eigen::Vector3d w;
    Eigen::Vector3d a;

    sensor_msgs::Imu msg;
    ros::Time rcv_stamp;
    bool imu_init;
    Imu_Data_t();
    void feed(sensor_msgs::ImuConstPtr pMsg);
};

class State_Data_t {
  public:
    mavros_msgs::State current_state;

    State_Data_t();
    void feed(mavros_msgs::StateConstPtr pMsg);
};

class Command_Data_t {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d p;
    Eigen::Vector3d v;
    Eigen::Vector3d a;
    Eigen::Vector3d jerk;
    double yaw;
    double head_rate;
    bool cmd_init;
    quadrotor_msgs::PositionCommand msg;
    ros::Time rcv_stamp;

    Command_Data_t();
    void feed(quadrotor_msgs::PositionCommandConstPtr pMsg);
};

#endif