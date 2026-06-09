/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Sept. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the header file of the PX4 control finite-state machine for
 *                   FlyCo simulation.
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

#ifndef __PX4CTRLFSM_H
#define __PX4CTRLFSM_H

#include <ros/ros.h>
#include <ros/assert.h>

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <mavros_msgs/SetMode.h>

#include "input.h"
#include "hovthrkf.h"
#include "controller.h"

class PX4CtrlFSM
{
public:
	Parameter_t& param;

	RC_Data_t rc_data;
	State_Data_t state_data;
	Odom_Data_t odom_data;
	Imu_Data_t imu_data;
	Command_Data_t cmd_data;

	Controller& controller;
	HovThrKF& hov_thr_kf;

	// ros::Publisher des_pose_pub;
	ros::Publisher traj_start_trigger_pub;

	Eigen::Vector4d hover_pose;

	enum State_t
	{
		MANUAL_CTRL,             // px4ctrl is deactived. FCU is controled by the remote controller only
		AUTO_HOVER,	 			// px4ctrl is actived, it will keep the drone hover from odom measurments while waiting for commands from PositionCommand topic.
		CMD_CTRL				 // px4ctrl is actived, and controling the drone.
	};

	PX4CtrlFSM(Parameter_t &, Controller &, HovThrKF &);
	void process();
	bool rc_is_received(const ros::Time& now_time);
	bool cmd_is_received(const ros::Time& now_time);
	bool odom_is_received(const ros::Time& now_time);
	bool imu_is_received(const ros::Time& now_time);
	bool px4_init();


private:
	State_t state;
	// ---- control related ----
	void process_hover_control(Controller_Output_t& u, SO3_Controller_Output_t& u_so3);
	void process_cmd_control(Controller_Output_t& u, SO3_Controller_Output_t& u_so3);
	// void publish_desire(const Desired_State_t& des);

	// ---- tools ----
	double get_yaw_from_odom();
	void align_with_imu(Controller_Output_t& u);
	void set_hov_with_odom();

	void toggle_offboard_mode(bool on_off);  // It will only try to toggle once, so not blocked.

	void publish_trigger(const nav_msgs::Odometry& odom_msg);
};

#endif