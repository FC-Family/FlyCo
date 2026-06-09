/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Sept. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the header file of the PX4 low-level controller for FlyCo
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

#ifndef __CONTROLLER_H
#define __CONTROLLER_H

#include <sensor_msgs/Imu.h>
#include <mavros_msgs/AttitudeTarget.h>
#include <tf/tf.h>

#include "input.h"

struct Desired_State_t
{
	Eigen::Vector3d p;
	Eigen::Vector3d v;
	double yaw;
	double head_rate;
	Eigen::Quaterniond q;
	Eigen::Vector3d a;
	Eigen::Vector3d jerk;
};

struct Controller_Output_t
{
    static constexpr double CTRL_YAW_RATE = 1.0;
    static constexpr double CTRL_YAW = 0.0;

	double roll;
	double pitch;
	double yaw;
	double thrust;
	double roll_rate;
	double pitch_rate;
	double yaw_rate;
	double yaw_mode; // if yaw_mode > 0, CTRL_YAW;
				// if yaw_mode < 0, CTRL_YAW_RATE
	Eigen::Quaterniond orientation;
	double normalized_thrust;

	Eigen::Vector3d des_v_real;
};

struct SO3_Controller_Output_t
{
	Eigen::Matrix3d Rdes;
	Eigen::Vector3d Fdes;
	double net_force;
};

class Controller
{
public:
	Parameter_t& param;

	ros::Publisher ctrl_FCU_pub;
	ros::Publisher debug_roll_pub;
	ros::Publisher debug_pitch_pub;
	ros::ServiceClient set_FCU_mode;

	Eigen::Matrix3d Kp;
	Eigen::Matrix3d Kv;
	Eigen::Matrix3d Kvi;
	Eigen::Matrix3d Ka;
	double Kyaw;

	Eigen::Vector3d int_e_v;

	Controller(Parameter_t&);
	void config_gain(const Parameter_t::Gain& gain);
	void config();
	void update(const Desired_State_t& des, const Odom_Data_t& odom, 
		Controller_Output_t& u, SO3_Controller_Output_t& u_so3
	);
	Controller_Output_t computeNominalReferenceInputs(
    const Desired_State_t& reference_state,
    const Odom_Data_t& attitude_estimate) const;

	Eigen::Quaterniond computeDesiredAttitude(
    const Eigen::Vector3d& desired_acceleration, const double reference_heading,
    const Eigen::Quaterniond& attitude_estimate) const;
	bool almostZero(const double value) const;
	bool almostZeroThrust(const double thrust_value) const;
	Eigen::Vector3d computeRobustBodyXAxis(
    const Eigen::Vector3d& x_B_prototype, const Eigen::Vector3d& x_C,
    const Eigen::Vector3d& y_C,
    const Eigen::Quaterniond& attitude_estimate) const; 
	Eigen::Vector3d computeFeedBackControlBodyrates(const Eigen::Quaterniond& desired_attitude,
    const Eigen::Quaterniond& attitude_estimate);
	Eigen::Vector3d computePIDErrorAcc(
    const Odom_Data_t& state_estimate,
    const Desired_State_t& reference_state);
	
	void publish_ctrl(const Controller_Output_t& u, const ros::Time& stamp);
	void publish_zero_ctrl(const ros::Time& stamp);

private:
	bool is_configured;
};

#endif
