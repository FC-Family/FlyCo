/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Sept. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the header file of PX4 control parameters for FlyCo
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

#ifndef	__PX4CTRLPARAM_H
#define __PX4CTRLPARAM_H

#include <ros/ros.h>

class Parameter_t
{
public:
	struct Gain
	{
		double Kp0,Kp1,Kp2;
		double Kv0,Kv1,Kv2;
		double Kvi0,Kvi1,Kvi2;
		double Ka0,Ka1,Ka2;
		double Kyaw;
		double Krp;
	};

	struct Hover
	{
		bool use_hov_percent_kf;
		
		double percent_lower_limit;
		double percent_higher_limit;
	};

	struct MsgTimeout
	{
		double odom;
		double rc;
		double cmd;
		double imu;
	};

	Gain hover_gain, track_gain;
	Hover hover;
	MsgTimeout msg_timeout;

	double mass;
	double gra;
	double hov_percent;
	double full_thrust;
	

	double ctrl_rate;
	

	bool use_yaw_rate_ctrl;
	bool perform_aerodynamics_compensation;
	double pxy_error_max;
	double vxy_error_max;
	double pz_error_max;
	double vz_error_max;
	double yaw_error_max;
	Parameter_t();
	void config_from_ros_handle(const ros::NodeHandle& nh);
	void init();
	void config_full_thrust(double hov);
private:	

	template<typename TName, typename TVal>
	void read_essential_param(const ros::NodeHandle& nh, const TName& name, TVal& val)
	{
		if (nh.getParam(name, val))
		{
			// pass
		}
		else
		{
			ROS_ERROR_STREAM("Read param: " << name << " failed.");
			ROS_BREAK();
		}
	};
};

#endif