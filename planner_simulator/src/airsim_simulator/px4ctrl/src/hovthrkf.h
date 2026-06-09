/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Sept. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the header file of hover-thrust Kalman filtering for FlyCo
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

#ifndef __HOVTHRKF_H
#define __HOVTHRKF_H

#include <Eigen/Dense>
#include <ros/ros.h>
#include "input.h"

class HovThrKF
{
public:
	Parameter_t& param;
	ros::Publisher hov_thr_pub;

	HovThrKF(Parameter_t&);
	void init();
	void process(double u);
	void update(double a);
	double get_hov_thr();
	void set_hov_thr(double hov);
	void simple_update(Eigen::Quaterniond q, double u, Eigen::Vector3d acc);
	void simple_update(Eigen::Vector3d des_v, Eigen::Vector3d odom_v);
private:
	Eigen::VectorXd x;
	Eigen::MatrixXd P;
	Eigen::MatrixXd Q;
	Eigen::MatrixXd F;
	Eigen::MatrixXd H;
	Eigen::MatrixXd B;
	Eigen::MatrixXd R;
};

#endif