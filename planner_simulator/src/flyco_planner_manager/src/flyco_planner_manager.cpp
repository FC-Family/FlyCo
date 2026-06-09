/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Sept. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the main file of mapping and planning manager node of FlyCo.
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

namespace flyco
{

void FlyCoPlanManager::setStart(Eigen::VectorXd& input)
{
  this->start_pose_.resize(input.size());
  this->start_pose_ = input;

  return;
}

void FlyCoPlanManager::init(ros::NodeHandle& nh)
{
  // * Params Initialization
  nh.param("flyco_planner/sim",       this->sim_flag_, false);
  nh.param("flyco_planner/init_traj", this->init_traj_, false);
  this->nh_ = nh;
  this->en_activate_ = false;
  this->already_activate_ = false;
  this->auto_mode_ = true;
  
  // * Module Initialization
  this->dual_mapping_.reset(new SDFMap);
  this->dual_mapping_->setXYoffset(this->start_pose_(0), this->start_pose_(1));
  this->dual_mapping_->initMap(nh);
  this->dual_mapping_->initHC(nh);

  if (this->auto_mode_ == true)
  {
    this->planning_.reset(new FCPlanner_PP_Node);
    this->planning_->init(nh);
    this->planning_->setMap(this->dual_mapping_);
  }
  else
  {
    this->manual_.reset(new Manual_Node);
    this->manual_->init(nh);
  }

  // * ROS Initialization
  this->joy_sub_ = nh.subscribe("/joy", 50, &FlyCoPlanManager::joyCallback, this);

  return;
}

void FlyCoPlanManager::run()
{
  ROS_INFO("\033[42;37m[Indication]\033[47;32m <1> Firstly, please press button 'Y' to start initial flight! \033[0m");
  ROS_INFO("\033[42;37m[Indication]\033[47;32m <2> After initial flight, please press button 'X' to start all services of planner! \033[0m");

  if (this->init_traj_)
  {
    // * Initial Trajectory Generation
    this->init_traj_generator_.reset(new Init_Traj);
    this->init_traj_generator_->init(this->nh_);
    this->init_traj_generator_->setMap(this->dual_mapping_);
  }

  this->activate_timer_ = this->nh_.createTimer(ros::Duration(0.1), &FlyCoPlanManager::activateCallback, this);

  return;
}

void FlyCoPlanManager::activate()
{
  if (this->auto_mode_ == true)
  {
    ROS_INFO("\033[42;37m[FlyCo]\033[47;32m Auto Mode! \033[0m");
    this->planning_->startAllService();
    ros::spin();
    this->planning_->stopAllService();
    ros::Duration(1.0).sleep();
  }
  else
  {
    ROS_INFO("\033[43;37m[FlyCo]\033[47;33m Manual Mode! \033[0m");
  }
  
  return;
}

void FlyCoPlanManager::joyCallback(const sensor_msgs::Joy::ConstPtr& Joy)
{
  sensor_msgs::Joy joy_msg;
  if (Joy) joy_msg = *Joy;

  // Button 'X' to activate all services of planning
  if (joy_msg.buttons[2] == true) this->en_activate_ = true;

  return;
}

void FlyCoPlanManager::activateCallback(const ros::TimerEvent& e)
{
  if (this->en_activate_ && !this->already_activate_)
  {
    ROS_WARN("[FlyCo] Activate......");
    this->already_activate_ = true;
    this->activate();
  }

  return;
}

} // namespace flyco