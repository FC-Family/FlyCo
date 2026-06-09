/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Feb. 2025
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the header file of Init_Traj class, which implements the
 *                   generation of initial trajectory in FlyCo.
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

#ifndef _INIT_TRAJ_H_
#define _INIT_TRAJ_H_

#include "plan_env/sdf_map.h"
#include "hierarchical_coverage_planner/hctraj.h"
#include "plan_utils/path_tools.hpp"
#include "gcopter/voxel_map.hpp"
#include "gcopter/trajectory.hpp"
#include "quadrotor_msgs/PolynomialTrajGroup.h"
#include <ros/ros.h>
#include <ros/package.h>
#include <Eigen/Core>
#include <sensor_msgs/Joy.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <yaml-cpp/yaml.h>
#include <tf/tf.h>

using namespace std;
using std::shared_ptr;
using std::unique_ptr;

namespace flyco
{

class SDFMap;
class TrajGenerator;

class Init_Traj
{

public:
  Init_Traj();
  ~Init_Traj();
  /* Func */
  void init(ros::NodeHandle& nh);
  void setMap(shared_ptr<SDFMap>& map);

private:
  /* Utils */
  shared_ptr<SDFMap> mapping_ = nullptr;
  unique_ptr<TrajGenerator> traj_generator_ = nullptr;
  /* Param */
  bool start_gen_ = false;
  bool finish_init_ = false;
  bool en_inherit_ = false;
  int traj_id_ = -1;
  int path_size_ = 0;
  int num_path_ = 0;
  string init_path_file_;
  /* Func */
  void run();
  void readInitialPoints();
  void trajConverter(const Trajectory<7> &pos, const Trajectory<7> &ori, quadrotor_msgs::PolynomialTrajGroup &msg, const ros::Time &cur_stamp, int &traj_id);
  void runCallback(const ros::TimerEvent& event);
  void joyCallback(const sensor_msgs::Joy::ConstPtr& Joy);
  void odomCallback(const nav_msgs::OdometryConstPtr& msg);
  void gimbalCallback(const geometry_msgs::QuaternionStampedConstPtr& msg);
  /* ROS Service */
  ros::Timer task_timer_;
  ros::Publisher traj_pub_;
  ros::Subscriber joy_sub_, odom_sub_, gimbal_sub_;
  /* Data */
  nav_msgs::Odometry odom_;
  geometry_msgs::QuaternionStamped gimbal_;
  vector<Eigen::VectorXd> waypoints_;
  vector<bool> indicators_;
  vector<vector<Eigen::VectorXd>> paths_;
  vector<vector<bool>> paths_indicators_;
};

}

#endif