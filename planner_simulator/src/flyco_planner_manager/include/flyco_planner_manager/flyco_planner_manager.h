/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Sept. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the header file of FlyCoPlanManager class, which implements
 *                   the mapping and planning manager node of FlyCo.
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

#ifndef _FLYCO_PLANNER_MANAGER_H_
#define _FLYCO_PLANNER_MANAGER_H_

#include "plan_env/sdf_map.h"
#include "hierarchical_coverage_planner/planner_node.h"
#include "hierarchical_coverage_planner/manual_node.h"
#include "hierarchical_coverage_planner/init_traj.h"
#include <ros/ros.h>
#include <Eigen/Core>
#include <sensor_msgs/Joy.h>

using namespace std;
using std::shared_ptr;
using std::unique_ptr;

namespace flyco
{
  class SDFMap;
  class FCPlanner_PP_Node;
  class Manual_Node;
  class Init_Traj;
  
  class FlyCoPlanManager
  {
    public:
      FlyCoPlanManager(){
      }
      ~FlyCoPlanManager(){
      }
      /* Func */
      void setStart(Eigen::VectorXd& input);
      void init(ros::NodeHandle& nh);
      void run();
    
    private:
      /* Module */
      shared_ptr<SDFMap> dual_mapping_ = nullptr;
      unique_ptr<FCPlanner_PP_Node> planning_ = nullptr;
      unique_ptr<Manual_Node> manual_ = nullptr;
      unique_ptr<Init_Traj> init_traj_generator_ = nullptr;
      /* Param */
      ros::NodeHandle nh_;
      Eigen::VectorXd start_pose_;
      bool auto_mode_ = true;
      bool sim_flag_ = false;
      bool init_traj_ = false;
      bool en_activate_ = false, already_activate_ = false;
      /* Func */
      void activate();
      void joyCallback(const sensor_msgs::Joy::ConstPtr& Joy);
      void activateCallback(const ros::TimerEvent& e);
      /* ROS Service */
      ros::Subscriber joy_sub_;
      ros::Timer activate_timer_;
  };


} // namespace flyco

#endif