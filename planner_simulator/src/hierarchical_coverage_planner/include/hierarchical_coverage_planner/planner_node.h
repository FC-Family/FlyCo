/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Jun. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the header file of FCPlanner_PP_Node class, which implements
 *                   the whole asynchronous planning framework of FlyCo.
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

#ifndef _FCPLANNER_PP_NODE_H_
#define _FCPLANNER_PP_NODE_H_

#include "active_perception/perception_utils.h"
#include "plan_env/sdf_map.h"
#include "gcopter/voxel_map.hpp"
#include "gcopter/geo_utils.hpp"
#include "gcopter/trajectory.hpp"
#include "hierarchical_coverage_planner/fcplanner_pp.h"
#include "hierarchical_coverage_planner/hctraj.h"
#include "hierarchical_coverage_planner/local_replan.h"
#include "plan_utils/visibility_st.hpp"
#include "quadrotor_msgs/Mesh.h"
#include "quadrotor_msgs/PolynomialTrajGroup.h"
#include "quadrotor_msgs/EigenVectorArray.h"
#include "quadrotor_msgs/BEVBox.h"
#include <ros/package.h>
#include <Eigen/Core>
#include <embree3/rtcore.h>
#include <yaml-cpp/yaml.h>
#include <boost/circular_buffer.hpp>
#include <mutex>
#include <thread>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Joy.h>
#include <std_msgs/Bool.h>
#include <iomanip>
#include <pcl/filters/random_sample.h>

using namespace std;
using std::unique_ptr;
using std::shared_ptr;

#define NEG_INFINITY                 (-1e6)

namespace flyco
{

class PerceptionUtils;
class SDFMap;
class FCPlanner_PP;
class TrajGenerator;
class Local_Replan;
class PlanningVisualization;

enum GLOBAL_STATE { GLOBAL_SILENCE, GLOBAL_PLAN, GLOBAL_EXEC, GLOBAL_FINISHED };
enum LOCAL_STATE { LOCAL_SILENCE, LOCAL_PLAN, LOCAL_EXEC, LOCAL_FINISHED };
class FCPlanner_PP_Node
{

public:
  FCPlanner_PP_Node(){
  }
  ~FCPlanner_PP_Node(){
  }
  /* Func */
  void init(ros::NodeHandle& nh);
  void setMap(shared_ptr<SDFMap>& map);
  void FCPlannerv1(Eigen::MatrixXd& mesh_V, Eigen::MatrixXi& mesh_F);
  void startAllService();
  void stopAllService();
  /* Operations */
  void setPlanStart(Eigen::VectorXd start_pose);
  void setLocalStart(Eigen::VectorXd local_start, Eigen::Vector3d vel, Eigen::Vector3d acc, Eigen::Vector3d pyd, Eigen::Vector3d pyd_dot);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  /* Task */
  std::thread global_planning_thread_;
  std::thread local_planning_thread_;
  std::thread progress_thread_;
  std::thread safety_thread_;
  std::thread vis_thread_;
  int global_planning_running_ = 1;
  int local_planning_running_ = 1;
  int progress_running_ = 1;
  int safety_running_ = 1;
  int vis_running_ = 1;
  /* Mutex */
  std::mutex global_input_mtx_;
  std::mutex g_l_intersec_mtx_;
  std::mutex local_output_mtx_;
  std::mutex vis_mtx_;
  /* ROS Rate */
  unique_ptr<ros::Rate> global_planning_rate_ = nullptr;
  unique_ptr<ros::Rate> local_planning_rate_ = nullptr;
  unique_ptr<ros::Rate> progress_rate_ = nullptr;
  unique_ptr<ros::Rate> safety_rate_ = nullptr;
  unique_ptr<ros::Rate> vis_rate_ = nullptr;
  /* Param */
  bool visFlag = true, static_state_global_ = true, static_state_local_ = true, global_succeed_ = true;
  double freq_ = 0.0, res_ = 0.0, drone_radius_ = 0.0, suspension_target_ = 0.0, safe_radius_ = 0.0;
  double global_fps_ = 0.0, local_fps_ = 0.0, progress_fps_ = 0.0, safety_fps_ = 0.0, vis_fps_ = 0.0;
  double global_plan_time_ = 0.0, global_latency_ = 0.0;
  double replan_time_ = 0.0, replan_full_exec_time_ = 0.0, replan_periodic_time_ = 0.0, replan_proportion_ = 0.0;
  double safety_horizon_ = 0.0;
  double ground_height_ = 0.0, top_height_ = 0.0;
  int sample_size_ = 0;
  double vis_inflation_ = 0.0;

  bool bev_seg_ = false;
  double bev_min_x_ = 0.0, bev_max_x_ = 0.0, bev_min_y_ = 0.0, bev_max_y_ = 0.0;

  bool en_inherit_ = false;
  bool last_rb_pressed_ = false;
  /* Func */
  void globalPlanThread();
  void localPlanThread();
  void progressThread();
  void safetyThread();
  void visThread();
  void globalPlan(Eigen::VectorXd& cur_pose, Eigen::Vector3d& cur_vel, Eigen::MatrixXd& mesh_V, Eigen::MatrixXi& mesh_F, vector<Eigen::VectorXd>& last_global_path, vector<bool>& last_global_waypt_indicators, bool& succeed);
  void localPlan(Eigen::MatrixXd& mesh_V, Eigen::MatrixXi& mesh_F, vector<Eigen::VectorXd>& g_path, vector<bool>& g_indi, pcl::PointCloud<pcl::PointXYZ>::Ptr& g_hc, pcl::PointCloud<pcl::PointXYZ>::Ptr& g_model, vector<Eigen::VectorXd>& g_prior, vector<Eigen::Vector3d>& g_next_sub_inliers, vector<Eigen::VectorXd>& g_next_sub_vps, const Eigen::VectorXd& local_start, const Eigen::Vector3d vel, const Eigen::Vector3d acc, const Eigen::Vector3d pyd, const Eigen::Vector3d pyd_dot, bool& succeed, bool& feasible);
  void visualization();
  /* Utils */
  void meshCallback(const quadrotor_msgs::Mesh& mesh_msg);
  void triggerPlanning();
  void joyCallback(const sensor_msgs::Joy::ConstPtr& Joy);
  void pauseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg);
  void execTrajCallback(const quadrotor_msgs::EigenVectorArrayConstPtr& msg);
  void odomCallback(const nav_msgs::OdometryConstPtr& msg);
  void gimbalCallback(const geometry_msgs::QuaternionStampedConstPtr& msg);
  void bevCallback(const quadrotor_msgs::BEVBoxConstPtr& msg);

  shared_ptr<SDFMap> mapping_utils_ = nullptr;
  unique_ptr<FCPlanner_PP> path_planner_ = nullptr;
  unique_ptr<TrajGenerator> motion_planner_ = nullptr;
  unique_ptr<PerceptionUtils> percep_utils_ = nullptr;
  unique_ptr<Local_Replan> local_replanner_ = nullptr;
  unique_ptr<PlanningVisualization> vis_utils_ = nullptr;
  /* Tools */
  void stopGlobalPlanningService();
  void stopLocalPlanningService();
  void stopProgressService();
  void stopSafetyService();
  void stopVisService();
  void trajConverter(const Trajectory<7> &pos, const Trajectory<7> &ori, quadrotor_msgs::PolynomialTrajGroup &msg, const ros::Time &cur_stamp, int &traj_id);
  /* Data */
  voxel_map::VoxelMap voxelMap;
  vector<Eigen::VectorXd> last_global_path_, local_path_, local_updated_path_, local_used_g_path_, local_used_g_prior_;
  vector<bool> last_global_waypt_indicators_;
  vector<Eigen::Vector3d> local_used_g_next_sub_inliers_;
  vector<Eigen::VectorXd> local_used_g_next_sub_vps_;
  Eigen::Vector3d min_bound_, max_bound_;
  Eigen::VectorXd vis_global_start_;
  Eigen::MatrixXd vis_mesh_V_;
  Eigen::MatrixXi vis_mesh_F_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_hc_, local_visible_;
  vector<Eigen::VectorXd> exec_traj_;
  nav_msgs::Odometry odom_;
  geometry_msgs::QuaternionStamped gimbal_;
  /* Path */
  ros::Time last_global_time_;
  /* Traj */
  ros::Time last_start_time_, newest_start_time_;
  Trajectory<7> last_pos_traj_, last_orientation_traj_, newest_pos_traj_, newest_orientation_traj_; // orientation_traj_: [yaw (world frame -> NWU), pitch (gimbal frame -> NED), 0]
  double last_duration_, newest_duration_;
  int traj_id_ = 1;
  /* Buffer */
  boost::circular_buffer<Eigen::MatrixXd> buffer_mesh_V_;
  boost::circular_buffer<Eigen::MatrixXi> buffer_mesh_F_;
  boost::circular_buffer<vector<Eigen::VectorXd>> buffer_g_paths_;
  boost::circular_buffer<vector<bool>> buffer_g_waypt_indicators_;
  boost::circular_buffer<vector<Eigen::VectorXd>> buffer_g_prior_paths_;
  boost::circular_buffer<pcl::PointCloud<pcl::PointXYZ>::Ptr> buffer_g_hc_;
  boost::circular_buffer<pcl::PointCloud<pcl::PointXYZ>::Ptr> buffer_g_model_;
  boost::circular_buffer<ros::Time> buffer_start_times_;
  boost::circular_buffer<Trajectory<7>> buffer_pos_trajs_;
  boost::circular_buffer<Trajectory<7>> buffer_ori_trajs_;
  boost::circular_buffer<vector<Eigen::VectorXd>> buffer_last_paths_;
  boost::circular_buffer<vector<bool>> buffer_last_wp_indicators_;
  /* Trigger */
  boost::circular_buffer<GLOBAL_STATE> global_state_;
  boost::circular_buffer<LOCAL_STATE> local_state_;
  vector<string> global_state_str_ = {"GLOBAL_SILENCE", "GLOBAL_PLAN", "GLOBAL_EXEC", "GLOBAL_FINISHED"};
  vector<string> local_state_str_ = {"LOCAL_SILENCE", "LOCAL_PLAN", "LOCAL_EXEC", "LOCAL_FINISHED"};
  std::mutex global_state_mtx_;
  std::mutex local_state_mtx_;
  /* Global Start */
  Eigen::VectorXd cur_pos_;
  Eigen::Vector3d cur_vel_;
  /* Local Start */
  Eigen::VectorXd local_start_;
  Eigen::Vector3d local_start_vel_, local_start_acc_;
  Eigen::Vector3d local_start_pyd_, local_start_pyd_dot_;
  /* ROS Service */
  ros::Subscriber pred_mesh_sub_, joy_sub_, pause_sub_, exec_sub_, odom_sub_, gimbal_sub_, bev_sub_;
  ros::Publisher local_traj_pub_, finish_pub_, brake_pub_, local_replan_path_pub_;
  /* Visulization */
  Eigen::Vector3d temp_min_bound, temp_max_bound;
  pcl::PointCloud<pcl::PointXYZ>::Ptr temp_local_hc;
  vector<Eigen::VectorXd> temp_local_path;
  vector<Eigen::VectorXd> temp_local_updated_path;
  vector<Eigen::VectorXd> temp_local_used_g_path;
  vector<Eigen::VectorXd> temp_local_used_g_prior;
  Eigen::MatrixXd temp_mesh_V;
  Eigen::MatrixXi temp_mesh_F;
}; // namespace flyco

inline void FCPlanner_PP_Node::setPlanStart(Eigen::VectorXd start_pose)
{
  this->cur_pos_ = start_pose;
  this->cur_vel_ = Eigen::Vector3d::Zero();

  return;
}

inline void FCPlanner_PP_Node::setLocalStart(Eigen::VectorXd local_start, Eigen::Vector3d vel, Eigen::Vector3d acc, Eigen::Vector3d pyd, Eigen::Vector3d pyd_dot)
{
  this->local_start_ = local_start;
  this->local_start_vel_ = vel;
  this->local_start_acc_ = acc;
  this->local_start_pyd_ = pyd;
  this->local_start_pyd_dot_ = pyd_dot;

  return;
}

// ! --------------------------- Utils ---------------------------

inline void FCPlanner_PP_Node::meshCallback(const quadrotor_msgs::Mesh& mesh_msg)
{
  Eigen::MatrixXd mesh_V;
  Eigen::MatrixXi mesh_F;
  mesh_V.resize(mesh_msg.vertex_num, 3);
  mesh_F.resize(mesh_msg.face_num, 3);

  for (int i=0; i<(int)mesh_msg.vertex_num; ++i)
  {
    mesh_V(i, 0) = mesh_msg.vertex_matrix.data[i*3];
    mesh_V(i, 1) = mesh_msg.vertex_matrix.data[i*3+1];
    mesh_V(i, 2) = mesh_msg.vertex_matrix.data[i*3+2];
  }

  for (int i=0; i<(int)mesh_msg.face_num; ++i)
  {
    mesh_F(i, 0) = mesh_msg.face_matrix.data[i*3];
    mesh_F(i, 1) = mesh_msg.face_matrix.data[i*3+1];
    mesh_F(i, 2) = mesh_msg.face_matrix.data[i*3+2];
  }

  
  if (!this->global_input_mtx_.try_lock())
  {
    return;
  }
  else
  {
    this->buffer_mesh_V_.push_back(mesh_V);
    this->buffer_mesh_F_.push_back(mesh_F);

    this->global_input_mtx_.unlock();

    // ROS_INFO("mesh_V rows: %d, mesh_V cols: %d", this->buffer_mesh_V_.back().rows(), this->buffer_mesh_V_.back().cols());
    // ROS_INFO("mesh_F rows: %d, mesh_F cols: %d", this->buffer_mesh_F_.back().rows(), this->buffer_mesh_F_.back().cols());
    // ROS_ERROR("Received mesh data. -> %d, %d", this->buffer_mesh_V_.size(), this->buffer_mesh_F_.size());
  }

  return;
}

inline void FCPlanner_PP_Node::triggerPlanning()
{
  if (this->cur_pos_(0) == NEG_INFINITY || this->cur_pos_(1) == NEG_INFINITY || this->cur_pos_(2) == NEG_INFINITY)
  {
    Eigen::Vector3d plan_pos(odom_.pose.pose.position.x, odom_.pose.pose.position.y, odom_.pose.pose.position.z);

    tf::Quaternion tfQuat_body;
    tf::quaternionMsgToTF(odom_.pose.pose.orientation, tfQuat_body);
    tf::Matrix3x3 m_body(tfQuat_body);
    double roll_body, pitch_body, yaw_body;
    m_body.getRPY(roll_body, pitch_body, yaw_body);

    tf::Quaternion tfQuat;
    tf::quaternionMsgToTF(gimbal_.quaternion, tfQuat);
    tf::Matrix3x3 m(tfQuat);
    double roll_gimbal, pitch_gimbal, yaw_gimbal;
    m.getRPY(roll_gimbal, pitch_gimbal, yaw_gimbal);

    Eigen::VectorXd start_pose(5);
    start_pose << plan_pos(0), plan_pos(1), plan_pos(2), pitch_gimbal, yaw_body;

    this->setPlanStart(start_pose);
    this->setLocalStart(start_pose, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  }

  ros::Duration(0.5).sleep();

  ROS_INFO("\033[41;37m[FlyCo]\033[47;31m Successfully select the drone starting flight from \n (%lf, %lf, %lf) in [World->NWU] frame with pitch %lf deg and yaw %lf deg! \033[0m", this->cur_pos_(0), this->cur_pos_(1), this->cur_pos_(2), this->cur_pos_(3)*180.0/M_PI, this->cur_pos_(4)*180.0/M_PI);
  
  ROS_ERROR("Received trigger data.");

  bool receving_global = false;
  bool receving_local = false;
  while (!receving_global && !receving_local)
  {
    if (this->global_state_mtx_.try_lock())
    {
      this->global_state_.push_back(GLOBAL_PLAN);
      this->global_state_mtx_.unlock();
      receving_global = true;
    }
    else
    {
      continue;
    }
    
    if (this->local_state_mtx_.try_lock())
    {
      this->local_state_.push_back(LOCAL_PLAN);
      this->local_state_mtx_.unlock();
      receving_local = true;
    }
    else
    {
      continue;
    }
  }

  return;
}

inline void FCPlanner_PP_Node::joyCallback(const sensor_msgs::Joy::ConstPtr& Joy)
{
  if (!Joy) return;

  constexpr int JOY_RB_BUTTON = 5;
  bool rb_pressed = Joy->buttons.size() > JOY_RB_BUTTON && Joy->buttons[JOY_RB_BUTTON] == true;

  if (rb_pressed && !this->last_rb_pressed_)
  {
    ROS_WARN("\033[43;37m[FlyCo Joy]\033[47;33m RB pressed: trigger planning. \033[0m");
    this->triggerPlanning();
  }

  this->last_rb_pressed_ = rb_pressed;

  return;
}

inline void FCPlanner_PP_Node::pauseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)
{
  ROS_WARN("Received pause data.");

  this->static_state_global_ = true;
  this->static_state_local_ = true;
  
  bool receving_global = false;
  bool receving_local = false; 

  while (!receving_global && !receving_local)
  {
    if (this->global_state_mtx_.try_lock())
    {
      this->global_state_.push_back(GLOBAL_SILENCE);
      this->global_state_mtx_.unlock();
      receving_global = true;
    }
    else
    {
      continue;
    }
    
    if (this->local_state_mtx_.try_lock())
    {
      this->local_state_.push_back(LOCAL_SILENCE);
      this->local_state_mtx_.unlock();
      receving_local = true;
    }
    else
    {
      continue;
    }
  }

  return;
}

inline void FCPlanner_PP_Node::execTrajCallback(const quadrotor_msgs::EigenVectorArrayConstPtr& msg)
{
  const quadrotor_msgs::EigenVectorArray &traj_msg = *msg;

  this->exec_traj_.clear();
  for (const auto& array : traj_msg.vectors)
  {
    Eigen::VectorXd vec(array.data.size());
    for (size_t i = 0; i < array.data.size(); ++i)
    {
      vec[i] = array.data[i];
    }
    this->exec_traj_.push_back(vec);
  }

  return;
}

inline void FCPlanner_PP_Node::odomCallback(const nav_msgs::OdometryConstPtr& msg)
{
  this->odom_ = *msg;

  return;
}

inline void FCPlanner_PP_Node::gimbalCallback(const geometry_msgs::QuaternionStampedConstPtr& msg)
{
  this->gimbal_ = *msg;

  return;
}

inline void FCPlanner_PP_Node::bevCallback(const quadrotor_msgs::BEVBoxConstPtr& msg)
{
  const quadrotor_msgs::BEVBox &bev_msg = *msg;

  this->bev_seg_ = bev_msg.en_bev_box == 1 ? true : false;

  if (!this->bev_seg_) return;

  this->bev_min_x_ = bev_msg.min_x;
  this->bev_max_x_ = bev_msg.max_x;
  this->bev_min_y_ = bev_msg.min_y;
  this->bev_max_y_ = bev_msg.max_y; 

  return;
}

// ! --------------------------- Tools ---------------------------

inline void FCPlanner_PP_Node::stopGlobalPlanningService()
{
  this->global_planning_running_ = 0;

  return;
}

inline void FCPlanner_PP_Node::stopLocalPlanningService()
{
  this->local_planning_running_ = 0;

  return;
}

inline void FCPlanner_PP_Node::stopProgressService()
{
  this->progress_running_ = 0;

  return;
}

inline void FCPlanner_PP_Node::stopSafetyService()
{
  this->safety_running_ = 0;

  return;
}

inline void FCPlanner_PP_Node::stopVisService()
{
  this->vis_running_ = 0;

  return;
}

inline void FCPlanner_PP_Node::trajConverter(const Trajectory<7> &pos, const Trajectory<7> &ori, quadrotor_msgs::PolynomialTrajGroup &msg, const ros::Time &cur_stamp, int &traj_id)
{
  msg.trajectory_id = traj_id;
  msg.header.stamp = cur_stamp;
  msg.action = msg.ACTION_ADD;
  msg.start_time = cur_stamp;

  int pn_pos = pos.getPieceNum();
  for (int i=0; i<pn_pos; ++i)
  {
    quadrotor_msgs::PolynomialMatrix piece_pos;
    piece_pos.num_dim = pos[i].getDim();
    piece_pos.num_order = pos[i].getDegree();
    piece_pos.duration = pos[i].getDuration();
    auto cMat = pos[i].getCoeffMat();
    piece_pos.data.assign(cMat.data(),cMat.data() + cMat.rows()*cMat.cols());
    msg.pos_traj.push_back(piece_pos);
  }

  int pn_ori = ori.getPieceNum();
  for (int i=0; i<pn_ori; ++i)
  {
    quadrotor_msgs::PolynomialMatrix piece_ori;
    piece_ori.num_dim = ori[i].getDim();
    piece_ori.num_order = ori[i].getDegree();
    piece_ori.duration = ori[i].getDuration();
    auto cMat = ori[i].getCoeffMat();
    piece_ori.data.assign(cMat.data(),cMat.data() + cMat.rows()*cMat.cols());
    msg.ori_traj.push_back(piece_ori);
  }

  return;
}

} // namespace flyco

#endif
