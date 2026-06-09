/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Jun. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the main file of asynchronous planning framework in FlyCo.
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

#include "hierarchical_coverage_planner/planner_node.h"

namespace flyco
{

void FCPlanner_PP_Node::init(ros::NodeHandle& nh)
{
  // * Module Initialization
  this->path_planner_.reset(new FCPlanner_PP);
  this->motion_planner_.reset(new TrajGenerator);
  this->local_replanner_.reset(new Local_Replan);
  this->percep_utils_.reset(new PerceptionUtils);
  this->vis_utils_.reset(new PlanningVisualization(nh));

  this->path_planner_->init(nh);
  this->motion_planner_->init(nh);
  this->local_replanner_->init(nh);
  this->percep_utils_->init(nh);

  // * Params Initialization
  nh.param("viewpoint_manager/GroundPos", this->ground_height_, -1.0);
  nh.param("viewpoint_manager/TopPos", this->top_height_, -1.0);
  nh.param("hcopp/TrajSampleFrequency", this->freq_, -1.0);
  nh.param("hcmap/resolution", this->res_, -1.0);
  nh.param("hcplanner/mesh_sample_size", this->sample_size_, -1);
  nh.param("hctraj/drone_radius", this->drone_radius_, -1.0);
  nh.param("hcplanner/suspension_rate", this->suspension_target_, -1.0);
  nh.param("viewpoint_manager/vis_inf", vis_inflation_, 0.5);
  nh.param("fsm/global_planning_fps", this->global_fps_, -1.0);
  nh.param("fsm/local_planning_fps", this->local_fps_, -1.0);
  nh.param("fsm/progress_fps", this->progress_fps_, -1.0);
  nh.param("fsm/safety_fps", this->safety_fps_, -1.0);
  nh.param("fsm/vis_fps", this->vis_fps_, -1.0);
  nh.param("fsm/global_plan_time", this->global_plan_time_, -1.0);
  nh.param("fsm/global_plan_latency", this->global_latency_, -1.0);
  nh.param("fsm/replan_time", this->replan_time_, -1.0);
  nh.param("fsm/replan_full_exec", this->replan_full_exec_time_, -1.0);
  nh.param("fsm/replan_periodic", this->replan_periodic_time_, -1.0);
  nh.param("fsm/replan_proportion", this->replan_proportion_, -1.0);
  nh.param("fsm/safety_horizon", this->safety_horizon_, -1.0);
  nh.param("flyco_planner/enable_inherit", this->en_inherit_, false);
  
  this->bev_seg_ = false;
  this->static_state_global_ = true;
  this->static_state_local_ = true;
  this->last_global_time_ = ros::Time(0.0);
  this->last_start_time_ = ros::Time(0.0);
  this->newest_start_time_ = ros::Time(0.0);
  this->odom_.pose.pose.position.x = -1e3;
  this->odom_.pose.pose.position.y = -1e3;
  this->odom_.pose.pose.position.z = -1e3;
  this->safe_radius_ = this->drone_radius_ + this->res_;

  // * Buffer Initialization
  int size = 10;
  this->buffer_mesh_V_ = boost::circular_buffer<Eigen::MatrixXd>(size);
  this->buffer_mesh_F_ = boost::circular_buffer<Eigen::MatrixXi>(size);
  this->buffer_g_paths_ = boost::circular_buffer<vector<Eigen::VectorXd>>(size);
  this->buffer_g_waypt_indicators_ = boost::circular_buffer<vector<bool>>(size);
  this->buffer_g_prior_paths_ = boost::circular_buffer<vector<Eigen::VectorXd>>(size);
  this->buffer_g_hc_ = boost::circular_buffer<pcl::PointCloud<pcl::PointXYZ>::Ptr>(size);
  this->buffer_g_model_ = boost::circular_buffer<pcl::PointCloud<pcl::PointXYZ>::Ptr>(size);
  this->buffer_start_times_ = boost::circular_buffer<ros::Time>(2);
  this->buffer_pos_trajs_ = boost::circular_buffer<Trajectory<7>>(2);
  this->buffer_ori_trajs_ = boost::circular_buffer<Trajectory<7>>(2);
  this->buffer_last_paths_ = boost::circular_buffer<vector<Eigen::VectorXd>>(2);
  this->buffer_last_wp_indicators_ = boost::circular_buffer<vector<bool>>(2);

  // * Trigger Initialization
  this->global_state_ = boost::circular_buffer<GLOBAL_STATE>(1);
  this->local_state_ = boost::circular_buffer<LOCAL_STATE>(1);
  this->global_state_.push_back(GLOBAL_SILENCE);
  this->local_state_.push_back(LOCAL_SILENCE);

  // * Service Initialization
  this->pred_mesh_sub_ = nh.subscribe("/prediction/predicted_mesh", 1, &FCPlanner_PP_Node::meshCallback, this);
  this->joy_sub_ = nh.subscribe("/joy", 50, &FCPlanner_PP_Node::joyCallback, this);
  this->pause_sub_ = nh.subscribe("/initialpose", 1, &FCPlanner_PP_Node::pauseCallback, this);
  this->exec_sub_ = nh.subscribe("/traj_server/exec_traj_waypts", 1, &FCPlanner_PP_Node::execTrajCallback, this);
  this->odom_sub_ = nh.subscribe("/flyco_planner/body_pose", 1, &FCPlanner_PP_Node::odomCallback, this);
  this->gimbal_sub_ = nh.subscribe("/flyco_planner/gimbal_pose", 1, &FCPlanner_PP_Node::gimbalCallback, this);
  this->bev_sub_ = nh.subscribe("/prediction/bev_box", 1, &FCPlanner_PP_Node::bevCallback, this);
  this->local_traj_pub_ = nh.advertise<quadrotor_msgs::PolynomialTrajGroup>("/local_newest_traj", 1);
  this->finish_pub_ = nh.advertise<std_msgs::Bool>("/flight_end", 1);
  this->brake_pub_ = nh.advertise<std_msgs::Bool>("/emergency_brake", 1);
  this->local_replan_path_pub_ = nh.advertise<quadrotor_msgs::EigenVectorArray>("/local_newest_path", 1);

  // * Thread Initialization
  this->global_planning_rate_ = make_unique<ros::Rate>(ros::Rate(this->global_fps_));
  this->local_planning_rate_ = make_unique<ros::Rate>(ros::Rate(this->local_fps_));
  this->progress_rate_ = make_unique<ros::Rate>(ros::Rate(this->progress_fps_));
  this->safety_rate_ = make_unique<ros::Rate>(ros::Rate(this->safety_fps_));
  this->vis_rate_ = make_unique<ros::Rate>(ros::Rate(this->vis_fps_));

  // * Start Initialization
  this->cur_pos_.resize(5);
  this->cur_pos_ << NEG_INFINITY, NEG_INFINITY, NEG_INFINITY, NEG_INFINITY, NEG_INFINITY;
  this->local_start_.resize(5);
  this->local_start_ << NEG_INFINITY, NEG_INFINITY, NEG_INFINITY, NEG_INFINITY, NEG_INFINITY;

  ROS_INFO("\033[31m[FC-Planner-v2 in FlyCo] Initialized! \033[32m");
}

void FCPlanner_PP_Node::setMap(shared_ptr<SDFMap>& map)
{
  path_planner_->setMap(map);
  this->mapping_utils_ = map;

  return;
}

void FCPlanner_PP_Node::FCPlannerv1(Eigen::MatrixXd& mesh_V, Eigen::MatrixXi& mesh_F)
{  
  // * Visualization
  path_planner_->skeleton_operator->visFlag = true;
  path_planner_->visFlag = true;
  motion_planner_->visFlag = true;

  // * Set Input
  path_planner_->setInput(mesh_V, mesh_F);

  // * Path Planning
  vector<Eigen::VectorXd> last_global_path;
  vector<bool> last_global_indi;
  this->global_succeed_ = path_planner_->plan(this->cur_pos_, this->cur_vel_, last_global_path, last_global_indi, true);
  int path_size = path_planner_->PR.FullPath_.size();
  Eigen::Vector3d now_posi = path_planner_->PR.FullPath_[1].head(3);
  vector<Eigen::Vector3d> wps;
  vector<int> wps_waypt; 
  Eigen::Vector3d inter_wp;
  double hcoppL = (path_planner_->PR.FullPath_[2].head(3)-now_posi).norm();
  int wpID = 0;
  for (int i=2; i<(int)path_size; ++i)
  {
    inter_wp = path_planner_->PR.FullPath_[i].head(3);
    if (i<path_size-1)
      hcoppL += (path_planner_->PR.FullPath_[i+1].head(3)-path_planner_->PR.FullPath_[i].head(3)).norm();
    wps.push_back(inter_wp);
    if (path_planner_->PR.waypoints_indicators_[i] == true)
      wps_waypt.push_back(wpID);
    wpID++;
  }
  vector<bool> wps_waypt_indicators(wps.size(), false);
  for (int w=0; w<(int)wps_waypt.size(); ++w)
    wps_waypt_indicators[wps_waypt[w]] = true;
  
  vector<double> pitchs;
  vector<double> yaws;
  for (int i=1; i<path_size; ++i)
    yaws.push_back(path_planner_->PR.FullPath_[i].tail(1)(0));
  if ((int)path_planner_->PR.FullPath_[0].size() == 5)
  {
    for (int i=1; i<path_size; ++i)
      pitchs.push_back(path_planner_->PR.FullPath_[i](3));
  }

  ROS_INFO("\033[31m[PathAnalyzer] full path size = %d. \033[32m", (int)path_planner_->PR.FullPath_.size());
  ROS_INFO("\033[31m[PathAnalyzer] all viewpoints quantity = %d. \033[32m", path_planner_->viewpointNum);
  ROS_INFO("\033[31m[PathAnalyzer] all waypoints quantity = %d. \033[32m", (int)wps_waypt.size());
  ROS_INFO("\033[31m[PathAnalyzer] path coverage rate = %lf %%. \033[32m", path_planner_->coverage*100.0);
  ROS_INFO("\033[31m[PathAnalyzer] path length = %lf m.\033[32m", hcoppL);
  ROS_INFO("\033[31m[PathAnalyzer] system computation latency = %lf ms.\033[32m", path_planner_->plan_time);

  // * Prepare voxelMap
  double sfc_prog = path_planner_->corridorProgress;
  voxelMap = voxel_map::VoxelMap(path_planner_->xyz, path_planner_->offset, res_);
  for (int i = 0; i < (int)path_planner_->skeleton_operator->P.vmap_pts_->points.size(); ++i)
  {
    pcl::PointXYZ pt = path_planner_->skeleton_operator->P.vmap_pts_->points[i];
    voxelMap.setOccupied(Eigen::Vector3d(pt.x, pt.y, pt.z));
  }
  // most time-consuming
  voxelMap.dilate(std::ceil(drone_radius_ / voxelMap.getScale()));

  // * Motion Planning 
  motion_planner_->HCTraj(now_posi, wps, wps_waypt_indicators, pitchs, yaws, voxelMap, sfc_prog, 1.3, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  ROS_INFO("\033[35m[Traj] --- <Trajectory Generation finished> --- \033[35m");

  // * Trajectory Coverage Evaluation
  ROS_INFO("\033[36m[CoverageAnalyzer] Wait for CoverageAnalyzer...... \033[32m");
  motion_planner_->outputTraj(freq_);
  double coverageRate = path_planner_->trajCoverageEval(motion_planner_->TrajPose);
  motion_planner_->outputCloud(path_planner_->visible_group, freq_);
  ROS_INFO("\033[36m[CoverageAnalyzer] FC-Planner-v2 trajectory coverage rate = %lf %%. \033[32m", coverageRate*100.0);

  return;
}

void FCPlanner_PP_Node::startAllService()
{
  this->global_planning_thread_ = std::thread(&FCPlanner_PP_Node::globalPlanThread, this);
  this->local_planning_thread_ = std::thread(&FCPlanner_PP_Node::localPlanThread, this);
  this->progress_thread_ = std::thread(&FCPlanner_PP_Node::progressThread, this);
  this->safety_thread_ = std::thread(&FCPlanner_PP_Node::safetyThread, this);
  this->vis_thread_ = std::thread(&FCPlanner_PP_Node::visThread, this);

  ROS_INFO("\033[31m[FlyCo] All services started! \033[32m");

  return;
}

void FCPlanner_PP_Node::stopAllService()
{  
  if (this->global_planning_thread_.joinable())
  {
    this->stopGlobalPlanningService();
    this->global_planning_thread_.join();
  }

  if (this->local_planning_thread_.joinable())
  {
    this->stopLocalPlanningService();
    this->local_planning_thread_.join();
  }

  if (this->progress_thread_.joinable())
  {
    this->stopProgressService();
    this->progress_thread_.join();
  }

  if (this->safety_thread_.joinable())
  {
    this->stopSafetyService();
    this->safety_thread_.join();
  }

  if (this->vis_thread_.joinable())
  {
    this->stopVisService();
    this->vis_thread_.join();
  }

  return;
}

/*
 * @brief: Global Planning Thread (Timer) : plan -> sleep -> plan -> sleep -> ...
 */
void FCPlanner_PP_Node::globalPlanThread()
{
  Trajectory<7> last_pos_traj, last_ori_traj;
  Trajectory<7> new_pos_traj, new_ori_traj;
  ros::Time last_traj_start_time, new_traj_start_time;
  vector<Eigen::VectorXd> new_path, last_path;
  vector<bool> new_indi, last_indi;
  bool global_finish = false;
  bool global_continue = true;
  ros::Time cur_time;
  Eigen::MatrixXd mesh_V;
  Eigen::MatrixXi mesh_F;

  while (this->global_planning_running_)
  {
    GLOBAL_STATE global_state;
    // * Read Global State
    if (this->global_state_mtx_.try_lock())
    {
      if (this->global_state_.empty())
      {
        this->global_state_mtx_.unlock();
        this->global_planning_rate_->sleep();
        continue;
      }
      else
      {
        global_state = this->global_state_.back();
        this->global_state_mtx_.unlock();
      }
    }
    else
    {
      this->global_planning_rate_->sleep();
      continue;
    }

    ROS_INFO_STREAM_THROTTLE(2.0, "\033[1m\033[37m\033[41m[GLOBAL_PLANNING_THREAD] : \033[7m\033[37m\033[41m" << this->global_state_str_[int(global_state)].c_str());

    switch (global_state)
    {
      case GLOBAL_SILENCE:
      {
        this->global_planning_rate_->sleep();
        break;
      }

      case GLOBAL_FINISHED:
      {
        ROS_INFO_THROTTLE(1.0, "\033[1m\033[30m\033[43m[FlyCo] : ---------------------------- <Flight Finished> ---------------------------- \033[7m\033[30m\033[43m");
        
        this->global_planning_rate_->sleep();
        break;
      }

      case GLOBAL_PLAN:
      {
        ROS_ERROR("global finish: %d", (int)global_finish);

        if (global_finish == false)
        {

        // * Input Interface
        if (this->global_input_mtx_.try_lock())
        {
          if (this->buffer_mesh_V_.empty() || this->buffer_mesh_F_.empty())
          {
            this->global_input_mtx_.unlock();
            this->global_planning_rate_->sleep();
            continue;
          }
          else
          {
            mesh_V = this->buffer_mesh_V_.back();
            mesh_F = this->buffer_mesh_F_.back();
            this->global_input_mtx_.unlock();
          }
        }
        else
        {
          this->global_planning_rate_->sleep();
          continue;
        }

        // * Determine Plan Start
        if (this->static_state_global_ == false)
        {
          if (this->local_output_mtx_.try_lock())
          {
            if (this->buffer_pos_trajs_.empty() || this->buffer_ori_trajs_.empty() || this->buffer_start_times_.empty() || this->buffer_last_paths_.empty() || this->buffer_last_wp_indicators_.empty())
            {
              this->local_output_mtx_.unlock();
              this->global_planning_rate_->sleep();
              continue;
            }
            else
            {
              if ((int)this->buffer_start_times_.size() == 1)
              {
                new_pos_traj = this->buffer_pos_trajs_.back();
                new_ori_traj = this->buffer_ori_trajs_.back();
                new_traj_start_time = this->buffer_start_times_.back();
                last_traj_start_time = ros::Time(0.0);
                new_path = this->buffer_last_paths_.back();
                new_indi = this->buffer_last_wp_indicators_.back();
              }

              if ((int)this->buffer_start_times_.size() == 2)
              {
                last_pos_traj = this->buffer_pos_trajs_.front();
                last_ori_traj = this->buffer_ori_trajs_.front();
                new_pos_traj = this->buffer_pos_trajs_.back();
                new_ori_traj = this->buffer_ori_trajs_.back();
                last_traj_start_time = this->buffer_start_times_.front();
                new_traj_start_time = this->buffer_start_times_.back();
                last_path = this->buffer_last_paths_.front();
                last_indi = this->buffer_last_wp_indicators_.front();
                new_path = this->buffer_last_paths_.back();
                new_indi = this->buffer_last_wp_indicators_.back();
              }
              this->local_output_mtx_.unlock();
            }
          }
          else
          {
            this->global_planning_rate_->sleep();
            continue;
          }

          // ros::Time t_cur = ros::Time::now() + ros::Duration(this->global_latency_); // ! future time
          ros::Time t_cur = ros::Time::now();
          Trajectory<7> pos_traj, ori_traj;
          vector<Eigen::VectorXd> r_path;
          vector<bool> r_indi;
          double t_span = 0.0, duration = -1.0;

          double t_old = (t_cur - last_traj_start_time).toSec();
          double t_new = (t_cur - new_traj_start_time).toSec();
          if (t_old > 0 && t_new < 0)
          {
            pos_traj = last_pos_traj;
            ori_traj = last_ori_traj;
            // r_path = last_path;
            // r_indi = last_indi;
            r_path = new_path;
            r_indi = new_indi;
            t_span = t_old;
          }
          else if (t_new >= 0)
          {
            pos_traj = new_pos_traj;
            ori_traj = new_ori_traj;
            r_path = new_path;
            r_indi = new_indi;
            t_span = t_new;
          }
          duration = pos_traj.getTotalDuration();

          if (t_span < duration)
          {
            Eigen::Vector3d pos = pos_traj.getPos(t_span);
            double pitch = ori_traj.getPitch(t_span);
            double yaw = ori_traj.getYaw(t_span);
            this->cur_pos_ << pos(0), pos(1), pos(2), pitch, yaw;
            this->cur_vel_ = pos_traj.getVel(t_span);
          }
          else
          {
            t_span = duration;

            Eigen::Vector3d pos = pos_traj.getPos(t_span);
            double pitch = ori_traj.getPitch(t_span);
            double yaw = ori_traj.getYaw(t_span);
            this->cur_pos_ << pos(0), pos(1), pos(2), pitch, yaw;
            this->cur_vel_ = pos_traj.getVel(t_span);
          }

          this->last_global_path_ = r_path;
          this->last_global_waypt_indicators_ = r_indi;

        }
        else
        {
          if (this->en_inherit_)
          {
            this->last_global_path_.clear();
            this->last_global_waypt_indicators_.clear();

            string filename = ros::package::getPath("flyco_planner_manager") + "/inherit_data/local_path.yaml";
            YAML::Node path_config = YAML::LoadFile(filename);
            for (const auto& point : path_config["local_path"])
            {
              Eigen::VectorXd wp(5);
              wp << point[0].as<double>(), point[1].as<double>(), point[2].as<double>(), point[3].as<double>(), point[4].as<double>();
              bool indi = point[5].as<double>() > 0.0 ? true : false;

              this->last_global_path_.push_back(wp);
              this->last_global_waypt_indicators_.push_back(indi);
            }
          }
        }

        // * Global Path Planning
        this->globalPlan(this->cur_pos_, this->cur_vel_, mesh_V, mesh_F, this->last_global_path_, this->last_global_waypt_indicators_, global_continue);
        last_path.clear();
        this->last_global_time_ = ros::Time::now();

        // * Flight Suspension Condition
        if (global_continue == false)
        { 
          if (this->global_state_mtx_.try_lock())
          {
            if (this->local_state_mtx_.try_lock())
            {
              this->global_state_.push_back(GLOBAL_FINISHED);
              this->local_state_.push_back(LOCAL_FINISHED);
              this->local_state_mtx_.unlock();
              this->global_state_mtx_.unlock();

              std_msgs::Bool finish_singal;
              finish_singal.data = true;
              this->finish_pub_.publish(finish_singal); 
            }
            else
            {
              this->global_state_mtx_.unlock();
            }
          }

          this->global_planning_rate_->sleep();
          continue;
        }

        global_finish = true;
        this->static_state_global_ = false;
        }

        // * Output Interface
        if (!this->g_l_intersec_mtx_.try_lock())
        {
          ROS_ERROR("Global Local Intersection Mutex is locked!");
          cout << "current global finish: " << global_finish << endl;
          this->global_planning_rate_->sleep();
          continue;
        }
        else
        {
          buffer_g_paths_.push_back(path_planner_->PR.FullPath_);
          buffer_g_waypt_indicators_.push_back(path_planner_->PR.waypoints_indicators_);
          buffer_g_prior_paths_.push_back(path_planner_->PR.prior_path);
          buffer_g_hc_.push_back(path_planner_->PR.cur_hc_map_);
          buffer_g_model_.push_back(path_planner_->PR.uncovered_model_local_);
          this->g_l_intersec_mtx_.unlock();
        }

        if (this->vis_mtx_.try_lock())
        {
          this->vis_global_start_ = this->cur_pos_;
          this->vis_mesh_V_ = mesh_V;
          this->vis_mesh_F_ = mesh_F;

          this->vis_mtx_.unlock();
        }

        if (this->global_state_mtx_.try_lock())
        {
          this->global_state_.push_back(GLOBAL_EXEC);
          this->global_state_mtx_.unlock();
          global_finish = false;
        }

        this->global_planning_rate_->sleep();
        break;
      }

      case GLOBAL_EXEC:
      {
        cur_time = ros::Time::now();
        double t_cur = (cur_time - this->last_global_time_).toSec();

        if (t_cur > this->global_plan_time_)
        {
          if (this->global_state_mtx_.try_lock())
          {
            this->global_state_.push_back(GLOBAL_PLAN);
            this->global_state_mtx_.unlock();
            ROS_WARN("++++++++++++ Global replan due to periodic call! ++++++++++++");
          }
        }
        
        this->global_planning_rate_->sleep();
        break;
      }
    }
  }

  return;
}

/*
 * @brief: Local Planning Thread (Timer) : silence -> plan (blockage) -> silence -> silence -> ...
 */
void FCPlanner_PP_Node::localPlanThread()
{
  // ? useful
  ros::Time cur_time;
  ros::Time t_plan;
  bool local_finish = false;
  ros::Time local_traj_start_time;
  quadrotor_msgs::PolynomialTrajGroup msg;
  vector<Eigen::VectorXd> local_replan_path;
  vector<bool> local_replan_indi;
  bool local_continue = true, local_feasible = true;

  while (this->local_planning_running_)
  {
    LOCAL_STATE local_state;
    // * Read Local State
    if (this->local_state_mtx_.try_lock())
    {
      if (this->local_state_.empty())
      {
        this->local_state_mtx_.unlock();
        this->local_planning_rate_->sleep();
        continue;
      }
      else
      {
        local_state = this->local_state_.back();
        this->local_state_mtx_.unlock();
      }
    }
    else
    {
      this->local_planning_rate_->sleep();
      continue;
    }

    ROS_INFO_STREAM_THROTTLE(2.0, "\033[1m\033[37m\033[44m[LOCAL_REPLANNING_THREAD] : \033[7m\033[37m\033[44m" << this->local_state_str_[int(local_state)].c_str());
    
    switch (local_state)
    {
      case LOCAL_SILENCE:
      {        
        this->local_planning_rate_->sleep();
        break;
      }

      case LOCAL_FINISHED:
      {
        this->local_planning_rate_->sleep();
        break;
      }

      case LOCAL_PLAN:
      {        
        if (local_finish == false)
        {

          t_plan = ros::Time::now() + ros::Duration(this->replan_time_);
          if (this->static_state_local_ == false)
          {          
            double t_old = (t_plan - this->last_start_time_).toSec();
            double t_new = (t_plan - this->newest_start_time_).toSec();

            if (t_old > 0 && t_new < 0)
            {
              Eigen::Vector3d plan_pos = this->last_pos_traj_.getPos(t_old);
              double plan_pitch = this->last_orientation_traj_.getPitch(t_old);
              double plan_yaw = this->last_orientation_traj_.getYaw(t_old);
              this->local_start_.resize(5);
              this->local_start_ << plan_pos(0), plan_pos(1), plan_pos(2), plan_pitch, plan_yaw;
              this->local_start_vel_ = this->last_pos_traj_.getVel(t_old);
              this->local_start_acc_ = this->last_pos_traj_.getAcc(t_old);

              double pitch_vel = this->last_orientation_traj_.getPitchd(t_old);
              double yaw_vel = this->last_orientation_traj_.getYawd(t_old);
              this->local_start_pyd_ << pitch_vel, yaw_vel, 0.0;
              double pitch_acc = this->last_orientation_traj_.getPitchdd(t_old);
              double yaw_acc = this->last_orientation_traj_.getYawdd(t_old);
              this->local_start_pyd_dot_ << pitch_acc, yaw_acc, 0.0;
            }
            else if (t_new >= 0)
            {
              Eigen::Vector3d plan_pos = this->newest_pos_traj_.getPos(t_new);
              double plan_pitch = this->newest_orientation_traj_.getPitch(t_new);
              double plan_yaw = this->newest_orientation_traj_.getYaw(t_new);
              this->local_start_.resize(5);
              this->local_start_ << plan_pos(0), plan_pos(1), plan_pos(2), plan_pitch, plan_yaw;
              this->local_start_vel_ = this->newest_pos_traj_.getVel(t_new);
              this->local_start_acc_ = this->newest_pos_traj_.getAcc(t_new);

              double pitch_vel = this->newest_orientation_traj_.getPitchd(t_new);
              double yaw_vel = this->newest_orientation_traj_.getYawd(t_new);
              this->local_start_pyd_ << pitch_vel, yaw_vel, 0.0;
              double pitch_acc = this->newest_orientation_traj_.getPitchdd(t_new);
              double yaw_acc = this->newest_orientation_traj_.getYawdd(t_new);
              this->local_start_pyd_dot_ << pitch_acc, yaw_acc, 0.0;
            }
          }

          Eigen::MatrixXd mesh_V;
          Eigen::MatrixXi mesh_F;
          vector<Eigen::VectorXd> global_path;
          vector<bool> global_waypt_indicators;
          vector<Eigen::VectorXd> global_prior_path;
          pcl::PointCloud<pcl::PointXYZ>::Ptr global_hc (new pcl::PointCloud<pcl::PointXYZ>);
          pcl::PointCloud<pcl::PointXYZ>::Ptr global_model (new pcl::PointCloud<pcl::PointXYZ>);
          vector<Eigen::Vector3d> global_next_sub_inliers;
          vector<Eigen::VectorXd> global_next_sub_vps;

          // * Input Interface
          if (this->g_l_intersec_mtx_.try_lock())
          {
            if (this->global_input_mtx_.try_lock())
            {
              if (this->buffer_mesh_V_.empty() || this->buffer_mesh_F_.empty() || this->buffer_g_paths_.empty() || this->buffer_g_waypt_indicators_.empty() || this->buffer_g_prior_paths_.empty() || this->buffer_g_hc_.empty())
              {
                this->global_input_mtx_.unlock();
                this->g_l_intersec_mtx_.unlock();
                this->local_planning_rate_->sleep();
                continue;
              }
              else
              {
                mesh_V = this->buffer_mesh_V_.back();
                mesh_F = this->buffer_mesh_F_.back();
                global_path = this->buffer_g_paths_.back();
                global_waypt_indicators = this->buffer_g_waypt_indicators_.back();
                global_prior_path = this->buffer_g_prior_paths_.back();
                *global_hc = *this->buffer_g_hc_.back();
                *global_model = *this->buffer_g_model_.back();

                this->global_input_mtx_.unlock();
                this->g_l_intersec_mtx_.unlock();
              }
            }
            else
            {
              this->g_l_intersec_mtx_.unlock();

              this->local_planning_rate_->sleep();
              continue;
            }
          }

          // * Local Motion Replanning
          this->localPlan(mesh_V, mesh_F, global_path, global_waypt_indicators, global_hc, global_model, global_prior_path, global_next_sub_inliers, global_next_sub_vps, this->local_start_, this->local_start_vel_, this->local_start_acc_, this->local_start_pyd_, this->local_start_pyd_dot_, local_continue, local_feasible);

          if (!local_feasible)
          {
            this->local_planning_rate_->sleep();
            continue;
          }

          // * Flight Suspension Condition
          if (local_continue == false)
          {            
            if (this->global_state_mtx_.try_lock())
            {
              if (this->local_state_mtx_.try_lock())
              {
                this->global_state_.push_back(GLOBAL_FINISHED);
                this->local_state_.push_back(LOCAL_FINISHED);
                this->local_state_mtx_.unlock();
                this->global_state_mtx_.unlock();

                std_msgs::Bool finish_singal;
                finish_singal.data = true;
                this->finish_pub_.publish(finish_singal);
              }
              else
              {
                this->global_state_mtx_.unlock();
              }
            }

            this->local_planning_rate_->sleep();
            continue;
          }

          Trajectory<7> pos_traj, ori_traj;
          this->local_replanner_->getLocalTraj(pos_traj, ori_traj);
          this->local_replanner_->getReplanPathWithIndi(local_replan_path, local_replan_indi);

          ROS_WARN("------------ Local Path->Traj Continuity Check ------------");
          if ((this->local_start_.head(3) - pos_traj.getPos(0)).norm() > 1e-3)
          {
            cout << "input start : " << this->local_start_.head(3).transpose() << endl;
            cout << "output start : " << pos_traj.getPos(0).transpose() << endl;
            ROS_ERROR("------------ Local Path->Traj Start Point Error ------------");
          }
          
          // * Output Interface
          if (!this->local_output_mtx_.try_lock())
          {
            this->local_planning_rate_->sleep();
            continue;
          }
          else
          {
            this->buffer_pos_trajs_.push_back(pos_traj);
            this->buffer_ori_trajs_.push_back(ori_traj);
            this->buffer_start_times_.push_back(t_plan);
            this->buffer_last_paths_.push_back(local_replan_path);
            this->buffer_last_wp_indicators_.push_back(local_replan_indi);
            this->local_output_mtx_.unlock();
            
            this->last_pos_traj_.clear();
            this->last_orientation_traj_.clear();
            this->last_pos_traj_ = this->newest_pos_traj_;
            this->last_orientation_traj_ = this->newest_orientation_traj_;
            this->newest_pos_traj_.clear();
            this->newest_orientation_traj_.clear();
            this->newest_pos_traj_ = pos_traj;
            this->newest_orientation_traj_ = ori_traj;
            this->last_duration_ = this->last_pos_traj_.getTotalDuration();
            this->newest_duration_ = this->newest_pos_traj_.getTotalDuration();

            if (this->static_state_local_ == true)
            {
              this->newest_start_time_ = t_plan;
            }
            else
            {
              this->last_start_time_ = this->newest_start_time_;
              this->newest_start_time_ = t_plan;

              ROS_WARN("------------ Local Trajectory Continuity Check ------------");
              double t_bound = (this->newest_start_time_ - this->last_start_time_).toSec();
              Eigen::Vector3d old_pos = this->last_pos_traj_.getPos(t_bound);
              Eigen::Vector3d new_pos = this->newest_pos_traj_.getPos(0);
              if ((old_pos - new_pos).norm() > 1e-3)
              {
                cout << "old pos : " << old_pos.transpose() << endl;
                cout << "new pos : " << new_pos.transpose() << endl;
                ROS_ERROR("------------ Local Trajectory Bound Error ------------");
              }
            }

            quadrotor_msgs::EigenVectorArray path_msg;
            path_msg.header.stamp = ros::Time::now();

            for (int l=0; l<(int)local_replan_path.size(); ++l)
            {
              std_msgs::Float64MultiArray array;
              for (int v=0; v<(int)local_replan_path[l].size(); ++v)
              {
                array.data.push_back(local_replan_path[l](v));
              }
              double indi = local_replan_indi[l] == true ? 1.0 : 0.0;
              array.data.push_back(indi);
              path_msg.vectors.push_back(array);
            }

            this->local_replan_path_pub_.publish(path_msg);
            
            local_traj_start_time = this->newest_start_time_;

            quadrotor_msgs::PolynomialTrajGroup temp_msg;
            this->trajConverter(this->newest_pos_traj_, this->newest_orientation_traj_, temp_msg, local_traj_start_time, this->traj_id_);

            msg = temp_msg;
            this->traj_id_++;
          }

          local_finish = true;
          this->static_state_local_ = false;

        }

        if (this->local_state_mtx_.try_lock())
        {
          this->local_state_.push_back(LOCAL_EXEC);
          this->local_state_mtx_.unlock();
          local_finish = false;
          this->local_traj_pub_.publish(msg);
        }

        this->local_planning_rate_->sleep();
        break;
      }
    
      case LOCAL_EXEC:
      {
        // * Replan Condition -> 1) periodic 2) traj fully executed
        bool pass = false, new_traj = false; 

        cur_time = ros::Time::now();
        double t_old = (cur_time - this->last_start_time_).toSec();
        double t_new = (cur_time - this->newest_start_time_).toSec();
        double time_to_end = 0.0;
        double t_cur = 0.0;
        double duration = 0.0;

        if (t_new >= 0)
        {
          new_traj = true;

          t_cur = t_new;
          duration = this->newest_duration_;
          time_to_end = duration - t_cur;
          if (time_to_end < this->replan_full_exec_time_)
          {
            if (this->local_state_mtx_.try_lock())
            {
              this->local_state_.push_back(LOCAL_PLAN);
              this->local_state_mtx_.unlock();
              pass = true;
              ROS_WARN("------------ Replan due to full execution! ------------");
            }
          }
        }
        else if (t_old > 0 && t_new < 0)
        {
          t_cur = t_old;
          duration = this->last_duration_;
        }

        if (pass == false && new_traj == true)
        {
          if (t_cur > this->replan_proportion_ * duration)
          {
            if (this->local_state_mtx_.try_lock())
            {
              this->local_state_.push_back(LOCAL_PLAN);
              this->local_state_mtx_.unlock();
              ROS_WARN("------------ Replan due to periodic call! ------------");
            }
          }
        }

        this->local_planning_rate_->sleep();
        break;
      }
    }
  }

  return;
}

void FCPlanner_PP_Node::progressThread()
{
  RTCScene cur_scene = nullptr;
  double coverage_rate = 0.0;
  int total_points = 0, covered_points = 0;
  bool finish = false;

  while (this->progress_running_)
  {
    ROS_INFO_STREAM_THROTTLE(1.0, "\033[1m\033[37m\033[46m[PROGRESS_THREAD] Flight Coverage : \033[7m\033[37m\033[46m" << std::fixed << std::setprecision(2) << coverage_rate << " %, total: " << total_points << ", covered: " << covered_points);
    
    if (cur_scene != nullptr)
    {
      rtcReleaseScene(cur_scene);
      cur_scene = nullptr;
    }

    Eigen::MatrixXd mesh_V;
    Eigen::MatrixXi mesh_F;

    if (finish == false)
    {

    // * Input Interface
    if (this->global_input_mtx_.try_lock())
    {
      if (this->buffer_mesh_V_.empty() || this->buffer_mesh_F_.empty())
      {
        this->global_input_mtx_.unlock();
        this->global_planning_rate_->sleep();
        continue;
      }
      else
      {
        mesh_V = this->buffer_mesh_V_.back();
        mesh_F = this->buffer_mesh_F_.back();
        this->global_input_mtx_.unlock();
      }
    }
    else
    {
      this->progress_rate_->sleep();
      continue;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cur_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cur_filter_cloud (new pcl::PointCloud<pcl::PointXYZ>);

    // * Matrix to Mesh
    visibility_st::mesh2scene(mesh_V, mesh_F, cur_scene);

    // * Matrix to PointCloud
    visibility_st::mesh2pcd(mesh_V, mesh_F, this->res_, cur_cloud);
    if ((int)cur_cloud->points.size() > this->sample_size_)
    {
      pcl::RandomSample<pcl::PointXYZ> rs;
      rs.setInputCloud(cur_cloud);
      rs.setSample(this->sample_size_);
      rs.filter(*cur_cloud);
    }

    for (int i=0; i<(int)cur_cloud->points.size(); ++i)
    {
      if (cur_cloud->points[i].z > this->ground_height_ && cur_cloud->points[i].z < this->top_height_)
        cur_filter_cloud->points.push_back(cur_cloud->points[i]);
    }

    // * Calculate Current Coverage Rate
    vector<bool> coverage_map((int)cur_filter_cloud->points.size(), false);

    // * If bev segmentation is enabled by human's intention
    if (this->bev_seg_)
    {
      for (int i=0; i<(int)cur_filter_cloud->points.size(); ++i)
      {
        pcl::PointXYZ pt = cur_filter_cloud->points[i];
        if (pt.x > this->bev_max_x_ || pt.x < this->bev_min_x_ || pt.y > this->bev_max_y_ || pt.y < this->bev_min_y_)
          coverage_map[i] = true;
      }
    }

    for (int i=0; i<(int)cur_filter_cloud->points.size(); ++i)
    {
      pcl::PointXYZ pt = cur_filter_cloud->points[i];
      for (int j=0; j<(int)this->exec_traj_.size(); ++j)
      {
        Eigen::Vector3d vp_pos = this->exec_traj_[j].head(3);
        double vp_pitch = this->exec_traj_[j](3), vp_yaw = this->exec_traj_[j](4);
        Eigen::Vector3d pt_pos(pt.x, pt.y, pt.z);

        this->percep_utils_->setPose_PY(vp_pos, vp_pitch, vp_yaw);
        if (this->percep_utils_->insideFOV(pt_pos) == false)
          continue;
        
        bool visible = visibility_st::visibilityCheck(vp_pos, pt_pos, this->vis_inflation_, cur_scene);
        if (visible == true)
        {
          coverage_map[i] = true;
          break;
        }
      }
    }

    int cover_count = count(coverage_map.begin(), coverage_map.end(), true);
    coverage_rate = 100.0 * (double)cover_count / (double)cur_filter_cloud->points.size();
    total_points = (int)cur_filter_cloud->points.size();
    covered_points = cover_count;
  
    }

    if (coverage_rate > this->suspension_target_)
    {
      finish = true;
      // * Flight Suspension Condition
      if (this->global_state_mtx_.try_lock())
      {
        if (this->local_state_mtx_.try_lock())
        {
          this->global_state_.push_back(GLOBAL_FINISHED);
          this->local_state_.push_back(LOCAL_FINISHED);
          this->local_state_mtx_.unlock();
          this->global_state_mtx_.unlock();

          std_msgs::Bool finish_singal;
          finish_singal.data = true;
          this->finish_pub_.publish(finish_singal);
        }
        else
        {
          this->global_state_mtx_.unlock();
        }
      }
    }

    this->progress_rate_->sleep();
  }

  return;
}

void FCPlanner_PP_Node::safetyThread()
{
  Trajectory<7> last_pos_traj, last_ori_traj;
  Trajectory<7> new_pos_traj, new_ori_traj;
  ros::Time last_traj_start_time = ros::Time(0.0), new_traj_start_time = ros::Time(0.0);
  bool safe = true, info_pass = false;
  vector<Eigen::VectorXd> empty_path = {};
  vector<bool> empty_indi = {};
  
  while (this->safety_running_)
  {
    ROS_INFO_STREAM_THROTTLE(2.0, "\033[1m\033[37m\033[43m[SAFETY_THREAD] : \033[7m\033[37m\033[43m" << "Running......");

    LOCAL_STATE temp_local_state;

    if (this->local_state_mtx_.try_lock())
    {
      if (this->local_state_.empty())
      {
        this->local_state_mtx_.unlock();
        this->safety_rate_->sleep();
        continue;
      }
      else
      {
        temp_local_state = this->local_state_.back();
        this->local_state_mtx_.unlock();
      }
    }
    else
    {
      this->safety_rate_->sleep();
      continue;
    }

    if (temp_local_state == LOCAL_STATE::LOCAL_EXEC)
    {
      // * Input Interface
      if (this->local_output_mtx_.try_lock())
      {
        if (this->buffer_pos_trajs_.empty() || this->buffer_ori_trajs_.empty() || this->buffer_start_times_.empty())
        {
          this->local_output_mtx_.unlock();
          this->safety_rate_->sleep();
          continue;
        }
        else
        {
          if ((int)this->buffer_start_times_.size() == 1)
          {
            new_pos_traj = this->buffer_pos_trajs_.back();
            new_ori_traj = this->buffer_ori_trajs_.back();
            new_traj_start_time = this->buffer_start_times_.back();
            last_traj_start_time = ros::Time(0.0);
          }

          if ((int)this->buffer_start_times_.size() == 2)
          {
            last_pos_traj = this->buffer_pos_trajs_.front();
            last_ori_traj = this->buffer_ori_trajs_.front();
            new_pos_traj = this->buffer_pos_trajs_.back();
            new_ori_traj = this->buffer_ori_trajs_.back();
            last_traj_start_time = this->buffer_start_times_.front();
            new_traj_start_time = this->buffer_start_times_.back();
          }
          this->local_output_mtx_.unlock();
        }
      }
      else
      {
        this->safety_rate_->sleep();
        continue;
      }

      // * Traj Safety Check 
      if (info_pass == false)
      {
        double t_now = (ros::Time::now() - new_traj_start_time).toSec();
        double duration = new_pos_traj.getTotalDuration();
        if (t_now >= 0)
        {
          Eigen::Vector3d cur_pos = new_pos_traj.getPos(t_now);
          double radius = 0.0;
          Eigen::Vector3d future_pos;
          double future_t = 0.02;
          while (radius < this->safety_horizon_ && t_now + future_t < duration)
          {
            future_pos = new_pos_traj.getPos(t_now + future_t);
            if (this->mapping_utils_->getSafety(future_pos, this->drone_radius_, this->drone_radius_, 0.5*this->drone_radius_) == false)
            {
              ROS_ERROR("collision is detected at (%lf, %lf, %lf)!!!", future_pos(0), future_pos(1), future_pos(2));
              safe = false;
              info_pass = true;
              break;
            }
            radius = (future_pos - cur_pos).norm();
            future_t += 0.02;
          }
        }
      }

      if (!safe)
      {
        ROS_ERROR("------------ Replan due to collision detected! ------------");        
        if (this->global_state_mtx_.try_lock())
        {
          if (this->local_state_mtx_.try_lock())
          {
            if (this->local_output_mtx_.try_lock())
            {
              // this->global_state_.push_back(GLOBAL_PLAN);
              this->local_state_.push_back(LOCAL_PLAN);
              // this->buffer_last_paths_.push_back(empty_path);
              // this->buffer_last_wp_indicators_.push_back(empty_indi);
              this->local_output_mtx_.unlock();
              this->local_state_mtx_.unlock();
              this->global_state_mtx_.unlock();
              info_pass = false;
              safe = true;
            }
            else
            {
              this->local_state_mtx_.unlock();
              this->global_state_mtx_.unlock();
            }
          }
          else
          {
            this->global_state_mtx_.unlock();
          }
        }
      }
    }

    this->safety_rate_->sleep();
  }

  return;
}

void FCPlanner_PP_Node::visThread()
{
  this->temp_local_hc.reset(new pcl::PointCloud<pcl::PointXYZ>);
  
  while(this->vis_running_)
  {
    ROS_INFO_STREAM_THROTTLE(2.0, "\033[1m\033[37m\033[42m[VISUALIZATION_THREAD] : \033[7m\033[37m\033[42m" << "Running......");

    this->visualization();
    
    this->vis_rate_->sleep();
  }

  return;
}

void FCPlanner_PP_Node::globalPlan(Eigen::VectorXd& cur_pose, Eigen::Vector3d& cur_vel, Eigen::MatrixXd& mesh_V, Eigen::MatrixXi& mesh_F, vector<Eigen::VectorXd>& last_global_path, vector<bool>& last_global_waypt_indicators, bool& succeed)
{
  auto t1 = chrono::high_resolution_clock::now();

  this->path_planner_->visFlag = false;

  // * Reset
  this->path_planner_->reset();

  // * Set Input
  this->path_planner_->setInput(mesh_V, mesh_F);

  // * Global Path Planning
  this->global_succeed_ = path_planner_->plan(cur_pose, cur_vel, last_global_path, last_global_waypt_indicators, false);
  succeed = this->global_succeed_;

  auto t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> g_ms = t2 - t1;
  double g_time = (double)g_ms.count();
  ROS_INFO("\033[33m[FlyCo] Global Planning : %lf ms. \033[32m", g_time);

  return;
}

void FCPlanner_PP_Node::localPlan(Eigen::MatrixXd& mesh_V, Eigen::MatrixXi& mesh_F, vector<Eigen::VectorXd>& g_path, vector<bool>& g_indi, pcl::PointCloud<pcl::PointXYZ>::Ptr& g_hc, pcl::PointCloud<pcl::PointXYZ>::Ptr& g_model, vector<Eigen::VectorXd>& g_prior, vector<Eigen::Vector3d>& g_next_sub_inliers, vector<Eigen::VectorXd>& g_next_sub_vps, const Eigen::VectorXd& local_start, const Eigen::Vector3d vel, const Eigen::Vector3d acc, const Eigen::Vector3d pyd, const Eigen::Vector3d pyd_dot, bool& succeed, bool& feasible)
{ 
  auto t1 = chrono::high_resolution_clock::now();

  cout << "Local Working......" << endl;

  this->local_replanner_->reset();

  // * Input Interface
  this->local_replanner_->setMap(this->mapping_utils_);
  this->local_replanner_->setMesh(mesh_V, mesh_F);
  this->local_replanner_->setGlobalPath(g_path, g_indi);
  this->local_replanner_->setGlobalPriorPath(g_prior);
  this->local_replanner_->setGlobalHCMap(g_hc);
  this->local_replanner_->setGlobalModel(g_model);
  this->local_replanner_->setGlobalNextSub(g_next_sub_inliers, g_next_sub_vps);

  // * Local Motion Replanning
  succeed = this->local_replanner_->replan(feasible, local_start, vel, acc, pyd, pyd_dot);

  // * Output Interface -> Visualization
  bool lock_vis = false;

  if (this->vis_mtx_.try_lock()) lock_vis = true;

  if (lock_vis && succeed)
  {
    this->local_replanner_->getUsedGlobalRoute(this->local_used_g_path_, this->local_used_g_prior_);
    this->local_replanner_->getUsedGlobalNextSub(this->local_used_g_next_sub_inliers_, this->local_used_g_next_sub_vps_);
    this->local_replanner_->getLocalPath(this->local_path_);
    this->local_replanner_->getLocalUpdatedPath(this->local_updated_path_);
    this->local_replanner_->getLocalRegion(this->local_hc_, this->min_bound_, this->max_bound_);

    this->vis_mtx_.unlock();
  }
  else
  {
    if (lock_vis) this->vis_mtx_.unlock();
  }

  auto t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> l_ms = t2 - t1;
  double l_time = (double)l_ms.count();
  ROS_INFO("\033[33m[FlyCo] Local Planning : %lf ms. \033[32m", l_time);

  return;
}

void FCPlanner_PP_Node::visualization()
{
  if (visFlag == true)
  { 
    if (this->vis_mtx_.try_lock())
    {
      if (this->local_used_g_path_.empty() || this->local_hc_ == nullptr || this->local_path_.empty() || this->local_updated_path_.empty())
      {
        this->vis_mtx_.unlock();
        return;
      }
      else
      {
        temp_min_bound = this->min_bound_;
        temp_max_bound = this->max_bound_;
        *temp_local_hc = *this->local_hc_;
        temp_local_path.clear();
        temp_local_path = this->local_path_;
        temp_local_updated_path.clear();
        temp_local_updated_path = this->local_updated_path_;
        temp_local_used_g_path.clear();
        temp_local_used_g_path = this->local_used_g_path_;
        temp_local_used_g_prior.clear();
        temp_local_used_g_prior = this->local_used_g_prior_;
        temp_mesh_V = this->vis_mesh_V_;
        temp_mesh_F = this->vis_mesh_F_;
        
        this->vis_mtx_.unlock();
      }
    }
    else
    {
      return;
    }
    
    // * Local Path Vis
    vector<vector<Eigen::Vector3d>> list1, list2;
    vector<Eigen::Vector3d> l1_, l2_;
    Eigen::Vector3d pos_;
    double pitch_, yaw_;
    for (int i = 0; i < (int)temp_local_path.size(); ++i)
    {
      pos_(0) = temp_local_path[i](0);
      pos_(1) = temp_local_path[i](1);
      pos_(2) = temp_local_path[i](2);
      pitch_ = temp_local_path[i](3);
      yaw_ = temp_local_path[i](4);

      percep_utils_->setPose_PY(pos_, pitch_, yaw_);
      percep_utils_->getFOV_PY(l1_, l2_);

      list1.push_back(l1_);
      list2.push_back(l2_);
    }
    
    // * Updated Local Path Vis
    vector<vector<Eigen::Vector3d>> list1_u, list2_u;
    vector<Eigen::Vector3d> l1_u, l2_u;
    Eigen::Vector3d pos_u;
    double pitch_u, yaw_u;
    for (int i = 0; i < (int)temp_local_updated_path.size(); ++i)
    {
      pos_u(0) = temp_local_updated_path[i](0);
      pos_u(1) = temp_local_updated_path[i](1);
      pos_u(2) = temp_local_updated_path[i](2);
      pitch_u = temp_local_updated_path[i](3);
      yaw_u = temp_local_updated_path[i](4);

      percep_utils_->setPose_PY(pos_u, pitch_u, yaw_u);
      percep_utils_->getFOV_PY(l1_u, l2_u);

      list1_u.push_back(l1_u);
      list2_u.push_back(l2_u);
    }
    
    if (temp_local_hc != nullptr && temp_max_bound.x() > temp_min_bound.x() && temp_max_bound.y() > temp_min_bound.y() && temp_max_bound.z() > temp_min_bound.z())
      vis_utils_->publishLocalRegion(temp_local_hc, temp_min_bound, temp_max_bound);
    if ((int)temp_local_path.size() > 0)
      vis_utils_->publishCurPath(temp_local_path, list1, list2);
    if ((int)temp_local_updated_path.size() > 0)
      vis_utils_->publishUpdatedCurPath(temp_local_updated_path, list1_u, list2_u);
    if ((int)this->vis_global_start_.size() == 5)
      vis_utils_->publishGlobalStart(this->vis_global_start_);
    if ((int)temp_mesh_V.rows() > 0 && (int)temp_mesh_F.rows() > 0)
      vis_utils_->publishPredMesh(temp_mesh_V, temp_mesh_F);
    if ((int)temp_local_used_g_path.size() > 0)
      vis_utils_->publishLocalUsedGlobal(temp_local_used_g_path, temp_local_used_g_prior);

  }

  return;
}

} // namespace flyco
