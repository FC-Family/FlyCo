/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Feb. 2025
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the main algorithm of initial trajectory generation in FlyCo.
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

#include "hierarchical_coverage_planner/init_traj.h"

namespace flyco
{
  Init_Traj::Init_Traj(){
  }

  Init_Traj::~Init_Traj(){
  }

  void Init_Traj::init(ros::NodeHandle& nh)
  {
    // * Params Initialization
    nh.param("flyco_planner/enable_inherit", this->en_inherit_, false);
    nh.param("flyco_planner/init_path_file", this->init_path_file_, std::string(""));
    this->readInitialPoints();
    this->start_gen_ = false;
    this->finish_init_ = false;
    this->traj_id_ = -this->path_size_ + 1;

    // * Module Initialization
    this->traj_generator_.reset(new TrajGenerator);
    this->traj_generator_->init(nh);

    // * ROS Initialization
    this->task_timer_ = nh.createTimer(ros::Duration(0.1), &Init_Traj::runCallback, this);
    this->traj_pub_ = nh.advertise<quadrotor_msgs::PolynomialTrajGroup>("/local_newest_traj", 1);
    this->joy_sub_ = nh.subscribe("/joy", 50, &Init_Traj::joyCallback, this);
    this->odom_sub_ = nh.subscribe("/flyco_planner/body_pose", 1, &Init_Traj::odomCallback, this);
    this->gimbal_sub_ = nh.subscribe("/flyco_planner/gimbal_pose", 1, &Init_Traj::gimbalCallback, this);

    return;
  }

  void Init_Traj::setMap(shared_ptr<SDFMap>& map)
  {
    this->mapping_ = map;
    this->traj_generator_->setMap(this->mapping_);

    return;
  }

  /**
   * @brief traj_id_ -> -1 for initial img traj while 0 for initial pc traj
   */
  void Init_Traj::run()
  {
    if (this->traj_id_ > 0) return;

    ros::Time traj_start_time = ros::Time::now() + ros::Duration(3.0);
    // * Process Start Pose
    Eigen::VectorXd start_pose(5);

    Eigen::Vector3d plan_pos(this->odom_.pose.pose.position.x, this->odom_.pose.pose.position.y, this->odom_.pose.pose.position.z);

    tf::Quaternion tfQuat_body;
    tf::quaternionMsgToTF(this->odom_.pose.pose.orientation, tfQuat_body);
    tf::Matrix3x3 m_body(tfQuat_body);
    double roll_body, pitch_body, yaw_body;
    m_body.getRPY(roll_body, pitch_body, yaw_body);

    tf::Quaternion tfQuat;
    tf::quaternionMsgToTF(this->gimbal_.quaternion, tfQuat);
    tf::Matrix3x3 m(tfQuat);
    double roll_gimbal, pitch_gimbal, yaw_gimbal;
    m.getRPY(roll_gimbal, pitch_gimbal, yaw_gimbal);

    start_pose << plan_pos(0), plan_pos(1), plan_pos(2), pitch_gimbal, yaw_body;
    this->waypoints_.push_back(start_pose);
    this->indicators_.push_back(false);

    // * Read Initial Waypoints
    this->waypoints_.insert(this->waypoints_.end(), this->paths_[this->num_path_].begin(), this->paths_[this->num_path_].end());
    this->indicators_.insert(this->indicators_.end(), this->paths_indicators_[this->num_path_].begin(), this->paths_indicators_[this->num_path_].end());

    if ((int)waypoints_.size() == 1)
    {
      ROS_WARN("\033[41;37m[FlyCo]\033[47;31m No Initial Waypoints Found! \033[0m");
      return;
    }

    // * Interpolate Path
    vector<Eigen::VectorXd> updated_waypoints;
    vector<bool> updated_indicators;
    for (int i=0; i<(int)this->waypoints_.size()-1; i++)
    {
      updated_waypoints.push_back(this->waypoints_[i]);
      updated_indicators.push_back(this->indicators_[i]);

      Eigen::VectorXd p1 = this->waypoints_[i];
      Eigen::VectorXd p2 = this->waypoints_[i+1];

      double dist = (p2.head(3) - p1.head(3)).norm();
      int num = ceil(dist / 1.0);
      double step = dist / num;

      vector<Eigen::VectorXd> piece_waypts;
      path_tools::pieceInterpolate(p1, p2, step, piece_waypts);
      if (piece_waypts.empty()) continue;

      vector<bool> piece_indicators(piece_waypts.size(), true);
      updated_waypoints.insert(updated_waypoints.end(), piece_waypts.begin(), piece_waypts.end());
      updated_indicators.insert(updated_indicators.end(), piece_indicators.begin(), piece_indicators.end());
    }
    updated_waypoints.push_back(this->waypoints_.back());
    updated_indicators.push_back(this->indicators_.back());

    this->waypoints_.clear();
    this->indicators_.clear();
    this->waypoints_ = updated_waypoints;
    this->indicators_ = updated_indicators;

    // * Generate Trajectory
    Eigen::Vector3d now_posi = this->waypoints_.front().head(3);
    vector<Eigen::Vector3d> wps;
    vector<bool> wps_waypt_indicators;
    vector<double> pitchs, yaws;

    wps.resize((int)this->waypoints_.size()-1);
    wps_waypt_indicators.resize((int)this->waypoints_.size()-1);
    pitchs.resize((int)this->waypoints_.size());
    yaws.resize((int)this->waypoints_.size());

    for (int i=0; i<(int)this->waypoints_.size(); ++i)
    {
      if (i > 0)
      {
        wps[i-1] = this->waypoints_[i].head(3);
        wps_waypt_indicators[i-1] = this->indicators_[i];
      }  
      pitchs[i] = this->waypoints_[i](3);
      yaws[i] = this->waypoints_[i](4);
    }

    voxel_map::VoxelMap vmap;
    double sfc = 0.0;
    this->traj_generator_->visFlag = true;
    this->traj_generator_->HCTraj(now_posi, wps, wps_waypt_indicators, pitchs, yaws, vmap, sfc, 1.7, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    Trajectory<7> pos_traj = this->traj_generator_->minco_traj;
    Trajectory<7> ori_traj = this->traj_generator_->minco_orientation_traj;

    // * Publish Initial Trajectory
    quadrotor_msgs::PolynomialTrajGroup traj_msg;
    this->trajConverter(pos_traj, ori_traj, traj_msg, traj_start_time, this->traj_id_);
    this->traj_pub_.publish(traj_msg);

    if (this->traj_id_ >= 0)
    {
      // * Release
      this->traj_generator_.reset(new TrajGenerator);
      this->traj_pub_ = ros::Publisher();
      this->joy_sub_ = ros::Subscriber();
      this->odom_sub_ = ros::Subscriber();
      this->gimbal_sub_ = ros::Subscriber();
    }

    ROS_INFO("\033[41;37m[FlyCo]\033[47;31m Initial -> [ %d ]-th Trajectory Generation...... \033[0m", this->traj_id_);
    ros::Duration sleep_time = ros::Duration(0.5);
    sleep_time.sleep();
    this->finish_init_ = false;
    this->traj_id_++;
    this->num_path_++;
    this->waypoints_.clear();
    this->indicators_.clear();
    this->traj_generator_->reset();

    return;
  }

  void Init_Traj::readInitialPoints()
  {
    string package_path = ros::package::getPath("flyco_planner_manager");
    string filename;

    if (this->en_inherit_)
    {
      filename = package_path + "/inherit_data/inherit_point.yaml";
      YAML::Node config = YAML::LoadFile(filename);
      vector<Eigen::VectorXd> waypts;
      vector<bool> indicators;
      for (const auto& point : config["waypoints"])
      {
        Eigen::VectorXd wp(5);
        
        wp << point[0].as<double>(), point[1].as<double>(), point[2].as<double>(), point[3].as<double>()*M_PI/180.0, point[4].as<double>()*M_PI/180.0;

        waypts.push_back(wp);
        indicators.push_back(false);
      }
      this->paths_.push_back(waypts);
      this->paths_indicators_.push_back(indicators);

      this->path_size_ = (int)this->paths_.size();
    }
    else
    {
      filename = this->init_path_file_;
      YAML::Node config = YAML::LoadFile(filename);
      for (const auto& path : config["path"])
      {
        vector<Eigen::VectorXd> waypts;
        vector<bool> indicators;
        for (const auto& point : path["waypoints"])
        {
          Eigen::VectorXd wp(5);
          
          wp << point[0].as<double>(), point[1].as<double>(), point[2].as<double>(), point[3].as<double>()*M_PI/180.0, point[4].as<double>()*M_PI/180.0;

          waypts.push_back(wp);
          indicators.push_back(false);
        }
        this->paths_.push_back(waypts);
        this->paths_indicators_.push_back(indicators);
      }

      this->path_size_ = (int)this->paths_.size();
    }

    return;
  }

  void Init_Traj::trajConverter(const Trajectory<7> &pos, const Trajectory<7> &ori, quadrotor_msgs::PolynomialTrajGroup &msg, const ros::Time &cur_stamp, int &traj_id)
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

  void Init_Traj::runCallback(const ros::TimerEvent& event)
  {
    if (this->start_gen_ && !this->finish_init_)
    {
      this->start_gen_ = false;
      this->finish_init_ = true;
      this->run();
    }

    return;
  }

  void Init_Traj::joyCallback(const sensor_msgs::Joy::ConstPtr& Joy)
  {
    sensor_msgs::Joy joy_msg;
    if (Joy) joy_msg = *Joy;

    // Button 'Y' to start initial trajectory generation
    if (joy_msg.buttons[3] == true) 
    {
      this->start_gen_ = true;
    }

    return;
  }

  void Init_Traj::odomCallback(const nav_msgs::OdometryConstPtr& msg)
  {
    this->odom_ = *msg;

    return;
  }

  void Init_Traj::gimbalCallback(const geometry_msgs::QuaternionStampedConstPtr& msg)
  {
    this->gimbal_ = *msg;

    return;
  }

}