/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Jun. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file implements tool functions of path computation in FlyCo.
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

#ifndef _PATH_TOOLS_HPP_
#define _PATH_TOOLS_HPP_

#include <Eigen/Core>

using namespace std;

namespace path_tools
{
  inline Eigen::Vector3d pyToVec(Eigen::Vector2d &pitch_yaw)
  {
    Eigen::Vector3d vec;
    vec(0) = cos(pitch_yaw(0)) * cos(pitch_yaw(1));
    vec(1) = cos(pitch_yaw(0)) * sin(pitch_yaw(1));
    vec(2) = sin(pitch_yaw(0));

    return vec;
  }

  inline Eigen::Vector2d vecToPy(Eigen::Vector3d &vec)
  {
    Eigen::Vector2d PY;
    double pitch = std::asin(vec.z() / (vec.norm() + 1e-3)); // calculate pitch angle
    double yaw = std::atan2(vec.y(), vec.x());               // calculate yaw angle

    PY(0) = pitch;
    PY(1) = yaw;

    return PY;
  }

  inline void posInterpolate(const Eigen::VectorXd& start, const Eigen::VectorXd& end, const double interval, vector<Eigen::Vector3d>& inter_pos)
  {
    Eigen::Vector3d dir = (end.head(3) - start.head(3)).normalized();
    double length = (end.head(3) - start.head(3)).norm();
    int piece = ceil(length / interval);
    double new_interval = length / piece;
    
    if (piece == 1) return;

    for (int i=1; i<piece; ++i)
    {
      Eigen::Vector3d temp_pos = start.head(3) + i * new_interval * dir;
      inter_pos.push_back(temp_pos);
    }

    return;
  }

  inline void orienInterpolate(const Eigen::VectorXd& start, const Eigen::VectorXd& end, vector<Eigen::Vector3d>& waypts_, vector<Eigen::VectorXd>& updates_waypts_)
  {
    int dof = start.size(), angle = dof-3;
    double dist_gap = 1e-2;
    double whole_dist = 0.0;
    vector<Eigen::VectorXd> effect_vps_;
    vector<Eigen::VectorXd> seg_posi_;
    double yaw_start = 0.0, yaw_end = 0.0, pitch_start = 0.0, pitch_end = 0.0;
    int cal_flag_yaw = 0, cal_flag_pitch = 0;
    int cal_dir_yaw = 0, cal_dir_pitch = 0;

    if (angle == 1)
    {
      yaw_start = start(3)*180.0/M_PI;
      yaw_end = end(3)*180.0/M_PI;
      cal_flag_yaw = yaw_end - yaw_start > 0? 1:-1;
      cal_dir_yaw = abs(yaw_end - yaw_start)>180.0? -1:1;
    }
    else
    {
      yaw_start = start(4)*180.0/M_PI;
      yaw_end = end(4)*180.0/M_PI;
      cal_flag_yaw = yaw_end - yaw_start > 0? 1:-1;
      cal_dir_yaw = abs(yaw_end - yaw_start)>180.0? -1:1;
      pitch_start = start(3)*180.0/M_PI;
      pitch_end = end(3)*180.0/M_PI;
      cal_flag_pitch = ((pitch_end - pitch_start) > 0)? 1:-1;
      cal_dir_pitch = abs(pitch_end - pitch_start)>180.0? -1:1;
    }
    
    effect_vps_.push_back(start);
    for (auto x:waypts_)
    {
      if ((x-start.head(3)).norm() > dist_gap && (x-end.head(3)).norm() > dist_gap)
      {
        Eigen::VectorXd aug_x; aug_x.resize(dof);
        aug_x(0) = x(0); aug_x(1) = x(1); aug_x(2) = x(2);
        effect_vps_.push_back(aug_x);
      }
    }
    effect_vps_.push_back(end);

    seg_posi_.push_back(start);
    seg_posi_.insert(seg_posi_.end(), effect_vps_.begin(), effect_vps_.end());
    seg_posi_.push_back(end);
    for (int i=0; i<(int)seg_posi_.size()-1; ++i)
      whole_dist += (seg_posi_[i+1].head(3)-seg_posi_[i].head(3)).norm();
    
    if ((int)effect_vps_.size() > 0)
    {
      double yaw_gap = abs(yaw_end - yaw_start)>180.0? (360.0-abs(yaw_end - yaw_start)):abs(yaw_end - yaw_start);
      if (angle == 1)
      {
        for (int i=1; i<(int)effect_vps_.size()-1; ++i)
        {
          double e_dist = (effect_vps_[i].head(3)-start.head(3)).norm();
          effect_vps_[i](3) = (yaw_start + cal_dir_yaw*cal_flag_yaw*yaw_gap*e_dist/whole_dist)*M_PI/180.0;
          while (effect_vps_[i](3) < -M_PI)
            effect_vps_[i](3) += 2 * M_PI;
          while (effect_vps_[i](3) > M_PI)
            effect_vps_[i](3) -= 2 * M_PI;

          updates_waypts_.push_back(effect_vps_[i]);
        }
      }
      else
      {
        double pitch_gap = abs(pitch_end - pitch_start)>180.0? (360.0-abs(pitch_end - pitch_start)):abs(pitch_end - pitch_start);
        for (int i=1; i<(int)effect_vps_.size()-1; ++i)
        {
          double e_dist = (effect_vps_[i].head(3)-start.head(3)).norm();
          effect_vps_[i](3) = (pitch_start + cal_dir_pitch*cal_flag_pitch*pitch_gap*e_dist/whole_dist)*M_PI/180.0;
          effect_vps_[i](4) = (yaw_start + cal_dir_yaw*cal_flag_yaw*yaw_gap*e_dist/whole_dist)*M_PI/180.0;
          while (effect_vps_[i](3) < -M_PI)
            effect_vps_[i](3) += 2 * M_PI;
          while (effect_vps_[i](3) > M_PI)
            effect_vps_[i](3) -= 2 * M_PI;
          while (effect_vps_[i](4) < -M_PI)
            effect_vps_[i](4) += 2 * M_PI;
          while (effect_vps_[i](4) > M_PI)
            effect_vps_[i](4) -= 2 * M_PI;
          
          updates_waypts_.push_back(effect_vps_[i]);
        }
      }
    }

    return;
  }

  inline void pieceInterpolate(const Eigen::VectorXd& start, const Eigen::VectorXd& end, const double interval, vector<Eigen::VectorXd>& piece_waypts)
  {
    vector<Eigen::Vector3d> inter_pos;
    posInterpolate(start, end, interval, inter_pos);
    if ((int)inter_pos.size() == 0) return;

    orienInterpolate(start, end, inter_pos, piece_waypts);

    return;
  }

  inline void nextInterpolate(const Eigen::VectorXd& start, const Eigen::VectorXd& end, const double interval, Eigen::VectorXd& next)
  {
    vector<Eigen::Vector3d> inter_pos;
    posInterpolate(start, end, interval, inter_pos);
    if ((int)inter_pos.size() == 0) return;
    
    vector<Eigen::VectorXd> next_waypts;
    orienInterpolate(start, end, inter_pos, next_waypts);

    next = next_waypts.front();

    return;
  }

  inline void posDirVec(const Eigen::VectorXd& pt, Eigen::Vector3d& pos, Eigen::Vector3d& dir)
  {
    pos = pt.head(3);
    Eigen::Vector2d pitch_yaw(pt(3), pt(4));
    dir = pyToVec(pitch_yaw);

    return;
  }

  inline double estPathTime(Eigen::Vector3d cur_vel, Eigen::Vector3d start, Eigen::Vector3d end, double vm, double am)
  {
    double time_cost = 0.0;
    
    Eigen::Vector3d cur_vel_dir = cur_vel.normalized();
    Eigen::Vector3d dir = (end - start).normalized();
    double dist = (end - start).norm();
    double cur_vel_norm = cur_vel.norm();
    double theta;
    if (cur_vel_norm < 1e-2)
    {
      theta = 0.0;
    }
    else
    {
      theta = acos(cur_vel_dir.dot(dir));
    }

    double t_upper = (vm - cur_vel_norm*cos(theta)) / am;
    double s_upper = cur_vel_norm*cos(theta)*t_upper + 0.5*am*t_upper*t_upper;

    if (s_upper < dist)
    {
      time_cost = t_upper + (dist - s_upper) / vm;
    }
    else
    {
      double cur_v_square = (cur_vel_norm*cos(theta))*(cur_vel_norm*cos(theta));
      time_cost = (sqrt(cur_v_square + 2*am*dist) - cur_vel_norm*cos(theta))/am;
    }

    return time_cost;
  }
}

#endif