/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Apr. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the main algorithm of perception utils in FlyCo.
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

#include "active_perception/perception_utils.h"

namespace flyco {
PerceptionUtils::PerceptionUtils(){
}

PerceptionUtils::~PerceptionUtils(){
}

void PerceptionUtils::init(ros::NodeHandle& nh)
{
  // * Params Initialization
  nh.param("perception_utils/top_angle", top_angle_, -1.0);
  nh.param("perception_utils/left_angle", left_angle_, -1.0);
  nh.param("perception_utils/right_angle", right_angle_, -1.0);

  nh.param("perception_utils/max_dist", max_dist_, -1.0);
  nh.param("perception_utils/vis_dist", vis_dist_, -1.0);

  n_top_ << 0.0, sin(M_PI/2 - top_angle_), cos(M_PI/2 - top_angle_);
  n_bottom_ << 0.0, -sin(M_PI/2 - top_angle_), cos(M_PI/2 - top_angle_);

  n_left_ << sin(M_PI/2 - left_angle_), 0.0, cos(M_PI/2 - left_angle_);
  n_right_ << -sin(M_PI/2 - right_angle_), 0.0, cos(M_PI/2 - right_angle_);

  left_angle_shrink_ = this->shrink_factor_ * left_angle_;
  right_angle_shrink_ = this->shrink_factor_ * right_angle_;
  top_angle_shrink_ = this->shrink_factor_ * top_angle_;

  n_top_shrink_ << 0.0, sin(M_PI/2 - top_angle_shrink_), cos(M_PI/2 - top_angle_shrink_);
  n_bottom_shrink_ << 0.0, -sin(M_PI/2 - top_angle_shrink_), cos(M_PI/2 - top_angle_shrink_);

  n_left_shrink_ << sin(M_PI/2 - left_angle_shrink_), 0.0, cos(M_PI/2 - left_angle_shrink_);
  n_right_shrink_ << -sin(M_PI/2 - right_angle_shrink_), 0.0, cos(M_PI/2 - right_angle_shrink_);

  left_angle_gs_ = left_angle_shrink_;
  right_angle_gs_ = right_angle_shrink_;
  top_angle_gs_ = 0.7 * top_angle_shrink_;

  n_top_gs_ << 0.0, sin(M_PI/2 - top_angle_gs_), cos(M_PI/2 - top_angle_gs_);
  n_bottom_gs_ << 0.0, -sin(M_PI/2 - top_angle_gs_), cos(M_PI/2 - top_angle_gs_);

  n_left_gs_ << sin(M_PI/2 - left_angle_gs_), 0.0, cos(M_PI/2 - left_angle_gs_);
  n_right_gs_ << -sin(M_PI/2 - right_angle_gs_), 0.0, cos(M_PI/2 - right_angle_gs_);

  T_cb_ << 0, -1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1;
  T_bc_ = T_cb_.inverse();

  // * FOV vertices in body frame, for FOV visualization
  double hor = vis_dist_ * tan(left_angle_);
  double vert = vis_dist_ * tan(top_angle_);
  Eigen::Vector3d origin(0, 0, 0);
  Eigen::Vector3d left_up(vis_dist_, hor, vert);
  Eigen::Vector3d left_down(vis_dist_, hor, -vert);
  Eigen::Vector3d right_up(vis_dist_, -hor, vert);
  Eigen::Vector3d right_down(vis_dist_, -hor, -vert);

  cam_vertices1_.push_back(origin);
  cam_vertices2_.push_back(left_up);
  cam_vertices1_.push_back(origin);
  cam_vertices2_.push_back(left_down);
  cam_vertices1_.push_back(origin);
  cam_vertices2_.push_back(right_up);
  cam_vertices1_.push_back(origin);
  cam_vertices2_.push_back(right_down);

  cam_vertices1_.push_back(left_up);
  cam_vertices2_.push_back(right_up);
  cam_vertices1_.push_back(right_up);
  cam_vertices2_.push_back(right_down);
  cam_vertices1_.push_back(right_down);
  cam_vertices2_.push_back(left_down);
  cam_vertices1_.push_back(left_down);
  cam_vertices2_.push_back(left_up);
}

void PerceptionUtils::setShrinkFactor(const double& factor)
{
  this->shrink_factor_ = factor;

  return;
}

void PerceptionUtils::setPose_PY(const Eigen::Vector3d& pos, const double& pitch, const double& yaw)
{
  pos_ = pos;
  pitch_ = pitch;
  yaw_ = yaw;
  // Transform the normals of camera FOV
  Eigen::Matrix3d Rwb_y, Rwb_p;
  Rwb_y << cos(yaw_), -sin(yaw_), 0.0, sin(yaw_), cos(yaw_), 0.0, 0.0, 0.0, 1.0;
  Rwb_p << cos(pitch_), 0.0, -sin(pitch_), 0.0, 1.0, 0.0, sin(pitch_), 0.0, cos(pitch_);
  Eigen::Vector3d pc = pos_;
  Eigen::Matrix4d T_wb = Eigen::Matrix4d::Identity();
  T_wb.block<3, 3>(0, 0) = Rwb_y * Rwb_p;
  T_wb.block<3, 1>(0, 3) = pc;
  Eigen::Matrix4d T_wc = T_wb * T_bc_;
  Eigen::Matrix3d R_wc = T_wc.block<3, 3>(0, 0);
  normals_ = { n_top_, n_bottom_, n_left_, n_right_ };
  for (auto& n : normals_)
  {
    n = R_wc * n;
  }

  shrink_normals_ = { n_top_shrink_, n_bottom_shrink_, n_left_shrink_, n_right_shrink_ };
  for (auto& n : shrink_normals_)
  {
    n = R_wc * n;
  }

  ground_shrink_normals_ = { n_top_gs_, n_bottom_gs_, n_left_gs_, n_right_gs_ };
  for (auto& n : ground_shrink_normals_)
  {
    n = R_wc * n;
  }

  return;
}

void PerceptionUtils::setPoseGimbal(const Eigen::Vector3d& pos, const double& pitch_gimbal, const double& yaw_gimbal)
{
  pos_ = pos;
  pitch_ = pitch_gimbal;
  yaw_ = yaw_gimbal;
  // Transform the normals of camera FOV
  Eigen::Matrix3d Rwb_y, Rwb_p;
  Rwb_y << cos(-yaw_), -sin(-yaw_), 0.0, sin(-yaw_), cos(-yaw_), 0.0, 0.0, 0.0, 1.0;
  Rwb_p << cos(pitch_), 0.0, -sin(pitch_), 0.0, 1.0, 0.0, sin(pitch_), 0.0, cos(pitch_);
  Eigen::Vector3d pc = pos_;
  Eigen::Matrix4d T_wb = Eigen::Matrix4d::Identity();
  T_wb.block<3, 3>(0, 0) = Rwb_y * Rwb_p;
  T_wb.block<3, 1>(0, 3) = pc;
  Eigen::Matrix4d T_wc = T_wb * T_bc_;
  Eigen::Matrix3d R_wc = T_wc.block<3, 3>(0, 0);
  normals_ = { n_top_, n_bottom_, n_left_, n_right_ };
  for (auto& n : normals_)
  {
    n = R_wc * n;
  }

  shrink_normals_ = { n_top_shrink_, n_bottom_shrink_, n_left_shrink_, n_right_shrink_ };
  for (auto& n : shrink_normals_)
  {
    n = R_wc * n;
  }

  ground_shrink_normals_ = { n_top_gs_, n_bottom_gs_, n_left_gs_, n_right_gs_ };
  for (auto& n : ground_shrink_normals_)
  {
    n = R_wc * n;
  }

  return;
}

void PerceptionUtils::getFOV_PY(vector<Eigen::Vector3d>& list1, vector<Eigen::Vector3d>& list2)
{
  list1.clear();
  list2.clear();

  // Get info for visualizing FOV at (pos, yaw)
  Eigen::Matrix3d Rwb_y, Rwb_p;
  Rwb_y << cos(yaw_), -sin(yaw_), 0.0, sin(yaw_), cos(yaw_), 0.0, 0.0, 0.0, 1.0;
  Rwb_p << cos(pitch_), 0.0, -sin(pitch_), 0.0, 1.0, 0.0, sin(pitch_), 0.0, cos(pitch_);
  for (int i = 0; i < (int)cam_vertices1_.size(); ++i) {
    auto p1 = Rwb_y * Rwb_p * cam_vertices1_[i] + pos_;
    auto p2 = Rwb_y * Rwb_p * cam_vertices2_[i] + pos_;
    list1.push_back(p1);
    list2.push_back(p2);
  }

  return;
}

void PerceptionUtils::getFOVGimbal(vector<Eigen::Vector3d>& list1, vector<Eigen::Vector3d>& list2)
{
  list1.clear();
  list2.clear();

  // Get info for visualizing FOV at (pos, yaw)
  Eigen::Matrix3d Rwb_y, Rwb_p;
  Rwb_y << cos(yaw_), -sin(-yaw_), 0.0, sin(-yaw_), cos(-yaw_), 0.0, 0.0, 0.0, 1.0;
  Rwb_p << cos(pitch_), 0.0, -sin(pitch_), 0.0, 1.0, 0.0, sin(pitch_), 0.0, cos(pitch_);
  for (int i = 0; i < (int)cam_vertices1_.size(); ++i) {
    auto p1 = Rwb_y * Rwb_p * cam_vertices1_[i] + pos_;
    auto p2 = Rwb_y * Rwb_p * cam_vertices2_[i] + pos_;
    list1.push_back(p1);
    list2.push_back(p2);
  }

  return;
}

bool PerceptionUtils::insideFOV(const Eigen::Vector3d& point) 
{
  Eigen::Vector3d dir = point - pos_;
  if (dir.norm() > max_dist_) return false;

  dir.normalize();
  for (auto n : normals_) {
    if (dir.dot(n) < 0.0) return false;
  }
  return true;
}

bool PerceptionUtils::insideShrinkFOV(const Eigen::Vector3d& point)
{
  Eigen::Vector3d dir = point - pos_;
  if (dir.norm() > max_dist_) return false;

  dir.normalize();
  for (auto n : this->shrink_normals_) 
  {
    if (dir.dot(n) < 0.0) return false;
  }
  return true;
}

bool PerceptionUtils::insideGSFOV(const Eigen::Vector3d& point)
{
  Eigen::Vector3d dir = point - pos_;
  if (dir.norm() > max_dist_) return false;

  dir.normalize();
  for (auto n : this->ground_shrink_normals_) 
  {
    if (dir.dot(n) < 0.0) return false;
  }
  return true;
}

bool PerceptionUtils::insideFOVInflate(const Eigen::Vector3d& point, const double& inflate_dist) 
{
  Eigen::Vector3d dir = point - pos_;
  if (dir.norm() > (max_dist_+inflate_dist)) return false;

  dir.normalize();
  for (auto n : normals_) {
    if (dir.dot(n) < 0.0) return false;
  }
  return true;
}

void PerceptionUtils::getCamMat(vector<Eigen::Vector3d>& cv1, vector<Eigen::Vector3d>& cv2, const double& dist)
{
  double hor = dist * tan(left_angle_);
  double vert = dist * tan(top_angle_);

  Eigen::Vector3d origin(0, 0, 0);
  Eigen::Vector3d left_up(dist, hor, vert);
  Eigen::Vector3d left_down(dist, hor, -vert);
  Eigen::Vector3d right_up(dist, -hor, vert);
  Eigen::Vector3d right_down(dist, -hor, -vert);

  cv1.push_back(origin);
  cv2.push_back(left_up);
  cv1.push_back(origin);
  cv2.push_back(left_down);
  cv1.push_back(origin);
  cv2.push_back(right_up);
  cv1.push_back(origin);
  cv2.push_back(right_down);

  cv1.push_back(left_up);
  cv2.push_back(right_up);
  cv1.push_back(right_up);
  cv2.push_back(right_down);
  cv1.push_back(right_down);
  cv2.push_back(left_down);
  cv1.push_back(left_down);
  cv2.push_back(left_up); 

  return;
}

void PerceptionUtils::getCamMatShrink(vector<Eigen::Vector3d>& cv1, vector<Eigen::Vector3d>& cv2, const double& dist)
{
  double hor = dist * tan(left_angle_shrink_);
  double vert = dist * tan(top_angle_shrink_);

  Eigen::Vector3d origin(0, 0, 0);
  Eigen::Vector3d left_up(dist, hor, vert);
  Eigen::Vector3d left_down(dist, hor, -vert);
  Eigen::Vector3d right_up(dist, -hor, vert);
  Eigen::Vector3d right_down(dist, -hor, -vert);

  cv1.push_back(origin);
  cv2.push_back(left_up);
  cv1.push_back(origin);
  cv2.push_back(left_down);
  cv1.push_back(origin);
  cv2.push_back(right_up);
  cv1.push_back(origin);
  cv2.push_back(right_down);

  cv1.push_back(left_up);
  cv2.push_back(right_up);
  cv1.push_back(right_up);
  cv2.push_back(right_down);
  cv1.push_back(right_down);
  cv2.push_back(left_down);
  cv1.push_back(left_down);
  cv2.push_back(left_up); 

  return;
}

void PerceptionUtils::getFOVBoundingBox(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax) {
  double left = yaw_ + left_angle_;
  double right = yaw_ - right_angle_;
  double up = top_angle_;
  double down = -top_angle_;
  Eigen::Vector3d left_pt = pos_ + max_dist_ * Eigen::Vector3d(cos(left), sin(left), 0);
  Eigen::Vector3d right_pt = pos_ + max_dist_ * Eigen::Vector3d(cos(right), sin(right), 0);
  Eigen::Vector3d up_pt = pos_ + max_dist_ * Eigen::Vector3d(cos(yaw_), sin(yaw_), sin(up));
  Eigen::Vector3d down_pt = pos_ + max_dist_ * Eigen::Vector3d(cos(yaw_), sin(yaw_), sin(down));
  vector<Eigen::Vector3d> points = { left_pt, right_pt, up_pt, down_pt };
  // if (left > 0 && right < 0)
  //   points.push_back(pos_ + max_dist_ * Vector3d(1, 0, 0));
  // else if (left > M_PI/2 && right < M_PI/2)
  //   points.push_back(pos_ + max_dist_ * Vector3d(0, 1, 0));
  // else if (left > -M_PI/2 && right < -M_PI/2)
  //   points.push_back(pos_ + max_dist_ * Vector3d(0, -1, 0));
  // else if ((left > M_PI && right < M_PI) || (left > -M_PI && right < -M_PI))
  //   points.push_back(pos_ + max_dist_ * Vector3d(-1, 0, 0));

  bmax = bmin = pos_;
  for (auto p : points) {
    bmax = bmax.array().max(p.array());
    bmin = bmin.array().min(p.array());
  }
}

void PerceptionUtils::getFOVBoundingBoxPYInflate(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax, const double& inflate_dist)
{  
  double cur_dist_ = max_dist_ + inflate_dist;
  
  vector<Eigen::Vector3d> cv1, cv2;
  getCamMat(cv1, cv2, cur_dist_);
  
  vector<Eigen::Vector3d> points;

  Eigen::Matrix3d Rwb_y, Rwb_p;
  Rwb_y << cos(yaw_), -sin(yaw_), 0.0, sin(yaw_), cos(yaw_), 0.0, 0.0, 0.0, 1.0;
  Rwb_p << cos(pitch_), 0.0, -sin(pitch_), 0.0, 1.0, 0.0, sin(pitch_), 0.0, cos(pitch_);
  for (int i = 0; i < (int)cv1.size(); ++i) {
    auto p1 = Rwb_y * Rwb_p * cv1[i] + pos_;
    auto p2 = Rwb_y * Rwb_p * cv2[i] + pos_;
    points.push_back(p1);
    points.push_back(p2);
  }

  bmax = bmin = pos_;
  for (const auto& p : points) 
  {
    bmax = bmax.array().max(p.array());
    bmin = bmin.array().min(p.array());
  }

  // Inflate the bounding box
  bmax(0) += 2*inflate_dist;
  bmax(1) += 2*inflate_dist;
  bmax(2) += 2*inflate_dist;
  bmin(0) -= 2*inflate_dist;
  bmin(1) -= 2*inflate_dist;
  bmin(2) -= 2*inflate_dist;
}

void PerceptionUtils::getHRepresentationFovInflate(Eigen::MatrixX4d& H, const double& inflate_dist)
{
  double h_dist = inflate_dist;

  vector<Eigen::Vector3d> cv1, cv2;
  getCamMatShrink(cv1, cv2, h_dist);

  // * points: {origin, left_up, right_up, right_down, left_down}
  vector<Eigen::Vector3d> points;
  Eigen::Matrix3d Rwb_y, Rwb_p;
  Rwb_y << cos(yaw_), -sin(yaw_), 0.0, sin(yaw_), cos(yaw_), 0.0, 0.0, 0.0, 1.0;
  Rwb_p << cos(pitch_), 0.0, -sin(pitch_), 0.0, 1.0, 0.0, sin(pitch_), 0.0, cos(pitch_);
  for (int i = 3; i < (int)cv1.size(); ++i) 
  {
    auto p1 = Rwb_y * Rwb_p * cv1[i] + pos_;
    points.push_back(p1);
  }
  
  // * get vertices of 5 planes -> {top, bottom, left, right, far}
  vector<Eigen::Vector3d> top = { points[0], points[2], points[1] }; // origin, right_up, left_up
  vector<Eigen::Vector3d> bottom = { points[0], points[4], points[3] }; // origin, left_down, right_down
  vector<Eigen::Vector3d> left = { points[0], points[1], points[4] }; // origin, left_up, left_down
  vector<Eigen::Vector3d> right = { points[0], points[3], points[2] }; // origin, right_down, right_up
  vector<Eigen::Vector3d> far = { points[1], points[2], points[3], points[4] }; // left_up, right_up, right_down, left_down

  vector<vector<Eigen::Vector3d>> vertices = { top, bottom, left, right, far };

  // * get H representation of 5 planes
  // Each row of hPoly is defined by h0, h1, h2, h3 as
  // h0*x + h1*y + h2*z + h3 <= 0 -> (h0, h1, h2) unit normal, h3 distance to origin
  H.resize(5, 4);
  for (int i = 0; i < 5; ++i) 
  {
    Eigen::Vector3d n = (vertices[i][1] - vertices[i][0]).cross(vertices[i][2] - vertices[i][0]).normalized();
    H.row(i) << n(0), n(1), n(2), -n.dot(vertices[i][0]);
  }
}

}  // namespace flyco