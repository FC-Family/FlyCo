/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Apr. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the header file of PerceptionUtils class, which implements
 *                   perception utils for coverage in FlyCo.
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

#ifndef _PERCEPTION_UTILS_H_
#define _PERCEPTION_UTILS_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <iostream>
#include <memory>
#include <vector>

using std::shared_ptr;
using std::unique_ptr;
using std::vector;

namespace flyco {
class PerceptionUtils {
public:
  PerceptionUtils();
  ~PerceptionUtils();

  void init(ros::NodeHandle& nh);
  void setShrinkFactor(const double& factor);
  void setPose_PY(const Eigen::Vector3d& pos, const double& pitch, const double& yaw);
  void setPoseGimbal(const Eigen::Vector3d& pos, const double& pitch_gimbal, const double& yaw_gimbal);
  void getFOV_PY(vector<Eigen::Vector3d>& list1, vector<Eigen::Vector3d>& list2);
  void getFOVGimbal(vector<Eigen::Vector3d>& list1, vector<Eigen::Vector3d>& list2);
  bool insideFOV(const Eigen::Vector3d& point);
  bool insideShrinkFOV(const Eigen::Vector3d& point);
  bool insideGSFOV(const Eigen::Vector3d& point);
  bool insideFOVInflate(const Eigen::Vector3d& point, const double& inflate_dist);
  void getCamMat(vector<Eigen::Vector3d>& cv1, vector<Eigen::Vector3d>& cv2, const double& dist);
  void getCamMatShrink(vector<Eigen::Vector3d>& cv1, vector<Eigen::Vector3d>& cv2, const double& dist);
  void getFOVBoundingBox(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax);
  void getFOVBoundingBoxPYInflate(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax, const double& inflate_dist);
  void getHRepresentationFovInflate(Eigen::MatrixX4d& H, const double& inflate_dist);

  /* Param */
  double max_dist_;

private:
  // Data
  // Current camera pos and yaw
  Eigen::Vector3d pos_;
  double pitch_, yaw_;
  // Camera plane's normals in world frame
  vector<Eigen::Vector3d> normals_, shrink_normals_, ground_shrink_normals_;

  /* Params */
  // Sensing range of camera
  double left_angle_, right_angle_, top_angle_, vis_dist_;
  double shrink_factor_ = 1.0;
  double left_angle_shrink_, right_angle_shrink_, top_angle_shrink_;
  double left_angle_gs_, right_angle_gs_, top_angle_gs_;
  // Normal vectors of camera FOV planes in camera frame
  Eigen::Vector3d n_top_, n_bottom_, n_left_, n_right_;
  Eigen::Vector3d n_top_shrink_, n_bottom_shrink_, n_left_shrink_, n_right_shrink_;
  Eigen::Vector3d n_top_gs_, n_bottom_gs_, n_left_gs_, n_right_gs_;
  // Transform between camera and body
  Eigen::Matrix4d T_cb_, T_bc_;
  // FOV vertices in body frame
  vector<Eigen::Vector3d> cam_vertices1_, cam_vertices2_;
};

} // namespace flyco
#endif