/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Apr. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the header file of RayCaster class, which implements
 *                   visibility and safety check in FlyCo.
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

#ifndef RAYCAST_H_
#define RAYCAST_H_

#include <Eigen/Eigen>
#include <vector>

double signum(double x);

double mod(double value, double modulus);

double intbound(double s, double ds);

void Raycast(const Eigen::Vector3d& start, const Eigen::Vector3d& end, const Eigen::Vector3d& min,
             const Eigen::Vector3d& max, int& output_points_cnt, Eigen::Vector3d* output);

void Raycast(const Eigen::Vector3d& start, const Eigen::Vector3d& end, const Eigen::Vector3d& min,
             const Eigen::Vector3d& max, std::vector<Eigen::Vector3d>* output);

class RayCaster {
private:
  /* data */
  Eigen::Vector3d start_;
  Eigen::Vector3d end_;
  Eigen::Vector3d direction_;
  Eigen::Vector3d min_;
  Eigen::Vector3d max_;
  int x_;
  int y_;
  int z_;
  int endX_;
  int endY_;
  int endZ_;
  double maxDist_;
  double dx_;
  double dy_;
  double dz_;
  int stepX_;
  int stepY_;
  int stepZ_;
  double tMaxX_;
  double tMaxY_;
  double tMaxZ_;
  double tDeltaX_;
  double tDeltaY_;
  double tDeltaZ_;
  double dist_;

  Eigen::Vector3d inter_;
  int interX_;
  int interY_;
  int interZ_;
  Eigen::Vector3d dir_p, dir_n;
  double maxDp, maxDn;
  double dx_p, dy_p, dz_p;
  double dx_n, dy_n, dz_n;
  int stepXp, stepYp, stepZp;
  int stepXn, stepYn, stepZn;
  double tMaxXp, tMaxYp, tMaxZp;
  double tMaxXn, tMaxYn, tMaxZn;
  double tDeltaXp, tDeltaYp, tDeltaZp;
  double tDeltaXn, tDeltaYn, tDeltaZn;

  int step_num_;

  double resolution_;
  Eigen::Vector3d offset_;
  Eigen::Vector3d half_;

public:
  RayCaster(/* args */) {
  }
  ~RayCaster() {
  }

  void setParams(const double& res, const Eigen::Vector3d& origin);
  bool input(const Eigen::Vector3d& start, const Eigen::Vector3d& end);
  bool biInput(const Eigen::Vector3d& start, const Eigen::Vector3d& end);
  bool nextId(Eigen::Vector3i& idx);
  bool biNextId(Eigen::Vector3i& idx, Eigen::Vector3i& idxR);
  bool nextPos(Eigen::Vector3d& pos);

  // deprecated
  bool setInput(const Eigen::Vector3d& start, const Eigen::Vector3d& end /* , const Eigen::Vector3d& min,
                const Eigen::Vector3d& max */);
  bool step(Eigen::Vector3d& ray_pt);
};

#endif  // RAYCAST_H_