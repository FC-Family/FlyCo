/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Apr. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the header file of Astar class, which implements safe path
 *                   searching in FlyCo.
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

#ifndef _ASTAR2_H
#define _ASTAR2_H

#include "plan_env/sdf_map.h"
#include "path_searching/matrix_hash.h"
#include <ros/ros.h>
#include <ros/console.h>
#include <Eigen/Eigen>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <queue>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace flyco {
class Node {
public:
  Eigen::Vector3i index;
  Eigen::Vector3d position;
  double g_score, f_score;
  Node* parent;

  /* -------------------- */
  Node() {
    parent = NULL;
  }
  ~Node(){};
};
typedef Node* NodePtr;

class NodeComparator0 {
public:
  bool operator()(NodePtr node1, NodePtr node2) {
    return node1->f_score > node2->f_score;
  }
};

class Astar {
public:
  Astar();
  ~Astar();
  enum { REACH_END = 1, NO_PATH = 2 };

  void init_hc(ros::NodeHandle& nh);
  void setMap(const SDFMap::Ptr& hc_map_);
  void reset();
  int hc_search(const Eigen::Vector3d& start_pt, const Eigen::Vector3d& end_pt);
  int hc_occ_search(const Eigen::Vector3d& start_pt, const Eigen::Vector3d& end_pt);
  int wholeSearch(const Eigen::Vector3d& start_pt, const Eigen::Vector3d& end_pt);
  void setResolution(const double& res);
  static double pathLength(const vector<Eigen::Vector3d>& path);

  std::vector<Eigen::Vector3d> getPath();
  std::vector<Eigen::Vector3d> getVisited();
  double getEarlyTerminateCost();

  double lambda_heu_hc_;
  double max_search_time_;
  bool zFlag;
  double groundz, safeheight; 

private:
  void backtrack(const NodePtr& end_node, const Eigen::Vector3d& end);
  int wholeSearchAtStep(const Eigen::Vector3d& start_pt, const Eigen::Vector3d& end_pt, const int step);
  bool isWholeSearchPointSafe(const Eigen::Vector3d& pos);
  bool isWholeSearchSegmentSafe(const Eigen::Vector3d& start, const Eigen::Vector3d& end);
  void shortenWholeSearchPath();
  void posToIndex(const Eigen::Vector3d& pt, Eigen::Vector3i& idx);
  double getDiagHeu(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2);
  double getManhHeu(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2);
  double getEuclHeu(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2);

  // main data structure
  vector<NodePtr> path_node_pool_;
  int use_node_num_, iter_num_;
  std::priority_queue<NodePtr, std::vector<NodePtr>, NodeComparator0> open_set_;
  std::unordered_map<Eigen::Vector3i, NodePtr, matrix_hash<Eigen::Vector3i>> open_set_map_;
  std::unordered_map<Eigen::Vector3i, int, matrix_hash<Eigen::Vector3i>> close_set_map_;
  std::vector<Eigen::Vector3d> path_nodes_;
  double early_terminate_cost_;

  SDFMap::Ptr map_hc_;

  // parameter
  double margin_;
  int allocate_num_ = 100000;
  double tie_breaker_;
  double resolution_, inv_resolution_;
  int whole_search_max_step_ = 3;
  double whole_search_shorten_interval_ = 1.0;
  Eigen::Vector3d map_size_3d_, origin_;
};

}  // namespace flyco

#endif