/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Apr. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the header file of HCSolver class, which implements basic
 *                   solver for planning modules in FlyCo.
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

#ifndef _HCSOLVER_H_
#define _HCSOLVER_H_

#include "path_searching/astar2.h"
#include "plan_env/raycast.h"
#include "plan_env/sdf_map.h"
#include "plan_utils/path_tools.hpp"
#include "plan_utils/solver_tools.hpp"
#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/surface/convex_hull.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <random>
#include <math.h>
#include <thread>
#include <unistd.h>
#include <mutex>
#include <condition_variable>
#include <iomanip>
#include <unordered_set>

using namespace std;
using std::unique_ptr;
using std::shared_ptr;

class RayCaster;

#define COST_INFINITY                 (1e5)

namespace flyco
{

struct Site
{
  Site *Pred;
  Site *Suc;
  bool Start;
  bool End;
  double X;
  double Y;
  double Z;
  double Pitch;
  double Yaw;
  int GlobalID;
  int LocalID;
};
// * Consistent planning data structure
struct PairVectorXdCompare
{
  bool operator()(const pair<Eigen::VectorXd, Eigen::VectorXd>& lhs, const pair<Eigen::VectorXd, Eigen::VectorXd>& rhs) const
  {
    if (lhs.first.size() != rhs.first.size())
    {
      return lhs.first.size() < rhs.first.size();
    }
    for (int i = 0; i < lhs.first.size(); ++i)
    {
      if (lhs.first[i] != rhs.first[i])
      {
        return lhs.first[i] < rhs.first[i];
      }
    }

    if (lhs.second.size() != rhs.second.size())
    {
      return lhs.second.size() < rhs.second.size();
    }
    for (int i = 0; i < lhs.second.size(); ++i)
    {
      if (lhs.second[i] != rhs.second[i])
      {
        return lhs.second[i] < rhs.second[i];
      }
    }

    return false;
  }
};

struct Pair 
{
  int first;
  int second;

  Pair(int a, int b) {
      if (a < b) {
          first = a;
          second = b;
      } else {
          first = b;
          second = a;
      }
  }

  bool operator<(const Pair& other) const {
      return std::tie(first, second) < std::tie(other.first, other.second);
  }
};

struct AstarResults
{
  double cost = -1.0;
  vector<Eigen::VectorXd> path = {}; // 5-DoF: [x, y, z, pitch, yaw], inter path N-2 dimension
};

struct Vector2iCompare
{
  bool operator()(const Eigen::Vector2i &v1, const Eigen::Vector2i &v2) const
  {
    if (v1(0) != v2(0))
      return v1(0) < v2(0);
    return v1(1) < v2(1);
  }
};

struct VectorXdCompare
{
  bool operator()(const Eigen::VectorXd &v1, const Eigen::VectorXd &v2) const
  {
    for (int i = 0; i < v1.size(); ++i)
    {
      if (v1(i) != v2(i))
        return v1(i) < v2(i);
    }
    return false;
  }
};

struct Vector3dCompare
{
    bool operator()(const Eigen::Vector3d& a, const Eigen::Vector3d& b) const {
        for (int i = 0; i < 3; ++i) {
            if (std::abs(a[i] - b[i]) > 1e-6) {
                return a[i] < b[i];
            }
        }
        return false;
    }
};

class Astar;
class SDFMap;

class HCSolver
{
public:
  HCSolver();
  ~HCSolver();

  /* Func */
  void init(ros::NodeHandle& nh);
  void setStart(Eigen::Vector3d& start_);
  void setMap(SDFMap::Ptr& hcmap);
  void setRayCaster(double& res_, Eigen::Vector3d& origin_);
  void setRayCasterReal(double& res_, Eigen::Vector3d& origin_);
  vector<int> GlobalSubspaceSequence(map<int, vector<Eigen::VectorXd>>& sub_vps);
  map<int, vector<int>> GlobalBoundaryPoints(map<int, vector<Eigen::VectorXd>>& sub_vps, vector<int>& globalseq);
  vector<int> topoGlobalSeq(map<int, vector<Eigen::VectorXd>>& sub_vps, vector<Eigen::Vector3d>& topoNodes, map<vector<int>, int>& topoEdges, Eigen::MatrixXd& topoAdjMat, vector<vector<int>>& all_nodes_idx, vector<Eigen::Vector3d>& all_nodes);
  map<int, vector<int>> topoBoundaryPoints(map<int, vector<Eigen::VectorXd>>& sub_vps, vector<int>& globalseq, map<int, Eigen::Vector3d>& nearestPts, map<vector<int>, vector<Eigen::Vector3d>>& ctrlPts);
  tuple<map<int, vector<Eigen::VectorXd>>, map<int, vector<vector<Eigen::VectorXd>>>> LocalConditionalPath(
    map<int, vector<Eigen::VectorXd>>& sub_vps, map<int, vector<int>>& global_boundary, bool turn);
  tuple<vector<Eigen::VectorXd>, vector<vector<Eigen::VectorXd>>> CoverageFullPath(
    Eigen::VectorXd& cur_pos, vector<int>& globalseq, map<int, vector<Eigen::VectorXd>>& localviewpts, map<int, vector<vector<Eigen::VectorXd>>>& localwaypts);
  vector<Eigen::VectorXd> LocalRefine(vector<Eigen::Vector3d>& Joints,double& Range, vector<Eigen::VectorXd>& Path, bool turn, double time_bound, bool all_refine);
  void consistencyRefine(const vector<Eigen::VectorXd>& path, const Eigen::Vector3d& vel, vector<Eigen::VectorXd>& refine_path, vector<bool>& indi);
  /* // * Consistent Planning Func */
  void parallelPairPathSearching(const map<int, vector<Eigen::VectorXd>>& sub_vps);
  void ATSPSolver(const map<pair<Eigen::VectorXd, Eigen::VectorXd>, AstarResults, PairVectorXdCompare> astar_results, const int sub_idx, const int start_idx, const vector<Eigen::VectorXd> viewpoints, const bool start_end, const int end_idx);
  void parallelSubPlanning(const map<int, vector<Eigen::VectorXd>>& remaining_vps, const map<int, vector<int>>& remaining_boundary, const vector<int>& remaining_subs, const vector<int>& remaining_order, vector<Eigen::VectorXd>& remaining_full_path);
  /* // * Consistent Planning Data */
  map<int, map<pair<Eigen::VectorXd, Eigen::VectorXd>, AstarResults, PairVectorXdCompare>> sub_pair_astar_results_; 
  map<int, map<Eigen::VectorXd, bool, VectorXdCompare>> sub_waypoints_indicators_;
  map<Eigen::VectorXd, bool, VectorXdCompare> refined_waypoints_indicators_;
  /* Data */
  vector<Eigen::Vector3d> centroids;
  vector<vector<Eigen::Vector3d>> JointVps;
  map<int, vector<Eigen::VectorXd>> sub_paths_;
  /* Tools */
  void AngleInterpolation(Eigen::VectorXd& start, Eigen::VectorXd& end, vector<Eigen::Vector3d>& waypts_, vector<Eigen::VectorXd>& updates_waypts_);
  double search_Path(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, vector<Eigen::Vector3d>& path);
  void findBridgePath(Eigen::VectorXd& p1, Eigen::VectorXd& p2, double& cost, vector<Eigen::VectorXd>& path, vector<bool>& indicators, const double res, bool only_occ);
  void findRealBridgePath(Eigen::VectorXd& p1, Eigen::VectorXd& p2, double& cost, vector<Eigen::VectorXd>& path, vector<bool>& indicators);

  /* Decomp Planning */
  void decompGlobalSeq(const vector<Eigen::VectorXd>& vps, const vector<vector<int>>& vps_set, const vector<Eigen::VectorXd>& LAST_vps, const vector<vector<int>>& LAST_vps_set, const vector<int>& LAST_global_seq, const int& reused_sets, const int& updated_sets, const int& match_k, vector<int>& global_seq, vector<Eigen::Vector3d>& global_path);
  void decompGlobalPath(const Eigen::VectorXd& start_pose, const Eigen::Vector3d& start_vel, const vector<Eigen::VectorXd>& prior_remaining, const vector<Eigen::VectorXd>& vps, const vector<vector<int>>& vps_set, const vector<int>& global_seq, const vector<Eigen::Vector3d>& global_seq_path, const int& reused_num, vector<Eigen::VectorXd>& full_path, vector<bool>& full_indicators);
  void decompReusedPath(Eigen::Vector3d start_vel, vector<Eigen::VectorXd>& vps_pose, vector<Eigen::VectorXd>& sub_path, vector<bool>& sub_indicators);
  void decompFollowPath(vector<Eigen::VectorXd>& vps_pose, vector<Eigen::VectorXd>& sub_path, vector<bool>& sub_indicators);

  /* Ground Priority Flight */
  void groundFindPath(Eigen::Vector3d start_vel, vector<Eigen::VectorXd>& vps_pose, vector<Eigen::VectorXd>& sub_path, vector<bool>& sub_indicators);
  bool findClosestBoundaryPoint(const Eigen::Vector3d& pt, const Eigen::Vector3d& nr, Eigen::Vector3d& closest_pt, double vp_box_min_x, double vp_box_max_x, double vp_box_min_y, double vp_box_max_y); 

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  /* Multi-thread lock */
  std::mutex path_finder_mtx, cp_mtx_;
  std::condition_variable con_var;
  /* Param */
  int precision_ = 0, system_back_ = 0, local_system_back_ = 0, boundary_system_back_ = 0, consistency_system_back_ = 0, next_system_back_ = 0;
  string GlobalSolver_ = "", BoundarySolver_ = "";
  string GlobalPar_ = "", GlobalProF_ = "", GlobalResult_ = "";
  string LocalFolder_ = "";
  int GlobalRuns_ = 0;
  int LocalRuns_ = 0;
  bool tripod_head_trigger_ = false;
  double vm_ = 0.0, am_ = 0.0, yd_ = 0.0, ydd_ = 0.0, jm_ = 0.0, amean_ = 0.0;
  double bridge_length_ = 0.0, tempLength = 0.0;
  int local2optNum = 0;
  double astarSearchingRes = 0.0;
  int swapTimes = 0;
  double global_range_ = 0.0;
  double cvx_range_ = 0.0;
  /* Data */
  Eigen::Vector3d solver_start_;
  map<int, Eigen::MatrixXd> local_sub_costmat_;
  map<int, Eigen::MatrixXd> local_sub_phymat_;
  map<vector<double>, double> allAstarCost_; // vector: [i_pos(0), i_pos(1), i_pos(2), j_pos(0), j_pos(1), j_pos(2)] --> Astar cost
  map<vector<double>, vector<Eigen::VectorXd>> allAstarPath_; // vector: [i_pos(0), i_pos(1), i_pos(2), j_pos(0), j_pos(1), j_pos(2)] --> Astar search path
  map<int, vector<int>> local_sub_init_idx_; // boundary-conditioned initial ids of sub-space viewpoints
  // vector<int>: sub_id, viewpoint1, viewpoint2; vector<Eigen::Vector3d>: waypoints from Astar
  map<vector<int>, vector<Eigen::VectorXd>> path_waypts_;
  map<int, vector<Eigen::VectorXd>> local_sub_path_viewpts_;
  map<int, vector<vector<Eigen::VectorXd>>> local_sub_path_waypts_;
  /* Refine Data */
  vector<Eigen::Vector3d> LocalVps;
  map<Eigen::Vector3d, int, Vector3dCompare> RefineID;
  vector<Site*> AllPathSite;
  vector<Site*> LocalSite;
  double inter_cost = 0.0;
  /* Sub-space Param */
  double phy_dist = 0.0, opt_dist = 0.0;
  /* Utils */ 
  unique_ptr<Astar> astar_ = nullptr;
  unique_ptr<RayCaster> solve_raycaster_ = nullptr;
  unique_ptr<RayCaster> real_raycaster_ = nullptr;
  shared_ptr<SDFMap> solve_map_ = nullptr;
  Eigen::MatrixXd GlobalCostMat(Eigen::Vector3d& start_, vector<Eigen::Vector3d>& targets);
  void GlobalParWrite();
  void GlobalProblemWrite(Eigen::MatrixXd& costMat);
  vector<int> GlobalResultsRead();
  int FindSphereNearestPoint(Eigen::Vector3d& spherePtA, Eigen::Vector3d& spherePtB, vector<Eigen::VectorXd>& vps);
  void LocalFindPath(vector<Eigen::VectorXd> vps, vector<int> boundary_, int sub_id_, bool turn);
  void LocalPathFinder(vector<Eigen::VectorXd> vps, vector<int> boundary_, int sub_id_);
  void solveConsistentPath(const vector<Eigen::VectorXd>& pts, const Eigen::Vector3d& vel, vector<Eigen::VectorXd>& refine_path, vector<bool>& indi);
  /* Tools */
  double compute_Cost(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const double& y1, const double& y2, vector<Eigen::Vector3d>& path);
  double computeTimeCost(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const double& y1, const double& y2, vector<Eigen::Vector3d>& path);
  double compute_Cost_tripod_head(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const double& y1, const double& y2, vector<Eigen::Vector3d>& path);
  double computeTimeCost_tripod_head(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const double& y1, const double& y2, vector<Eigen::Vector3d>& path);
  void ConstructKDTree(vector<Eigen::VectorXd>& pathlist, pcl::KdTreeFLANN<pcl::PointXYZ>& tree);
  void Make2Opt(Site *s1, Site *s2, Site *s3, Site *s4, bool turn);
  void Make3Opt(Site *s1, Site *s2, Site *s3, Site *s4, Site *s5, Site *s6, bool turn);
  void RandomLocal2Opt(bool turn);
  void RandomLocal3Opt(bool turn);
  void shortenPath(vector<Eigen::Vector3d>& path);
  double kOptTimeCost(vector<Eigen::Vector3d>& path);
  /* Utils */
  double pathTimeUniAcc(Eigen::Vector3d& pred, Eigen::Vector3d& cur, Eigen::Vector3d& suc);
  vector<string> split(string str, string pattern);
  vector<int> generalizedEulerianPath(Eigen::MatrixXd& adj, int start_vertex);
  int fourPointsNearestPoint(Eigen::Vector3d& spherePtA, Eigen::Vector3d& spherePtB, Eigen::Vector3d& centerA, Eigen::Vector3d& centerB, vector<Eigen::VectorXd>& vps);
  /* // * Consistent Planning Utils */
  void subPathSearching(int sub_idx, vector<Eigen::VectorXd> vps);
  int findBoundary(Eigen::Vector3d& spherePtA, Eigen::Vector3d& spherePtB, vector<Eigen::VectorXd>& vps, int excluded_idx);
};

inline vector<string> HCSolver::split(string str, string pattern)
{
  vector<string> ret;
  if (pattern.empty()) return ret;
  size_t start = 0, index = str.find_first_of(pattern, 0);
  while (index != str.npos)
  {
    if (start != index)
      ret.push_back(str.substr(start, index - start));
    start = index + 1;
    index = str.find_first_of(pattern, start);
  }
  if (!str.substr(start).empty())
    ret.push_back(str.substr(start));
  return ret;
}

} // namespace flyco

#endif