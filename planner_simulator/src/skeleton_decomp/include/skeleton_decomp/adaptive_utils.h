/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    May. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the header file of adaptive_utils class, which implements some
 *                   utils for robust and adaptive skeleton extraction in FlyCo.
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

#ifndef _ADAPTIVE_UTILS_H_
#define _ADAPTIVE_UTILS_H_

#include "plan_utils/visibility_st.hpp"
#include <ros/ros.h>
#include <igl/read_triangle_mesh.h>
#include <igl/signed_distance.h>
#include <embree3/rtcore.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <random>

using namespace std;
using std::unique_ptr;
using std::shared_ptr;

namespace flyco
{

struct Vector3dComp
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

class adaptive_utils
{
public:
  adaptive_utils(){
  }
  ~adaptive_utils(){
  }

  void init(ros::NodeHandle& nh);
  void setScene(RTCScene input_scene);
  void create_scene();
  /* Func */
  vector<int> fps(const pcl::PointCloud<pcl::PointNormal>::Ptr cloud, int num_samples);
  Eigen::MatrixXd cluster_proc(int& queryIdx, Eigen::MatrixXd& neighbors, Eigen::MatrixXd& pts_mat, double& eps);
  vector<int> sdf_check(Eigen::MatrixXd& pts_mat);
  vector<int> raytracing_check(Eigen::MatrixXd& pts_mat, Eigen::MatrixXf& pts_dir, vector<vector<Eigen::Vector3d>>& intersection_pts);
  bool ray_tracing_public(Eigen::Vector3d& pointA, Eigen::Vector3d& pointB);
  bool ray_tracing_public(Eigen::Vector3d& pointA, Eigen::Vector3d& pointB, RTCScene query_scene);
  void ray_tracing_public_batch(const vector<Eigen::Vector3d>& points, RTCScene query_scene, Eigen::MatrixXi& graph, double& min_edge_dist);
  bool cross_ray_tracing(Eigen::Vector3d& point, Eigen::Vector3f& dir);
  bool max_progress(Eigen::Vector3d& p, Eigen::Vector3d& n, double& prog);
  void set_free_raytracing();
  /* Param */
  double resolution_;
  bool independent;
  /* Data */
  string mesh_file = "";
  pcl::PointCloud<pcl::PointXYZ>::Ptr sampled_cloud;
  Eigen::MatrixXd mesh_V;
  Eigen::MatrixXi mesh_F;
  RTCScene scene = nullptr;

  typedef std::shared_ptr<adaptive_utils> Ptr;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  /* Param */
  string fps_tool = "", input_file = "", output_file = "";
  int system_back_ = 0;
  /* Data */
  map<string, int> point_idx_str_;
  vector<Eigen::Vector3d> inter_pts;
  /* Utils */
  void mesh2pcd(double& resolution);
  vector<int> findNeighbors(Eigen::Vector3d& pointP, vector<Eigen::Vector3d>& neighborhood, double& epsilon);
  void assignVisited(vector<int>& neighbors, vector<bool>& visited);
  int calNewVisited(vector<int>& neighbors, vector<bool>& visited);
  vector<Eigen::Vector3d> adaptiveDensityCluster(Eigen::Vector3d& pointP, vector<Eigen::Vector3d>& neighborhood, double& epsilon);
  vector<RTCHit> get_all_intersections(RTCScene scene, RTCRay ray);
  bool ray_tracing(Eigen::Vector3d& point, Eigen::Vector3f& direction);
};

}

#endif
