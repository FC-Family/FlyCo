/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Sept. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the header file of SDFMap class, which implements dual
 *                   volumetric mapping in FlyCo.
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

#ifndef _SDF_MAP_H
#define _SDF_MAP_H

#include "plan_env/map_ros.h"
#include "plan_env/raycast.h"
#include <ros/ros.h>
#include <ros/package.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>
#include <pcl/io/pcd_io.h>
#include <pcl/surface/convex_hull.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <tuple>
#include <queue>
#include <chrono>
#include <unordered_map>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>

using namespace std;
using std::shared_ptr;
using std::unique_ptr;

namespace cv {
class Mat;
}

class RayCaster;

namespace flyco {
struct MapParam;
struct MapData;
struct HCMapParam;
struct HCMapData;
class MapROS;

class SDFMap {
public:
  SDFMap();
  ~SDFMap();

  enum OCCUPANCY { UNKNOWN, FREE, OCCUPIED, HC_INTERNAL };

  Eigen::Vector3d min_bound, max_bound;

  void initMap(ros::NodeHandle& nh);
  void setXYoffset(double x_offset, double y_offset);
  void getXYoffset(double& x_offset, double& y_offset);
  void inputPointCloud(const pcl::PointCloud<pcl::PointXYZ>& points, const int& point_num,
                       const Eigen::Vector3d& camera_pos, bool air);
  void inputFreePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& points);

  void posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& id);
  void posToIndex_hc(const Eigen::Vector3d& pos, Eigen::Vector3i& id);
  void indexToPos(const Eigen::Vector3i& id, Eigen::Vector3d& pos);
  void indexToPos_hc(const Eigen::Vector3i& id, Eigen::Vector3d& pos);
  void boundIndex(Eigen::Vector3i& id);
  int toAddress(const Eigen::Vector3i& id);
  int toAddress(const int& x, const int& y, const int& z);
  int toAddress_hc(const Eigen::Vector3i& id);
  int toAddress_hc(const int& x, const int& y, const int& z);
  bool isInMap(const Eigen::Vector3d& pos);
  bool isInMap(const Eigen::Vector3i& idx);
  bool isInBox(const Eigen::Vector3i& id);
  bool isInBox(const Eigen::Vector3d& pos);
  void boundBox(Eigen::Vector3d& low, Eigen::Vector3d& up);
  int getOccupancy(const Eigen::Vector3d& pos);
  int getOccupancy(const Eigen::Vector3i& id);
  int getInflateOccupancy(const Eigen::Vector3d& pos);
  int getInflateOccupancy(const Eigen::Vector3i& id);
  double getDistance(const Eigen::Vector3d& pos);
  double getDistance(const Eigen::Vector3i& id);
  double getDistWithGrad(const Eigen::Vector3d& pos, Eigen::Vector3d& grad);
  void updateESDF3d();
  void resetBuffer();
  void resetBuffer(const Eigen::Vector3d& min, const Eigen::Vector3d& max);
  void getRegion(Eigen::Vector3d& ori, Eigen::Vector3d& size);
  void getBox(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax);
  void getUpdatedBox(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax, bool reset = false);
  double getResolution();
  int getVoxelNum();
  void getOccMap(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
  void inv_address(int& idx, Eigen::Vector3i& pos);
  void getLocalEnv(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, Eigen::Vector3d& min, Eigen::Vector3d& max);
  void getLocalOccEnv(pcl::PointCloud<pcl::PointXYZ>::Ptr& occ_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& env_cloud, Eigen::Vector3d& min, Eigen::Vector3d& max);
  bool getSafety(const Eigen::Vector3d& pos, const double x_range, const double y_range, const double z_range);
  void getGlobalBox(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax);

  /* Utils */
  unique_ptr<MapParam> mp_;

  // ! /* hierarchical_coverage_planner */ --------------------------------------------
  bool zFlag; double zPos;
  int checkSize, inflate_num;
  void initHC(ros::NodeHandle& nh);
  void setHC(pcl::PointCloud<pcl::PointXYZ>::Ptr& model);
  void initHCMap(ros::NodeHandle& nh, pcl::PointCloud<pcl::PointXYZ>::Ptr& model);
  void InternalSpace(map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr>& seg_cloud, Eigen::MatrixXd& vertices, map<int, vector<int>>& segments);
  void OuterCheck(vector<Eigen::VectorXd>& outers);
  vector<Eigen::Vector3d> points_in_plane(pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud, Eigen::Vector3d& point_on_plane, Eigen::Vector3d& plane_normal, double thickness);
  bool isInMap_hc(const Eigen::Vector3d& pos);
  bool isInMap_hc(const Eigen::Vector3i& idx);
  bool safety_check(Eigen::Vector3d& pos);
  bool safety_check(Eigen::Vector3i& id);
  bool occCheck(Eigen::Vector3d& pos);
  bool occCheck(Eigen::Vector3i& id);
  bool freeCheck(Eigen::Vector3d& pos);
  bool freeCheck(Eigen::Vector3i& id);
  int get_Internal(const Eigen::Vector3d& pos);
  int get_Internal(const Eigen::Vector3i& id);

  void setInternal(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const unordered_map<int, Eigen::Vector3d>& cor_inliers);
  void resetHCMap(const pcl::PointCloud<pcl::PointXYZ>::Ptr& model);
  void resetHCVirtualBound(const double min_x, const double max_x, const double min_y, const double max_y);
  void resetHCTop(const double min_x, const double max_x, const double min_y, const double max_y);
  void getGlobalHCMap(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
  void getLocalHCMap(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, Eigen::Vector3d& min, Eigen::Vector3d& max);
  bool getHCSafety(const Eigen::Vector3d& pos, const double x_range, const double y_range, const double z_range);
  double getDistance_hc(const Eigen::Vector3d& pos);
  double getDistance_hc(const Eigen::Vector3i& id);
  void updateESDF3dHCMap();

  unique_ptr<HCMapParam> hcmp_;
  unique_ptr<HCMapData> hcmd_;
  unique_ptr<RayCaster> internal_cast_;
  // ! /* hierarchical_coverage_planner */ --------------------------------------------

  // * inheritance flight
  void loadInheritMap();
  void loadInheritOffset();
  void getOccPcd(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
  void getFreePcd(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

private:
  void clearAndInflateLocalMap();
  void clearAndMapTarget(const pcl::PointCloud<pcl::PointXYZ>& points, const int& point_num, const Eigen::Vector3d& camera_pos);
  void clearAndMapEnvironment(const pcl::PointCloud<pcl::PointXYZ>& points, const int& point_num, const Eigen::Vector3d& camera_pos);
  void inflatePoint(const Eigen::Vector3i& pt, int step, int z_step, vector<Eigen::Vector3i>& pts);
  void setCacheOccupancy(const int& adr, const int& occ);
  Eigen::Vector3d closetPointInMap(const Eigen::Vector3d& pt, const Eigen::Vector3d& camera_pt);
  template <typename F_get_val, typename F_set_val>
  void fillESDF(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int dim);
  
  bool airsim_flag_ = false;
  double z_min = 0.0;
  double cur_x_ = 0.0, cur_y_ = 0.0;

  bool en_inherit_ = false;

  unique_ptr<MapData> md_;
  unique_ptr<MapROS> mr_;
  unique_ptr<RayCaster> caster_;

  friend MapROS;

public:
  typedef std::shared_ptr<SDFMap> Ptr;
  typedef std::unique_ptr<SDFMap> HCPtr;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct MapParam {
  // map properties
  Eigen::Vector3d map_origin_, map_size_;
  Eigen::Vector3d map_min_boundary_, map_max_boundary_;
  Eigen::Vector3i map_voxel_num_;
  double resolution_, resolution_inv_;
  double obstacles_inflation_, target_obstacles_inflation_, obs_inf_z_, target_obs_inf_z_;
  double virtual_ceil_height_, ground_height_;
  Eigen::Vector3i box_min_, box_max_;
  Eigen::Vector3d box_mind_, box_maxd_;
  double default_dist_;
  bool optimistic_, signed_dist_;
  // map fusion
  double p_hit_, p_miss_, p_min_, p_max_, p_occ_;  // occupancy probability
  double prob_hit_log_, prob_miss_log_, clamp_min_log_, clamp_max_log_, min_occupancy_log_;  // logit
  double max_ray_length_;
  double local_bound_inflate_;
  int local_map_margin_;
  double unknown_flag_;
  double x_origin_back_, y_origin_back_;
};

struct HCMapParam
{
  double resolution_, resolution_inv_, proj_interval, thickness, size_inflate;
  double checkScale;
  Eigen::Vector3d map_origin_, map_size_;
  Eigen::Vector3d map_min_boundary_, map_max_boundary_;
  Eigen::Vector3i map_voxel_num_;
  Eigen::Vector3i box_min_, box_max_;
  Eigen::Vector3d box_mind_, box_maxd_;
  Eigen::Vector3i map_origin_idx_;
  double x_size_, y_size_, z_size_, z_min_;
};

struct MapData {
  // Occupancy
  vector<double> occupancy_buffer_;
  vector<char> occupancy_buffer_inflate_;
  // target & environment flags
  vector<char> target_buffer_;
  vector<bool> target_publish_buffer_;
  vector<char> env_buffer_;
  // ESDF
  vector<double> distance_buffer_neg_;
  vector<double> distance_buffer_;
  vector<double> tmp_buffer1_;
  vector<double> tmp_buffer2_;
  // data for updating
  vector<short> count_hit_, count_miss_;
  vector<char> flag_rayend_;
  char raycast_num_;
  queue<int> cache_voxel_;
  Eigen::Vector3i local_bound_min_, local_bound_max_;
  Eigen::Vector3d update_min_, update_max_;
  bool reset_updated_box_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct HCMapData
{
  std::vector<char> occupancy_buffer_hc_;
  std::vector<char> occupancy_inflate_buffer_hc_;
  std::vector<char> occupancy_buffer_internal_;
  pcl::PointCloud<pcl::PointXYZ> occ_cloud;
  pcl::PointCloud<pcl::PointXYZ> internal_cloud;
  std::vector<bool> seg_occ_visited_buffer_;
  // ESDF
  vector<double> distance_buffer_neg_hc_;
  vector<double> distance_buffer_hc_;
  vector<double> tmp_buffer1_hc_;
  vector<double> tmp_buffer2_hc_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline void SDFMap::posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& id) {
  for (int i = 0; i < 3; ++i)
    id(i) = floor((pos(i) - mp_->map_origin_(i)) * mp_->resolution_inv_);
}

inline void SDFMap::posToIndex_hc(const Eigen::Vector3d& pos, Eigen::Vector3i& id) {
  for (int i = 0; i < 3; ++i)
    id(i) = floor((pos(i) - hcmp_->map_origin_(i)) * hcmp_->resolution_inv_);
}

inline void SDFMap::indexToPos(const Eigen::Vector3i& id, Eigen::Vector3d& pos) {
  for (int i = 0; i < 3; ++i)
    pos(i) = (id(i) + 0.5) * mp_->resolution_ + mp_->map_origin_(i);
}

inline void SDFMap::indexToPos_hc(const Eigen::Vector3i& id, Eigen::Vector3d& pos) {
  for (int i = 0; i < 3; ++i)
    pos(i) = (id(i) + 0.5) * hcmp_->resolution_ + hcmp_->map_origin_(i);
}


inline void SDFMap::boundIndex(Eigen::Vector3i& id) {
  Eigen::Vector3i id1;
  id1(0) = max(min(id(0), mp_->map_voxel_num_(0) - 1), 0);
  id1(1) = max(min(id(1), mp_->map_voxel_num_(1) - 1), 0);
  id1(2) = max(min(id(2), mp_->map_voxel_num_(2) - 1), 0);
  id = id1;
}

inline int SDFMap::toAddress(const int& x, const int& y, const int& z) {
  return x * mp_->map_voxel_num_(1) * mp_->map_voxel_num_(2) + y * mp_->map_voxel_num_(2) + z;
}

inline int SDFMap::toAddress(const Eigen::Vector3i& id) {
  return toAddress(id[0], id[1], id[2]);
}

inline int SDFMap::toAddress_hc(const int& x, const int& y, const int& z) {
  int xi = x-hcmp_->map_origin_idx_(0); int yi = y-hcmp_->map_origin_idx_(1); int zi = z-hcmp_->map_origin_idx_(2); 
  return xi * hcmp_->map_voxel_num_(1) * hcmp_->map_voxel_num_(2) + yi * hcmp_->map_voxel_num_(2) + zi;
}

inline int SDFMap::toAddress_hc(const Eigen::Vector3i& id) {
  return toAddress_hc(id[0], id[1], id[2]);
}

inline void SDFMap::inv_address(int& idx, Eigen::Vector3i& pos)
{
  int x_inv = int(idx/(mp_->map_voxel_num_(1) * mp_->map_voxel_num_(2)));
  int y_inv = int((idx-x_inv*mp_->map_voxel_num_(1) * mp_->map_voxel_num_(2))/mp_->map_voxel_num_(2));
  int z_inv = idx - x_inv * mp_->map_voxel_num_(1) * mp_->map_voxel_num_(2) - y_inv * mp_->map_voxel_num_(2);

  pos(0) = x_inv;
  pos(1) = y_inv;
  pos(2) = z_inv;
}

inline bool SDFMap::isInMap(const Eigen::Vector3d& pos) {
  if (pos(0) < mp_->map_min_boundary_(0) + 1e-4 || pos(1) < mp_->map_min_boundary_(1) + 1e-4 ||
      pos(2) < mp_->map_min_boundary_(2) + 1e-4)
    return false;
  if (pos(0) > mp_->map_max_boundary_(0) - 1e-4 || pos(1) > mp_->map_max_boundary_(1) - 1e-4 ||
      pos(2) > mp_->map_max_boundary_(2) - 1e-4)
    return false;
  return true;
}

inline bool SDFMap::isInMap(const Eigen::Vector3i& idx) {
  if (idx(0) < 0 || idx(1) < 0 || idx(2) < 0) return false;
  if (idx(0) > mp_->map_voxel_num_(0) - 1 || idx(1) > mp_->map_voxel_num_(1) - 1 ||
      idx(2) > mp_->map_voxel_num_(2) - 1)
    return false;
  return true;
}

inline bool SDFMap::isInBox(const Eigen::Vector3i& id) {
  for (int i = 0; i < 3; ++i) {
    if (id[i] < mp_->box_min_[i] || id[i] >= mp_->box_max_[i]) {
      return false;
    }
  }
  return true;
}

inline bool SDFMap::isInBox(const Eigen::Vector3d& pos) {
  for (int i = 0; i < 3; ++i) {
    if (pos[i] <= mp_->box_mind_[i] || pos[i] >= mp_->box_maxd_[i]) {
      return false;
    }
  }
  return true;
}

inline void SDFMap::boundBox(Eigen::Vector3d& low, Eigen::Vector3d& up) {
  for (int i = 0; i < 3; ++i) {
    low[i] = max(low[i], mp_->box_mind_[i]);
    up[i] = min(up[i], mp_->box_maxd_[i]);
  }
}

inline int SDFMap::getOccupancy(const Eigen::Vector3i& id) {
  if (!isInMap(id)) return -1;
  double occ = md_->occupancy_buffer_[toAddress(id)];
  if (occ < mp_->clamp_min_log_ - 1e-3) return UNKNOWN;
  if (occ > mp_->min_occupancy_log_) return OCCUPIED;
  return FREE;
}

inline int SDFMap::getOccupancy(const Eigen::Vector3d& pos) {
  Eigen::Vector3i id;
  posToIndex(pos, id);
  return getOccupancy(id);
}

inline bool SDFMap::isInMap_hc(const Eigen::Vector3d& pos) {
  if (pos(0) < hcmp_->map_min_boundary_(0) + 1e-4 || pos(1) < hcmp_->map_min_boundary_(1) + 1e-4 ||
      pos(2) < hcmp_->map_min_boundary_(2) + 1e-4)
    return false;
  if (pos(0) > hcmp_->map_max_boundary_(0) - 1e-4 || pos(1) > hcmp_->map_max_boundary_(1) - 1e-4 ||
      pos(2) > hcmp_->map_max_boundary_(2) - 1e-4)
    return false;
  return true;
}

inline bool SDFMap::isInMap_hc(const Eigen::Vector3i& idx) {
  if (idx(0) < 0 || idx(1) < 0 || idx(2) < 0) return false;
  if (idx(0) > hcmp_->map_voxel_num_(0) - 1 || idx(1) > hcmp_->map_voxel_num_(1) - 1 ||
      idx(2) > hcmp_->map_voxel_num_(2) - 1)
    return false;
  return true;
}

inline bool SDFMap::safety_check(Eigen::Vector3i& id)
{
  if (!isInMap_hc(id)) return false;
  
  if (hcmd_->occupancy_buffer_hc_[toAddress_hc(id)] == 1 || hcmd_->occupancy_buffer_internal_[toAddress_hc(id)] == 1 || hcmd_->occupancy_inflate_buffer_hc_[toAddress_hc(id)] == 1)
    return false;

  return true;
}

inline bool SDFMap::safety_check(Eigen::Vector3d& pos)
{
  if (zFlag == true)
  {
    if (pos(2) < zPos)
      return false;
  }

  Eigen::Vector3i id;
  posToIndex_hc(pos, id);
  return safety_check(id);
}

inline bool SDFMap::occCheck(Eigen::Vector3i& id)
{
  if (!isInMap_hc(id)) return false;

  if (hcmd_->occupancy_buffer_hc_[toAddress_hc(id)] == 1)
    return false;
  
  return true;
}

inline bool SDFMap::occCheck(Eigen::Vector3d& pos)
{
  Eigen::Vector3i id;
  posToIndex_hc(pos, id);
  return occCheck(id);
}

inline bool SDFMap::freeCheck(Eigen::Vector3i& id)
{
  // only for viewpoint_manager demo
  if (!isInMap_hc(id)) return false;
  if (hcmd_->occupancy_buffer_hc_[toAddress_hc(id)] == 2)
    return true;
  
  return false;
}

inline bool SDFMap::freeCheck(Eigen::Vector3d& pos)
{
  // only for viewpoint_manager demo
  Eigen::Vector3i id;
  posToIndex_hc(pos, id);
  return occCheck(id);
}

inline int SDFMap::get_Internal(const Eigen::Vector3i& id)
{
  if (!isInMap_hc(id)) return -1;
  if (hcmd_->occupancy_buffer_internal_[toAddress_hc(id)] == 1) 
    return HC_INTERNAL;
  
  return -1;
}

inline int SDFMap::get_Internal(const Eigen::Vector3d& pos)
{
  Eigen::Vector3i id;
  posToIndex_hc(pos, id);
  return get_Internal(id);
}

inline int SDFMap::getInflateOccupancy(const Eigen::Vector3i& id) {
  if (!isInMap(id)) return -1;
  return int(md_->occupancy_buffer_inflate_[toAddress(id)]);
}

inline int SDFMap::getInflateOccupancy(const Eigen::Vector3d& pos) {
  Eigen::Vector3i id;
  posToIndex(pos, id);
  return getInflateOccupancy(id);
}

inline double SDFMap::getDistance(const Eigen::Vector3i& id) {
  if (!isInMap(id)) return -1;
  return md_->distance_buffer_[toAddress(id)];
}

inline double SDFMap::getDistance(const Eigen::Vector3d& pos) {
  Eigen::Vector3i id;
  posToIndex(pos, id);
  return getDistance(id);
}

inline double SDFMap::getDistance_hc(const Eigen::Vector3i& id) {
  if (!isInMap_hc(id)) return -1;
  return hcmd_->distance_buffer_hc_[toAddress_hc(id)];
}

inline double SDFMap::getDistance_hc(const Eigen::Vector3d& pos) {
  Eigen::Vector3i id;
  posToIndex_hc(pos, id);
  return getDistance_hc(id);
}

inline void SDFMap::inflatePoint(const Eigen::Vector3i& pt, int step, int z_step, vector<Eigen::Vector3i>& pts) {
  int num = 0;

  /* ---------- + shape inflate ---------- */
  // for (int x = -step; x <= step; ++x)
  // {
  //   if (x == 0)
  //     continue;
  //   pts[num++] = Eigen::Vector3i(pt(0) + x, pt(1), pt(2));
  // }
  // for (int y = -step; y <= step; ++y)
  // {
  //   if (y == 0)
  //     continue;
  //   pts[num++] = Eigen::Vector3i(pt(0), pt(1) + y, pt(2));
  // }
  // for (int z = -1; z <= 1; ++z)
  // {
  //   pts[num++] = Eigen::Vector3i(pt(0), pt(1), pt(2) + z);
  // }

  /* ---------- all inflate ---------- */
  for (int x = -step; x <= step; ++x)
    for (int y = -step; y <= step; ++y)
      for (int z = -z_step; z <= z_step; ++z) {
        if (x == 0 && y == 0 && z == 0)
          continue;
        pts.push_back(Eigen::Vector3i(pt(0) + x, pt(1) + y, pt(2) + z));
      }
}

}
#endif