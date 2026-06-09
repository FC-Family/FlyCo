/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Jun. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file implements tool functions of visibility constraints in FlyCo.
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

#ifndef _VISIBILITY_ST_HPP_
#define _VISIBILITY_ST_HPP_

#include <ros/ros.h>
#include <Eigen/Core>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <embree3/rtcore.h>
#include <chrono>
#include <random>
#include <numeric> 
#include <thread>
#include <future>

using namespace std;

namespace visibility_st
{
  inline double dist_ = -1.0;
  inline double left_angle_ = 0.0;
  inline double top_angle_ = 0.0;
  inline double resolution_ = 0.0;
  inline pcl::KdTreeFLANN<pcl::PointXYZ> inliers_kdtree_;
  inline pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_kdtree_ptr;

  inline void setParams(const double& dist, const double& left_angle, const double& top_angle, const double& resolution)
  {
    dist_ = dist;
    left_angle_ = left_angle;
    top_angle_ = top_angle;
    resolution_ = resolution;

    return;
  }

  inline void setInliers(pcl::PointCloud<pcl::PointXYZ>::Ptr& inliers)
  {
    inliers_kdtree_ = pcl::KdTreeFLANN<pcl::PointXYZ>();
    inliers_kdtree_ptr.reset(new pcl::PointCloud<pcl::PointXYZ>);
    inliers_kdtree_.setInputCloud(inliers);
    *inliers_kdtree_ptr = *inliers;

    return;
  }

  inline void mesh2scene(Eigen::MatrixXd& mesh_V, Eigen::MatrixXi& mesh_F, RTCScene& scene)
  {
    RTCDevice device;
    device = rtcNewDevice(NULL);
    scene = rtcNewScene(device);
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    float* vertices = (float*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), mesh_V.rows());
    for (int i = 0; i < (int)mesh_V.rows(); ++i) {
      vertices[3 * i + 0] = mesh_V(i, 0);
      vertices[3 * i + 1] = mesh_V(i, 1);
      vertices[3 * i + 2] = mesh_V(i, 2);
    }

    unsigned int* indices = (unsigned int*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned int), mesh_F.rows());
    for (int i = 0; i < (int)mesh_F.rows(); ++i) {
      indices[3 * i + 0] = mesh_F(i, 0);
      indices[3 * i + 1] = mesh_F(i, 1);
      indices[3 * i + 2] = mesh_F(i, 2);
    }

    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);

    rtcReleaseDevice(device);

    return;
  }

  inline void mesh2pcd(Eigen::MatrixXd& mesh_V, Eigen::MatrixXi& mesh_F, double& resolution, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
  {

    const int num_faces = mesh_F.rows();
    std::vector<double> areas(num_faces);
    double total_area = 0.0;

    for (int i = 0; i < num_faces; ++i) 
    {
        Eigen::Vector3d v1 = mesh_V.row(mesh_F(i, 0));
        Eigen::Vector3d v2 = mesh_V.row(mesh_F(i, 1));
        Eigen::Vector3d v3 = mesh_V.row(mesh_F(i, 2));

        double area = 0.5 * ((v2 - v1).cross(v3 - v1)).norm();
        areas[i] = area;
        total_area += area;
    }

    int total_points = static_cast<int>(total_area / (resolution * resolution));

    std::vector<double> cumulative_areas(num_faces);
    std::partial_sum(areas.begin(), areas.end(), cumulative_areas.begin());

    std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> sampled_points(total_points);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < total_points; ++i) 
    {
      double random_area = dist(rng) * total_area;
      auto it = std::lower_bound(cumulative_areas.begin(), cumulative_areas.end(), random_area);
      int triangle_idx = std::distance(cumulative_areas.begin(), it);

      Eigen::Vector3d v1 = mesh_V.row(mesh_F(triangle_idx, 0));
      Eigen::Vector3d v2 = mesh_V.row(mesh_F(triangle_idx, 1));
      Eigen::Vector3d v3 = mesh_V.row(mesh_F(triangle_idx, 2));

      double r1 = dist(rng);
      double r2 = dist(rng);
      double sqrt_r1 = std::sqrt(r1);
      double u = 1.0 - sqrt_r1;
      double v = r2 * sqrt_r1;

      Eigen::Vector3d point = u * v1 + v * v2 + (1.0 - u - v) * v3;
      sampled_points[i] = pcl::PointXYZ(point(0), point(1), point(2));
    }

    cloud->points = std::move(sampled_points);

    return;
  }

  // no collision -> true, collision -> false
  inline bool rayTracing(const Eigen::Vector3d& pointA, const Eigen::Vector3d& pointB, const RTCScene& scene)
  {
    Eigen::Vector3d direction_d = (pointB - pointA).normalized();
    Eigen::Vector3f direction = direction_d.cast<float>();
    double length = (pointB - pointA).norm();

    RTCRayHit rayhit;
    RTCRay ray;
    ray.org_x = pointA(0);
    ray.org_y = pointA(1);
    ray.org_z = pointA(2);
    ray.dir_x = direction(0);
    ray.dir_y = direction(1);
    ray.dir_z = direction(2);
    ray.tnear = 0;
    ray.tfar = length;
    ray.time = 0.0f;
    ray.mask = -1;
    ray.id = 0;
    ray.flags = 0;
    rayhit.ray = ray;
    RTCHit hit;
    hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit = hit;

    RTCIntersectContext context;
    rtcInitIntersectContext(&context);
    rtcIntersect1(scene, &context, &rayhit);

    if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) 
      return true;
    else
      return false;
  }

  inline pcl::PointCloud<pcl::PointXYZ>::Ptr getAABB(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const Eigen::Vector3d& min_bound, const Eigen::Vector3d& max_bound)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZ>());
 
    if (cloud->points.size() == 0) return cluster;

    // x-axis
    float x_low = min_bound(0), x_up = max_bound(0);
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new
        pcl::FieldComparison<pcl::PointXYZ>("x", pcl::ComparisonOps::GE, x_low)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new
        pcl::FieldComparison<pcl::PointXYZ>("x", pcl::ComparisonOps::LT, x_up)));
    // y-axis
    float y_low = min_bound(1), y_up = max_bound(1);
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new
        pcl::FieldComparison<pcl::PointXYZ>("y", pcl::ComparisonOps::GE, y_low)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new
        pcl::FieldComparison<pcl::PointXYZ>("y", pcl::ComparisonOps::LT, y_up)));
    // z-axis
    float z_low = min_bound(2), z_up = max_bound(2);
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new 
        pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::GE, z_low)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new
        pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::LT, z_up)));
    
    pcl::ConditionalRemoval<pcl::PointXYZ> condrem;
    condrem.setCondition(range_cond);
    condrem.setInputCloud(cloud);
    condrem.setKeepOrganized(false); // Remove the points out of range
    // Apply filter
    condrem.filter(*cluster); 

    return cluster;
  }

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

  inline bool visibilityCheck(const Eigen::Vector3d& viewpoint, const Eigen::Vector3d& surface_pt, const double& res, const RTCScene& scene)
  {
    Eigen::Vector3d vis_dir = (viewpoint - surface_pt).normalized();
    Eigen::Vector3d inflate_pt = surface_pt + vis_dir * res;
    bool visible = rayTracing(viewpoint, inflate_pt, scene);

    return visible;
  }

  inline void pcl2Eigen(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcl, Eigen::Matrix4Xd& pcl_eigen)
  {
    pcl_eigen = Eigen::Matrix4Xd::Zero(4, pcl->points.size());
    for (int i=0; i<(int)pcl->points.size(); ++i)
    {
      pcl_eigen(0, i) = pcl->points[i].x;
      pcl_eigen(1, i) = pcl->points[i].y;
      pcl_eigen(2, i) = pcl->points[i].z;
      pcl_eigen(3, i) = 1.0;
    }

    return;
  }

  inline vector<RTCHit> getAllIntersections(RTCScene scene, RTCRay ray)
  {
    vector<RTCHit> intersections;

    int check_times = 0;

    while (true) 
    {
      RTCHit hit;
      hit.geomID = RTC_INVALID_GEOMETRY_ID;

      RTCRayHit rayhit;
      rayhit.ray = ray;
      rayhit.hit = hit;

      RTCIntersectContext context;
      rtcInitIntersectContext(&context);
      rtcIntersect1(scene, &context, &rayhit);
      
      // remove the point if it is too close to the surface
      if (rayhit.ray.tfar < 1e-2 && check_times == 0)
        break;

      if (rayhit.ray.tfar < 1e-3)
        break;

      // No more intersections
      if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) 
        break;

      // Add the intersection to the list
      intersections.push_back(rayhit.hit);

      // Move the ray origin to the intersection point and continue
      ray.org_x += (1+1e-5)*rayhit.ray.tfar * rayhit.ray.dir_x;
      ray.org_y += (1+1e-5)*rayhit.ray.tfar * rayhit.ray.dir_y;
      ray.org_z += (1+1e-5)*rayhit.ray.tfar * rayhit.ray.dir_z;
      ray.tnear = 0;
      ray.tfar = INFINITY;

      check_times++;
    }

    return intersections;
  }

  inline bool boundlessTracing(Eigen::Vector3d& point, Eigen::Vector3f& direction, RTCScene& scene, bool en_one_check)
  {
    int num_intersections = 0;

    RTCRay ray;
    ray.org_x = point(0);
    ray.org_y = point(1);
    ray.org_z = point(2);
    ray.dir_x = direction(0);
    ray.dir_y = direction(1);
    ray.dir_z = direction(2);
    ray.tnear = 0;
    ray.tfar = INFINITY;
    ray.time = 0.0f;
    ray.mask = -1;
    ray.id = 0;
    ray.flags = 0;

    vector<RTCHit> intersections = getAllIntersections(scene, ray);
    num_intersections += (int)intersections.size();

    if (!en_one_check)
    {
      if (num_intersections % 2 != 0)
      {
        return true;
      }
      else
      {
        return false;
      }
    }
    else
    {
      if (num_intersections % 2 > 0)
      {
        return true;
      }
      else
      {
        return false;
      }
    }
  }

  inline bool insideCheck(Eigen::Vector3d& pos, double pitch, double yaw, RTCScene& scene, bool en_one_check)
  {
    // 16 directions
    double pitch_real = 0.0;
    int N = 16, threshold = 3, out_count = 0;
    for (int i=0; i<N; ++i)
    {
      double new_yaw = yaw + i * 2 * M_PI / N;
      Eigen::Vector2d new_py(pitch_real, new_yaw);
      Eigen::Vector3d new_dir = pyToVec(new_py);
      new_dir(2) = 0.0;
      new_dir.normalize();
      Eigen::Vector3f new_dir_f(new_dir(0), new_dir(1), new_dir(2));
      
      bool inside = boundlessTracing(pos, new_dir_f, scene, en_one_check);

      if (inside == false)
      {
        out_count++;
      }

      // if (out_count > threshold)
      // {
      //   return false;
      // }
    }

    if (out_count > threshold)
    {
      return false;
    }

    return true;
  }

  inline bool inlierCheck(Eigen::Vector3d& pos, RTCScene& scene)
  {
    vector<int> nearest(1);
    vector<float> nn_squared_distance(1);
    pcl::PointXYZ check_pt;
    check_pt.x = pos(0); check_pt.y = pos(1); check_pt.z = pos(2);

    inliers_kdtree_.nearestKSearch(check_pt, 1, nearest, nn_squared_distance);
    Eigen::Vector3d nearest_pos(inliers_kdtree_ptr->points[nearest[0]].x, inliers_kdtree_ptr->points[nearest[0]].y, inliers_kdtree_ptr->points[nearest[0]].z);

    bool no_collision = rayTracing(pos, nearest_pos, scene);

    return no_collision;
  }

  inline bool getFirstIntersection(const Eigen::Vector3d& start, const Eigen::Vector3d& end, RTCScene scene, Eigen::Vector3d& intersectionPoint)
  {
    RTCRayHit rayhit;
    rayhit.ray.org_x = static_cast<float>(start(0));
    rayhit.ray.org_y = static_cast<float>(start(1));
    rayhit.ray.org_z = static_cast<float>(start(2));
    rayhit.ray.dir_x = static_cast<float>(end(0) - start(0));
    rayhit.ray.dir_y = static_cast<float>(end(1) - start(1));
    rayhit.ray.dir_z = static_cast<float>(end(2) - start(2));

    const float dir_length = sqrt(rayhit.ray.dir_x * rayhit.ray.dir_x +
                                  rayhit.ray.dir_y * rayhit.ray.dir_y +
                                  rayhit.ray.dir_z * rayhit.ray.dir_z);
    
    rayhit.ray.dir_x /= dir_length;
    rayhit.ray.dir_y /= dir_length;
    rayhit.ray.dir_z /= dir_length;

    rayhit.ray.tnear = 0.0f;                    
    rayhit.ray.tfar = dir_length;               
    rayhit.ray.time = 0.0f;                    
    rayhit.ray.mask = -1;                       
    rayhit.ray.id = 0;                          
    rayhit.ray.flags = 0;                      
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

    RTCIntersectContext context;
    rtcInitIntersectContext(&context);

    rtcIntersect1(scene, &context, &rayhit);

    if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) return false;

    intersectionPoint(0) = start(0) + rayhit.ray.tfar * rayhit.ray.dir_x;
    intersectionPoint(1) = start(1) + rayhit.ray.tfar * rayhit.ray.dir_y;
    intersectionPoint(2) = start(2) + rayhit.ray.tfar * rayhit.ray.dir_z;

    return true;
  }

}

#endif