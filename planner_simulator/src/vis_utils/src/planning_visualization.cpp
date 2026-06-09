/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Apr. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the main functions of visualization tools for FlyCo.
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

#include <vis_utils/planning_visualization.h> 
#include <cmath>

using std::cout;
using std::endl;
namespace {
ros::Publisher& inlierCandidatesPublisher(ros::NodeHandle& node)
{
  static ros::Publisher pub = node.advertise<sensor_msgs::PointCloud2>("/skeleton_decomp_debug/inlier_candidates", 1);
  return pub;
}

ros::Publisher& strictInlierCandidatesPublisher(ros::NodeHandle& node)
{
  static ros::Publisher pub = node.advertise<sensor_msgs::PointCloud2>("/skeleton_decomp_debug/inlier_candidates_strict", 1);
  return pub;
}

ros::Publisher& relaxedInlierCandidatesPublisher(ros::NodeHandle& node)
{
  static ros::Publisher pub = node.advertise<sensor_msgs::PointCloud2>("/skeleton_decomp_debug/inlier_candidates_relaxed", 1);
  return pub;
}

ros::Publisher& rejectedInlierCandidatesPublisher(ros::NodeHandle& node)
{
  static ros::Publisher pub = node.advertise<sensor_msgs::PointCloud2>("/skeleton_decomp_debug/inlier_candidates_rejected", 1);
  return pub;
}

ros::Publisher& dcrosaTopologyVerticesPublisher(ros::NodeHandle& node)
{
  static ros::Publisher pub = node.advertise<sensor_msgs::PointCloud2>("/skeleton_decomp_debug/dcrosa_topo_vertices", 1);
  return pub;
}

ros::Publisher& dcrosaTopologyEdgesPublisher(ros::NodeHandle& node)
{
  static ros::Publisher pub = node.advertise<visualization_msgs::Marker>("/skeleton_decomp_debug/dcrosa_topo_edges", 1);
  return pub;
}

ros::Publisher& dcrosaTopologyBranchesPublisher(ros::NodeHandle& node)
{
  static ros::Publisher pub = node.advertise<visualization_msgs::MarkerArray>("/skeleton_decomp_debug/dcrosa_topo_branches", 1);
  return pub;
}

void publishEigenPointCloud(const Eigen::MatrixXd& points, ros::Publisher& pub)
{
  pcl::PointCloud<pcl::PointXYZ> cloud;
  if (points.cols() < 3)
    return;

  cloud.points.reserve(points.rows());
  pcl::PointXYZ pt;
  for (Eigen::Index i=0; i<points.rows(); ++i)
  {
    if (!std::isfinite(points(i,0)) || !std::isfinite(points(i,1)) || !std::isfinite(points(i,2)))
      continue;

    pt.x = points(i,0);
    pt.y = points(i,1);
    pt.z = points(i,2);
    cloud.points.push_back(pt);
  }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = "world";

  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  pub.publish(cloud_msg);
}
}

namespace flyco {
PlanningVisualization::PlanningVisualization(ros::NodeHandle& nh) {
  node = nh;
  inlierCandidatesPublisher(node);
  strictInlierCandidatesPublisher(node);
  relaxedInlierCandidatesPublisher(node);
  rejectedInlierCandidatesPublisher(node);
  dcrosaTopologyVerticesPublisher(node);
  dcrosaTopologyEdgesPublisher(node);
  dcrosaTopologyBranchesPublisher(node);

  /* ROSA */
  pcloud_pub_ = node.advertise<sensor_msgs::PointCloud2>("/skeleton_decomp_vis/input_cloud", 10);
  mesh_pub_ = node.advertise<visualization_msgs::Marker>("/skeleton_decomp_vis/input_mesh", 1);
  normal_pub_ = node.advertise<visualization_msgs::MarkerArray>("/skeleton_decomp_vis/input_normal", 10);
  rosa_orientation_pub_ = node.advertise<visualization_msgs::MarkerArray>("/skeleton_decomp_vis/rosa_orientation", 10);
  drosa_pub_ = node.advertise<sensor_msgs::PointCloud2>("/skeleton_decomp_vis/drosa_pts", 10);
  decomp_pub_ = node.advertise<visualization_msgs::MarkerArray>("/skeleton_decomp_vis/branches", 10);
  sub_space_pub_ = node.advertise<sensor_msgs::PointCloud2>("/skeleton_decomp_vis/sub_space", 1);

  /* ROSA Debug Vis */
  inliers_pub_ = node.advertise<sensor_msgs::PointCloud2>("/skeleton_decomp_debug/inliers", 1);
  inliers_intersection_pub_ = node.advertise<visualization_msgs::MarkerArray>("/skeleton_decomp_debug/inliers_intersec", 1);
  inliers_isec_pub_ = node.advertise<sensor_msgs::PointCloud2>("/skeleton_decomp_debug/isec", 1);
  vis_graph_pub_ = node.advertise<visualization_msgs::Marker>("/skeleton_decomp_debug/vis_graph", 1);
  subspace_cluster_pub_ = node.advertise<visualization_msgs::MarkerArray>("/skeleton_decomp_debug/subspace_cluster", 10);
  intra_edge_pub_ = node.advertise<visualization_msgs::Marker>("/skeleton_decomp_debug/intra_edges", 10);
  subspace_skeleton_pub_ = node.advertise<visualization_msgs::MarkerArray>("/skeleton_decomp_debug/subspace_skeleton", 10);

  /* Global Planner */
  init_vps_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/init_vps", 1);
  hcopp_occ_pub_ = node.advertise<sensor_msgs::PointCloud2>("/hcopp/occupied", 1);
  hcopp_internal_pub_ = node.advertise<sensor_msgs::PointCloud2>("/hcopp/internal", 1);
  hcopp_uncovered_pub_ = node.advertise<sensor_msgs::PointCloud2>("/hcopp/uncovered_area", 1);
  hcopp_global_uncovered_pub_ = node.advertise<sensor_msgs::PointCloud2>("/hcopp/global_uncovered_area", 1);
  hcopp_correctnormal_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/correctNormals", 10);
  hcopp_sub_finalvps_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/sub_finalvps", 1);
  hcopp_globalseq_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/global_seq", 1);
  hcopp_globalboundary_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/global_boundary", 1);
  hcopp_local_path_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/local_paths", 1);
  hcopp_full_path_pub_ = node.advertise<visualization_msgs::Marker>("/hcopp/HCOPP_Path", 1);
  hcopp_full_waypts_pub_ = node.advertise<visualization_msgs::Marker>("/hcopp/HCOPP_Waypts", 1);
  jointSphere_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/JointSphere", 1);
  hcoppYaw_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/yaw_traj_", 1);
  global_start_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/global_start_", 1);
  before_consistency_path_pub_ = node.advertise<visualization_msgs::Marker>("/hcopp/before_consistency_path", 1);
  global_next_sub_consistency_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/global_next_sub_consistency", 1);
  global_vp_vis_graph_pub_ = node.advertise<visualization_msgs::Marker>("/hcopp/vps_vis_graph", 1);
  global_vp_decomp_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/vps_decomp", 1);
  global_decomp_path_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/decomp_global_path", 1);

  /* Flight */
  exec_path_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/exec_part", 1);

  /* Local Planner */
  local_pub_ = node.advertise<visualization_msgs::MarkerArray>("/local_planning/local_tour", 10);
  localVP_pub_ = node.advertise<visualization_msgs::MarkerArray>("/local_planning/vp_local", 10);
  local_path_order_pub_ = node.advertise<visualization_msgs::MarkerArray>("/local_planning/local_tour_order", 1);
  localReg_pub_ = node.advertise<sensor_msgs::PointCloud2>("/local_planning/local_region", 10);
  localBox_pub_ = node.advertise<visualization_msgs::Marker>("/local_planning/local_box", 10);
  local_updated_pub_ = node.advertise<visualization_msgs::MarkerArray>("/local_planning/local_updated_tour", 10);
  localVP_updated_pub_ = node.advertise<visualization_msgs::MarkerArray>("/local_planning/updated_vp_local", 10);
  local_updated_path_order_pub_ = node.advertise<visualization_msgs::MarkerArray>("/local_planning/local_updated_tour_order", 1);
  local_used_g_path_pub_ = node.advertise<visualization_msgs::MarkerArray>("/local_planning/local_used_global_path", 1);
  local_used_g_prior_pub_ = node.advertise<visualization_msgs::MarkerArray>("/local_planning/local_used_global_prior", 1);

  /* Prediction */
  pred_mesh_pub_ = node.advertise<visualization_msgs::Marker>("/prediction/mesh", 1);

  red_list.reserve(100);
  green_list.reserve(100);
  blue_list.reserve(100);
  for (int i = 0; i < 100; i++) 
  {
    double red = (double)rand() / (double)RAND_MAX;
    double green = (double)rand() / (double)RAND_MAX;
    double blue = (double)rand() / (double)RAND_MAX;
    red_list.push_back(red);
    green_list.push_back(green);
    blue_list.push_back(blue);
  }
}

void PlanningVisualization::publishSurface(const pcl::PointCloud<pcl::PointXYZ>& input_cloud)
{
  pcl::PointCloud<pcl::PointXYZ> cloud_pred;
  cloud_pred = input_cloud;

  cloud_pred.width = cloud_pred.points.size();
  cloud_pred.height = 1;
  cloud_pred.is_dense = true;
  cloud_pred.header.frame_id = "world";
  
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud_pred, cloud_msg);
  pcloud_pub_.publish(cloud_msg);
}

void PlanningVisualization::publishMesh(std::string& mesh)
{
  visualization_msgs::Marker meshModel;
  meshModel.header.frame_id = "world";
  meshModel.header.stamp = ros::Time::now();
  meshModel.id = 0;
  meshModel.ns = "mesh";
  meshModel.type = visualization_msgs::Marker::MESH_RESOURCE;
  meshModel.color.r = 128.0/256.0;
  meshModel.color.g = 128.0/256.0;
  meshModel.color.b = 128.0/256.0;
  meshModel.color.a = 1.0;
  meshModel.scale.x = 1.0;
  meshModel.scale.y = 1.0;
  meshModel.scale.z = 1.0;
  meshModel.pose.orientation.w = 1.0;
  meshModel.mesh_resource = "file://";
  meshModel.mesh_resource += mesh;
  meshModel.mesh_use_embedded_materials = true;

  mesh_pub_.publish(meshModel);
}

void PlanningVisualization::publishSurfaceNormal(const pcl::PointCloud<pcl::PointXYZ>& input_cloud, const pcl::PointCloud<pcl::Normal>& normals)
{
  visualization_msgs::MarkerArray pcloud_normals;
  int counter = 0;
  double scale = 3.0;
  for (int i=0; i<(int)input_cloud.points.size(); ++i)
  {
    visualization_msgs::Marker nm;
    nm.header.frame_id = "world";
    nm.header.stamp = ros::Time::now();
    nm.id = counter;
    nm.type = visualization_msgs::Marker::ARROW;
    nm.action = visualization_msgs::Marker::ADD;

    nm.pose.orientation.w = 1.0;
    nm.scale.x = 0.2;
    nm.scale.y = 0.3;
    nm.scale.z = 0.2;

    geometry_msgs::Point pt_;
    pt_.x = input_cloud.points[i].x;
    pt_.y = input_cloud.points[i].y;
    pt_.z = input_cloud.points[i].z;
    nm.points.push_back(pt_);

    pt_.x = input_cloud.points[i].x + scale*normals.points[i].normal_x;
    pt_.y = input_cloud.points[i].y + scale*normals.points[i].normal_y;
    pt_.z = input_cloud.points[i].z + scale*normals.points[i].normal_z;
    nm.points.push_back(pt_);

    nm.color.r = 0.1;
    nm.color.g = 0.2;
    nm.color.b = 0.7;
    nm.color.a = 1.0;
    
    pcloud_normals.markers.push_back(nm);
    counter++;
  }

  normal_pub_.publish(pcloud_normals);
}

void PlanningVisualization::publishROSAOrientation(const pcl::PointCloud<pcl::PointXYZ>& input_cloud, const pcl::PointCloud<pcl::Normal>& normals)
{
  visualization_msgs::MarkerArray pcloud_normals;
  int counter = 0;
  double scale = 3.0;
  for (int i=0; i<(int)input_cloud.points.size(); ++i)
  {
    visualization_msgs::Marker nm;
    nm.header.frame_id = "world";
    nm.header.stamp = ros::Time::now();
    nm.id = counter;
    nm.type = visualization_msgs::Marker::ARROW;
    nm.action = visualization_msgs::Marker::ADD;

    nm.pose.orientation.w = 1.0;
    nm.scale.x = 0.2;
    nm.scale.y = 0.3;
    nm.scale.z = 0.2;

    geometry_msgs::Point pt_;
    pt_.x = input_cloud.points[i].x;
    pt_.y = input_cloud.points[i].y;
    pt_.z = input_cloud.points[i].z;
    nm.points.push_back(pt_);

    pt_.x = input_cloud.points[i].x + scale*normals.points[i].normal_x;
    pt_.y = input_cloud.points[i].y + scale*normals.points[i].normal_y;
    pt_.z = input_cloud.points[i].z + scale*normals.points[i].normal_z;
    nm.points.push_back(pt_);

    nm.color.r = 0.2;
    nm.color.g = 0.7;
    nm.color.b = 0.1;
    nm.color.a = 1.0;
    
    pcloud_normals.markers.push_back(nm);
    counter++;
  }

  rosa_orientation_pub_.publish(pcloud_normals);
}

void PlanningVisualization::publish_dROSA(const pcl::PointCloud<pcl::PointXYZ>& local_region)
{
  pcl::PointCloud<pcl::PointXYZ> cloud_pred;
  cloud_pred = local_region;

  cloud_pred.width = cloud_pred.points.size();
  cloud_pred.height = 1;
  cloud_pred.is_dense = true;
  cloud_pred.header.frame_id = "world";
  
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud_pred, cloud_msg);
  drosa_pub_.publish(cloud_msg);
}

void PlanningVisualization::publishInliers(const pcl::PointCloud<pcl::PointXYZ>& inliers, const pcl::PointCloud<pcl::PointXYZ>& all_pts, const Eigen::MatrixXf& directions, const vector<vector<Eigen::Vector3d>>& isec_pts)
{
  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud = inliers;

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = "world";

  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  inliers_pub_.publish(cloud_msg);
  
  // intersection
  visualization_msgs::MarkerArray inliers_inter;
  int counter = 0;
  double scale = 1000.0;
  for (int i=0; i<(int)all_pts.points.size(); ++i)
  {
    visualization_msgs::Marker nm;
    nm.header.frame_id = "world";
    nm.header.stamp = ros::Time::now();
    nm.id = counter;
    nm.type = visualization_msgs::Marker::ARROW;
    nm.action = visualization_msgs::Marker::ADD;

    nm.pose.orientation.w = 1.0;
    nm.scale.x = 0.2;
    nm.scale.y = 0.3;
    nm.scale.z = 0.2;

    geometry_msgs::Point pt_;
    pt_.x = all_pts.points[i].x;
    pt_.y = all_pts.points[i].y;
    pt_.z = all_pts.points[i].z;
    nm.points.push_back(pt_);

    pt_.x = all_pts.points[i].x + scale*directions(i,0);
    pt_.y = all_pts.points[i].y + scale*directions(i,1);
    pt_.z = all_pts.points[i].z + scale*directions(i,2);
    nm.points.push_back(pt_);

    nm.color.r = 0.1;
    nm.color.g = 0.2;
    nm.color.b = 0.8;
    nm.color.a = 0.8;

    inliers_inter.markers.push_back(nm);
    counter++;
  }

  inliers_intersection_pub_.publish(inliers_inter);

  pcl::PointCloud<pcl::PointXYZ> cloud_isec;
  pcl::PointXYZ p;
  for (int i=0; i<(int)isec_pts.size(); ++i)
  {
    for (int j=0; j<(int)isec_pts[i].size(); ++j)
    {
      p.x = isec_pts[i][j](0);
      p.y = isec_pts[i][j](1);
      p.z = isec_pts[i][j](2);
      cloud_isec.points.push_back(p);
    }
  }

  cloud_isec.width = cloud_isec.points.size();
  cloud_isec.height = 1;
  cloud_isec.is_dense = true;
  cloud_isec.header.frame_id = "world";

  sensor_msgs::PointCloud2 cloud_isec_msg;
  pcl::toROSMsg(cloud_isec, cloud_isec_msg);
  inliers_isec_pub_.publish(cloud_isec_msg);
}


void PlanningVisualization::publishInlierCandidates(const Eigen::MatrixXd& candidates)
{
  publishEigenPointCloud(candidates, inlierCandidatesPublisher(node));
}

void PlanningVisualization::publishInlierCandidateClasses(const Eigen::MatrixXd& strict_candidates,
                                                          const Eigen::MatrixXd& relaxed_candidates,
                                                          const Eigen::MatrixXd& rejected_candidates)
{
  publishEigenPointCloud(strict_candidates, strictInlierCandidatesPublisher(node));
  publishEigenPointCloud(relaxed_candidates, relaxedInlierCandidatesPublisher(node));
  publishEigenPointCloud(rejected_candidates, rejectedInlierCandidatesPublisher(node));
}

void PlanningVisualization::publishVisGraph(const vector<Eigen::Vector3d>& inliers, const Eigen::MatrixXi& vis_graph, const vector<vector<int>>& subspace_sets)
{
  visualization_msgs::Marker lines;
  lines.header.frame_id = "world";
  lines.header.stamp = ros::Time::now();
  lines.id = 0;
  lines.ns = "visibility_graph";
  lines.type = visualization_msgs::Marker::LINE_LIST;
  lines.action = visualization_msgs::Marker::ADD;

  lines.pose.orientation.w = 1.0;
  lines.scale.x = 0.07;

  lines.color.r = 0.8;
  lines.color.g = 0.2;
  lines.color.b = 0.2;
  lines.color.a = 0.8;

  geometry_msgs::Point p1, p2;
  for (int i=0; i<vis_graph.rows(); ++i)
  {
    for (int j=0; j<vis_graph.cols(); ++j)
    {
      if (vis_graph(i,j) == 1 && i > j)
      {
        if (i >= (int)inliers.size() || j >= (int)inliers.size())
          continue;
        p1.x = inliers[i](0); p1.y = inliers[i](1); p1.z = inliers[i](2);
        p2.x = inliers[j](0); p2.y = inliers[j](1); p2.z = inliers[j](2);
        lines.points.push_back(p1);
        lines.points.push_back(p2);
      }
    }
  }

  vis_graph_pub_.publish(lines);
  
  visualization_msgs::MarkerArray subspace_cluster;
  int counter = 0;
  for (int i=0; i<(int)subspace_sets.size(); ++i)
  {
    vector<int> x = subspace_sets[i];
    for (auto idx:x)
    {
      if (idx < 0 || idx >= (int)inliers.size())
        continue;
      visualization_msgs::Marker mk;
      mk.header.frame_id = "world";
      mk.header.stamp = ros::Time::now();
      mk.id = counter;
      mk.ns = "subspace_cluster";
      mk.type = visualization_msgs::Marker::SPHERE;
      mk.action = visualization_msgs::Marker::ADD;

      mk.pose.orientation.w = 1.0;
      mk.scale.x = 1.5;
      mk.scale.y = 1.5;
      mk.scale.z = 1.5;

      mk.color.r = red_list[i % red_list.size()];
      mk.color.g = green_list[i % green_list.size()];
      mk.color.b = blue_list[i % blue_list.size()];
      mk.color.a = 0.8;

      mk.pose.position.x = inliers[idx](0);
      mk.pose.position.y = inliers[idx](1);
      mk.pose.position.z = inliers[idx](2);
      subspace_cluster.markers.push_back(mk);
      counter++;
    }
  }
  
  subspace_cluster_pub_.publish(subspace_cluster);

  return;
}

void PlanningVisualization::publishIntraEdge(const vector<Eigen::MatrixX3d>& intra_edges)
{
  visualization_msgs::Marker lines;
  lines.header.frame_id = "world";
  lines.header.stamp = ros::Time::now();
  lines.id = 0;
  lines.type = visualization_msgs::Marker::LINE_LIST;
  lines.action = visualization_msgs::Marker::ADD;

  lines.pose.orientation.w = 1.0;
  lines.scale.x = 0.15;

  lines.color.r = 0.2;
  lines.color.g = 0.8;
  lines.color.b = 0.4;
  lines.color.a = 1.0;

  geometry_msgs::Point p1, p2;
  for (int i=0; i<(int)intra_edges.size(); ++i)
  {
    p1.x = intra_edges[i](0,0); p1.y = intra_edges[i](0,1); p1.z = intra_edges[i](0,2);
    p2.x = intra_edges[i](1,0); p2.y = intra_edges[i](1,1); p2.z = intra_edges[i](1,2);
    lines.points.push_back(p1);
    lines.points.push_back(p2);
  }

  intra_edge_pub_.publish(lines);
}

void PlanningVisualization::publishSubspaceSkeleton(const vector<vector<Eigen::Vector3d>>& skeleton_paths)
{
  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker lines;
  lines.header.frame_id = "world";
  lines.header.stamp = ros::Time::now();
  lines.id = 0;
  lines.ns = "subspace_skeleton_lines";
  lines.type = visualization_msgs::Marker::LINE_LIST;
  lines.action = visualization_msgs::Marker::ADD;
  lines.pose.orientation.w = 1.0;
  lines.scale.x = 0.22;
  lines.color.r = 1.0;
  lines.color.g = 0.85;
  lines.color.b = 0.05;
  lines.color.a = 1.0;

  visualization_msgs::Marker endpoints;
  endpoints.header = lines.header;
  endpoints.id = 1;
  endpoints.ns = "subspace_skeleton_endpoints";
  endpoints.type = visualization_msgs::Marker::SPHERE_LIST;
  endpoints.action = visualization_msgs::Marker::ADD;
  endpoints.pose.orientation.w = 1.0;
  endpoints.scale.x = 0.8;
  endpoints.scale.y = 0.8;
  endpoints.scale.z = 0.8;
  endpoints.color.r = 1.0;
  endpoints.color.g = 0.45;
  endpoints.color.b = 0.0;
  endpoints.color.a = 1.0;

  geometry_msgs::Point p1, p2;
  for (const auto& path:skeleton_paths)
  {
    if (path.size() < 2)
      continue;
    for (int i=1; i<(int)path.size(); ++i)
    {
      if (!path[i-1].allFinite() || !path[i].allFinite())
        continue;
      p1.x = path[i-1](0); p1.y = path[i-1](1); p1.z = path[i-1](2);
      p2.x = path[i](0);   p2.y = path[i](1);   p2.z = path[i](2);
      lines.points.push_back(p1);
      lines.points.push_back(p2);
    }

    if (path.front().allFinite())
    {
      p1.x = path.front()(0); p1.y = path.front()(1); p1.z = path.front()(2);
      endpoints.points.push_back(p1);
    }
    if (path.back().allFinite())
    {
      p2.x = path.back()(0); p2.y = path.back()(1); p2.z = path.back()(2);
      endpoints.points.push_back(p2);
    }
  }

  marker_array.markers.push_back(lines);
  marker_array.markers.push_back(endpoints);
  subspace_skeleton_pub_.publish(marker_array);
}

void PlanningVisualization::publishDcrosaTopology(const Eigen::MatrixXd& skelver,
                                                  const Eigen::MatrixXi& skeladj,
                                                  const vector<vector<Eigen::Vector3d>>& skeleton_paths)
{
  publishEigenPointCloud(skelver, dcrosaTopologyVerticesPublisher(node));

  visualization_msgs::Marker edges;
  edges.header.frame_id = "world";
  edges.header.stamp = ros::Time::now();
  edges.id = 0;
  edges.ns = "dcrosa_topology_edges";
  edges.type = visualization_msgs::Marker::LINE_LIST;
  edges.action = visualization_msgs::Marker::ADD;
  edges.pose.orientation.w = 1.0;
  edges.scale.x = 0.08;
  edges.color.r = 0.05;
  edges.color.g = 0.9;
  edges.color.b = 1.0;
  edges.color.a = 0.9;

  geometry_msgs::Point p1, p2;
  if (skelver.cols() >= 3 && skeladj.rows() == skeladj.cols())
  {
    for (int i=0; i<skeladj.rows() && i<skelver.rows(); ++i)
    {
      for (int j=i+1; j<skeladj.cols() && j<skelver.rows(); ++j)
      {
        if (skeladj(i,j) != 1)
          continue;

        if (!std::isfinite(skelver(i,0)) || !std::isfinite(skelver(i,1)) || !std::isfinite(skelver(i,2)) ||
            !std::isfinite(skelver(j,0)) || !std::isfinite(skelver(j,1)) || !std::isfinite(skelver(j,2)))
          continue;

        p1.x = skelver(i,0); p1.y = skelver(i,1); p1.z = skelver(i,2);
        p2.x = skelver(j,0); p2.y = skelver(j,1); p2.z = skelver(j,2);
        edges.points.push_back(p1);
        edges.points.push_back(p2);
      }
    }
  }
  dcrosaTopologyEdgesPublisher(node).publish(edges);

  visualization_msgs::MarkerArray branch_markers;
  for (int i=0; i<(int)skeleton_paths.size(); ++i)
  {
    const auto& path = skeleton_paths[i];
    if ((int)path.size() < 2)
      continue;

    visualization_msgs::Marker branch;
    branch.header.frame_id = "world";
    branch.header.stamp = ros::Time::now();
    branch.id = i;
    branch.ns = "dcrosa_topology_branches";
    branch.type = visualization_msgs::Marker::LINE_STRIP;
    branch.action = visualization_msgs::Marker::ADD;
    branch.pose.orientation.w = 1.0;
    branch.scale.x = 0.16;
    branch.color.r = red_list[i % red_list.size()];
    branch.color.g = green_list[i % green_list.size()];
    branch.color.b = blue_list[i % blue_list.size()];
    branch.color.a = 1.0;

    for (const Eigen::Vector3d& point:path)
    {
      if (!point.allFinite())
        continue;
      geometry_msgs::Point marker_pt;
      marker_pt.x = point(0);
      marker_pt.y = point(1);
      marker_pt.z = point(2);
      branch.points.push_back(marker_pt);
    }

    if ((int)branch.points.size() >= 2)
      branch_markers.markers.push_back(branch);
  }
  dcrosaTopologyBranchesPublisher(node).publish(branch_markers);
}

void PlanningVisualization::publishSubSpace(vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& space)
{
  srand((int)time(0));
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_pcl_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointXYZRGB p;
  int red, blue, green;
  for (int i=0; i<(int)space.size(); ++i)
  {
    red = rand()%255;
    blue = rand()%255;
    green = rand()%255;
    for (int j=0; j<(int)space[i]->points.size(); ++j)
    {
      p.x = space[i]->points[j].x;
      p.y = space[i]->points[j].y;
      p.z = space[i]->points[j].z;
      // color
      p.r = red;
      p.g = green;
      p.b = blue;
      colored_pcl_ptr->points.push_back(p);
    }
  }

  colored_pcl_ptr->width = colored_pcl_ptr->points.size();
  colored_pcl_ptr->height = 1;
  colored_pcl_ptr->is_dense = true;
  colored_pcl_ptr->header.frame_id = "world";

  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*colored_pcl_ptr, cloud_msg);
  sub_space_pub_.publish(cloud_msg);
}

void PlanningVisualization::publishOccupied(pcl::PointCloud<pcl::PointXYZ>& occupied)
{
  pcl::PointCloud<pcl::PointXYZ> cloud_pred;
  cloud_pred = occupied;

  cloud_pred.width = cloud_pred.points.size();
  cloud_pred.height = 1;
  cloud_pred.is_dense = true;
  cloud_pred.header.frame_id = "world";
  
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud_pred, cloud_msg);
  hcopp_occ_pub_.publish(cloud_msg);
}

void PlanningVisualization::publishInternal(pcl::PointCloud<pcl::PointXYZ>& internal)
{
  pcl::PointCloud<pcl::PointXYZ> cloud_pred;
  cloud_pred = internal;

  cloud_pred.width = cloud_pred.points.size();
  cloud_pred.height = 1;
  cloud_pred.is_dense = true;
  cloud_pred.header.frame_id = "world";
  
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud_pred, cloud_msg);
  hcopp_internal_pub_.publish(cloud_msg);
}

void PlanningVisualization::publishUncovered(pcl::PointCloud<pcl::PointXYZ>& uncovered)
{
  pcl::PointCloud<pcl::PointXYZ> cloud_pred;
  cloud_pred = uncovered;

  cloud_pred.width = cloud_pred.points.size();
  cloud_pred.height = 1;
  cloud_pred.is_dense = true;
  cloud_pred.header.frame_id = "world";
  
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud_pred, cloud_msg);
  hcopp_uncovered_pub_.publish(cloud_msg);
}

void PlanningVisualization::publishGlobalUncovered(pcl::PointCloud<pcl::PointXYZ>& uncovered)
{
  pcl::PointCloud<pcl::PointXYZ> cloud_pred;
  cloud_pred = uncovered;

  cloud_pred.width = cloud_pred.points.size();
  cloud_pred.height = 1;
  cloud_pred.is_dense = true;
  cloud_pred.header.frame_id = "world";
  
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud_pred, cloud_msg);
  hcopp_global_uncovered_pub_.publish(cloud_msg);
}

void PlanningVisualization::publishRevisedNormal(const pcl::PointCloud<pcl::PointXYZ>& input_cloud, const pcl::PointCloud<pcl::Normal>& normals)
{
  visualization_msgs::MarkerArray pcloud_normals;
  int counter = 0;
  double scale = 5.0;
  for (int i=0; i<(int)input_cloud.points.size(); ++i)
  {
    visualization_msgs::Marker nm;
    nm.header.frame_id = "world";
    nm.header.stamp = ros::Time::now();
    nm.id = counter;
    nm.type = visualization_msgs::Marker::ARROW;
    nm.action = visualization_msgs::Marker::ADD;

    nm.pose.orientation.w = 1.0;
    nm.scale.x = 0.2;
    nm.scale.y = 0.3;
    nm.scale.z = 0.2;

    geometry_msgs::Point pt_;
    pt_.x = input_cloud.points[i].x;
    pt_.y = input_cloud.points[i].y;
    pt_.z = input_cloud.points[i].z;
    nm.points.push_back(pt_);

    pt_.x = input_cloud.points[i].x + scale*normals.points[i].normal_x;
    pt_.y = input_cloud.points[i].y + scale*normals.points[i].normal_y;
    pt_.z = input_cloud.points[i].z + scale*normals.points[i].normal_z;
    nm.points.push_back(pt_);

    nm.color.r = 0.1;
    nm.color.g = 0.2;
    nm.color.b = 0.7;
    nm.color.a = 1.0;
    
    pcloud_normals.markers.push_back(nm);
    counter++;
  }

  hcopp_correctnormal_pub_.publish(pcloud_normals);
}

void PlanningVisualization::publishFinalFOV(map<int, vector<vector<Eigen::Vector3d>>>& list1, map<int, vector<vector<Eigen::Vector3d>>>& list2, map<int, vector<double>>& yaws)
{
  srand((int)time(0));
  int red, blue, green;
  visualization_msgs::MarkerArray final_vps;
  visualization_msgs::MarkerArray vps_drones;
  int counter = 0;
  int sub_id;
  for (auto& sub_fov:list1)
  {
    sub_id = sub_fov.first;
    red = rand()%255;
    blue = rand()%255;
    green = rand()%255;

    for (int j=0; j<(int)sub_fov.second.size(); ++j)
    {
      visualization_msgs::Marker mk;
      mk.header.frame_id = "world";
      mk.header.stamp = ros::Time::now();
      mk.id = counter;
      mk.ns = "sub_viewpoints";
      mk.type = visualization_msgs::Marker::LINE_LIST;
      mk.color.r = red;
      mk.color.g = blue;
      mk.color.b = green;
      mk.color.a = 1.0;
      mk.scale.x = 0.15;
      mk.scale.y = 0.15;
      mk.scale.z = 0.15;
      mk.pose.orientation.w = 1.0;

      // visualization_msgs::Marker meshROS;
      // meshROS.header.frame_id = "world";
      // meshROS.header.stamp = ros::Time::now();
      // meshROS.id = counter;
      // meshROS.ns = "drones_mesh";
      // meshROS.type = visualization_msgs::Marker::MESH_RESOURCE;
      // meshROS.color.r = 0.8;
      // meshROS.color.g = 0.0;
      // meshROS.color.b = 0.5;
      // meshROS.color.a = 1.0;
      // meshROS.scale.x = 8.0;
      // meshROS.scale.y = 8.0;
      // meshROS.scale.z = 9.0;
      // meshROS.pose.orientation.w = 1.0;
      // meshROS.mesh_resource = "file://";
      // meshROS.mesh_resource += droneMesh;

      // double yaw = yaws[sub_id][j];
      // double yaw_mesh = yaw;
      // meshROS.pose.orientation.x = 0.0;
      // meshROS.pose.orientation.y = 0.0;
      // meshROS.pose.orientation.z = sin(0.5*yaw_mesh);
      // meshROS.pose.orientation.w = cos(0.5*yaw_mesh);
      // meshROS.pose.position.x = list1[sub_id][j][0](0);
      // meshROS.pose.position.y = list1[sub_id][j][0](1);
      // meshROS.pose.position.z = list1[sub_id][j][0](2);

      geometry_msgs::Point pt;
      for (int i = 0; i < int(sub_fov.second[j].size()); ++i) {
        pt.x = list1[sub_id][j][i](0);
        pt.y = list1[sub_id][j][i](1);
        pt.z = list1[sub_id][j][i](2);
        mk.points.push_back(pt);

        pt.x = list2[sub_id][j][i](0);
        pt.y = list2[sub_id][j][i](1);
        pt.z = list2[sub_id][j][i](2);
        mk.points.push_back(pt);
      }

      // vps_drones.markers.push_back(meshROS);
      final_vps.markers.push_back(mk);
      counter++;
    }
  }

  hcopp_sub_finalvps_pub_.publish(final_vps);
}

void PlanningVisualization::publishGlobalSeq(Eigen::Vector3d& start_, vector<Eigen::Vector3d>& sub_rep, vector<int>& global_seq)
{
  vector<Eigen::Vector3d> total_site_;
  total_site_.push_back(start_);
  total_site_.insert(total_site_.begin()+1, sub_rep.begin(), sub_rep.end());
  vector<int> total_seq_;
  total_seq_.push_back(0);
  for (auto x:global_seq)
    total_seq_.push_back(x+1);

  visualization_msgs::MarkerArray global_results;
  int counter = 0;

  visualization_msgs::Marker begin;
  begin.header.frame_id = "world";
  begin.header.stamp = ros::Time::now();
  begin.id = counter;
  begin.ns = "current_pose";
  begin.type = visualization_msgs::Marker::SPHERE;
  begin.color.r = 1.0;
  begin.color.g = 0.0;
  begin.color.b = 0.0;
  begin.color.a = 1.0;
  begin.scale.x = 1.5;
  begin.scale.y = 1.5;
  begin.scale.z = 1.5;
  begin.pose.orientation.w = 1.0;
  begin.pose.position.x = start_(0);
  begin.pose.position.y = start_(1);
  begin.pose.position.z = start_(2);

  global_results.markers.push_back(begin);
  counter++;

  for (int i=0; i<(int)sub_rep.size(); ++i)
  {
    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.id = counter;
    mk.ns = "current_pose";
    mk.type = visualization_msgs::Marker::CUBE;
    mk.color.r = 0.0;
    mk.color.g = 0.0;
    mk.color.b = 1.0;
    mk.color.a = 1.0;
    mk.scale.x = 1.5;
    mk.scale.y = 1.5;
    mk.scale.z = 1.5;
    mk.pose.orientation.w = 1.0;
    mk.pose.position.x = sub_rep[i](0);
    mk.pose.position.y = sub_rep[i](1);
    mk.pose.position.z = sub_rep[i](2);

    global_results.markers.push_back(mk);
    counter++;
  }

  for (int j=0; j<(int)total_seq_.size()-1; ++j)
  {
    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.id = counter;
    mk.ns = "current_pose";
    mk.type = visualization_msgs::Marker::LINE_LIST;
    mk.color.r = 0.0;
    mk.color.g = 1.0;
    mk.color.b = 0.0;
    mk.color.a = 1.0;
    mk.scale.x = 1.5;
    mk.scale.y = 1.5;
    mk.scale.z = 1.5;
    mk.pose.orientation.w = 1.0;

    geometry_msgs::Point pt;
    pt.x = total_site_[total_seq_[j]](0);
    pt.y = total_site_[total_seq_[j]](1);
    pt.z = total_site_[total_seq_[j]](2);
    mk.points.push_back(pt);

    pt.x = total_site_[total_seq_[j+1]](0);
    pt.y = total_site_[total_seq_[j+1]](1);
    pt.z = total_site_[total_seq_[j+1]](2);
    mk.points.push_back(pt);

    global_results.markers.push_back(mk);
    counter++;
  }

  hcopp_globalseq_pub_.publish(global_results);
}

void PlanningVisualization::publishGlobalBoundary(Eigen::Vector3d& start_, map<int, vector<int>>& boundary_id_, map<int, vector<Eigen::VectorXd>>& sub_vps, vector<int>& global_seq)
{
  vector<Eigen::Vector3d> boundaries;
  boundaries.push_back(start_);
  Eigen::Vector3d b_s, b_e;
  for (auto id:global_seq)
  {
    if ((int)boundary_id_.find(id)->second.size() == 2)
    {
      b_s(0) = sub_vps.find(id)->second[boundary_id_.find(id)->second[0]](0);
      b_s(1) = sub_vps.find(id)->second[boundary_id_.find(id)->second[0]](1);
      b_s(2) = sub_vps.find(id)->second[boundary_id_.find(id)->second[0]](2);
      boundaries.push_back(b_s);
      b_e(0) = sub_vps.find(id)->second[boundary_id_.find(id)->second[1]](0);
      b_e(1) = sub_vps.find(id)->second[boundary_id_.find(id)->second[1]](1);
      b_e(2) = sub_vps.find(id)->second[boundary_id_.find(id)->second[1]](2);
      boundaries.push_back(b_e);
    }
    else
    {
      b_s(0) = sub_vps.find(id)->second[boundary_id_.find(id)->second[0]](0);
      b_s(1) = sub_vps.find(id)->second[boundary_id_.find(id)->second[0]](1);
      b_s(2) = sub_vps.find(id)->second[boundary_id_.find(id)->second[0]](2);
      boundaries.push_back(b_s);
    }
  }
  
  visualization_msgs::MarkerArray boundary_results;
  int counter = 0;
  for (int i=0; i<(int)boundaries.size(); ++i)
  {
    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.id = counter;
    mk.ns = "current_pose";
    mk.type = visualization_msgs::Marker::CUBE;
    mk.color.r = 0.0;
    mk.color.g = 1.0;
    mk.color.b = 1.0;
    mk.color.a = 1.0;
    mk.scale.x = 1.2;
    mk.scale.y = 1.2;
    mk.scale.z = 1.2;
    mk.pose.orientation.w = 1.0;
    mk.pose.position.x = boundaries[i](0);
    mk.pose.position.y = boundaries[i](1);
    mk.pose.position.z = boundaries[i](2);

    boundary_results.markers.push_back(mk);
    counter++;
  }

  for (int j=0; j<(int)boundaries.size()-1; ++j)
  {
    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.id = counter;
    mk.ns = "current_pose";
    mk.type = visualization_msgs::Marker::LINE_LIST;
    mk.color.r = 1.0;
    mk.color.g = 0.0;
    mk.color.b = 0.0;
    mk.color.a = 1.0;
    mk.scale.x = 1.0;
    mk.scale.y = 1.0;
    mk.scale.z = 1.0;
    mk.pose.orientation.w = 1.0;

    geometry_msgs::Point pt;
    pt.x = boundaries[j](0);
    pt.y = boundaries[j](1);
    pt.z = boundaries[j](2);
    mk.points.push_back(pt);

    pt.x = boundaries[j+1](0);
    pt.y = boundaries[j+1](1);
    pt.z = boundaries[j+1](2);
    mk.points.push_back(pt);

    boundary_results.markers.push_back(mk);
    counter++;
  }

  hcopp_globalboundary_pub_.publish(boundary_results);
}

void PlanningVisualization::publishLocalPath(map<int, vector<Eigen::VectorXd>>& sub_paths_)
{
  srand((int)time(0));
  int red, blue, green;
  vector<Eigen::VectorXd> vps_;
  int counter = 0;
  visualization_msgs::MarkerArray local_results;
  for (const auto& pair:sub_paths_)
  {
    vps_ = pair.second;
    red = rand()%255;
    blue = rand()%255;
    green = rand()%255;

    for (int i=0; i<(int)vps_.size()-1; ++i)
    {
      visualization_msgs::Marker mk;
      mk.header.frame_id = "world";
      mk.header.stamp = ros::Time::now();
      mk.id = counter;
      mk.ns = "current_pose";
      mk.type = visualization_msgs::Marker::LINE_LIST;
      mk.color.r = red;
      mk.color.g = blue;
      mk.color.b = green;
      mk.color.a = 1.0;
      mk.scale.x = 1.0;
      mk.scale.y = 1.0;
      mk.scale.z = 1.0;
      mk.pose.orientation.w = 1.0;

      geometry_msgs::Point pt;
      pt.x = vps_[i](0);
      pt.y = vps_[i](1);
      pt.z = vps_[i](2);
      mk.points.push_back(pt);

      pt.x = vps_[i+1](0);
      pt.y = vps_[i+1](1);
      pt.z = vps_[i+1](2);
      mk.points.push_back(pt);

      local_results.markers.push_back(mk);
      counter++;
    }
  }

  hcopp_local_path_pub_.publish(local_results);
}

void PlanningVisualization::publishHCOPPPath(vector<Eigen::VectorXd>& fullpath_)
{
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.id = 0;
  mk.ns = "global_path";
  mk.type = visualization_msgs::Marker::LINE_LIST;
  mk.color.r = 0.0;
  mk.color.g = 1.0;
  mk.color.b = 0.0;
  mk.color.a = 0.5;
  mk.scale.x = 0.4;
  mk.scale.y = 0.4;
  mk.scale.z = 0.4;
  mk.pose.orientation.w = 1.0;

  mk.action = visualization_msgs::Marker::DELETEALL;
  hcopp_full_path_pub_.publish(mk);
  geometry_msgs::Point pt;
  for (int i=0; i<(int)fullpath_.size()-1; ++i)
  {
    pt.x = fullpath_[i](0);
    pt.y = fullpath_[i](1);
    pt.z = fullpath_[i](2);
    mk.points.push_back(pt);

    pt.x = fullpath_[i+1](0);
    pt.y = fullpath_[i+1](1);
    pt.z = fullpath_[i+1](2);
    mk.points.push_back(pt);
  }

  mk.action = visualization_msgs::Marker::ADD;
  hcopp_full_path_pub_.publish(mk);

  visualization_msgs::Marker mk_pts;
  mk_pts.header.frame_id = "world";
  mk_pts.header.stamp = ros::Time::now();
  mk_pts.id = 0;
  mk_pts.ns = "global_path_waypts";
  mk_pts.type = visualization_msgs::Marker::SPHERE_LIST;
  mk_pts.color.r = 0.0;
  mk_pts.color.g = 0.0;
  mk_pts.color.b = 0.0;
  mk_pts.color.a = 0.7;
  mk_pts.scale.x = 0.6;
  mk_pts.scale.y = 0.6;
  mk_pts.scale.z = 0.6;
  mk_pts.pose.orientation.w = 1.0;

  mk_pts.action = visualization_msgs::Marker::DELETEALL;
  hcopp_full_waypts_pub_.publish(mk_pts);

  for (int i=0; i<(int)fullpath_.size(); ++i)
  {
    pt.x = fullpath_[i](0);
    pt.y = fullpath_[i](1);
    pt.z = fullpath_[i](2);
    mk_pts.points.push_back(pt);
  }

  mk_pts.action = visualization_msgs::Marker::ADD;
  hcopp_full_waypts_pub_.publish(mk_pts);
}

void PlanningVisualization::publishBeforeConsistencyPath(const vector<Eigen::VectorXd>& path)
{
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.id = 0;
  mk.ns = "before_consis_path";
  mk.type = visualization_msgs::Marker::LINE_LIST;
  mk.color.r = 0.6;
  mk.color.g = 0.0;
  mk.color.b = 0.3;
  mk.color.a = 1.0;
  mk.scale.x = 0.2;
  mk.scale.y = 0.2;
  mk.scale.z = 0.2;
  mk.pose.orientation.w = 1.0;

  mk.action = visualization_msgs::Marker::DELETEALL;
  before_consistency_path_pub_.publish(mk);
  
  geometry_msgs::Point pt;
  for (int i=0; i<(int)path.size()-1; ++i)
  {
    pt.x = path[i](0);
    pt.y = path[i](1);
    pt.z = path[i](2);
    mk.points.push_back(pt);

    pt.x = path[i+1](0);
    pt.y = path[i+1](1);
    pt.z = path[i+1](2);
    mk.points.push_back(pt);
  }

  mk.action = visualization_msgs::Marker::ADD;
  before_consistency_path_pub_.publish(mk);
}

void PlanningVisualization::publishGlobalNextSubConsistency(const vector<Eigen::Vector3d>& last, const vector<Eigen::Vector3d>& current)
{
  visualization_msgs::MarkerArray two_inliers;

  double last_scale = 0.7, current_scale = 0.8;

  visualization_msgs::Marker last_marker;
  last_marker.header.frame_id = "world";
  last_marker.header.stamp = ros::Time::now();
  last_marker.id = 0;
  last_marker.ns = "last_subspace";
  last_marker.type = visualization_msgs::Marker::CUBE_LIST;
  last_marker.color.r = 0.0;
  last_marker.color.g = 0.8;
  last_marker.color.b = 0.5;
  last_marker.color.a = 1.0;
  last_marker.scale.x = last_scale;
  last_marker.scale.y = last_scale;
  last_marker.scale.z = last_scale;
  last_marker.pose.orientation.w = 1.0;

  for (int i=0; i<(int)last.size(); ++i)
  {
    geometry_msgs::Point pt;
    pt.x = last[i](0);
    pt.y = last[i](1);
    pt.z = last[i](2);
    last_marker.points.push_back(pt);
  }
  two_inliers.markers.push_back(last_marker);

  visualization_msgs::Marker current_marker;
  current_marker.header.frame_id = "world";
  current_marker.header.stamp = ros::Time::now();
  current_marker.id = 1;
  current_marker.ns = "current_subspace";
  current_marker.type = visualization_msgs::Marker::SPHERE_LIST;
  current_marker.color.r = 0.0;
  current_marker.color.g = 0.0;
  current_marker.color.b = 1.0;
  current_marker.color.a = 0.6;
  current_marker.scale.x = current_scale;
  current_marker.scale.y = current_scale;
  current_marker.scale.z = current_scale;
  current_marker.pose.orientation.w = 1.0;

  for (int i=0; i<(int)current.size(); ++i)
  {
    geometry_msgs::Point pt;
    pt.x = current[i](0);
    pt.y = current[i](1);
    pt.z = current[i](2);
    current_marker.points.push_back(pt);
  }
  two_inliers.markers.push_back(current_marker);

  global_next_sub_consistency_pub_.publish(two_inliers);

  return;
}

void PlanningVisualization::publishGlobalVPVisGraph(const vector<Eigen::VectorXd>& vps, const Eigen::MatrixXi& vp_vis_graph, const vector<vector<int>>& decomp_vps)
{
  visualization_msgs::Marker lines;
  lines.header.frame_id = "world";
  lines.header.stamp = ros::Time::now();
  lines.id = 0;
  lines.ns = "vps_vis_graph";
  lines.type = visualization_msgs::Marker::LINE_LIST;
  lines.action = visualization_msgs::Marker::ADD;

  lines.pose.orientation.w = 1.0;
  lines.scale.x = 0.05;

  lines.color.r = 0.3;
  lines.color.g = 0.1;
  lines.color.b = 0.8;
  lines.color.a = 0.5;

  geometry_msgs::Point p1, p2;
  for (int i=0; i<vp_vis_graph.rows(); ++i)
  {
    for (int j=0; j<vp_vis_graph.cols(); ++j)
    {
      if (vp_vis_graph(i,j) == 1 && i > j)
      {
        p1.x = vps[i](0); p1.y = vps[i](1); p1.z = vps[i](2);
        p2.x = vps[j](0); p2.y = vps[j](1); p2.z = vps[j](2);
        lines.points.push_back(p1);
        lines.points.push_back(p2);
      }
    }
  }

  global_vp_vis_graph_pub_.publish(lines);

  visualization_msgs::MarkerArray decomp_vps_markers;
  for (int i=0; i<(int)decomp_vps.size(); ++i)
  {
    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.id = i;
    mk.ns = "decomp_vps";
    mk.type = visualization_msgs::Marker::SPHERE_LIST;
    mk.color.r = this->red_list[i];
    mk.color.g = this->green_list[i];
    mk.color.b = this->blue_list[i];
    mk.color.a = 1.0;
    mk.scale.x = 0.8;
    mk.scale.y = 0.8;
    mk.scale.z = 0.8;
    mk.pose.orientation.w = 1.0;

    for (int j=0; j<(int)decomp_vps[i].size(); ++j)
    {
      geometry_msgs::Point pt;
      pt.x = vps[decomp_vps[i][j]](0);
      pt.y = vps[decomp_vps[i][j]](1);
      pt.z = vps[decomp_vps[i][j]](2);
      mk.points.push_back(pt);
    }

    decomp_vps_markers.markers.push_back(mk);
  }

  global_vp_decomp_pub_.publish(decomp_vps_markers);

  return;
}

void PlanningVisualization::publishDecompGlobalPath(const vector<Eigen::Vector3d>& global_path)
{
  visualization_msgs::MarkerArray global_path_markers;

  visualization_msgs::Marker pts;
  pts.header.frame_id = "world";
  pts.header.stamp = ros::Time::now();
  pts.id = 0;
  pts.ns = "decomp_global_path";
  pts.type = visualization_msgs::Marker::CUBE_LIST;
  pts.color.r = 0.0;
  pts.color.g = 0.0;
  pts.color.b = 0.0;
  pts.color.a = 0.5;
  pts.scale.x = 1.2;
  pts.scale.y = 1.2;
  pts.scale.z = 1.2;
  pts.pose.orientation.w = 1.0;

  for (int i=0; i<(int)global_path.size(); ++i)
  {
    geometry_msgs::Point pt;
    pt.x = global_path[i](0);
    pt.y = global_path[i](1);
    pt.z = global_path[i](2);
    pts.points.push_back(pt);
  }

  global_path_markers.markers.push_back(pts);

  if ((int)global_path.size() > 1)
  {
    visualization_msgs::Marker lines;
    lines.header.frame_id = "world";
    lines.header.stamp = ros::Time::now();
    lines.id = 1;
    lines.ns = "decomp_global_path";
    lines.type = visualization_msgs::Marker::LINE_LIST;
    lines.color.r = 0.5;
    lines.color.g = 0.5;
    lines.color.b = 0.0;
    lines.color.a = 0.7;
    lines.scale.x = 0.6;
    lines.scale.y = 0.6;
    lines.scale.z = 0.6;
    lines.pose.orientation.w = 1.0;

    geometry_msgs::Point p1, p2;
    for (int i=0; i<(int)global_path.size()-1; ++i)
    {
      p1.x = global_path[i](0); p1.y = global_path[i](1); p1.z = global_path[i](2);
      p2.x = global_path[i+1](0); p2.y = global_path[i+1](1); p2.z = global_path[i+1](2);
      lines.points.push_back(p1);
      lines.points.push_back(p2);
    }

    global_path_markers.markers.push_back(lines);
  }

  global_decomp_path_pub_.publish(global_path_markers);

  return;
}

void PlanningVisualization::publishJointSphere(vector<Eigen::Vector3d>& joints, double& radius, vector<vector<Eigen::Vector3d>>& InnerVps)
{
  visualization_msgs::MarkerArray JointSpheres;
  int counter = 0, vpscount = 0;

  for (int i=0; i<(int)joints.size(); ++i)
  {
    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.id = counter;
    mk.ns = "JointSphere";
    mk.type = visualization_msgs::Marker::SPHERE;
    mk.color.r = 1.0;
    mk.color.g = 0.5;
    mk.color.b = 0.0;
    mk.color.a = 0.5;
    mk.scale.x = 2*radius;
    mk.scale.y = 2*radius;
    mk.scale.z = 2*radius;
    mk.pose.orientation.w = 1.0;
    mk.pose.position.x = joints[i](0);
    mk.pose.position.y = joints[i](1);
    mk.pose.position.z = joints[i](2);

    JointSpheres.markers.push_back(mk);
    counter++;

    for (int j=0; j<(int)InnerVps[i].size(); ++j)
    {
      visualization_msgs::Marker vp;
      vp.header.frame_id = "world";
      vp.header.stamp = ros::Time::now();
      vp.id = vpscount;
      vp.ns = "JointVp";
      vp.type = visualization_msgs::Marker::CUBE;
      vp.color.r = 0.0;
      vp.color.g = 0.0;
      vp.color.b = 1.0;
      vp.color.a = 1.0;
      vp.scale.x = 1.5;
      vp.scale.y = 1.5;
      vp.scale.z = 1.5;
      vp.pose.orientation.w = 1.0;
      vp.pose.position.x = InnerVps[i][j](0);
      vp.pose.position.y = InnerVps[i][j](1);
      vp.pose.position.z = InnerVps[i][j](2);

      JointSpheres.markers.push_back(vp);
      vpscount++;
    }
  }

  jointSphere_pub_.publish(JointSpheres);
}

void PlanningVisualization::publishYawTraj(vector<Eigen::Vector3d>& waypt, vector<double>& yaw)
{
  visualization_msgs::MarkerArray yaw_traj;
  int counter = 0;
  double scale = 3.0;
  for (int i=0; i<(int)yaw.size(); ++i)
  {
    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.id = counter;
    mk.ns = "yaw_traj";
    mk.type = visualization_msgs::Marker::ARROW;
    mk.pose.orientation.w = 1.0;
    mk.scale.x = 0.2;
    mk.scale.y = 0.4;
    mk.scale.z = 0.3;
    mk.color.r = 0.0;
    mk.color.g = 0.0;
    mk.color.b = 1.0;
    mk.color.a = 1.0;

    geometry_msgs::Point pt_;
    pt_.x = waypt[i](0);
    pt_.y = waypt[i](1);
    pt_.z = waypt[i](2);
    mk.points.push_back(pt_);

    pt_.x = waypt[i](0) + scale*cos(yaw[i]);
    pt_.y = waypt[i](1) + scale*sin(yaw[i]);
    pt_.z = waypt[i](2);
    mk.points.push_back(pt_);

    yaw_traj.markers.push_back(mk);
    counter++;
  }

  hcoppYaw_pub_.publish(yaw_traj);
}

void PlanningVisualization::publishInitVps(pcl::PointCloud<pcl::PointNormal>::Ptr& init_vps)
{
  visualization_msgs::MarkerArray init_vps_markers;
  int counter = 0;
  double scale = 3.0;

  for (int i=0; i<(int)init_vps->points.size(); ++i)
  {
    visualization_msgs::Marker nm;
    nm.header.frame_id = "world";
    nm.header.stamp = ros::Time::now();
    nm.id = counter;
    nm.ns = "init_vps_dir";
    nm.type = visualization_msgs::Marker::ARROW;
    nm.action = visualization_msgs::Marker::ADD;

    nm.pose.orientation.w = 1.0;
    nm.scale.x = 0.2;
    nm.scale.y = 0.3;
    nm.scale.z = 0.2;

    geometry_msgs::Point pt_;
    pt_.x = init_vps->points[i].x;
    pt_.y = init_vps->points[i].y;
    pt_.z = init_vps->points[i].z;
    nm.points.push_back(pt_);

    pt_.x = init_vps->points[i].x + scale*init_vps->points[i].normal_x;
    pt_.y = init_vps->points[i].y + scale*init_vps->points[i].normal_y;
    pt_.z = init_vps->points[i].z + scale*init_vps->points[i].normal_z;

    nm.points.push_back(pt_);

    nm.color.r = 0.0;
    nm.color.g = 0.4;
    nm.color.b = 0.9;
    nm.color.a = 0.8;

    init_vps_markers.markers.push_back(nm);
    counter++;

    visualization_msgs::Marker pos;
    pos.header.frame_id = "world";
    pos.header.stamp = ros::Time::now();
    pos.id = counter;
    pos.ns = "init_vps_pos";
    pos.type = visualization_msgs::Marker::SPHERE;
    pos.color.r = 0.0;
    pos.color.g = 0.0;
    pos.color.b = 0.0;
    pos.color.a = 1.0;
    pos.scale.x = 0.5;
    pos.scale.y = 0.5;
    pos.scale.z = 0.5;
    pos.pose.orientation.w = 1.0;
    pos.pose.position.x = init_vps->points[i].x;
    pos.pose.position.y = init_vps->points[i].y;
    pos.pose.position.z = init_vps->points[i].z;

    init_vps_markers.markers.push_back(pos);
    counter++;
  }

  init_vps_pub_.publish(init_vps_markers);
}

void PlanningVisualization::publishLocalRegion(pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, Eigen::Vector3d& min_bound, Eigen::Vector3d& max_bound)
{
  pcl::PointCloud<pcl::PointXYZ> cloud_pred;
  for (auto p:input_cloud->points)
    cloud_pred.points.push_back(p);

  cloud_pred.width = cloud_pred.points.size();
  cloud_pred.height = 1;
  cloud_pred.is_dense = true;
  cloud_pred.header.frame_id = "world";
  
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud_pred, cloud_msg);
  localReg_pub_.publish(cloud_msg);

  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.id = 0;
  mk.ns = "local_box";
  mk.type = visualization_msgs::Marker::CUBE;
  mk.color.r = 0.5;
  mk.color.g = 1.0;
  mk.color.b = 0.5;
  mk.color.a = 0.2;
  mk.scale.x = max_bound(0) - min_bound(0);
  mk.scale.y = max_bound(1) - min_bound(1);
  mk.scale.z = max_bound(2) - min_bound(2);
  mk.pose.position.x = (max_bound(0) + min_bound(0)) / 2.0;
  mk.pose.position.y = (max_bound(1) + min_bound(1)) / 2.0;
  mk.pose.position.z = (max_bound(2) + min_bound(2)) / 2.0;
  mk.pose.orientation.w = 1.0;

  localBox_pub_.publish(mk);
}

void PlanningVisualization::publishCurPath(const vector<Eigen::VectorXd>& local_path, const vector<vector<Eigen::Vector3d>>& list1, const vector<vector<Eigen::Vector3d>>& list2)
{
  double scale_line = 0.1;
  double scale_sphere = 0.25;
  double scale_wp = 0.1;
  
  if (!path_markers_.markers.empty()) {
    for (auto& marker : path_markers_.markers) {
      marker.action = visualization_msgs::Marker::DELETE;
    }
    local_pub_.publish(path_markers_);
    path_markers_.markers.clear();
  }

  // * path
  int counter = 0;
  visualization_msgs::MarkerArray path_markers;
  
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.id = counter;
  mk.type = visualization_msgs::Marker::LINE_LIST;
  mk.color.r = 0.0;
  mk.color.g = 1.0;
  mk.color.b = 0.0;
  mk.color.a = 1.0;
  mk.scale.x = scale_line;
  mk.scale.y = scale_line;
  mk.scale.z = scale_line;
  mk.pose.orientation.w = 1.0;

  for (int i=0; i<(int)local_path.size()-1; ++i)
  {
    geometry_msgs::Point pt;
    pt.x = local_path[i](0);
    pt.y = local_path[i](1);
    pt.z = local_path[i](2);
    mk.points.push_back(pt);

    pt.x = local_path[i+1](0);
    pt.y = local_path[i+1](1);
    pt.z = local_path[i+1](2);
    mk.points.push_back(pt);
  }

  path_markers.markers.push_back(mk);
  counter++;

  visualization_msgs::Marker sp;
  sp.header.frame_id = "world";
  sp.header.stamp = ros::Time::now();
  sp.id = counter;
  sp.type = visualization_msgs::Marker::SPHERE_LIST;
  sp.color.r = 0.0;
  sp.color.g = 0.0;
  sp.color.b = 1.0;
  sp.color.a = 1.0;
  sp.scale.x = scale_sphere;
  sp.scale.y = scale_sphere;
  sp.scale.z = scale_sphere;
  sp.pose.orientation.w = 1.0;

  for (int i=0; i<(int)local_path.size(); ++i)
  {
    geometry_msgs::Point pt;
    pt.x = local_path[i](0);
    pt.y = local_path[i](1);
    pt.z = local_path[i](2);
    sp.points.push_back(pt);
  }

  path_markers.markers.push_back(sp);
  counter++;

  local_pub_.publish(path_markers);
  path_markers_ = path_markers;

  // * path order
  int counter_order = 0;
  visualization_msgs::MarkerArray orders;
  for (int i=0; i<(int)local_path.size(); ++i)
  {
    visualization_msgs::Marker text_marker;
    text_marker.header.frame_id = "world";
    text_marker.header.stamp = ros::Time::now();
    text_marker.id = counter_order;
    text_marker.ns = "local_path_order";
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.text = std::to_string(i);
    text_marker.pose.position.x = local_path[i](0);
    text_marker.pose.position.y = local_path[i](1);
    text_marker.pose.position.z = local_path[i](2);
    text_marker.scale.z = 4*scale_sphere;
    text_marker.color.r = 0.0;
    text_marker.color.g = 0.0;
    text_marker.color.b = 0.0;
    text_marker.color.a = 1.0;
    orders.markers.push_back(text_marker);
    counter_order++;
  }

  local_path_order_pub_.publish(orders);

  if (!vp_set_.markers.empty()) {
    for (auto& marker : vp_set_.markers) {
      marker.action = visualization_msgs::Marker::DELETE;
    }
    localVP_pub_.publish(vp_set_);
    vp_set_.markers.clear();
  }

  // * waypoints
  visualization_msgs::MarkerArray vp_set;
  int counter_wp = 0;
  for (int j=0; j<(int)list1.size(); ++j)
  {
    visualization_msgs::Marker wp;
    wp.header.frame_id = "world";
    wp.header.stamp = ros::Time::now();
    wp.id = counter_wp;
    wp.ns = "local_waypoints";
    wp.type = visualization_msgs::Marker::LINE_LIST;
    wp.color.r = j == 0? 1.0:0.0;
    wp.color.g = 0.0;
    wp.color.b = 0.0;
    wp.color.a = 1.0;
    wp.scale.x = scale_wp;
    wp.scale.y = scale_wp;
    wp.scale.z = scale_wp;
    wp.pose.orientation.w = 1.0;

    geometry_msgs::Point pt;
    for (int i = 0; i < (int)list1[j].size(); ++i) 
    {
      pt.x = list1[j][i](0);
      pt.y = list1[j][i](1);
      pt.z = list1[j][i](2);
      wp.points.push_back(pt);

      pt.x = list2[j][i](0);
      pt.y = list2[j][i](1);
      pt.z = list2[j][i](2);
      wp.points.push_back(pt);
    }

    vp_set.markers.push_back(wp);
    counter_wp++;
  }

  localVP_pub_.publish(vp_set);
  vp_set_ = vp_set;
}

void PlanningVisualization::publishUpdatedCurPath(const vector<Eigen::VectorXd>& local_path, const vector<vector<Eigen::Vector3d>>& list1, const vector<vector<Eigen::Vector3d>>& list2)
{
  double scale_line = 0.1;
  double scale_sphere = 0.25;
  double scale_wp = 0.1;
  
  if (!path_markers_updated_.markers.empty()) {
    for (auto& marker : path_markers_updated_.markers) {
      marker.action = visualization_msgs::Marker::DELETE;
    }
    local_updated_pub_.publish(path_markers_updated_);
    path_markers_updated_.markers.clear();
  }

  // * path
  int counter = 0;
  visualization_msgs::MarkerArray path_markers;
  
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.id = counter;
  mk.type = visualization_msgs::Marker::LINE_LIST;
  mk.color.r = red_list[50];
  mk.color.g = green_list[50];
  mk.color.b = blue_list[50];
  mk.color.a = 1.0;
  mk.scale.x = scale_line;
  mk.scale.y = scale_line;
  mk.scale.z = scale_line;
  mk.pose.orientation.w = 1.0;

  for (int i=0; i<(int)local_path.size()-1; ++i)
  {
    geometry_msgs::Point pt;
    pt.x = local_path[i](0);
    pt.y = local_path[i](1);
    pt.z = local_path[i](2);
    mk.points.push_back(pt);

    pt.x = local_path[i+1](0);
    pt.y = local_path[i+1](1);
    pt.z = local_path[i+1](2);
    mk.points.push_back(pt);
  }

  path_markers.markers.push_back(mk);
  counter++;

  visualization_msgs::Marker sp;
  sp.header.frame_id = "world";
  sp.header.stamp = ros::Time::now();
  sp.id = counter;
  sp.type = visualization_msgs::Marker::SPHERE_LIST;
  sp.color.r = red_list[40];
  sp.color.g = green_list[40];
  sp.color.b = blue_list[40];
  sp.color.a = 1.0;
  sp.scale.x = scale_sphere;
  sp.scale.y = scale_sphere;
  sp.scale.z = scale_sphere;
  sp.pose.orientation.w = 1.0;

  for (int i=0; i<(int)local_path.size(); ++i)
  {
    geometry_msgs::Point pt;
    pt.x = local_path[i](0);
    pt.y = local_path[i](1);
    pt.z = local_path[i](2);
    sp.points.push_back(pt);
  }

  path_markers.markers.push_back(sp);
  counter++;

  // * path order
  int counter_order = 0;
  visualization_msgs::MarkerArray orders;
  for (int i=0; i<(int)local_path.size(); ++i)
  {
    visualization_msgs::Marker text_marker;
    text_marker.header.frame_id = "world";
    text_marker.header.stamp = ros::Time::now();
    text_marker.id = counter_order;
    text_marker.ns = "local_updated_path_order";
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.text = std::to_string(i);
    text_marker.pose.position.x = local_path[i](0);
    text_marker.pose.position.y = local_path[i](1);
    text_marker.pose.position.z = local_path[i](2);
    text_marker.scale.z = 4*scale_sphere;
    text_marker.color.r = 1.0;
    text_marker.color.g = 0.0;
    text_marker.color.b = 0.0;
    text_marker.color.a = 1.0;
    orders.markers.push_back(text_marker);
    counter_order++;
  }

  local_updated_path_order_pub_.publish(orders);

  local_updated_pub_.publish(path_markers);
  path_markers_updated_ = path_markers;

  if (!vp_set_updated_.markers.empty()) {
    for (auto& marker : vp_set_updated_.markers) {
      marker.action = visualization_msgs::Marker::DELETE;
    }
    localVP_updated_pub_.publish(vp_set_updated_);
    vp_set_updated_.markers.clear();
  }

  // * waypoints
  visualization_msgs::MarkerArray vp_set;
  int counter_wp = 0;
  for (int j=0; j<(int)list1.size(); ++j)
  {
    visualization_msgs::Marker wp;
    wp.header.frame_id = "world";
    wp.header.stamp = ros::Time::now();
    wp.id = counter_wp;
    wp.ns = "local_updated_waypoints";
    wp.type = visualization_msgs::Marker::LINE_LIST;
    wp.color.r = j == 0? 1.0:red_list[18];
    wp.color.g = j == 0? 0.0:green_list[18];
    wp.color.b = j == 0? 0.0:blue_list[18];
    wp.color.a = 1.0;
    wp.scale.x = scale_wp;
    wp.scale.y = scale_wp;
    wp.scale.z = scale_wp;
    wp.pose.orientation.w = 1.0;

    geometry_msgs::Point pt;
    for (int i = 0; i < (int)list1[j].size(); ++i) 
    {
      pt.x = list1[j][i](0);
      pt.y = list1[j][i](1);
      pt.z = list1[j][i](2);
      wp.points.push_back(pt);

      pt.x = list2[j][i](0);
      pt.y = list2[j][i](1);
      pt.z = list2[j][i](2);
      wp.points.push_back(pt);
    }

    vp_set.markers.push_back(wp);
    counter_wp++;
  }

  localVP_updated_pub_.publish(vp_set);
  vp_set_updated_ = vp_set;
}

void PlanningVisualization::publishLocalUsedGlobal(const vector<Eigen::VectorXd>& path, const vector<Eigen::VectorXd>& prior_path)
{
  double path_line_scale = 0.2, path_sphere_scale = 0.3;
  double prior_line_scale = 0.2, prior_sphere_scale = 0.4;
  
  visualization_msgs::MarkerArray path_marker;
  path_marker.markers.clear();

  local_used_g_path_pub_.publish(path_marker);

  if ((int)path.size() > 1)
  {
    visualization_msgs::Marker line;
    line.header.frame_id = "world";
    line.header.stamp = ros::Time::now();
    line.id = 0;
    line.ns = "local_used_global_line";
    line.type = visualization_msgs::Marker::LINE_LIST;
    line.color.r = 0.0;
    line.color.g = 1.0;
    line.color.b = 0.0;
    line.color.a = 0.5;
    line.scale.x = path_line_scale;
    line.scale.y = path_line_scale;
    line.scale.z = path_line_scale;
    line.pose.orientation.w = 1.0;
    for (int i=0; i<(int)path.size()-1; ++i)
    {
      geometry_msgs::Point p1, p2;
      p1.x = path[i](0);
      p1.y = path[i](1);
      p1.z = path[i](2);
      p2.x = path[i+1](0);
      p2.y = path[i+1](1);
      p2.z = path[i+1](2);
      line.points.push_back(p1);
      line.points.push_back(p2);
    }
    path_marker.markers.push_back(line);
  }

  visualization_msgs::Marker sphere;
  sphere.header.frame_id = "world";
  sphere.header.stamp = ros::Time::now();
  sphere.id = 1;
  sphere.ns = "local_used_global_sphere";
  sphere.type = visualization_msgs::Marker::SPHERE_LIST;
  sphere.scale.x = path_sphere_scale;
  sphere.scale.y = path_sphere_scale;
  sphere.scale.z = path_sphere_scale;
  sphere.color.r = 0.0;
  sphere.color.g = 0.0;
  sphere.color.b = 0.0;
  sphere.color.a = 1.0;
  sphere.pose.orientation.w = 1.0;

  visualization_msgs::Marker text_marker;
  text_marker.header.frame_id = "world";
  text_marker.header.stamp = ros::Time::now();
  text_marker.ns = "local_used_global_sphere_idx";
  text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  text_marker.color.r = 1.0;
  text_marker.color.g = 0.0;
  text_marker.color.b = 0.0;
  text_marker.color.a = 1.0;
  text_marker.scale.z = 0.6;

  for (int i=1; i<(int)path.size(); ++i)
  {
    geometry_msgs::Point pt;
    pt.x = path[i](0);
    pt.y = path[i](1);
    pt.z = path[i](2);
    sphere.points.push_back(pt);

    text_marker.id = i;
    text_marker.pose.position = pt;
    text_marker.text = std::to_string(i);
    // path_marker.markers.push_back(text_marker);
  }
  path_marker.markers.push_back(sphere);

  visualization_msgs::Marker start_cube;
  start_cube.header.frame_id = "world";
  start_cube.header.stamp = ros::Time::now();
  start_cube.id = 2;
  start_cube.ns = "local_used_global_start";
  start_cube.type = visualization_msgs::Marker::CUBE;
  start_cube.color.r = 0.0;
  start_cube.color.g = 0.5;
  start_cube.color.b = 0.5;
  start_cube.color.a = 1.0;
  start_cube.scale.x = path_sphere_scale;
  start_cube.scale.y = path_sphere_scale;
  start_cube.scale.z = path_sphere_scale;
  start_cube.pose.orientation.w = 1.0;
  start_cube.pose.position.x = path[0](0);
  start_cube.pose.position.y = path[0](1);
  start_cube.pose.position.z = path[0](2);
  path_marker.markers.push_back(start_cube);

  local_used_g_path_pub_.publish(path_marker);

  visualization_msgs::MarkerArray prior_path_marker;
  prior_path_marker.markers.clear();

  local_used_g_prior_pub_.publish(prior_path_marker);

  if ((int)prior_path.size() > 1)
  {
    visualization_msgs::Marker prior_line;
    prior_line.header.frame_id = "world";
    prior_line.header.stamp = ros::Time::now();
    prior_line.id = 0;
    prior_line.ns = "local_used_global_prior_line";
    prior_line.type = visualization_msgs::Marker::LINE_LIST;
    prior_line.color.r = 1.0;
    prior_line.color.g = 0.0;
    prior_line.color.b = 0.0;
    prior_line.color.a = 0.7;
    prior_line.scale.x = prior_line_scale;
    prior_line.scale.y = prior_line_scale;
    prior_line.scale.z = prior_line_scale;
    prior_line.pose.orientation.w = 1.0;
    for (int i=0; i<(int)prior_path.size()-1; ++i)
    {
      geometry_msgs::Point p1, p2;
      p1.x = prior_path[i](0);
      p1.y = prior_path[i](1);
      p1.z = prior_path[i](2);
      p2.x = prior_path[i+1](0);
      p2.y = prior_path[i+1](1);
      p2.z = prior_path[i+1](2);
      prior_line.points.push_back(p1);
      prior_line.points.push_back(p2);
    }
    prior_path_marker.markers.push_back(prior_line);
  }

  visualization_msgs::Marker prior_sphere;
  prior_sphere.header.frame_id = "world";
  prior_sphere.header.stamp = ros::Time::now();
  prior_sphere.id = 1;
  prior_sphere.ns = "local_used_global_prior_sphere";
  prior_sphere.type = visualization_msgs::Marker::SPHERE_LIST;
  prior_sphere.color.r = 0.0;
  prior_sphere.color.g = 0.0;
  prior_sphere.color.b = 1.0;
  prior_sphere.color.a = 1.0;
  prior_sphere.scale.x = prior_sphere_scale;
  prior_sphere.scale.y = prior_sphere_scale;
  prior_sphere.scale.z = prior_sphere_scale;
  prior_sphere.pose.orientation.w = 1.0;
  for (int i=0; i<(int)prior_path.size(); ++i)
  {
    geometry_msgs::Point pt;
    pt.x = prior_path[i](0);
    pt.y = prior_path[i](1);
    pt.z = prior_path[i](2);
    prior_sphere.points.push_back(pt);
  }
  prior_path_marker.markers.push_back(prior_sphere);

  local_used_g_prior_pub_.publish(prior_path_marker);

  return;
}

void PlanningVisualization::publishPredMesh(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces)
{
  visualization_msgs::Marker mesh_marker;
  mesh_marker.header.frame_id = "world";
  mesh_marker.header.stamp = ros::Time::now();
  mesh_marker.ns = "mesh";
  mesh_marker.id = 0;
  mesh_marker.type = visualization_msgs::Marker::TRIANGLE_LIST;
  mesh_marker.pose.orientation.w = 1.0;
  mesh_marker.scale.x = 1.0;
  mesh_marker.scale.y = 1.0;
  mesh_marker.scale.z = 1.0;
  mesh_marker.color.a = 0.5;
  mesh_marker.color.r = 0.4;
  mesh_marker.color.g = 0.3;
  mesh_marker.color.b = 0.5;

  mesh_marker.action = visualization_msgs::Marker::DELETEALL;
  pred_mesh_pub_.publish(mesh_marker);

  for (int i = 0; i < faces.rows(); ++i)
  {
      for (int j = 0; j < faces.cols(); ++j)
      {
          int idx = faces(i, j);
          geometry_msgs::Point p;
          p.x = vertices(idx, 0);
          p.y = vertices(idx, 1);
          p.z = vertices(idx, 2);
          mesh_marker.points.push_back(p);
      }
  }

  mesh_marker.action = visualization_msgs::Marker::ADD;
  pred_mesh_pub_.publish(mesh_marker);
}

void PlanningVisualization::publishGlobalStart(const Eigen::VectorXd& start_)
{
  Eigen::Vector3d start_pos = start_.head(3);
  double pitch = start_(3), yaw = start_(4);

  Eigen::Vector3d vec;
  vec(0) = cos(pitch) * cos(yaw);
  vec(1) = cos(pitch) * sin(yaw);
  vec(2) = sin(pitch);

  visualization_msgs::MarkerArray marker_array;

  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.id = 0;
  mk.type = visualization_msgs::Marker::ARROW;
  mk.action = visualization_msgs::Marker::ADD;
  mk.color.r = 1.0;
  mk.color.g = 0.0;
  mk.color.b = 0.0;
  mk.color.a = 1.0;
  mk.scale.x = 0.3;
  mk.scale.y = 0.5;
  mk.scale.z = 0.3;
  mk.pose.orientation.w = 1.0;
  geometry_msgs::Point pt;
  pt.x = start_pos(0);
  pt.y = start_pos(1);
  pt.z = start_pos(2);
  mk.points.push_back(pt);

  pt.x = start_pos(0) + 2.0*vec(0);
  pt.y = start_pos(1) + 2.0*vec(1);
  pt.z = start_pos(2) + 2.0*vec(2);
  mk.points.push_back(pt);

  marker_array.markers.push_back(mk);

  visualization_msgs::Marker pos;
  pos.header.frame_id = "world";
  pos.header.stamp = ros::Time::now();
  pos.id = 1;
  pos.type = visualization_msgs::Marker::SPHERE;
  pos.action = visualization_msgs::Marker::ADD;
  pos.color.r = 0.0;
  pos.color.g = 0.0;
  pos.color.b = 0.0;
  pos.color.a = 1.0;
  pos.scale.x = 0.5;
  pos.scale.y = 0.5;
  pos.scale.z = 0.5;
  pos.pose.orientation.w = 1.0;
  pos.pose.position.x = start_pos(0);
  pos.pose.position.y = start_pos(1);
  pos.pose.position.z = start_pos(2);

  marker_array.markers.push_back(pos);

  global_start_pub_.publish(marker_array);
}

void PlanningVisualization::publishExecPart(const vector<Eigen::VectorXd>& path)
{
  double sphere_scale = 0.25;
  visualization_msgs::MarkerArray part;

  visualization_msgs::Marker wp;
  wp.header.frame_id = "world";
  wp.header.stamp = ros::Time::now();
  wp.id = 0;
  wp.ns = "exec_path_wp";
  wp.type = visualization_msgs::Marker::SPHERE_LIST;
  wp.color.r = 1.0;
  wp.color.g = 0.0;
  wp.color.b = 0.0;
  wp.color.a = 1.0;
  wp.scale.x = sphere_scale;
  wp.scale.y = sphere_scale;
  wp.scale.z = sphere_scale;
  wp.pose.orientation.w = 1.0;

  for (int i=0; i<(int)path.size(); ++i)
  {
    geometry_msgs::Point pt;
    pt.x = path[i](0);
    pt.y = path[i](1);
    pt.z = path[i](2);
    wp.points.push_back(pt);
  }
  part.markers.push_back(wp);

  // if ((int)path.size() > 1)
  // {
  //   double line_scale = 0.1;
  //   visualization_msgs::Marker line;
  //   line.header.frame_id = "world";
  //   line.header.stamp = ros::Time::now();
  //   line.id = 1;
  //   line.ns = "exec_path_line";
  //   line.type = visualization_msgs::Marker::LINE_LIST;
  //   line.color.r = 0.1;
  //   line.color.g = 0.1;
  //   line.color.b = 0.1;
  //   line.color.a = 0.8;
  //   line.scale.x = line_scale;
  //   line.scale.y = line_scale;
  //   line.scale.z = line_scale;
  //   line.pose.orientation.w = 1.0;

  //   for (int i=0; i<(int)path.size()-1; ++i)
  //   {
  //     geometry_msgs::Point pt;
  //     pt.x = path[i](0);
  //     pt.y = path[i](1);
  //     pt.z = path[i](2);
  //     line.points.push_back(pt);

  //     pt.x = path[i+1](0);
  //     pt.y = path[i+1](1);
  //     pt.z = path[i+1](2);
  //     line.points.push_back(pt);
  //   }
  //   part.markers.push_back(line);
  // }

  exec_path_pub_.publish(part);
}

}
