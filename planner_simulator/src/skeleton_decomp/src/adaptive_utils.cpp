/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Apr. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the main algorithm of utils for better skeleton extraction in
 *                   FlyCo.
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

#include "skeleton_decomp/adaptive_utils.h"

namespace flyco
{

void adaptive_utils::init(ros::NodeHandle& nh)
{
  /* Params */
  nh.param("fps/cpu", fps_tool, string("null"));
  nh.param("fps/input", input_file, string("null"));
  nh.param("fps/output", output_file, string("null"));
  nh.param("sk/input_mesh", mesh_file, string("null"));
  nh.param("input/resolution", resolution_, -1.0);
  nh.param("input/independent", independent, true);
}

void adaptive_utils::setScene(RTCScene input_scene)
{
  this->scene = input_scene;
}

void adaptive_utils::create_scene()
{
  /* Mesh & Scene */
  // ! most time-consuming -> directly provide mesh_V and mesh_F
  // ! mesh_V -> (N,3), mesh_F -> (M,3)
  if (independent == true)
  {
    igl::read_triangle_mesh(mesh_file, mesh_V, mesh_F);
    cout << "Mesh Vertices: " << mesh_V.rows() << ", " << mesh_V.cols() << endl;
    cout << "Mesh Faces: " << mesh_F.rows() << ", " << mesh_F.cols() << endl;
    visibility_st::mesh2scene(mesh_V, mesh_F, this->scene);
  }
  
  // Mesh to PCD
  auto m2p_t1 = chrono::high_resolution_clock::now();
  sampled_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
  visibility_st::mesh2pcd(mesh_V, mesh_F, resolution_, sampled_cloud);
  auto m2p_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, std::milli> m2p_ms = m2p_t2 - m2p_t1;
  double m2p_time = (double)m2p_ms.count();
  ROS_INFO("\033[34m[ADP] Mesh to PCD time = %lf ms.\033[32m", m2p_time);
}

void adaptive_utils::mesh2pcd(double& resolution)
{
  auto m2p_t1 = chrono::high_resolution_clock::now();

  sampled_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

  for (int i = 0; i < (int)mesh_F.rows(); ++i)
  {
    Eigen::Vector3d v1 = mesh_V.row(mesh_F(i, 0));
    Eigen::Vector3d v2 = mesh_V.row(mesh_F(i, 1));
    Eigen::Vector3d v3 = mesh_V.row(mesh_F(i, 2));

    double a = (v2 - v1).norm();
    double b = (v3 - v2).norm();
    double c = (v1 - v3).norm();
    double s = (a + b + c) / 2.0;
    double area = sqrt(s * (s - a) * (s - b) * (s - c));

    int points_per_face = max(1, (int) (area / (resolution * resolution)));

    for (int j = 0; j < points_per_face; ++j)
    {
      double r1 = ((double) rand() / (RAND_MAX));
      double r2 = ((double) rand() / (RAND_MAX));
      double sqrt_r1 = sqrt(r1);
      double one_minus_sqrt_r1 = (1 - sqrt_r1);
      double sqrt_r1_times_r2 = sqrt_r1 * r2;

      Eigen::Vector3d point = (one_minus_sqrt_r1 * v1) + (sqrt_r1_times_r2 * v2) + ((1 - r2) * sqrt_r1 * v3);
      sampled_cloud->points.push_back(pcl::PointXYZ(point(0), point(1), point(2)));
    }
  }

  sampled_cloud->width = sampled_cloud->points.size();
  sampled_cloud->height = 1;
  sampled_cloud->is_dense = true;

  auto m2p_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, std::milli> m2p_ms = m2p_t2 - m2p_t1;
  double m2p_time = (double)m2p_ms.count();
  ROS_INFO("\033[34m[ADP] Mesh to PCD time = %lf ms.\033[32m", m2p_time);
}

vector<int> adaptive_utils::fps(const pcl::PointCloud<pcl::PointNormal>::Ptr cloud, int num_samples)
{
  vector<int> samples;
  /* Write input */
  ofstream input(input_file);
  for (int i = 0; i < (int)cloud->points.size(); i++)
  {
    Eigen::Vector3d pt(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
    for (int j = 0; j < 3; ++j)
    {
      pt[j] = std::round(pt[j] * 1e6) / 1e6;
    }

    stringstream ss;
    ss << fixed << setprecision(6) << cloud->points[i].x << " " << cloud->points[i].y << " " << cloud->points[i].z;
    point_idx_str_[ss.str()] = i;

    if (i == (int)cloud->points.size() - 1)
      input << ss.str();
    else
      input << ss.str() << endl;
  }
  input.close();

  /* FPS command */
  string command_ = "cd " + fps_tool + " && ./kdline tree_high " + to_string(num_samples) + " " + input_file;
  const char* charPtr = command_.c_str();
  system_back_ = system(charPtr);

  /* Read output */
  ifstream output(output_file);
  string line;
  while (getline(output, line))
  {
    istringstream iss(line);
    double x, y, z;
    if (!(iss >> fixed >> setprecision(6) >> x >> y >> z)) {
      if (stoi(line) == -1) break;
    }
    stringstream ss;
    ss << fixed << setprecision(6) << x << " " << y << " " << z;
    string vec_str = ss.str();

    if (point_idx_str_.find(vec_str) == point_idx_str_.end())
    {
      cout << vec_str << endl;
      ROS_ERROR("Point not found in the original cloud.");
      continue;
    }
    else
      samples.push_back(point_idx_str_[vec_str]);
  }
  output.close();

  return samples;
}

Eigen::MatrixXd adaptive_utils::cluster_proc(int& queryIdx, Eigen::MatrixXd& neighbors, Eigen::MatrixXd& pts_mat, double& eps)
{
  vector<int> n_idx;

  Eigen::Vector3d ptP = pts_mat.row(queryIdx);
  vector<Eigen::Vector3d> n_pts;
  n_idx.push_back(queryIdx);
  n_pts.push_back(ptP);

  map<Eigen::Vector3d, int, Vector3dComp> NEIGH_pt_id;
  for (int i=0; i<(int)neighbors.rows(); ++i)
  {
    n_idx.push_back(neighbors(i,0));
    n_pts.push_back(pts_mat.row(neighbors(i,0)));
    NEIGH_pt_id[pts_mat.row(neighbors(i,0))] = neighbors(i,0);
  }

  /* Clustering */
  vector<int> pointsInPartA;
  vector<Eigen::Vector3d> cluster = adaptiveDensityCluster(ptP, n_pts, eps);
  for (int i=0; i<(int)cluster.size(); ++i)
  {
    if (NEIGH_pt_id.find(cluster[i]) != NEIGH_pt_id.end())
    {
      int inid = NEIGH_pt_id.find(cluster[i])->second;
      pointsInPartA.push_back(inid); 
    }
  }

  Eigen::MatrixXd only_idxs;
  if (pointsInPartA.size() > 1)
  {
    only_idxs.resize((int)pointsInPartA.size() - 1, 1);
    for (int i = 0; i < (int)pointsInPartA.size() - 1; ++i)
      only_idxs(i, 0) = pointsInPartA[i + 1];
  }
  else
  {
    only_idxs.resize(0, 1);
  }

  return only_idxs;
}

vector<int> adaptive_utils::findNeighbors(Eigen::Vector3d& pointP, vector<Eigen::Vector3d>& neighborhood, double& epsilon)
{
  vector<int> neighbors;
  int neighborhood_size = neighborhood.size();
  
  for (int i=0; i<neighborhood_size; ++i)
  {
    Eigen::Vector3d pointN = neighborhood[i];
    double distance = (pointN-pointP).norm();
    if (epsilon >= distance)
      neighbors.push_back(i);
  }

  return neighbors;
}

void adaptive_utils::assignVisited(vector<int>& neighbors, vector<bool>& visited)
{
  int neighbors_size = neighbors.size();
  
  for (int i=0; i<neighbors_size; ++i)
    visited[neighbors[i]] = true;
}

int adaptive_utils::calNewVisited(vector<int>& neighbors, vector<bool>& visited)
{
  int newVisited = 0;
  for (int i=0; i<(int)neighbors.size(); ++i)
  {
    if (!visited[neighbors[i]])
    {
      newVisited++;
    }
  }

  return newVisited;
}

vector<Eigen::Vector3d> adaptive_utils::adaptiveDensityCluster(Eigen::Vector3d& pointP, vector<Eigen::Vector3d>& neighborhood, double& epsilon)
{
  // find the cluster that pointP belongs to.
  vector<int> neighbors = findNeighbors(pointP, neighborhood, epsilon);
  vector<bool> visited(neighborhood.size(), false);
  vector<int> totalNeighbors;
  vector<int> currentAllNeighbors;
  totalNeighbors.insert(totalNeighbors.end(), neighbors.begin(), neighbors.end());
  vector<Eigen::Vector3d> cluster;

  int maxVisited = 0;
  maxVisited = calNewVisited(neighbors, visited);
  assignVisited(neighbors, visited);
  currentAllNeighbors = neighbors;

  while (maxVisited > 0)
  {
    maxVisited = 0;
    vector<int> tempNeighbors;
    for (int i=0; i<(int)currentAllNeighbors.size(); ++i)
    {
      vector<int> neighbors2 = findNeighbors(neighborhood[currentAllNeighbors[i]], neighborhood, epsilon);
      maxVisited = max(maxVisited, calNewVisited(neighbors2, visited));
      for (int j=0; j<(int)neighbors2.size(); ++j)
      {
        if (visited[neighbors2[j]] == false)
        {
          tempNeighbors.push_back(neighbors2[j]);
          totalNeighbors.push_back(neighbors2[j]);
        }
      }
      assignVisited(neighbors2, visited); 
    }
    
    currentAllNeighbors = tempNeighbors;
  }

  for (int i=0; i<(int)totalNeighbors.size(); ++i)
    cluster.push_back(neighborhood[totalNeighbors[i]]);

  return cluster;
}

vector<int> adaptive_utils::sdf_check(Eigen::MatrixXd& pts_mat)
{
  auto sc_t1 = std::chrono::high_resolution_clock::now();
  // true -> inlier, false -> outlier
  vector<int> inside;

  vector<bool> inside_flag(pts_mat.rows());

  Eigen::MatrixXd S;
  Eigen::VectorXi I;
  Eigen::MatrixXd C;
  Eigen::MatrixXd N;

  igl::signed_distance(pts_mat, mesh_V, mesh_F, igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL, S, I, C, N);

  for (int i = 0; i < (int)pts_mat.rows(); ++i) 
  {
    inside_flag[i] = S(i) < -0.1;
    if (inside_flag[i] == true)
      inside.push_back(i);
  }

  auto sc_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> sc_ms = sc_t2 - sc_t1;
  double sc_time = (double)sc_ms.count();
  ROS_INFO("\033[34m[SSD] Mesh Check time = %lf ms.\033[32m", sc_time);

  return inside;
}

vector<int> adaptive_utils::raytracing_check(Eigen::MatrixXd& pts_mat, Eigen::MatrixXf& pts_dir, vector<vector<Eigen::Vector3d>>& intersection_pts)
{
  vector<int> inside;

  // Raytracing -> four points
  for (int j = 0; j < pts_mat.rows(); ++j)
  {
    Eigen::Vector3d point = pts_mat.row(j);
    Eigen::Vector3f direction = pts_dir.row(j);
    Eigen::Vector3f direction_inv = -pts_dir.row(j);
    Eigen::Vector3f ortho_(direction(1), -direction(0), 0);
    Eigen::Vector3f ortho_inv(-direction(1), direction(0), 0);

    inter_pts.clear();
    bool inlier = ray_tracing(point, direction);
    intersection_pts[j] = inter_pts;
    bool inlier_inv = ray_tracing(point, direction_inv);
    bool inlier_ortho = ray_tracing(point, ortho_);
    bool inlier_ortho_inv = ray_tracing(point, ortho_inv);

    if (inlier == true && inlier_inv == true && inlier_ortho == true && inlier_ortho_inv == true)
    {
      inside.push_back(j);
    }
  }

  return inside;
}

bool adaptive_utils::cross_ray_tracing(Eigen::Vector3d& point, Eigen::Vector3f& dir)
{
  bool outside = false;

  Eigen::Vector3f direction_inv = dir;
  Eigen::Vector3f ortho_(dir(1), -dir(0), 0);
  Eigen::Vector3f ortho_inv(-dir(1), dir(0), 0);

  bool inlier = ray_tracing(point, dir);
  bool inlier_inv = ray_tracing(point, direction_inv);
  bool inlier_ortho = ray_tracing(point, ortho_);
  bool inlier_ortho_inv = ray_tracing(point, ortho_inv);

  if (inlier == false && inlier_inv == false && inlier_ortho == false && inlier_ortho_inv == false)
  {
    outside = true;
  }

  return outside;
}

// no collision -> true, collision -> false
bool adaptive_utils::ray_tracing_public(Eigen::Vector3d& pointA, Eigen::Vector3d& pointB)
{
  return ray_tracing_public(pointA, pointB, scene);
}

bool adaptive_utils::ray_tracing_public(Eigen::Vector3d& pointA, Eigen::Vector3d& pointB, RTCScene query_scene)
{
  Eigen::Vector3f direction = (pointB - pointA).cast<float>();
  double length = direction.norm();
  if (length < 1e-6 || query_scene == nullptr)
    return true;
  direction.normalize();

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
  rtcIntersect1(query_scene, &context, &rayhit);

  if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) 
    return true;
  else
    return false;
}

void adaptive_utils::ray_tracing_public_batch(const vector<Eigen::Vector3d>& points,
                                              RTCScene query_scene,
                                              Eigen::MatrixXi& graph,
                                              double& min_edge_dist)
{
  min_edge_dist = INFINITY;

  const int point_num = (int)points.size();
  graph.resize(point_num, point_num);
  graph.setZero();

  if (point_num <= 0)
    return;

  if (query_scene == nullptr)
  {
    graph.setOnes();
    graph.diagonal().setOnes();
    return;
  }

  RTCIntersectContext context;
  for (int i=0; i<point_num; ++i)
  {
    graph(i,i) = 1;
    const Eigen::Vector3d& pointA = points[i];
    if (!pointA.allFinite())
      continue;

    for (int j=0; j<i; ++j)
    {
      const Eigen::Vector3d& pointB = points[j];
      if (!pointB.allFinite())
        continue;

      Eigen::Vector3f direction = (pointB - pointA).cast<float>();
      const double length = direction.norm();
      if (length < 1e-6)
      {
        graph(i,j) = 1;
        graph(j,i) = 1;
        continue;
      }
      direction.normalize();

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

      rtcInitIntersectContext(&context);
      rtcIntersect1(query_scene, &context, &rayhit);

      if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
      {
        graph(i,j) = 1;
        graph(j,i) = 1;
        if (length < min_edge_dist)
          min_edge_dist = length;
      }
    }
  }
}

bool adaptive_utils::max_progress(Eigen::Vector3d& p, Eigen::Vector3d& n, double& prog)
{
  bool compress = false;
  
  Eigen::Vector3f dir;
  dir << n(0), n(1), n(2);
  dir.normalize();

  RTCRayHit rayhit_a2b;
  RTCRay ray_a2b;
  ray_a2b.org_x = p(0);
  ray_a2b.org_y = p(1);
  ray_a2b.org_z = p(2);
  ray_a2b.dir_x = dir(0);
  ray_a2b.dir_y = dir(1);
  ray_a2b.dir_z = dir(2);
  ray_a2b.tnear = 0;
  ray_a2b.tfar = INFINITY;
  ray_a2b.time = 0.0f;
  ray_a2b.mask = -1;
  ray_a2b.id = 0;
  ray_a2b.flags = 0;
  rayhit_a2b.ray = ray_a2b;
  RTCHit hit_a2b;
  hit_a2b.geomID = RTC_INVALID_GEOMETRY_ID;
  rayhit_a2b.hit = hit_a2b;

  RTCIntersectContext context_a2b;
  rtcInitIntersectContext(&context_a2b);
  rtcIntersect1(scene, &context_a2b, &rayhit_a2b);

  if (rayhit_a2b.hit.geomID != RTC_INVALID_GEOMETRY_ID)
  {
    prog = 0.5*rayhit_a2b.ray.tfar;
    if (prog > 0.5*resolution_)
      compress = true;
  }
  
  return compress;
}

bool adaptive_utils::ray_tracing(Eigen::Vector3d& point, Eigen::Vector3f& direction)
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

  vector<RTCHit> intersections = get_all_intersections(scene, ray);
  num_intersections += (int)intersections.size();
  // cout << "Num intersections: " << num_intersections << endl;

  if (num_intersections % 2 != 0)
  {
    return true;
  }
  else
  {
    return false;
  }
}

vector<RTCHit> adaptive_utils::get_all_intersections(RTCScene scene, RTCRay ray)
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
    if (rayhit.ray.tfar < 0.5*resolution_ && check_times == 0)
      break;

    // cout << "tfar: " << rayhit.ray.tfar << endl;
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

    Eigen::Vector3d isec_pt(ray.org_x, ray.org_y, ray.org_z);
    // cout << "intersection: " << isec_pt.transpose() << endl;
    inter_pts.push_back(isec_pt);
    check_times++;
  }

  return intersections;
}

void adaptive_utils::set_free_raytracing()
{
  rtcReleaseScene(scene);
}

}
