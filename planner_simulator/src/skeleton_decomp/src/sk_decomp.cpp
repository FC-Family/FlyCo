/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    May. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the main algorithm of a new robust and adaptive skeleton
 *                   extraction, decomposition, and space allocation in FlyCo.
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

#include "skeleton_decomp/sk_decomp.h"

namespace flyco
{
namespace
{
double clampUnit(double value)
{
  return std::max(-1.0, std::min(1.0, value));
}

double pointSegmentDistanceSq(const Eigen::Vector3d& point,
                              const Eigen::Vector3d& seg_start,
                              const Eigen::Vector3d& seg_end)
{
  if (!point.allFinite() || !seg_start.allFinite() || !seg_end.allFinite())
    return std::numeric_limits<double>::infinity();

  const Eigen::Vector3d seg = seg_end - seg_start;
  const double len_sq = seg.squaredNorm();
  if (len_sq <= 1e-12)
    return (point - seg_start).squaredNorm();

  const double t = std::max(0.0, std::min(1.0, (point - seg_start).dot(seg) / len_sq));
  const Eigen::Vector3d proj = seg_start + t * seg;
  return (point - proj).squaredNorm();
}

double medianPositive(vector<double> values)
{
  values.erase(std::remove_if(values.begin(), values.end(), [](double value) {
    return !std::isfinite(value) || value <= 0.0;
  }), values.end());

  if (values.empty())
    return 0.0;

  const size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  double median = values[mid];
  if (values.size() % 2 == 0)
  {
    std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
    median = 0.5 * (median + values[mid - 1]);
  }

  return median;
}

int countGraphEdges(const Eigen::MatrixXi& graph)
{
  if (graph.rows() != graph.cols())
    return 0;

  int edge_num = 0;
  for (int i=0; i<graph.rows(); ++i)
    for (int j=i+1; j<graph.cols(); ++j)
      if (graph(i,j) == 1)
        edge_num++;
  return edge_num;
}

int countGraphTriangles(const Eigen::MatrixXi& graph)
{
  if (graph.rows() != graph.cols())
    return 0;

  int triangle_num = 0;
  for (int i=0; i<graph.rows(); ++i)
    for (int j=i+1; j<graph.cols(); ++j)
    {
      if (graph(i,j) != 1)
        continue;
      for (int k=j+1; k<graph.cols(); ++k)
      {
        if (graph(i,k) == 1 && graph(j,k) == 1)
          triangle_num++;
      }
    }
  return triangle_num;
}

Eigen::MatrixXd candidateRowsToMatrix(const Eigen::MatrixXd& candidates, const vector<int>& rows)
{
  Eigen::MatrixXd out(rows.size(), 3);
  for (int i=0; i<(int)rows.size(); ++i)
  {
    const int row = rows[i];
    if (row >= 0 && row < candidates.rows() && candidates.cols() >= 3)
      out.row(i) = candidates.row(row).leftCols(3);
  }
  return out;
}

void appendFarthestSubset(const vector<Eigen::Vector3d>& source,
                          int target_total,
                          vector<Eigen::Vector3d>& output)
{
  if (source.empty() || target_total <= (int)output.size())
    return;

  const int n = (int)source.size();
  vector<char> selected(n, 0);
  vector<double> min_dist_sq(n, std::numeric_limits<double>::infinity());

  auto add_idx = [&](int idx)
  {
    if (idx < 0 || idx >= n || selected[idx] || !source[idx].allFinite())
      return;
    selected[idx] = 1;
    output.push_back(source[idx]);
    for (int j=0; j<n; ++j)
    {
      if (selected[j] || !source[j].allFinite())
        continue;
      const double d2 = (source[j] - source[idx]).squaredNorm();
      if (d2 < min_dist_sq[j])
        min_dist_sq[j] = d2;
    }
  };

  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  int valid_num = 0;
  for (const Eigen::Vector3d& point:source)
  {
    if (!point.allFinite())
      continue;
    center += point;
    valid_num++;
  }
  if (valid_num <= 0)
    return;
  center /= valid_num;

  int first_idx = -1;
  double best_center_dist = -1.0;
  for (int i=0; i<n; ++i)
  {
    if (!source[i].allFinite())
      continue;
    const double d2 = (source[i] - center).squaredNorm();
    if (d2 > best_center_dist)
    {
      best_center_dist = d2;
      first_idx = i;
    }
  }
  add_idx(first_idx);

  while ((int)output.size() < target_total)
  {
    int best_idx = -1;
    double best_dist = -1.0;
    for (int i=0; i<n; ++i)
    {
      if (selected[i] || !source[i].allFinite())
        continue;
      if (min_dist_sq[i] > best_dist)
      {
        best_dist = min_dist_sq[i];
        best_idx = i;
      }
    }
    if (best_idx < 0)
      break;
    add_idx(best_idx);
  }
}
}

void sk_decomp::init(ros::NodeHandle& nh)
{
  auto a_t1 = chrono::high_resolution_clock::now();
  /* Params */
  nh.param("sk/input_mesh", input_mesh, string("null"));
  nh.param("sk/cal_number", calNum, -1);
  nh.param("sk/est_number", ne_KNN, -1);
  nh.param("sk/ds_number", estNum, -1);
  nh.param("sk/selected_num", selected_num, -1);
  nh.param("sk/k_KNN", k_KNN, -1);
  nh.param("sk/iteration_extraction", iter_ext, -1);
  nh.param("sk/orientation_convergence_thresh", orientation_convergence_thresh_, 1e-3);
  nh.param("sk/iteration_shrinking", iter_shr, -1);
  nh.param("sk/relaxed_candidate_dist", relaxed_candidate_dist_, -1.0);
  nh.param("sk/hybrid_decomp_max_points", hybrid_decomp_max_points_, 50);
  nh.param("sk/hybrid_component_radius_scale", hybrid_component_radius_scale_, 0.0);
  nh.param("sk/hybrid_projection_weight", hybrid_projection_weight_, 1.0);
  nh.param("sk/hybrid_min_component_size", hybrid_min_component_size_, 4);
  nh.param("sk/hybrid_dcrosa_iter", hybrid_dcrosa_iter_, 1);
  nh.param("sk/hybrid_dcrosa_confidence_th", hybrid_dcrosa_confidence_th_, 0.5);
  nh.param("sk/hybrid_dcrosa_neighbor_blend", hybrid_dcrosa_neighbor_blend_, 1.0);
  nh.param("sk/hybrid_smooth_iter", hybrid_smooth_iter_, 0);
  nh.param("sk/dcrosa_topology_sample_radius", dcrosa_topology_sample_radius_, 0.0);
  nh.param("sk/dcrosa_topology_min_support", dcrosa_topology_min_support_, 2);
  nh.param("sk/dcrosa_guided_min_branch_len_scale", dcrosa_guided_min_branch_len_scale_, 1.0);
  nh.param("sk/dcrosa_guided_confidence_ratio", dcrosa_guided_confidence_ratio_, 0.75);
  nh.param("sk/dcrosa_guided_min_group_size", dcrosa_guided_min_group_size_, 3);
  /* Initialization */
  pt_downsample_voxel_size = 0.02;

  adp_utils_.reset(new adaptive_utils);
  adp_utils_->init(nh);
  /* Visualization */
  vis_utils_.reset(new PlanningVisualization(nh));
  visFlag = true;
  vis_timer_ = nh.createTimer(ros::Duration(0.1), &sk_decomp::visualization, this);

  auto a_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> a_ms = a_t2 - a_t1;
  double a_time = (double)a_ms.count();
  ROS_INFO("\033[34m[SSD] Initialization time = %lf ms.\033[34m", a_time);

  ROS_INFO("\033[34m[SSD] Initialized!\033[34m");
}

void sk_decomp::set_mesh(Eigen::MatrixXd& mesh_V, Eigen::MatrixXi& mesh_F)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  vis_data_ready_.store(false);
  adp_utils_->mesh_V = mesh_V;
  adp_utils_->mesh_F = mesh_F;
}

void sk_decomp::main()
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  vis_data_ready_.store(false);
  dcrosa_branch_subspace_ready_ = false;

  set_input();

  auto ssd_t1 = chrono::high_resolution_clock::now();

  process_data();
  inliers_extraction();
  inliers_shrinking();
  build_dcrosa_topology();
  inliers_check();
  build_branch_subspace_graph();
  apply_dcrosa_guided_decomposition();
  if (!dcrosa_branch_subspace_ready_)
    P.subspace_sets = visibility_subspace_decomp(P.visibility_graph, &P.node_subspace_map);
  if (!dcrosa_branch_subspace_ready_)
    merge_fallback_subspaces();
  build_subspace_skeleton_paths();
  high_level_vis_graph();
  distribute_ori_cloud();
  vis_data_ready_.store(true);

  auto ssd_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> ssd_ms = ssd_t2 - ssd_t1;
  double ssd_time = (double)ssd_ms.count();
  ROS_INFO("\033[34m[SSD] SSD latency = %lf ms.\033[34m", ssd_time);
  ROS_INFO("\033[35m[SSD] --- <SSD finished> --- \033[35m");
}

void sk_decomp::setScene(RTCScene input_scene)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  vis_data_ready_.store(false);
  adp_utils_->setScene(input_scene);
}

void sk_decomp::set_input()
{
  auto a_t1 = chrono::high_resolution_clock::now();

  adp_utils_->create_scene();
  load_pcd();

  auto a_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> a_ms = a_t2 - a_t1;
  double a_time = (double)a_ms.count();
  ROS_INFO("\033[34m[SSD] set input time = %lf ms.\033[34m", a_time);
}

// ? tips: small 'calNum' for faster computation
void sk_decomp::load_pcd()
{
  auto init_t1 = chrono::high_resolution_clock::now();

  P.pts_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  P.vmap_pts_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  P.normals_.reset(new pcl::PointCloud<pcl::Normal>);
  // positions -> read
  *P.pts_ = *adp_utils_->sampled_cloud;
  if ((int)P.pts_->points.size() > calNum)
  {
    pcl::RandomSample<pcl::PointXYZ> rs_o;
    rs_o.setInputCloud(P.pts_);
    rs_o.setSample(calNum);
    rs_o.filter(*P.pts_);
  }

  *P.vmap_pts_ = *P.pts_;

  // normals -> pcl estimate
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(P.pts_);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  ne.setSearchMethod(tree);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  ne.setKSearch(ne_KNN);
  ne.compute(*cloud_normals);
  *P.normals_ = *cloud_normals;

  auto init_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, std::milli> init_ms = init_t2 - init_t1;
  double init_time = (double)init_ms.count();
  ROS_INFO("\033[34m[ADP] PCD loading time = %lf ms.\033[32m", init_time);

  return;
}

void sk_decomp::process_data()
{
  pcd_size_ = P.pts_->points.size();
  P.pts_mat.resize(pcd_size_, 3);
  P.nrs_mat.resize(pcd_size_, 3);
  for (int i=0; i<pcd_size_; ++i)
  {
    P.pts_mat(i,0) = P.pts_->points[i].x;
    P.pts_mat(i,1) = P.pts_->points[i].y;
    P.pts_mat(i,2) = P.pts_->points[i].z;
  }

  normalize();

  pset.resize(selected_num, 3);
  vset.resize(selected_num, 3);
  vvar.resize(selected_num, 1);
  P.datas = new double[pcd_size_*3]();
  for (int idx=0; idx<pcd_size_; ++idx)
  {
    P.datas[idx] = P.pts_->points[idx].x;
    P.datas[idx+pcd_size_] = P.pts_->points[idx].y;
    P.datas[idx+2*pcd_size_] = P.pts_->points[idx].z;
  }
}

void sk_decomp::inliers_compression()
{
  for (int i=0; i<selected_num; ++i)
  {
    Eigen::Vector3d p = P.selected_pts_mat.row(i);
    Eigen::Vector3d p_w;
    p_w(0) = p(0)*P.scale + P.center(0);
    p_w(1) = p(1)*P.scale + P.center(1);
    p_w(2) = p(2)*P.scale + P.center(2);
    Eigen::Vector3d n_a = P.selected_nrs_mat.row(i);
    Eigen::Vector3d n_b = -P.selected_nrs_mat.row(i);

    double prog_a = 0.0, prog_b = 0.0;
    bool c_a = adp_utils_->max_progress(p_w, n_a, prog_a);
    bool c_b = adp_utils_->max_progress(p_w, n_b, prog_b);

    if (c_a == true)
    {
      Eigen::Vector3d pc_a = p_w + prog_a*n_a;
      pc_set.push_back(pc_a);
      pc_cor_idx.push_back(i);
    }

    if (c_b == true)
    {
      Eigen::Vector3d pc_b = p_w + prog_b*n_b;
      pc_set.push_back(pc_b);
      pc_cor_idx.push_back(i);
    }
  }
}

void sk_decomp::inliers_extraction()
{
  auto ie_t1 = chrono::high_resolution_clock::now();

  Extra_Del ed_;
  inliers_initialize(P.selected_pts_, P.selected_normals_);
  // surface neighbors
  P.surf_neighs.clear();
  vector<int> temp_surf(k_KNN);
  vector<float> nn_squared_distance(k_KNN);
  pcl::KdTreeFLANN<pcl::PointXYZ> surf_kdtree;
  surf_kdtree.setInputCloud(P.selected_pts_);
  pcl::PointXYZ search_point_surf;
  for (int i=0; i<selected_num; ++i)
  {
    temp_surf.clear();
    nn_squared_distance.clear();
    search_point_surf = P.selected_pts_->points[i];
    surf_kdtree.nearestKSearch(search_point_surf, k_KNN, temp_surf, nn_squared_distance);
    P.surf_neighs.push_back(temp_surf);
  }
  // inlier estimation
  Eigen::Vector3d var_p, var_v, new_v;
  Eigen::MatrixXd indxs, extract_normals;
  /* --- optimize orientations --- */
  for (int n=0; n<iter_ext; ++n)
  {
    const Eigen::MatrixXd vprev = vset;
    Eigen::MatrixXd vnew = Eigen::MatrixXd::Zero(selected_num, 3);
    for (int pidx=0; pidx<selected_num; ++pidx)
    {
      var_p = pset.row(pidx);
      var_v = vset.row(pidx);
      int oriIdx = P.selected_idx_map_[pidx];
      indxs = rosa_compute_active_samples(oriIdx, var_p, var_v);
      extract_normals = ed_.rows_ext_M(indxs, P.nrs_mat);
      vnew.row(pidx) = compute_symmetrynormal(extract_normals).transpose();
      new_v = vnew.row(pidx);

      if ((int)extract_normals.rows() > 0)
        vvar(pidx,0) = symmnormal_variance(new_v, extract_normals);
      else
        vvar(pidx,0) = 0.0;
    }
    Eigen::MatrixXd offset(vvar.rows(), vvar.cols());
    offset.setOnes();
    offset = 0.00001*offset;
    vvar = (vvar.cwiseAbs2().cwiseAbs2()+offset).cwiseInverse();
    vset = vnew;
    // orientation smoothing
    vector<int> surf_;
    Eigen::MatrixXi snidxs;
    Eigen::MatrixXd snidxs_d, vset_ex, vvar_ex;
    for (int i=0; i<1; ++i)
    {
      for (int p=0; p<selected_num; ++p)
      {
        surf_.clear();
        surf_ = P.surf_neighs[p];
        snidxs.resize(surf_.size(), 1);
        snidxs = Eigen::Map<Eigen::MatrixXi>(surf_.data(), surf_.size(), 1);
        snidxs_d = snidxs.cast<double>();
        vset_ex = ed_.rows_ext_M(snidxs_d, vset);
        vvar_ex = ed_.rows_ext_M(snidxs_d, vvar);
        vset.row(p) = symmnormal_smooth(vset_ex, vvar_ex);
      }
      vnew = vset;
    }

    double max_orientation_change = 0.0;
    int valid_orientation_changes = 0;
    if (orientation_convergence_thresh_ > 0.0)
    {
      for (int pidx=0; pidx<selected_num; ++pidx)
      {
        Eigen::Vector3d prev_v = vprev.row(pidx).transpose();
        Eigen::Vector3d next_v = vset.row(pidx).transpose();
        const double prev_norm = prev_v.norm();
        const double next_norm = next_v.norm();
        if (prev_norm <= 1e-9 || next_norm <= 1e-9 || !prev_v.allFinite() || !next_v.allFinite())
          continue;

        prev_v /= prev_norm;
        next_v /= next_norm;
        const double cos_angle = std::max(-1.0, std::min(1.0, std::abs(prev_v.dot(next_v))));
        max_orientation_change = std::max(max_orientation_change, 1.0 - cos_angle);
        valid_orientation_changes++;
      }
    }

    if (orientation_convergence_thresh_ > 0.0 && valid_orientation_changes > 0 && max_orientation_change < orientation_convergence_thresh_)
    {
      ROS_INFO("\033[32m[SSD] orientation refinement converged at iteration %d/%d (max angular change metric: %.6g < %.6g).\033[32m",
               n + 1,
               iter_ext,
               max_orientation_change,
               orientation_convergence_thresh_);
      break;
    }
  }
  /* --- compute positions --- */
  vector<int> poorIdx;
  pcl::PointCloud<pcl::PointXYZ>::Ptr goodPts (new pcl::PointCloud<pcl::PointXYZ>);
  map<Eigen::Vector3d, Eigen::Vector3d, Vector3dCompareFunc> goodPtsPset;
  Eigen::Vector3d var_p_p, var_v_p, centroid;
  Eigen::MatrixXd indxs_p, extract_pts, extract_nrs;
  for (int pIdx=0; pIdx<selected_num; ++pIdx)
  {
    var_p_p = pset.row(pIdx);
    var_v_p = vset.row(pIdx);
    int oriIdx = P.selected_idx_map_[pIdx];
    bool valid_centroid = false;
    if (hybrid_component_radius_scale_ <= 0.0)
    {
      indxs_p = rosa_compute_active_samples(oriIdx, var_p_p, var_v_p);
      if ((int)indxs_p.rows() > 0)
      {
        extract_pts = ed_.rows_ext_M(indxs_p, P.pts_mat);
        extract_nrs = ed_.rows_ext_M(indxs_p, P.nrs_mat);
        centroid = closest_projection_point(extract_pts, extract_nrs);
        valid_centroid = centroid.allFinite() && centroid.cwiseAbs().maxCoeff() < 1.0;
        if (!valid_centroid)
        {
          // ROSA-main-like robust fallback: keep the same active sample set but
          // avoid dropping this skeleton node when the projection system is singular.
          centroid = extract_pts.colwise().mean();
          valid_centroid = centroid.allFinite();
        }
        else if (hybrid_projection_weight_ < 1.0)
        {
          const Eigen::Vector3d mean = extract_pts.colwise().mean();
          const double projection_weight = std::max(0.0, std::min(1.0, hybrid_projection_weight_));
          centroid = projection_weight * centroid + (1.0 - projection_weight) * mean;
        }
      }
    }
    else
    {
      vector<vector<int>> sample_sets = hybrid_active_sample_sets(oriIdx, var_p_p, var_v_p);
      valid_centroid = robust_drosa_point(sample_sets, centroid);
    }

    if (valid_centroid && abs(centroid(0)) < 1 && abs(centroid(1)) < 1 && abs(centroid(2)) < 1)
    {
      pset.row(pIdx) = centroid;
      pcl::PointXYZ goodPoint;
      Eigen::Vector3d goodPointP;
      goodPoint = P.selected_pts_->points[pIdx];
      goodPointP(0) = P.selected_pts_->points[pIdx].x;
      goodPointP(1) = P.selected_pts_->points[pIdx].y;
      goodPointP(2) = P.selected_pts_->points[pIdx].z;
      goodPts->points.push_back(goodPoint);
      goodPtsPset[goodPointP] = centroid;
    }
    else
      poorIdx.push_back(pIdx);
  }

  pcl::KdTreeFLANN<pcl::PointXYZ> sk_tree;
  if (goodPts->points.empty())
  {
    ROS_WARN("\033[33m[SSD] hybrid dROSA produced no valid points; keep initialized selected samples as fallback.\033[33m");
    return;
  }
  sk_tree.setInputCloud(goodPts);
  int pair = 1;
  vector<int> pair_id(pair);
  vector<float> s_distance(pair);
  for (int pp=0; pp<(int)poorIdx.size(); ++pp)
  {
    pcl::PointXYZ search_point;
    search_point.x = P.selected_pts_->points[poorIdx[pp]].x;
    search_point.y = P.selected_pts_->points[poorIdx[pp]].y;
    search_point.z = P.selected_pts_->points[poorIdx[pp]].z;
    pair_id.clear();
    s_distance.clear();
    sk_tree.nearestKSearch(search_point, pair, pair_id, s_distance);
    Eigen::Vector3d pairpos;
    pairpos(0) = goodPts->points[pair_id[0]].x;
    pairpos(1) = goodPts->points[pair_id[0]].y;
    pairpos(2) = goodPts->points[pair_id[0]].z;
    Eigen::Vector3d goodrp = goodPtsPset.find(pairpos)->second;
    pset.row(poorIdx[pp]) = goodrp;
  }

  hybrid_dcrosa();
  smooth_hybrid_drosa();

  auto ie_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> ie_ms = ie_t2 - ie_t1;
  double ie_time = (double)ie_ms.count();
  ROS_INFO("\033[32m[SSD] inliers extraction time = %lf ms.\033[32m", ie_time);
}

void sk_decomp::inliers_shrinking()
{
  auto is_t1 = chrono::high_resolution_clock::now();

  for (int n=0; n<iter_shr; ++n)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pset_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pset_cloud->width = pset.rows();
    pset_cloud->height = 1;
    pset_cloud->points.resize(pset_cloud->width * pset_cloud->height);
    for (size_t i = 0; i < pset_cloud->points.size(); ++i)
    {
      pset_cloud->points[i].x = pset(i, 0);
      pset_cloud->points[i].y = pset(i, 1);
      pset_cloud->points[i].z = pset(i, 2);
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> pset_tree;
    pset_tree.setInputCloud(pset_cloud);

    // calculate confidence
    Eigen::VectorXd conf = Eigen::VectorXd::Zero(pset.rows());
    Eigen::MatrixXd newpset2;
    newpset2.resize(selected_num, 3);
    newpset2 = pset;
    double CONFIDENCE_TH = 0.5;

    vector<int> pointIdxNKNSearch(k_KNN);
    vector<float> pointNKNSquaredDistance(k_KNN);
    for (int i=0; i<(int)pset.rows(); ++i)
    {
      pointIdxNKNSearch.clear();
      pointNKNSquaredDistance.clear();
      pset_tree.nearestKSearch(pset_cloud->points[i], k_KNN, pointIdxNKNSearch, pointNKNSquaredDistance);

      Eigen::MatrixXd neighbors(k_KNN, 3);
      for (int j=0; j<k_KNN; ++j)
        neighbors.row(j) = pset.row(pointIdxNKNSearch[j]);

      Eigen::Vector3d local_mean = neighbors.colwise().mean();
      neighbors.rowwise() -= local_mean.transpose();

      BDCSVD<Eigen::MatrixXd> svd(neighbors, ComputeThinU | ComputeThinV);
      conf(i) = svd.singularValues()(0) / svd.singularValues().sum();
      // compute linear projection
      if (conf(i) < CONFIDENCE_TH) continue;

      newpset2.row(i) = svd.matrixU().col(0).transpose() * ( svd.matrixU().col(0) * (pset.row(i) - local_mean.transpose()) ) + local_mean.transpose();
    }

    pset = newpset2;
  }

  auto is_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> is_ms = is_t2 - is_t1;
  double is_time = (double)is_ms.count();
  ROS_INFO("\033[32m[SSD] inliers shrinking time = %lf ms.\033[32m", is_time);
}

vector<vector<int>> sk_decomp::hybrid_active_sample_sets(int idx, Eigen::Vector3d& p_cut, Eigen::Vector3d& v_cut)
{
  vector<vector<int>> sample_sets;
  vector<int> isoncut(pcd_size_, 0);
  pcloud_isoncut(p_cut, v_cut, isoncut, P.datas, pcd_size_);

  vector<int> inplane;
  inplane.reserve(pcd_size_);
  for (int i=0; i<pcd_size_; ++i)
  {
    if (isoncut[i] == 1)
      inplane.push_back(i);
  }
  if (inplane.empty())
    return sample_sets;

  vector<int> seed_component;
  if (hybrid_component_radius_scale_ > 0.0)
  {
    unordered_map<int, int> inplane_local;
    inplane_local.reserve(inplane.size());
    for (int i=0; i<(int)inplane.size(); ++i)
      inplane_local[inplane[i]] = i;

    if (idx >= 0 && idx < pcd_size_ && inplane_local.find(idx) != inplane_local.end())
    {
      const double radius = std::max(1e-4, hybrid_component_radius_scale_ * epsilon);
      const double radius_sq = radius * radius;
      vector<char> visited(inplane.size(), 0);
      queue<int> q;
      q.push(idx);
      visited[inplane_local[idx]] = 1;

      while (!q.empty())
      {
        int cur = q.front();
        q.pop();
        seed_component.push_back(cur);
        const Eigen::Vector3d cur_pt = P.pts_mat.row(cur);
        for (int nb:inplane)
        {
          auto local_it = inplane_local.find(nb);
          if (local_it == inplane_local.end() || visited[local_it->second])
            continue;
          const Eigen::Vector3d nb_pt = P.pts_mat.row(nb);
          if ((cur_pt - nb_pt).squaredNorm() <= radius_sq)
          {
            visited[local_it->second] = 1;
            q.push(nb);
          }
        }
      }
    }

    if ((int)seed_component.size() >= hybrid_min_component_size_)
      sample_sets.push_back(seed_component);
  }

  vector<int> adaptive_cluster;
  if (sample_sets.empty())
  {
    Eigen::MatrixXd clusterIdxs(inplane.size(), 1);
    for (int i=0; i<(int)inplane.size(); ++i)
      clusterIdxs(i,0) = inplane[i];
    Eigen::MatrixXd cluster_idxs = adp_utils_->cluster_proc(idx, clusterIdxs, P.pts_mat, epsilon);
    adaptive_cluster.reserve(cluster_idxs.rows());
    for (int i=0; i<(int)cluster_idxs.rows(); ++i)
    {
      int id = (int)cluster_idxs(i,0);
      if (id >= 0 && id < pcd_size_)
        adaptive_cluster.push_back(id);
    }
    if ((int)adaptive_cluster.size() >= hybrid_min_component_size_)
      sample_sets.push_back(adaptive_cluster);
  }

  if (sample_sets.empty())
  {
    if (!seed_component.empty())
      sample_sets.push_back(seed_component);
    else if (!adaptive_cluster.empty())
      sample_sets.push_back(adaptive_cluster);
    else
      sample_sets.push_back(inplane);
  }

  return sample_sets;
}

bool sk_decomp::estimate_drosa_from_indices(const vector<int>& indices, Eigen::Vector3d& point, double& confidence)
{
  confidence = 0.0;
  if ((int)indices.size() < 3)
    return false;

  Eigen::MatrixXd extract_pts(indices.size(), 3);
  Eigen::MatrixXd extract_nrs(indices.size(), 3);
  int valid_num = 0;
  for (int idx:indices)
  {
    if (idx < 0 || idx >= P.pts_mat.rows() || idx >= P.nrs_mat.rows())
      continue;
    extract_pts.row(valid_num) = P.pts_mat.row(idx);
    extract_nrs.row(valid_num) = P.nrs_mat.row(idx);
    valid_num++;
  }

  if (valid_num < 3)
    return false;

  extract_pts.conservativeResize(valid_num, 3);
  extract_nrs.conservativeResize(valid_num, 3);

  Eigen::Vector3d mean = extract_pts.colwise().mean();
  Eigen::Vector3d projected = closest_projection_point(extract_pts, extract_nrs);
  if (!projected.allFinite() || projected.cwiseAbs().maxCoeff() > 1.0)
  {
    point = mean;
    confidence = 0.25;
    return point.allFinite();
  }

  confidence = 1.0;
  const double projection_weight = std::min(1.0, std::max(0.0, hybrid_projection_weight_));
  point = projection_weight * projected + (1.0 - projection_weight) * mean;
  return point.allFinite();
}

bool sk_decomp::robust_drosa_point(const vector<vector<int>>& sample_sets, Eigen::Vector3d& point)
{
  double confidence = 0.0;
  for (const vector<int>& sample_set:sample_sets)
  {
    if (estimate_drosa_from_indices(sample_set, point, confidence))
      return true;
  }

  return false;
}

void sk_decomp::hybrid_dcrosa()
{
  if (hybrid_dcrosa_iter_ <= 0 || pset.rows() <= 1)
    return;

  auto dc_t1 = chrono::high_resolution_clock::now();

  const double neighbor_blend = std::max(0.0, std::min(1.0, hybrid_dcrosa_neighbor_blend_));
  const int neighbor_num = std::max(2, std::min(k_KNN, (int)pset.rows()));

  for (int iter=0; iter<hybrid_dcrosa_iter_; ++iter)
  {
    Eigen::MatrixXd averaged = pset;
    for (int i=0; i<pset.rows(); ++i)
    {
      if (i >= (int)P.surf_neighs.size() || P.surf_neighs[i].empty())
        continue;

      Eigen::Vector3d mean = Eigen::Vector3d::Zero();
      int valid_num = 0;
      for (int neigh_idx:P.surf_neighs[i])
      {
        if (neigh_idx < 0 || neigh_idx >= pset.rows())
          continue;
        mean += pset.row(neigh_idx).transpose();
        valid_num++;
      }
      if (valid_num <= 0)
        continue;

      mean /= valid_num;
      averaged.row(i) = (neighbor_blend * mean + (1.0 - neighbor_blend) * pset.row(i).transpose()).transpose();
    }
    pset = averaged;


    pcl::PointCloud<pcl::PointXYZ>::Ptr pset_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pset_cloud->width = pset.rows();
    pset_cloud->height = 1;
    pset_cloud->points.resize(pset_cloud->width * pset_cloud->height);
    for (size_t i=0; i<pset_cloud->points.size(); ++i)
    {
      pset_cloud->points[i].x = pset(i, 0);
      pset_cloud->points[i].y = pset(i, 1);
      pset_cloud->points[i].z = pset(i, 2);
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> pset_tree;
    pset_tree.setInputCloud(pset_cloud);

    Eigen::MatrixXd projected = pset;
    vector<int> pointIdxNKNSearch;
    vector<float> pointNKNSquaredDistance;
    for (int i=0; i<(int)pset.rows(); ++i)
    {
      pointIdxNKNSearch.clear();
      pointNKNSquaredDistance.clear();
      if (pset_tree.nearestKSearch(pset_cloud->points[i],
                                   neighbor_num,
                                   pointIdxNKNSearch,
                                   pointNKNSquaredDistance) <= 0)
        continue;

      Eigen::MatrixXd neighbors(pointIdxNKNSearch.size(), 3);
      int valid_num = 0;
      for (int idx:pointIdxNKNSearch)
      {
        if (idx < 0 || idx >= pset.rows())
          continue;
        neighbors.row(valid_num++) = pset.row(idx);
      }
      if (valid_num < 3)
        continue;
      neighbors.conservativeResize(valid_num, 3);

      const Eigen::Vector3d local_mean = neighbors.colwise().mean();
      Eigen::MatrixXd centered = neighbors.rowwise() - local_mean.transpose();
      BDCSVD<Eigen::MatrixXd> svd(centered, ComputeThinU | ComputeThinV);
      const Eigen::VectorXd singular_values = svd.singularValues();
      const double sigma_sum = singular_values.sum();
      if (sigma_sum <= 1e-9)
        continue;

      const double confidence = singular_values(0) / sigma_sum;
      if (confidence < hybrid_dcrosa_confidence_th_)
        continue;

      const Eigen::Vector3d dir = svd.matrixV().col(0);
      const Eigen::Vector3d point = pset.row(i).transpose();
      projected.row(i) = (local_mean + dir * dir.dot(point - local_mean)).transpose();
    }
    pset = projected;
  }

  auto dc_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> dc_ms = dc_t2 - dc_t1;
  ROS_INFO("\033[32m[SSD] hybrid dcrosa time = %lf ms.\033[32m", (double)dc_ms.count());
}

void sk_decomp::smooth_hybrid_drosa()
{
  if (hybrid_smooth_iter_ <= 0 || pset.rows() <= 2)
    return;

  const int neighbor_num = std::max(3, std::min(k_KNN, (int)pset.rows()));
  for (int iter=0; iter<hybrid_smooth_iter_; ++iter)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pset_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pset_cloud->points.reserve(pset.rows());
    pcl::PointXYZ pt;
    for (int i=0; i<pset.rows(); ++i)
    {
      pt.x = pset(i,0);
      pt.y = pset(i,1);
      pt.z = pset(i,2);
      pset_cloud->points.push_back(pt);
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> pset_tree;
    pset_tree.setInputCloud(pset_cloud);
    Eigen::MatrixXd smoothed = pset;
    vector<int> nearest;
    vector<float> nn_squared_distance;
    for (int i=0; i<pset.rows(); ++i)
    {
      nearest.clear();
      nn_squared_distance.clear();
      if (pset_tree.nearestKSearch(pset_cloud->points[i], neighbor_num, nearest, nn_squared_distance) <= 0)
        continue;

      Eigen::Vector3d mean = Eigen::Vector3d::Zero();
      int valid_num = 0;
      for (int idx:nearest)
      {
        if (idx < 0 || idx >= pset.rows())
          continue;
        mean += pset.row(idx).transpose();
        valid_num++;
      }
      if (valid_num > 0)
      {
        mean /= valid_num;
        smoothed.row(i) = (0.85 * pset.row(i).transpose() + 0.15 * mean).transpose();
      }
    }
    pset = smoothed;
  }
}

void sk_decomp::build_dcrosa_topology()
{
  P.dcrosa_skelver.resize(0, 0);
  P.dcrosa_skeladj.resize(0, 0);
  P.dcrosa_skeleton_paths.clear();
  dcrosa_topology_effective_radius_ = 0.0;

  if (pset.rows() <= 1)
    return;

  auto topo_t1 = chrono::high_resolution_clock::now();

  const int sample_num = (int)pset.rows();
  vector<Eigen::Vector3d> dcrosa_points(sample_num, Eigen::Vector3d::Zero());
  for (int i=0; i<sample_num; ++i)
  {
    const Eigen::Vector3d p_norm = pset.row(i).transpose();
    dcrosa_points[i] = p_norm * P.scale + P.center;
  }

  vector<double> nearest_dist;
  nearest_dist.reserve(sample_num);
  for (int i=0; i<sample_num; ++i)
  {
    double best = std::numeric_limits<double>::infinity();
    for (int j=0; j<sample_num; ++j)
    {
      if (i == j)
        continue;
      const double d = (dcrosa_points[i] - dcrosa_points[j]).norm();
      if (std::isfinite(d) && d < best)
        best = d;
    }
    if (std::isfinite(best))
      nearest_dist.push_back(best);
  }

  const double median_nn = medianPositive(nearest_dist);
  double auto_radius = std::max(2.0 * median_nn, 2.0 * P.scale * pt_downsample_voxel_size);
  if (!std::isfinite(auto_radius) || auto_radius <= 1e-6)
    auto_radius = std::max(1e-3, P.scale * pt_downsample_voxel_size);
  const double sample_radius = dcrosa_topology_sample_radius_ > 0.0 ? dcrosa_topology_sample_radius_ : auto_radius;
  dcrosa_topology_effective_radius_ = sample_radius;
  const double sample_radius_sq = sample_radius * sample_radius;

  vector<int> corresp(sample_num, -1);
  vector<double> min_dist_sq(sample_num, std::numeric_limits<double>::infinity());
  vector<Eigen::Vector3d> vertices;
  vertices.reserve(sample_num);

  auto addVertex = [&](int idx)
  {
    if (idx < 0 || idx >= sample_num || !dcrosa_points[idx].allFinite())
      return false;

    const int vertex_id = (int)vertices.size();
    vertices.push_back(dcrosa_points[idx]);
    for (int i=0; i<sample_num; ++i)
    {
      if (!dcrosa_points[i].allFinite())
        continue;
      const double d2 = (dcrosa_points[i] - dcrosa_points[idx]).squaredNorm();
      if (d2 <= sample_radius_sq && (corresp[i] < 0 || d2 < min_dist_sq[i]))
      {
        corresp[i] = vertex_id;
        min_dist_sq[i] = d2;
      }
      else if (d2 < min_dist_sq[i])
      {
        min_dist_sq[i] = d2;
      }
    }
    return true;
  };

  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  int valid_num = 0;
  for (const Eigen::Vector3d& p:dcrosa_points)
  {
    if (!p.allFinite())
      continue;
    mean += p;
    valid_num++;
  }
  if (valid_num <= 0)
    return;
  mean /= valid_num;

  int first_idx = 0;
  double farthest_from_mean = -1.0;
  for (int i=0; i<sample_num; ++i)
  {
    if (!dcrosa_points[i].allFinite())
      continue;
    const double d2 = (dcrosa_points[i] - mean).squaredNorm();
    if (d2 > farthest_from_mean)
    {
      farthest_from_mean = d2;
      first_idx = i;
    }
  }
  addVertex(first_idx);

  while ((int)vertices.size() < sample_num)
  {
    int best_idx = -1;
    double best_dist_sq = sample_radius_sq;
    for (int i=0; i<sample_num; ++i)
    {
      if (!dcrosa_points[i].allFinite() || corresp[i] >= 0)
        continue;
      if (min_dist_sq[i] > best_dist_sq)
      {
        best_dist_sq = min_dist_sq[i];
        best_idx = i;
      }
    }

    if (best_idx < 0)
      break;
    addVertex(best_idx);
  }

  for (int i=0; i<sample_num; ++i)
  {
    if (corresp[i] >= 0 || !dcrosa_points[i].allFinite() || vertices.empty())
      continue;

    int best_vertex = -1;
    double best_dist_sq = std::numeric_limits<double>::infinity();
    for (int v=0; v<(int)vertices.size(); ++v)
    {
      const double d2 = (dcrosa_points[i] - vertices[v]).squaredNorm();
      if (d2 < best_dist_sq)
      {
        best_dist_sq = d2;
        best_vertex = v;
      }
    }
    corresp[i] = best_vertex;
  }

  if (vertices.empty())
    return;

  vector<int> support(vertices.size(), 0);
  for (int vertex_id:corresp)
  {
    if (vertex_id >= 0 && vertex_id < (int)support.size())
      support[vertex_id]++;
  }

  const int min_support = std::max(1, dcrosa_topology_min_support_);
  vector<int> old_to_valid(vertices.size(), -1);
  vector<Eigen::Vector3d> valid_vertices;
  valid_vertices.reserve(vertices.size());
  for (int v=0; v<(int)vertices.size(); ++v)
  {
    if (support[v] >= min_support || (int)vertices.size() <= 2)
    {
      old_to_valid[v] = (int)valid_vertices.size();
      valid_vertices.push_back(vertices[v]);
    }
  }

  if (valid_vertices.empty())
  {
    const int keep_num = std::min(2, (int)vertices.size());
    for (int v=0; v<keep_num; ++v)
    {
      old_to_valid[v] = (int)valid_vertices.size();
      valid_vertices.push_back(vertices[v]);
    }
  }

  for (int i=0; i<sample_num; ++i)
  {
    if (corresp[i] < 0)
      continue;

    int mapped = old_to_valid[corresp[i]];
    if (mapped < 0)
    {
      double best_dist_sq = std::numeric_limits<double>::infinity();
      for (int v=0; v<(int)valid_vertices.size(); ++v)
      {
        const double d2 = (dcrosa_points[i] - valid_vertices[v]).squaredNorm();
        if (d2 < best_dist_sq)
        {
          best_dist_sq = d2;
          mapped = v;
        }
      }
    }
    corresp[i] = mapped;
  }
  vertices.swap(valid_vertices);

  Eigen::MatrixXi adj = Eigen::MatrixXi::Zero(vertices.size(), vertices.size());
  for (int i=0; i<sample_num; ++i)
  {
    if (i >= (int)P.surf_neighs.size())
      continue;

    const int vi = corresp[i];
    if (vi < 0 || vi >= adj.rows())
      continue;

    for (int neigh_idx:P.surf_neighs[i])
    {
      if (neigh_idx < 0 || neigh_idx >= sample_num)
        continue;

      const int vj = corresp[neigh_idx];
      if (vj < 0 || vj >= adj.rows() || vi == vj)
        continue;

      adj(vi, vj) = 1;
      adj(vj, vi) = 1;
    }
  }

  if (adj.rows() > 1)
  {
    vector<char> connected(adj.rows(), 0);
    connected[0] = 1;
    int connected_num = 1;
    while (connected_num < adj.rows())
    {
      int best_i = -1, best_j = -1;
      double best_dist_sq = std::numeric_limits<double>::infinity();
      for (int i=0; i<adj.rows(); ++i)
      {
        if (!connected[i])
          continue;
        for (int j=0; j<adj.rows(); ++j)
        {
          if (connected[j])
            continue;
          const double d2 = (vertices[i] - vertices[j]).squaredNorm();
          if (d2 < best_dist_sq)
          {
            best_dist_sq = d2;
            best_i = i;
            best_j = j;
          }
        }
      }
      if (best_i < 0 || best_j < 0)
        break;
      adj(best_i, best_j) = 1;
      adj(best_j, best_i) = 1;
      connected[best_j] = 1;
      connected_num++;
    }
  }

  const int raw_edge_num = countGraphEdges(adj);
  const int raw_triangle_num = countGraphTriangles(adj);

  if (adj.rows() > 1)
  {
    struct CandidateEdge
    {
      int a;
      int b;
      double w;
    };

    vector<CandidateEdge> candidate_edges;
    candidate_edges.reserve(raw_edge_num);
    for (int i=0; i<adj.rows(); ++i)
    {
      for (int j=i+1; j<adj.cols(); ++j)
      {
        if (adj(i,j) != 1)
          continue;
        const double dist = (vertices[i] - vertices[j]).norm();
        if (!std::isfinite(dist) || dist <= 1e-9)
          continue;
        candidate_edges.push_back({i, j, dist});
      }
    }

    sort(candidate_edges.begin(), candidate_edges.end(), [](const CandidateEdge& lhs, const CandidateEdge& rhs) {
      return lhs.w < rhs.w;
    });

    vector<int> parent(adj.rows());
    iota(parent.begin(), parent.end(), 0);
    function<int(int)> findRoot = [&](int x) -> int {
      while (parent[x] != x)
      {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };
    auto unite = [&](int a, int b) -> bool {
      int ra = findRoot(a);
      int rb = findRoot(b);
      if (ra == rb)
        return false;
      parent[rb] = ra;
      return true;
    };

    Eigen::MatrixXi backbone_adj = Eigen::MatrixXi::Zero(adj.rows(), adj.cols());
    for (int i=0; i<backbone_adj.rows(); ++i)
      backbone_adj(i,i) = 0;

    for (const CandidateEdge& edge:candidate_edges)
    {
      if (!unite(edge.a, edge.b))
        continue;
      backbone_adj(edge.a, edge.b) = 1;
      backbone_adj(edge.b, edge.a) = 1;
    }

    // If surface-KNN produced disconnected components, connect them with the
    // nearest geometric bridges. This preserves one skeleton backbone without
    // reintroducing the dense local triangle mesh.
    bool connected_all = false;
    while (!connected_all)
    {
      vector<int> roots(adj.rows(), 0);
      unordered_map<int, vector<int>> components;
      for (int i=0; i<adj.rows(); ++i)
      {
        roots[i] = findRoot(i);
        components[roots[i]].push_back(i);
      }
      if (components.size() <= 1)
      {
        connected_all = true;
        break;
      }

      int best_i = -1, best_j = -1;
      double best_dist_sq = std::numeric_limits<double>::infinity();
      for (auto it_a=components.begin(); it_a!=components.end(); ++it_a)
      {
        auto it_b = it_a;
        ++it_b;
        for (; it_b!=components.end(); ++it_b)
        {
          for (int a:it_a->second)
          {
            for (int b:it_b->second)
            {
              const double d2 = (vertices[a] - vertices[b]).squaredNorm();
              if (d2 < best_dist_sq)
              {
                best_dist_sq = d2;
                best_i = a;
                best_j = b;
              }
            }
          }
        }
      }

      if (best_i < 0 || best_j < 0)
        break;

      backbone_adj(best_i, best_j) = 1;
      backbone_adj(best_j, best_i) = 1;
      unite(best_i, best_j);
    }

    adj.swap(backbone_adj);
  }

  const int backbone_edge_num = countGraphEdges(adj);
  const int backbone_triangle_num = countGraphTriangles(adj);

  {
    bool collapsed = true;
    int collapse_iter = 0;
    const int collapse_iter_limit = std::max(1, (int)vertices.size() * (int)vertices.size());
    while (collapsed && collapse_iter++ < collapse_iter_limit)
    {
      collapsed = false;
      int edge_a = -1, edge_b = -1;
      double shortest = std::numeric_limits<double>::infinity();

      for (int i=0; i<adj.rows(); ++i)
      {
        for (int j=i+1; j<adj.cols(); ++j)
        {
          if (adj(i,j) != 1)
            continue;

          bool triangle_edge = false;
          for (int k=0; k<adj.rows(); ++k)
          {
            if (k == i || k == j)
              continue;
            if (adj(i,k) == 1 && adj(j,k) == 1)
            {
              triangle_edge = true;
              break;
            }
          }
          if (!triangle_edge)
            continue;

          const double len = (vertices[i] - vertices[j]).norm();
          if (len < shortest)
          {
            shortest = len;
            edge_a = i;
            edge_b = j;
          }
        }
      }

      if (edge_a < 0 || edge_b < 0)
        break;

      vertices[edge_a] = 0.5 * (vertices[edge_a] + vertices[edge_b]);
      for (int k=0; k<adj.rows(); ++k)
      {
        if (adj(edge_b,k) == 1 && k != edge_a)
        {
          adj(edge_a,k) = 1;
          adj(k,edge_a) = 1;
        }
      }
      adj(edge_a, edge_a) = 0;
      for (int i=0; i<sample_num; ++i)
        if (corresp[i] == edge_b)
          corresp[i] = edge_a;

      collapsed = true;
      vector<int> remap(vertices.size(), -1);
      vector<Eigen::Vector3d> compact_vertices;
      compact_vertices.reserve(vertices.size() - 1);
      for (int i=0; i<(int)vertices.size(); ++i)
      {
        if (i == edge_b)
          continue;
        remap[i] = (int)compact_vertices.size();
        compact_vertices.push_back(vertices[i]);
      }

      Eigen::MatrixXi compact_adj = Eigen::MatrixXi::Zero(compact_vertices.size(), compact_vertices.size());
      for (int i=0; i<adj.rows(); ++i)
      {
        if (remap[i] < 0)
          continue;
        for (int j=0; j<adj.cols(); ++j)
        {
          if (remap[j] < 0 || i == j || adj(i,j) != 1)
            continue;
          compact_adj(remap[i], remap[j]) = 1;
        }
      }

      for (int& c:corresp)
      {
        if (c >= 0 && c < (int)remap.size())
          c = remap[c];
      }
      vertices.swap(compact_vertices);
      adj.swap(compact_adj);
    }
  }

  const int final_triangle_num = countGraphTriangles(adj);

  P.dcrosa_skelver.resize(vertices.size(), 3);
  for (int i=0; i<(int)vertices.size(); ++i)
    P.dcrosa_skelver.row(i) = vertices[i].transpose();
  P.dcrosa_skeladj = adj;
  P.dcrosa_skeleton_paths = extract_dcrosa_branches(P.dcrosa_skelver, P.dcrosa_skeladj);

  const int edge_num = countGraphEdges(P.dcrosa_skeladj);

  auto topo_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> topo_ms = topo_t2 - topo_t1;
  ROS_INFO("\033[32m[SSD] dcROSA topology: samples=%d, vertices=%d, edges=%d, branches=%zu, radius=%lf, tri(raw/backbone/final)=%d/%d/%d, edge(raw/backbone/final)=%d/%d/%d, time=%lf ms.\033[32m",
           sample_num,
           (int)P.dcrosa_skelver.rows(),
           edge_num,
           P.dcrosa_skeleton_paths.size(),
           sample_radius,
           raw_triangle_num,
           backbone_triangle_num,
           final_triangle_num,
           raw_edge_num,
           backbone_edge_num,
           edge_num,
           (double)topo_ms.count());
}

vector<vector<Eigen::Vector3d>> sk_decomp::extract_dcrosa_branches(const Eigen::MatrixXd& skelver,
                                                                   const Eigen::MatrixXi& skeladj) const
{
  vector<vector<Eigen::Vector3d>> paths;
  const int n = skeladj.rows();
  if (n <= 0 || skeladj.cols() != n || skelver.rows() < n || skelver.cols() < 3)
    return paths;

  vector<vector<int>> adj(n);
  for (int i=0; i<n; ++i)
  {
    for (int j=0; j<n; ++j)
    {
      if (i != j && skeladj(i,j) == 1)
        adj[i].push_back(j);
    }
  }

  vector<vector<char>> visited(n);
  for (int i=0; i<n; ++i)
    visited[i].assign(adj[i].size(), 0);

  auto markVisited = [&](int a, int b)
  {
    for (int k=0; k<(int)adj[a].size(); ++k)
    {
      if (adj[a][k] == b)
      {
        visited[a][k] = 1;
        break;
      }
    }
  };

  auto edgeVisited = [&](int a, int b) -> bool
  {
    for (int k=0; k<(int)adj[a].size(); ++k)
      if (adj[a][k] == b)
        return visited[a][k] != 0;
    return true;
  };

  auto emitPath = [&](const vector<int>& ids)
  {
    if ((int)ids.size() < 2)
      return;
    vector<Eigen::Vector3d> path;
    path.reserve(ids.size());
    for (int id:ids)
    {
      if (id < 0 || id >= skelver.rows())
        return;
      Eigen::Vector3d point = skelver.row(id).transpose();
      if (!point.allFinite())
        return;
      path.push_back(point);
    }
    paths.push_back(path);
  };

  vector<int> key_nodes;
  for (int i=0; i<n; ++i)
  {
    if ((int)adj[i].size() != 2 && !adj[i].empty())
      key_nodes.push_back(i);
  }

  for (int start:key_nodes)
  {
    for (int next:adj[start])
    {
      if (edgeVisited(start, next))
        continue;

      vector<int> path{start, next};
      int prev = start;
      int cur = next;
      markVisited(prev, cur);
      markVisited(cur, prev);

      while ((int)adj[cur].size() == 2)
      {
        int candidate = adj[cur][0] == prev ? adj[cur][1] : adj[cur][0];
        if (edgeVisited(cur, candidate))
          break;
        prev = cur;
        cur = candidate;
        markVisited(prev, cur);
        markVisited(cur, prev);
        path.push_back(cur);
      }
      emitPath(path);
    }
  }

  for (int i=0; i<n; ++i)
  {
    for (int next:adj[i])
    {
      if (edgeVisited(i, next))
        continue;

      vector<int> path{i, next};
      int prev = i;
      int cur = next;
      markVisited(prev, cur);
      markVisited(cur, prev);

      while (cur != i && (int)adj[cur].size() == 2)
      {
        int candidate = adj[cur][0] == prev ? adj[cur][1] : adj[cur][0];
        if (edgeVisited(cur, candidate))
          break;
        prev = cur;
        cur = candidate;
        markVisited(prev, cur);
        markVisited(cur, prev);
        path.push_back(cur);
      }
      emitPath(path);
    }
  }

  return paths;
}

double sk_decomp::dcrosa_branch_length(const vector<Eigen::Vector3d>& path) const
{
  double length = 0.0;
  for (int i=1; i<(int)path.size(); ++i)
  {
    if (!path[i-1].allFinite() || !path[i].allFinite())
      continue;
    const double seg_len = (path[i] - path[i-1]).norm();
    if (std::isfinite(seg_len))
      length += seg_len;
  }
  return length;
}

int sk_decomp::nearest_dcrosa_branch(const Eigen::Vector3d& point,
                                     const vector<char>* valid_branch_mask,
                                     double* best_dist_sq_out,
                                     double* second_best_dist_sq_out) const
{
  if (best_dist_sq_out != nullptr)
    *best_dist_sq_out = std::numeric_limits<double>::infinity();
  if (second_best_dist_sq_out != nullptr)
    *second_best_dist_sq_out = std::numeric_limits<double>::infinity();

  if (!point.allFinite())
    return -1;

  double best_dist_sq = std::numeric_limits<double>::infinity();
  double second_best_dist_sq = std::numeric_limits<double>::infinity();
  int best_branch = -1;
  for (int i=0; i<(int)P.dcrosa_skeleton_paths.size(); ++i)
  {
    if (valid_branch_mask != nullptr &&
        (i >= (int)valid_branch_mask->size() || !(*valid_branch_mask)[i]))
      continue;

    const vector<Eigen::Vector3d>& path = P.dcrosa_skeleton_paths[i];
    if ((int)path.size() < 2)
      continue;

    double path_dist_sq = std::numeric_limits<double>::infinity();
    for (int j=1; j<(int)path.size(); ++j)
    {
      const double dist_sq = pointSegmentDistanceSq(point, path[j-1], path[j]);
      if (dist_sq < path_dist_sq)
        path_dist_sq = dist_sq;
    }

    if (!std::isfinite(path_dist_sq))
      continue;

    if (path_dist_sq < best_dist_sq)
    {
      second_best_dist_sq = best_dist_sq;
      best_dist_sq = path_dist_sq;
      best_branch = i;
    }
    else if (path_dist_sq < second_best_dist_sq)
    {
      second_best_dist_sq = path_dist_sq;
    }
  }

  if (best_branch >= 0)
  {
    if (best_dist_sq_out != nullptr)
      *best_dist_sq_out = best_dist_sq;
    if (second_best_dist_sq_out != nullptr)
      *second_best_dist_sq_out = second_best_dist_sq;
    return best_branch;
  }

  if (valid_branch_mask != nullptr)
    return -1;

  for (int i=0; i<P.dcrosa_skelver.rows(); ++i)
  {
    const Eigen::Vector3d vertex = P.dcrosa_skelver.row(i).transpose();
    if (!vertex.allFinite())
      continue;
    const double dist_sq = (point - vertex).squaredNorm();
    if (dist_sq < best_dist_sq)
    {
      best_dist_sq = dist_sq;
      best_branch = i;
    }
  }

  if (best_dist_sq_out != nullptr)
    *best_dist_sq_out = best_dist_sq;
  if (second_best_dist_sq_out != nullptr)
    *second_best_dist_sq_out = second_best_dist_sq;
  return best_branch;
}

void sk_decomp::apply_dcrosa_guided_decomposition()
{
  if (P.dcrosa_skeleton_paths.empty())
    return;

  auto guide_t1 = chrono::high_resolution_clock::now();
  const vector<Eigen::Vector3d>& decomp_points = active_decomp_points();
  if (decomp_points.empty())
    return;

  const double base_radius = std::isfinite(dcrosa_topology_effective_radius_) && dcrosa_topology_effective_radius_ > 1e-6
                               ? dcrosa_topology_effective_radius_
                               : std::max(1e-3, P.scale * pt_downsample_voxel_size);
  const double min_branch_len = std::max(0.0, dcrosa_guided_min_branch_len_scale_) * base_radius;
  const double confidence_ratio = std::max(0.0, dcrosa_guided_confidence_ratio_);
  const double confidence_ratio_sq = confidence_ratio * confidence_ratio;
  const int min_group_size = std::max(1, dcrosa_guided_min_group_size_);

  vector<double> branch_lengths(P.dcrosa_skeleton_paths.size(), 0.0);
  vector<char> valid_branch_mask(P.dcrosa_skeleton_paths.size(), 0);
  int valid_branch_num = 0;
  int short_branch_num = 0;
  for (int i=0; i<(int)P.dcrosa_skeleton_paths.size(); ++i)
  {
    branch_lengths[i] = dcrosa_branch_length(P.dcrosa_skeleton_paths[i]);
    const bool valid_branch = P.dcrosa_skeleton_paths.size() <= 1 || branch_lengths[i] >= min_branch_len;
    valid_branch_mask[i] = valid_branch ? 1 : 0;
    if (valid_branch)
      valid_branch_num++;
    else
      short_branch_num++;
  }

  if (valid_branch_num <= 0)
  {
    ROS_WARN("\033[33m[SSD] dcROSA-guided decomp skipped: no branch longer than %lf m.\033[33m",
             min_branch_len);
    return;
  }

  const int before = (int)P.subspace_sets.size();

  unordered_map<int, vector<int>> branch_groups;
  vector<int> fallback_group;
  int assigned_num = 0;
  int ambiguous_num = 0;
  int fallback_num = 0;
  int tiny_group_merge_num = 0;

  for (int node_idx=0; node_idx<(int)decomp_points.size(); ++node_idx)
  {
    double best_dist_sq = std::numeric_limits<double>::infinity();
    double second_best_dist_sq = std::numeric_limits<double>::infinity();
    const int branch_id = nearest_dcrosa_branch(decomp_points[node_idx],
                                                &valid_branch_mask,
                                                &best_dist_sq,
                                                &second_best_dist_sq);
    bool confident = branch_id >= 0 && std::isfinite(best_dist_sq);
    if (confident && std::isfinite(second_best_dist_sq))
      confident = best_dist_sq <= confidence_ratio_sq * second_best_dist_sq;

    if (branch_id >= 0)
    {
      branch_groups[branch_id].push_back(node_idx);
      assigned_num++;
      if (!confident)
        ambiguous_num++;
    }
    else
    {
      fallback_group.push_back(node_idx);
      fallback_num++;
    }
  }

  if (!fallback_group.empty() && !branch_groups.empty())
  {
    auto dominant = branch_groups.begin();
    for (auto it=branch_groups.begin(); it!=branch_groups.end(); ++it)
    {
      if (it->second.size() > dominant->second.size())
        dominant = it;
    }
    dominant->second.insert(dominant->second.end(), fallback_group.begin(), fallback_group.end());
    fallback_group.clear();
  }

  while (branch_groups.size() > 1)
  {
    auto tiny_it = branch_groups.end();
    for (auto it=branch_groups.begin(); it!=branch_groups.end(); ++it)
    {
      if (it->second.empty())
        continue;
      if ((int)it->second.size() >= min_group_size)
        continue;
      if (tiny_it == branch_groups.end() || it->second.size() < tiny_it->second.size())
        tiny_it = it;
    }

    if (tiny_it == branch_groups.end())
      break;

    Eigen::Vector3d tiny_center = Eigen::Vector3d::Zero();
    int tiny_count = 0;
    for (int node_idx:tiny_it->second)
    {
      if (node_idx >= 0 && node_idx < (int)decomp_points.size() && decomp_points[node_idx].allFinite())
      {
        tiny_center += decomp_points[node_idx];
        tiny_count++;
      }
    }

    if (tiny_count <= 0)
      tiny_center = Eigen::Vector3d::Zero();
    else
      tiny_center /= tiny_count;

    auto target_it = branch_groups.end();
    double target_score = std::numeric_limits<double>::infinity();
    for (auto it=branch_groups.begin(); it!=branch_groups.end(); ++it)
    {
      if (it == tiny_it || it->second.empty())
        continue;

      double best_dist_sq = std::numeric_limits<double>::infinity();
      for (int node_idx:it->second)
      {
        if (node_idx < 0 || node_idx >= (int)decomp_points.size() || !decomp_points[node_idx].allFinite())
          continue;
        const double dist_sq = (tiny_center - decomp_points[node_idx]).squaredNorm();
        if (dist_sq < best_dist_sq)
          best_dist_sq = dist_sq;
      }

      if (!std::isfinite(best_dist_sq))
        best_dist_sq = (double)it->second.size();
      best_dist_sq /= std::max(1.0, (double)it->second.size());
      if (best_dist_sq < target_score)
      {
        target_score = best_dist_sq;
        target_it = it;
      }
    }

    if (target_it == branch_groups.end())
      break;

    target_it->second.insert(target_it->second.end(), tiny_it->second.begin(), tiny_it->second.end());
    branch_groups.erase(tiny_it);
    tiny_group_merge_num++;
  }

  vector<vector<int>> guided_sets;
  guided_sets.reserve(branch_groups.size() + (fallback_group.empty() ? 0 : 1));

  vector<pair<int, vector<int>>> ordered_groups;
  ordered_groups.reserve(branch_groups.size());
  for (auto& pair:branch_groups)
  {
    sort(pair.second.begin(), pair.second.end());
    ordered_groups.push_back({pair.first, pair.second});
  }
  sort(ordered_groups.begin(), ordered_groups.end(), [](const pair<int, vector<int>>& lhs,
                                                         const pair<int, vector<int>>& rhs) {
    return lhs.first < rhs.first;
  });

  for (auto& pair:ordered_groups)
  {
    if (!pair.second.empty())
      guided_sets.push_back(pair.second);
  }

  if (!fallback_group.empty())
  {
    sort(fallback_group.begin(), fallback_group.end());
    guided_sets.push_back(fallback_group);
  }

  if (guided_sets.empty())
    return;

  P.subspace_sets = guided_sets;
  fill_subspace_map(P.subspace_sets, P.node_subspace_map);
  dcrosa_branch_subspace_ready_ = true;

  auto guide_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> guide_ms = guide_t2 - guide_t1;
  ROS_INFO("\033[32m[SSD] dcROSA branch subspace: before=%d, after=%zu, points=%zu, branches=%zu, valid=%d, short=%d, assigned=%d, ambiguous=%d, fallback=%d, tiny_merge=%d, min_len=%lf, conf_ratio=%lf, min_group=%d, fallback_merge=skipped, time=%lf ms.\033[32m",
           before,
           P.subspace_sets.size(),
           decomp_points.size(),
           P.dcrosa_skeleton_paths.size(),
           valid_branch_num,
           short_branch_num,
           assigned_num,
           ambiguous_num,
           fallback_num,
           tiny_group_merge_num,
           min_branch_len,
           confidence_ratio,
           min_group_size,
           (double)guide_ms.count());
}

void sk_decomp::inliers_check()
{
  auto sc_t1 = chrono::high_resolution_clock::now();

  int num = (int)pset.rows();
  int compress_num = (int)pc_set.size();
  pset_check.resize(num+compress_num,3);
  pset_dir.resize(num+compress_num,3);
  vector<int> candidate_ref_indices;
  candidate_ref_indices.reserve(num+compress_num);
  intersec_pts.clear();
  intersec_pts.reserve(num+compress_num);
  intersec_pts.resize(num+compress_num);

  for (int i=0; i<num; ++i)
  {
    Eigen::Vector3d corresp_pt(P.selected_pts_->points[i].x, P.selected_pts_->points[i].y, P.selected_pts_->points[i].z);
    Eigen::Vector3d skl_pt = pset.row(i);
    Eigen::Vector3d dir = corresp_pt - skl_pt;
    // ground-vertical assumption
    dir(2) = 0.0;
    dir.normalize();

    pset_check(i,0) = pset(i,0)*P.scale + P.center(0);
    pset_check(i,1) = pset(i,1)*P.scale + P.center(1);
    pset_check(i,2) = pset(i,2)*P.scale + P.center(2);
    pset_dir(i,0) = dir(0);
    pset_dir(i,1) = dir(1);
    pset_dir(i,2) = dir(2);
    candidate_ref_indices.push_back(i);
  }
  for (int j=0; j<compress_num; ++j)
  {
    Eigen::Vector3d corresp_pt(P.selected_pts_->points[pc_cor_idx[j]].x, P.selected_pts_->points[pc_cor_idx[j]].y, P.selected_pts_->points[pc_cor_idx[j]].z);
    Eigen::Vector3d skl_pt = pc_set[j];
    Eigen::Vector3d dir = corresp_pt - skl_pt;
    // ground-vertical assumption
    dir(2) = 0.0;
    dir.normalize();

    pset_check.row(num+j) = pc_set[j];
    pset_dir(num+j,0) = dir(0);
    pset_dir(num+j,1) = dir(1);
    pset_dir(num+j,2) = dir(2);
    candidate_ref_indices.push_back(pc_cor_idx[j]);
  }

  // inliers safety check
  auto strict_t1 = chrono::high_resolution_clock::now();
  vector<int> inlier_idxs = adp_utils_->raytracing_check(pset_check, pset_dir, intersec_pts);
  auto strict_t2 = chrono::high_resolution_clock::now();
  vector<char> strict_mask(pset_check.rows(), 0);
  for (auto idx:inlier_idxs)
  {
    if (idx >= 0 && idx < (int)strict_mask.size())
      strict_mask[idx] = 1;
  }

  vector<int> strict_rows;
  vector<int> relaxed_rows;
  vector<int> rejected_rows;
  strict_rows.reserve(inlier_idxs.size());
  relaxed_rows.reserve(pset_check.rows());
  rejected_rows.reserve(pset_check.rows());

  double relaxed_threshold = relaxed_candidate_dist_;
  if (relaxed_threshold <= 0.0)
  {
    relaxed_threshold = std::max(adp_utils_ != nullptr ? adp_utils_->resolution_ : 0.0,
                                 P.scale * pt_downsample_voxel_size);
  }

  const double relaxed_threshold_sq = relaxed_threshold * relaxed_threshold;
  auto classify_t1 = chrono::high_resolution_clock::now();
  for (int i=0; i<(int)pset_check.rows(); ++i)
  {
    if (strict_mask[i])
    {
      strict_rows.push_back(i);
      continue;
    }

    bool near_surface = false;
    if (i < (int)candidate_ref_indices.size())
    {
      const Eigen::Vector3d candidate = pset_check.row(i);
      const int ref_idx = candidate_ref_indices[i];
      if (candidate.allFinite() && ref_idx >= 0 && ref_idx < (int)P.selected_pts_->points.size())
      {
        const pcl::PointXYZ& ref_pt = P.selected_pts_->points[ref_idx];
        const Eigen::Vector3d ref_pt_w(ref_pt.x*P.scale + P.center(0),
                                       ref_pt.y*P.scale + P.center(1),
                                       ref_pt.z*P.scale + P.center(2));
        near_surface = (candidate - ref_pt_w).squaredNorm() <= relaxed_threshold_sq;

        if (!near_surface && ref_idx < (int)P.surf_neighs.size())
        {
          const vector<int>& neighs = P.surf_neighs[ref_idx];
          for (int neigh_idx:neighs)
          {
            if (neigh_idx < 0 || neigh_idx >= (int)P.selected_pts_->points.size())
              continue;
            const pcl::PointXYZ& neigh_pt = P.selected_pts_->points[neigh_idx];
            const Eigen::Vector3d neigh_pt_w(neigh_pt.x*P.scale + P.center(0),
                                             neigh_pt.y*P.scale + P.center(1),
                                             neigh_pt.z*P.scale + P.center(2));
            if ((candidate - neigh_pt_w).squaredNorm() <= relaxed_threshold_sq)
            {
              near_surface = true;
              break;
            }
          }
        }
      }
    }

    if (near_surface)
      relaxed_rows.push_back(i);
    else
      rejected_rows.push_back(i);
  }
  auto classify_t2 = chrono::high_resolution_clock::now();

  P.strict_inlier_candidates = candidateRowsToMatrix(pset_check, strict_rows);
  P.relaxed_inlier_candidates = candidateRowsToMatrix(pset_check, relaxed_rows);
  P.rejected_inlier_candidates = candidateRowsToMatrix(pset_check, rejected_rows);

  ROS_INFO("\033[32m[SSD] candidate classification: strict=%zu, relaxed=%zu, rejected=%zu, relaxed_threshold=%lf.\033[32m",
           strict_rows.size(), relaxed_rows.size(), rejected_rows.size(), relaxed_threshold);

  inliers_size = (int)inlier_idxs.size();
  P.inliers.clear();
  P.inliers.reserve(inliers_size);
  Eigen::Vector3d inlier;
  for (int i=0; i<inliers_size; ++i)
  {
    inlier = pset_check.row(inlier_idxs[i]);
    P.inliers.push_back(inlier);
  }
  // voxelization
  pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  inliers_cloud->width = inliers_size;
  inliers_cloud->height = 1;
  inliers_cloud->points.resize(inliers_cloud->width * inliers_cloud->height);
  for (size_t i = 0; i < inliers_cloud->points.size(); ++i)
  {
    inliers_cloud->points[i].x = P.inliers[i](0);
    inliers_cloud->points[i].y = P.inliers[i](1);
    inliers_cloud->points[i].z = P.inliers[i](2);
  }

  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(inliers_cloud);
  vg.setLeafSize(P.scale*pt_downsample_voxel_size, P.scale*pt_downsample_voxel_size, P.scale*pt_downsample_voxel_size);
  vg.filter(*inliers_cloud);

  P.inliers.clear();
  for (size_t i = 0; i < inliers_cloud->points.size(); ++i)
  {
    Eigen::Vector3d inlier;
    inlier(0) = inliers_cloud->points[i].x;
    inlier(1) = inliers_cloud->points[i].y;
    inlier(2) = inliers_cloud->points[i].z;
    P.inliers.push_back(inlier);
  }
  const int strict_voxelized_num = (int)P.inliers.size();

  vector<Eigen::Vector3d> relaxed_points;
  relaxed_points.reserve(relaxed_rows.size());
  for (int row:relaxed_rows)
  {
    if (row >= 0 && row < pset_check.rows() && pset_check.row(row).allFinite())
      relaxed_points.push_back(pset_check.row(row));
  }

  const int configured_cap = hybrid_decomp_max_points_ > 0 ? hybrid_decomp_max_points_ : std::numeric_limits<int>::max();
  const int max_decomp_points = std::max(strict_voxelized_num, configured_cap);
  P.branch_decomp_points.clear();
  P.branch_decomp_points.reserve(std::min(max_decomp_points, strict_voxelized_num + (int)relaxed_points.size()));
  for (const Eigen::Vector3d& strict_inlier:P.inliers)
    P.branch_decomp_points.push_back(strict_inlier);
  appendFarthestSubset(relaxed_points, max_decomp_points, P.branch_decomp_points);
  P.decomp_points = P.branch_decomp_points.empty() ? P.inliers : P.branch_decomp_points;
  ROS_INFO("\033[32m[SSD] branch decomp points prepared: strict_voxelized=%d, relaxed=%zu/%zu, total=%zu, cap=%d.\033[32m",
           strict_voxelized_num,
           P.branch_decomp_points.size() >= (size_t)strict_voxelized_num ? P.branch_decomp_points.size() - strict_voxelized_num : 0,
           relaxed_rows.size(),
           P.branch_decomp_points.size(),
           hybrid_decomp_max_points_);
  // construct visibility graph
  auto vis_graph_t1 = chrono::high_resolution_clock::now();
  P.visibility_graph = build_visibility_graph(P.inliers, adp_utils_->scene, edge_min_dist);
  P.strict_visibility_graph = P.visibility_graph;
  auto vis_graph_t2 = chrono::high_resolution_clock::now();

  auto sc_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> sc_ms = sc_t2 - sc_t1;
  double sc_time = (double)sc_ms.count();
  chrono::duration<double, milli> strict_ms = strict_t2 - strict_t1;
  chrono::duration<double, milli> classify_ms = classify_t2 - classify_t1;
  chrono::duration<double, milli> vis_graph_ms = vis_graph_t2 - vis_graph_t1;
  ROS_INFO("\033[32m[SSD] inliers check breakdown: strict_ray=%lf ms, classify=%lf ms, visibility_graph=%lf ms.\033[32m",
           (double)strict_ms.count(), (double)classify_ms.count(), (double)vis_graph_ms.count());
  ROS_INFO("\033[32m[SSD] inliers check time = %lf ms.\033[32m", sc_time);
}


Eigen::MatrixXi sk_decomp::build_visibility_graph(const vector<Eigen::Vector3d>& points, RTCScene query_scene, double& min_edge_dist)
{
  Eigen::MatrixXi graph;
  if (adp_utils_ == nullptr)
  {
    min_edge_dist = INFINITY;
    graph.resize((int)points.size(), (int)points.size());
    graph.setIdentity();
    return graph;
  }
  adp_utils_->ray_tracing_public_batch(points, query_scene, graph, min_edge_dist);
  return graph;
}

const vector<Eigen::Vector3d>& sk_decomp::active_decomp_points() const
{
  return P.decomp_points.empty() ? P.inliers : P.decomp_points;
}

const vector<vector<int>>& sk_decomp::active_subspace_sets() const
{
  return P.subspace_sets;
}

void sk_decomp::build_branch_subspace_graph()
{
  if (P.branch_decomp_points.empty() || P.dcrosa_skeleton_paths.empty())
    return;

  P.decomp_points = P.branch_decomp_points;
  const int point_num = (int)P.decomp_points.size();
  P.visibility_graph.resize(point_num, point_num);
  P.visibility_graph.setIdentity();
  if (std::isfinite(edge_min_dist) && edge_min_dist > 1e-6)
    return;
  edge_min_dist = std::max(1e-3, P.scale * pt_downsample_voxel_size);
}

void sk_decomp::build_strict_internal_inliers()
{
  P.strict_dense_inliers.reset(new pcl::PointCloud<pcl::PointXYZ>);
  P.dense_inliers.reset(new pcl::PointCloud<pcl::PointXYZ>);
  P.pt_inlier_pairs.clear();

  if (P.inliers.empty() || P.ori_pts_ == nullptr)
    return;

  const double strict_delta = std::max(1e-3, P.scale * pt_downsample_voxel_size);
  auto pushStrictPoint = [&](const Eigen::Vector3d& point)
  {
    if (!point.allFinite())
      return;
    pcl::PointXYZ pt;
    pt.x = point(0);
    pt.y = point(1);
    pt.z = point(2);
    P.strict_dense_inliers->points.push_back(pt);
  };

  for (const Eigen::Vector3d& inlier:P.inliers)
    pushStrictPoint(inlier);

  if (P.strict_visibility_graph.rows() == P.strict_visibility_graph.cols() &&
      P.strict_visibility_graph.rows() <= (int)P.inliers.size())
  {
    for (int i=0; i<P.strict_visibility_graph.rows(); ++i)
    {
      for (int j=0; j<i; ++j)
      {
        if (P.strict_visibility_graph(i,j) != 1)
          continue;

        const Eigen::Vector3d p1 = P.inliers[i];
        const Eigen::Vector3d p2 = P.inliers[j];
        const double dist = (p2-p1).norm();
        if (!std::isfinite(dist) || dist <= strict_delta)
          continue;

        const Eigen::Vector3d dir = (p2-p1) / dist;
        const int num = floor(dist / strict_delta);
        for (int k=1; k<num; ++k)
          pushStrictPoint(p1 + dir * k * strict_delta);
      }
    }
  }

  P.strict_dense_inliers->width = P.strict_dense_inliers->points.size();
  P.strict_dense_inliers->height = 1;
  P.strict_dense_inliers->is_dense = true;
  *P.dense_inliers = *P.strict_dense_inliers;

  if (P.dense_inliers->points.empty())
    return;

  pcl::KdTreeFLANN<pcl::PointXYZ> strict_tree;
  strict_tree.setInputCloud(P.dense_inliers);
  vector<int> nearest(1);
  vector<float> nn_squared_distance(1);
  for (int i=0; i<(int)P.ori_pts_->points.size(); ++i)
  {
    nearest.clear();
    nn_squared_distance.clear();
    if (strict_tree.nearestKSearch(P.ori_pts_->points[i], 1, nearest, nn_squared_distance) <= 0 || nearest.empty())
      continue;

    const pcl::PointXYZ& cor = P.dense_inliers->points[nearest[0]];
    P.pt_inlier_pairs[i] = Eigen::Vector3d(cor.x, cor.y, cor.z);
  }
}

void sk_decomp::fill_subspace_map(const vector<vector<int>>& subspace_sets, unordered_map<int, int>& node_subspace_map) const
{
  node_subspace_map.clear();
  for (int i=0; i<(int)subspace_sets.size(); ++i)
  {
    for (int idx:subspace_sets[i])
      node_subspace_map[idx] = i;
  }
}

vector<vector<int>> sk_decomp::visibility_subspace_decomp(const Eigen::MatrixXi& graph, unordered_map<int, int>* node_subspace_map)
{
  auto cd_t1 = chrono::high_resolution_clock::now();

  int numNodes = graph.rows();
  vector<vector<int>> subspaceSets;
  if (graph.cols() != numNodes || numNodes <= 0)
  {
    if (node_subspace_map != nullptr)
      node_subspace_map->clear();
    return subspaceSets;
  }

  // calculate degree
  vector<int> degrees(numNodes, 0);
  for (int i = 0; i < numNodes; ++i)
    for (int j = 0; j < numNodes; ++j)
      if (graph(i, j) == 1)
        degrees[i]++;

  // sort nodes
  vector<int> nodeOrder(numNodes);
  iota(nodeOrder.begin(), nodeOrder.end(), 0);
  sort(nodeOrder.begin(), nodeOrder.end(), [&](int a, int b) {
    return degrees[a] > degrees[b];
  });

  for (int node : nodeOrder)
  {
    bool assigned = false;
    for (auto& subspaceSet : subspaceSets)
    {
      if (canJoinVisibilitySubspace(graph, subspaceSet, node))
      {
        subspaceSet.push_back(node);
        assigned = true;
        break;
      }
    }

    if (!assigned)
    {
      vector<int> newSubspaceSet{node};
      subspaceSets.push_back(newSubspaceSet);
    }
  }

  vector<vector<int>> filteredSubspaceSets;
  filteredSubspaceSets = subspaceSets;

  if (node_subspace_map != nullptr)
    fill_subspace_map(filteredSubspaceSets, *node_subspace_map);

  auto cd_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> cd_ms = cd_t2 - cd_t1;
  double cd_time = (double)cd_ms.count();
  ROS_INFO("\033[32m[SSD] visibility graph subspace decomp time = %lf ms.\033[32m", cd_time);

  return filteredSubspaceSets;
}

void sk_decomp::merge_fallback_subspaces()
{
  auto cst_t1 = chrono::high_resolution_clock::now();
  const vector<Eigen::Vector3d>& decomp_points = active_decomp_points();
  const int merge_before = (int)P.subspace_sets.size();

  double sim_threshold = 15.0;
  // data preparation
  P.subspace_centers.clear();
  P.subspace_centers.resize(P.subspace_sets.size(), Eigen::Vector3d::Zero());
  P.subspace_maindir.clear();
  P.subspace_merge_sets.clear();
  P.subspace_states.clear();
  P.hl_vis_neighs.clear();
  for (int i=0; i<(int)P.subspace_sets.size(); ++i)
  {
    P.subspace_states[i] = true;
    bool valid_subspace_set = !P.subspace_sets[i].empty();
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    for (int j=0; j<(int)P.subspace_sets[i].size(); ++j)
    {
      if (P.subspace_sets[i][j] < 0 || P.subspace_sets[i][j] >= (int)decomp_points.size())
      {
        valid_subspace_set = false;
        break;
      }
      center += decomp_points[P.subspace_sets[i][j]];
    }
    if (!valid_subspace_set)
    {
      P.subspace_states[i] = false;
      continue;
    }
    center /= P.subspace_sets[i].size();
    P.subspace_centers[i] = center;

    P.subspace_merge_sets[i] = P.subspace_sets[i];
    if ((int)P.subspace_sets[i].size() < 3)
    {
      if ((int)P.subspace_sets[i].size() == 1)
      {
        P.subspace_maindir[i] = Eigen::Vector3d::UnitX();
        continue;
      }
      if ((int)P.subspace_sets[i].size() == 2)
      {
        Eigen::Vector3d dir = (decomp_points[P.subspace_sets[i][1]]-decomp_points[P.subspace_sets[i][0]]).normalized();
        if (!dir.allFinite())
          dir = Eigen::Vector3d::UnitX();
        P.subspace_maindir[i] = dir;
      }
    }
    else
    {
      Eigen::MatrixXd A;
      A.resize(P.subspace_sets[i].size(),3);
      for (int j=0; j<(int)P.subspace_sets[i].size(); ++j)
        A.row(j) = decomp_points[P.subspace_sets[i][j]];
      Eigen::Vector3d maindir = PCA(A);
      if (!maindir.allFinite())
        maindir = Eigen::Vector3d::UnitX();
      P.subspace_maindir[i] = maindir;
    }
  }

  // process single-node fallback subspaces
  for (auto pair:P.subspace_merge_sets)
  {
    if (pair.second.size() == 1)
    {
      int inlier_idx = pair.second[0];
      vector<int> neighbors, neigh_subspace;
      for (int i=0; i<(int)P.visibility_graph.cols(); ++i)
        if (P.visibility_graph(inlier_idx,i) == 1 && inlier_idx != i)
          neighbors.push_back(i);
      if ((int)neighbors.size() == 0)
      {
        P.subspace_states[pair.first] = false;
        continue;
      }
      Eigen::Vector3d temp_dir = Eigen::Vector3d::Zero();
      for (auto x:neighbors)
      {
        auto subspace_it = P.node_subspace_map.find(x);
        if (subspace_it == P.node_subspace_map.end())
          continue;
        neigh_subspace.push_back(subspace_it->second);
        Eigen::Vector3d dir = (decomp_points[x]-decomp_points[inlier_idx]).normalized();
        if (dir.allFinite())
          temp_dir += dir;
      }
      if (neigh_subspace.empty() || temp_dir.norm() < 1e-6)
      {
        P.subspace_states[pair.first] = false;
        continue;
      }
      temp_dir.normalize();
      int mostFrequent = findMostFrequentElement(neigh_subspace);

      auto maindir_it = P.subspace_maindir.find(mostFrequent);
      if (maindir_it == P.subspace_maindir.end() || !maindir_it->second.allFinite())
      {
        P.subspace_states[pair.first] = false;
        continue;
      }
      Eigen::Vector3d before_maindir = maindir_it->second;
      double angle = 180.0*acos(clampUnit(temp_dir.dot(before_maindir)))/M_PI;
      double similarity = min(angle, 180.0-angle);
      if (similarity < sim_threshold)
        P.subspace_merge_sets[mostFrequent].push_back(inlier_idx);

      P.subspace_states[pair.first] = false;
    }
  }

  // construct high-level visibility graph
  P.hl_vis_graph.resize(P.subspace_merge_sets.size(), P.subspace_merge_sets.size());
  P.hl_vis_graph.setZero();
  for (int i=0; i<(int)P.subspace_merge_sets.size(); ++i)
  {
    if (P.subspace_states[i] == false) continue;
    for (int j=0; j<(int)P.subspace_merge_sets.size(); ++j)
    {
      if (i <= j) continue;
      if (P.subspace_states[j] == false) continue;
      int i_idx = -1, j_idx = -1;
      bool con_flag = gen_hlvis_edge(P.subspace_merge_sets[i], P.subspace_merge_sets[j], i_idx, j_idx);
      if (con_flag == true)
      {
        P.hl_vis_graph(i,j) = 1;
        P.hl_vis_graph(j,i) = 1;
        P.hl_vis_neighs[{i,j}] = {i_idx, j_idx};
        P.hl_vis_neighs[{j,i}] = {j_idx, i_idx};
      }
    }
  }

  // main direction similarity graph
  P.dir_sim_graph.resize(P.subspace_merge_sets.size(), P.subspace_merge_sets.size());
  P.dir_sim_graph.setOnes();
  P.dir_sim_graph = P.dir_sim_graph*10000.0;
  for (int i=0; i<(int)P.subspace_merge_sets.size(); ++i)
  {
    if (P.subspace_states[i] == false) continue;
    for (int j=0; j<(int)P.subspace_merge_sets.size(); ++j)
    {
      if (i <= j) continue;
      if (P.subspace_states[j] == false) continue;
      if (P.hl_vis_graph(i,j) == 0) continue;

      if (P.subspace_maindir.find(i) == P.subspace_maindir.end() ||
          P.subspace_maindir.find(j) == P.subspace_maindir.end())
        continue;
      double angle = 180.0*acos(clampUnit(P.subspace_maindir[i].dot(P.subspace_maindir[j])))/M_PI;
      double similarity = min(angle, 180.0-angle);
      P.dir_sim_graph(i,j) = similarity;
      P.dir_sim_graph(j,i) = similarity;
    }
  }
  // convert P.dir_sim_graph to mask: element = 1 if similarity < sim_threshold
  Eigen::MatrixXi sim_mask = Eigen::MatrixXi::Zero(P.subspace_merge_sets.size(), P.subspace_merge_sets.size());
  for (int i=0; i<(int)P.subspace_merge_sets.size(); ++i)
    for (int j=0; j<(int)P.subspace_merge_sets.size(); ++j)
      if (P.dir_sim_graph(i,j) < sim_threshold)
        sim_mask(i,j) = 1;

  // center maindir consistency graph
  P.center_dir_cst_graph.resize(P.subspace_merge_sets.size(), P.subspace_merge_sets.size());
  P.center_dir_cst_graph.setOnes();
  P.center_dir_cst_graph = P.center_dir_cst_graph*10000.0;
  for (int i=0; i<(int)P.subspace_merge_sets.size(); ++i)
  {
    if (P.subspace_states[i] == false) continue;
    for (int j=0; j<(int)P.subspace_merge_sets.size(); ++j)
    {
      if (i <= j) continue;
      if (P.subspace_states[j] == false) continue;
      if (P.hl_vis_graph(i,j) == 0) continue;

      Eigen::Vector3d center_dir = (P.subspace_centers[j]-P.subspace_centers[i]).normalized();
      if (!center_dir.allFinite() ||
          P.subspace_maindir.find(i) == P.subspace_maindir.end() ||
          P.subspace_maindir.find(j) == P.subspace_maindir.end())
        continue;

      double angle_i = 180.0*acos(clampUnit(P.subspace_maindir[i].dot(center_dir)))/M_PI;
      double angle_j = 180.0*acos(clampUnit(P.subspace_maindir[j].dot(center_dir)))/M_PI;
      double similarity_i = min(angle_i, 180.0-angle_i);
      double similarity_j = min(angle_j, 180.0-angle_j);
      P.center_dir_cst_graph(i,j) = (similarity_i+similarity_j)/2.0;
      P.center_dir_cst_graph(j,i) = (similarity_i+similarity_j)/2.0;
    }
  }
  // convert P.center_dir_cst_graph to mask: element = 1 if similarity < sim_threshold
  Eigen::MatrixXi cst_mask = Eigen::MatrixXi::Zero(P.subspace_merge_sets.size(), P.subspace_merge_sets.size());
  for (int i=0; i<(int)P.subspace_merge_sets.size(); ++i)
    for (int j=0; j<(int)P.subspace_merge_sets.size(); ++j)
      if (P.center_dir_cst_graph(i,j) < sim_threshold)
        cst_mask(i,j) = 1;

  // local connected dir consistency graph
  P.local_dir_cst_graph.resize(P.subspace_merge_sets.size(), P.subspace_merge_sets.size());
  P.local_dir_cst_graph.setOnes();
  P.local_dir_cst_graph = P.local_dir_cst_graph*10000.0;
  for (int i=0; i<(int)P.subspace_merge_sets.size(); ++i)
  {
    if (P.subspace_states[i] == false) continue;
    for (int j=0; j<(int)P.subspace_merge_sets.size(); ++j)
    {
      if (i <= j) continue;
      if (P.subspace_states[j] == false) continue;
      if (P.hl_vis_graph(i,j) == 0) continue;

      vector<int> neighs = P.hl_vis_neighs[{i,j}];
      if (neighs.size() < 2 ||
          neighs[0] < 0 || neighs[1] < 0 ||
          neighs[0] >= (int)P.subspace_merge_sets[i].size() ||
          neighs[1] >= (int)P.subspace_merge_sets[j].size())
        continue;
      const int node_i = P.subspace_merge_sets[i][neighs[0]];
      const int node_j = P.subspace_merge_sets[j][neighs[1]];
      if (node_i < 0 || node_j < 0 ||
          node_i >= (int)decomp_points.size() ||
          node_j >= (int)decomp_points.size())
        continue;
      Eigen::Vector3d local_dir = (decomp_points[node_i]-decomp_points[node_j]).normalized();
      if (!local_dir.allFinite() ||
          P.subspace_maindir.find(i) == P.subspace_maindir.end() ||
          P.subspace_maindir.find(j) == P.subspace_maindir.end())
        continue;

      double angle_i = 180.0*acos(clampUnit(P.subspace_maindir[i].dot(local_dir)))/M_PI;
      double angle_j = 180.0*acos(clampUnit(P.subspace_maindir[j].dot(local_dir)))/M_PI;
      double similarity_i = min(angle_i, 180.0-angle_i);
      double similarity_j = min(angle_j, 180.0-angle_j);
      P.local_dir_cst_graph(i,j) = max(similarity_i, similarity_j);
      P.local_dir_cst_graph(j,i) = max(similarity_i, similarity_j);
    }
  }
  // convert P.local_dir_cst_graph to mask: element = 1 if similarity < sim_threshold
  Eigen::MatrixXi local_cst_mask = Eigen::MatrixXi::Zero(P.subspace_merge_sets.size(), P.subspace_merge_sets.size());
  for (int i=0; i<(int)P.subspace_merge_sets.size(); ++i)
    for (int j=0; j<(int)P.subspace_merge_sets.size(); ++j)
      if (P.local_dir_cst_graph(i,j) < sim_threshold)
        local_cst_mask(i,j) = 1;

  // merge graph: element-wise multiplication of P.hl_vis_graph, sim_mask, and cst_mask
  Eigen::MatrixXi merge_graph = P.hl_vis_graph.cwiseProduct(sim_mask).cwiseProduct(cst_mask).cwiseProduct(local_cst_mask);

  for (int i=0; i<(int)P.subspace_merge_sets.size(); ++i)
  {
    if (P.subspace_states[i] == true)
      merge_graph(i,i) = 1;
  }

  // DFS cluster
  dfs_cluster ds;
  vector<vector<int>> components = ds.find_directly_connected_components(merge_graph);
  for(const auto& component : components)
  {
    if (component.size() > 1)
    {
      for (int i=1; i<(int)component.size(); ++i)
      {
        P.subspace_merge_sets[component[0]].insert(P.subspace_merge_sets[component[0]].end(), P.subspace_merge_sets[component[i]].begin(), P.subspace_merge_sets[component[i]].end());
        P.subspace_states[component[i]] = false;
      }
    }
  }

  // update fallback subspace sets
  vector<vector<int>> new_subspace_sets;
  for (auto subspace_state:P.subspace_states)
  {
    if (subspace_state.second == true)
      new_subspace_sets.push_back(P.subspace_merge_sets[subspace_state.first]);
  }

  P.subspace_sets = new_subspace_sets;
  fill_subspace_map(P.subspace_sets, P.node_subspace_map);

  auto cst_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> cst_ms = cst_t2 - cst_t1;
  double cst_time = (double)cst_ms.count();
  ROS_INFO("\033[32m[SSD] fallback subspace merge: before=%d, after=%zu, time=%lf ms.\033[32m",
           merge_before, P.subspace_sets.size(), cst_time);
}

void sk_decomp::build_subspace_skeleton_paths()
{
  auto csk_t1 = chrono::high_resolution_clock::now();
  P.subspace_skeleton_paths.clear();
  const vector<vector<int>>& subspace_sets = active_subspace_sets();
  P.subspace_skeleton_paths.reserve(subspace_sets.size());

  int valid_path_num = 0;
  for (const vector<int>& subspace:subspace_sets)
  {
    vector<int> subspace_set = subspace;
    vector<vector<Eigen::Vector3d>> paths;
    subspace_representative(subspace_set, paths);
    for (const auto& path:paths)
    {
      if ((int)path.size() < 2)
        continue;
      valid_path_num++;
      P.subspace_skeleton_paths.push_back(path);
    }
  }

  auto csk_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> csk_ms = csk_t2 - csk_t1;
  ROS_INFO("\033[32m[SSD] subspace skeleton paths: subspaces=%zu, paths=%zu, valid=%d, source=%s, time=%lf ms.\033[32m",
           subspace_sets.size(),
           P.subspace_skeleton_paths.size(),
           valid_path_num,
           dcrosa_branch_subspace_ready_ ? "dcrosa_branch" : "visibility_subspace",
           (double)csk_ms.count());
}

void sk_decomp::distribute_ori_cloud()
{
  auto doc_t1 = chrono::high_resolution_clock::now();
  const vector<Eigen::Vector3d>& decomp_points = active_decomp_points();
  const vector<vector<int>>& subspace_sets = active_subspace_sets();

  // data preparation
  double norm_delta = P.scale*pt_downsample_voxel_size>edge_min_dist? P.scale*pt_downsample_voxel_size:edge_min_dist;
  if (!std::isfinite(norm_delta) || norm_delta <= 1e-6)
    norm_delta = std::max(1e-3, P.scale * pt_downsample_voxel_size);

  P.inlier_sub.clear();
  P.tg_pt_idx.clear();
  P.pt_inlier_pairs.clear();
  P.sub_space_scale.clear();
  P.sub_space_ptnr.clear();
  P.sub_space_scale.reserve(subspace_sets.size());
  P.sub_space_ptnr.reserve(subspace_sets.size());
  P.dense_inliers.reset(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr target (new pcl::PointCloud<pcl::PointXYZ>);
  int pt_idx = 0;

  pcl::PointXYZ pt;
  for (int i=0; i<(int)subspace_sets.size(); ++i)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr zerocloud (new pcl::PointCloud<pcl::PointXYZ>);
    P.sub_space_scale.push_back(zerocloud);
    pcl::PointCloud<pcl::PointNormal>::Ptr zero_ptnr (new pcl::PointCloud<pcl::PointNormal>);
    P.sub_space_ptnr.push_back(zero_ptnr);

    for (int j=0; j<(int)subspace_sets[i].size(); ++j)
    {
      const int j_node = subspace_sets[i][j];
      if (j_node < 0 || j_node >= (int)decomp_points.size())
        continue;
      for (int k=0; k<(int)subspace_sets[i].size(); ++k)
      {
        const int k_node = subspace_sets[i][k];
        if (k_node < 0 || k_node >= (int)decomp_points.size())
          continue;
        if (j == k)
        {
          pt.x = decomp_points[j_node](0);
          pt.y = decomp_points[j_node](1);
          pt.z = decomp_points[j_node](2);
          target->points.push_back(pt);
          P.inlier_sub[pt_idx] = i;
          P.tg_pt_idx[pt] = pt_idx;
          pt_idx++;
        }
        if (j > k)
        {
          Eigen::Vector3d p1 = decomp_points[j_node];
          Eigen::Vector3d p2 = decomp_points[k_node];
          double dist = (p2-p1).norm();
          if (!std::isfinite(dist) || dist <= 1e-6)
            continue;
          Eigen::Vector3d dir = (p2-p1) / dist;
          int num = floor(dist/norm_delta);
          for (int l=1; l<num; ++l)
          {
            pt.x = p1(0)+dir(0)*l*norm_delta;
            pt.y = p1(1)+dir(1)*l*norm_delta;
            pt.z = p1(2)+dir(2)*l*norm_delta;
            target->points.push_back(pt);
            P.inlier_sub[pt_idx] = i;
            P.tg_pt_idx[pt] = pt_idx;
            pt_idx++;
          }
        }
      }
    }
  }

  // select seeds
  int upper_bound = (int)(0.2*calNum);
  if ((int)target->points.size() > upper_bound)
  {
    pcl::RandomSample<pcl::PointXYZ> rs;
    rs.setInputCloud(target);
    rs.setSample(upper_bound);
    rs.filter(*target);

    unordered_map<int, int> temp_inlier_sub;
    int temp_idx = 0;
    pcl::PointXYZ pt;
    for (int i=0; i<(int)target->points.size(); ++i)
    {
      pt = target->points[i];
      if (P.tg_pt_idx.find(pt) != P.tg_pt_idx.end())
      {
        temp_inlier_sub[temp_idx] = P.inlier_sub[P.tg_pt_idx[pt]];
        temp_idx++;
      }
    }

    P.inlier_sub.clear();
    P.inlier_sub = temp_inlier_sub;
  }

  // kdtree search: use active branch-subspace nodes only for subspace partition.
  if (!target->points.empty() && P.ori_pts_ != nullptr && P.ori_nrs_ != nullptr)
  {
    pcl::KdTreeFLANN<pcl::PointXYZ> distri_tree;
    distri_tree.setInputCloud(target);
    vector<int> nearest(1);
    vector<float> nn_squared_distance(1);

    const int ori_num = std::min((int)P.ori_pts_->points.size(), (int)P.ori_nrs_->points.size());
    for (int i=0; i<ori_num; ++i)
    {
      pcl::PointXYZ ori_pt = P.ori_pts_->points[i];
      pcl::Normal ori_nr = P.ori_nrs_->points[i];
      pcl::PointNormal ptnr;
      ptnr.x = ori_pt.x; ptnr.y = ori_pt.y; ptnr.z = ori_pt.z;
      ptnr.normal_x = ori_nr.normal_x; ptnr.normal_y = ori_nr.normal_y; ptnr.normal_z = ori_nr.normal_z;
      nearest.clear();
      nn_squared_distance.clear();
      if (distri_tree.nearestKSearch(ori_pt, 1, nearest, nn_squared_distance) <= 0 || nearest.empty())
        continue;

      auto sub_it = P.inlier_sub.find(nearest[0]);
      if (sub_it == P.inlier_sub.end() || sub_it->second < 0 || sub_it->second >= (int)P.sub_space_scale.size())
        continue;

      int sub_idx = sub_it->second;
      P.sub_space_scale[sub_idx]->points.push_back(ori_pt);
      P.sub_space_ptnr[sub_idx]->points.push_back(ptnr);
    }
  }

  // Keep downstream visibility/internal safety strict: relaxed branch points guide
  // subspace partition only, while inlier pairs stay inside-mesh strict points.
  build_strict_internal_inliers();

  // print size
  // for (int i=0; i<(int)P.sub_space_scale.size(); ++i)
  //   cout << "sub space " << i << " size: " << P.sub_space_scale[i]->points.size() << endl;

  auto doc_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> doc_ms = doc_t2 - doc_t1;
  double doc_time = (double)doc_ms.count();
  ROS_INFO("\033[32m[SSD] distribute original cloud time = %lf ms.\033[32m", doc_time);
}

void sk_decomp::high_level_vis_graph()
{
  auto hv_t1 = chrono::high_resolution_clock::now();
  const vector<Eigen::Vector3d>& decomp_points = active_decomp_points();
  const vector<vector<int>>& subspace_sets = active_subspace_sets();
  // initialize high-level visibility graph
  P.hl_vis_sets.clear();
  P.hl_vis_neighs.clear();
  P.hl_vis_sets.reserve(subspace_sets.size());
  P.hl_vis_graph.resize(subspace_sets.size(), subspace_sets.size());
  P.hl_vis_graph.setZero();

  for (int i=0; i<(int)subspace_sets.size(); ++i)
  {
    hlvis_graph_node h_node;
    h_node.subspace_idx = i;
    h_node.base_nodes = subspace_sets[i];
    // calculate the center of base_nodes
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    int valid_num = 0;
    for (int j=0; j<(int)subspace_sets[i].size(); ++j)
    {
      const int node_idx = subspace_sets[i][j];
      if (node_idx < 0 || node_idx >= (int)decomp_points.size())
        continue;
      center += decomp_points[node_idx];
      valid_num++;
    }
    if (valid_num > 0)
      center /= valid_num;
    h_node.subspace_center = center;

    P.hl_vis_sets.push_back(h_node);
  }
  // construct hlvis_edge
  for (int i=0; i<(int)P.hl_vis_sets.size(); ++i)
    for (int j=0; j<(int)P.hl_vis_sets.size(); ++j)
    {
      if (i <= j) continue;
      int i_idx = -1, j_idx = -1;
      bool con_flag = gen_hlvis_edge(P.hl_vis_sets[i].base_nodes, P.hl_vis_sets[j].base_nodes, i_idx, j_idx);
      if (con_flag == true)
      {
        P.hl_vis_graph(i,j) = 1;
        P.hl_vis_graph(j,i) = 1;
        P.hl_vis_sets[i].connected_subspaces.push_back(j);
        P.hl_vis_sets[j].connected_subspaces.push_back(i);
        P.hl_vis_sets[i].boundary_nodes.push_back(i_idx);
        P.hl_vis_sets[j].boundary_nodes.push_back(j_idx);
        Eigen::Vector2i ij_edge(i_idx, j_idx);
        Eigen::Vector2i ji_edge(j_idx, i_idx);
        P.hl_vis_sets[i].edges[j] = ij_edge;
        P.hl_vis_sets[j].edges[i] = ji_edge;
      }
    }
  // intra paths
  for (int i=0; i<(int)P.hl_vis_graph.rows(); ++i)
    for (int j=0; j<(int)P.hl_vis_graph.cols(); ++j)
    {
      if (i <= j) continue;
      if (P.hl_vis_graph(i,j) == 1)
      {
        vector<Eigen::Vector3d> path;
        Eigen::Vector2i edge_idx = P.hl_vis_sets[i].edges[j];
        int i_idx = edge_idx(0);
        int j_idx = edge_idx(1);

        if (i_idx < 0 || j_idx < 0) continue;
        if (i_idx >= (int)P.hl_vis_sets[i].base_nodes.size()) continue;
        if (j_idx >= (int)P.hl_vis_sets[j].base_nodes.size()) continue;
        const int i_node = P.hl_vis_sets[i].base_nodes[i_idx];
        const int j_node = P.hl_vis_sets[j].base_nodes[j_idx];
        if (i_node < 0 || j_node < 0) continue;
        if (i_node >= (int)decomp_points.size() || j_node >= (int)decomp_points.size()) continue;

        Eigen::Vector3d p1 = decomp_points[i_node];
        Eigen::Vector3d p2 = decomp_points[j_node];
        path = {p1, p2};
      }
    }

  auto hv_t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> hv_ms = hv_t2 - hv_t1;
  double hv_time = (double)hv_ms.count();
  ROS_INFO("\033[32m[SSD] high-level graph construct time = %lf ms.\033[32m", hv_time);
}

// ! /* --------------------- Utils --------------------- */

void sk_decomp::inliers_initialize(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::Normal>::Ptr& normals)
{
  int psize = (int)cloud->points.size();
  // initialize positions
  for (int i=0; i<psize; ++i)
  {
    pset(i,0) = cloud->points[i].x;
    pset(i,1) = cloud->points[i].y;
    pset(i,2) = cloud->points[i].z;
  }
  // initialize orientations
  Eigen::Matrix3d M;
  Eigen::Vector3d normal_v;
  for (int j=0; j<psize; ++j)
  {
    normal_v(0) = normals->points[j].normal_x;
    normal_v(1) = normals->points[j].normal_y;
    normal_v(2) = normals->points[j].normal_z;
    M = create_orthonormal_frame(normal_v);
    vset.row(j) = M.row(1);
  }
}

Eigen::Matrix3d sk_decomp::create_orthonormal_frame(Eigen::Vector3d& v)
{
  // ! /* random process for generating orthonormal basis */
  v = v/v.norm();
  double TH_ZERO = 1e-10;
  srand((unsigned)time(NULL));
  Eigen::Matrix3d M=Eigen::Matrix3d::Zero();
  M(0,0) = v(0);
  M(0,1) = v(1);
  M(0,2) = v(2);
  Eigen::Vector3d new_vec, temp_vec;
  for (int i=1; i<3; ++i)
  {
    new_vec.setRandom();
    new_vec = new_vec/new_vec.norm();
    while (abs(1.0-v.dot(new_vec)) < TH_ZERO)
    {
    new_vec.setRandom();
    new_vec = new_vec/new_vec.norm();
    }
    for (int j=0; j<i; ++j)
    {
    temp_vec = (new_vec - new_vec.dot(M.row(j))*(M.row(j).transpose()));
    new_vec = temp_vec/temp_vec.norm();
    }
    M(i,0) = new_vec(0); M(i,1) = new_vec(1); M(i,2) = new_vec(2);
  }

  return M;
}

Eigen::MatrixXd sk_decomp::rosa_compute_active_samples(int& idx, Eigen::Vector3d& p_cut, Eigen::Vector3d& v_cut)
{
  Eigen::MatrixXd out_indxs(pcd_size_, 1);
  vector<int> isoncut(pcd_size_, 0);
  pcloud_isoncut(p_cut, v_cut, isoncut, P.datas, pcd_size_);

  Eigen::MatrixXd clusterIdxs;
  vector<int> inplane;
  for (int i=0; i<pcd_size_; ++i)
  {
    if (isoncut[i] == 1)
      inplane.push_back(i);
  }

  clusterIdxs.resize(inplane.size(),1);
  for (int i=0; i<(int)clusterIdxs.size(); ++i)
    clusterIdxs(i,0) = inplane[i];
  Eigen::MatrixXd cluster_idxs = adp_utils_->cluster_proc(idx, clusterIdxs, P.pts_mat, epsilon);
  out_indxs = cluster_idxs;

  return out_indxs;
}

void sk_decomp::pcloud_isoncut(Eigen::Vector3d& p_cut, Eigen::Vector3d& v_cut, vector<int>& isoncut, double*& datas, int& size)
{
  DataWrapper data;
  data.factory(datas, size);
  vector<double> p(3); p[0] = p_cut(0); p[1] = p_cut(1); p[2] = p_cut(2);
  vector<double> n(3); n[0] = v_cut(0); n[1] = v_cut(1); n[2] = v_cut(2);
  distance_query(data, p, n, delta, isoncut);
}

void sk_decomp::distance_query(DataWrapper& data, const vector<double>& Pp, const vector<double>& Np, double delta, vector<int>& isoncut)
{
  vector<double> P(3);
  for (int pIdx=0; pIdx < data.length(); pIdx++)
  {
    data(pIdx, P);
    if(fabs( Np[0]*(Pp[0]-P[0]) + Np[1]*(Pp[1]-P[1]) + Np[2]*(Pp[2]-P[2]) ) < delta)
      isoncut[pIdx] = 1;
  }
}

Eigen::Vector3d sk_decomp::compute_symmetrynormal(Eigen::MatrixXd& local_normals)
{
  Eigen::Matrix3d M;
  Eigen::Vector3d vec;
  double alpha = 0.0;
  int size = local_normals.rows();
  double Vxx, Vyy, Vzz, Vxy, Vyx, Vxz, Vzx, Vyz, Vzy;
  Vxx = (1.0+alpha)*local_normals.col(0).cwiseAbs2().sum()/size - pow(local_normals.col(0).sum(), 2)/pow(size, 2);
  Vyy = (1.0+alpha)*local_normals.col(1).cwiseAbs2().sum()/size - pow(local_normals.col(1).sum(), 2)/pow(size, 2);
  Vzz = (1.0+alpha)*local_normals.col(2).cwiseAbs2().sum()/size - pow(local_normals.col(2).sum(), 2)/pow(size, 2);
  Vxy = 2*(1.0+alpha)*(local_normals.col(0).cwiseProduct(local_normals.col(1))).sum()/size - 2*local_normals.col(0).sum()*local_normals.col(1).sum()/pow(size, 2);
  Vyx = Vxy;
  Vxz = 2*(1.0+alpha)*(local_normals.col(0).cwiseProduct(local_normals.col(2))).sum()/size - 2*local_normals.col(0).sum()*local_normals.col(2).sum()/pow(size, 2);
  Vzx = Vxz;
  Vyz = 2*(1.0+alpha)*(local_normals.col(1).cwiseProduct(local_normals.col(2))).sum()/size - 2*local_normals.col(1).sum()*local_normals.col(2).sum()/pow(size, 2);
  Vzy = Vyz;
  M << Vxx, Vxy, Vxz, Vyx, Vyy, Vyz, Vzx, Vzy, Vzz;

  BDCSVD<MatrixXd> svd(M, ComputeThinU | ComputeThinV);
  Eigen::Matrix3d U = svd.matrixU();
  vec = U.col(M.cols()-1);

  return vec;
}

double sk_decomp::symmnormal_variance(Eigen::Vector3d& symm_nor, Eigen::MatrixXd& local_normals)
{
  Eigen::MatrixXd repmat;
  Eigen::VectorXd alpha;
  int num = local_normals.rows();
  repmat.resize(num,3);
  for (int i=0; i<num; ++i)
    repmat.row(i) = symm_nor;
  alpha = local_normals.cwiseProduct(repmat).rowwise().sum();
  int n = alpha.size();
  double var;
  if (n>1)
    var = (n+1)*(alpha.squaredNorm()/(n+1) - alpha.mean()*alpha.mean())/n;
  else
    var = alpha.squaredNorm()/(n+1) - alpha.mean()*alpha.mean();

  return var;
}

Eigen::Vector3d sk_decomp::symmnormal_smooth(Eigen::MatrixXd& V, Eigen::MatrixXd& w)
{
  Eigen::Matrix3d M;
  Eigen::Vector3d vec;
  double Vxx, Vyy, Vzz, Vxy, Vyx, Vxz, Vzx, Vyz, Vzy;
  Vxx = (w.cwiseProduct(V.col(0).cwiseAbs2())).sum();
  Vyy = (w.cwiseProduct(V.col(1).cwiseAbs2())).sum();
  Vzz = (w.cwiseProduct(V.col(2).cwiseAbs2())).sum();
  Vxy = (w.cwiseProduct(V.col(0)).cwiseProduct(V.col(1))).sum();
  Vyx = Vxy;
  Vxz = (w.cwiseProduct(V.col(0)).cwiseProduct(V.col(2))).sum();
  Vzx = Vxz;
  Vyz = (w.cwiseProduct(V.col(1)).cwiseProduct(V.col(2))).sum();
  Vzy = Vyz;
  M << Vxx, Vxy, Vxz, Vyx, Vyy, Vyz, Vzx, Vzy, Vzz;

  BDCSVD<MatrixXd> svd(M, ComputeThinU | ComputeThinV);
  Eigen::Matrix3d U = svd.matrixU();
  vec = U.col(0);

  return vec;
}

Eigen::Vector3d sk_decomp::closest_projection_point(Eigen::MatrixXd& P, Eigen::MatrixXd& V)
{
  Eigen::Vector3d vec;
  Eigen::VectorXd Lix2, Liy2, Liz2;
  Lix2 = V.col(0).cwiseAbs2();
  Liy2 = V.col(1).cwiseAbs2();
  Liz2 = V.col(2).cwiseAbs2();

  Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
  Eigen::Vector3d B = Eigen::Vector3d::Zero();

  M(0,0) = (Liy2+Liz2).sum();
  M(0,1) = -(V.col(0).cwiseProduct(V.col(1))).sum();
  M(0,2) = -(V.col(0).cwiseProduct(V.col(2))).sum();
  B(0) = (P.col(0).cwiseProduct(Liy2 + Liz2)).sum() - (V.col(0).cwiseProduct(V.col(1)).cwiseProduct(P.col(1))).sum() - (V.col(0).cwiseProduct(V.col(2)).cwiseProduct(P.col(2))).sum();
  M(1,0) = -(V.col(1).cwiseProduct(V.col(0))).sum();
  M(1,1) = (Lix2 + Liz2).sum();
  M(1,2) = -(V.col(1).cwiseProduct(V.col(2))).sum();
  B(1) = (P.col(1).cwiseProduct(Lix2 + Liz2)).sum() - (V.col(1).cwiseProduct(V.col(0)).cwiseProduct(P.col(0))).sum() - (V.col(1).cwiseProduct(V.col(2)).cwiseProduct(P.col(2))).sum();
  M(2,0) = -(V.col(2).cwiseProduct(V.col(0))).sum();
  M(2,1) = -(V.col(2).cwiseProduct(V.col(1))).sum();
  M(2,2) = (Lix2 + Liy2).sum();
  B(2) = (P.col(2).cwiseProduct(Lix2 + Liy2)).sum() - (V.col(2).cwiseProduct(V.col(0)).cwiseProduct(P.col(0))).sum() - (V.col(2).cwiseProduct(V.col(1)).cwiseProduct(P.col(1))).sum();

  if (abs(M.determinant()) < 1e-3)
    vec << 1e8, 1e8, 1e8;
  else
    vec = M.inverse()*B;

  return vec;
}

bool sk_decomp::canJoinVisibilitySubspace(const Eigen::MatrixXi& graph, const vector<int>& subspace_set, int node)
{
  for (int setNode : subspace_set)
  {
    if (graph(setNode, node) == 0) return false;
  }

  return true;
}

bool sk_decomp::gen_hlvis_edge(vector<int>& subspace_set_1, vector<int>& subspace_set_2, int& idx_1, int& idx_2)
{
  bool connected = false;
  const vector<Eigen::Vector3d>& decomp_points = active_decomp_points();

  double min_dist = 1e8;
  for (int i=0; i<(int)subspace_set_1.size(); ++i)
  {
    int id_a = subspace_set_1[i];
    if (id_a < 0 || id_a >= (int)decomp_points.size() || id_a >= P.visibility_graph.rows())
      continue;
    for (int j=0; j<(int)subspace_set_2.size(); ++j)
    {
      int id_b = subspace_set_2[j];
      if (id_b < 0 || id_b >= (int)decomp_points.size() || id_b >= P.visibility_graph.cols())
        continue;
      Eigen::Vector3d p1 = decomp_points[id_a];
      Eigen::Vector3d p2 = decomp_points[id_b];
      if (P.visibility_graph(id_a, id_b) == 1)
      {
        double dist = (p1-p2).norm();
        if (dist < min_dist)
        {
          min_dist = dist;
          idx_1 = i;
          idx_2 = j;
          connected = true;
        }
      }
    }
  }

  return connected;
}

void sk_decomp::subspace_representative(vector<int>& subspace_set, vector<Eigen::Vector3d>& path)
{
  path.clear();
  const vector<Eigen::Vector3d>& decomp_points = active_decomp_points();
  if ((int)subspace_set.size() < 3)
  {
    if ((int)subspace_set.size() == 1)
    {
      if (subspace_set[0] < 0 || subspace_set[0] >= (int)decomp_points.size())
        return;
      Eigen::Vector3d dir=Eigen::Vector3d::Random();
      dir.normalize();
      Eigen::Vector3d end=decomp_points[subspace_set[0]]+dir*1e-3;
      path = {decomp_points[subspace_set[0]], end};
    }
    else
    {
      if (subspace_set[0] < 0 || subspace_set[1] < 0 ||
          subspace_set[0] >= (int)decomp_points.size() ||
          subspace_set[1] >= (int)decomp_points.size())
        return;
      path = {decomp_points[subspace_set[0]], decomp_points[subspace_set[1]]};
    }
  }
  else
  {
    // data prepration
    vector<Eigen::Vector3d> pts;
    Eigen::MatrixXd A;
    A.resize(subspace_set.size(),3);
    for (int i=0; i<(int)subspace_set.size(); ++i)
    {
      if (subspace_set[i] < 0 || subspace_set[i] >= (int)decomp_points.size())
        return;
      pts.push_back(decomp_points[subspace_set[i]]);
      A.row(i) = decomp_points[subspace_set[i]];
    }
    // PCA
    Eigen::Vector3d vec;
    Eigen::Vector3d centroid = A.colwise().mean();
    Eigen::Matrix3d cov = (A.rowwise() - centroid.transpose()).transpose() * (A.rowwise() - centroid.transpose()) / double(A.rows() - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    vec = eig.eigenvectors().col(2);
    // find representative path
    double minProjection = std::numeric_limits<double>::infinity();
    double maxProjection = -std::numeric_limits<double>::infinity();
    Eigen::Vector3d minPoint, maxPoint;
    for (const Eigen::Vector3d& point : pts)
    {
      double projection = point.dot(vec);
      if (projection < minProjection)
      {
        minProjection = projection;
        minPoint = point;
      }
      if (projection > maxProjection)
      {
        maxProjection = projection;
        maxPoint = point;
      }
    }

    path = {minPoint, maxPoint};
  }
}

void sk_decomp::subspace_representative(vector<int>& subspace_set, vector<vector<Eigen::Vector3d>>& paths)
{
  paths.clear();
  vector<Eigen::Vector3d> fallback_path;
  subspace_representative(subspace_set, fallback_path);
  if ((int)subspace_set.size() < 4)
  {
    if ((int)fallback_path.size() >= 2)
      paths.push_back(fallback_path);
    return;
  }

  const vector<Eigen::Vector3d>& decomp_points = active_decomp_points();
  const int n = (int)subspace_set.size();
  Eigen::MatrixXd A(n, 3);
  for (int i=0; i<n; ++i)
  {
    if (subspace_set[i] < 0 || subspace_set[i] >= (int)decomp_points.size())
    {
      if ((int)fallback_path.size() >= 2)
        paths.push_back(fallback_path);
      return;
    }
    A.row(i) = decomp_points[subspace_set[i]];
  }

  Eigen::Vector3d centroid = A.colwise().mean();
  Eigen::Matrix3d cov = (A.rowwise() - centroid.transpose()).transpose() *
                        (A.rowwise() - centroid.transpose()) / double(A.rows() - 1);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
  if (eig.info() != Eigen::Success)
  {
    if ((int)fallback_path.size() >= 2)
      paths.push_back(fallback_path);
    return;
  }

  Eigen::Vector3d evals = eig.eigenvalues().cwiseMax(0.0);
  const double primary_eval = evals(2);
  const double secondary_ratio = primary_eval > 1e-9 ? evals(1) / primary_eval : 0.0;
  constexpr double kLineLikeSecondaryRatio = 0.15;
  if (secondary_ratio < kLineLikeSecondaryRatio)
  {
    if ((int)fallback_path.size() >= 2)
      paths.push_back(fallback_path);
    return;
  }

  const double fallback_len = (int)fallback_path.size() >= 2 ? (fallback_path.back() - fallback_path.front()).norm() : 0.0;
  const double min_branch_len = std::max(1e-3, 0.08 * fallback_len);
  const int k_max_neighbors = std::min(4, n - 1);

  struct CandidateEdge
  {
    int a;
    int b;
    double w;
  };

  vector<CandidateEdge> candidate_edges;
  for (int i=0; i<n; ++i)
  {
    vector<pair<double, int>> neighs;
    neighs.reserve(n - 1);
    const int node_i = subspace_set[i];
    for (int j=0; j<n; ++j)
    {
      if (i == j)
        continue;
      const int node_j = subspace_set[j];
      if (node_i < 0 || node_j < 0 ||
          node_i >= P.visibility_graph.rows() || node_j >= P.visibility_graph.cols())
        continue;
      if (P.visibility_graph(node_i, node_j) != 1)
        continue;
      const double dist = (decomp_points[node_i] - decomp_points[node_j]).norm();
      if (!std::isfinite(dist) || dist <= 1e-9)
        continue;
      neighs.push_back({dist, j});
    }
    sort(neighs.begin(), neighs.end(), [](const pair<double, int>& lhs, const pair<double, int>& rhs) {
      return lhs.first < rhs.first;
    });
    const int keep_num = std::min(k_max_neighbors, (int)neighs.size());
    for (int k=0; k<keep_num; ++k)
    {
      int a = i;
      int b = neighs[k].second;
      if (a > b)
        std::swap(a, b);
      candidate_edges.push_back({a, b, neighs[k].first});
    }
  }

  if (candidate_edges.empty())
  {
    if ((int)fallback_path.size() >= 2)
      paths.push_back(fallback_path);
    return;
  }

  sort(candidate_edges.begin(), candidate_edges.end(), [](const CandidateEdge& lhs, const CandidateEdge& rhs) {
    if (lhs.a != rhs.a)
      return lhs.a < rhs.a;
    if (lhs.b != rhs.b)
      return lhs.b < rhs.b;
    return lhs.w < rhs.w;
  });
  vector<CandidateEdge> unique_edges;
  for (const CandidateEdge& edge:candidate_edges)
  {
    if (edge.a == edge.b)
      continue;
    if (!unique_edges.empty() && unique_edges.back().a == edge.a && unique_edges.back().b == edge.b)
      continue;
    unique_edges.push_back(edge);
  }
  candidate_edges.swap(unique_edges);

  sort(candidate_edges.begin(), candidate_edges.end(), [](const CandidateEdge& lhs, const CandidateEdge& rhs) {
    return lhs.w < rhs.w;
  });

  vector<int> parent(n);
  iota(parent.begin(), parent.end(), 0);
  function<int(int)> findRoot = [&](int x) -> int {
    while (parent[x] != x)
    {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  auto unite = [&](int a, int b) -> bool {
    int ra = findRoot(a);
    int rb = findRoot(b);
    if (ra == rb)
      return false;
    parent[rb] = ra;
    return true;
  };

  vector<vector<int>> adj(n);
  int mst_edge_num = 0;
  for (const CandidateEdge& edge:candidate_edges)
  {
    if (!unite(edge.a, edge.b))
      continue;
    adj[edge.a].push_back(edge.b);
    adj[edge.b].push_back(edge.a);
    mst_edge_num++;
  }

  if (mst_edge_num <= 0)
  {
    if ((int)fallback_path.size() >= 2)
      paths.push_back(fallback_path);
    return;
  }

  vector<int> degree(n, 0);
  vector<int> key_nodes;
  key_nodes.reserve(n);
  for (int i=0; i<n; ++i)
  {
    degree[i] = (int)adj[i].size();
    if (degree[i] != 2 && degree[i] > 0)
      key_nodes.push_back(i);
  }

  vector<vector<bool>> visited(n);
  for (int i=0; i<n; ++i)
    visited[i].assign(adj[i].size(), false);

  auto markVisited = [&](int a, int b) {
    for (int k=0; k<(int)adj[a].size(); ++k)
    {
      if (adj[a][k] == b)
      {
        visited[a][k] = true;
        break;
      }
    }
  };

  auto edgeVisited = [&](int a, int b) -> bool {
    for (int k=0; k<(int)adj[a].size(); ++k)
      if (adj[a][k] == b)
        return visited[a][k];
    return true;
  };

  auto pathLength = [&](const vector<int>& local_path) -> double {
    double length = 0.0;
    for (int i=1; i<(int)local_path.size(); ++i)
      length += (decomp_points[subspace_set[local_path[i]]] - decomp_points[subspace_set[local_path[i-1]]]).norm();
    return length;
  };

  auto emitLocalPath = [&](const vector<int>& local_path) {
    if ((int)local_path.size() < 2)
      return;
    if (pathLength(local_path) < min_branch_len && n > 4)
      return;
    vector<Eigen::Vector3d> path;
    path.reserve(local_path.size());
    for (int local_idx:local_path)
      path.push_back(decomp_points[subspace_set[local_idx]]);
    paths.push_back(path);
  };

  for (int start:key_nodes)
  {
    for (int next:adj[start])
    {
      if (edgeVisited(start, next))
        continue;

      vector<int> local_path;
      local_path.push_back(start);
      int prev = start;
      int cur = next;
      markVisited(prev, cur);
      markVisited(cur, prev);
      local_path.push_back(cur);

      while (degree[cur] == 2)
      {
        int candidate = -1;
        for (int nb:adj[cur])
        {
          if (nb != prev)
          {
            candidate = nb;
            break;
          }
        }
        if (candidate < 0 || edgeVisited(cur, candidate))
          break;

        prev = cur;
        cur = candidate;
        markVisited(prev, cur);
        markVisited(cur, prev);
        local_path.push_back(cur);
      }

      emitLocalPath(local_path);
    }
  }

  for (int i=0; i<n; ++i)
  {
    for (int nb:adj[i])
    {
      if (edgeVisited(i, nb))
        continue;
      vector<int> local_path;
      local_path.push_back(i);
      int prev = i;
      int cur = nb;
      markVisited(prev, cur);
      markVisited(cur, prev);
      local_path.push_back(cur);

      while (cur != i && degree[cur] == 2)
      {
        int candidate = -1;
        for (int next:adj[cur])
        {
          if (next != prev)
          {
            candidate = next;
            break;
          }
        }
        if (candidate < 0 || edgeVisited(cur, candidate))
          break;

        prev = cur;
        cur = candidate;
        markVisited(prev, cur);
        markVisited(cur, prev);
        local_path.push_back(cur);
      }
      emitLocalPath(local_path);
    }
  }

  if (paths.empty() && (int)fallback_path.size() >= 2)
    paths.push_back(fallback_path);
}

Eigen::Vector3d sk_decomp::PCA(Eigen::MatrixXd& A)
{
  Eigen::Vector3d vec;
  Eigen::Vector3d centroid = A.colwise().mean();
  Eigen::Matrix3d cov = (A.rowwise() - centroid.transpose()).transpose() * (A.rowwise() - centroid.transpose()) / double(A.rows() - 1);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
  vec = eig.eigenvectors().col(2);

  return vec;
}

int sk_decomp::findMostFrequentElement(const vector<int>& vec)
{
  map<int, int> countMap;

  for (int num : vec) {
      countMap[num]++;
  }

  int maxCount = 0;
  int mostFrequentElement = vec[0];

  for (const auto& pair : countMap) {
      if (pair.second > maxCount) {
          maxCount = pair.second;
          mostFrequentElement = pair.first;
      }
  }

  return mostFrequentElement;
}

void sk_decomp::stopVisualization()
{
  visFlag.store(false);
  if (vis_timer_.isValid())
  {
    vis_timer_.stop();
  }
}

bool sk_decomp::visualizationDataReadyUnsafe() const
{
  if (!vis_data_ready_.load())
    return false;
  if (vis_utils_ == nullptr)
    return false;
  if (P.ori_pts_ == nullptr || P.selected_pts_ == nullptr || P.selected_normals_ == nullptr)
    return false;
  if ((int)P.selected_normals_->points.size() != (int)P.selected_pts_->points.size())
    return false;
  if (vset.rows() < (int)P.selected_normals_->points.size() || vset.cols() < 3)
    return false;
  if (pset.cols() != 0 && pset.cols() < 3)
    return false;
  if (pset_check.cols() != 0 && pset_check.cols() < 3)
    return false;
  if (pset_dir.rows() != 0 && pset_dir.rows() < pset.rows())
    return false;
  if (pset_dir.cols() != 0 && pset_dir.cols() < 3)
    return false;
  if (P.visibility_graph.rows() != P.visibility_graph.cols())
    return false;
  const vector<Eigen::Vector3d>& decomp_points = active_decomp_points();
  if (P.visibility_graph.rows() > 0 && P.visibility_graph.rows() > (int)decomp_points.size())
    return false;

  return true;
}

bool sk_decomp::validVisGraphUnsafe() const
{
  if (P.hl_vis_graph.rows() == 0 || P.hl_vis_graph.cols() == 0)
    return false;
  if (P.hl_vis_graph.rows() != P.hl_vis_graph.cols())
    return false;
  if ((int)P.hl_vis_sets.size() < P.hl_vis_graph.rows())
    return false;

  return true;
}

void sk_decomp::visualization(const ros::TimerEvent& e)
{
  if (!visFlag.load())
    return;

  std::lock_guard<std::mutex> lock(data_mutex_);
  if (!visualizationDataReadyUnsafe())
    return;

  const vector<Eigen::Vector3d>& decomp_points = active_decomp_points();
  const vector<vector<int>>& subspace_sets = active_subspace_sets();
  pcl::PointCloud<pcl::PointXYZ> pcloud, calcloud, drosa, inliers;
  pcl::PointCloud<pcl::Normal> normals, orientations;
  pcl::PointXYZ pt_;
  pcl::Normal n_, o_;

  for (int i=0; i<(int)P.ori_pts_->points.size(); ++i)
  {
    pt_.x = P.ori_pts_->points[i].x;
    pt_.y = P.ori_pts_->points[i].y;
    pt_.z = P.ori_pts_->points[i].z;
    pcloud.points.push_back(pt_);
  }

  for (int j=0; j<(int)P.selected_pts_->points.size(); ++j)
  {
    pt_.x = P.selected_pts_->points[j].x*P.scale + P.center(0);
    pt_.y = P.selected_pts_->points[j].y*P.scale + P.center(1);
    pt_.z = P.selected_pts_->points[j].z*P.scale + P.center(2);
    calcloud.points.push_back(pt_);
  }

  const int normal_size = std::min((int)P.selected_normals_->points.size(), (int)calcloud.points.size());
  for (int k=0; k<normal_size; ++k)
  {
    n_.normal_x = P.selected_normals_->points[k].normal_x;
    n_.normal_y = P.selected_normals_->points[k].normal_y;
    n_.normal_z = P.selected_normals_->points[k].normal_z;
    o_.normal_x = vset(k,0);
    o_.normal_y = vset(k,1);
    o_.normal_z = vset(k,2);
    normals.points.push_back(n_);
    orientations.points.push_back(o_);
  }

  for (int j=0; j<(int)pset.rows(); ++j)
  {
    pt_.x = pset(j,0)*P.scale + P.center(0);
    pt_.y = pset(j,1)*P.scale + P.center(1);
    pt_.z = pset(j,2)*P.scale + P.center(2);
    drosa.push_back(pt_);
  }

  for (int i=0; i<(int)P.inliers.size(); ++i)
  {
    pt_.x = P.inliers[i](0);
    pt_.y = P.inliers[i](1);
    pt_.z = P.inliers[i](2);
    inliers.points.push_back(pt_);
  }

  vector<Eigen::MatrixX3d> intra_edges;
  if (validVisGraphUnsafe())
  {
    for (int i=0; i<(int)P.hl_vis_graph.rows(); ++i)
      for (int j=0; j<(int)P.hl_vis_graph.cols(); ++j)
      {
        if (i <= j) continue;
        if (P.hl_vis_graph(i,j) != 1) continue;

        const auto edge_it = P.hl_vis_sets[i].edges.find(j);
        if (edge_it == P.hl_vis_sets[i].edges.end()) continue;

        const Eigen::Vector2i& edge_idx = edge_it->second;
        const int i_idx = edge_idx(0);
        const int j_idx = edge_idx(1);

        if (i_idx < 0 || j_idx < 0) continue;
        if (i_idx >= (int)P.hl_vis_sets[i].base_nodes.size()) continue;
        if (j_idx >= (int)P.hl_vis_sets[j].base_nodes.size()) continue;

        const int i_node = P.hl_vis_sets[i].base_nodes[i_idx];
        const int j_node = P.hl_vis_sets[j].base_nodes[j_idx];
        if (i_node < 0 || j_node < 0) continue;
        if (i_node >= (int)decomp_points.size() || j_node >= (int)decomp_points.size()) continue;

        Eigen::MatrixX3d edge;
        edge.resize(2,3);
        edge.row(0) = decomp_points[i_node];
        edge.row(1) = decomp_points[j_node];
        intra_edges.push_back(edge);
      }
  }

  vis_utils_->publishMesh(input_mesh);
  vis_utils_->publishSurface(pcloud);
  if ((int)calcloud.points.size() == (int)normals.points.size())
    vis_utils_->publishSurfaceNormal(calcloud, normals);
  if ((int)calcloud.points.size() == (int)orientations.points.size())
    vis_utils_->publishROSAOrientation(calcloud, orientations);
  vis_utils_->publish_dROSA(drosa);
  if (pset_check.cols() >= 3)
    vis_utils_->publishInlierCandidates(pset_check);
  vis_utils_->publishInlierCandidateClasses(P.strict_inlier_candidates,
                                            P.relaxed_inlier_candidates,
                                            P.rejected_inlier_candidates);
  if (pset_dir.rows() >= static_cast<Eigen::Index>(drosa.points.size()) && pset_dir.cols() >= 3)
    vis_utils_->publishInliers(inliers, drosa, pset_dir, intersec_pts);
  vis_utils_->publishIntraEdge(intra_edges);
  if (P.dcrosa_skelver.rows() > 0 && P.dcrosa_skeladj.rows() > 0)
    vis_utils_->publishDcrosaTopology(P.dcrosa_skelver, P.dcrosa_skeladj, P.dcrosa_skeleton_paths);
  vis_utils_->publishSubspaceSkeleton(P.subspace_skeleton_paths);
  vis_utils_->publishSubSpace(P.sub_space_scale);

  if (!decomp_points.empty() && P.visibility_graph.rows() > 0 && P.visibility_graph.rows() <= (int)decomp_points.size() && !subspace_sets.empty())
  {
    vis_utils_->publishVisGraph(decomp_points, P.visibility_graph, subspace_sets);
  }
}

} // namespace flyco
