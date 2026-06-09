/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Oct. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file implements convex decomposition of visibility graph in FlyCo.
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

#ifndef _CVX_DECOMP_HPP_
#define _CVX_DECOMP_HPP_

#include <Eigen/Core>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_set>

using namespace std;

namespace cvx_decomp
{
  inline bool isConvex(const Eigen::MatrixXi& graph, const vector<int>& convex_set, int node)
  {
    for (int setNode : convex_set) 
    {
      if (graph(setNode, node) == 0) return false;
    }

    return true;
  }

  inline bool isConvexSizeSt(const double& max_range, const Eigen::MatrixXi& graph, const vector<Eigen::Vector3d>& pts, const vector<int>& convex_set, int node)
  {
    // * convexity
    for (int setNode : convex_set) 
    {
      if (graph(setNode, node) == 0) return false;
    }
    
    // * size constraint
    Eigen::MatrixXd all_points(3, convex_set.size() + 1);
    for (size_t i = 0; i < convex_set.size(); ++i) 
      all_points.col(i) = pts[convex_set[i]];
    all_points.col(convex_set.size()) = pts[node];

    Eigen::Vector3d centroid = all_points.rowwise().mean();
    Eigen::VectorXd distances = (all_points.colwise() - centroid).colwise().norm();
    double max_distance = distances.maxCoeff();
    
    if (max_distance > max_range) return false;

    return true;
  }

  inline vector<vector<int>> cvxDecomp(const Eigen::MatrixXi& graph)
  {
    int num_nodes = graph.rows();
    vector<vector<int>> convex_sets;

    // calculate degree
    vector<int> degrees(num_nodes, 0);
    for (int i = 0; i < num_nodes; ++i)
      for (int j = 0; j < num_nodes; ++j)
        if (graph(i, j) == 1)
          degrees[i]++;
    
    // sort nodes
    vector<int> node_order(num_nodes);
    iota(node_order.begin(), node_order.end(), 0);
    sort(node_order.begin(), node_order.end(), [&](int a, int b) {
        return degrees[a] > degrees[b];
    });
    
    for (int node : node_order)
    {
      bool assigned = false;
      for (auto& set : convex_sets)
      {
        if (isConvex(graph, set, node))
        {
          set.push_back(node);
          assigned = true;
          break;
        }
      }

      if (!assigned)
      {
        vector<int> new_set{node};
        convex_sets.push_back(new_set);
      }
    }

    return convex_sets;
  }

  inline vector<vector<int>> cvxDecompSizeSt(const Eigen::MatrixXi& graph, const vector<Eigen::Vector3d>& pts, const double& max_range)
  {
    int num_nodes = graph.rows();
    vector<vector<int>> convex_sets;

    // calculate degree
    vector<int> degrees(num_nodes, 0);
    for (int i = 0; i < num_nodes; ++i)
      for (int j = 0; j < num_nodes; ++j)
        if (graph(i, j) == 1)
          degrees[i]++;
    
    // sort nodes
    vector<int> node_order(num_nodes);
    iota(node_order.begin(), node_order.end(), 0);
    sort(node_order.begin(), node_order.end(), [&](int a, int b) {
        return degrees[a] > degrees[b];
    });

    for (int node : node_order)
    {
      bool assigned = false;
      for (auto& set : convex_sets)
      {
        if (isConvexSizeSt(max_range, graph, pts, set, node))
        {
          set.push_back(node);
          assigned = true;
          break;
        }
      }

      if (!assigned)
      {
        vector<int> new_set{node};
        convex_sets.push_back(new_set);
      }
    }

    return convex_sets;
  }

  struct Cluster 
  {
    Eigen::Vector3d centroid;          
    std::vector<int> point_indices;    
    double max_radius;                

    Cluster(const Eigen::Vector3d& init_centroid, int first_point_index) 
    {
      centroid = init_centroid;
      point_indices.push_back(first_point_index);
      max_radius = 0.0;
    }
  };

  inline void updateCluster(const std::vector<Eigen::Vector3d>& points, Cluster& cluster) 
  {
    Eigen::Vector3d new_centroid(0, 0, 0);
    for (int idx : cluster.point_indices) 
    {
      new_centroid += points[idx];
    }
    new_centroid /= cluster.point_indices.size();
    cluster.centroid = new_centroid;

    double max_radius = 0.0;
    for (int idx : cluster.point_indices) 
    {
      double dist = (points[idx] - cluster.centroid).norm();
      if (dist > max_radius) 
      {
        max_radius = dist;
      }
    }
    cluster.max_radius = max_radius;

    return;
  }

  inline vector<vector<int>> decompCvxSizeUni(const Eigen::MatrixXi& visibility, const vector<Eigen::Vector3d>& points, const double& m)
  {
    vector<vector<int>> decomp_sets;

    int N = points.size();
    std::vector<Cluster> clusters;
    clusters.emplace_back(points[0], 0);

    std::vector<int> point_assignments(N, -1);
    point_assignments[0] = 0;

    bool changed = true;
    int max_iterations = 100;
    int iteration = 0;

    while (changed && iteration < max_iterations)
    {
      changed = false;
      iteration++;

      for (int i = 0; i < N; ++i)
      {
        double min_distance = std::numeric_limits<double>::max();
        int best_cluster = -1;

        for (size_t j = 0; j < clusters.size(); ++j)
        {
          Cluster& cluster = clusters[j];
          double distance_to_centroid = (points[i] - cluster.centroid).norm();
          if (distance_to_centroid <= m)
          {
            if (isConvex(visibility, cluster.point_indices, i))
            {
              std::vector<int> temp_indices = cluster.point_indices;
              temp_indices.push_back(i);

              Eigen::Vector3d temp_centroid = cluster.centroid;
              temp_centroid = (temp_centroid * cluster.point_indices.size() + points[i]) / (cluster.point_indices.size() + 1);

              double max_radius = 0.0;
              for (int idx : temp_indices)
              {
                double dist = (points[idx] - temp_centroid).norm();
                if (dist > max_radius) max_radius = dist;
              }

              if (max_radius <= m)
              {
                if (distance_to_centroid < min_distance)
                {
                  min_distance = distance_to_centroid;
                  best_cluster = j;
                }
              }
            }
          }
        }

        if (best_cluster != -1 && point_assignments[i] != best_cluster)
        {
          if (point_assignments[i] != -1) 
          {
            Cluster& old_cluster = clusters[point_assignments[i]];
            old_cluster.point_indices.erase(std::remove(old_cluster.point_indices.begin(), old_cluster.point_indices.end(), i), old_cluster.point_indices.end());
            updateCluster(points, old_cluster);
          }

          Cluster& cluster = clusters[best_cluster];
          cluster.point_indices.push_back(i);
          updateCluster(points, cluster);

          point_assignments[i] = best_cluster;
          changed = true;
        }
        else if (point_assignments[i] == -1) 
        {
          clusters.emplace_back(points[i], i);
          point_assignments[i] = clusters.size() - 1;
          changed = true;
        }
      }

      for (Cluster& cluster : clusters) 
        updateCluster(points, cluster);
    }

    for (size_t i = 0; i < clusters.size(); ++i)
      decomp_sets.push_back(clusters[i].point_indices);

    return decomp_sets;
  }

}

#endif