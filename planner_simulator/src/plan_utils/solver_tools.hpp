/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Oct. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file implements tool functions of path solver in FlyCo.
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

#ifndef _SOLVER_TOOLS_HPP_
#define _SOLVER_TOOLS_HPP_

#include <Eigen/Core>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <vector>
#include <limits>
#include <queue>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;
typedef pcl::PointXYZ PointT;

namespace solver_tools
{
  inline int findMediod(const vector<Eigen::VectorXd>& points, const vector<int>& indices)
  {
    int mediod = -1;
    double min_cost = numeric_limits<double>::max();
    for (int i=0; i<(int)indices.size(); ++i)
    {
      double cost = 0.0;
      for (int j=0; j<(int)indices.size(); ++j)
      {
        if (i == j) continue;
        cost += (points[indices[i]].head(3) - points[indices[j]].head(3)).norm();
      }
      if (cost < min_cost)
      {
        min_cost = cost;
        mediod = i;
      }
    }

    return mediod;
  }

  inline Eigen::Vector3d findCentroid(const vector<Eigen::VectorXd>& points, const vector<int>& indices)
  {
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (int i=0; i<(int)indices.size(); ++i)
      centroid += points[indices[i]].head(3);
    centroid /= (double)indices.size();

    return centroid;
  }

  struct Node 
  {
    int index;
    double distance;
    bool operator>(const Node& other) const 
    {
      return distance > other.distance;
    }
  };
  /**
   * @return the shortest path from start to goal in the graph -> [start, ..., goal]
   */
  inline vector<int> dijkstra(const Eigen::MatrixXi& graph, const vector<Eigen::VectorXd>& points, int start, int goal)
  {
    int n = graph.rows();
    vector<double> distances(n, numeric_limits<double>::infinity());
    vector<int> previous(n, -1);
    priority_queue<Node, vector<Node>, greater<Node>> pq;

    distances[start] = 0.0;
    pq.push({start, 0.0});

    while (!pq.empty())
    {
      Node current = pq.top();
      pq.pop();

      if (current.index == goal) break;

      for (int neighbor = 0; neighbor < n; ++neighbor) 
      {
        if (graph(current.index, neighbor) == 1) 
        {
          double new_dist = distances[current.index] + (points[current.index].head<3>() - points[neighbor].head<3>()).norm();
          if (new_dist < distances[neighbor]) 
          {
            distances[neighbor] = new_dist;
            previous[neighbor] = current.index;
            pq.push({neighbor, new_dist});
          }
        }
      }
    }

    vector<int> path;
    for (int at = goal; at != -1; at = previous[at]) 
    {
      path.push_back(at);
    }
    reverse(path.begin(), path.end());

    if (path.size() == 1 && path[0] != start) 
    {
      return {}; // No path found
    }

    return path;
  }

  inline vector<int> ATSP(const string& file_folder, const string& solver_folder, const Eigen::MatrixXd& cost_mat, const int& run, const int& precision)
  {
    int dim = cost_mat.rows();
    
    // * Interface
    string par = file_folder+"/solver_tools" +".par";
    string prob = file_folder+"/solver_tools" +".tsp";
    string sol = file_folder+"/solver_tools" +".txt";

    // * Write the param file
    ofstream par_file(par);
    par_file << "PROBLEM_FILE = " << prob << "\n";
    par_file << "GAIN23 = NO\n";
    par_file << "OUTPUT_TOUR_FILE =" << sol << "\n";
    par_file << "RUNS = " << to_string(run) << "\n";
    par_file.close();

    // * Write the problem file
    ofstream prob_file(prob);
    string prob_spec = "NAME : Next_sub\nTYPE : ATSP\nDIMENSION : " + to_string(dim) +
                       "\nEDGE_WEIGHT_TYPE : EXPLICIT"
                       "\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n";
    prob_file << prob_spec;
    for (int i=0; i<dim; ++i)
    {
      for (int j=0; j<dim; ++j)
      {
        int int_cost = min(cost_mat(i,j), 100000.0)*precision;
        prob_file << to_string(int_cost) << " ";
      }
      prob_file << "\n";
    }
    prob_file << "EOF";
    prob_file.close();

    // * Run the solver
    string command = "cd " + solver_folder + " && ./LKH " + par;
    const char* charPtr = command.c_str();
    int system_back = system(charPtr);

    // * Read the results
    vector<int> results;
    ifstream res_file(sol);
    string res;
    while (getline(res_file, res)) 
      if (res.compare("TOUR_SECTION") == 0) break;
    while (getline(res_file, res)) 
    {
    int id = stoi(res);
    if (id == -1) break;
    results.push_back(id - 1);
    }
    res_file.close();

    return results;
  }

  inline pcl::PointCloud<PointT>::Ptr convertToPCLCloud(const vector<Eigen::Vector3d>& points)
  {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    cloud->width = points.size();
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->points.reserve(points.size());
    for (const auto& pt : points) 
    {
        cloud->points.emplace_back(PointT(pt.x(), pt.y(), pt.z()));
    }

    return cloud;
  }

  /**
   * @brief calculate the single chamfer distance B->A
   */
  inline double computeSingleDirectionChamfer(const vector<Eigen::Vector3d>& A, const vector<Eigen::Vector3d>& B)
  {
    if ((int)A.size() == 0 || (int)B.size() == 0)
    {
      ROS_WARN("Warning: Empty point clouds provided for Chamfer distance calculation.");
      return 1e7;
    }

    pcl::PointCloud<PointT>::Ptr cloudA = convertToPCLCloud(A);
    pcl::PointCloud<PointT>::Ptr cloudB = convertToPCLCloud(B);

    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloudA);

    double totalDistance = 0.0;
    int validPoints = 0;

    int K=1;
    vector<int> pointIdxNKNSearch(K);
    vector<float> pointNKNSquaredDistance(K);

    for (const auto& point : cloudB->points) 
    {
        if (kdtree.nearestKSearch(point, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) 
        {
        totalDistance += sqrt(pointNKNSquaredDistance[0]);
        validPoints++;
        }
    }

    if (validPoints == 0)
    {
        ROS_WARN("Warning: No valid nearest neighbors found.");
        double max_dist = 1e7;
        return max_dist;
    }
    
    return totalDistance / static_cast<double>(validPoints);
  }

  /**
   * @brief calculate the bi-directional chamfer distance mean(B->A, A->B)
   */
  inline double computeBidirectionalChamfer(const std::vector<Eigen::Vector3d>& A, const std::vector<Eigen::Vector3d>& B)
  {
    if ((int)A.size() == 0 || (int)B.size() == 0)
    {
      ROS_WARN("Warning: Empty point clouds provided for Chamfer distance calculation.");
      return 1e7;
    }

    // Convert input vectors to PCL point clouds
    pcl::PointCloud<PointT>::Ptr cloudA = convertToPCLCloud(A);
    pcl::PointCloud<PointT>::Ptr cloudB = convertToPCLCloud(B);

    // KdTree for nearest neighbor search
    pcl::KdTreeFLANN<PointT> kdtreeA;
    pcl::KdTreeFLANN<PointT> kdtreeB;

    // Set input clouds for the kd-trees
    kdtreeA.setInputCloud(cloudA);
    kdtreeB.setInputCloud(cloudB);

    double totalDistanceAtoB = 0.0;
    double totalDistanceBtoA = 0.0;
    int validPointsAtoB = 0;
    int validPointsBtoA = 0;

    int K = 1; // Nearest neighbor search
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);

    // Compute distances from B to A (first direction)
    for (const auto& point : cloudB->points) 
    {
      if (kdtreeA.nearestKSearch(point, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) 
      {
        totalDistanceAtoB += std::sqrt(pointNKNSquaredDistance[0]);
        validPointsAtoB++;
      }
    }

    // Compute distances from A to B (second direction)
    for (const auto& point : cloudA->points) 
    {
      if (kdtreeB.nearestKSearch(point, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) 
      {
        totalDistanceBtoA += std::sqrt(pointNKNSquaredDistance[0]);
        validPointsBtoA++;
      }
    }

    // Handle edge cases where there are no valid points
    if (validPointsAtoB == 0 || validPointsBtoA == 0)
    {
      ROS_WARN("Warning: No valid nearest neighbors found in one or both directions.");
      return 1e7; // Return a very large distance as an error indicator
    }

    // Compute the bidirectional Chamfer Distance
    double averageDistanceAtoB = totalDistanceAtoB / static_cast<double>(validPointsAtoB);
    double averageDistanceBtoA = totalDistanceBtoA / static_cast<double>(validPointsBtoA);

    return (averageDistanceAtoB + averageDistanceBtoA) / 2.0;
  }

}

#endif