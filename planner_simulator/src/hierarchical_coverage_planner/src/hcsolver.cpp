/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Apr. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the main algorithm of basic solver for planning in FlyCo.
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

#include "hierarchical_coverage_planner/hcsolver.h"

const double MAX = 1e6;

namespace flyco
{
  HCSolver::HCSolver(){
  }

  HCSolver::~HCSolver(){
  }

  void HCSolver::init(ros::NodeHandle& nh)
  {
    // * Module Initialization
    this->astar_.reset(new Astar);
    this->astar_->init_hc(nh);

    this->solve_raycaster_.reset(new RayCaster);
    this->real_raycaster_.reset(new RayCaster);

    // * Params Initialization
    nh.param("hcplanner/precision_", precision_, -1);
    nh.param("hcplanner/global_solver", GlobalSolver_, string("null"));
    nh.param("hcplanner/global_par_file", GlobalPar_, string("null"));
    nh.param("hcplanner/global_problem_file", GlobalProF_, string("null"));
    nh.param("hcplanner/global_result_file", GlobalResult_, string("null"));
    nh.param("hcplanner/global_runs", GlobalRuns_, -1);
    nh.param("hcplanner/local_folder", LocalFolder_, string("null"));
    nh.param("hcplanner/local_runs", LocalRuns_, -1);
    nh.param("hcplanner/boundary_solver", BoundarySolver_, string("null"));
    nh.param("hcplanner/tripod_head", tripod_head_trigger_, false);
    nh.param("hcplanner/vmax_", vm_, -1.0);
    nh.param("hcplanner/amax_", am_, -1.0);
    nh.param("hcplanner/jmax_", jm_, -1.0);
    nh.param("hcplanner/ydmax_", yd_, -1.0);
    nh.param("hcplanner/yddmax_", ydd_, -1.0);
    nh.param("hcplanner/amean_", amean_, -1.0);
    nh.param("hcplanner/local2opt_trial", local2optNum, -1);
    nh.param("astar/resolution_astar", astarSearchingRes, -1.0);
    nh.param("hcplanner/global_solving", global_range_, -1.0);
    nh.param("hcplanner/max_cvx_range", cvx_range_, -1.0);
  }
  
  void HCSolver::setStart(Eigen::Vector3d& start_)
  {
    solver_start_ = start_;
  }

  void HCSolver::setMap(SDFMap::Ptr& hcmap)
  {
    this->solve_map_ = hcmap;
    astar_->setMap(hcmap);
  }

  void HCSolver::setRayCaster(double& res_, Eigen::Vector3d& origin_)
  {
    solve_raycaster_->setParams(res_, origin_);

    return;
  }

  void HCSolver::setRayCasterReal(double& res_, Eigen::Vector3d& origin_)
  {
    real_raycaster_->setParams(res_, origin_);

    return;
  }

  vector<int> HCSolver::GlobalSubspaceSequence(map<int, vector<Eigen::VectorXd>>& sub_vps)
  {
    auto global_t1 = chrono::high_resolution_clock::now();
    
    vector<int> GlobalSeq;
    
    centroids.clear();
    Eigen::Vector3d sub_cen = Eigen::Vector3d::Zero();
    for (const auto& pair:sub_vps)
    {
      sub_cen(0) = 0.0; sub_cen(1) = 0.0; sub_cen(2) = 0.0;
      for (auto pose:pair.second)
      {
        sub_cen(0) += pose(0);
        sub_cen(1) += pose(1);
        sub_cen(2) += pose(2);
      }
      sub_cen = sub_cen/(double)pair.second.size();
      centroids.push_back(sub_cen);
    }

    if ((int)sub_vps.size() > 2)
    {
      /* write par file */
      GlobalParWrite();
      /* construct ATSP cost matrix */
      Eigen::MatrixXd GloablCostMat;
      GloablCostMat = GlobalCostMat(solver_start_, centroids);
      /* write problem file */
      GlobalProblemWrite(GloablCostMat);
      /* ATSP solving */
      string command_ = "cd " + GlobalSolver_ + " && ./LKH " + GlobalPar_;
      const char* charPtr = command_.c_str();
      system_back_=system(charPtr);
      /* read solution results */
      GlobalSeq = GlobalResultsRead();
    }
    else
    {
      vector<double> dist_;
      for (const auto& pair:sub_vps)
      {
        int id = pair.first;
        dist_.push_back((centroids[id]-solver_start_).norm());
      }
      vector<int> idx(dist_.size());
      iota(idx.begin(), idx.end(), 0);
      sort(idx.begin(), idx.end(), [&dist_](int i1, int i2) {return dist_[i1] < dist_[i2];});

      GlobalSeq = idx;
    }

    auto global_t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> global_ms = global_t2 - global_t1;
    double global_time = (double)global_ms.count();
    ROS_INFO("\033[33m[Planner] global sequence planning time = %lf ms.\033[32m", global_time);

    return GlobalSeq;
  }

  map<int, vector<int>> HCSolver::GlobalBoundaryPoints(map<int, vector<Eigen::VectorXd>>& sub_vps, vector<int>& globalseq)
  { 
    auto boundary_t1 = chrono::high_resolution_clock::now();
    
    map<int, vector<int>> boundary_ids_;

    if ((int)sub_vps.size() == 1)
    {
      int sub_id = -1;
      pcl::PointCloud<pcl::PointXYZ>::Ptr oneCloud (new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointXYZ onept, startpt;
      for (const auto& pair:sub_vps)
      {
        sub_id = pair.first;
        for (auto vp:pair.second)
        {
          onept.x = vp(0); onept.y = vp(1); onept.z = vp(2);
          oneCloud->points.push_back(onept);
        }
      }
      pcl::KdTreeFLANN<pcl::PointXYZ> oneTree;
      oneTree.setInputCloud(oneCloud);
      vector<int> nearest(1);
      vector<float> nn_squared_distance(1);
      startpt.x = solver_start_(0); startpt.y = solver_start_(1); startpt.z = solver_start_(2);  
      oneTree.nearestKSearch(startpt, 1, nearest, nn_squared_distance);

      vector<int> boundary; boundary.push_back(nearest[0]);
      boundary_ids_[sub_id] = boundary;
      
      return boundary_ids_;
    }
    
    vector<Eigen::Vector3d> global_site_; global_site_.push_back(solver_start_);
    vector<int> site_seq_; site_seq_.push_back(0);
    for (auto x:globalseq)
    {
      global_site_.push_back(centroids[x]);
      site_seq_.push_back(x);
    }

    /* find start and end of each sub-space */
    Eigen::Vector3d start_s_site_, start_e_site_, end_s_site_, end_e_site_;
    int sub_space_id_;
    int start_id, end_id;
    vector<int> temp_sub_ids_;
    vector<Eigen::VectorXd> sub_space_vps_;
    for (int i=0; i<(int)global_site_.size()-1; ++i)
    {
      vector<int>().swap(temp_sub_ids_);
      sub_space_id_ = site_seq_[i+1];
      sub_space_vps_ = sub_vps.find(sub_space_id_)->second;
      // for start_id
      start_s_site_ = global_site_[i];
      start_e_site_ = global_site_[i+1];
      start_id = FindSphereNearestPoint(start_s_site_, start_e_site_, sub_space_vps_);
      temp_sub_ids_.push_back(start_id);
      // for end_id, notably, final sub-space is without end constraint.
      end_s_site_ = global_site_[i+1];
      if (i+2 < (int)global_site_.size())
      {
        end_e_site_ = global_site_[i+2];
        end_id = FindSphereNearestPoint(end_s_site_, end_e_site_, sub_space_vps_);
        temp_sub_ids_.push_back(end_id);
      }
      boundary_ids_[sub_space_id_] = temp_sub_ids_;
    }

    auto boundary_t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> boundary_ms = boundary_t2 - boundary_t1;
    double boundary_time = (double)boundary_ms.count();
    ROS_INFO("\033[33m[Planner] local boundary selection time = %lf ms.\033[32m", boundary_time);

    return boundary_ids_;
  }

  vector<int> HCSolver::topoGlobalSeq(map<int, vector<Eigen::VectorXd>>& sub_vps, vector<Eigen::Vector3d>& topoNodes, map<vector<int>, int>& topoEdges, Eigen::MatrixXd& topoAdjMat, vector<vector<int>>& all_nodes_idx, vector<Eigen::Vector3d>& all_nodes)
  {
    auto global_t1 = chrono::high_resolution_clock::now();

    vector<int> GlobalSeq;
    
    centroids.clear();
    Eigen::Vector3d sub_cen = Eigen::Vector3d::Zero();
    for (const auto& pair:sub_vps)
    {
      sub_cen(0) = 0.0; sub_cen(1) = 0.0; sub_cen(2) = 0.0;
      for (auto pose:pair.second)
      {
        sub_cen(0) += pose(0);
        sub_cen(1) += pose(1);
        sub_cen(2) += pose(2);
      }
      sub_cen = sub_cen/(double)pair.second.size();
      centroids.push_back(sub_cen);
    }

    if ((int)sub_vps.size() < 3)
    {
      vector<double> dist_;
      for (const auto& pair:sub_vps)
      {
        int id = pair.first;
        dist_.push_back((centroids[id]-solver_start_).norm());
      }
      vector<int> idx(dist_.size());
      iota(idx.begin(), idx.end(), 0);
      sort(idx.begin(), idx.end(), [&dist_](int i1, int i2) {return dist_[i1] < dist_[i2];});

      GlobalSeq = idx;
    }
    else
    {
      // * Select Start Node
      int nearest_cvx = -1;
      double min_dist = MAX;
      for (int i=0; i<(int)all_nodes_idx.size(); ++i)
      {
        for (int j=0; j<(int)all_nodes_idx[i].size(); ++j)
        {
          int idx = all_nodes_idx[i][j];
          double dist = (all_nodes[idx]-solver_start_).norm();
          if (dist < min_dist)
          {
            min_dist = dist;
            nearest_cvx = i;
          }
        }
      }

      double dist_a = (topoNodes[2*nearest_cvx] - solver_start_).norm();
      double dist_b = (topoNodes[2*nearest_cvx+1] - solver_start_).norm();
      int nearestID = (dist_a < dist_b) ? 2*nearest_cvx : 2*nearest_cvx+1;

      // * Generalized Eulerian Path
      vector<int> vPath = generalizedEulerianPath(topoAdjMat, nearestID);
      // cout << "Eulerian path: " << endl;
      // for (auto x:vPath)
      //   cout << x << " ";
      // cout << endl;

      vector<bool> visited(sub_vps.size(), false);
      for (int i=0; i<(int)vPath.size()-1; ++i)
      {
        int a = vPath[i];
        int b = vPath[i+1];
        vector<int> edge = {a, b};
        if (topoEdges.find(edge) != topoEdges.end())
        {
          int sub_id = topoEdges.find(edge)->second;
          if (visited[sub_id] == false)
          {
            visited[sub_id] = true;
            GlobalSeq.push_back(sub_id);
          }
        }
      }
    }

    auto global_t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> global_ms = global_t2 - global_t1;
    double global_time = (double)global_ms.count();
    ROS_INFO("\033[33m[Planner] global sequence planning time = %lf ms.\033[32m", global_time);

    return GlobalSeq;
  }

  map<int, vector<int>> HCSolver::topoBoundaryPoints(map<int, vector<Eigen::VectorXd>>& sub_vps, vector<int>& globalseq, map<int, Eigen::Vector3d>& nearestPts, map<vector<int>, vector<Eigen::Vector3d>>& ctrlPts)
  {
    auto boundary_t1 = chrono::high_resolution_clock::now();

    map<int, vector<int>> topoBoundaryID;

    if ((int)globalseq.size() == 1)
    {
      int sub_id = -1;
      pcl::PointCloud<pcl::PointXYZ>::Ptr oneCloud (new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointXYZ onept, startpt;
      for (const auto& pair:sub_vps)
      {
        sub_id = pair.first;
        for (auto vp:pair.second)
        {
          onept.x = vp(0); onept.y = vp(1); onept.z = vp(2);
          oneCloud->points.push_back(onept);
        }
      }
      pcl::KdTreeFLANN<pcl::PointXYZ> oneTree;
      oneTree.setInputCloud(oneCloud);
      vector<int> nearest(1);
      vector<float> nn_squared_distance(1);
      startpt.x = solver_start_(0); startpt.y = solver_start_(1); startpt.z = solver_start_(2);  
      oneTree.nearestKSearch(startpt, 1, nearest, nn_squared_distance);

      vector<int> boundary; boundary.push_back(nearest[0]);
      topoBoundaryID[sub_id] = boundary;

      return topoBoundaryID;
    }

    vector<Eigen::Vector3d> global_site_ = {solver_start_, nearestPts.find(globalseq[0])->second}; 
    for (int i=0; i<(int)globalseq.size()-1; ++i)
    {
      if (ctrlPts.find({globalseq[i], globalseq[i+1]}) != ctrlPts.end())
      {
        vector<Eigen::Vector3d> ctrlPts_ = ctrlPts.find({globalseq[i], globalseq[i+1]})->second;
        global_site_.insert(global_site_.end(), ctrlPts_.begin(), ctrlPts_.end());
      }
      else
      {
        ROS_ERROR("No control points found between %d and %d.", globalseq[i], globalseq[i+1]);
        exit(1);
      }
    }

    vector<Eigen::Vector3d> global_centers; global_centers.push_back(solver_start_);
    for (int i=0; i<(int)globalseq.size(); ++i)
      global_centers.push_back(centroids[globalseq[i]]);

    for (int i=0; i<(int)globalseq.size(); ++i)
    {
      vector<int> tempBoundary;
      vector<Eigen::VectorXd> subSpaceVps = sub_vps.find(globalseq[i])->second;
      // * start id
      Eigen::Vector3d startSsite = global_site_[2*i];
      Eigen::Vector3d startEsite = global_site_[2*i+1];
      Eigen::Vector3d startScen = global_centers[i];
      Eigen::Vector3d startEcen = global_centers[i+1];
      int startID = fourPointsNearestPoint(startSsite, startEsite, startScen, startEcen, subSpaceVps);
      tempBoundary.push_back(startID);
      // * end id
      if (i < (int)globalseq.size()-1)
      {
        Eigen::Vector3d endSsite = global_site_[2*i+2];
        Eigen::Vector3d endEsite = global_site_[2*i+3];
        Eigen::Vector3d endScen = global_centers[i+1];
        Eigen::Vector3d endEcen = global_centers[i+2];
        int endID = fourPointsNearestPoint(endSsite, endEsite, endScen, endEcen, subSpaceVps);
        tempBoundary.push_back(endID);
      }

      topoBoundaryID[globalseq[i]] = tempBoundary;
    }

    auto boundary_t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> boundary_ms = boundary_t2 - boundary_t1;
    double boundary_time = (double)boundary_ms.count();
    ROS_INFO("\033[33m[Planner] local boundary selection time = %lf ms.\033[32m", boundary_time);

    return topoBoundaryID;
  }

  tuple<map<int, vector<Eigen::VectorXd>>, map<int, vector<vector<Eigen::VectorXd>>>> HCSolver::LocalConditionalPath(map<int, vector<Eigen::VectorXd>>& sub_vps, map<int, vector<int>>& global_boundary, bool turn)
  {
    auto plan_t1 = chrono::high_resolution_clock::now();
    
    auto hcopp_astar_t1 = std::chrono::high_resolution_clock::now();
    for (const auto& pair:sub_vps)
    {
      LocalFindPath(pair.second, global_boundary.find(pair.first)->second, pair.first, turn);
    }
    auto hcopp_astar_t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> hcopp_astar_ms = hcopp_astar_t2 - hcopp_astar_t1;
    double hcopp_astar_time = (double)hcopp_astar_ms.count();
    ROS_INFO("\033[33m[Planner] path searching time = %lf ms.\033[32m", hcopp_astar_time);

    int sub_space_id_, num=0;
    int numThreads = static_cast<int>(sub_vps.size());
    std::thread threads[numThreads];

    for (const auto& pair:sub_vps)
    {
      sub_space_id_ = pair.first;
      threads[num] = std::thread(&HCSolver::LocalPathFinder, this, pair.second, global_boundary.find(sub_space_id_)->second, sub_space_id_);
      num++;
    }

    /* Waiting all threads to the end... */
    for (int i = 0; i < numThreads; i++)
    {
      if (threads[i].joinable())
        threads[i].join();
    }
    
    auto plan_t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> plan_ms = plan_t2 - plan_t1;
    double local_time = (double)plan_ms.count();
    ROS_INFO("\033[33m[Planner] parallel local path planning time = %lf ms.\033[32m", local_time);

    return make_tuple(local_sub_path_viewpts_, local_sub_path_waypts_);
  }

  tuple<vector<Eigen::VectorXd>, vector<vector<Eigen::VectorXd>>> HCSolver::CoverageFullPath(
    Eigen::VectorXd& cur_pos, vector<int>& globalseq, map<int, vector<Eigen::VectorXd>>& localviewpts, map<int, vector<vector<Eigen::VectorXd>>>& localwaypts)
  { 
    if (globalseq.empty()) 
    {
      ROS_ERROR("Global sequence is empty.");
      exit(1);
    }

    for (const auto& id : globalseq) 
    {
      if (localviewpts.find(id) == localviewpts.end()) {
        ROS_ERROR("No viewpoints found for sub-space %d.", id);
        exit(1);
      }
      if (localwaypts.find(id) == localwaypts.end()) {
        ROS_ERROR("No waypoints found for sub-space %d.", id);
        exit(1);
      }
    }
    
    vector<Eigen::VectorXd> full_viewpts_;
    vector<vector<Eigen::VectorXd>> full_waypts_;
    Eigen::Vector3d p1, p2;
    vector<Eigen::Vector3d> waypts_;
    vector<Eigen::VectorXd> UPwaypts_;

    full_viewpts_.push_back(cur_pos);
    /* start path */
    waypts_.clear();
    UPwaypts_.clear();
    waypts_ = {};
    UPwaypts_ = {};
    full_waypts_.push_back(UPwaypts_);
    Eigen::Vector3d lastPos = full_viewpts_.back().head(3);
    // ! Assert
    if (localviewpts.find(globalseq[0]) == localviewpts.end())
    {
      ROS_ERROR("No viewpoints found in the first sub-space.");
      exit(1);
    }
    // ! Assert
    for (auto pose:localviewpts.find(globalseq[0])->second)
    {
      Eigen::Vector3d curPos = pose.head(3);
      if ((lastPos-curPos).norm() < 1e-3)
        continue;
      full_viewpts_.push_back(pose);
    }
    // ! Assert
    if (localwaypts.find(globalseq[0]) == localwaypts.end())
    {
      ROS_ERROR("No waypoints found in the first sub-space.");
      exit(1);
    }
    // ! Assert
    full_waypts_.insert(full_waypts_.end(), localwaypts.find(globalseq[0])->second.begin(), localwaypts.find(globalseq[0])->second.end());

    /* following paths */
    for (int i=0; i<(int)globalseq.size()-1; ++i)
    {
      vector<Eigen::Vector3d>().swap(waypts_);
      vector<Eigen::VectorXd>().swap(UPwaypts_);
      p1 = localviewpts.find(globalseq[i])->second.back().head(3);
      p2 = localviewpts.find(globalseq[i+1])->second.front().head(3);
      bridge_length_ = search_Path(p1, p2, waypts_);
      // ! Assert
      if (bridge_length_ < 0) 
      {
        ROS_ERROR("Failed to find a path between sub-spaces %d and %d.", globalseq[i], globalseq[i+1]);
        exit(1);
      }
      // ! Assert
      if ((int)waypts_.size() > 2)
      {
        AngleInterpolation(localviewpts.find(globalseq[i])->second.back(), localviewpts.find(globalseq[i+1])->second.front(), waypts_, UPwaypts_);
        full_waypts_.push_back(UPwaypts_);
      }
      else
        full_waypts_.push_back({});

      Eigen::Vector3d lastPos = full_viewpts_.back().head(3);
      // ! Assert
      if (localviewpts.find(globalseq[i+1]) == localviewpts.end())
      {
        ROS_ERROR("No viewpoints found in the %d sub-space.", i+1);
        exit(1);
      }
      // ! Assert
      for (auto pose:localviewpts.find(globalseq[i+1])->second)
      {
        Eigen::Vector3d curPos = pose.head(3);
        if ((lastPos-curPos).norm() < 1e-3)
          continue;
        full_viewpts_.push_back(pose);
      }
      // ! Assert
      if (localwaypts.find(globalseq[i+1]) == localwaypts.end())
      {
        ROS_ERROR("No waypoints found in the %d sub-space.", i+1);
        exit(1);
      }
      // ! Assert
      full_waypts_.insert(full_waypts_.end(), localwaypts.find(globalseq[i+1])->second.begin(), localwaypts.find(globalseq[i+1])->second.end());
    }

    return make_tuple(full_viewpts_, full_waypts_);
  }

  vector<Eigen::VectorXd> HCSolver::LocalRefine(vector<Eigen::Vector3d>& Joints, double& Range, vector<Eigen::VectorXd>& Path, bool turn, double time_bound, bool all_refine)
  {
    if ((int)Joints.size() == 0 && all_refine == false)
    {
      return Path;
    }
    
    vector<Eigen::VectorXd> RefinedPath, FinalPath;
    
    // ! Random Local 2-opt/3-opt
    RefinedPath = Path;
    this->JointVps.clear();
    vector<int> innervpsID, finalID;

    if (all_refine == false)
    {
      pcl::KdTreeFLANN<pcl::PointXYZ> PathTree;
      vector<int> indxs;
      vector<float> radius_squared_distance;
      pcl::PointXYZ serachPt;
      ConstructKDTree(RefinedPath, PathTree);
      for (auto jt:Joints)
      {
        serachPt.x = jt(0); serachPt.y = jt(1); serachPt.z = jt(2);
        PathTree.radiusSearch(serachPt, Range, indxs, radius_squared_distance);
        innervpsID.insert(innervpsID.end(), indxs.begin(), indxs.end());
        vector<Eigen::Vector3d> innervps;
        for (int k=0; k<(int)indxs.size(); ++k)
          innervps.push_back(RefinedPath[indxs[k]].head(3));
        JointVps.push_back(innervps);
      }
    }
    else
    {
      vector<Eigen::Vector3d> innervps;
      for (int i=0; i<(int)RefinedPath.size(); ++i)
      {
        innervps.push_back(RefinedPath[i].head(3));
        innervpsID.push_back(i);
      }
      JointVps.push_back(innervps);
    }

    set<int> uniqueSet(innervpsID.begin(), innervpsID.end());
    for (const auto& num : uniqueSet)
      finalID.push_back(num);

    this->LocalVps.clear();
    for (auto id:finalID)
      LocalVps.push_back(RefinedPath[id].head(3));

    this->RefineID.clear();
    for (int i=0; i<(int)RefinedPath.size(); ++i)
      RefineID[RefinedPath[i].head(3)] = i;
    
    double refinelength = 0.0;
    for (int i=0; i<(int)RefinedPath.size()-1; ++i)
      refinelength += (RefinedPath[i+1].head(3)-RefinedPath[i].head(3)).norm();

    /* Local random 2 Opt */
    this->AllPathSite.clear();
    this->LocalSite.clear();

    auto twoopt_t1 = std::chrono::high_resolution_clock::now();
    // * data prepare START
    int gID = 0;
    Site* StartSite = new Site;
    StartSite->X = RefinedPath.front()(0); StartSite->Y = RefinedPath.front()(1); StartSite->Z = RefinedPath.front()(2);
    StartSite->Pitch = RefinedPath.front().size() == 5? RefinedPath.front()(3):0.0;
    StartSite->Yaw = RefinedPath.front().tail(1)(0); 
    StartSite->Start = true; StartSite->End = false;
    StartSite->GlobalID = gID;
    StartSite->LocalID = -1;
    gID++;
    AllPathSite.push_back(StartSite);
    for (int i=1; i<(int)RefinedPath.size()-1; ++i)
    {
      Site* tempSite = new Site;
      tempSite->X = RefinedPath[i](0); tempSite->Y = RefinedPath[i](1); tempSite->Z = RefinedPath[i](2);
      tempSite->Pitch = RefinedPath[i].size() == 5? RefinedPath[i](3):0.0;
      tempSite->Yaw = RefinedPath[i].tail(1)(0);
      tempSite->Start = false; tempSite->End = false;
      tempSite->GlobalID = gID;
      tempSite->LocalID = -1;
      gID++;
      AllPathSite.push_back(tempSite);
    }
    Site* EndSite = new Site;
    EndSite->X = RefinedPath.back()(0); EndSite->Y = RefinedPath.back()(1); EndSite->Z = RefinedPath.back()(2);
    EndSite->Pitch = RefinedPath.back().size() == 5? RefinedPath.back()(3):0.0;
    EndSite->Yaw = RefinedPath.back().tail(1)(0); 
    EndSite->Start = false; EndSite->End = true;
    EndSite->GlobalID = gID;
    EndSite->LocalID = -1;
    AllPathSite.push_back(EndSite);

    AllPathSite.front()->Pred = nullptr;
    AllPathSite.front()->Suc = AllPathSite[1];
    for (int i=1; i<(int)AllPathSite.size()-1; ++i)
    {
      AllPathSite[i]->Pred = AllPathSite[i-1];
      AllPathSite[i]->Suc = AllPathSite[i+1];
    }
    AllPathSite.back()->Pred = AllPathSite[(int)AllPathSite.size()-2];
    AllPathSite.back()->Suc = nullptr;
    
    int count = 0;
    for (auto vp:LocalVps)
    {
      int id = RefineID.find(vp)->second;

      AllPathSite[id]->LocalID = count;
      LocalSite.push_back(AllPathSite[id]);
      count++;
    }
    // * data prepare END

    // * Local 2-opt/3-opt START
    swapTimes = 0;
    int attempts = 0;
    double time_upper = time_bound;

    auto t1 = chrono::high_resolution_clock::now();
    auto t2 = chrono::high_resolution_clock::now(); 

    if ((int)LocalSite.size() > 0)
    {
      for (int i=0; i<local2optNum; ++i)
      {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        double prob_local = dis(gen);
        if (prob_local > 0.2)
        {
          RandomLocal2Opt(turn);
        }
        else
        {
          RandomLocal3Opt(turn);
        }

        attempts++;
        t2 = chrono::high_resolution_clock::now();
        double cur_time_cost = (double)chrono::duration<double, std::milli>(t2 - t1).count();
        if (cur_time_cost > time_upper)
          break;
      }
    }
    ROS_INFO("\033[37m[LocalRefine] total local search attempts: %d, valid local search attempts: %d. \033[32m", attempts, swapTimes);
    // * Local 2-opt/3-opt END
    
    // * obtain final path START
    Site *StartNode = new Site;
    for (auto n:AllPathSite)
    {
      if (n->Start == true)
        StartNode = n;
    }
    // ROS_WARN("updated path with %d points: ", (int)AllPathSite.size());
    // cout << StartNode->GlobalID << " -> ";

    Eigen::VectorXd pose; pose.resize(5);
    pose << StartNode->X, StartNode->Y, StartNode->Z, StartNode->Pitch, StartNode->Yaw;
    FinalPath.push_back(pose);
    Eigen::VectorXd last_pose;
    
    Site *LastNode = new Site; LastNode = StartNode;
    Site *NextNode = new Site;
    NextNode->End = false;
    while (NextNode->End != true)
    {
      NextNode = LastNode->Suc;
      // cout << NextNode->GlobalID << " -> ";
      pose(0) = NextNode->X; pose(1) = NextNode->Y; pose(2) = NextNode->Z; pose(3) = NextNode->Pitch; pose(4) = NextNode->Yaw;
      last_pose = FinalPath.back();
      // --- START insert collision-free interpath ---
      vector<Eigen::Vector3d> ori_path;
      vector<Eigen::VectorXd> updated_path;
      Eigen::Vector3d last_pos = last_pose.head(3);
      Eigen::Vector3d now_pos = pose.head(3);
      inter_cost = search_Path(last_pos, now_pos, ori_path);
      if ((int)ori_path.size() > 2)
      {
        AngleInterpolation(last_pose, pose, ori_path, updated_path);
        for (auto p:updated_path)
          refined_waypoints_indicators_[p] = true;
      }
      else
        updated_path = {last_pose, pose};
      
      if ((int)updated_path.size() > 2)
      {
        for (int k=1; k<(int)updated_path.size()-1; ++k)
          FinalPath.push_back(updated_path[k]);
      }
      // --- END insert collision-free interpath ---

      FinalPath.push_back(pose);
      LastNode = NextNode;
    }
    // cout << endl;

    auto twoopt_t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> twoopt_ms = twoopt_t2 - twoopt_t1;
    // * obtain final path END

    double local2optlength = 0.0;
    for (int i=0; i<(int)FinalPath.size()-1; ++i)
      local2optlength += (FinalPath[i+1].head(3)-FinalPath[i].head(3)).norm();
    
    ROS_INFO("\033[37m[LocalRefine] local path refinement improvement = %lf m.\033[32m", refinelength-local2optlength);

    return FinalPath;
  }

  void HCSolver::consistencyRefine(const vector<Eigen::VectorXd>& path, const Eigen::Vector3d& vel, vector<Eigen::VectorXd>& refine_path, vector<bool>& indi)
  {
    auto t1 = chrono::high_resolution_clock::now();

    // * Motion Consistent Path Solver
    this->solveConsistentPath(path, vel, refine_path, indi);

    auto t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms = t2 - t1;
    double time = (double)ms.count();
    ROS_INFO("\033[33m[GlobalConsistency] motion consistent path refinement time = %lf ms.\033[32m", time);
  }

  // ? Consistent Planning Func
  void HCSolver::parallelPairPathSearching(const map<int, vector<Eigen::VectorXd>>& sub_vps)
  {
    for (const auto& pair:sub_vps)
    {
      int sub_idx = pair.first;
      vector<Eigen::VectorXd> vps_set = pair.second;
      subPathSearching(sub_idx, vps_set);
    }

    return;
  }

  /**
   * @param astar_results Astar results of this sub-space.
   * @param sub_idx       Sub-space index.
   * @param start_idx     Start index.
   * @param viewpoints    Viewpoints in one sub-space, N-dim.
   * @param start_end     Whether the start and end constraint.
   * @param end_idx       End index.
   * @param vps_path      Path of viewpoints, N-dim.
   */
  void HCSolver::ATSPSolver(const map<pair<Eigen::VectorXd, Eigen::VectorXd>, AstarResults, PairVectorXdCompare> astar_results, const int sub_idx, const int start_idx, const vector<Eigen::VectorXd> viewpoints, const bool start_end, const int end_idx)
  {
    vector<Eigen::VectorXd> vps_path;
    map<Eigen::VectorXd, bool, VectorXdCompare> sub_waypt_indi;
    // * Prepare the cost viewpoints
    vector<Eigen::VectorXd> re_viewpoints = {viewpoints[start_idx]};
    for (int i=0; i<(int)viewpoints.size(); ++i)
    {
      if (start_end == true && i == end_idx) continue;

      if (i != start_idx)
        re_viewpoints.push_back(viewpoints[i]);
    }
    if (start_end == true)
      re_viewpoints.push_back(viewpoints[end_idx]);

    if ((int)re_viewpoints.size() < 3)
    {
      if ((int)viewpoints.size() == 1)
      {
        vps_path = viewpoints;
        sub_waypt_indi[viewpoints.front()] = false;
      }
      else
      {
        Eigen::VectorXd vp_i = re_viewpoints.front();
        Eigen::VectorXd vp_j = re_viewpoints.back();
        vps_path.push_back(vp_i);
        sub_waypt_indi[vp_i] = false;
        if (astar_results.find(make_pair(vp_i, vp_j)) != astar_results.end())
        {
          AstarResults astar_result = astar_results.find(make_pair(vp_i, vp_j))->second;
          vps_path.insert(vps_path.end(), astar_result.path.begin(), astar_result.path.end());
          for (auto p:astar_result.path)
            sub_waypt_indi[p] = true;
        }
        vps_path.push_back(vp_j);
        sub_waypt_indi[vp_j] = false;
      }
    }
    else
    {
      // * Construct the cost matrix
      int dim = re_viewpoints.size();
      Eigen::MatrixXd cost_mat;
      cost_mat.resize(dim, dim);
      cost_mat.setZero();

      for (int i=1; i<dim; ++i)
      {
        for (int j=1; j<dim; ++j)
        {
          if (i<=j) continue;
          
          Eigen::VectorXd vp_i = re_viewpoints[i];
          Eigen::VectorXd vp_j = re_viewpoints[j];

          double cost = 0.0;
          if (astar_results.find(make_pair(vp_i, vp_j)) != astar_results.end())
          {
            AstarResults astar_result = astar_results.find(make_pair(vp_i, vp_j))->second;
            cost = astar_result.cost;
          }
          else
            cost = (vp_i.head(3)-vp_j.head(3)).norm();
          cost_mat(i,j) = cost;
          cost_mat(j,i) = cost;
        }
      }

      Eigen::VectorXd start_vp = re_viewpoints.front();
      for (int i=1; i<dim; ++i)
      {
        Eigen::VectorXd vp_i = re_viewpoints[i];

        double cost = 0.0;
        if (astar_results.find(make_pair(start_vp, vp_i)) != astar_results.end())
        {
          AstarResults astar_result = astar_results.find(make_pair(start_vp, vp_i))->second;
          cost = astar_result.cost;
        }
        else
          cost = (start_vp.head(3)-vp_i.head(3)).norm();

        cost_mat(0, i) = cost;
      }
      
      if (start_end == true)
      {
        for (int i=1; i<dim-1; ++i)
          cost_mat(dim-1, i) = COST_INFINITY;
      }
      
      // * Interface
      string next_par = LocalFolder_+"/cp_"+ to_string(sub_idx) +".par";
      string next_prob = LocalFolder_+"/cp_"+ to_string(sub_idx) +".tsp";
      string next_sol = LocalFolder_+"/cp_"+ to_string(sub_idx) +".txt";

      // * Write the param file
      ofstream par_file(next_par);
      par_file << "PROBLEM_FILE = " << next_prob << "\n";
      par_file << "GAIN23 = NO\n";
      par_file << "OUTPUT_TOUR_FILE =" << next_sol << "\n";
      par_file << "RUNS = " << to_string(LocalRuns_) << "\n";
      par_file.close();

      // * Write the problem file
      ofstream prob_file(next_prob);
      string prob_spec = "NAME : Next_sub\nTYPE : ATSP\nDIMENSION : " + to_string(dim) +
        "\nEDGE_WEIGHT_TYPE : "
        "EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n";
      prob_file << prob_spec;
      for (int i=0; i<dim; ++i)
      {
        for (int j=0; j<dim; ++j)
        {
          int int_cost = cost_mat(i,j)*precision_;
          prob_file << to_string(int_cost) << " ";
        }
        prob_file << "\n";
      }
      prob_file << "EOF";
      prob_file.close();

      // * Run the solver
      string next_command = "cd " + GlobalSolver_ + " && ./LKH " + next_par;
      const char* next_charPtr = next_command.c_str();
      next_system_back_ = system(next_charPtr);

      // * Read the results
      vector<int> results;
      ifstream res_file(next_sol);
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

      // * Reorder the viewpoints
      for (int i=0; i<(int)results.size(); ++i)
      {
        int idx = results[i];
        vps_path.push_back(re_viewpoints[idx]);
        if (i < (int)results.size()-1)
        {
          Eigen::VectorXd vp_i = re_viewpoints[idx];
          Eigen::VectorXd vp_j = re_viewpoints[results[i+1]];
          if (astar_results.find(make_pair(vp_i, vp_j)) != astar_results.end())
          {
            AstarResults astar_result = astar_results.find(make_pair(vp_i, vp_j))->second;
            vps_path.insert(vps_path.end(), astar_result.path.begin(), astar_result.path.end());
            for (auto p:astar_result.path)
              sub_waypt_indi[p] = true;
          }
        }
      }
    }

    bool append = false;
    while (!append)
    {
      if (this->cp_mtx_.try_lock())
      {
        this->sub_paths_[sub_idx] = vps_path;
        this->sub_waypoints_indicators_[sub_idx] = sub_waypt_indi;
        this->cp_mtx_.unlock();
        append = true;
      }
      else
        ros::Duration(0.001).sleep();
    }

    return;
  }

  /**
   * @param remaining_vps       Viewpoints in remaining sub-spaces.
   * @param remaining_boundary  Boundary of remaining sub-spaces.
   * @param remaining_subs      Remaining sub-space global indices.
   * @param remaining_order     Remaining sub-space order of remaining_subs.
   * @param remaining_full_path Full path of remaining sub-spaces.
   */
  void HCSolver::parallelSubPlanning(const map<int, vector<Eigen::VectorXd>>& remaining_vps, const map<int, vector<int>>& remaining_boundary, const vector<int>& remaining_subs, const vector<int>& remaining_order, vector<Eigen::VectorXd>& remaining_full_path)
  {
    int sub_idx = -1, num = 0;
    map<pair<Eigen::VectorXd, Eigen::VectorXd>, AstarResults, PairVectorXdCompare> astar_results;
    vector<int> boundary;
    vector<Eigen::VectorXd> vps;
    bool start_end_st = false;
    int start_idx = -1, end_idx = -1;
    int num_threads = static_cast<int>(remaining_subs.size());
    std::thread threads[num_threads];

    for (const auto& pair:remaining_vps)
    {
      sub_idx = remaining_subs[pair.first];
      astar_results = this->sub_pair_astar_results_[sub_idx];
      boundary = remaining_boundary.find(pair.first)->second;
      start_end_st = ((int)boundary.size() == 2) ? true : false;
      vps = pair.second;
      start_idx = boundary.front();
      if (start_end_st == true)
        end_idx = boundary.back();

      threads[num] = std::thread(&HCSolver::ATSPSolver, this, astar_results, sub_idx, start_idx, vps, start_end_st, end_idx);
      num++;
    }

    for (int i = 0; i < num_threads; i++)
    {
      if (threads[i].joinable())
        threads[i].join();
    }

    Eigen::VectorXd last_end, next_start;
    for (int i=0; i<(int)remaining_order.size(); ++i)
    {
      int sub_order = remaining_order[i];
      int sub_idx = remaining_subs[sub_order];
      vector<Eigen::VectorXd> sub_path = this->sub_paths_[sub_idx];

      remaining_full_path.insert(remaining_full_path.end(), sub_path.begin(), sub_path.end());
      if (i < (int)remaining_order.size()-1)
      {
        last_end = sub_path.back();
        next_start = remaining_vps.find(remaining_order[i+1])->second.front();
        vector<Eigen::Vector3d> inter_pos_path;
        vector<Eigen::VectorXd> inter_path;
        inter_cost = search_Path(last_end.head(3), next_start.head(3), inter_pos_path);
        if ((int)inter_pos_path.size() > 2)
          AngleInterpolation(last_end, next_start, inter_pos_path, inter_path);
        else
          inter_path = {};
        remaining_full_path.insert(remaining_full_path.end(), inter_path.begin(), inter_path.end());
      }
    }

    return;
  }

  // ! /* Decomp Planning */
  /**
   * @brief Consistent global sequence planning.
   */
  void HCSolver::decompGlobalSeq(const vector<Eigen::VectorXd>& vps, const vector<vector<int>>& vps_set, const vector<Eigen::VectorXd>& LAST_vps, const vector<vector<int>>& LAST_vps_set, const vector<int>& LAST_global_seq, const int& reused_sets, const int& updated_sets, const int& match_k, vector<int>& global_seq, vector<Eigen::Vector3d>& global_path)
  {
    // * step 1: Find the centroids of each set
    vector<Eigen::Vector3d> vps_centroids;
    vps_centroids.reserve(vps_set.size());
    for (int i=0; i<(int)vps_set.size(); ++i)
    {
      Eigen::Vector3d centroid = solver_tools::findCentroid(vps, vps_set[i]);
      vps_centroids.push_back(centroid);
    }

    // * step 2: Reused viewpoints sets
    for (int i=0; i<updated_sets; ++i)
      global_seq.push_back(i);
    
    // * step 3: Chamfer matching
    if (!LAST_global_seq.empty() && updated_sets > 0)
    {
      vector<bool> align_state(vps_set.size(), false);
      for (int i=0; i<updated_sets; ++i) align_state[i] = true;
      int range = reused_sets + match_k < (int)LAST_global_seq.size() ? reused_sets + match_k : (int)LAST_global_seq.size();

      vector<vector<Eigen::Vector3d>> this_vps_pos;
      for (int i=0; i<(int)vps_set.size(); ++i)
      {
        vector<Eigen::Vector3d> temp_set;
        for (int j=0; j<(int)vps_set[i].size(); ++j) temp_set.push_back(vps[vps_set[i][j]].head(3));
        this_vps_pos.push_back(temp_set);
      }

      vector<vector<Eigen::Vector3d>> last_vps_pos;
      for (int i=0; i<(int)LAST_vps_set.size(); ++i)
      {
        vector<Eigen::Vector3d> temp_set;
        for (int j=0; j<(int)LAST_vps_set[i].size(); ++j) temp_set.push_back(LAST_vps[LAST_vps_set[i][j]].head(3));
        last_vps_pos.push_back(temp_set);
      }
      
      for (int i=reused_sets; i<range; ++i)
      {
        int last_sub_idx = LAST_global_seq[i];
        vector<Eigen::Vector3d> last_vps = last_vps_pos[last_sub_idx];

        double min_dist = 1e6;
        int best_align = -1;

        for (int j=0; j<(int)this_vps_pos.size(); ++j)
        {
          if (align_state[j] == true) continue;

          vector<Eigen::Vector3d> cur_vps = this_vps_pos[j];
          double chamfer_distance = solver_tools::computeSingleDirectionChamfer(last_vps, cur_vps);
          if (chamfer_distance < min_dist)
          {
            min_dist = chamfer_distance;
            best_align = j;
          }
        }

        if (best_align != -1)
        {          
          double consist_dist = (vps_centroids[global_seq.back()] - vps_centroids[best_align]).norm();
          if (consist_dist < 3 * this->cvx_range_)
          {
            align_state[best_align] = true;
            global_seq.push_back(best_align);
          }
          else
          {
            ROS_WARN("Too far for keeping consistency! -> %f", consist_dist);
            break;
          }
        }
      }
    }

    auto t1 = chrono::high_resolution_clock::now();

    // * step 4: ATSP for remaining sets
    vector<int> remaining_sets;
    if (!global_seq.empty()) remaining_sets = {global_seq.back()};
    unordered_set<int> in_seq_set(global_seq.begin(), global_seq.end());
    for (int i=0; i<(int)vps_set.size(); ++i)
    {
      if (in_seq_set.find(i) == in_seq_set.end()) remaining_sets.push_back(i);
    }

    if ((int)remaining_sets.size() < 3)
      global_seq.insert(global_seq.end(), remaining_sets.begin()+1, remaining_sets.end());
    else
    {
      int dim = remaining_sets.size();
      Eigen::MatrixXd cost_mat; cost_mat.resize(dim, dim); cost_mat.setZero();

      for (int i=1; i<dim; ++i)
        for (int j=1; j<dim; ++j)
        {
          if (i<=j) continue;
          Eigen::Vector3d c_i = vps_centroids[remaining_sets[i]];
          Eigen::Vector3d c_j = vps_centroids[remaining_sets[j]];

          double cost = 0.0;
          vector<Eigen::VectorXd> path;
          vector<bool> indi;
          Eigen::VectorXd p_i, p_j;
          p_i.resize(5); p_j.resize(5);
          p_i << c_i(0), c_i(1), c_i(2), 0.0, 0.0;
          p_j << c_j(0), c_j(1), c_j(2), 0.0, 0.0;

          double c_dist = (c_i - c_j).norm();
          if (c_dist < 3*this->cvx_range_)
          {
            this->findBridgePath(p_i, p_j, cost, path, indi, 2*astarSearchingRes, true);
          }
          else
          {
            cost = COST_INFINITY;
          }

          cost_mat(i,j) = cost;
          cost_mat(j,i) = cost;
        }
      
      Eigen::Vector3d c_first = vps_centroids[remaining_sets.front()];
      Eigen::VectorXd p_first; p_first.resize(5);
      p_first << c_first(0), c_first(1), c_first(2), 0.0, 0.0;
      for (int i=1; i<dim; ++i)
      {
        Eigen::Vector3d c_i = vps_centroids[remaining_sets[i]];

        double cost = 0.0;
        vector<Eigen::VectorXd> path;
        vector<bool> indi;
        Eigen::VectorXd c_i_vec; c_i_vec.resize(5);
        c_i_vec << c_i(0), c_i(1), c_i(2), 0.0, 0.0;

        this->findBridgePath(p_first, c_i_vec, cost, path, indi, 2*astarSearchingRes, true);
        
        cost_mat(0,i) = cost;
      }

      vector<int> results = solver_tools::ATSP(this->LocalFolder_, this->GlobalSolver_, cost_mat, this->LocalRuns_, this->precision_);

      for (int i=1; i<(int)results.size(); ++i)
        global_seq.push_back(remaining_sets[results[i]]);
    }

    auto t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms = t2 - t1;
    double time = (double)ms.count();
    ROS_INFO("\033[33m[GlobalConsistency] ATSP for remaining sets time = %lf ms.\033[32m", time);

    // * step 5: Obtain global path
    for (int i=0; i<(int)global_seq.size(); ++i)
    {
      global_path.push_back(vps_centroids[global_seq[i]]);
    }

    return;
  }

  /**
   * @brief Serial ATSP for first-reused_num sets, start-end-constrained TSP for the following sets.
   * @param global_seq      N-dim -> 0-th, 1-st, 2-nd, ... N-1-th
   * @param global_seq_path N-dim -> centroid_0, centroid_1, ... centroid_N
   */
  void HCSolver::decompGlobalPath(const Eigen::VectorXd& start_pose, const Eigen::Vector3d& start_vel, const vector<Eigen::VectorXd>& prior_remaining, const vector<Eigen::VectorXd>& vps, const vector<vector<int>>& vps_set, const vector<int>& global_seq, const vector<Eigen::Vector3d>& global_seq_path, const int& reused_num, vector<Eigen::VectorXd>& full_path, vector<bool>& full_indicators)
  {
    full_path.push_back(start_pose);
    full_indicators.push_back(false);
    
    vector<Eigen::VectorXd> sub_path;
    vector<bool> sub_indicators;

    // * step 1: serial ATSP for reused viewpoints sets
    Eigen::VectorXd sub_start_pose = start_pose;
    vector<Eigen::VectorXd> task_vps_pose = {sub_start_pose};
    for (int i=0; i<reused_num; ++i)
      for (auto idx:vps_set[global_seq[i]]) 
        task_vps_pose.push_back(vps[idx]);
    
    for (int i=0; i<(int)prior_remaining.size(); ++i)
      task_vps_pose.push_back(prior_remaining[i]);

    sub_path.clear();
    sub_indicators.clear();
    this->decompReusedPath(start_vel, task_vps_pose, sub_path, sub_indicators);
    full_path.insert(full_path.end(), sub_path.begin()+1, sub_path.end());
    full_indicators.insert(full_indicators.end(), sub_indicators.begin()+1, sub_indicators.end());

    double cur_length = 0.0;
    for (int i=0; i<(int)full_path.size()-1; ++i)
      cur_length += (full_path[i+1].head(3)-full_path[i].head(3)).norm();
    
    if (cur_length > this->global_range_)
      return;

    // * step 2: find start and end boundary & solve the constrained TSP for the following sets
    int following_num = global_seq.size() - reused_num;
    if (following_num > 0)
    {
      double bridge_cost = 0.0;
      vector<Eigen::VectorXd> brigde_path;
      vector<bool> bridge_indicators;

      vector<Eigen::Vector3d> ref_pts = {full_path.back().head(3)};
      for (int i=reused_num; i<(int)global_seq.size(); ++i) ref_pts.push_back(global_seq_path[i]);
      
      Eigen::VectorXd last_end_pose;
      int start_idx = -1, end_idx = -1;
      for (int i=reused_num; i<(int)global_seq.size(); ++i)
      {
        last_end_pose = full_path.back();

        int ref_idx = i - reused_num;
        start_idx = -1, end_idx = -1;

        vector<Eigen::VectorXd> task_vps_pose;
        for (auto idx:vps_set[global_seq[i]]) task_vps_pose.push_back(vps[idx]); 

        if ((int)task_vps_pose.size() == 1)
        {
          brigde_path.clear();
          bridge_indicators.clear();
          this->findBridgePath(last_end_pose, task_vps_pose.front(), bridge_cost, brigde_path, bridge_indicators, astarSearchingRes, false);
          full_path.insert(full_path.end(), brigde_path.begin(), brigde_path.end());
          full_indicators.insert(full_indicators.end(), bridge_indicators.begin(), bridge_indicators.end());
          full_path.push_back(task_vps_pose.front());
          full_indicators.push_back(false);

          continue;
        }

        // determine start viewpoint
        start_idx = this->findBoundary(ref_pts[ref_idx], ref_pts[ref_idx+1], task_vps_pose, -1);

        // determine end viewpoint
        if (i != (int)global_seq.size()-1) end_idx = this->findBoundary(ref_pts[ref_idx+1], ref_pts[ref_idx+2], task_vps_pose, start_idx);

        // update viewpoints set
        vector<Eigen::VectorXd> updated_task_vps_pose = {task_vps_pose[start_idx]};
        for (int j=0; j<(int)task_vps_pose.size(); ++j)
        {
          if (j != start_idx && j != end_idx) updated_task_vps_pose.push_back(task_vps_pose[j]);
        }
        if (i != (int)global_seq.size()-1)  updated_task_vps_pose.push_back(task_vps_pose[end_idx]);

        // add bridge path
        brigde_path.clear();
        bridge_indicators.clear();
        this->findBridgePath(last_end_pose, updated_task_vps_pose.front(), bridge_cost, brigde_path, bridge_indicators, astarSearchingRes, false);
        full_path.insert(full_path.end(), brigde_path.begin(), brigde_path.end());
        full_indicators.insert(full_indicators.end(), bridge_indicators.begin(), bridge_indicators.end());

        // solve the constrained TSP
        sub_path.clear();
        sub_indicators.clear();
        this->decompFollowPath(updated_task_vps_pose, sub_path, sub_indicators);
        full_path.insert(full_path.end(), sub_path.begin(), sub_path.end());
        full_indicators.insert(full_indicators.end(), sub_indicators.begin(), sub_indicators.end());
      
        double cur_length = 0.0;
        for (int i=0; i<(int)full_path.size()-1; ++i)
          cur_length += (full_path[i+1].head(3)-full_path[i].head(3)).norm();
        
        if (cur_length > this->global_range_)
          return;
      }
    }

    return;
  }

  void HCSolver::decompReusedPath(Eigen::Vector3d start_vel, vector<Eigen::VectorXd>& vps_pose, vector<Eigen::VectorXd>& sub_path, vector<bool>& sub_indicators)
  {
    double astar_cost = 0.0;
    vector<Eigen::VectorXd> updated_ij_path;
    vector<bool> updated_ij_indicators;
    
    if ((int)vps_pose.size() < 3)
    {
      if ((int)vps_pose.size() == 1)
      {
        sub_path = vps_pose;
        sub_indicators.push_back(false);
      }
      else
      {
        updated_ij_path.clear();
        updated_ij_indicators.clear();

        this->findBridgePath(vps_pose.front(), vps_pose.back(), astar_cost, updated_ij_path, updated_ij_indicators, astarSearchingRes, false);

        sub_path.push_back(vps_pose.front());
        sub_path.insert(sub_path.end(), updated_ij_path.begin(), updated_ij_path.end());
        sub_path.push_back(vps_pose.back());

        sub_indicators.push_back(false);
        sub_indicators.insert(sub_indicators.end(), updated_ij_indicators.begin(), updated_ij_indicators.end());
        sub_indicators.push_back(false);
      }

      return;
    }
    
    int dim = vps_pose.size();
    Eigen::MatrixXd cost_mat; cost_mat.resize(dim, dim); cost_mat.setZero();

    map<Pair, vector<Eigen::VectorXd>> ij_inter_paths;
    map<Pair, vector<bool>> ij_inter_indicators;

    // * inter cost
    for (int i=1; i<dim; ++i)
      for (int j=1; j<dim; ++j)
      {
        if (i<=j) continue;
        
        Eigen::VectorXd v_i = vps_pose[i];
        Eigen::VectorXd v_j = vps_pose[j];

        updated_ij_path.clear();
        updated_ij_indicators.clear();
        this->findBridgePath(v_i, v_j, astar_cost, updated_ij_path, updated_ij_indicators, astarSearchingRes, false);

        // astar_cost += 10.0 * (v_i(2) + v_j(2));

        cost_mat(i,j) = astar_cost;
        cost_mat(j,i) = astar_cost;

        ij_inter_paths[Pair(i,j)] = updated_ij_path;
        ij_inter_indicators[Pair(i,j)] = updated_ij_indicators;
        ij_inter_paths[Pair(j,i)] = updated_ij_path;
        ij_inter_indicators[Pair(j,i)] = updated_ij_indicators;
      }
    
    // * intra cost
    Eigen::VectorXd start = vps_pose.front();
    for (int i=1; i<dim; ++i)
    {
      Eigen::VectorXd vp = vps_pose[i];

      updated_ij_path.clear();
      updated_ij_indicators.clear();
      this->findRealBridgePath(start, vp, astar_cost, updated_ij_path, updated_ij_indicators);

      Eigen::Vector3d dir_vec = (vp.head(3) - start.head(3)).normalized();
      double direction_cost = 2.0 * (1.0 - dir_vec.dot(start_vel.normalized()));

      cost_mat(0,i) = astar_cost + direction_cost;

      ij_inter_paths[Pair(0,i)] = updated_ij_path;
      ij_inter_indicators[Pair(0,i)] = updated_ij_indicators;
      ij_inter_paths[Pair(i,0)] = updated_ij_path;
      ij_inter_indicators[Pair(i,0)] = updated_ij_indicators;
    }

    // * solving
    vector<int> results = solver_tools::ATSP(this->LocalFolder_, this->GlobalSolver_, cost_mat, this->LocalRuns_, this->precision_);
    for (int i=0; i<(int)results.size()-1; ++i)
    {
      sub_path.push_back(vps_pose[results[i]]);
      sub_indicators.push_back(false);

      Pair ij = Pair(results[i], results[i+1]);

      if (ij_inter_paths.find(ij) != ij_inter_paths.end()) sub_path.insert(sub_path.end(), ij_inter_paths[ij].begin(), ij_inter_paths[ij].end());
      if (ij_inter_indicators.find(ij) != ij_inter_indicators.end()) sub_indicators.insert(sub_indicators.end(), ij_inter_indicators[ij].begin(), ij_inter_indicators[ij].end());
    }
    sub_path.push_back(vps_pose[results.back()]);
    sub_indicators.push_back(false);

    return;
  }

  void HCSolver::decompFollowPath(vector<Eigen::VectorXd>& vps_pose, vector<Eigen::VectorXd>& sub_path, vector<bool>& sub_indicators)
  {
    if ((int)vps_pose.size() < 3)
    {
      sub_path = vps_pose;
      vector<bool> temp_indicators(vps_pose.size(), true);
      sub_indicators = temp_indicators;

      return;
    }
    else
    {
      int dim = vps_pose.size();
      Eigen::MatrixXd cost_mat; cost_mat.resize(dim, dim); cost_mat.setZero();

      for (int i=1; i<dim; ++i)
        for (int j=1; j<dim; ++j)
        {
          if (i<=j) continue;
          double cost = (vps_pose[i].head(3)-vps_pose[j].head(3)).norm();

          // cost += 10.0 * (vps_pose[i](2) + vps_pose[j](2));

          cost_mat(i,j) = cost;
          cost_mat(j,i) = cost;
        }
      
      // * start contraint
      Eigen::VectorXd start = vps_pose.front();
      for (int i=1; i<dim; ++i)
      {
        Eigen::VectorXd v_i = vps_pose[i];
        double cost = (start.head(3)-v_i.head(3)).norm();

        // cost += 10.0 * (start(2) + v_i(2));

        cost_mat(0,i) = cost;
      }

      // * end constraint
      // for (int i=1; i<dim-1; ++i)
      //   cost_mat(dim-1, i) = COST_INFINITY;

      // * solving
      vector<int> results = solver_tools::ATSP(this->LocalFolder_, this->GlobalSolver_, cost_mat, this->LocalRuns_, this->precision_);
      for (int i=0; i<(int)results.size(); ++i)
      {
        sub_path.push_back(vps_pose[results[i]]);
        sub_indicators.push_back(false);
      }
    }

    return;
  }

  void HCSolver::groundFindPath(Eigen::Vector3d start_vel, vector<Eigen::VectorXd>& vps_pose, vector<Eigen::VectorXd>& sub_path, vector<bool>& sub_indicators)
  {
    if ((int)vps_pose.size() < 3)
    {
      sub_path = vps_pose;
      sub_indicators = vector<bool>(vps_pose.size(), false);

      return;
    }
    
    int dim = vps_pose.size();
    Eigen::MatrixXd cost_mat; cost_mat.resize(dim, dim); cost_mat.setZero();

    // * inter cost
    for (int i=1; i<dim; ++i)
      for (int j=1; j<dim; ++j)
      {
        if (i<=j) continue;
        
        Eigen::VectorXd v_i = vps_pose[i];
        Eigen::VectorXd v_j = vps_pose[j];

        double pos_cost = (v_i.head(3)-v_j.head(3)).norm() / this->vm_;
        double yaw_cost = 2.0*min(abs(v_i(4)-v_j(4)), 2*M_PI-abs(v_i(4)-v_j(4))) / this->yd_;

        double cost = max(pos_cost, yaw_cost);
        // double cost = pos_cost;

        cost_mat(i,j) = cost;
        cost_mat(j,i) = cost;
      }
    
    // * intra cost
    Eigen::VectorXd start = vps_pose.front();
    for (int i=1; i<dim; ++i)
    {
      Eigen::VectorXd vp = vps_pose[i];

      // double dynamic_cost = path_tools::estPathTime(start_vel, start.head(3), vp.head(3), this->vm_, this->amean_);

      // cost_mat(0,i) = 20.0*dynamic_cost;

      double pos_cost = (start.head(3)-vp.head(3)).norm() / this->vm_;
      // double yaw_cost = 2.0*min(abs(start(4)-vp(4)), 2*M_PI-abs(start(4)-vp(4))) / this->yd_;

      // double cost = max(pos_cost, yaw_cost);
      double cost = pos_cost;

      cost_mat(0,i) = cost;
    }

    // * solving
    vector<int> results = solver_tools::ATSP(this->LocalFolder_, this->GlobalSolver_, cost_mat, this->LocalRuns_, this->precision_);

    for (int i=0; i<(int)results.size(); ++i)
    {
      sub_path.push_back(vps_pose[results[i]]);
      sub_indicators.push_back(false);
    }
    
    return;
  }

  bool HCSolver::findClosestBoundaryPoint(const Eigen::Vector3d& pt, const Eigen::Vector3d& nr, Eigen::Vector3d& closest_pt, double vp_box_min_x, double vp_box_max_x, double vp_box_min_y, double vp_box_max_y)
  {
    double t_min = 1e6;
    if (nr(0) != 0) 
    {
      double t_x_min = (vp_box_min_x - pt(0)) / nr(0);
      if (t_x_min > 0) t_min = std::min(t_min, t_x_min);

      double t_x_max = (vp_box_max_x - pt(0)) / nr(0);
      if (t_x_max > 0) t_min = std::min(t_min, t_x_max);
    }

    if (nr(1) != 0) 
    {
      double t_y_min = (vp_box_min_y - pt(1)) / nr(1);
      if (t_y_min > 0) t_min = std::min(t_min, t_y_min);

      double t_y_max = (vp_box_max_y - pt(1)) / nr(1);
      if (t_y_max > 0) t_min = std::min(t_min, t_y_max);
    }

    if (t_min == 1e6)
    {
      return false;
    }

    closest_pt = pt + t_min * nr;

    return true;
  }

  // ! ------------------------------------- Utils -------------------------------------
  Eigen::MatrixXd HCSolver::GlobalCostMat(Eigen::Vector3d& start_, vector<Eigen::Vector3d>& targets)
  {
    vector<Eigen::Vector3d> total_;
    total_.push_back(start_);
    total_.insert(total_.begin()+1, targets.begin(), targets.end());
    Eigen::Vector3d vec_i, vec_j;
    double cost_;

    int dim = (int)total_.size();
    Eigen::MatrixXd costmat_;
    costmat_.resize(dim, dim);
    costmat_.setZero();

    for (int i=1; i<dim; ++i)
      for (int j=1; j<dim; ++j)
      {
        vec_i = total_[i];
        vec_j = total_[j];
        cost_ = (vec_i-vec_j).norm();
        costmat_(i,j) = cost_;
      }
    
    for (int k=0; k<dim; ++k)
    {
      vec_i = total_[0];
      vec_j = total_[k];
      cost_ = (vec_i-vec_j).norm();
      costmat_(0,k) = cost_;
    }

    return costmat_;
  }

  void HCSolver::GlobalParWrite()
  {
    ofstream par_file(GlobalPar_);
    par_file << "PROBLEM_FILE = " << GlobalProF_ << "\n";
    par_file << "GAIN23 = NO\n";
    par_file << "OUTPUT_TOUR_FILE =" << GlobalResult_ << "\n";
    par_file << "RUNS = " << to_string(GlobalRuns_) << "\n";
    par_file.close();
  }

  void HCSolver::GlobalProblemWrite(Eigen::MatrixXd& costMat)
  {
    const int dimension = costMat.rows();
    ofstream prob_file(GlobalProF_);
    string prob_spec = "NAME : global\nTYPE : ATSP\nDIMENSION : " + to_string(dimension) +
      "\nEDGE_WEIGHT_TYPE : "
      "EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n";
    prob_file << prob_spec;

    for (int i=0; i<dimension; ++i)
    {
      for (int j=0; j<dimension; ++j)
      {
        int int_cost = costMat(i,j)*precision_;
        prob_file << to_string(int_cost) << " ";
      }
      prob_file << "\n";
    }

    prob_file << "EOF";
    prob_file.close();
  }

  vector<int> HCSolver::GlobalResultsRead()
  {
    vector<int> results;

    ifstream res_file(GlobalResult_);
    string res;
    while (getline(res_file, res)) 
      if (res.compare("TOUR_SECTION") == 0) break;
    
    while (getline(res_file, res)) 
    {
      int id = stoi(res);
      if (id == 1)
        continue;
      if (id == -1) break;
      results.push_back(id - 2);
    }
    res_file.close();

    return results;
  }

  int HCSolver::FindSphereNearestPoint(Eigen::Vector3d& spherePtA, Eigen::Vector3d& spherePtB, vector<Eigen::VectorXd>& vps)
  {
    int nearestID = -1;
    Eigen::Vector3d vp_pos_;
    double dist_min_ = MAX;

    for (int i=0; i<(int)vps.size(); ++i) 
    {
      vp_pos_(0) = vps[i](0); vp_pos_(1) = vps[i](1); vp_pos_(2) = vps[i](2); 
      // Compute the distance between the point and the line segment AB
      Eigen::Vector3d AP = vp_pos_ - spherePtA;
      Eigen::Vector3d BP = vp_pos_ - spherePtB;

      double within_dist_ = AP.norm() + BP.norm();
      if (within_dist_ < dist_min_)
      {
        dist_min_ = within_dist_;
        nearestID = i;
      }
    }

    return nearestID;
  }
  /* the path from p1 to p2 */
  double HCSolver::search_Path(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, vector<Eigen::Vector3d>& path)
  {
    /* try to connect two points with straight line */
    Eigen::MatrixXd ori_path;
    bool safe = true;

    Eigen::Vector3i idx;
    solve_raycaster_->input(p1, p2);
    while (solve_raycaster_->nextId(idx)) 
    {
      if (!solve_map_->isInMap_hc(idx))
      {
        safe = false;
        break;
      }

      if (solve_map_->hcmd_->occupancy_inflate_buffer_hc_[solve_map_->toAddress_hc(idx)] == 1 || solve_map_->hcmd_->occupancy_buffer_internal_[solve_map_->toAddress_hc(idx)] == 1) 
      {
        safe = false;
        break;
      }
      
      if (safe == false)
        break;
    }

    if (safe) 
    {
      path = { p1, p2 };

      phy_dist = (p1-p2).norm();

      return (p1-p2).norm();
    }
    /* Search a path using decreasing resolution */
    vector<double> res = { astarSearchingRes };
    vector<double> length_candidate = {};
    vector<vector<Eigen::Vector3d>> path_candidate = {};
    int success = 0;
    for (int k = 0; k < (int)res.size(); ++k) 
    {
      astar_->reset();
      astar_->setResolution(res[k]);

      double this_time_length = 0.0;
      vector<Eigen::Vector3d> this_time_path = {};
      if (astar_->hc_search(p1, p2) == Astar::REACH_END) 
      {
        this_time_path = astar_->getPath();
        shortenPath(this_time_path);
        this_time_length = astar_->pathLength(this_time_path);

        success++;
      }
      else
      {
        this_time_length = COST_INFINITY;
        this_time_path = {};
      }

      length_candidate.push_back(this_time_length);
      path_candidate.push_back(this_time_path);
    }
    
    if (success > 0)
    {
      auto min_it = min_element(length_candidate.begin(), length_candidate.end());
      int min_index = distance(length_candidate.begin(), min_it);

      phy_dist = length_candidate[min_index];
      path = path_candidate[min_index];

      return length_candidate[min_index];
    }

    /* Use Astar early termination cost as an estimate */
    path = { p1, p2 };
    phy_dist = COST_INFINITY;
    return COST_INFINITY;
  }

  void HCSolver::shortenPath(vector<Eigen::Vector3d>& path)
  {
    if (path.empty()) 
    {
      ROS_ERROR("Empty path to shorten");
      return;
    }
    // Shorten the tour, only critical intermediate points are reserved.
    const double dist_thresh = 5.0;
    vector<Eigen::Vector3d> short_tour = { path.front() };
    for (int i = 1; i < (int)path.size() - 1; ++i) 
    {
      if ((path[i] - short_tour.back()).norm() > dist_thresh)
        short_tour.push_back(path[i]);
      else 
      {
        // Add waypoints to shorten path only to avoid collision
        solve_raycaster_->input(short_tour.back(), path[i + 1]);
        Eigen::Vector3i idx;
        while (solve_raycaster_->nextId(idx)) 
        {
          if (!solve_map_->isInMap_hc(idx))
          {
            short_tour.push_back(path[i]);
            break;
          }
          
          if (solve_map_->hcmd_->occupancy_buffer_hc_[solve_map_->toAddress_hc(idx)] == 1 || solve_map_->hcmd_->occupancy_buffer_internal_[solve_map_->toAddress_hc(idx)] == 1) 
          {
            short_tour.push_back(path[i]);
            break;
          }
        }
      }
    }
    if ((path.back() - short_tour.back()).norm() > 1e-3) 
      short_tour.push_back(path.back());

    path = short_tour;
  }

  /* for the platform without tripod head */
  double HCSolver::compute_Cost(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const double& y1, const double& y2, vector<Eigen::Vector3d>& path)
  {
    /* cost of position change */
    double pos_dist = search_Path(p1, p2, path);
    double pos_time = -1.0;

    pos_time = pos_dist/vm_;
    pos_time = min(pos_time, COST_INFINITY);
    
    return pos_time;
  }
  /* uniform acceleration time cost model */
  double HCSolver::computeTimeCost(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const double& y1, const double& y2, vector<Eigen::Vector3d>& path)
  {
    double pos_dist = search_Path(p1, p2, path);
    double pos_cost = 0.0;
    if ((int)path.size() > 2)
    {
      pos_cost += (path[1]-path[0]).norm()/vm_;
      for (int i=0; i<(int)path.size()-1; ++i)
        pos_cost += pathTimeUniAcc(path[i-1], path[i], path[i+1]);
    }
    else
    {
      pos_cost = pos_dist/vm_;
    }

    double angle_diff = fabs(y2 - y1);
    angle_diff = min(angle_diff, 2 * M_PI - angle_diff);
    double angle_cost = angle_diff/yd_;

     return max(pos_cost, angle_cost);
  }
  /* given A, B, C three waypoints, uniform acceleration motion model to calculate time cost of BC */
  double HCSolver::pathTimeUniAcc(Eigen::Vector3d& pred, Eigen::Vector3d& cur, Eigen::Vector3d& suc)
  {
    // 1. calculate the angle between two segments
    double theta = acos((pred-cur).normalized().dot((suc-cur).normalized()));
    // 2. calculate the upper accelerating distance
    double s = pow(vm_,2)*sin(theta)*cos(theta/2)/am_;
    double l = (cur-suc).norm();
    // 3. calculate time cost
    double time = 0.0;
    if (s < l)
    {
      time = l/vm_ + 2*vm_*pow(sin(theta/2),3)/(am_);
    }
    else
    {
      double temp = pow(vm_,2)*pow(cos(theta),2) + 2*am_*sin(theta/2)*l;
      time = (pow(temp, 0.5)-vm_*cos(theta))/(am_*sin(theta/2));
    }

    double flightCost = time;

    return flightCost;
  }

  /* for the platform with tripod head */
  double HCSolver::compute_Cost_tripod_head(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const double& y1, const double& y2, vector<Eigen::Vector3d>& path)
  {
    /* cost of position change */
    double pos_dist = search_Path(p1, p2, path);
    double pos_time = -1.0;

    pos_time = pos_dist/vm_;
    
    return pos_time;
  }

  double HCSolver::computeTimeCost_tripod_head(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const double& y1, const double& y2, vector<Eigen::Vector3d>& path)
  {
    double pos_dist = search_Path(p1, p2, path);
    double pos_cost = 0.0;
    if ((int)path.size() > 2)
    {
      pos_cost += (path[1]-path[0]).norm()/vm_;
      for (int i=0; i<(int)path.size()-1; ++i)
        pos_cost += pathTimeUniAcc(path[i-1], path[i], path[i+1]);
    }
    else
    {
      pos_cost = pos_dist/vm_;
    }

    return pos_cost;
  }

  void HCSolver::LocalFindPath(vector<Eigen::VectorXd> vps, vector<int> boundary_, int sub_id_, bool turn)
  {
    int start_id = -1, end_id = -1;
    vector<int> local_id_init_;
    bool bilateral = false;

    if (boundary_.size() == 2)
    {
      start_id = boundary_[0];
      end_id = boundary_[1];
      local_id_init_.push_back(start_id);
      for (int id=0; id<(int)vps.size(); ++id)
      {
        if (id != start_id && id != end_id)
          local_id_init_.push_back(id);
      }
      local_id_init_.push_back(end_id);
      bilateral = true;
    }
    else if (boundary_.size() == 1)
    {
      start_id = boundary_[0];
      local_id_init_.push_back(start_id);
      for (int id=0; id<(int)vps.size(); ++id)
      {
        if (id != start_id)
          local_id_init_.push_back(id);
      }
      bilateral = false;
    }
    /* local initial idxs conditioned on boundary */
    local_sub_init_idx_[sub_id_] = local_id_init_;
    
    int dim = (int)local_id_init_.size();
    Eigen::MatrixXd costmat_, phymat_;
    costmat_ = Eigen::MatrixXd::Zero(dim, dim);
    phymat_ = Eigen::MatrixXd::Zero(dim, dim);
    
    Eigen::Vector3d pos_i, pos_j;
    double yaw_i, yaw_j;
    double pitch_i, pitch_j;
    double cost_ij_;
    vector<Eigen::Vector3d> ij_path_;
    vector<Eigen::VectorXd> updated_ij_path_;
    vector<double> ij_pos_;

    for (int i=1; i<(int)local_id_init_.size(); ++i)
      for (int j=1; j<(int)local_id_init_.size(); ++j)
      {
        vector<Eigen::Vector3d>().swap(ij_path_);
        vector<Eigen::VectorXd>().swap(updated_ij_path_);
        pos_i = vps[local_id_init_[i]].head(3);
        pos_j = vps[local_id_init_[j]].head(3);
        pitch_i = vps[local_id_init_[i]](3);
        pitch_j = vps[local_id_init_[j]](3);
        yaw_i = vps[local_id_init_[i]].tail(1)(0);
        yaw_j = vps[local_id_init_[j]].tail(1)(0);

        double p_diff = min(abs(pitch_i-pitch_j), 2*M_PI-abs(pitch_i-pitch_j));
        double y_diff = min(abs(yaw_i-yaw_j), 2*M_PI-abs(yaw_i-yaw_j));
        double a1, a2;
        if (p_diff > y_diff)
        {
          a1 = pitch_i;
          a2 = pitch_j;
        }
        else
        {
          a1 = yaw_i;
          a2 = yaw_j;
        }
        
        if ((pos_i - pos_j).norm() < 20.0)
        {
        if (tripod_head_trigger_ == true)
        {
          if (turn == true)
            cost_ij_ = computeTimeCost_tripod_head(pos_i, pos_j, a1, a2, ij_path_);
          else
            cost_ij_ = compute_Cost_tripod_head(pos_i, pos_j, a1, a2, ij_path_);
        }
        else
        {
          if (turn == true)
            cost_ij_ = computeTimeCost(pos_i, pos_j, a1, a2, ij_path_);
          else
            cost_ij_ = compute_Cost(pos_i, pos_j, a1, a2, ij_path_);
        }
        }
        else
        {
          cost_ij_ = COST_INFINITY;
        }
        vector<int> path_index_ = {sub_id_, local_id_init_[i], local_id_init_[j]};
        costmat_(i,j) = cost_ij_;
        phymat_(i,j) = phy_dist;
        ij_pos_ = {pos_i(0), pos_i(1), pos_i(2), pos_j(0), pos_j(1), pos_j(2)};
        allAstarCost_[ij_pos_] = cost_ij_;
        if ((int)ij_path_.size() > 2)
        {
          AngleInterpolation(vps[local_id_init_[i]], vps[local_id_init_[j]], ij_path_, updated_ij_path_);
        }
        else
          updated_ij_path_ = {vps[local_id_init_[i]], vps[local_id_init_[j]]};
        path_waypts_[path_index_] = updated_ij_path_;
        allAstarPath_[ij_pos_] = updated_ij_path_;
      }
    /* start constraint */
    for (int k=0; k<dim; ++k)
    {
      vector<Eigen::Vector3d>().swap(ij_path_);
      vector<Eigen::VectorXd>().swap(updated_ij_path_);

      pos_i = vps[local_id_init_[0]].head(3);
      pos_j = vps[local_id_init_[k]].head(3);
      pitch_i = vps[local_id_init_[0]](3);
      pitch_j = vps[local_id_init_[k]](3);
      yaw_i = vps[local_id_init_[0]].tail(1)(0);
      yaw_j = vps[local_id_init_[k]].tail(1)(0);

      double p_diff = min(abs(pitch_i-pitch_j), 2*M_PI-abs(pitch_i-pitch_j));
      double y_diff = min(abs(yaw_i-yaw_j), 2*M_PI-abs(yaw_i-yaw_j));
      double a1, a2;
      if (p_diff > y_diff)
      {
        a1 = pitch_i;
        a2 = pitch_j;
      }
      else
      {
        a1 = yaw_i;
        a2 = yaw_j;
      }

      if ((pos_i - pos_j).norm() < 20.0)
      {
      if (tripod_head_trigger_ == true)
      {
        if (turn == true)
          cost_ij_ = computeTimeCost_tripod_head(pos_i, pos_j, a1, a2, ij_path_);
        else
          cost_ij_ = compute_Cost_tripod_head(pos_i, pos_j, a1, a2, ij_path_);
      }
      else
      {
        if (turn == true)
          cost_ij_ = computeTimeCost(pos_i, pos_j, a1, a2, ij_path_);
        else
          cost_ij_ = compute_Cost(pos_i, pos_j, a1, a2, ij_path_);
      }
      }
      else
      {
        cost_ij_ = COST_INFINITY;
      }

      vector<int> path_index_ = {sub_id_, local_id_init_[0], local_id_init_[k]};
      costmat_(0,k) = cost_ij_;
      phymat_(0,k) = phy_dist;
      ij_pos_ = {pos_i(0), pos_i(1), pos_i(2), pos_j(0), pos_j(1), pos_j(2)};
      allAstarCost_[ij_pos_] = cost_ij_;
      if ((int)ij_path_.size() > 2)
        AngleInterpolation(vps[local_id_init_[0]], vps[local_id_init_[k]], ij_path_, updated_ij_path_);
      else
        updated_ij_path_ = {vps[local_id_init_[0]], vps[local_id_init_[k]]};
      path_waypts_[path_index_] = updated_ij_path_;
      allAstarPath_[ij_pos_] = updated_ij_path_;
    }
    /* end constraint */
    if (bilateral == true)
    {
      for (int m=1; m<dim-1; ++m)
      {
        costmat_(dim-1, m) = COST_INFINITY;
        phymat_(dim-1, m) = COST_INFINITY;
      }
    }

    local_sub_costmat_[sub_id_] = costmat_;
    local_sub_phymat_[sub_id_] = phymat_;
  }

  void HCSolver::LocalPathFinder(vector<Eigen::VectorXd> vps, vector<int> boundary_, int sub_id_)
  {
    Eigen::MatrixXd phyMat = local_sub_phymat_.find(sub_id_)->second;

    if ((int)vps.size() < 3 || (int)phyMat.rows() < 3)
    {
      vector<Eigen::VectorXd> viewpointsSpec;
      vector<vector<Eigen::VectorXd>> waypointsSpec;

      viewpointsSpec.push_back(vps[boundary_[0]]);
      if ((int)vps.size() == 2)
      {
        waypointsSpec.push_back({});
        if (vps[0](0) == vps[boundary_[0]](0) && vps[0](1) == vps[boundary_[0]](1) && vps[0](2) == vps[boundary_[0]](2))
          viewpointsSpec.push_back(vps[1]);
        else
          viewpointsSpec.push_back(vps[0]);
      }
      else
      {
        waypointsSpec.push_back({});
        viewpointsSpec.push_back(vps[0]);
      }

      {
        std::unique_lock<std::mutex> lock(path_finder_mtx);
        local_sub_path_viewpts_[sub_id_] = viewpointsSpec;
        local_sub_path_waypts_[sub_id_] = waypointsSpec;
        lock.unlock();
      }
    }
    else
    {
    vector<int> init_id_ = local_sub_init_idx_.find(sub_id_)->second;
    /* prepare related files */
    string LocalParF_ = LocalFolder_+"/sub_"+to_string(sub_id_)+".par";
    string LocalProbF_ = LocalFolder_+"/sub_"+to_string(sub_id_)+".tsp";
    string LocalResultF_ = LocalFolder_+"/sub_"+to_string(sub_id_)+"_solution.txt";
    
    vector<Eigen::VectorXd> path_viewpointsPrior;
    vector<vector<Eigen::VectorXd>> path_waypointsPrior;
    /* preparation end */

    /* write parameter file */
    ofstream par_file(LocalParF_);
    par_file << "PROBLEM_FILE = " << LocalProbF_ << "\n";
    par_file << "GAIN23 = NO\n";
    par_file << "OUTPUT_TOUR_FILE =" << LocalResultF_ << "\n";
    par_file << "RUNS = " << to_string(LocalRuns_) << "\n";
    par_file.close();
    /* write parameter end */

    /* write problem file */
    Eigen::MatrixXd costMat = local_sub_costmat_.find(sub_id_)->second;
    ostringstream v_s, a_s;
    v_s << fixed << setprecision(1) << vm_;
    a_s << fixed << setprecision(1) << am_;
    const int dimension = costMat.rows();
    ofstream prob_file(LocalProbF_);
    string prob_spec;

    prob_spec = "NAME : local_" + to_string(sub_id_) +
    "\nTYPE : ATSP\nDIMENSION : " + to_string(dimension) +
    "\nEDGE_WEIGHT_TYPE : "
    "EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX" + 
    "\nEDGE_WEIGHT_SECTION\n";

    prob_file << prob_spec;
    
    for (int i=0; i<dimension; ++i)
    {
      for (int j=0; j<dimension; ++j)
      {
        int int_cost = costMat(i,j)*precision_;
        prob_file << to_string(int_cost) << " ";
      }
      prob_file << "\n";
    }

    prob_file << "EOF";
    prob_file.close();
    /* write problem end */

    /* TSP solving... based on the initial idxs */
    string local_command_;
    local_command_ = "cd " + GlobalSolver_ + " && ./LKH " + LocalParF_;
    const char* local_charPtr = local_command_.c_str();
    local_system_back_ = system(local_charPtr);
    /* TSP solved */

    /* read results */
    vector<int> results;
    ifstream res_file(LocalResultF_);
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
    // viewpt_0, [waypts_0], viewpt_1, [waypts_1], viewpt_2, ...
    vector<Eigen::VectorXd> path_viewpoints_;
    vector<vector<Eigen::VectorXd>> path_waypoints_;

    // vector<Eigen::VectorXd> full;

    for (int i=0; i<(int)results.size(); ++i)
    {
      
      // full.push_back(vps[init_id_[results[i]]]);

      path_viewpoints_.push_back(vps[init_id_[results[i]]]);
      if (i < (int)results.size()-1)
      {
        vector<int> retrieval_ = {sub_id_, init_id_[results[i]], init_id_[results[i+1]]};
        vector<Eigen::VectorXd> inter_pts_;
        if ((int)path_waypts_.find(retrieval_)->second.size()>2)
        {
          for (int m=1; m<(int)path_waypts_.find(retrieval_)->second.size()-1; ++m)
            inter_pts_.push_back(path_waypts_.find(retrieval_)->second[m]);
          // full.insert(full.end(), inter_pts_.begin(), inter_pts_.end());
        }
        else
          inter_pts_ = {};
        path_waypoints_.push_back(inter_pts_);
      }
    }

    // double length = 0.0;
    // for (int i=0; i<(int)full.size()-1; ++i)
    //   length += (full[i]-full[i+1]).norm();
    // cout << "Subspace " << sub_id_ << " length: " << length << endl;
    
    /* read results end */

    /* record results --> Prior as LowerBound*/
    vector<Eigen::VectorXd> viewpointsFinal;
    vector<vector<Eigen::VectorXd>> waypointsFinal;

    viewpointsFinal = path_viewpoints_;
    waypointsFinal = path_waypoints_;

    {
      std::unique_lock<std::mutex> lock(path_finder_mtx);
      local_sub_path_viewpts_[sub_id_] = path_viewpoints_;
      local_sub_path_waypts_[sub_id_] = path_waypoints_;
      lock.unlock();
    }
    /* record results end */
    }
    
    con_var.notify_all();
  }  
  /* 
  INPUT : [START, WAYPOINTS, END], size_t : N or [WAYPOINTS], size_t : N-2
  OUTPUT : [UPDATED WAYPOINTS], size_t : N-2 
  */
  void HCSolver::AngleInterpolation(Eigen::VectorXd& start, Eigen::VectorXd& end, vector<Eigen::Vector3d>& waypts_, vector<Eigen::VectorXd>& updates_waypts_)
  {
    int dof = start.size(), angle = dof-3;
    double dist_gap = 1e-2;
    double whole_dist = 0.0;
    vector<Eigen::VectorXd> effect_vps_;
    vector<Eigen::VectorXd> seg_posi_;
    double yaw_start = 0.0, yaw_end = 0.0, pitch_start = 0.0, pitch_end = 0.0;
    int cal_flag_yaw = 0, cal_flag_pitch = 0;
    int cal_dir_yaw = 0, cal_dir_pitch = 0;

    if (angle == 1)
    {
      yaw_start = start(3)*180.0/M_PI;
      yaw_end = end(3)*180.0/M_PI;
      cal_flag_yaw = yaw_end - yaw_start > 0? 1:-1;
      cal_dir_yaw = abs(yaw_end - yaw_start)>180.0? -1:1;
    }
    else
    {
      yaw_start = start(4)*180.0/M_PI;
      yaw_end = end(4)*180.0/M_PI;
      cal_flag_yaw = yaw_end - yaw_start > 0? 1:-1;
      cal_dir_yaw = abs(yaw_end - yaw_start)>180.0? -1:1;
      pitch_start = start(3)*180.0/M_PI;
      pitch_end = end(3)*180.0/M_PI;
      cal_flag_pitch = ((pitch_end - pitch_start) > 0)? 1:-1;
      cal_dir_pitch = abs(pitch_end - pitch_start)>180.0? -1:1;
    }
    
    effect_vps_.push_back(start);
    for (auto x:waypts_)
    {
      if ((x-start.head(3)).norm() > dist_gap && (x-end.head(3)).norm() > dist_gap)
      {
        Eigen::VectorXd aug_x; aug_x.resize(dof);
        aug_x(0) = x(0); aug_x(1) = x(1); aug_x(2) = x(2);
        effect_vps_.push_back(aug_x);
      }
    }
    effect_vps_.push_back(end);

    seg_posi_.push_back(start);
    seg_posi_.insert(seg_posi_.end(), effect_vps_.begin(), effect_vps_.end());
    seg_posi_.push_back(end);
    for (int i=0; i<(int)seg_posi_.size()-1; ++i)
      whole_dist += (seg_posi_[i+1].head(3)-seg_posi_[i].head(3)).norm();
    
    if ((int)effect_vps_.size() > 0)
    {
      double yaw_gap = abs(yaw_end - yaw_start)>180.0? (360.0-abs(yaw_end - yaw_start)):abs(yaw_end - yaw_start);
      if (angle == 1)
      {
        for (int i=1; i<(int)effect_vps_.size()-1; ++i)
        {
          double e_dist = (effect_vps_[i].head(3)-start.head(3)).norm();
          effect_vps_[i](3) = (yaw_start + cal_dir_yaw*cal_flag_yaw*yaw_gap*e_dist/whole_dist)*M_PI/180.0;
          while (effect_vps_[i](3) < -M_PI)
            effect_vps_[i](3) += 2 * M_PI;
          while (effect_vps_[i](3) > M_PI)
            effect_vps_[i](3) -= 2 * M_PI;

          updates_waypts_.push_back(effect_vps_[i]);
        }
      }
      else
      {
        double pitch_gap = abs(pitch_end - pitch_start)>180.0? (360.0-abs(pitch_end - pitch_start)):abs(pitch_end - pitch_start);
        for (int i=1; i<(int)effect_vps_.size()-1; ++i)
        {
          double e_dist = (effect_vps_[i].head(3)-start.head(3)).norm();
          effect_vps_[i](3) = (pitch_start + cal_dir_pitch*cal_flag_pitch*pitch_gap*e_dist/whole_dist)*M_PI/180.0;
          effect_vps_[i](4) = (yaw_start + cal_dir_yaw*cal_flag_yaw*yaw_gap*e_dist/whole_dist)*M_PI/180.0;
          while (effect_vps_[i](3) < -M_PI)
            effect_vps_[i](3) += 2 * M_PI;
          while (effect_vps_[i](3) > M_PI)
            effect_vps_[i](3) -= 2 * M_PI;
          while (effect_vps_[i](4) < -M_PI)
            effect_vps_[i](4) += 2 * M_PI;
          while (effect_vps_[i](4) > M_PI)
            effect_vps_[i](4) -= 2 * M_PI;
          
          updates_waypts_.push_back(effect_vps_[i]);
        }
      }
    }

    return;
  }
  
  void HCSolver::ConstructKDTree(vector<Eigen::VectorXd>& pathlist, pcl::KdTreeFLANN<pcl::PointXYZ>& tree)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr PathCloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ p;
    for (auto node:pathlist)
    {
      p.x = node(0); p.y = node(1); p.z = node(2);
      PathCloud->points.push_back(p);
    }
    tree.setInputCloud(PathCloud);
  }
  /* time cost for k-opt */
  double HCSolver::kOptTimeCost(vector<Eigen::Vector3d>& path)
  {
    double cost = 0.0;
    if ((int)path.size() > 2)
    {
      cost += (path[1]-path[0]).norm()/vm_;
      for (int i=0; i<(int)path.size()-1; ++i)
        cost += pathTimeUniAcc(path[i-1], path[i], path[i+1]);
    }
    else
    {
      cost = (path[1]-path[0]).norm()/vm_;
    }

    return cost;
  }

  /* [s1, s2] [s3, s4] --> [s1, s4] [s2, s3] */
  /* Reverse neighbor: s1, s3, NoReverse neighbor: s2, s4 */
  void HCSolver::Make2Opt(Site *s1, Site *s2, Site *s3, Site *s4, bool turn)
  {
    // determine s4
    // Site *s4 = new Site;
    Site *Last = new Site;
    Site *Next = new Site;
    Site *ReverseStart = new Site;
    Site *ReverseEnd = new Site;

    Eigen::Vector3d A, B, C, D, E, F, G, H;
    bool seqF = true, endF = true;
    if (s2 == s1->Suc)
    {
      if (s3->Start == true)
        return;

      /* A->B->s1->s2->C->D, E->F->s4->s3->G->H */
      s4 = s3->Pred;
      seqF = true;
      // A << s1->Pred->Pred->X, s1->Pred->Y, s1->Pred->Z;
      // B << s1->Pred->X, s1->Pred->Y, s1->Pred->Z;
      // C << s2->Suc->X, s2->Suc->Y, s2->Suc->Z;
      // D << s2->Suc->Suc->X, s2->Suc->Suc->Y, s2->Suc->Suc->Z;
      // E << s4->Pred->Pred->X, s4->Pred->Pred->Y, s4->Pred->Pred->Z;
      // F << s4->Pred->X, s4->Pred->Y, s4->Pred->Z;
      // G << s3->Suc->X, s3->Suc->Y, s3->Suc->Z;
      // H << s3->Suc->Suc->X, s3->Suc->Suc->Y, s3->Suc->Suc->Z;
    }
    if (s2 == s1->Pred)
    {
      if (s3->End == true)
        return;
      
      /* D->C->s2->s1->B->A, H->G->s3->s4->F->E */
      s4 = s3->Suc;
      seqF = false;
      // A << s1->Suc->Suc->X, s1->Suc->Suc->Y, s1->Suc->Suc->Z;
      // B << s1->Suc->X, s1->Suc->Y, s1->Suc->Z;
      // C << s2->Pred->X, s2->Pred->Y, s2->Pred->Z;
      // D << s2->Pred->Pred->X, s2->Pred->Pred->Y, s2->Pred->Pred->Z;
      // E << s4->Suc->Suc->X, s4->Suc->Suc->Y, s4->Suc->Suc->Z;
      // F << s4->Suc->X, s4->Suc->Y, s4->Suc->Z;
      // G << s3->Pred->X, s3->Pred->Y, s3->Pred->Z;
      // H << s3->Pred->Pred->X, s3->Pred->Pred->Y, s3->Pred->Pred->Z;
    }
    // determine edge endpoints
    vector<Site*> swapSet = {s1, s2, s3, s4};
    Site *start = new Site; Site *end = new Site;
    start = AllPathSite.front();
    end = AllPathSite.back();

    int sflag = 0;
    Last = start;
    if (s1 == start || s2 == start || s3 == start || s4 == start)
      sflag++;
    while (sflag != 2)
    {
      Next = Last->Suc;
      int curid = Next->GlobalID;
      for (auto s:swapSet)
      {
        if (curid == s->GlobalID)
          sflag++;
      }
      Last = Next;
    }
    ReverseStart = Last;

    sflag = 0;
    Last = end;
    if (s1 == end || s2 == end || s3 == end || s4 == end)
      sflag++;
    while (sflag != 2)
    {
      Next = Last->Pred;
      int curid = Next->GlobalID;
      for (auto s:swapSet)
      {
        if (curid == s->GlobalID)
          sflag++;
      }
      Last = Next;
    }
    ReverseEnd = Last;
    if (s2 == ReverseStart || s2 == ReverseEnd)
      endF = true;
    if (s1 == ReverseStart || s1 == ReverseEnd)
      endF = false;
    // calculate gain
    double beforeCost, afterCost, Gain;
    Eigen::Vector3d x1, x2, x3, x4;
    x1 << s1->X, s1->Y, s1->Z;
    x2 << s2->X, s2->Y, s2->Z;
    x3 << s3->X, s3->Y, s3->Z;
    x4 << s4->X, s4->Y, s4->Z;

    double yaw_time_b1, yaw_time_b2;
    double pitch_time_b1, pitch_time_b2;
    yaw_time_b1 = min(abs(s1->Yaw-s2->Yaw), 2 * M_PI - abs(s1->Yaw-s2->Yaw))/yd_;
    yaw_time_b2 = min(abs(s3->Yaw-s4->Yaw), 2 * M_PI - abs(s3->Yaw-s4->Yaw))/yd_;
    pitch_time_b1 = min(abs(s1->Pitch-s2->Pitch), 2 * M_PI - abs(s1->Pitch-s2->Pitch))/yd_;
    pitch_time_b2 = min(abs(s3->Pitch-s4->Pitch), 2 * M_PI - abs(s3->Pitch-s4->Pitch))/yd_;
    // beforeCost = max((x1-x2).norm()/vm_, yaw_time_b1) + max((x3-x4).norm()/vm_, yaw_time_b2);
    beforeCost = max(max((x1-x2).norm()/vm_, yaw_time_b1), pitch_time_b1) + max(max((x3-x4).norm()/vm_, yaw_time_b2), pitch_time_b2);

    double yaw_time_a1, yaw_time_a2;
    double pitch_time_a1, pitch_time_a2;
    yaw_time_a1 = min(abs(s1->Yaw-s4->Yaw), 2 * M_PI - abs(s1->Yaw-s4->Yaw))/yd_;
    yaw_time_a2 = min(abs(s2->Yaw-s3->Yaw), 2 * M_PI - abs(s2->Yaw-s3->Yaw))/yd_;
    pitch_time_a1 = min(abs(s1->Pitch-s4->Pitch), 2 * M_PI - abs(s1->Pitch-s4->Pitch))/yd_;
    pitch_time_a2 = min(abs(s2->Pitch-s3->Pitch), 2 * M_PI - abs(s2->Pitch-s3->Pitch))/yd_;
    // afterCost = max((x1-x4).norm()/vm_, yaw_time_a1)+ max((x2-x3).norm()/vm_, yaw_time_a2);
    afterCost = max(max((x1-x4).norm()/vm_, yaw_time_a1), pitch_time_a1) + max(max((x2-x3).norm()/vm_, yaw_time_a2), pitch_time_a2);

    Gain = beforeCost - afterCost;
    // swap
    if (Gain > 0)
    {
      bool swap = false;
      if (turn == true)
      {
        double dist_gain = 0.0;
        vector<Eigen::Vector3d> before_path;
        vector<Eigen::Vector3d> after_path;
        vector<Eigen::Vector3d> turn_path_1, turn_path_2, turn_path_3, turn_path_4;
        double cost_1 = 0.0, cost_2 = 0.0, cost_3 = 0.0, cost_4 = 0.0;
        double angle1, angle2, angle3, angle4;
        if (seqF == true)
        {
          if (endF == true)
          {
            /* Before: B->s1->s2->C->D + F->s4->s3->G->H */
            /* After:  B->s1->s4->F->E + C->s2->s3->G->H */
            before_path.clear();
            if (yaw_time_b1 > pitch_time_b1)
            {
              angle1 = s1->Yaw;
              angle2 = s2->Yaw;
            }
            else
            {
              angle1 = s1->Pitch;
              angle2 = s2->Pitch;
            }
            opt_dist = computeTimeCost(x1, x2, angle1, angle2, before_path);
            turn_path_1.push_back(B);
            turn_path_1.insert(turn_path_1.end(), before_path.begin(), before_path.end());
            turn_path_1.push_back(C);
            turn_path_1.push_back(D);
            cost_1 = kOptTimeCost(turn_path_1);
            dist_gain += opt_dist;

            before_path.clear();
            if (yaw_time_b2 > pitch_time_b2)
            {
              angle3 = s3->Yaw;
              angle4 = s4->Yaw;
            }
            else
            {
              angle3 = s3->Pitch;
              angle4 = s4->Pitch;
            }
            opt_dist = computeTimeCost(x4, x3, angle4, angle3, before_path);
            turn_path_2.push_back(F);
            turn_path_2.insert(turn_path_2.end(), before_path.begin(), before_path.end());
            turn_path_2.push_back(G);
            turn_path_2.push_back(H);
            cost_2 = kOptTimeCost(turn_path_2);
            dist_gain += opt_dist;

            after_path.clear();
            if (yaw_time_a1 > pitch_time_a1)
            {
              angle1 = s1->Yaw;
              angle4 = s4->Yaw;
            }
            else
            {
              angle1 = s1->Pitch;
              angle4 = s4->Pitch;
            }
            opt_dist = computeTimeCost(x1, x4, angle1, angle4, after_path);
            turn_path_3.push_back(B);
            turn_path_3.insert(turn_path_3.end(), after_path.begin(), after_path.end());
            turn_path_3.push_back(F);
            turn_path_3.push_back(E);
            cost_3 = kOptTimeCost(turn_path_3);
            dist_gain -= opt_dist;

            after_path.clear();
            if (yaw_time_a2 > pitch_time_a2)
            {
              angle2 = s2->Yaw;
              angle3 = s3->Yaw;
            }
            else
            {
              angle2 = s2->Pitch;
              angle3 = s3->Pitch;
            }
            opt_dist = computeTimeCost(x2, x3, angle2, angle3, after_path);
            turn_path_4.push_back(C);
            turn_path_4.insert(turn_path_4.end(), after_path.begin(), after_path.end());
            turn_path_4.push_back(G);
            turn_path_4.push_back(H);
          }
          else
          {
            /* Before: B->s1->s2->C->D + F->s4->s3->G->H */
            /* After:  F->s4->s1->B->A + G->s3->s2->C->D */
            before_path.clear();
            if (yaw_time_b1 > pitch_time_b1)
            {
              angle1 = s1->Yaw;
              angle2 = s2->Yaw;
            }
            else
            {
              angle1 = s1->Pitch;
              angle2 = s2->Pitch;
            }
            opt_dist = computeTimeCost(x1, x2, angle1, angle2, before_path);
            turn_path_1.push_back(B);
            turn_path_1.insert(turn_path_1.end(), before_path.begin(), before_path.end());
            turn_path_1.push_back(C);
            turn_path_1.push_back(D);
            cost_1 = kOptTimeCost(turn_path_1);
            dist_gain += opt_dist;

            before_path.clear();
            if (yaw_time_b2 > pitch_time_b2)
            {
              angle3 = s3->Yaw;
              angle4 = s4->Yaw;
            }
            else
            {
              angle3 = s3->Pitch;
              angle4 = s4->Pitch;
            }
            opt_dist = computeTimeCost(x4, x3, angle4, angle3, before_path);
            turn_path_2.push_back(F);
            turn_path_2.insert(turn_path_2.end(), before_path.begin(), before_path.end());
            turn_path_2.push_back(G);
            turn_path_2.push_back(H);
            cost_2 = kOptTimeCost(turn_path_2);
            dist_gain += opt_dist;

            after_path.clear();
            if (yaw_time_a1 > pitch_time_a1)
            {
              angle1 = s1->Yaw;
              angle4 = s4->Yaw;
            }
            else
            {
              angle1 = s1->Pitch;
              angle4 = s4->Pitch;
            }
            opt_dist = computeTimeCost(x4, x1, angle4, angle1, after_path);
            turn_path_3.push_back(F);
            turn_path_3.insert(turn_path_3.end(), after_path.begin(), after_path.end());
            turn_path_3.push_back(B);
            turn_path_3.push_back(A);
            cost_3 = kOptTimeCost(turn_path_3);
            dist_gain -= opt_dist;

            after_path.clear();
            if (yaw_time_a2 > pitch_time_a2)
            {
              angle2 = s2->Yaw;
              angle3 = s3->Yaw;
            }
            else
            {
              angle2 = s2->Pitch;
              angle3 = s3->Pitch;
            }
            opt_dist = computeTimeCost(x3, x2, angle3, angle2, after_path);
            turn_path_4.push_back(G);
            turn_path_4.insert(turn_path_4.end(), after_path.begin(), after_path.end());
            turn_path_4.push_back(C);
            turn_path_4.push_back(D);
            cost_4 = kOptTimeCost(turn_path_4);
            dist_gain -= opt_dist;
          }
        }
        else
        {
          if (endF == true)
          {
            /* Before: C->s2->s1->B->A + G->s3->s4->F->E */
            /* After:  F->s4->s1->B->A + G->s3->s2->C->D */
            before_path.clear();
            if (yaw_time_b1 > pitch_time_b1)
            {
              angle1 = s1->Yaw;
              angle2 = s2->Yaw;
            }
            else
            {
              angle1 = s1->Pitch;
              angle2 = s2->Pitch;
            }
            opt_dist = computeTimeCost(x2, x1, angle2, angle1, before_path);
            turn_path_1.push_back(C);
            turn_path_1.insert(turn_path_1.end(), before_path.begin(), before_path.end());
            turn_path_1.push_back(B);
            turn_path_1.push_back(A);
            cost_1 = kOptTimeCost(turn_path_1);
            dist_gain += opt_dist;

            before_path.clear();
            if (yaw_time_b2 > pitch_time_b2)
            {
              angle3 = s3->Yaw;
              angle4 = s4->Yaw;
            }
            else
            {
              angle3 = s3->Pitch;
              angle4 = s4->Pitch;
            }
            opt_dist = computeTimeCost(x3, x4, angle3, angle4, before_path);
            turn_path_2.push_back(G);
            turn_path_2.insert(turn_path_2.end(), before_path.begin(), before_path.end());
            turn_path_2.push_back(F);
            turn_path_2.push_back(E);
            cost_2 = kOptTimeCost(turn_path_2);
            dist_gain += opt_dist;

            after_path.clear();
            if (yaw_time_a1 > pitch_time_a1)
            {
              angle1 = s1->Yaw;
              angle4 = s4->Yaw;
            }
            else
            {
              angle1 = s1->Pitch;
              angle4 = s4->Pitch;
            }
            opt_dist = computeTimeCost(x4, x1, angle4, angle1, after_path);
            turn_path_3.push_back(F);
            turn_path_3.insert(turn_path_3.end(), after_path.begin(), after_path.end());
            turn_path_3.push_back(B);
            turn_path_3.push_back(A);
            cost_3 = kOptTimeCost(turn_path_3);
            dist_gain -= opt_dist;

            after_path.clear();
            if (yaw_time_a2 > pitch_time_a2)
            {
              angle2 = s2->Yaw;
              angle3 = s3->Yaw;
            }
            else
            {
              angle2 = s2->Pitch;
              angle3 = s3->Pitch;
            }
            opt_dist = computeTimeCost(x3, x2, angle3, angle2, after_path);
            turn_path_4.push_back(G);
            turn_path_4.insert(turn_path_4.end(), after_path.begin(), after_path.end());
            turn_path_4.push_back(C);
            turn_path_4.push_back(D);
            cost_4 = kOptTimeCost(turn_path_4);
            dist_gain -= opt_dist;
          }
          else
          {
            /* Before: C->s2->s1->B->A + G->s3->s4->F->E */
            /* After:  B->s1->s4->F->E + C->s2->s3->G->H */
            before_path.clear();
            if (yaw_time_b1 > pitch_time_b1)
            {
              angle1 = s1->Yaw;
              angle2 = s2->Yaw;
            }
            else
            {
              angle1 = s1->Pitch;
              angle2 = s2->Pitch;
            }
            opt_dist = computeTimeCost(x2, x1, angle2, angle1, before_path);
            turn_path_1.push_back(C);
            turn_path_1.insert(turn_path_1.end(), before_path.begin(), before_path.end());
            turn_path_1.push_back(B);
            turn_path_1.push_back(A);
            cost_1 = kOptTimeCost(turn_path_1);
            dist_gain += opt_dist;
            
            before_path.clear();
            if (yaw_time_b2 > pitch_time_b2)
            {
              angle3 = s3->Yaw;
              angle4 = s4->Yaw;
            }
            else
            {
              angle3 = s3->Pitch;
              angle4 = s4->Pitch;
            }
            opt_dist = computeTimeCost(x3, x4, angle3, angle4, before_path);
            turn_path_2.push_back(G);
            turn_path_2.insert(turn_path_2.end(), before_path.begin(), before_path.end());
            turn_path_2.push_back(F);
            turn_path_2.push_back(E);
            cost_2 = kOptTimeCost(turn_path_2);
            dist_gain += opt_dist;

            after_path.clear();
            if (yaw_time_a1 > pitch_time_a1)
            {
              angle1 = s1->Yaw;
              angle4 = s4->Yaw;
            }
            else
            {
              angle1 = s1->Pitch;
              angle4 = s4->Pitch;
            }
            opt_dist = computeTimeCost(x1, x4, angle1, angle4, after_path);
            turn_path_3.push_back(B);
            turn_path_3.insert(turn_path_3.end(), after_path.begin(), after_path.end());
            turn_path_3.push_back(F);
            turn_path_3.push_back(E);
            cost_3 = kOptTimeCost(turn_path_3);
            dist_gain -= opt_dist;

            after_path.clear();
            if (yaw_time_a2 > pitch_time_a2)
            {
              angle2 = s2->Yaw;
              angle3 = s3->Yaw;
            }
            else
            {
              angle2 = s2->Pitch;
              angle3 = s3->Pitch;
            }
            opt_dist = computeTimeCost(x2, x3, angle2, angle3, after_path);
            turn_path_4.push_back(C);
            turn_path_4.insert(turn_path_4.end(), after_path.begin(), after_path.end());
            turn_path_4.push_back(G);
            turn_path_4.push_back(H);
            cost_4 = kOptTimeCost(turn_path_4);
            dist_gain -= opt_dist;
          }
        }
        double phy_gain = cost_1+cost_2-cost_3-cost_4;
        if (phy_gain > 0 || dist_gain > 0)
        // if (dist_gain > 0)
        {
          swap = true;
        }
      }
      else
      {
        double angle1, angle2, angle3, angle4;
        if (yaw_time_b1 > pitch_time_b1)
        {
          angle1 = s1->Yaw;
          angle2 = s2->Yaw;
        }
        else
        {
          angle1 = s1->Pitch;
          angle2 = s2->Pitch;
        }
        vector<Eigen::Vector3d> before_path;
        double dist_1 = compute_Cost(x1, x2, angle1, angle2, before_path);
        before_path.clear();
        if (yaw_time_b2 > pitch_time_b2)
        {
          angle3 = s3->Yaw;
          angle4 = s4->Yaw;
        }
        else
        {
          angle3 = s3->Pitch;
          angle4 = s4->Pitch;
        }
        double dist_2 = compute_Cost(x3, x4, angle3, angle4, before_path);

        vector<Eigen::Vector3d> after_path;
        if (yaw_time_a1 > pitch_time_a1)
        {
          angle1 = s1->Yaw;
          angle4 = s4->Yaw;
        }
        else
        {
          angle1 = s1->Pitch;
          angle4 = s4->Pitch;
        }
        double dist_3 = compute_Cost(x1, x4, angle1, angle4, after_path);
        after_path.clear();
        if (yaw_time_a2 > pitch_time_a2)
        {
          angle2 = s2->Yaw;
          angle3 = s3->Yaw;
        }
        else
        {
          angle2 = s2->Pitch;
          angle3 = s3->Pitch;
        }
        double dist_4 = compute_Cost(x2, x3, angle2, angle3, after_path);
        double phy_gain = dist_1+dist_2-dist_3-dist_4;
        if (phy_gain > 0)
          swap = true;
      }

      if (swap == true)
      {
      swapTimes++;
      // apply the 2-opt move by reversing the order of the edges between the endpoints
      //* e.g. start -> ... -> s1 -> s2 -> ... -> s4 -> s3 -> ... -> end 
      //* Reverse tour between s2 and s4
      if (s2 == s1->Suc)
      {
        s1->Suc = s4;
        s2->Pred = s3;
        s3->Pred = s2;
        s4->Suc = s1;
      }
      if (s2 == s1->Pred)
      {
        s1->Pred = s4;
        s2->Suc = s3;
        s3->Suc = s2;
        s4->Pred = s1;
      }

      Site *TempPred = new Site;
      Site *TempSuc = new Site;
      Site *Terminal = new Site;

      if (ReverseStart == ReverseEnd->Pred)
      {
        TempPred = ReverseStart->Pred;
        TempSuc = ReverseStart->Suc;
        ReverseStart->Pred = TempSuc;
        ReverseStart->Suc = TempPred;

        TempPred = ReverseEnd->Pred;
        TempSuc = ReverseEnd->Suc;
        ReverseEnd->Pred = TempSuc;
        ReverseEnd->Suc = TempPred;
      }
      else
      {
        Last = ReverseStart;
        Terminal = ReverseEnd->Suc;
        while (Last != Terminal)
        {
          Next = Last->Suc;
          TempPred = Last->Pred;
          TempSuc = Last->Suc;
          Last->Pred = TempSuc;
          Last->Suc = TempPred;
          Last = Next;
        }
      }

      } 
    }
  }

  /* [s1, s2] [s3, s4] [s5, s6] --> (1/8)*transformation */
  void HCSolver::Make3Opt(Site *s1, Site *s2, Site *s3, Site *s4, Site *s5, Site *s6, bool turn)
  {
    std::random_device rdprob2;
    std::mt19937 genprob2(rdprob2());
    std::uniform_real_distribution<> disprob2(0.0, 1.0);
    double prob2 = disprob2(genprob2);
    if (prob2 > 0.5)
    {
      Make2Opt(s1, s2, s3, s4, turn);

      std::random_device rdprob3;
      std::mt19937 genprob3(rdprob3());
      std::uniform_real_distribution<> disprob3(0.0, 1.0);
      double prob3 = disprob3(genprob3);
      if (prob3 > 0.5)
      {
        std::random_device rdprob4;
        std::mt19937 genprob4(rdprob4());
        std::uniform_real_distribution<> disprob4(0.0, 1.0);
        double prob4 = disprob4(genprob4);
        if (prob4 > 0.5)
        {
          if (s1->Start == true)
            return;

          s2 = s1->Pred;
        }
        else
        {
          if (s1->End == true)
            return;

          s2 = s1->Suc;
        }
        if (s5 != s2->Pred && s5 != s2->Suc && s5 != s1->Pred && s5 != s1->Suc)
          Make2Opt(s1, s2, s5, s6, turn);
      }
      else
      {
        std::random_device rdprob5;
        std::mt19937 genprob5(rdprob5());
        std::uniform_real_distribution<> disprob5(0.0, 1.0);
        double prob5 = disprob5(genprob5);
        if (prob5 > 0.5)
        {
          if (s3->Start == true)
            return;

          s4 = s3->Pred;
        }
        else
        {
          if (s3->End == true)
            return;
          
          s4 = s3->Suc;
        }
        if (s5 != s4->Pred && s5 != s4->Suc && s5 != s3->Pred && s5 != s3->Suc)
          Make2Opt(s3, s4, s5, s6, turn);
      }
    }
    else
    {
      Make2Opt(s1, s2, s5, s6, turn);

      std::random_device rdprob6;
      std::mt19937 genprob6(rdprob6());
      std::uniform_real_distribution<> disprob6(0.0, 1.0);
      double prob6 = disprob6(genprob6);
      if (prob6 > 0.5)
      {
        std::random_device rdprob7;
        std::mt19937 genprob7(rdprob7());
        std::uniform_real_distribution<> disprob7(0.0, 1.0);
        double prob7 = disprob7(genprob7);
        if (prob7 > 0.5)
        {
          if (s1->Start == true)
            return;

          s2 = s1->Pred;
        }
        else
        {
          if (s1->End == true)
            return;

          s2 = s1->Suc;
        }
        if (s3 != s2->Pred && s3 != s2->Suc && s3 != s1->Pred && s3 != s1->Suc)
          Make2Opt(s1, s2, s3, s4, turn);
      }
      else
      {
        std::random_device rdprob8;
        std::mt19937 genprob8(rdprob8());
        std::uniform_real_distribution<> disprob8(0.0, 1.0);
        double prob8 = disprob8(genprob8);
        if (prob8 > 0.5)
        {
          if (s5->Start == true)
            return;
          
          s6 = s5->Pred;
        }
        else
        {
          if (s5->End == true)
            return;

          s6 = s5->Suc;
        }
        if (s3 != s6->Pred && s3 != s6->Suc && s3 != s5->Pred && s3 != s5->Suc)
          Make2Opt(s5, s6, s3, s4, turn);
      }
    }
  }

  void HCSolver::RandomLocal2Opt(bool turn)
  {
    Site *t1 = new Site;
    Site *t2 = new Site;
    Site *t3 = new Site;
    Site *t4 = new Site;
    vector<int> excludeID;
    vector<Site*> remainSite;

    int localNum = LocalSite.size();
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, localNum - 1);
    int randomInt = dis(gen);
    t1 = LocalSite[randomInt];
    excludeID.push_back(t1->GlobalID);
    
    // if (t1->Start == true)
    //   ROS_ERROR("-------------------------------- t1 is Start!");
    // if (t1->End == true)
    //   ROS_ERROR("t1 is End! --------------------------------");

    // if (t1->Start != true)
    //   ROS_ERROR("t1->Pred GlobalID: %d, t1->Pred LocalID: %d", t1->Pred->GlobalID, t1->Pred->LocalID);
    // if (t1->End != true)
    //   ROS_WARN("t1->Suc GlobalID: %d, t1->Suc LocalID: %d", t1->Suc->GlobalID, t1->Suc->LocalID);

    std::random_device rdprob;
    std::mt19937 genprob(rdprob());
    std::uniform_real_distribution<> disprob(0.0, 1.0);
    double prob = disprob(genprob);
    if (prob > 0.5)
    {
      if (t1->Start == true)
        return;

      /* avoid t3->t2->t1 since t4 will be t2*/
      t2 = t1->Pred;
      excludeID.push_back(t2->GlobalID); 
      if (t2->Start != true)
        excludeID.push_back(t2->Pred->GlobalID);
    }
    else
    {
      if (t1->End == true)
        return;

      /* avoid t1->t2->t3 since t4 will be t2*/
      t2 = t1->Suc;
      excludeID.push_back(t2->GlobalID); 
      if (t2->End != true)
        excludeID.push_back(t2->Suc->GlobalID);
    }

    Eigen::Vector3d posT1; posT1 << t1->X, t1->Y, t1->Z;
    Eigen::Vector3d posT2; posT2 << t2->X, t2->Y, t2->Z;
    Eigen::Vector3d vecT1T2 = (posT2 - posT1).normalized();

    for (auto s:LocalSite)
    {
      bool in = true;
      for (auto ex:excludeID)
      {
        if (s->GlobalID == ex)
        {
          in = false;
          break;
        }
      }

      Eigen::Vector3d posCandidate; posCandidate << s->X, s->Y, s->Z;
      if (in == true && (posCandidate-posT2).norm() < 20.0)
      {
        Eigen::Vector3d posCandidatePred, posCandidateSuc;
        Eigen::Vector3d vecCandidatePred, vecCandidateSuc;
        double diffAnglePred, diffAngleSuc, diffSel;

        if (s->Start != true)
        {
          posCandidatePred << s->Pred->X, s->Pred->Y, s->Pred->Z;
          vecCandidatePred = (posCandidate - posCandidatePred).normalized();
          diffAnglePred = acos(vecT1T2.dot(vecCandidatePred));
        }
        else
          diffAnglePred = M_PI;
        
        if (s->End != true)
        {
          posCandidateSuc << s->Suc->X, s->Suc->Y, s->Suc->Z;
          vecCandidateSuc = (posCandidateSuc - posCandidate).normalized();
          diffAngleSuc = acos(vecT1T2.dot(vecCandidateSuc));
        }
        else
          diffAngleSuc = M_PI;
        
        diffSel = min(diffAnglePred, diffAngleSuc);

        if (diffSel > M_PI/36.0)
          remainSite.push_back(s);
      }
    }

    if ((int)remainSite.size() > 0)
    {
      int remainNum = remainSite.size();
      random_device rdR;
      mt19937 genR(rdR());
      uniform_int_distribution<> disR(0, remainNum - 1);
      int randomIntRemain = disR(genR);
      t3 = remainSite[randomIntRemain];

      Make2Opt(t1, t2, t3, t4, turn);
    }
  }

  void HCSolver::RandomLocal3Opt(bool turn)
  {
    Site *t1 = new Site;
    Site *t2 = new Site;
    Site *t3 = new Site;
    Site *t4 = new Site;
    Site *t5 = new Site;
    Site *t6 = new Site;
    vector<int> excludeID;
    vector<Site*> remainSite;

    int localNum = LocalSite.size();
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, localNum - 1);
    int randomInt = dis(gen);
    t1 = LocalSite[randomInt];
    excludeID.push_back(t1->GlobalID);
    
    std::random_device rdprob;
    std::mt19937 genprob(rdprob());
    std::uniform_real_distribution<> disprob(0.0, 1.0);
    double prob = disprob(genprob);
    
    if (prob > 0.5)
    {
      if (t1->Start == true)
        return;
      
      /* avoid t3->t2->t1 since t4 will be t2*/
      t2 = t1->Pred;
      excludeID.push_back(t2->GlobalID); 
      if (t2->Start != true)
        excludeID.push_back(t2->Pred->GlobalID);
    }
    else
    {
      if (t1->End == true)
        return;
      
      /* avoid t1->t2->t3 since t4 will be t2*/
      t2 = t1->Suc;
      excludeID.push_back(t2->GlobalID); 
      if (t2->End != true)
        excludeID.push_back(t2->Suc->GlobalID);
    }
    Eigen::Vector3d posT1; posT1 << t1->X, t1->Y, t1->Z;
    Eigen::Vector3d posT2; posT2 << t2->X, t2->Y, t2->Z;
    Eigen::Vector3d vecT1T2 = (posT2 - posT1).normalized();

    for (auto s:LocalSite)
    {
      if (s->Start == true)
        continue;

      bool in = true;
      for (auto ex:excludeID)
      {
        if (s->GlobalID == ex)
        {
          in = false;
          break;
        }
      }
      Eigen::Vector3d posCandidate; posCandidate << s->X, s->Y, s->Z;
      if (in == true && (posCandidate-posT2).norm() < 20.0)
      {
        Eigen::Vector3d posCandidatePred, posCandidateSuc;
        Eigen::Vector3d vecCandidatePred, vecCandidateSuc;
        double diffAnglePred, diffAngleSuc, diffSel;

        if (s->Start != true)
        {
          posCandidatePred << s->Pred->X, s->Pred->Y, s->Pred->Z;
          vecCandidatePred = (posCandidate - posCandidatePred).normalized();
          diffAnglePred = acos(vecT1T2.dot(vecCandidatePred));
        }
        else
          diffAnglePred = M_PI;

        if (s->End != true)
        {
          posCandidateSuc << s->Suc->X, s->Suc->Y, s->Suc->Z;
          vecCandidateSuc = (posCandidateSuc - posCandidate).normalized();
          diffAngleSuc = acos(vecT1T2.dot(vecCandidateSuc));
        }
        else
          diffAngleSuc = M_PI;

        diffSel = min(diffAnglePred, diffAngleSuc);

        if (diffSel > M_PI/36.0)
          remainSite.push_back(s);
      }
    }

    if ((int)remainSite.size() > 0)
    {
      int remainNum = remainSite.size();
      random_device rdR;
      mt19937 genR(rdR());
      uniform_int_distribution<> disR(0, remainNum - 1);
      int randomIntRemain = disR(genR);
      t3 = remainSite[randomIntRemain];

      /* choose t5 & t6 */
      vector<int> excludeID2;
      vector<Site*> remainSite2;
      excludeID2.push_back(t1->GlobalID); 
      if (t1->Start != true)
        excludeID2.push_back(t1->Pred->GlobalID);
      if (t1->End != true)
        excludeID2.push_back(t1->Suc->GlobalID);
      excludeID2.push_back(t2->GlobalID); 
      if (t2->Start != true)
        excludeID2.push_back(t2->Pred->GlobalID);
      if (t2->End != true)
        excludeID2.push_back(t2->Suc->GlobalID);
      excludeID2.push_back(t3->GlobalID);
      if (t3->Start != true)
        excludeID2.push_back(t3->Pred->GlobalID);
      if (t3->End != true)
        excludeID2.push_back(t3->Suc->GlobalID);

      Eigen::Vector3d posT3; posT3 << t3->X, t3->Y, t3->Z;
      Eigen::Vector3d posT4; posT4 << t3->Pred->X, t3->Pred->Y, t3->Pred->Z;
      Eigen::Vector3d vecT3T4 = (posT4 - posT3).normalized();

      for (auto s:LocalSite)
      {
        bool in = true;
        for (auto ex:excludeID2)
        {
          if (s->GlobalID == ex)
          {
            in = false;
            break;
          }
        }
        Eigen::Vector3d posCandidate; posCandidate << s->X, s->Y, s->Z;
        if (in == true && (posCandidate-posT3).norm() < 20.0)
        {
          Eigen::Vector3d posCandidatePred, posCandidateSuc;
          Eigen::Vector3d vecCandidatePred, vecCandidateSuc;
          double diffAnglePred, diffAngleSuc, diffSel;

          if (s->Start != true)
          {
            posCandidatePred << s->Pred->X, s->Pred->Y, s->Pred->Z;
            vecCandidatePred = (posCandidate - posCandidatePred).normalized();
            diffAnglePred = acos(vecT3T4.dot(vecCandidatePred));
          }
          else
            diffAnglePred = M_PI;
          
          if (s->End != true)
          {
            posCandidateSuc << s->Suc->X, s->Suc->Y, s->Suc->Z;
            vecCandidateSuc = (posCandidateSuc - posCandidate).normalized();
            diffAngleSuc = acos(vecT3T4.dot(vecCandidateSuc));
          }
          else
            diffAngleSuc = M_PI;
          
          diffSel = min(diffAnglePred, diffAngleSuc);

          if (diffSel > M_PI/36.0)
            remainSite2.push_back(s);
        }
      }

      if ((int)remainSite2.size() > 0)
      {
        int remainNum2 = remainSite2.size();
        random_device rdR2;
        mt19937 genR2(rdR2());
        uniform_int_distribution<> disR2(0, remainNum2 - 1);
        int randomIntRemain2 = disR2(genR2);
        t5 = remainSite2[randomIntRemain2];

        Make3Opt(t1, t2, t3, t4, t5, t6, turn);
      }
    }
  }

  /* solve this path as a Chinese Postman Problem (CPP) */
  vector<int> HCSolver::generalizedEulerianPath(Eigen::MatrixXd& adj, int start_vertex)
  {
    vector<int> path;

    // [start_vertex, 0, 1, 2, ..., n]
    vector<int> convertVertices;
    convertVertices.push_back(start_vertex);
    for (int i=0; i<(int)adj.rows(); ++i)
    {
      if (i == start_vertex)
        continue;
      else
        convertVertices.push_back(i);
    }
    
    int vertexNum = (int)convertVertices.size();
    int edgeNum = 0;
    for (int i=0; i<(int)adj.rows(); ++i)
      for (int j=0; j<(int)adj.cols(); ++j)
        if (adj(i,j) != 0.0 && adj(i,j) != MAX && i<j)
          edgeNum++;
    
    string problem_file = BoundarySolver_ + "/problem.txt";
    ofstream prob_file(problem_file);
    prob_file << to_string(vertexNum) << "\n";
    prob_file << to_string(edgeNum) << "\n";
    for (int i=0; i<(int)convertVertices.size(); ++i)
      for (int j=0; j<(int)convertVertices.size(); ++j)
        {
          if (adj(convertVertices[i], convertVertices[j]) != 0.0 && adj(convertVertices[i], convertVertices[j]) != MAX && i < j)
          {
            int cost = (int)(100*adj(convertVertices[i], convertVertices[j]));
            prob_file << to_string(i) << " " << to_string(j) << " " << to_string(cost) << "\n";
          }
        }
    
    prob_file.close();

    string command_ = "cd " + BoundarySolver_ + " && ./chinese -f problem.txt";
    const char* charPtr = command_.c_str();
    boundary_system_back_=system(charPtr);

    string result_file = BoundarySolver_ + "/solution.txt";
    ifstream sol_file(result_file);
    string line;
    getline(sol_file, line);
    if (stoi(line) < 0)
    {
      path = {};
      return path;
    }

    while (getline(sol_file, line)) 
      if (line.compare("PATH") == 0) break;
    
    getline(sol_file, line);
    vector<string> result = split(line, " ");
    for (int i=0; i<(int)result.size(); ++i)
    {
      path.push_back(convertVertices[stoi(result[i])]);
    }

    return path;
  }

  int HCSolver::fourPointsNearestPoint(Eigen::Vector3d& spherePtA, Eigen::Vector3d& spherePtB, Eigen::Vector3d& centerA, Eigen::Vector3d& centerB, vector<Eigen::VectorXd>& vps)
  {
    int nearestID = -1;
    Eigen::Vector3d vp_pos_;
    double dist_min_ = MAX;

    for (int i=0; i<(int)vps.size(); ++i) 
    {
      vp_pos_(0) = vps[i](0); vp_pos_(1) = vps[i](1); vp_pos_(2) = vps[i](2); 
      // Compute the distance between the point and the line segment AB
      Eigen::Vector3d AP = vp_pos_ - spherePtA;
      Eigen::Vector3d BP = vp_pos_ - spherePtB;
      Eigen::Vector3d CP = vp_pos_ - centerA;
      Eigen::Vector3d DP = vp_pos_ - centerB;
      double within_dist_ = AP.norm() + BP.norm() + CP.norm() + DP.norm();
      if (within_dist_ < dist_min_)
      {
        dist_min_ = within_dist_;
        nearestID = i;
      }
    }

    return nearestID;
  }

  void HCSolver::solveConsistentPath(const vector<Eigen::VectorXd>& pts, const Eigen::Vector3d& vel, vector<Eigen::VectorXd>& refine_path, vector<bool>& indi)
  {
    double dist_max = COST_INFINITY;
    int dim = (int)pts.size();
    Eigen::MatrixXd cost_matrix;
    cost_matrix = Eigen::MatrixXd::Zero(dim, dim);
    map<Eigen::Vector2i, vector<Eigen::VectorXd>, Vector2iCompare> inter_path_map;

    // * Construct Cost Matrix
    Eigen::VectorXd pose_i, pose_j;
    Eigen::Vector3d pos_i, pos_j, dir_ij;
    double pitch_i, pitch_j, yaw_i, yaw_j;
    double cost_ij, turn_cost_ij, cost_ij_total;
    vector<Eigen::Vector3d> ij_path;
    vector<Eigen::VectorXd> updated_ij_path;

    for (int i=0; i<dim; ++i)
      for (int j=0; j<dim; ++j)
      {
        ij_path.clear();
        updated_ij_path.clear();

        if (i == j || j == 0)
          continue;
        
        if (i == dim-1 && j != 0)
        {
          cost_matrix(i,j) = dist_max;
          continue;
        }

        pose_i = pts[i];
        pose_j = pts[j];
        pos_i = pose_i.head(3);
        pos_j = pose_j.head(3);
        pitch_i = pose_i(3);
        pitch_j = pose_j(3);
        yaw_i = pose_i(4);
        yaw_j = pose_j(4);

        double p_diff = min(abs(pitch_i-pitch_j), 2*M_PI-abs(pitch_i-pitch_j));
        double y_diff = min(abs(yaw_i-yaw_j), 2*M_PI-abs(yaw_i-yaw_j));
        double ai = p_diff>y_diff?pitch_i:yaw_i;
        double aj = p_diff>y_diff?pitch_j:yaw_j;

        cost_ij = compute_Cost(pos_i, pos_j, ai, aj, ij_path);
        if ((int)ij_path.size() > 2)
          AngleInterpolation(pose_i, pose_j, ij_path, updated_ij_path);
        else
          updated_ij_path = {pose_i, pose_j};
        
        Eigen::Vector2i key_ij = Eigen::Vector2i(i,j);
        inter_path_map[key_ij] = updated_ij_path;

        dir_ij = (pos_j - pos_i).normalized();
        double vel_diff = (vm_*dir_ij - vel).norm() / amean_;
        turn_cost_ij = vel_diff;

        cost_ij_total = cost_ij + turn_cost_ij;
        
        cost_matrix(i,j) = cost_ij_total;
      }

    // * Prepare File
    string par_F = LocalFolder_+"/global_consistency.par";
    string pro_F = LocalFolder_+"/global_consistency.tsp";
    string res_F = LocalFolder_+"/global_consistency.txt";

    // * Write Problem
    ofstream par_file(par_F);
    par_file << "PROBLEM_FILE = " << pro_F << "\n";
    par_file << "GAIN23 = NO\n";
    par_file << "OUTPUT_TOUR_FILE =" << res_F << "\n";
    par_file << "RUNS = " << to_string(LocalRuns_) << "\n";
    par_file.close();

    ofstream pro_file(pro_F);
    string prob_spec;
    prob_spec = "NAME : Global_Consistency\nTYPE : ATSP\nDIMENSION : "
     + to_string(dim) +
    "\nEDGE_WEIGHT_TYPE : "
    "EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX" + 
    "\nEDGE_WEIGHT_SECTION\n";
    pro_file << prob_spec;
    for (int i=0; i<dim; ++i)
    {
      for (int j=0; j<dim; ++j)
      {
        int int_cost = cost_matrix(i,j)*precision_;
        pro_file << to_string(int_cost) << " ";
      }
      pro_file << "\n";
    }
    pro_file << "EOF";
    pro_file.close();

    // * Run LKH Solver
    string cmd;
    cmd = "cd " + GlobalSolver_ + " && ./LKH " + par_F;
    const char* char_ptr = cmd.c_str();
    consistency_system_back_ = system(char_ptr);

    // * Read Result
    vector<int> results;
    ifstream res_file(res_F);
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

    // * Refine Path
    std::stringstream ss;
    ss << "\033[37m[GlobalConsistency] rectified seq : \033[32m ";
    for (int j=0; j<(int)results.size(); ++j)
    {
      ss << to_string(results[j]);
      if (j != (int)results.size()-1)
        ss << " -> ";
    }
    ss << "\033[0m";
    ROS_INFO("%s", ss.str().c_str());

    for (int i=0; i<(int)results.size()-1; ++i)
    {
      refine_path.push_back(pts[results[i]]);
      indi.push_back(false);
      Eigen::Vector2i key = Eigen::Vector2i(results[i], results[i+1]);
      if (inter_path_map.find(key) != inter_path_map.end())
      {
        vector<Eigen::VectorXd> inter_path = inter_path_map[key];
        if ((int)inter_path.size() > 2)
        {
          refine_path.insert(refine_path.end(), inter_path.begin()+1, inter_path.end()-1);
          for (int j=1; j<(int)inter_path.size()-1; ++j)
            indi.push_back(true);
        }
      }
    }
    refine_path.push_back(pts[results.back()]);
    indi.push_back(false);

    return;
  }

  // ? Consistent Planning Func
  void HCSolver::subPathSearching(int sub_idx, vector<Eigen::VectorXd> vps)
  {    
    map<pair<Eigen::VectorXd, Eigen::VectorXd>, AstarResults, PairVectorXdCompare> temp_results;

    int dim = (int)vps.size();
    double max_cost = COST_INFINITY;

    // * Astar Search
    vector<Eigen::Vector3d> pos_inter_path;
    double cost = 0.0;
    vector<Eigen::VectorXd> updated_inter_path;
    for (int i=0; i<dim; ++i)
      for (int j=0; j<dim; ++j)
      {
        if (i == j)
          continue;
        
        Eigen::VectorXd vp_i = vps[i];
        Eigen::VectorXd vp_j = vps[j];
        Eigen::Vector3d pos_i = vp_i.head(3);
        Eigen::Vector3d pos_j = vp_j.head(3);

        pos_inter_path.clear();
        updated_inter_path.clear();

        if ((pos_i - pos_j).norm() > 20.0)
        {
          cost = max_cost;
          pos_inter_path = {pos_i, pos_j};
        }
        else
        {
          // ? Path Searching
          bool safe = true;
          double path_length = 0.0;
          Eigen::Vector3i idx;
          this->solve_raycaster_->input(pos_i, pos_j);
          while (this->solve_raycaster_->nextId(idx)) 
          {
            if (!this->solve_map_->isInMap_hc(idx))
            {
              safe = false;
              break;
            }
            
            if (this->solve_map_->hcmd_->occupancy_inflate_buffer_hc_[this->solve_map_->toAddress_hc(idx)] == 1 || this->solve_map_->hcmd_->occupancy_buffer_internal_[this->solve_map_->toAddress_hc(idx)] == 1) 
            {
              safe = false;
              break;
            }

            if (safe == false)
              break;
          }

          if (safe)
          {
            path_length = (pos_i - pos_j).norm();
            pos_inter_path = {pos_i, pos_j};
          }
          else
          {
            this->astar_->reset();
            this->astar_->setResolution(astarSearchingRes);
            if (this->astar_->hc_search(pos_i, pos_j) == Astar::REACH_END) 
            {
              pos_inter_path = this->astar_->getPath();
              shortenPath(pos_inter_path);
              path_length = this->astar_->pathLength(pos_inter_path);
            }
            else
            {
              cost = max_cost;
              pos_inter_path = {pos_i, pos_j};
            }
          }

          if ((int)pos_inter_path.size() > 2)
            AngleInterpolation(vp_i, vp_j, pos_inter_path, updated_inter_path);
          else
            updated_inter_path = {};

          double pos_time = path_length / vm_;
          cost = pos_time;
        }

        AstarResults temp_astar;
        temp_astar.cost = cost;
        temp_astar.path = updated_inter_path;
        temp_results[make_pair(vp_i, vp_j)] = temp_astar;
      }
    
    // * Data Sync
    this->sub_pair_astar_results_[sub_idx] = temp_results;
    
    return;
  }

  void HCSolver::findBridgePath(Eigen::VectorXd& p1, Eigen::VectorXd& p2, double& cost, vector<Eigen::VectorXd>& path, vector<bool>& indicators, const double res, bool only_occ)
  {
    Eigen::Vector3d pos_i = p1.head(3), pos_j = p2.head(3);
    vector<Eigen::Vector3d> ij_path;

    bool safe = true;

    Eigen::Vector3i idx;
    solve_raycaster_->input(pos_i, pos_j);
    while (solve_raycaster_->nextId(idx)) 
    {
      if (!solve_map_->isInMap_hc(idx))
      {
        safe = false;
        break;
      }

      if (solve_map_->hcmd_->occupancy_inflate_buffer_hc_[solve_map_->toAddress_hc(idx)] == 1 || solve_map_->hcmd_->occupancy_buffer_internal_[solve_map_->toAddress_hc(idx)] == 1) 
      {
        safe = false;
        break;
      }
      
      if (safe == false)
        break;
    }

    if (safe) 
    {
      ij_path = { pos_i, pos_j };
      cost = (pos_i-pos_j).norm();
    }
    else
    {
      astar_->reset();
      astar_->setResolution(res);
      int result;
      if (only_occ)
        result = astar_->hc_occ_search(pos_i, pos_j);
      else
        result = astar_->hc_search(pos_i, pos_j);
      if (result == Astar::REACH_END)
      {
        ij_path = astar_->getPath();
        shortenPath(ij_path);
        cost = astar_->pathLength(ij_path);
      }
      else
      {
        cost = COST_INFINITY;
        ij_path = {pos_i, pos_j};
      }
    }

    if ((int)ij_path.size() > 2) 
    {
      this->AngleInterpolation(p1, p2, ij_path, path);
      indicators = vector<bool>(path.size(), true);
    }
    else
    {
      path = {};
      indicators = {};
    }

    return;
  }

  void HCSolver::findRealBridgePath(Eigen::VectorXd& p1, Eigen::VectorXd& p2, double& cost, vector<Eigen::VectorXd>& path, vector<bool>& indicators)
  {
    Eigen::Vector3d pos_i = p1.head(3), pos_j = p2.head(3);
    vector<Eigen::Vector3d> ij_path;

    bool safe = true;
    Eigen::Vector3i idx;
    real_raycaster_->input(pos_i, pos_j);
    while (real_raycaster_->nextId(idx)) 
    {
      if (solve_map_->getOccupancy(idx) == SDFMap::OCCUPANCY::OCCUPIED || solve_map_->getInflateOccupancy(idx) == 1) 
      {
        safe = false;
        break;
      }
      
      if (safe == false)
        break;
    }

    if (safe) 
    {
      ij_path = { pos_i, pos_j };
      cost = (pos_i-pos_j).norm();
    }
    else
    {
      astar_->reset();
      astar_->setResolution(astarSearchingRes);
      int result;
      result = astar_->wholeSearch(pos_i, pos_j);
      if (result == Astar::REACH_END)
      {
        ij_path = astar_->getPath();
        cost = astar_->pathLength(ij_path);
      }
      else
      {
        cost = COST_INFINITY;
        ij_path = {pos_i, pos_j};
      }
    }

    if ((int)ij_path.size() > 2) 
    {
      this->AngleInterpolation(p1, p2, ij_path, path);
      indicators = vector<bool>(path.size(), true);
    }

    return;
  }

  int HCSolver::findBoundary(Eigen::Vector3d& spherePtA, Eigen::Vector3d& spherePtB, vector<Eigen::VectorXd>& vps, int excluded_idx)
  {
    int nearestID = -1;
    Eigen::Vector3d vp_pos_;
    double dist_min_ = MAX;

    for (int i=0; i<(int)vps.size(); ++i) 
    {
      vp_pos_(0) = vps[i](0); vp_pos_(1) = vps[i](1); vp_pos_(2) = vps[i](2); 
      // Compute the distance between the point and the line segment AB
      Eigen::Vector3d AP = vp_pos_ - spherePtA;
      Eigen::Vector3d BP = vp_pos_ - spherePtB;

      double within_dist_ = AP.norm() + BP.norm();
      if (within_dist_ < dist_min_ && i != excluded_idx)
      {
        dist_min_ = within_dist_;
        nearestID = i;
      }
    }

    return nearestID;
  }

} // namespace flyco