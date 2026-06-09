/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Apr. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the main algorithm of safe path searching using A* algorithm.
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

#include "path_searching/astar2.h"

using namespace std;
using namespace Eigen;

namespace flyco {
Astar::Astar() {
}

Astar::~Astar() {
  for (int i = 0; i < allocate_num_; i++)
  {
    if (path_node_pool_[i] != nullptr) {
      delete path_node_pool_[i];
      path_node_pool_[i] = nullptr;
    }
  }
}

void Astar::init_hc(ros::NodeHandle& nh)
{
  nh.param("astar/resolution_astar", resolution_, -1.0);
  nh.param("astar/lambda_heu_hc", lambda_heu_hc_, -1.0);
  nh.param("astar/max_search_time", max_search_time_, -1.0);
  nh.param("viewpoint_manager/zGround", zFlag, false);
  nh.param("viewpoint_manager/GroundPos", groundz, -1.0);
  nh.param("viewpoint_manager/safeHeight", safeheight, -1.0);

  tie_breaker_ = 1.0 + 1.0 / 1000;
  /* --- params --- */
  this->inv_resolution_ = 1.0 / resolution_;
  path_node_pool_.clear();
  path_node_pool_.resize(allocate_num_);
  for (int i = 0; i < allocate_num_; i++) {
    path_node_pool_[i] = new Node;
  }
  use_node_num_ = 0;
  iter_num_ = 0;
  early_terminate_cost_ = 0.0;

}

void Astar::setMap(const SDFMap::Ptr& hc_map_)
{
  this->map_hc_ = hc_map_;
}

void Astar::setResolution(const double& res) {
  resolution_ = res;
  this->inv_resolution_ = 1.0 / resolution_;
}

int Astar::hc_search(const Eigen::Vector3d& start_pt, const Eigen::Vector3d& end_pt)
{ 
  NodePtr cur_node = path_node_pool_[0];
  cur_node->parent = NULL;
  cur_node->position = start_pt;
  map_hc_->posToIndex_hc(start_pt, cur_node->index);
  cur_node->g_score = 0.0;
  cur_node->f_score = lambda_heu_hc_ * getDiagHeu(cur_node->position, end_pt);

  Eigen::Vector3i end_index;
  map_hc_->posToIndex_hc(end_pt, end_index);

  open_set_.push(cur_node);
  open_set_map_.insert(make_pair(cur_node->index, cur_node));
  use_node_num_ += 1;

  const auto t1 = ros::Time::now();
  /* ---------- search loop ---------- */
  while (!open_set_.empty()) {
    cur_node = open_set_.top();
    bool reach_end = abs(cur_node->index(0) - end_index(0)) <= 1 &&
        abs(cur_node->index(1) - end_index(1)) <= 1 && abs(cur_node->index(2) - end_index(2)) <= 1;
    if (reach_end) {
      backtrack(cur_node, end_pt);
      return REACH_END;
    }

    // Early termination if time up
    if ((ros::Time::now() - t1).toSec() > 0.05) {
      // std::cout << "early" << endl;
      early_terminate_cost_ = cur_node->g_score + getDiagHeu(cur_node->position, end_pt);
      return NO_PATH;
    }

    open_set_.pop();
    open_set_map_.erase(cur_node->index);
    close_set_map_.insert(make_pair(cur_node->index, 1));
    iter_num_ += 1;

    Eigen::Vector3d cur_pos = cur_node->position;
    Eigen::Vector3d nbr_pos;
    Eigen::Vector3d step;
    Eigen::Vector3d stepIdx;

    for (double dx = -resolution_; dx <= resolution_ + 1e-3; dx += resolution_)
      for (double dy = -resolution_; dy <= resolution_ + 1e-3; dy += resolution_)
        for (double dz = -resolution_; dz <= resolution_ + 1e-3; dz += resolution_) 
        {
          step << dx, dy, dz;
          if (step.norm() < 1e-3) continue;

          nbr_pos = cur_pos + step;
          Vector3i nbr_safe, nbr_safe_real;
          map_hc_->posToIndex_hc(nbr_pos, nbr_safe);
          map_hc_->posToIndex(nbr_pos, nbr_safe_real);

          // Check safety
          // if (zFlag == true)
          // {
          //   if (nbr_pos(2) < groundz + safeheight)
          //     continue;
          // }
          if (!map_hc_->isInMap_hc(nbr_safe)) continue;
          if (!map_hc_->isInMap(nbr_safe_real)) continue;
          if (map_hc_->get_Internal(nbr_safe) == SDFMap::OCCUPANCY::HC_INTERNAL || (int)map_hc_->hcmd_->occupancy_buffer_hc_[map_hc_->toAddress_hc(nbr_safe)] == 1 || (int) map_hc_->hcmd_->occupancy_inflate_buffer_hc_[map_hc_->toAddress_hc(nbr_safe)] == 1 || map_hc_->getOccupancy(nbr_safe_real) == SDFMap::OCCUPANCY::OCCUPIED)
          {
            continue;
          }

          bool safe = true;
          Vector3d dir = nbr_pos - cur_pos;
          double len = dir.norm();
          dir.normalize();
          for (double l = this->resolution_; l < len; l += this->resolution_) {
            Vector3d ckpt = cur_pos + l * dir;
            Vector3i ckpt_idx_, ckpt_idx_real;
            map_hc_->posToIndex_hc(ckpt, ckpt_idx_);
            map_hc_->posToIndex(ckpt, ckpt_idx_real);
            // if (zFlag == true)
            // {
            //   if (ckpt(2) < groundz + safeheight)
            //   {
            //     safe = false;
            //     break;
            //   }
            // }
            if (!map_hc_->isInMap_hc(ckpt_idx_)) continue;
            if (!map_hc_->isInMap(ckpt_idx_real)) continue;
            if (map_hc_->get_Internal(ckpt) == SDFMap::OCCUPANCY::HC_INTERNAL || (int)map_hc_->hcmd_->occupancy_buffer_hc_[map_hc_->toAddress_hc(ckpt_idx_)] == 1 || (int) map_hc_->hcmd_->occupancy_inflate_buffer_hc_[map_hc_->toAddress_hc(ckpt_idx_)] == 1 || map_hc_->getOccupancy(ckpt_idx_real) == SDFMap::OCCUPANCY::OCCUPIED) 
            {
              safe = false;
              break;
            }
          }
          if (!safe) continue;

          // Check not in close set
          Eigen::Vector3i nbr_idx;
          map_hc_->posToIndex_hc(nbr_pos, nbr_idx);
          if (close_set_map_.find(nbr_idx) != close_set_map_.end()) continue;

          NodePtr neighbor;
          double tmp_g_score = step.norm() + cur_node->g_score;
          auto node_iter = open_set_map_.find(nbr_idx);
          if (node_iter == open_set_map_.end()) {
            neighbor = path_node_pool_[use_node_num_];
            use_node_num_ += 1;
            if (use_node_num_ == allocate_num_) {
              // cout << "run out of node pool." << endl;
              return NO_PATH;
            }
            neighbor->index = nbr_idx;
            neighbor->position = nbr_pos;
          } else if (tmp_g_score < node_iter->second->g_score) {
            neighbor = node_iter->second;
          } else
            continue;

          neighbor->parent = cur_node;
          neighbor->g_score = tmp_g_score;
          neighbor->f_score = tmp_g_score + lambda_heu_hc_ * getDiagHeu(nbr_pos, end_pt);
          open_set_.push(neighbor);
          open_set_map_[nbr_idx] = neighbor;
        }
  }
  // cout << "open set empty, no path!" << endl;
  // cout << "use node num: " << use_node_num_ << endl;
  // cout << "iter num: " << iter_num_ << endl;
  return NO_PATH;
}

int Astar::hc_occ_search(const Eigen::Vector3d& start_pt, const Eigen::Vector3d& end_pt)
{
  NodePtr cur_node = path_node_pool_[0];
  cur_node->parent = NULL;
  cur_node->position = start_pt;
  map_hc_->posToIndex_hc(start_pt, cur_node->index);
  cur_node->g_score = 0.0;
  cur_node->f_score = lambda_heu_hc_ * getDiagHeu(cur_node->position, end_pt);

  Eigen::Vector3i end_index;
  map_hc_->posToIndex_hc(end_pt, end_index);

  open_set_.push(cur_node);
  open_set_map_.insert(make_pair(cur_node->index, cur_node));
  use_node_num_ += 1;

  const auto t1 = ros::Time::now();
  /* ---------- search loop ---------- */
  while (!open_set_.empty()) {
    cur_node = open_set_.top();
    bool reach_end = abs(cur_node->index(0) - end_index(0)) <= 1 &&
        abs(cur_node->index(1) - end_index(1)) <= 1 && abs(cur_node->index(2) - end_index(2)) <= 1;
    if (reach_end) {
      backtrack(cur_node, end_pt);
      return REACH_END;
    }

    // Early termination if time up
    if ((ros::Time::now() - t1).toSec() > 0.05) {
      // std::cout << "early" << endl;
      early_terminate_cost_ = cur_node->g_score + getDiagHeu(cur_node->position, end_pt);
      return NO_PATH;
    }

    open_set_.pop();
    open_set_map_.erase(cur_node->index);
    close_set_map_.insert(make_pair(cur_node->index, 1));
    iter_num_ += 1;

    Eigen::Vector3d cur_pos = cur_node->position;
    Eigen::Vector3d nbr_pos;
    Eigen::Vector3d step;
    Eigen::Vector3d stepIdx;

    for (double dx = -resolution_; dx <= resolution_ + 1e-3; dx += resolution_)
      for (double dy = -resolution_; dy <= resolution_ + 1e-3; dy += resolution_)
        for (double dz = -resolution_; dz <= resolution_ + 1e-3; dz += resolution_) 
        {
          step << dx, dy, dz;
          if (step.norm() < 1e-3) continue;

          nbr_pos = cur_pos + step;
          Vector3i nbr_safe, nbr_safe_real;
          map_hc_->posToIndex_hc(nbr_pos, nbr_safe);
          map_hc_->posToIndex(nbr_pos, nbr_safe_real);
          // Check safety
          // if (zFlag == true)
          // {
          //   if (nbr_pos(2) < groundz + safeheight)
          //     continue;
          // }
          if (!map_hc_->isInMap_hc(nbr_safe)) continue;
          if (!map_hc_->isInMap(nbr_safe_real)) continue;
          if (map_hc_->get_Internal(nbr_safe) == SDFMap::OCCUPANCY::HC_INTERNAL || (int)map_hc_->hcmd_->occupancy_buffer_hc_[map_hc_->toAddress_hc(nbr_safe)] == 1 || map_hc_->getOccupancy(nbr_safe_real) == SDFMap::OCCUPANCY::OCCUPIED)
          {
            continue;
          }

          bool safe = true;
          Vector3d dir = nbr_pos - cur_pos;
          double len = dir.norm();
          dir.normalize();
          for (double l = this->resolution_; l < len; l += this->resolution_) {
            Vector3d ckpt = cur_pos + l * dir;
            Vector3i ckpt_idx_, ckpt_idx_real;
            map_hc_->posToIndex_hc(ckpt, ckpt_idx_);
            map_hc_->posToIndex(ckpt, ckpt_idx_real);
            // if (zFlag == true)
            // {
            //   if (ckpt(2) < groundz + safeheight)
            //   {
            //     safe = false;
            //     break;
            //   }
            // }
            if (!map_hc_->isInMap_hc(ckpt_idx_)) continue;
            if (!map_hc_->isInMap(ckpt_idx_real)) continue;
            if (map_hc_->get_Internal(ckpt) == SDFMap::OCCUPANCY::HC_INTERNAL || (int)map_hc_->hcmd_->occupancy_buffer_hc_[map_hc_->toAddress_hc(ckpt_idx_)] == 1 || map_hc_->getOccupancy(ckpt_idx_real) == SDFMap::OCCUPANCY::OCCUPIED) 
            {
              safe = false;
              break;
            }
          }
          if (!safe) continue;

          // Check not in close set
          Eigen::Vector3i nbr_idx;
          map_hc_->posToIndex_hc(nbr_pos, nbr_idx);
          if (close_set_map_.find(nbr_idx) != close_set_map_.end()) continue;

          NodePtr neighbor;
          double tmp_g_score = step.norm() + cur_node->g_score;
          auto node_iter = open_set_map_.find(nbr_idx);
          if (node_iter == open_set_map_.end()) {
            neighbor = path_node_pool_[use_node_num_];
            use_node_num_ += 1;
            if (use_node_num_ == allocate_num_) {
              // cout << "run out of node pool." << endl;
              return NO_PATH;
            }
            neighbor->index = nbr_idx;
            neighbor->position = nbr_pos;
          } else if (tmp_g_score < node_iter->second->g_score) {
            neighbor = node_iter->second;
          } else
            continue;

          neighbor->parent = cur_node;
          neighbor->g_score = tmp_g_score;
          neighbor->f_score = tmp_g_score + lambda_heu_hc_ * getDiagHeu(nbr_pos, end_pt);
          open_set_.push(neighbor);
          open_set_map_[nbr_idx] = neighbor;
        }
  }
  // cout << "open set empty, no path!" << endl;
  // cout << "use node num: " << use_node_num_ << endl;
  // cout << "iter num: " << iter_num_ << endl;
  return NO_PATH;
}

int Astar::wholeSearch(const Eigen::Vector3d& start_pt, const Eigen::Vector3d& end_pt)
{
  const int original_step = std::max(1, static_cast<int>(std::round(this->resolution_ / std::max(1e-6, this->map_hc_->mp_->resolution_))));
  const int max_step = std::max(1, this->whole_search_max_step_);
  std::vector<int> cas_steps;
  cas_steps.reserve(max_step);
  for (int step = 1; step <= max_step; ++step) cas_steps.push_back(step);
  if (std::find(cas_steps.begin(), cas_steps.end(), original_step) == cas_steps.end())
    cas_steps.insert(cas_steps.begin(), original_step);

  for (const int step : cas_steps)
  {
    this->reset();
    int result = this->wholeSearchAtStep(start_pt, end_pt, step);
    if (result == REACH_END)
    {
      this->shortenWholeSearchPath();
      this->setResolution(this->map_hc_->mp_->resolution_ * original_step);
      ROS_INFO("\033[34mA* [Cascaded Whole Search] -> Succeed with step size : %d. \033[32m", step);
      return REACH_END;
    }
  }

  this->setResolution(this->map_hc_->mp_->resolution_ * original_step);
  return NO_PATH;
}

int Astar::wholeSearchAtStep(const Eigen::Vector3d& start_pt, const Eigen::Vector3d& end_pt, const int step)
{
  const int search_step = std::max(1, step);
  this->setResolution(this->map_hc_->mp_->resolution_ * search_step);

  NodePtr cur_node = path_node_pool_[0];
  cur_node->parent = NULL;
  cur_node->position = start_pt;
  map_hc_->posToIndex(start_pt, cur_node->index);
  cur_node->g_score = 0.0;
  cur_node->f_score = lambda_heu_hc_ * getDiagHeu(cur_node->position, end_pt);

  Eigen::Vector3i end_index;
  map_hc_->posToIndex(end_pt, end_index);

  open_set_.push(cur_node);
  open_set_map_.insert(make_pair(cur_node->index, cur_node));
  use_node_num_ += 1;

  const auto t1 = ros::Time::now();
  /* ---------- search loop ---------- */
  while (!open_set_.empty()) {
    cur_node = open_set_.top();
    bool reach_end = abs(cur_node->index(0) - end_index(0)) <= search_step &&
        abs(cur_node->index(1) - end_index(1)) <= search_step && abs(cur_node->index(2) - end_index(2)) <= search_step;
    if (reach_end) {
      backtrack(cur_node, end_pt);
      return REACH_END;
    }

    // Early termination if time up
    if ((ros::Time::now() - t1).toSec() > max_search_time_) {
      early_terminate_cost_ = cur_node->g_score + getDiagHeu(cur_node->position, end_pt);
      ROS_WARN("A* [Whole Search] @ %lf -> Early termination", this->resolution_);
      return NO_PATH;
    }

    open_set_.pop();
    open_set_map_.erase(cur_node->index);
    close_set_map_.insert(make_pair(cur_node->index, 1));
    iter_num_ += 1;

    const Eigen::Vector3d cur_pos = cur_node->position;
    const Eigen::Vector3i cur_idx = cur_node->index;
    Eigen::Vector3d nbr_pos;
    Eigen::Vector3d step_vec;
    Eigen::Vector3i nbr_idx;

    for (int dx = -search_step; dx <= search_step; dx += search_step)
      for (int dy = -search_step; dy <= search_step; dy += search_step)
        for (int dz = -search_step; dz <= search_step; dz += search_step) 
        {
          Eigen::Vector3i step_idx(dx, dy, dz);
          if (step_idx.squaredNorm() == 0) continue;

          nbr_idx = cur_idx + step_idx;
          if (close_set_map_.find(nbr_idx) != close_set_map_.end()) continue;
          if (!map_hc_->isInMap(nbr_idx)) continue;

          map_hc_->indexToPos(nbr_idx, nbr_pos);
          if (!isWholeSearchPointSafe(nbr_pos)) continue;
          if (!isWholeSearchSegmentSafe(cur_pos, nbr_pos)) continue;

          NodePtr neighbor;
          step_vec = nbr_pos - cur_pos;
          double tmp_g_score = step_vec.norm() + cur_node->g_score;
          auto node_iter = open_set_map_.find(nbr_idx);
          if (node_iter == open_set_map_.end()) {
            neighbor = path_node_pool_[use_node_num_];
            use_node_num_ += 1;
            if (use_node_num_ == allocate_num_) {
              cout << "run out of node pool." << endl;
              return NO_PATH;
            }
            neighbor->index = nbr_idx;
            neighbor->position = nbr_pos;
          } else if (tmp_g_score < node_iter->second->g_score) {
            neighbor = node_iter->second;
          } else
            continue;

          neighbor->parent = cur_node;
          neighbor->g_score = tmp_g_score;
          neighbor->f_score = tmp_g_score + lambda_heu_hc_ * getDiagHeu(nbr_pos, end_pt);
          open_set_.push(neighbor);
          open_set_map_[nbr_idx] = neighbor;
        }
  }

  return NO_PATH;
}

bool Astar::isWholeSearchPointSafe(const Eigen::Vector3d& pos)
{
  Eigen::Vector3i hc_idx, real_idx;
  map_hc_->posToIndex_hc(pos, hc_idx);
  map_hc_->posToIndex(pos, real_idx);

  if (!map_hc_->isInMap_hc(hc_idx)) return false;
  if (!map_hc_->isInMap(real_idx)) return false;
  if ((int)map_hc_->hcmd_->occupancy_inflate_buffer_hc_[map_hc_->toAddress_hc(hc_idx)] == 1) return false;
  if (map_hc_->getOccupancy(real_idx) == SDFMap::OCCUPANCY::OCCUPIED) return false;
  if (map_hc_->getInflateOccupancy(real_idx) != 0) return false;

  return true;
}

bool Astar::isWholeSearchSegmentSafe(const Eigen::Vector3d& start, const Eigen::Vector3d& end)
{
  Eigen::Vector3d dir = end - start;
  const double len = dir.norm();
  if (len < 1e-6) return true;
  dir.normalize();

  const double check_res = std::max(1e-3, this->map_hc_->mp_->resolution_);
  for (double l = check_res; l < len; l += check_res) 
  {
    Eigen::Vector3d ckpt = start + l * dir;
    if (!isWholeSearchPointSafe(ckpt)) return false;
  }

  return true;
}

void Astar::shortenWholeSearchPath()
{
  if ((int)this->path_nodes_.size() <= 2) return;

  const double interval = this->whole_search_shorten_interval_ > 1e-3 ? this->whole_search_shorten_interval_ : 2.0;
  vector<Eigen::Vector3d> shortened_path = {this->path_nodes_.front()};

  for (int i = 1; i < (int)this->path_nodes_.size() - 1; ++i)
  {
    if ((this->path_nodes_[i] - shortened_path.back()).norm() > interval)
      shortened_path.push_back(this->path_nodes_[i]);
    else if (!isWholeSearchSegmentSafe(shortened_path.back(), this->path_nodes_[i + 1]))
      shortened_path.push_back(this->path_nodes_[i]);
  }

  if ((this->path_nodes_.back() - shortened_path.back()).norm() > 1e-3)
    shortened_path.push_back(this->path_nodes_.back());

  if ((int)shortened_path.size() > 2 &&
      (shortened_path.back() - shortened_path[(int)shortened_path.size() - 2]).norm() < interval &&
      isWholeSearchSegmentSafe(shortened_path[(int)shortened_path.size() - 3], shortened_path.back()))
    shortened_path.erase(shortened_path.end() - 2);

  this->path_nodes_ = shortened_path;
}

double Astar::getEarlyTerminateCost() {
  return early_terminate_cost_;
}

void Astar::reset() {
  open_set_map_.clear();
  close_set_map_.clear();
  path_nodes_.clear();

  std::priority_queue<NodePtr, std::vector<NodePtr>, NodeComparator0> empty_queue;
  open_set_.swap(empty_queue);
  for (int i = 0; i < use_node_num_; i++) {
    path_node_pool_[i]->parent = NULL;
  }
  use_node_num_ = 0;
  iter_num_ = 0;
}

double Astar::pathLength(const vector<Eigen::Vector3d>& path) {
  double length = 0.0;
  if (path.size() < 2) return length;
  for (int i = 0; i < (int)path.size() - 1; ++i)
    length += (path[i + 1] - path[i]).norm();
  return length;
}

void Astar::backtrack(const NodePtr& end_node, const Eigen::Vector3d& end) {
  path_nodes_.push_back(end);
  path_nodes_.push_back(end_node->position);
  NodePtr cur_node = end_node;
  while (cur_node->parent != NULL) {
    cur_node = cur_node->parent;
    path_nodes_.push_back(cur_node->position);
  }
  reverse(path_nodes_.begin(), path_nodes_.end());
}

std::vector<Eigen::Vector3d> Astar::getPath() {
  return path_nodes_;
}

double Astar::getDiagHeu(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2) {
  double dx = fabs(x1(0) - x2(0));
  double dy = fabs(x1(1) - x2(1));
  double dz = fabs(x1(2) - x2(2));
  double h = 0.0;
  double diag = min(min(dx, dy), dz);
  dx -= diag;
  dy -= diag;
  dz -= diag;

  if (dx < 1e-4) {
    h = 1.0 * sqrt(3.0) * diag + sqrt(2.0) * min(dy, dz) + 1.0 * abs(dy - dz);
  }
  if (dy < 1e-4) {
    h = 1.0 * sqrt(3.0) * diag + sqrt(2.0) * min(dx, dz) + 1.0 * abs(dx - dz);
  }
  if (dz < 1e-4) {
    h = 1.0 * sqrt(3.0) * diag + sqrt(2.0) * min(dx, dy) + 1.0 * abs(dx - dy);
  }
  return tie_breaker_ * h;
}

double Astar::getManhHeu(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2) {
  double dx = fabs(x1(0) - x2(0));
  double dy = fabs(x1(1) - x2(1));
  double dz = fabs(x1(2) - x2(2));
  return tie_breaker_ * (dx + dy + dz);
}

double Astar::getEuclHeu(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2) {
  return tie_breaker_ * (x2 - x1).norm();
}

std::vector<Eigen::Vector3d> Astar::getVisited() {
  vector<Eigen::Vector3d> visited;
  for (int i = 0; i < use_node_num_; ++i)
    visited.push_back(path_node_pool_[i]->position);
  return visited;
}

void Astar::posToIndex(const Eigen::Vector3d& pt, Eigen::Vector3i& idx) {
  idx = ((pt - origin_) * inv_resolution_).array().floor().cast<int>();
}

} // namespace flyco
