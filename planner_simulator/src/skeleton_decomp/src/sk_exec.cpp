/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    May. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the execution file of skeleton-based space decomposition,
 *                   which individually implements the SSD module.
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
#include "plan_utils/visibility_st.hpp"
#include <igl/read_triangle_mesh.h>
#include <ros/ros.h>

using namespace flyco;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "sk_exec");
  ros::NodeHandle nh("~");

  string mesh_file;
  nh.param("sk/input_mesh", mesh_file, string("null"));
  if (mesh_file.empty() || mesh_file == "null")
  {
    ROS_ERROR("[sk_exec] Missing required param '~sk/input_mesh'.");
    return -1;
  }

  // Standalone SSD unit test path mirrors FCPlanner_PP::setInput() + plan():
  // load the mesh once here, provide mesh/scene directly to sk_decomp, and let
  // sk_decomp::main() reuse them instead of re-reading the mesh internally.
  nh.setParam("input/independent", false);

  sk_decomp sd;
  sd.init(nh);

  Eigen::MatrixXd mesh_V;
  Eigen::MatrixXi mesh_F;
  if (!igl::read_triangle_mesh(mesh_file, mesh_V, mesh_F) ||
      mesh_V.rows() <= 0 || mesh_F.rows() <= 0)
  {
    ROS_ERROR("[sk_exec] Failed to read mesh from '%s'.", mesh_file.c_str());
    return -1;
  }

  ROS_INFO("[sk_exec] Loaded mesh '%s': V=%ldx%ld, F=%ldx%ld.",
           mesh_file.c_str(),
           mesh_V.rows(), mesh_V.cols(),
           mesh_F.rows(), mesh_F.cols());

  RTCScene scene = nullptr;
  sd.set_mesh(mesh_V, mesh_F);
  visibility_st::mesh2scene(mesh_V, mesh_F, scene);
  if (scene == nullptr)
  {
    ROS_ERROR("[sk_exec] Failed to build Embree scene from '%s'.", mesh_file.c_str());
    return -1;
  }
  sd.setScene(scene);

  sd.main();
  ros::Duration(1.0).sleep();
  ros::spin();

  if (scene != nullptr)
    rtcReleaseScene(scene);
  return 0; 
}
