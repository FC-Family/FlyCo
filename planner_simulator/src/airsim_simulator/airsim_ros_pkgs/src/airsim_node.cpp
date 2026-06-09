/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Sept. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the execution file of the AirSim ROS wrapper node for FlyCo
 *                   simulation.
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

#include "ros/ros.h"
#include "airsim_ros_wrapper.h"
#include <ros/spinner.h>

int main(int argc, char ** argv)
{
    ros::init(argc, argv, "airsim_node");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    std::string host_ip = "localhost";
    nh_private.getParam("host_ip", host_ip);
    AirsimROSWrapper airsim_ros_wrapper(nh, nh_private, host_ip);

    if (airsim_ros_wrapper.is_used_img_timer_cb_queue_)
    {
        airsim_ros_wrapper.img_async_spinner_.start();
    }

    if (airsim_ros_wrapper.is_used_lidar_timer_cb_queue_)
    {
        airsim_ros_wrapper.lidar_async_spinner_.start();
    }

    ros::spin();

    return 0;
} 