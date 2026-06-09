/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Sept. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the joystick teleoperation node for FlyCo simulation.
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

#include<ros/ros.h>
#include<geometry_msgs/Twist.h>
#include <sensor_msgs/Joy.h>
#include<iostream>
#include <mavros_msgs/AttitudeTarget.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace std;

float y = 0;
mavros_msgs::AttitudeTarget v;
ros::Publisher pub;

void timecallback(const ros::TimerEvent& e)
{
    pub.publish(v);
}

void callback(const sensor_msgs::Joy::ConstPtr& Joy, ros::Publisher& pub)
{
    //mask
    v.type_mask = 1;
    //thrust
    float thrust = (Joy->axes[4] + 1)/8.425;
    v.thrust = thrust;
    //orientation
    float r = Joy->axes[0] * (-0.1);
    float p = -Joy->axes[1] * (0.1);
    y += Joy->axes[3] * (0.02);
    Eigen::AngleAxisf roll(r, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitch(p, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yaw(y, Eigen::Vector3f::UnitZ());
    Eigen::Quaternionf quater;
    quater = yaw * pitch * roll;//transfer
    //std::cout<<quater.x()<<std::endl;
    v.orientation.x = quater.x();
    v.orientation.y = quater.y();
    v.orientation.z = quater.z();
    v.orientation.w = quater.w();

    pub.publish(v);

}


int main(int argc,char** argv) {
    ros::init(argc, argv, "joy_node");
    ros::NodeHandle n;
    ros::Subscriber sub;

    ros::Timer exec_timer_ = n.createTimer(ros::Duration(0.01), &timecallback);
    pub = n.advertise<mavros_msgs::AttitudeTarget>("/setpoint_raw/attitude",1);
    sub = n.subscribe<sensor_msgs::Joy>("joy",10, boost::bind(&callback, _1, pub));
    ros::spin();

    return 0;
}