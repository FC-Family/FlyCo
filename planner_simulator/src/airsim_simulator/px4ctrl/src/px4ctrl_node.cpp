/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Sept. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the execution file of the PX4 control node for FlyCo
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

#include <ros/ros.h>
#include "PX4CtrlFSM.h"

//#include <quadrotor_msgs/SO3Command.h>
#include <geometry_msgs/PoseStamped.h>
#include <quadrotor_msgs/PositionCommand.h>
//#include <geometry_msgs/PoseWithCovarianceStamped.h>
//#include <std_msgs/Header.h>
//#include <geometry_msgs/Vector3Stamped.h>
#include <signal.h>
#include "std_msgs/Float32.h"

PX4CtrlFSM* pFSM;

void mySigintHandler(int sig) {
    ROS_INFO("[PX4Ctrl] exit...");
    ros::shutdown();
}

int main(int argc, char* argv[]) {
    ros::init(argc, argv, "px4ctrl");
    ros::NodeHandle nh("~");
    signal(SIGINT, mySigintHandler);
    Parameter_t param;
    Controller controller(param);
    HovThrKF hov_thr_kf(param);
    PX4CtrlFSM fsm(param, controller, hov_thr_kf);
    pFSM = &fsm;

    param.config_from_ros_handle(nh);
    param.init();//recompute the full thrust.
    fsm.hov_thr_kf.init();
    fsm.hov_thr_kf.set_hov_thr(param.hov_percent);//x(0) = hov_percent
    
    ROS_INFO("Initial value for hov_thr set to %.2f/%.2f",
             fsm.hov_thr_kf.get_hov_thr(),
             param.mass * param.gra / param.full_thrust);
    ROS_INFO("Hovering thrust kalman filter is %s.",
             param.hover.use_hov_percent_kf ? "used" : "NOT used");

    fsm.controller.config();//

    // ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>("/mavros/state", 
    //                                                              10,
    //                                                              boost::bind(&State_Data_t::feed, &fsm.state_data, _1));
    ros::Subscriber odom_sub =
        nh.subscribe<nav_msgs::Odometry>("odom",
                                         100,
                                         boost::bind(&Odom_Data_t::feed, &fsm.odom_data, _1),
                                         ros::VoidConstPtr(),
                                         ros::TransportHints().tcpNoDelay());


    ros::Subscriber cmd_sub = 
        nh.subscribe<quadrotor_msgs::PositionCommand>("cmd",
                                                      100,
                                                      boost::bind(&Command_Data_t::feed, &fsm.cmd_data, _1),
                                                      ros::VoidConstPtr(),
                                                      ros::TransportHints().tcpNoDelay());
    //All okay for both above

    ros::Subscriber imu_sub =
        nh.subscribe<sensor_msgs::Imu>("imu",
                                         100,
                                         boost::bind(&Imu_Data_t::feed, &fsm.imu_data, _1),
                                         ros::VoidConstPtr(),
                                         ros::TransportHints().tcpNoDelay());
    fsm.controller.ctrl_FCU_pub = nh.advertise<mavros_msgs::AttitudeTarget>("/setpoint_raw/attitude", 10);
    // fsm.controller.debug_roll_pub = nh.advertise<std_msgs::Float32>("/debug_roll",10);
    // fsm.controller.debug_pitch_pub = nh.advertise<std_msgs::Float32>("/debug_pitch",10);
    // fsm.traj_start_trigger_pub = nh.advertise<geometry_msgs::PoseStamped>("/traj_start_trigger", 10);
    ros::Rate r(param.ctrl_rate);
    // ---- process ----
    while (ros::ok()) {
        r.sleep();
        ros::spinOnce();
        if(fsm.px4_init())
            fsm.process();
    }

    return 0;
}
