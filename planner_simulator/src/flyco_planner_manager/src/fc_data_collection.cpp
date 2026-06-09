/*⭐⭐⭐******************************************************************⭐⭐⭐*
 * Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
 *                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
 * Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
 *                   https://gy920.github.io/
 * Date         :    Sept. 2024
 * E-mail       :    cfengag at connect dot ust dot hk
 *                   shverses at gmail dot com
 * Description  :    This file is the execution file of data collection in FlyCo.
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
#include <Eigen/Core>
#include <Eigen/Dense>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Joy.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>
#include <tf/tf.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <boost/circular_buffer.hpp>

using namespace std;
using std::shared_ptr;
using std::unique_ptr;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry, geometry_msgs::QuaternionStamped> SyncPolicyImagePoseGimbal;
typedef unique_ptr<message_filters::Synchronizer<SyncPolicyImagePoseGimbal>> SynchronizerImagePoseGimbal;

// Param
bool open_collection_ = false;
bool sim_flag_ = false;
bool collection_trigger_ = false;
bool collected_cam_info_ = false;
bool last_rb_pressed_ = false;
double collection_fps_ = 1.0;
double ext_x_, ext_y_, ext_z_;
Eigen::Vector3d cam2body_t_vector_;
Eigen::Matrix3d cam2body_R_matrix_;
string img_folder_;

double h_, w_, fx_, fy_, cx_, cy_; // intrinsic parameters
int img_id_ = 1;
ofstream intrinsic_file_, extrinsic_file_, flight_log_file_;
unique_ptr<ros::Rate> collection_rate_ = nullptr;

// ROS Service
ros::Subscriber joy_sub_, cam_sub_;
unique_ptr<message_filters::Subscriber<sensor_msgs::Image>> image_sub_;
unique_ptr<message_filters::Subscriber<nav_msgs::Odometry>> pose_sub_;
unique_ptr<message_filters::Subscriber<geometry_msgs::QuaternionStamped>> gimbal_sub_;
SynchronizerImagePoseGimbal sync_image_pose_gimbal_;
ros::Timer collection_timer_;

// Buffer
boost::circular_buffer<sensor_msgs::ImageConstPtr> buffer_img_;
boost::circular_buffer<nav_msgs::OdometryConstPtr> buffer_pose_;
boost::circular_buffer<geometry_msgs::QuaternionStampedConstPtr> buffer_quat_;

void triggerDataCollection()
{
    if (collection_trigger_ == true) return;

    ROS_ERROR("Received data collection trigger.");
    collection_trigger_ = true;

    return;
}

void joyCallback(const sensor_msgs::Joy::ConstPtr& Joy)
{
    if (!Joy) return;

    constexpr int JOY_RB_BUTTON = 5;
    bool rb_pressed = Joy->buttons.size() > JOY_RB_BUTTON && Joy->buttons[JOY_RB_BUTTON] == true;

    if (rb_pressed && !last_rb_pressed_)
    {
        ROS_WARN("[FlyCo Joy] RB pressed: trigger data collection.");
        triggerDataCollection();
    }

    last_rb_pressed_ = rb_pressed;

    return;
}

void camCallback(const sensor_msgs::CameraInfo& cam)
{
    if (collected_cam_info_ == false)
    {
        h_ = cam.height;
        w_ = cam.width;
        fx_ = cam.K[0];
        fy_ = cam.K[4];
        cx_ = cam.K[2];
        cy_ = cam.K[5];

        collected_cam_info_ = true;
        cam_sub_.shutdown();
    }

    return;
}

void dataCallback(const sensor_msgs::ImageConstPtr& img, const nav_msgs::OdometryConstPtr& pose, const geometry_msgs::QuaternionStampedConstPtr& quat)
{
    if (collection_trigger_ == false) return;

    buffer_img_.push_back(img);
    buffer_pose_.push_back(pose);
    buffer_quat_.push_back(quat);

    return;
}

void collectionCallback(const ros::TimerEvent& e)
{
    if (!collection_trigger_) return;
    if (buffer_img_.empty() || buffer_pose_.empty() || buffer_quat_.empty()) return;

    sensor_msgs::ImageConstPtr img = buffer_img_.back();
    nav_msgs::OdometryConstPtr pose = buffer_pose_.back();
    geometry_msgs::QuaternionStampedConstPtr quat = buffer_quat_.back();

    // * Process Odometry
    tf::Quaternion tfQuat;
    tf::quaternionMsgToTF(quat->quaternion, tfQuat);
    tf::Matrix3x3 m(tfQuat);
    double roll_gimbal, pitch_gimbal, yaw_gimbal;
    m.getRPY(roll_gimbal, pitch_gimbal, yaw_gimbal);

    Eigen::Vector3d body2world_t_vector;
    Eigen::Matrix3d body2world_R_matrix, gimbal2body_R_matrix;
    Eigen::Quaterniond body2world_R_quat, gimbal2body_R_quat;

    body2world_t_vector(0) = pose->pose.pose.position.x; // in world coordinate
    body2world_t_vector(1) = pose->pose.pose.position.y; // in world coordinate
    body2world_t_vector(2) = pose->pose.pose.position.z; // in world coordinate

    tf::Quaternion tfQuat_body;
    tf::quaternionMsgToTF(pose->pose.pose.orientation, tfQuat_body);
    tf::Matrix3x3 m_body(tfQuat_body);
    double roll_body, pitch_body, yaw_body;
    m_body.getRPY(roll_body, pitch_body, yaw_body);

    body2world_R_matrix = Eigen::AngleAxisd(yaw_body, Eigen::Vector3d::UnitZ()) *
                        Eigen::AngleAxisd(pitch_body, Eigen::Vector3d::UnitY()) *
                        Eigen::AngleAxisd(roll_body, Eigen::Vector3d::UnitX());
    body2world_R_quat = Eigen::Quaterniond(body2world_R_matrix);

    if (sim_flag_ == true)
    {
    // ! AirSim Simulation
    yaw_gimbal += yaw_body;
    pitch_gimbal += pitch_body;
    roll_gimbal = 0.0; // ? Fixed with Drone
    // ! ---------------------
    }

    gimbal2body_R_matrix = Eigen::AngleAxisd(yaw_gimbal, -Eigen::Vector3d::UnitZ()) *
                            Eigen::AngleAxisd(pitch_gimbal, -Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxisd(roll_gimbal, Eigen::Vector3d::UnitX()); 
    gimbal2body_R_quat = Eigen::Quaterniond(gimbal2body_R_matrix);

    Eigen::Vector3d camera_pos = body2world_R_matrix * cam2body_t_vector_ + body2world_t_vector;
    Eigen::Quaterniond camera_q = body2world_R_quat * gimbal2body_R_quat;

    // * Extrinsic of Image
    Eigen::Quaterniond ext_q = camera_q.inverse();
    Eigen::Vector3d ext_t = -camera_q.toRotationMatrix().inverse()*camera_pos;

    // * Process Image
    // -> Save Image
    string str, local_str;
    stringstream ss;
    ss << setfill('0') << setw(4) << img_id_;
    string name = ss.str();
    str = img_folder_ + "/imgs/image_" + name + ".jpg";
    local_str = "image_" + name + ".jpg";
    cv_bridge::CvImageConstPtr ptr = cv_bridge::toCvCopy(img, "bgr8");
    cv::imwrite(str, ptr->image);

    // -> Save Intrinsic
    intrinsic_file_.open(img_folder_ + "/intrinsic.txt", ios_base::app);
    intrinsic_file_ << to_string(img_id_) << " PINHOLE" << " " << to_string(w_) << " " << to_string(h_) << " " << to_string(fx_) << " " << to_string(fy_) << " " << to_string(cx_) << " " << to_string(cy_) << "\n";
    intrinsic_file_.close();

    // -> Save Extrinsic
    extrinsic_file_.open(img_folder_ + "/extrinsic.txt", ios_base::app);
    extrinsic_file_ << to_string(img_id_) << " " << to_string(ext_q.w()) << " " << to_string(ext_q.x()) << " " << to_string(ext_q.y()) << " " << to_string(ext_q.z()) << " " << to_string(ext_t(0)) << " " << to_string(ext_t(1)) << " " << to_string(ext_t(2)) << " 1 image_" << name + ".jpg" << "\n";
    extrinsic_file_ << "\n";
    extrinsic_file_.close();

    // -> Save Flight Log Pos
    // double yaw_cam = yaw_body * 180.0 / M_PI;
    // double pitch_cam = pitch_gimbal * 180.0 / M_PI;

    flight_log_file_.open(img_folder_ + "/FlightLog.txt", ios_base::app);
    flight_log_file_ << local_str << " " << to_string(camera_pos(0)) << " " << to_string(camera_pos(1)) << " " << to_string(camera_pos(2)) << "\n";
    // flight_log_file_ << name << " " << to_string(camera_pos(0)) << " " << to_string(camera_pos(1)) << " " << to_string(camera_pos(2)) << " " << to_string(yaw_cam) << " " << to_string(pitch_cam) << "\n";
    flight_log_file_.close();

    img_id_++;
   
    return;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "fc_data");
    ros::NodeHandle nh("~");

    nh.param("flyco_planner/collection",  open_collection_, true);
    if (!open_collection_)
    {
        ROS_WARN("Data collection is closed.");

        ros::Duration(1.0).sleep();
        ros::spin();

        return 0;
    }
    else
    {
        ROS_INFO("Data collection is open.");
    }

    nh.param("flyco_planner/sim",         sim_flag_, true);
    nh.param("flyco_planner/dc_fps",      collection_fps_, -1.0);
    nh.param("flyco_planner/img_dir",     img_folder_, string("null"));
    nh.param("flyco_planner/extrinsic_x", ext_x_, 0.0);
    nh.param("flyco_planner/extrinsic_y", ext_y_, 0.0);
    nh.param("flyco_planner/extrinsic_z", ext_z_, 0.0);
    cam2body_t_vector_ << ext_x_, ext_y_, ext_z_;
    double collection_time = 1.0/collection_fps_;

    // * Buffer
    int num = 20;
    buffer_img_ = boost::circular_buffer<sensor_msgs::ImageConstPtr>(num);
    buffer_pose_ = boost::circular_buffer<nav_msgs::OdometryConstPtr>(num);
    buffer_quat_ = boost::circular_buffer<geometry_msgs::QuaternionStampedConstPtr>(num);

    joy_sub_ = nh.subscribe("/joy", 50, &joyCallback);
    image_sub_.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh, "/flyco_planner/img", 10));
    pose_sub_.reset(new message_filters::Subscriber<nav_msgs::Odometry>(nh, "/flyco_planner/body_pose", 10));
    gimbal_sub_.reset(new message_filters::Subscriber<geometry_msgs::QuaternionStamped>(nh, "/flyco_planner/gimbal_pose", 10));
    sync_image_pose_gimbal_.reset(new message_filters::Synchronizer<SyncPolicyImagePoseGimbal>(
            SyncPolicyImagePoseGimbal(20), *image_sub_, *pose_sub_, *gimbal_sub_));
    sync_image_pose_gimbal_->registerCallback(boost::bind(&dataCallback, _1, _2, _3));
    if (sim_flag_ == true)
        cam_sub_ = nh.subscribe("/flyco_planner/cam_info", 1, &camCallback);
    else
    {
        nh.param("flyco_planner/cam_h",  h_, -1.0);
        nh.param("flyco_planner/cam_w",  w_, -1.0);
        nh.param("flyco_planner/cam_fx", fx_, -1.0);
        nh.param("flyco_planner/cam_fy", fy_, -1.0);
        nh.param("flyco_planner/cam_cx", cx_, -1.0);
        nh.param("flyco_planner/cam_cy", cy_, -1.0);
    }

    ofstream(img_folder_ + "/intrinsic.txt", std::ios_base::trunc).close();
    ofstream(img_folder_ + "/extrinsic.txt", std::ios_base::trunc).close();
    ofstream(img_folder_ + "/FlightLog.txt", std::ios_base::trunc).close();

    collection_timer_ = nh.createTimer(ros::Duration(collection_time), &collectionCallback);

    ros::Duration(1.0).sleep();
    ros::spin();

    return 0;
}
