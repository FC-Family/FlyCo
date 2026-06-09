# ⭐⭐⭐******************************************************************⭐⭐⭐
# Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
#                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
# Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
#                   https://gy920.github.io/
# Date         :    Jun. 2026
# E-mail       :    cfengag at connect dot ust dot hk
#                   shverses at gmail dot com
# Description  :    This file is part of the FlyCo Perception public ROS runtime.
# Copyright    :    Copyright (c) 2026 Chen Feng and Guiyong Zheng.
# License      :    PolyForm Noncommercial License 1.0.0
#                   <https://polyformproject.org/licenses/noncommercial/1.0.0/>
# Project      :    FlyCo: Foundation Model-Empowered Drones for Autonomous 3D Structure Scanning in Open-World Environments
# Website      :    https://hkust-aerial-robotics.github.io/FC-Planner/
# ⭐⭐⭐******************************************************************⭐⭐⭐

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2 as PointCloud2Msg
from visualization_msgs.msg import Marker

from perception_msgs.msg import Mesh

from mesh_predict.ros.messages import (
    build_mesh_marker,
    build_mesh_message,
    build_xyz_pointcloud,
)


class MeshPredictPublishers:
    """ROS publisher bundle for official runtime outputs."""

    def __init__(self, params):
        self.params = params
        self.bridge = CvBridge()

        self.mesh_pub = rospy.Publisher(
            self.params.predicted_mesh_topic, Mesh, queue_size=1
        )
        self.vis_mesh_pub = rospy.Publisher(
            self.params.mesh_vis_topic, Marker, queue_size=10
        )
        self.predict_pc_pub = rospy.Publisher(
            self.params.predicted_pointcloud_topic, PointCloud2Msg, queue_size=1
        )
        self.pure_predict_pc_pub = rospy.Publisher(
            "~pure_predicted_pointcloud", PointCloud2Msg, queue_size=1
        )
        self.debug_pc_pub = rospy.Publisher(
            "~debug_pointcloud", PointCloud2Msg, queue_size=1
        )
        self.partial_pc_pub = rospy.Publisher(
            self.params.partial_pointcloud_topic, PointCloud2Msg, queue_size=1
        )
        self.debug_img_pub = rospy.Publisher(
            self.params.predicted_input_imgs_topic, Image, queue_size=1
        )

    def publish_pointcloud_prediction(self, result):
        self.predict_pc_pub.publish(
            build_xyz_pointcloud(result["predicted_pointcloud"])
        )
        self.pure_predict_pc_pub.publish(
            build_xyz_pointcloud(result["pure_predicted_pointcloud"])
        )
        self.publish_debug_image(result["debug_image"])

    def publish_mesh_prediction(self, result):
        self.partial_pc_pub.publish(build_xyz_pointcloud(result["partial_pointcloud"]))
        self.publish_mesh(result["mesh_vertices"], result["mesh_faces"])
        self.publish_mesh_marker(result["mesh_vertices"], result["mesh_faces"])
        self.debug_pc_pub.publish(build_xyz_pointcloud(result["debug_pointcloud"]))

    def publish_mesh_marker(self, mesh_v, mesh_f):
        delete_marker = Marker()
        delete_marker.header.frame_id = "world"
        delete_marker.header.stamp = rospy.Time.now()
        delete_marker.ns = "mesh_vis"
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL
        self.vis_mesh_pub.publish(delete_marker)

        mesh_marker = build_mesh_marker(mesh_v, mesh_f)
        mesh_marker.action = Marker.ADD
        self.vis_mesh_pub.publish(mesh_marker)

    def publish_debug_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_msg = self.bridge.cv2_to_imgmsg(img, "8UC3")
        self.debug_img_pub.publish(img_msg)

    def publish_mesh(self, mesh_v, mesh_f):
        self.mesh_pub.publish(build_mesh_message(mesh_v, mesh_f))
