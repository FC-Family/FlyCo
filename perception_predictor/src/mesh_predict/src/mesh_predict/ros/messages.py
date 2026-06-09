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

import numpy as np
import rospy
from geometry_msgs.msg import Point
from perception_msgs.msg import Mesh
from sensor_msgs.msg import PointCloud2 as PointCloud2Msg, PointField
from std_msgs.msg import Float64MultiArray, Int64MultiArray, MultiArrayDimension
from visualization_msgs.msg import Marker


def build_xyz_pointcloud(points, frame_id="world"):
    """Serialize XYZ points into a PointCloud2 message."""

    points = np.asarray(points, dtype=np.float32)

    pc_msg = PointCloud2Msg()
    pc_msg.header.stamp = rospy.Time.now()
    pc_msg.header.frame_id = frame_id
    pc_msg.width = points.shape[0]
    pc_msg.height = 1
    pc_msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    pc_msg.is_bigendian = False
    pc_msg.point_step = 12
    pc_msg.row_step = 12 * points.shape[0]
    pc_msg.is_dense = True
    pc_msg.data = points.astype(np.float32, copy=False).tobytes()
    return pc_msg


def build_mesh_message(mesh_v, mesh_f, trajectory_id=1):
    """Serialize mesh vertices and faces into perception_msgs/Mesh."""

    mesh_v = np.asarray(mesh_v, dtype=np.float64)
    mesh_f = np.asarray(mesh_f, dtype=np.int64)

    mesh_msg = Mesh()
    mesh_msg.trajectory_id = trajectory_id
    mesh_msg.vertex_num = mesh_v.shape[0]
    mesh_msg.face_num = mesh_f.shape[0]

    v_pub = mesh_v.T
    vertex_matrix = Float64MultiArray()
    vertex_matrix.layout.dim = [
        MultiArrayDimension(
            label="height_V", size=v_pub.shape[0], stride=v_pub.shape[1]
        ),
        MultiArrayDimension(label="width_V", size=v_pub.shape[1], stride=1),
    ]
    vertex_matrix.data = mesh_v.flatten().tolist()

    f_pub = mesh_f.T
    face_matrix = Int64MultiArray()
    face_matrix.layout.dim = [
        MultiArrayDimension(
            label="height_F", size=f_pub.shape[0], stride=f_pub.shape[1]
        ),
        MultiArrayDimension(label="width_F", size=f_pub.shape[1], stride=1),
    ]
    face_matrix.data = mesh_f.flatten().tolist()

    mesh_msg.vertex_matrix = vertex_matrix
    mesh_msg.face_matrix = face_matrix
    return mesh_msg


def build_mesh_marker(mesh_v, mesh_f, frame_id="world"):
    """Build a TRIANGLE_LIST marker for RViz mesh visualization."""

    mesh_marker = Marker()
    mesh_marker.header.frame_id = frame_id
    mesh_marker.header.stamp = rospy.Time.now()
    mesh_marker.ns = "mesh_vis"
    mesh_marker.id = 0
    mesh_marker.type = Marker.TRIANGLE_LIST
    mesh_marker.pose.orientation.w = 1.0
    mesh_marker.scale.x = 1.0
    mesh_marker.scale.y = 1.0
    mesh_marker.scale.z = 1.0
    mesh_marker.color.a = 0.5
    mesh_marker.color.r = 0.4
    mesh_marker.color.g = 0.3
    mesh_marker.color.b = 0.5

    for face in mesh_f:
        for idx in face:
            point = Point()
            point.x = mesh_v[idx][0]
            point.y = mesh_v[idx][1]
            point.z = mesh_v[idx][2]
            mesh_marker.points.append(point)

    return mesh_marker
