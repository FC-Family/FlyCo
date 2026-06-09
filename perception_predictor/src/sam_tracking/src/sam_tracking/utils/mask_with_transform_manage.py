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

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
import tf2_ros
import tf
from perception_msgs.msg import MaskWithTransform


class TransformManager:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.mask_with_transform = MaskWithTransform()
        self.mask_with_transform.header = Header(
            stamp=rospy.Time.now(), frame_id="mask_T_frame_id"
        )
        self.mask_with_transform.mask_image = Image()
        self.mask_with_transform.transform = TransformStamped()

    def set_transform(
        self, transform_matrix, frame_id="world", child_frame_id="camera_frame", stamp=None
    ):
        if stamp is None:
            stamp = rospy.Time.now()
        translation = transform_matrix[:3, 3]
        quaternion = tf.transformations.quaternion_from_matrix(transform_matrix)
        self.mask_with_transform.header.stamp = stamp
        self.mask_with_transform.header.frame_id = frame_id
        self.mask_with_transform.transform.header.stamp = stamp
        self.mask_with_transform.transform.header.frame_id = frame_id
        self.mask_with_transform.transform.child_frame_id = child_frame_id
        self.mask_with_transform.transform.transform.translation.x = translation[0]
        self.mask_with_transform.transform.transform.translation.y = translation[1]
        self.mask_with_transform.transform.transform.translation.z = translation[2]
        self.mask_with_transform.transform.transform.rotation.x = quaternion[0]
        self.mask_with_transform.transform.transform.rotation.y = quaternion[1]
        self.mask_with_transform.transform.transform.rotation.z = quaternion[2]
        self.mask_with_transform.transform.transform.rotation.w = quaternion[3]

    def combine_and_transform(self, from_frame, to_frame):
        """
        根据给定的帧ID合成变换。
        from_frame: 源帧ID
        to_frame: 目标帧ID
        """
        try:
            combined_transform = self.tf_buffer.lookup_transform(
                to_frame, from_frame, rospy.Time(0), rospy.Duration(1.0)
            )
            self.mask_with_transform.transform = combined_transform
            return True
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr(f"Error in combining and publishing transforms: {e}")
            return False


if __name__ == "__main__":
    rospy.init_node("transform_manager_node")
    tm = TransformManager("/transform_output")

    # 组合变换
    if not tm.combine_and_transform("front_center_optical/static", "world_enu"):
        rospy.loginfo("Failed to combine MaskWithTransform message.")
