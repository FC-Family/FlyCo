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

from perception_msgs.msg import CompressMaskWithPose
from sensor_msgs.msg import CompressedImage

from sam_tracking.utils.utilts import compress_image, compress_transform


def build_compress_mask_with_pose(mask_rgba, transform, target_size=(640, 480)):
    mask = cv2.resize(mask_rgba, target_size)
    compress_mask_bytes = compress_image(mask)
    if compress_mask_bytes is None:
        return None

    compress_msg = CompressedImage()
    compress_msg.format = "jpeg"
    compress_msg.data = compress_mask_bytes

    compress_pose = compress_transform(transform)
    compress_mask_with_pose = CompressMaskWithPose()
    compress_mask_with_pose.mask = compress_msg
    compress_mask_with_pose.pose = compress_pose
    compress_mask_with_pose.header.stamp = transform.header.stamp
    compress_mask_with_pose.header.frame_id = "world"
    return compress_mask_with_pose
