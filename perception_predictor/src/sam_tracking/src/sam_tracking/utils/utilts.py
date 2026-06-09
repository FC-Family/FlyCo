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
import numpy as np
from cv_bridge import CvBridge
import tf.transformations as tft


def compress_image(image_msg):
    """
    压缩 sensor_msgs/Image 为 JPEG 格式。

    Args:
        image_msg (sensor_msgs/Image): ROS 图像消息。

    Returns:
        compressed_data (bytes): 压缩后的图像数据。
    """
    # 初始化 CvBridge 对象

    # 使用 OpenCV 压缩图像为 JPEG 格式
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # 压缩质量：90
    success, encoded_image = cv2.imencode(".jpg", image_msg, encode_param)

    if success:
        return encoded_image.tobytes()
    else:
        return None


def transform_to_matrix(transform_stamped):
    """
    将 geometry_msgs/TransformStamped 转换为 4×4 的变换矩阵。

    Args:
        transform_stamped (geometry_msgs/TransformStamped): ROS 变换消息。

    Returns:
        matrix (numpy.ndarray): 4×4 的变换矩阵。
    """
    # 提取平移和旋转信息
    translation = [
        transform_stamped.transform.translation.x,
        transform_stamped.transform.translation.y,
        transform_stamped.transform.translation.z,
    ]
    rotation = [
        transform_stamped.transform.rotation.x,
        transform_stamped.transform.rotation.y,
        transform_stamped.transform.rotation.z,
        transform_stamped.transform.rotation.w,
    ]

    # 构造 4×4 的变换矩阵
    matrix = tft.translation_matrix(translation) @ tft.quaternion_matrix(rotation)
    return matrix


def compress_transform(transform_stamped):
    return [
        transform_stamped.transform.translation.x,
        transform_stamped.transform.translation.y,
        transform_stamped.transform.translation.z,
        transform_stamped.transform.rotation.x,
        transform_stamped.transform.rotation.y,
        transform_stamped.transform.rotation.z,
        transform_stamped.transform.rotation.w,
    ]
