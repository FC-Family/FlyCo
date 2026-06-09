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

import json
import os

from sensor_msgs.msg import CameraInfo


def build_camera_info(camera_config):
    """Build a ROS CameraInfo message from a packaged JSON camera config."""

    if not camera_config or "camera" not in camera_config:
        raise ValueError("Invalid camera configuration")

    cam_info = CameraInfo()
    camera_data = camera_config["camera"]

    cam_info.distortion_model = camera_data.get("camera_model", "plumb_bob")

    intrinsics = camera_data["intrinsics"]
    cam_info.K = [0] * 9
    cam_info.K[0] = intrinsics[0]
    cam_info.K[4] = intrinsics[1]
    cam_info.K[2] = intrinsics[2]
    cam_info.K[5] = intrinsics[3]
    cam_info.K[8] = 1.0

    cam_info.D = camera_data.get("distortion_coeffs", [])

    cam_info.P = [0] * 12
    cam_info.P[0] = intrinsics[0]
    cam_info.P[5] = intrinsics[1]
    cam_info.P[2] = intrinsics[2]
    cam_info.P[6] = intrinsics[3]
    cam_info.P[10] = 1.0

    cam_info.width = camera_config.get("width", 1920)
    cam_info.height = camera_config.get("height", 1080)
    return cam_info


def load_camera_info(camera_param_file):
    """Load camera parameters from disk and convert them into CameraInfo."""

    if not os.path.exists(camera_param_file):
        raise FileNotFoundError(camera_param_file)

    with open(camera_param_file, "r") as file:
        camera_config = json.load(file)

    return build_camera_info(camera_config)
