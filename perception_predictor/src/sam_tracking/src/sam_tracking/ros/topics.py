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

DEFAULT_LEGACY_PROMPT_POINTS_TOPIC = "/query_points"
DEFAULT_LEGACY_PROMPT_TEXT_TOPIC = "/query_text"

DEFAULT_INPUT_IMAGE_TOPIC = "/tracking/input/image"
DEFAULT_INPUT_POINTCLOUD_TOPIC = "/tracking/input/pointcloud"
DEFAULT_INPUT_ODOMETRY_TOPIC = "/tracking/input/odometry"
DEFAULT_INPUT_DEPTH_MASK_TOPIC = "/tracking/input/depth_mask"

DEFAULT_PROMPT_POINTS_TOPIC = "/tracking/prompt/points"
DEFAULT_PROMPT_TEXT_TOPIC = "/tracking/prompt/text"

DEFAULT_MASK_IMAGE_TOPIC = "/tracking/mask/image"
DEFAULT_MASK_WITH_POSE_TOPIC = "/tracking/mask/with_pose"
DEFAULT_COMPRESSED_MASK_TOPIC = "/sam_tracking/compress_mask_and_transform"
DEFAULT_TARGET_POINTCLOUD_TOPIC = "/tracking/target_pointcloud"
DEFAULT_CLUSTER_GUIDE_POINTCLOUD_TOPIC = "/tracking/guider_pc"
DEFAULT_CLUSTER_TARGET_POINTCLOUD_TOPIC = "/tracking/target_pc"
DEFAULT_CLUSTER_ACCUMULATED_TARGET_POINTCLOUD_TOPIC = (
    "/cluster/target_pc_accumulated"
)
DEFAULT_CLUSTER_ENVIRONMENT_POINTCLOUD_TOPIC = "/tracking/env_pc"
DEFAULT_CLUSTER_PROJECTION_IMAGE_TOPIC = "/tracking/cluster/projection_image"
DEFAULT_CLUSTER_MASK_IMAGE_TOPIC = "/tracking/cluster/mask_image"
DEFAULT_POSE_TOPIC = "/tracking/pose"
