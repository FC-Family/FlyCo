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

from dataclasses import dataclass

import rospy

from .topics import (
    DEFAULT_CLUSTER_GUIDE_POINTCLOUD_TOPIC,
    DEFAULT_CLUSTER_ACCUMULATED_TARGET_POINTCLOUD_TOPIC,
    DEFAULT_CLUSTER_ENVIRONMENT_POINTCLOUD_TOPIC,
    DEFAULT_CLUSTER_MASK_IMAGE_TOPIC,
    DEFAULT_CLUSTER_PROJECTION_IMAGE_TOPIC,
    DEFAULT_CLUSTER_TARGET_POINTCLOUD_TOPIC,
    DEFAULT_COMPRESSED_MASK_TOPIC,
    DEFAULT_INPUT_DEPTH_MASK_TOPIC,
    DEFAULT_INPUT_IMAGE_TOPIC,
    DEFAULT_INPUT_ODOMETRY_TOPIC,
    DEFAULT_INPUT_POINTCLOUD_TOPIC,
    DEFAULT_LEGACY_PROMPT_POINTS_TOPIC,
    DEFAULT_LEGACY_PROMPT_TEXT_TOPIC,
    DEFAULT_MASK_IMAGE_TOPIC,
    DEFAULT_MASK_WITH_POSE_TOPIC,
    DEFAULT_POSE_TOPIC,
    DEFAULT_PROMPT_POINTS_TOPIC,
    DEFAULT_PROMPT_TEXT_TOPIC,
    DEFAULT_TARGET_POINTCLOUD_TOPIC,
)


@dataclass(frozen=True)
class TrackingRuntimeParams:
    min_height: float
    ckpt_path: str
    cluster_resolution: int
    camera_param: str
    camera_info_topic: str
    query_points_topic: str
    query_text_topic: str
    pc_topic: str
    img_topic: str
    odom_topic: str
    gimbal_topic: str
    airsim_gimbal_use_roll: bool
    depth_mask_topic: str
    airsim_camera_pose_enabled: bool
    tf_static_topic: str
    airsim_odom_frame: str
    airsim_camera_body_frame: str
    airsim_camera_optical_frame: str
    depth_guidance_enabled: bool
    depth_guidance_consistency_threshold: float
    depth_guidance_empty_mask_streak: int
    depth_guidance_reinit_cooldown_frames: int
    depth_guidance_min_mask_area: int
    depth_guidance_bbox_padding: int
    depth_guidance_max_reinit_bbox_area_ratio: float
    depth_guidance_max_reinit_bbox_width_ratio: float
    depth_guidance_max_reinit_bbox_height_ratio: float
    depth_guidance_reinit_on_inconsistency: bool
    depth_guidance_reinit_on_empty_streak: bool
    evf_text_init_enabled: bool
    evf_root: str
    evf_model_version: str
    evf_model_type: str
    evf_precision: str
    evf_device: str
    evf_image_size: int
    evf_semantic_level: bool
    evf_min_mask_area: int
    evf_max_init_points: int
    sam2_samurai_mode: bool
    sam2_depth_score_weight: float
    sam2_kf_score_weight: float
    sam2_memory_bank_geo_score_threshold: float
    cluster_target_pc_topic: str
    cluster_accumulated_target_pc_topic: str
    mask_image_topic: str
    mask_with_pose_topic: str
    compressed_mask_topic: str
    mask_pointcloud_topic: str
    cluster_pointcloud_topic: str
    filtered_pointcloud_topic: str
    pose_topic: str
    cluster_projection_topic: str
    cluster_mask_topic: str
    project_pc_topic: str
    debug_image_topic: str
    debug_pointcloud_topic: str
    cluster_projection_use_accumulated: bool
    cluster_projection_use_hpr: bool
    cluster_incremental_mode: bool
    cluster_input_max_points: int
    cluster_voxel_size: float


@dataclass(frozen=True)
class PromptAdapterParams:
    input_query_points_topic: str
    input_query_text_topic: str
    query_points_topic: str
    query_text_topic: str


def load_tracking_runtime_params() -> TrackingRuntimeParams:
    return TrackingRuntimeParams(
        min_height=rospy.get_param("~min_height", default=0.1),
        ckpt_path=rospy.get_param("~ckpt_path", default=""),
        cluster_resolution=rospy.get_param("~cluster_resolution", default=200),
        camera_param=rospy.get_param("~camera_param", default="config/camera.json"),
        camera_info_topic=rospy.get_param("~camera_info_topic", default=""),
        query_points_topic=rospy.get_param(
            "~query_points_topic", default=DEFAULT_PROMPT_POINTS_TOPIC
        ),
        query_text_topic=rospy.get_param(
            "~query_text_topic", default=DEFAULT_PROMPT_TEXT_TOPIC
        ),
        pc_topic=rospy.get_param("~pc_topic", default=DEFAULT_INPUT_POINTCLOUD_TOPIC),
        img_topic=rospy.get_param("~img_topic", default=DEFAULT_INPUT_IMAGE_TOPIC),
        odom_topic=rospy.get_param("~odom_topic", default=DEFAULT_INPUT_ODOMETRY_TOPIC),
        gimbal_topic=rospy.get_param("~gimbal_topic", default=""),
        airsim_gimbal_use_roll=rospy.get_param(
            "~airsim_gimbal_use_roll", default=False
        ),
        depth_mask_topic=rospy.get_param(
            "~depth_mask_topic", default=DEFAULT_INPUT_DEPTH_MASK_TOPIC
        ),
        airsim_camera_pose_enabled=rospy.get_param(
            "~airsim_camera_pose_enabled", default=False
        ),
        tf_static_topic=rospy.get_param("~tf_static_topic", default="/tf_static"),
        airsim_odom_frame=rospy.get_param(
            "~airsim_odom_frame", default="drone_1/odom_local_enu"
        ),
        airsim_camera_body_frame=rospy.get_param(
            "~airsim_camera_body_frame", default="front_center_body/static"
        ),
        airsim_camera_optical_frame=rospy.get_param(
            "~airsim_camera_optical_frame", default="front_center_optical/static"
        ),
        depth_guidance_enabled=rospy.get_param(
            "~depth_guidance_enabled", default=False
        ),
        depth_guidance_consistency_threshold=rospy.get_param(
            "~depth_guidance_consistency_threshold", default=0.2
        ),
        depth_guidance_empty_mask_streak=rospy.get_param(
            "~depth_guidance_empty_mask_streak", default=3
        ),
        depth_guidance_reinit_cooldown_frames=rospy.get_param(
            "~depth_guidance_reinit_cooldown_frames", default=8
        ),
        depth_guidance_min_mask_area=rospy.get_param(
            "~depth_guidance_min_mask_area", default=256
        ),
        depth_guidance_bbox_padding=rospy.get_param(
            "~depth_guidance_bbox_padding", default=8
        ),
        depth_guidance_max_reinit_bbox_area_ratio=rospy.get_param(
            "~depth_guidance_max_reinit_bbox_area_ratio", default=1.0
        ),
        depth_guidance_max_reinit_bbox_width_ratio=rospy.get_param(
            "~depth_guidance_max_reinit_bbox_width_ratio", default=1.0
        ),
        depth_guidance_max_reinit_bbox_height_ratio=rospy.get_param(
            "~depth_guidance_max_reinit_bbox_height_ratio", default=1.0
        ),
        depth_guidance_reinit_on_inconsistency=rospy.get_param(
            "~depth_guidance_reinit_on_inconsistency", default=True
        ),
        depth_guidance_reinit_on_empty_streak=rospy.get_param(
            "~depth_guidance_reinit_on_empty_streak", default=True
        ),
        evf_text_init_enabled=rospy.get_param(
            "~evf_text_init_enabled", default=False
        ),
        evf_root=rospy.get_param("~evf_root", default=""),
        evf_model_version=rospy.get_param(
            "~evf_model_version", default="YxZhang/evf-sam2"
        ),
        evf_model_type=rospy.get_param("~evf_model_type", default="sam2"),
        evf_precision=rospy.get_param("~evf_precision", default="fp16"),
        evf_device=rospy.get_param("~evf_device", default="cuda"),
        evf_image_size=rospy.get_param("~evf_image_size", default=224),
        evf_semantic_level=rospy.get_param("~evf_semantic_level", default=False),
        evf_min_mask_area=rospy.get_param("~evf_min_mask_area", default=256),
        evf_max_init_points=rospy.get_param("~evf_max_init_points", default=8),
        sam2_samurai_mode=rospy.get_param("~sam2_samurai_mode", default=True),
        sam2_depth_score_weight=rospy.get_param(
            "~sam2_depth_score_weight", default=0.7
        ),
        sam2_kf_score_weight=rospy.get_param(
            "~sam2_kf_score_weight", default=0.15
        ),
        sam2_memory_bank_geo_score_threshold=rospy.get_param(
            "~sam2_memory_bank_geo_score_threshold", default=0.0
        ),
        cluster_target_pc_topic=rospy.get_param(
            "~cluster_target_pc_topic",
            default=DEFAULT_CLUSTER_TARGET_POINTCLOUD_TOPIC,
        ),
        cluster_accumulated_target_pc_topic=rospy.get_param(
            "~cluster_accumulated_target_pc_topic",
            default=DEFAULT_CLUSTER_ACCUMULATED_TARGET_POINTCLOUD_TOPIC,
        ),
        mask_image_topic=rospy.get_param(
            "~mask_image_topic", default=DEFAULT_MASK_IMAGE_TOPIC
        ),
        mask_with_pose_topic=rospy.get_param(
            "~mask_with_pose_topic", default=DEFAULT_MASK_WITH_POSE_TOPIC
        ),
        compressed_mask_topic=rospy.get_param(
            "~compressed_mask_topic", default=DEFAULT_COMPRESSED_MASK_TOPIC
        ),
        mask_pointcloud_topic=rospy.get_param(
            "~mask_pointcloud_topic", default=DEFAULT_TARGET_POINTCLOUD_TOPIC
        ),
        cluster_pointcloud_topic=rospy.get_param(
            "~cluster_pointcloud_topic", default=DEFAULT_CLUSTER_GUIDE_POINTCLOUD_TOPIC
        ),
        filtered_pointcloud_topic=rospy.get_param(
            "~filtered_pointcloud_topic",
            default=DEFAULT_CLUSTER_ENVIRONMENT_POINTCLOUD_TOPIC,
        ),
        pose_topic=rospy.get_param("~pose_topic", default=DEFAULT_POSE_TOPIC),
        cluster_projection_topic=rospy.get_param(
            "~cluster_projection_topic", default=DEFAULT_CLUSTER_PROJECTION_IMAGE_TOPIC
        ),
        cluster_mask_topic=rospy.get_param(
            "~cluster_mask_topic", default=DEFAULT_CLUSTER_MASK_IMAGE_TOPIC
        ),
        project_pc_topic=rospy.get_param("~project_pc_topic", default="~project_pc"),
        debug_image_topic=rospy.get_param(
            "~debug_image_topic", default="~img_debug_topic"
        ),
        debug_pointcloud_topic=rospy.get_param(
            "~debug_pointcloud_topic", default="~debug_pointclouds"
        ),
        cluster_projection_use_accumulated=rospy.get_param(
            "~cluster_projection_use_accumulated", default=True
        ),
        cluster_projection_use_hpr=rospy.get_param(
            "~cluster_projection_use_hpr", default=True
        ),
        cluster_incremental_mode=rospy.get_param(
            "~cluster_incremental_mode", default=False
        ),
        cluster_input_max_points=rospy.get_param(
            "~cluster_input_max_points", default=1024
        ),
        cluster_voxel_size=rospy.get_param("~cluster_voxel_size", default=0.2),
    )


def load_prompt_adapter_params() -> PromptAdapterParams:
    return PromptAdapterParams(
        input_query_points_topic=rospy.get_param(
            "~input_query_points_topic", default=DEFAULT_LEGACY_PROMPT_POINTS_TOPIC
        ),
        input_query_text_topic=rospy.get_param(
            "~input_query_text_topic", default=DEFAULT_LEGACY_PROMPT_TEXT_TOPIC
        ),
        query_points_topic=rospy.get_param(
            "~query_points_topic", default=DEFAULT_PROMPT_POINTS_TOPIC
        ),
        query_text_topic=rospy.get_param(
            "~query_text_topic", default=DEFAULT_PROMPT_TEXT_TOPIC
        ),
    )
