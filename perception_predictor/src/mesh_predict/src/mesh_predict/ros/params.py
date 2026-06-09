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
from typing import Any, Dict, Optional

import rospy

from mesh_predict.runtime.resources import (
    discover_package_root,
    load_optional_yaml,
    resolve_existing_file,
)


@dataclass
class OfficialNodeParams:
    """Resolved ROS parameters for the official runtime node."""

    runtime_config_path: str
    mesh_pred_ckpt_path: Optional[str]
    filtered_pc_topic: str
    camera_param: Optional[str]
    camera_info_topic: str
    pc_topic: str
    img_pose_topic: str
    text_prompt_topic: str
    predicted_pointcloud_topic: str
    partial_pointcloud_topic: str
    predicted_mesh_topic: str
    mesh_vis_topic: str
    predicted_input_imgs_topic: str
    debug_dump_dir: str
    debug_dump_once: bool
    min_height: float
    mesh_min_z_offset: float
    keep_largest_mesh_component: bool
    nksr_scale: float
    recall_threshold: float
    recall_distance_threshold: float
    pose_alignment_mode: str


def _profile_value(profile: Dict[str, Any], key: str, default):
    return profile.get(key, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def load_node_params() -> OfficialNodeParams:
    """Load ROS parameters and apply optional profile defaults."""

    profile_config_path = rospy.get_param("~profile_config_path", default="")
    profile = load_optional_yaml(profile_config_path)
    profile_package_root = (
        discover_package_root(profile_config_path) if profile_config_path else None
    )
    camera_param = rospy.get_param(
        "~camera_param", default=_profile_value(profile, "camera_param", None)
    )
    if camera_param:
        camera_param = resolve_existing_file(
            "camera_param", camera_param, package_root=profile_package_root
        )

    return OfficialNodeParams(
        runtime_config_path=rospy.get_param("~runtime_config_path", default=None),
        mesh_pred_ckpt_path=rospy.get_param("~mesh_pred_ckpt_path", default=None),
        filtered_pc_topic=rospy.get_param("~filtered_pc_topic", default=""),
        camera_param=camera_param,
        camera_info_topic=rospy.get_param(
            "~camera_info_topic",
            default=_profile_value(profile, "camera_info_topic", ""),
        ),
        pc_topic=rospy.get_param(
            "~pc_topic",
            default=_profile_value(profile, "pc_topic", "/tracking/target_pc"),
        ),
        img_pose_topic=rospy.get_param(
            "~img_pose_topic",
            default=_profile_value(
                profile, "img_pose_topic", "/sam_tracking/compress_mask_and_transform"
            ),
        ),
        text_prompt_topic=rospy.get_param(
            "~text_prompt_topic",
            default=_profile_value(profile, "text_prompt_topic", "/text_prompt"),
        ),
        predicted_pointcloud_topic=rospy.get_param(
            "~predicted_pointcloud_topic",
            default=_profile_value(
                profile, "predicted_pointcloud_topic", "mesh_pred/pred_pc"
            ),
        ),
        partial_pointcloud_topic=rospy.get_param(
            "~partial_pointcloud_topic",
            default=_profile_value(
                profile, "partial_pointcloud_topic", "mesh_pred/partial_pc"
            ),
        ),
        predicted_mesh_topic=rospy.get_param(
            "~predicted_mesh_topic",
            default=_profile_value(
                profile, "predicted_mesh_topic", "/prediction/predicted_mesh"
            ),
        ),
        mesh_vis_topic=rospy.get_param(
            "~mesh_vis_topic",
            default=_profile_value(profile, "mesh_vis_topic", "/prediction/mesh_vis"),
        ),
        predicted_input_imgs_topic=rospy.get_param(
            "~predicted_input_imgs_topic",
            default=_profile_value(
                profile, "predicted_input_imgs_topic", "/prediction/input_image"
            ),
        ),
        debug_dump_dir=str(
            rospy.get_param(
                "~debug_dump_dir",
                default=_profile_value(profile, "debug_dump_dir", ""),
            )
        ),
        debug_dump_once=_as_bool(
            rospy.get_param(
                "~debug_dump_once",
                default=_profile_value(profile, "debug_dump_once", False),
            )
        ),
        min_height=float(
            rospy.get_param(
                "~min_height", default=_profile_value(profile, "min_height", 0.1)
            )
        ),
        mesh_min_z_offset=float(
            rospy.get_param(
                "~mesh_min_z_offset",
                default=_profile_value(profile, "mesh_min_z_offset", 0.3),
            )
        ),
        keep_largest_mesh_component=_as_bool(
            rospy.get_param(
                "~keep_largest_mesh_component",
                default=_profile_value(profile, "keep_largest_mesh_component", True),
            )
        ),
        nksr_scale=float(
            rospy.get_param(
                "~nksr_scale", default=_profile_value(profile, "nksr_scale", 0.3)
            )
        ),
        recall_threshold=float(
            rospy.get_param(
                "~recall_threshold",
                default=_profile_value(profile, "recall_threshold", 0.99),
            )
        ),
        recall_distance_threshold=float(
            rospy.get_param(
                "~recall_distance_threshold",
                default=_profile_value(profile, "recall_distance_threshold", 0.1),
            )
        ),
        pose_alignment_mode=str(
            rospy.get_param(
                "~pose_alignment_mode",
                default=_profile_value(profile, "pose_alignment_mode", "legacy_ros"),
            )
        ),
    )
