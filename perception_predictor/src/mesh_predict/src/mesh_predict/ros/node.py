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

from collections import deque

import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CameraInfo, PointCloud2 as PointCloud2Msg
from std_msgs.msg import String as StringMsg

from perception_msgs.msg import CompressMaskWithPose

from mesh_predict.core.orchestrator import (
    MeshPredictOrchestrator,
    format_waiting_reason,
)
from mesh_predict.ros.params import OfficialNodeParams
from mesh_predict.ros.publishers import MeshPredictPublishers
from mesh_predict.runtime.resources import OfficialRuntimeSpec


class MeshPredictNode:
    """Official ROS node wrapper around the runtime orchestrator."""

    def __init__(self, params: OfficialNodeParams, runtime_spec: OfficialRuntimeSpec):
        self.params = params
        self.runtime_spec = runtime_spec
        rospy.logwarn(
            "MeshPredict runtime config=%s model config=%s mesh checkpoint=%s",
            runtime_spec.runtime_config_path,
            runtime_spec.model_config_path,
            runtime_spec.mesh_pred_checkpoint,
        )
        self.publishers = MeshPredictPublishers(params)
        self.orchestrator = MeshPredictOrchestrator(
            runtime_spec=runtime_spec,
            min_height=params.min_height,
            mesh_min_z_offset=params.mesh_min_z_offset,
            keep_largest_component=params.keep_largest_mesh_component,
            nksr_scale=params.nksr_scale,
            recall_threshold=params.recall_threshold,
            recall_distance_threshold=params.recall_distance_threshold,
            pose_alignment_mode=params.pose_alignment_mode,
            debug_dump_dir=params.debug_dump_dir,
            debug_dump_once=params.debug_dump_once,
            device="cuda",
        )

        self.pending_point_messages = deque()
        self.pending_img_pose_messages = deque()
        self._pc_prediction_running = False

        self.camera_info_sub = None
        if self.params.camera_info_topic:
            self.camera_info_sub = rospy.Subscriber(
                self.params.camera_info_topic,
                CameraInfo,
                self.camera_info_callback,
            )
        self.pointcloud_sub = rospy.Subscriber(
            self.params.pc_topic, PointCloud2Msg, self.pointcloud_callback
        )
        self.compress_img_sub = rospy.Subscriber(
            self.params.img_pose_topic,
            CompressMaskWithPose,
            self.compress_image_transform_callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.text_prompt_sub = rospy.Subscriber(
            self.params.text_prompt_topic, StringMsg, self.text_prompt_callback
        )

        self.filtered_pc_sub = None
        if self.params.filtered_pc_topic:
            self.filtered_pc_sub = rospy.Subscriber(
                self.params.filtered_pc_topic,
                PointCloud2Msg,
                self.filtered_pc_callback,
                queue_size=1,
            )

        self.img_pose_process_timer = rospy.Timer(
            rospy.Duration(0.01), self.img_pose_process_timer_callback
        )
        self.pointcloud_process_timer = rospy.Timer(
            rospy.Duration(0.01), self.pointcloud_process_timer_callback
        )
        self.mesh_predict_timer = rospy.Timer(
            rospy.Duration(2), self.mesh_predict_timer_callback
        )
        self.pc_predict_timer = rospy.Timer(
            rospy.Duration(0.5), self.pc_predict_timer_callback
        )

        self.load_static_camera_info()
        rospy.logwarn("MeshPredictNode Init Done")

    def load_static_camera_info(self):
        camera_param_file = (
            self.params.camera_param or self.runtime_spec.camera_param_default
        )
        rospy.loginfo(f"Loading camera parameters from: {camera_param_file}")
        try:
            self.orchestrator.load_static_camera_info(camera_param_file)
            rospy.loginfo("Camera parameters loaded successfully.")
        except Exception as exc:
            rospy.logerr(f"Failed to load camera parameters: {exc}")

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.orchestrator.update_camera_info(msg)

    def pointcloud_callback(self, msg: PointCloud2Msg) -> None:
        self.pending_point_messages.append(msg)

    def compress_image_transform_callback(self, msg: CompressMaskWithPose) -> None:
        self.pending_img_pose_messages.append(msg)

    def text_prompt_callback(self, msg: StringMsg) -> None:
        self.orchestrator.set_text_prompt(msg.data)

    def filtered_pc_callback(self, ros_pc: PointCloud2Msg) -> None:
        try:
            gen = point_cloud2.read_points(
                ros_pc, field_names=("x", "y", "z"), skip_nans=True
            )
            pc_np = np.array(list(gen))
            self.orchestrator.append_filtered_pointcloud(pc_np)
            rospy.loginfo(f"Received filtered point cloud with {len(pc_np)} points.")
        except Exception as exc:
            rospy.logerr(f"Error processing filtered point cloud: {exc}")

    def img_pose_process_timer_callback(self, _event) -> None:
        if not self.pending_img_pose_messages:
            return

        img_pose_msg_list = list(self.pending_img_pose_messages)
        self.pending_img_pose_messages.clear()
        for msg in img_pose_msg_list:
            self.orchestrator.append_image_pose_message(msg)

    def pointcloud_process_timer_callback(self, _event) -> None:
        if len(self.pending_point_messages) < 5:
            return

        point_msg_list = list(self.pending_point_messages)[-3:]
        self.pending_point_messages.clear()
        self.orchestrator.append_pointcloud_messages(point_msg_list)

    def pc_predict_timer_callback(self, _event) -> None:
        if self._pc_prediction_running:
            rospy.logdebug("Skipping point cloud prediction while previous run is active")
            return

        self._pc_prediction_running = True
        try:
            result = self.orchestrator.run_pointcloud_prediction()
            if result is None:
                rospy.logdebug(format_waiting_reason(self.orchestrator))
                return

            self.publishers.publish_pointcloud_prediction(result)
            timings = result.get("timings", {})
            if result.get("debug_dump_path"):
                rospy.logwarn(
                    "Pointcloud debug dump saved to %s", result["debug_dump_path"]
                )
            rospy.logwarn(
                "PC predict timing total=%.3fs batch=%.3fs to_device=%.3fs "
                "infer=%.3fs decode=%.3fs post=%.3fs input_pts=%s partial_pts=%s pred_pts=%s",
                timings.get("total", -1.0),
                timings.get("batch", -1.0),
                timings.get("to_device", -1.0),
                timings.get("infer", -1.0),
                timings.get("decode", -1.0),
                timings.get("post", -1.0),
                timings.get("input_points", "?"),
                timings.get("partial_points", "?"),
                timings.get("predicted_points", "?"),
            )
        finally:
            self._pc_prediction_running = False

    def mesh_predict_timer_callback(self, _event) -> None:
        result = self.orchestrator.run_mesh_prediction()
        if result is None:
            rospy.logwarn("Waiting for mesh prediction metadata...")
            return

        if result["used_fallback"]:
            rospy.logwarn(
                "filtered point cloud is unavailable; falling back to predicted surface input"
            )
        if result.get("used_cached_mesh"):
            rospy.logwarn(
                "surface recall %.3f exceeded threshold %.3f; reusing previous mesh",
                result.get("surface_recall", -1.0),
                self.params.recall_threshold,
            )

        self.publishers.publish_mesh_prediction(result)
        timings = result.get("timings", {})
        rospy.logwarn(
            "Mesh predict timing total=%.3fs input=%.3fs surface=%.3fs "
            "recall=%.3fs pred_mesh=%.3fs post=%.3fs filtered_pts=%s "
            "partial_pts=%s surface_pts=%s vertices=%s faces=%s cached=%s",
            timings.get("total", -1.0),
            timings.get("input", -1.0),
            timings.get("surface", -1.0),
            timings.get("recall", -1.0),
            timings.get("pred_mesh", -1.0),
            timings.get("post", -1.0),
            timings.get("filtered_input_points", "?"),
            timings.get("partial_points", "?"),
            timings.get("surface_points", "?"),
            timings.get("mesh_vertices", "?"),
            timings.get("mesh_faces", "?"),
            result.get("used_cached_mesh", False),
        )
        rospy.logwarn("Mesh publish Done")

    def run(self):
        rospy.spin()
