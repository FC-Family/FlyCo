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
import sys

import cv2
import message_filters
import numpy as np
import rospy
import std_msgs.msg
import tf.transformations as tft
import tf_conversions
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import QuaternionStamped
from image_geometry import PinholeCameraModel
from message_filters import ApproximateTimeSynchronizer
from nav_msgs.msg import Odometry
from perception_msgs.msg import CompressMaskWithPose, MaskWithTransform
from sam_tracking.cluster import ClusterPostprocess, ClusterRuntime
from sam_tracking.cluster.cluster_postprocess import IncrementalClusterConfig
from sam_tracking.core import DepthGuidancePolicy, PromptSession, TrackingPipeline
from sam_tracking.core.pointcloud_projector import PointCloudProjector
from sam_tracking.providers.sam2_camera_provider import (
    EVFTextInitConfig,
    SAM2RuntimeConfig,
)
from sam_tracking.ros.message_builders import build_compress_mask_with_pose
from sam_tracking.ros.params import load_tracking_runtime_params
from sam_tracking.providers import Sam2CameraProvider
from sam_tracking.utils.mask_with_transform_manage import TransformManager
from sam_tracking.ros.topics import DEFAULT_PROMPT_POINTS_TOPIC
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Int32MultiArray as Int32MultiArrayMsg
from std_msgs.msg import String as StringMsg
from tf2_msgs.msg import TFMessage

try:
    import tf
    import tf2_geometry_msgs
    import tf2_ros
    import tf2_sensor_msgs.tf2_sensor_msgs
except Exception:
    sys.path.append("/usr/lib/python3/dist-packages/")
    import tf
    import tf2_geometry_msgs
    import tf2_ros
    import tf2_sensor_msgs.tf2_sensor_msgs

    sys.path.remove("/usr/lib/python3/dist-packages/")


class SAMTrackingManager:
    def __init__(self, node_name) -> None:
        rospy.init_node(name=node_name)
        self.runtime_params = load_tracking_runtime_params()
        self.min_height = self.runtime_params.min_height
        self.ckpt_path = self.runtime_params.ckpt_path
        self.cluster_resolution = self.runtime_params.cluster_resolution

        self.provider = Sam2CameraProvider(
            evf_text_init_config=EVFTextInitConfig(
                enabled=self.runtime_params.evf_text_init_enabled,
                evf_root=self.runtime_params.evf_root,
                model_version=self.runtime_params.evf_model_version,
                model_type=self.runtime_params.evf_model_type,
                precision=self.runtime_params.evf_precision,
                device=self.runtime_params.evf_device,
                image_size=self.runtime_params.evf_image_size,
                semantic_level=self.runtime_params.evf_semantic_level,
                min_mask_area=self.runtime_params.evf_min_mask_area,
                max_init_points=self.runtime_params.evf_max_init_points,
            ),
            sam2_runtime_config=SAM2RuntimeConfig(
                samurai_mode=self.runtime_params.sam2_samurai_mode,
                depth_score_weight=self.runtime_params.sam2_depth_score_weight,
                kf_score_weight=self.runtime_params.sam2_kf_score_weight,
                memory_bank_geo_score_threshold=self.runtime_params.sam2_memory_bank_geo_score_threshold,
            ),
        )
        self._bridge = CvBridge()

        self.camera_model = PinholeCameraModel()
        self.camera_info = None
        self.frame_idx = 0
        self.airsim_camera_pose_enabled = bool(
            self.runtime_params.airsim_camera_pose_enabled
        )
        self.airsim_gimbal_use_roll = bool(
            self.runtime_params.airsim_gimbal_use_roll
        )

        self.pointcloud_msgs = []
        self.query_points = []
        self.depth_mask_msg = None
        self.gimbal_msg = None
        self.query_text = ""
        self.cluster_depth_mask_max_stamp_delta = rospy.Duration(
            rospy.get_param("~cluster_depth_mask_max_stamp_delta", default=1.5)
        )
        self.gimbal_max_stamp_delta = rospy.Duration(
            rospy.get_param("~gimbal_max_stamp_delta", default=0.12)
        )
        # AirSim bag statistics in this repo are typically within 50 ms for
        # image/odom/lidar and within 6 ms for image/gimbal. Keep runtime
        # tolerances tight enough to avoid cross-frame projection drift.
        self.image_odom_sync_slop = 0.08
        self.pointcloud_max_stamp_delta = rospy.Duration(0.08)
        self.pointcloud_max_count = 4
        self.reinit_debug_dir = rospy.get_param("~reinit_debug_dir", default="")
        self._reinit_debug_count = 0
        self._guidance_debug_count = 0
        self._pending_reinit_next_debug = None
        if self.reinit_debug_dir:
            os.makedirs(self.reinit_debug_dir, exist_ok=True)

        self.static_transforms = {
            "body2cam": None,
            "cam2body": None,
        }
        self.airsim_camera_body_translation = np.array([0.125, 0.0, 0.0])
        # Match planner_simulator/airsim_ros_pkgs img2cam_R_matrix exactly.
        # Do not override this from /tf_static; that path is what previously
        # caused the AirSim projection chain to diverge.
        self.airsim_camera_optical_rotation = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        self._airsim_body_static_ready = False

        self.load_static_info()

        self.img_odom_data = []
        self.mask_T = TransformManager()
        self.prompt_session = PromptSession()
        self.depth_guidance = DepthGuidancePolicy(
            enabled=self.runtime_params.depth_guidance_enabled,
            consistency_threshold=self.runtime_params.depth_guidance_consistency_threshold,
            empty_mask_streak_threshold=self.runtime_params.depth_guidance_empty_mask_streak,
            reinit_cooldown_frames=self.runtime_params.depth_guidance_reinit_cooldown_frames,
            min_mask_area=self.runtime_params.depth_guidance_min_mask_area,
            bbox_padding=self.runtime_params.depth_guidance_bbox_padding,
            max_reinit_bbox_area_ratio=self.runtime_params.depth_guidance_max_reinit_bbox_area_ratio,
            max_reinit_bbox_width_ratio=self.runtime_params.depth_guidance_max_reinit_bbox_width_ratio,
            max_reinit_bbox_height_ratio=self.runtime_params.depth_guidance_max_reinit_bbox_height_ratio,
            reinit_on_inconsistency=self.runtime_params.depth_guidance_reinit_on_inconsistency,
            reinit_on_empty_streak=self.runtime_params.depth_guidance_reinit_on_empty_streak,
        )
        self.static_rotaion = np.array(
            [
                [0.98312, -0.00382055, 0.18292],
                [0, 0.999782, 0.020882],
                [-0.182959, -0.0205295, 0.982906],
            ]
        )
        self.pointcloud_projector = PointCloudProjector(
            camera_model=self.camera_model,
            static_rotation=self.static_rotaion,
            transform_pc_fn=self.transform_pc,
            filter_points_fn=lambda msg: self.filter_points_below_height(
                msg, self.min_height
            ),
        )
        self.cluster_postprocess = ClusterPostprocess(
            camera_model=self.camera_model,
            transform_pc_fn=self.transform_pc,
            static_rotation=self.static_rotaion,
            static_transform=self.static_transforms["body2cam"],
            image_size=(self.camera_info.width, self.camera_info.height),
            use_hpr=self.runtime_params.cluster_projection_use_hpr,
        )
        self.pipeline = TrackingPipeline(
            provider=self.provider,
            prompt_session=self.prompt_session,
            pointcloud_projector=self.pointcloud_projector,
            bridge=self._bridge,
            depth_guidance=self.depth_guidance,
        )
        self._point_stats_last_log_time = {}

        self.cloud_sub = rospy.Subscriber(
            self.runtime_params.pc_topic, PointCloud2, self.cloud_callback
        )
        self.img_sub = message_filters.Subscriber(self.runtime_params.img_topic, Image)
        self.odom_sub = message_filters.Subscriber(
            self.runtime_params.odom_topic, Odometry
        )
        self.camera_info_sub = None
        if self.runtime_params.camera_info_topic:
            self.camera_info_sub = rospy.Subscriber(
                self.runtime_params.camera_info_topic,
                CameraInfo,
                self.camera_info_callback,
            )
        self.tf_static_sub = None
        if self.airsim_camera_pose_enabled:
            self.tf_static_sub = rospy.Subscriber(
                self.runtime_params.tf_static_topic,
                TFMessage,
                self.tf_static_callback,
            )
        self.ats_image_odom = ApproximateTimeSynchronizer(
            [self.img_sub, self.odom_sub], 20, self.image_odom_sync_slop
        )
        self.ats_image_odom.registerCallback(self.img_odom_callback)
        self.depth_mask_sub = None
        if self.runtime_params.depth_mask_topic:
            self.depth_mask_sub = rospy.Subscriber(
                self.runtime_params.depth_mask_topic,
                Image,
                self.depth_mask_callback,
            )
        self.gimbal_sub = None
        if self.runtime_params.gimbal_topic:
            self.gimbal_sub = rospy.Subscriber(
                self.runtime_params.gimbal_topic,
                QuaternionStamped,
                self.gimbal_callback,
            )

        self.query_point_sub = rospy.Subscriber(
            self.runtime_params.query_points_topic or DEFAULT_PROMPT_POINTS_TOPIC,
            Int32MultiArrayMsg,
            self.query_point_callback,
        )
        self.query_text_sub = rospy.Subscriber(
            self.runtime_params.query_text_topic,
            StringMsg,
            self.query_text_callback,
        )

        self.img_pub = rospy.Publisher(
            self.runtime_params.mask_image_topic, Image, queue_size=1
        )
        self.img_debug_pub = rospy.Publisher(
            self.runtime_params.debug_image_topic, Image, queue_size=1
        )
        self.cluster_projection_pub = rospy.Publisher(
            self.runtime_params.cluster_projection_topic, Image, queue_size=1
        )
        self.cluster_mask_pub = rospy.Publisher(
            self.runtime_params.cluster_mask_topic, Image, queue_size=1
        )
        self.project_pc_pub = rospy.Publisher(
            self.runtime_params.project_pc_topic, Image, queue_size=1
        )

        self.transform_publisher = rospy.Publisher(
            self.runtime_params.mask_with_pose_topic, MaskWithTransform, queue_size=10
        )
        self.compress_mask_publisher = rospy.Publisher(
            self.runtime_params.compressed_mask_topic,
            CompressMaskWithPose,
            queue_size=10,
        )
        self.pc_pub = rospy.Publisher(
            self.runtime_params.mask_pointcloud_topic, PointCloud2, queue_size=1
        )
        self.dubug_pc_pub = rospy.Publisher(
            self.runtime_params.debug_pointcloud_topic, PointCloud2, queue_size=1
        )
        self.cluster_guide_pub = rospy.Publisher(
            self.runtime_params.cluster_pointcloud_topic, PointCloud2, queue_size=10
        )
        self.odom_pub = rospy.Publisher(
            self.runtime_params.pose_topic, Odometry, queue_size=1
        )
        self.cluster_runtime = ClusterRuntime(
            bridge=self._bridge,
            cluster_postprocess=self.cluster_postprocess,
            pointcloud_projector=self.pointcloud_projector,
            projection_pub=self.cluster_projection_pub,
            mask_pub=self.cluster_mask_pub,
            target_pc_topic=self.runtime_params.cluster_target_pc_topic,
            accumulated_target_pc_topic=self.runtime_params.cluster_accumulated_target_pc_topic,
            incremental_config=IncrementalClusterConfig(
                enabled=self.runtime_params.cluster_incremental_mode,
                input_max_points=self.runtime_params.cluster_input_max_points,
                voxel_size=self.runtime_params.cluster_voxel_size,
            ),
            projection_use_accumulated=self.runtime_params.cluster_projection_use_accumulated,
        )
        self.cluster_pc_sub = None
        if self.runtime_params.cluster_target_pc_topic:
            self.cluster_pc_sub = rospy.Subscriber(
                self.runtime_params.cluster_target_pc_topic,
                PointCloud2,
                self.cluster_pc_callback,
            )
        self.cluster_accumulated_pc_sub = None
        if self.runtime_params.cluster_accumulated_target_pc_topic:
            self.cluster_accumulated_pc_sub = rospy.Subscriber(
                self.runtime_params.cluster_accumulated_target_pc_topic,
                PointCloud2,
                self.cluster_accumulated_pc_callback,
            )

        self.tracking_timer = rospy.Timer(
            rospy.Duration(1 / 2.0), self.tracking_timer_callback
        )
        rospy.loginfo("SAMTrackingManager has been initialized")

    def create_camera_info(self, camera_config):
        if not camera_config or "camera" not in camera_config:
            rospy.logerr("Invalid camera configuration!")
            return None

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

        rospy.loginfo("CameraInfo message created successfully.")
        return cam_info

    def update_static_transforms_from_json(self, transform, mode="cam2body"):
        if len(transform) != 7:
            raise ValueError(
                "Transform must be a list with 7 elements: [x, y, z, qx, qy, qz, qw]"
            )

        from geometry_msgs.msg import Quaternion, TransformStamped, Vector3

        translation_msg = Vector3()
        translation_msg.x, translation_msg.y, translation_msg.z = transform[:3]

        rotation = Quaternion()
        rotation.x, rotation.y, rotation.z, rotation.w = transform[3:]

        transform_stamped = TransformStamped()
        transform_stamped.header.stamp = rospy.Time.now()
        transform_stamped.header.frame_id = "base_frame"
        transform_stamped.child_frame_id = (
            "camera_frame" if mode == "cam2body" else "body_frame"
        )
        transform_stamped.transform.translation = translation_msg
        transform_stamped.transform.rotation = rotation

        if mode == "cam2body":
            self.static_transforms["cam2body"] = transform_stamped
            if self.static_transforms["body2cam"] is None:
                self.static_transforms["body2cam"] = self.invert_transform(
                    transform_stamped
                )
        elif mode == "body2cam":
            self.static_transforms["body2cam"] = transform_stamped
            if self.static_transforms["cam2body"] is None:
                self.static_transforms["cam2body"] = self.invert_transform(
                    transform_stamped
                )
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'cam2body' or 'body2cam'.")

        rospy.loginfo(f"Updated static transform for {mode}: {transform}")

    def invert_transform(self, transform_stamped):
        from geometry_msgs.msg import Quaternion, TransformStamped

        translation = transform_stamped.transform.translation
        rotation = transform_stamped.transform.rotation

        inverse_rotation = Quaternion()
        inverse_rotation.x = -rotation.x
        inverse_rotation.y = -rotation.y
        inverse_rotation.z = -rotation.z
        inverse_rotation.w = rotation.w

        quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]
        rotation_matrix = tf_conversions.transformations.quaternion_matrix(quaternion)[
            :3, :3
        ]
        inverse_translation = -rotation_matrix.dot(
            [translation.x, translation.y, translation.z]
        )

        inverted_transform = TransformStamped()
        inverted_transform.header.stamp = transform_stamped.header.stamp
        inverted_transform.header.frame_id = transform_stamped.child_frame_id
        inverted_transform.child_frame_id = transform_stamped.header.frame_id

        inverted_transform.transform.translation.x = inverse_translation[0]
        inverted_transform.transform.translation.y = inverse_translation[1]
        inverted_transform.transform.translation.z = inverse_translation[2]
        inverted_transform.transform.rotation = inverse_rotation

        return inverted_transform

    def load_static_info(self):
        camera_param_file = self.runtime_params.camera_param
        rospy.loginfo(f"Loading camera parameters from: {camera_param_file}")

        if not os.path.exists(camera_param_file):
            rospy.logerr(f"Camera parameter file not found: {camera_param_file}")
            return

        try:
            with open(camera_param_file, "r") as file:
                camera_config = json.load(file)
                rospy.loginfo("Camera parameters loaded successfully.")
                self.update_static_transforms_from_json(
                    camera_config["results"]["init_T_lidar_camera"], "cam2body"
                )

                self.camera_info = self.create_camera_info(camera_config)
                self.camera_model.fromCameraInfo(self.camera_info)
        except Exception as e:
            rospy.logerr(f"Failed to load camera parameters: {e}")
            return

    def transform_pc(self, pc_np, transform):
        projected_points = point_cloud2.create_cloud(
            std_msgs.msg.Header(),
            [
                PointField("x", 0, PointField.FLOAT32, 1),
                PointField("y", 4, PointField.FLOAT32, 1),
                PointField("z", 8, PointField.FLOAT32, 1),
            ],
            pc_np,
        )
        transformed_pc_msg = tf2_sensor_msgs.do_transform_cloud(
            projected_points, transform
        )
        points = point_cloud2.read_points(
            transformed_pc_msg, field_names=("x", "y", "z"), skip_nans=True
        )
        return points

    def img_odom_callback(self, img_msg, odom_msg):
        self.img_odom_data.append({"img": img_msg, "odom": odom_msg})

    def cloud_callback(self, cloud_msg):
        if len(cloud_msg.data) == 0:
            return
        self.pointcloud_msgs.append(cloud_msg)

    def camera_info_callback(self, camera_info_msg):
        if self.camera_info is not None:
            old_fx = self.camera_model.fx()
            old_fy = self.camera_model.fy()
            old_cx = self.camera_model.cx()
            old_cy = self.camera_model.cy()
            if (
                abs(camera_info_msg.width - self.camera_info.width) > 0
                or abs(camera_info_msg.height - self.camera_info.height) > 0
                or abs(camera_info_msg.K[0] - old_fx) > 1e-3
                or abs(camera_info_msg.K[4] - old_fy) > 1e-3
                or abs(camera_info_msg.K[2] - old_cx) > 1e-3
                or abs(camera_info_msg.K[5] - old_cy) > 1e-3
            ):
                rospy.loginfo(
                    "CameraInfo override: %sx%s fx=%.3f fy=%.3f cx=%.3f cy=%.3f -> %sx%s fx=%.3f fy=%.3f cx=%.3f cy=%.3f",
                    self.camera_info.width,
                    self.camera_info.height,
                    old_fx,
                    old_fy,
                    old_cx,
                    old_cy,
                    camera_info_msg.width,
                    camera_info_msg.height,
                    camera_info_msg.K[0],
                    camera_info_msg.K[4],
                    camera_info_msg.K[2],
                    camera_info_msg.K[5],
                )
        self.camera_info = camera_info_msg
        self.camera_model.fromCameraInfo(camera_info_msg)
        self.cluster_postprocess.image_size = (
            camera_info_msg.width,
            camera_info_msg.height,
        )

    def tf_static_callback(self, tf_msg):
        for transform in tf_msg.transforms:
            if transform.header.frame_id != self.runtime_params.airsim_odom_frame:
                continue
            if transform.child_frame_id == self.runtime_params.airsim_camera_body_frame:
                self.airsim_camera_body_translation = np.array(
                    [
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z,
                    ]
                )
                self._airsim_body_static_ready = True
    def depth_mask_callback(self, depth_mask_msg):
        self.depth_mask_msg = depth_mask_msg

    def gimbal_callback(self, gimbal_msg):
        self.gimbal_msg = gimbal_msg

    def cluster_pc_callback(self, cluster_msg):
        self.cluster_runtime.update_current_target_pointcloud(cluster_msg)

    def cluster_accumulated_pc_callback(self, cluster_msg):
        self.cluster_runtime.update_accumulated_target_pointcloud(cluster_msg)

    def _pose_msg_to_matrix(self, position, orientation):
        translation = [position.x, position.y, position.z]
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        transform = tft.quaternion_matrix(quaternion)
        transform[:3, 3] = translation
        return transform

    def _build_camera_pose_from_airsim(self, odom_msg, image_stamp=None):
        body_to_world = self._pose_msg_to_matrix(
            odom_msg.pose.pose.position,
            odom_msg.pose.pose.orientation,
        )
        camera_translation_world = (
            body_to_world[:3, :3] @ self.airsim_camera_body_translation
        ) + body_to_world[:3, 3]
        camera_pose = np.eye(4)
        gimbal_rotation = self._build_airsim_gimbal_optical_rotation(
            odom_msg=odom_msg,
            image_stamp=image_stamp,
        )
        if gimbal_rotation is None:
            gimbal_rotation = body_to_world[:3, :3] @ self.airsim_camera_optical_rotation
        camera_pose[:3, :3] = gimbal_rotation
        camera_pose[:3, 3] = camera_translation_world
        return camera_pose

    def _build_airsim_gimbal_optical_rotation(self, odom_msg, image_stamp=None):
        if self.gimbal_msg is None:
            return None
        if not self._stamp_is_close(
            self.gimbal_msg.header.stamp,
            image_stamp,
            self.gimbal_max_stamp_delta,
        ):
            rospy.logwarn_throttle(
                2.0,
                "Skip AirSim gimbal pose with stale stamp: gimbal=%s image=%s max_delta=%.3fs",
                self.gimbal_msg.header.stamp.to_sec(),
                image_stamp.to_sec() if image_stamp is not None else -1.0,
                self.gimbal_max_stamp_delta.to_sec(),
            )
            return None

        body_quat = [
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w,
        ]
        gimbal_quat = [
            self.gimbal_msg.quaternion.x,
            self.gimbal_msg.quaternion.y,
            self.gimbal_msg.quaternion.z,
            self.gimbal_msg.quaternion.w,
        ]
        roll_body, pitch_body, yaw_body = tft.euler_from_quaternion(body_quat)
        roll_gimbal, pitch_gimbal, yaw_gimbal = tft.euler_from_quaternion(
            gimbal_quat
        )

        # Match Flyco's AirSim depth+gimbal convention: the gimbal message
        # carries yaw/pitch after AirSim's ENU/right-hand conversion. Roll is
        # optional for debugging because the original public path ignored it.
        yaw_gimbal += yaw_body
        pitch_gimbal += pitch_body
        if not self.airsim_gimbal_use_roll:
            roll_gimbal = 0.0
        gimbal_to_body = (
            tft.euler_matrix(0.0, 0.0, -yaw_gimbal)[:3, :3]
            @ tft.euler_matrix(0.0, -pitch_gimbal, 0.0)[:3, :3]
            @ tft.euler_matrix(roll_gimbal, 0.0, 0.0)[:3, :3]
        )
        body_to_world = tft.quaternion_matrix(body_quat)[:3, :3]
        return body_to_world @ gimbal_to_body @ self.airsim_camera_optical_rotation

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def tracking_timer_callback(self, event):
        if len(self.img_odom_data) == 0:
            rospy.logdebug_throttle(5.0, "No synchronized image and odometry yet")
            return

        latest = self.img_odom_data[-1]
        img_msg, odom_msg = (latest["img"], latest["odom"])

        self.img_odom_data.clear()

        if self.camera_info is None or len(self.pointcloud_msgs) == 0:
            rospy.loginfo("No camera info or pointcloud data")
            return

        body2world = self._pose_msg_to_matrix(
            odom_msg.pose.pose.position,
            odom_msg.pose.pose.orientation,
        )
        camera_pose = None
        if self.airsim_camera_pose_enabled:
            if not self._airsim_body_static_ready:
                rospy.loginfo_throttle(
                    5.0,
                    "Waiting for AirSim camera body translation from %s",
                    self.runtime_params.tf_static_topic,
                )
                return
            camera_pose = self._build_camera_pose_from_airsim(
                odom_msg,
                image_stamp=img_msg.header.stamp,
            )
        T = body2world
        img = cv2.cvtColor(self._bridge.imgmsg_to_cv2(img_msg), cv2.COLOR_BGR2RGB)
        orin_frame = img.copy()
        img = cv2.resize(img, (640, 480))

        ok, reason, provider_action, initial_mask = self.pipeline.sync_provider_state(
            frame_rgb=img
        )
        if not ok:
            rospy.logwarn(reason)
            return
        if provider_action is not None:
            rospy.loginfo("Provider action: %s", provider_action)
        if not self.provider.initialized:
            return

        if initial_mask is not None:
            tracked = self.pipeline.build_tracked_result_from_mask(
                frame_rgb=img,
                track_mask=initial_mask,
                mask_with_transform=self.mask_T.mask_with_transform,
            )
        else:
            depth_prompt_points = None
            depth_mask = self._load_depth_mask(img_msg.header.stamp)
            if depth_mask is None and self.depth_guidance.enabled:
                cluster_guidance = self._load_cluster_depth_mask(
                    body_to_world=T,
                    original_frame=orin_frame,
                    image_stamp=img_msg.header.stamp,
                    camera_pose=camera_pose,
                )
                if cluster_guidance is not None:
                    depth_mask = cluster_guidance.get("mask")
                    depth_prompt_points = cluster_guidance.get("prompt_points")
            tracked = self.pipeline.track_frame(
                frame_rgb=img,
                mask_with_transform=self.mask_T.mask_with_transform,
                depth_mask=depth_mask,
                depth_prompt_points=depth_prompt_points,
            )
        if tracked is None:
            return
        if tracked["state"] == "reinitialized":
            rospy.loginfo(
                "Depth guidance reinitialized provider: reason=%s, bbox=%s, depth_area=%s, consistency=%s",
                tracked["reinit_reason"],
                tracked["reinit_bbox"],
                tracked["depth_area"],
                tracked["depth_mask_iou"],
            )
            self._save_reinit_debug_event(
                frame_rgb=img,
                stamp=img_msg.header.stamp,
                tracked=tracked,
            )
            return
        if tracked.get("reinit_reason"):
            rospy.loginfo(
                "Depth guidance reinitialized provider: reason=%s, bbox=%s, depth_area=%s, consistency=%s",
                tracked["reinit_reason"],
                tracked["reinit_bbox"],
                tracked["depth_area"],
                tracked["depth_mask_iou"],
            )
            self._save_reinit_debug_event(
                frame_rgb=img,
                stamp=img_msg.header.stamp,
                tracked=tracked,
            )

        self.frame_idx += 1
        mask_image_msg = self._bridge.cv2_to_imgmsg(tracked["masked_rgb"], "rgb8")
        mask_image_msg.header.stamp = img_msg.header.stamp
        self.img_pub.publish(mask_image_msg)

        width, height = self.camera_info.width, self.camera_info.height
        mask_resized = cv2.resize(tracked["track_mask"], (width, height))

        points_msg_list = self._select_pointcloud_msgs_for_image(img_msg.header.stamp)
        self.pointcloud_msgs.clear()
        if len(points_msg_list) == 0:
            rospy.loginfo("No timestamp-aligned pointclouds")
            return

        projection = self.pipeline.project_target_points(
            pointcloud_msgs=points_msg_list,
            mask_resized=mask_resized,
            body_to_world=T,
            static_transforms=self.static_transforms,
            image_size=(width, height),
            camera_pose=camera_pose,
        )
        self._log_numpy_point_stats(
            "tracking selected_points_world", projection["selected_points_world"]
        )
        self._log_numpy_point_stats(
            "tracking direction_points", projection["direction_points"]
        )

        project_pc_uint8 = projection["project_pc"].astype(np.uint8)
        project_pc_msg = self._bridge.cv2_to_imgmsg(project_pc_uint8, "mono8")
        project_pc_msg.header.stamp = img_msg.header.stamp
        self.project_pc_pub.publish(project_pc_msg)

        time_stamp = img_msg.header.stamp
        if time_stamp == rospy.Time():
            time_stamp = rospy.Time.now()
        debug_point = np.array(projection["points_list"])
        debug_points_msg = self.pointcloud_projector.create_world_cloud(
            debug_point, time_stamp
        )
        self.dubug_pc_pub.publish(debug_points_msg)

        if len(projection["selected_points_world"]) != 0:
            rospy.loginfo("points count: %s", len(projection["selected_points_world"]))
            sam_mask_pc_msg = self.pointcloud_projector.create_world_cloud(
                projection["selected_points_world"], time_stamp
            )
            self.pc_pub.publish(sam_mask_pc_msg)
            self.odom_pub.publish(odom_msg)

            if len(projection["direction_points"]) != 0:
                self.cluster_runtime.publish_guide_pointcloud(
                    projection["direction_points"],
                    time_stamp,
                    self.cluster_guide_pub,
                )

            tracked["mask_rgba"][:, :, 3] = tracked["mask_rgba"][:, :, 3] * 255
            pose_to_publish = camera_pose if camera_pose is not None else body2world
            child_frame_id = (
                self.runtime_params.airsim_camera_optical_frame
                if camera_pose is not None
                else "drone_1"
            )
            self.mask_T.set_transform(
                pose_to_publish,
                frame_id="world",
                child_frame_id=child_frame_id,
                stamp=img_msg.header.stamp,
            )
            tracked["mask_with_transform_msg"].mask_image = self._bridge.cv2_to_imgmsg(
                tracked["mask_rgba"]
            )
            tracked["mask_with_transform_msg"].mask_image.header.stamp = (
                img_msg.header.stamp
            )
            self.transform_publisher.publish(tracked["mask_with_transform_msg"])
            compress_msg = build_compress_mask_with_pose(
                mask_rgba=tracked["mask_rgba"],
                transform=tracked["mask_with_transform_msg"].transform,
                target_size=(640, 480),
            )
            if compress_msg is not None:
                self.compress_mask_publisher.publish(compress_msg)
        else:
            rospy.loginfo("No points")

        rospy.loginfo("Publishing mask and transform")

        self.cluster_runtime.publish_projection_or_empty(
            body_to_world=T,
            original_frame=orin_frame,
            image_size=(width, height),
            current_points_cam=projection["points_list"],
            camera_pose=camera_pose,
            stamp=img_msg.header.stamp,
        )
        self._save_guidance_debug_frame(
            frame_rgb=img,
            stamp=img_msg.header.stamp,
            tracked=tracked,
            body_to_world=T,
            original_frame=orin_frame,
            current_points_cam=projection["points_list"],
            camera_pose=camera_pose,
        )
        self._save_reinit_next_debug_if_needed(
            frame_rgb=img,
            stamp=img_msg.header.stamp,
            tracked=tracked,
            body_to_world=T,
            original_frame=orin_frame,
            current_points_cam=projection["points_list"],
            camera_pose=camera_pose,
        )

    def _save_guidance_debug_frame(
        self,
        frame_rgb,
        stamp,
        tracked,
        body_to_world,
        original_frame,
        current_points_cam,
        camera_pose=None,
    ):
        if not self.reinit_debug_dir:
            return

        _, cluster_mask = self.cluster_runtime.project_target_mask(
            body_to_world=body_to_world,
            original_frame=original_frame,
            current_points_cam=current_points_cam,
            camera_pose=camera_pose,
        )
        self._guidance_debug_count += 1
        frame_dir = os.path.join(
            self.reinit_debug_dir,
            "guidance_%04d_t%.6f" % (self._guidance_debug_count, stamp.to_sec()),
        )
        os.makedirs(frame_dir, exist_ok=True)

        self._write_debug_image(
            os.path.join(frame_dir, "01_sam_mask.jpg"),
            self._annotate_debug_frame(
                self._overlay_mask(frame_rgb, tracked.get("track_mask"), color=(255, 0, 0)),
                stamp,
            ),
        )
        self._write_debug_image(
            os.path.join(frame_dir, "02_cluster_depth_mask.jpg"),
            self._annotate_debug_frame(
                self._overlay_mask(frame_rgb, cluster_mask, color=(0, 255, 0)),
                stamp,
            ),
        )
        self._write_debug_image(
            os.path.join(frame_dir, "03_reinit_prompt_empty.jpg"),
            self._annotate_debug_frame(self._empty_prompt_panel(frame_rgb), stamp),
        )
        self._write_debug_image(
            os.path.join(frame_dir, "triplet_sam_cluster_prompt.jpg"),
            self._annotate_debug_frame(
                self._make_guidance_triplet(
                    frame_rgb=frame_rgb,
                    sam_mask=tracked.get("track_mask"),
                    cluster_mask=cluster_mask,
                    prompt=None,
                ),
                stamp,
            ),
        )
        metadata = {
            "stamp": stamp.to_sec(),
            "frame_index": self._guidance_debug_count,
            "reinit_reason": tracked.get("reinit_reason"),
            "depth_mask_iou": tracked.get("depth_mask_iou"),
            "cluster_mask_area": (
                0 if cluster_mask is None else int(np.count_nonzero(cluster_mask))
            ),
            "sam_mask_area": int(np.count_nonzero(tracked.get("track_mask"))),
        }
        with open(os.path.join(frame_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        self._write_guidance_contact_sheet()

    def _save_reinit_debug_event(self, frame_rgb, stamp, tracked):
        if not self.reinit_debug_dir:
            return

        self._reinit_debug_count += 1
        event_dir = os.path.join(
            self.reinit_debug_dir,
            "reinit_%03d_t%.6f" % (self._reinit_debug_count, stamp.to_sec()),
        )
        os.makedirs(event_dir, exist_ok=True)

        pre_mask = tracked.get("pre_reinit_mask")
        depth_mask = tracked.get("reinit_depth_mask")
        after_mask = tracked.get("track_mask")
        prompt = tracked.get("reinit_prompt")

        self._write_debug_image(
            os.path.join(event_dir, "00_input.jpg"),
            self._annotate_debug_frame(self._rgb_to_bgr(frame_rgb), stamp),
        )
        self._write_debug_image(
            os.path.join(event_dir, "01_sam_before_reinit.jpg"),
            self._annotate_debug_frame(
                self._overlay_mask(frame_rgb, pre_mask, color=(255, 0, 0)),
                stamp,
            ),
        )
        self._write_debug_image(
            os.path.join(event_dir, "02_cluster_mask_for_reinit.jpg"),
            self._annotate_debug_frame(
                self._overlay_mask(frame_rgb, depth_mask, color=(0, 255, 0)),
                stamp,
            ),
        )
        self._write_debug_image(
            os.path.join(event_dir, "03_reinit_prompt.jpg"),
            self._annotate_debug_frame(self._draw_prompt(frame_rgb, prompt), stamp),
        )
        self._write_debug_image(
            os.path.join(event_dir, "triplet_sam_cluster_prompt.jpg"),
            self._annotate_debug_frame(
                self._make_reinit_triplet(frame_rgb, pre_mask, depth_mask, prompt),
                stamp,
            ),
        )
        self._write_debug_image(
            os.path.join(event_dir, "04_mask_after_prompt.jpg"),
            self._annotate_debug_frame(
                self._overlay_mask(frame_rgb, after_mask, color=(255, 0, 0)),
                stamp,
            ),
        )
        self._write_debug_summary(
            event_dir=event_dir,
            stamp=stamp,
            tracked=tracked,
            prompt=prompt,
        )
        self._pending_reinit_next_debug = {
            "event_dir": event_dir,
            "stamp": stamp,
        }
        self._write_reinit_contact_sheet()

    def _save_reinit_next_debug_if_needed(
        self,
        frame_rgb,
        stamp,
        tracked,
        body_to_world,
        original_frame,
        current_points_cam,
        camera_pose=None,
    ):
        if not self.reinit_debug_dir or self._pending_reinit_next_debug is None:
            return
        if stamp <= self._pending_reinit_next_debug["stamp"]:
            return

        event_dir = self._pending_reinit_next_debug["event_dir"]
        self._write_debug_image(
            os.path.join(event_dir, "05_next_tracking_mask.jpg"),
            self._annotate_debug_frame(
                self._overlay_mask(frame_rgb, tracked.get("track_mask"), color=(255, 0, 0)),
                stamp,
            ),
        )

        _, cluster_mask = self.cluster_runtime.project_target_mask(
            body_to_world=body_to_world,
            original_frame=original_frame,
            current_points_cam=current_points_cam,
            camera_pose=camera_pose,
        )
        self._write_debug_image(
            os.path.join(event_dir, "06_next_cluster_mask.jpg"),
            self._annotate_debug_frame(
                self._overlay_mask(frame_rgb, cluster_mask, color=(0, 255, 0)),
                stamp,
            ),
        )
        self._write_debug_image(
            os.path.join(event_dir, "07_next_input.jpg"),
            self._annotate_debug_frame(self._rgb_to_bgr(frame_rgb), stamp),
        )
        self._pending_reinit_next_debug = None
        self._write_reinit_contact_sheet()

    def _make_reinit_triplet(self, frame_rgb, pre_mask, depth_mask, prompt):
        panels = [
            self._overlay_mask(frame_rgb, pre_mask, color=(255, 0, 0)),
            self._overlay_mask(frame_rgb, depth_mask, color=(0, 255, 0)),
            self._draw_prompt(frame_rgb, prompt),
        ]
        labels = ["SAM before reinit", "Cluster/depth mask", "Reinit prompt"]
        return self._make_labeled_grid([panels], [labels])

    def _make_guidance_triplet(self, frame_rgb, sam_mask, cluster_mask, prompt):
        panels = [
            self._overlay_mask(frame_rgb, sam_mask, color=(255, 0, 0)),
            self._overlay_mask(frame_rgb, cluster_mask, color=(0, 255, 0)),
            self._draw_prompt(frame_rgb, prompt)
            if prompt is not None
            else self._empty_prompt_panel(frame_rgb),
        ]
        labels = ["SAM mask", "Cluster/depth mask", "Reinit prompt"]
        return self._make_labeled_grid([panels], [labels])

    def _write_guidance_contact_sheet(self):
        if not self.reinit_debug_dir:
            return
        rows = []
        labels = []
        frame_dirs = [
            os.path.join(self.reinit_debug_dir, name)
            for name in sorted(os.listdir(self.reinit_debug_dir))
            if name.startswith("guidance_")
            and os.path.isdir(os.path.join(self.reinit_debug_dir, name))
        ]
        for frame_dir in frame_dirs:
            image_paths = [
                os.path.join(frame_dir, "01_sam_mask.jpg"),
                os.path.join(frame_dir, "02_cluster_depth_mask.jpg"),
                os.path.join(frame_dir, "03_reinit_prompt_empty.jpg"),
            ]
            row = [cv2.imread(path) for path in image_paths]
            if any(image is None for image in row):
                continue
            frame_name = os.path.basename(frame_dir)
            rows.append(row)
            labels.append(
                [
                    "%s | SAM mask" % frame_name,
                    "%s | cluster/depth" % frame_name,
                    "%s | no prompt" % frame_name,
                ]
            )
        if not rows:
            return
        contact_sheet = self._make_labeled_grid(rows, labels)
        self._write_debug_image(
            os.path.join(self.reinit_debug_dir, "guidance_frames_nx3.jpg"),
            contact_sheet,
        )

    def _write_reinit_contact_sheet(self):
        if not self.reinit_debug_dir:
            return
        rows = []
        labels = []
        event_dirs = [
            os.path.join(self.reinit_debug_dir, name)
            for name in sorted(os.listdir(self.reinit_debug_dir))
            if name.startswith("reinit_")
            and os.path.isdir(os.path.join(self.reinit_debug_dir, name))
        ]
        for event_dir in event_dirs:
            image_paths = [
                os.path.join(event_dir, "01_sam_before_reinit.jpg"),
                os.path.join(event_dir, "02_cluster_mask_for_reinit.jpg"),
                os.path.join(event_dir, "03_reinit_prompt.jpg"),
            ]
            row = [cv2.imread(path) for path in image_paths]
            if any(image is None for image in row):
                continue
            event_name = os.path.basename(event_dir)
            rows.append(row)
            labels.append(
                [
                    "%s | SAM before" % event_name,
                    "%s | cluster/depth" % event_name,
                    "%s | prompt" % event_name,
                ]
            )
        if not rows:
            return
        contact_sheet = self._make_labeled_grid(rows, labels)
        self._write_debug_image(
            os.path.join(self.reinit_debug_dir, "reinit_events_nx3.jpg"),
            contact_sheet,
        )

    def _make_labeled_grid(self, rows, labels):
        normalized_rows = []
        for row in rows:
            target_h = min(image.shape[0] for image in row)
            normalized = []
            for image in row:
                if image.shape[0] != target_h:
                    width = int(round(image.shape[1] * target_h / float(image.shape[0])))
                    image = cv2.resize(image, (width, target_h))
                normalized.append(image)
            normalized_rows.append(normalized)

        cell_h = max(image.shape[0] for row in normalized_rows for image in row)
        cell_w = max(image.shape[1] for row in normalized_rows for image in row)
        label_h = 28
        margin = 6
        sheet_h = len(normalized_rows) * (cell_h + label_h + margin) + margin
        sheet_w = len(normalized_rows[0]) * (cell_w + margin) + margin
        sheet = np.full((sheet_h, sheet_w, 3), 32, dtype=np.uint8)
        for row_idx, row in enumerate(normalized_rows):
            for col_idx, image in enumerate(row):
                y0 = margin + row_idx * (cell_h + label_h + margin)
                x0 = margin + col_idx * (cell_w + margin)
                label = labels[row_idx][col_idx]
                cv2.putText(
                    sheet,
                    label,
                    (x0, y0 + 19),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )
                image_y0 = y0 + label_h
                sheet[
                    image_y0 : image_y0 + image.shape[0],
                    x0 : x0 + image.shape[1],
                ] = image
        return sheet

    def _write_debug_summary(self, event_dir, stamp, tracked, prompt):
        coords = []
        labels = []
        text = ""
        if prompt is not None:
            coords = np.asarray(prompt.get("coords", [])).tolist()
            labels = np.asarray(prompt.get("labels", [])).tolist()
            text = str(prompt.get("text", ""))
        summary = {
            "stamp": stamp.to_sec(),
            "reason": tracked.get("reinit_reason"),
            "bbox": tracked.get("reinit_bbox"),
            "depth_area": tracked.get("depth_area"),
            "consistency": tracked.get("depth_mask_iou"),
            "prompt": {
                "coords": coords,
                "labels": labels,
                "text": text,
            },
        }
        with open(os.path.join(event_dir, "metadata.json"), "w") as f:
            json.dump(summary, f, indent=2)

    def _draw_prompt(self, frame_rgb, prompt):
        image = self._rgb_to_bgr(frame_rgb)
        if prompt is None:
            return image

        coords = np.asarray(prompt.get("coords", []), dtype=np.int32)
        labels = np.asarray(prompt.get("labels", []), dtype=np.int32).reshape(-1)
        if len(coords) >= 2 and len(labels) >= 2 and labels[0] == 2 and labels[1] == 3:
            x0, y0 = coords[0]
            x1, y1 = coords[1]
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 255), 2)
            cv2.putText(
                image,
                "bbox",
                (x0, max(18, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        for point, label in zip(coords, labels):
            x, y = int(point[0]), int(point[1])
            if label == 1:
                color = (0, 255, 0)
                name = "pos"
            elif label == 0:
                color = (255, 0, 0)
                name = "neg"
            else:
                continue
            cv2.circle(image, (x, y), 5, color, thickness=-1)
            cv2.putText(
                image,
                name,
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
        return image

    def _empty_prompt_panel(self, frame_rgb):
        height, width = frame_rgb.shape[:2]
        image = np.full((height, width, 3), 32, dtype=np.uint8)
        cv2.putText(
            image,
            "no reinit prompt",
            (24, max(42, height // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
        return image

    def _overlay_mask(self, frame_rgb, mask, color):
        image = self._rgb_to_bgr(frame_rgb)
        if mask is None:
            return image
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        binary = mask > 0
        color_layer = np.zeros_like(image)
        color_layer[:, :] = color
        image[binary] = (image[binary] * 0.52 + color_layer[binary] * 0.48).astype(
            np.uint8
        )
        return image

    def _rgb_to_bgr(self, frame_rgb):
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def _write_debug_image(self, path, image):
        cv2.imwrite(path, image)

    def _format_gimbal_overlay_text(self, image_stamp):
        if not self.runtime_params.gimbal_topic:
            return "gimbal: disabled"
        if self.gimbal_msg is None:
            return "gimbal: no message"

        stamp_sec = (
            image_stamp.to_sec()
            if image_stamp is not None and image_stamp != rospy.Time()
            else None
        )
        gimbal_stamp = self.gimbal_msg.header.stamp
        gimbal_sec = (
            gimbal_stamp.to_sec()
            if gimbal_stamp is not None and gimbal_stamp != rospy.Time()
            else None
        )
        delta_text = "n/a"
        if stamp_sec is not None and gimbal_sec is not None:
            delta_text = "%.3fs" % abs(stamp_sec - gimbal_sec)

        quat = [
            self.gimbal_msg.quaternion.x,
            self.gimbal_msg.quaternion.y,
            self.gimbal_msg.quaternion.z,
            self.gimbal_msg.quaternion.w,
        ]
        roll, pitch, yaw = tft.euler_from_quaternion(quat)
        used_roll = roll if self.airsim_gimbal_use_roll else 0.0
        return (
            "gimbal raw/use rpy(deg): r=%.1f->%.1f p=%.1f y=%.1f dt=%s"
            % (
                np.degrees(roll),
                np.degrees(used_roll),
                np.degrees(pitch),
                np.degrees(yaw),
                delta_text,
            )
        )

    def _annotate_debug_frame(self, image, stamp):
        annotated = image.copy()
        line = self._format_gimbal_overlay_text(stamp)
        cv2.rectangle(annotated, (10, 10), (760, 54), (24, 24, 24), -1)
        cv2.putText(
            annotated,
            line,
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def _log_numpy_point_stats(self, name, points):
        now = rospy.get_time()
        last = self._point_stats_last_log_time.get(name)
        if last is not None and (now - last) < 1.0:
            return
        self._point_stats_last_log_time[name] = now
        if points is None:
            rospy.loginfo("%s: none", name)
            return
        points = np.asarray(points)
        if points.size == 0:
            rospy.loginfo("%s: empty", name)
            return
        if points.ndim != 2 or points.shape[1] < 3:
            rospy.loginfo("%s: unexpected shape=%s", name, tuple(points.shape))
            return
        z = points[:, 2]
        rospy.loginfo(
            "%s: n=%d z[min/p1/p5/med/p95/max]=%.3f/%.3f/%.3f/%.3f/%.3f/%.3f",
            name,
            len(points),
            float(np.min(z)),
            float(np.percentile(z, 1)),
            float(np.percentile(z, 5)),
            float(np.median(z)),
            float(np.percentile(z, 95)),
            float(np.max(z)),
        )

    def filter_points_below_height(self, point_cloud_msg, min_height):
        points = point_cloud2.read_points(
            point_cloud_msg, field_names=("x", "y", "z"), skip_nans=True
        )
        filtered_points = [point for point in points if point[2] >= min_height]
        header = point_cloud_msg.header
        fields = point_cloud_msg.fields
        filtered_point_cloud_msg = point_cloud2.create_cloud(
            header, fields, filtered_points
        )

        return filtered_point_cloud_msg

    def set_query_points(self, query_points):
        self.query_points = query_points
        self.prompt_session.set_query_points(query_points)

    def query_point_callback(self, msg):
        flat_data = msg.data
        self.query_points.clear()

        if len(flat_data) % 3 == 0:
            from geometry_msgs.msg import Point as PointMsg

            self.query_points = [
                PointMsg(x=flat_data[i], y=flat_data[i + 1], z=flat_data[i + 2])
                for i in range(0, len(flat_data), 3)
            ]

        rospy.loginfo(f"Received query points: {self.query_points}")
        self.prompt_session.set_query_points(self.query_points)

    def query_text_callback(self, msg):
        self.query_text = msg.data
        self.prompt_session.set_query_text(self.query_text)

    def _load_depth_mask(self, image_stamp=None):
        if self.depth_mask_msg is None:
            return None
        if not self._stamp_is_close(
            self.depth_mask_msg.header.stamp,
            image_stamp,
            self.cluster_depth_mask_max_stamp_delta,
        ):
            rospy.logwarn_throttle(
                2.0,
                "Skip depth mask with stale stamp: depth=%s image=%s max_delta=%.3fs",
                self.depth_mask_msg.header.stamp.to_sec(),
                image_stamp.to_sec() if image_stamp is not None else -1.0,
                self.cluster_depth_mask_max_stamp_delta.to_sec(),
            )
            return None

        return self._bridge.imgmsg_to_cv2(
            self.depth_mask_msg, desired_encoding="passthrough"
        )

    def _load_cluster_depth_mask(
        self,
        body_to_world,
        original_frame,
        image_stamp=None,
        camera_pose=None,
    ):
        target_stamp = self.cluster_runtime.target_stamp()
        if not self._stamp_is_close(
            target_stamp,
            image_stamp,
            self.cluster_depth_mask_max_stamp_delta,
        ):
            rospy.logwarn_throttle(
                2.0,
                "Skip cluster depth mask with stale target: target=%s image=%s max_delta=%.3fs",
                target_stamp.to_sec() if target_stamp is not None else -1.0,
                image_stamp.to_sec() if image_stamp is not None else -1.0,
                self.cluster_depth_mask_max_stamp_delta.to_sec(),
            )
            return None

        _, cluster_mask, metadata = self.cluster_runtime.project_target_mask(
            body_to_world=body_to_world,
            original_frame=original_frame,
            current_points_cam=None,
            camera_pose=camera_pose,
            return_metadata=True,
        )
        if cluster_mask is None or np.count_nonzero(cluster_mask) == 0:
            return None
        return {
            "mask": cluster_mask,
            "prompt_points": metadata.get("prompt_points", []),
        }

    def _stamp_is_close(self, data_stamp, image_stamp, max_delta):
        if data_stamp is None or image_stamp is None:
            return False
        if data_stamp == rospy.Time() or image_stamp == rospy.Time():
            return False
        return abs((data_stamp - image_stamp).to_sec()) <= max_delta.to_sec()

    def _select_pointcloud_msgs_for_image(self, image_stamp):
        if image_stamp is None or image_stamp == rospy.Time():
            return self.pointcloud_msgs[-self.pointcloud_max_count :]
        scored = []
        stale_count = 0
        for msg in self.pointcloud_msgs:
            stamp = msg.header.stamp
            if stamp is None or stamp == rospy.Time():
                continue
            delta = abs((stamp - image_stamp).to_sec())
            if delta <= self.pointcloud_max_stamp_delta.to_sec():
                scored.append((delta, msg))
            else:
                stale_count += 1
        if not scored:
            rospy.logwarn_throttle(
                2.0,
                "No pointcloud within %.3fs of image stamp %.6f; dropped=%d buffered=%d",
                self.pointcloud_max_stamp_delta.to_sec(),
                image_stamp.to_sec(),
                stale_count,
                len(self.pointcloud_msgs),
            )
            return []
        scored.sort(key=lambda item: item[0])
        selected = [msg for _, msg in scored[: self.pointcloud_max_count]]
        rospy.loginfo(
            "Selected %d/%d aligned pointclouds for image %.6f; best_delta=%.4fs worst_delta=%.4fs",
            len(selected),
            len(self.pointcloud_msgs),
            image_stamp.to_sec(),
            scored[0][0],
            scored[min(len(scored), self.pointcloud_max_count) - 1][0],
        )
        return selected

    def run(self):
        rospy.spin()
