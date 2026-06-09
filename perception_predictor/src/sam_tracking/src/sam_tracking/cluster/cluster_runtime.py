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


class ClusterRuntime:
    def __init__(
        self,
        bridge,
        cluster_postprocess,
        pointcloud_projector,
        projection_pub,
        mask_pub,
        target_pc_topic,
        accumulated_target_pc_topic=None,
        incremental_config=None,
        projection_use_accumulated=False,
    ):
        self.bridge = bridge
        self.cluster_postprocess = cluster_postprocess
        self.pointcloud_projector = pointcloud_projector
        self.cluster_projection_pub = projection_pub
        self.cluster_mask_pub = mask_pub
        self.cluster_target_pc_topic = target_pc_topic
        self.cluster_accumulated_target_pc_topic = accumulated_target_pc_topic
        self.projection_use_accumulated = bool(projection_use_accumulated)
        self.current_cluster_target_pc = None
        self.accumulated_cluster_target_pc = None
        rospy.loginfo(
            "Cluster projection source: %s",
            "accumulated target" if self.projection_use_accumulated else "current-frame target",
        )

    def _projection_target_pc(self):
        if self.projection_use_accumulated:
            return self.accumulated_cluster_target_pc
        return self.current_cluster_target_pc

    def has_target(self):
        return self._projection_target_pc() is not None

    def reset(self):
        self.current_cluster_target_pc = None
        self.accumulated_cluster_target_pc = None

    def update_current_target_pointcloud(self, cluster_msg):
        self.current_cluster_target_pc = cluster_msg

    def update_accumulated_target_pointcloud(self, cluster_msg):
        self.accumulated_cluster_target_pc = cluster_msg

    def publish_guide_pointcloud(self, direction_points, stamp, guide_pub):
        if direction_points is None or len(direction_points) == 0:
            return
        direction_pc_msg = self.pointcloud_projector.create_world_cloud(
            direction_points, stamp
        )
        guide_pub.publish(direction_pc_msg)

    def publish_projection_or_empty(
        self,
        body_to_world,
        original_frame,
        image_size,
        current_points_cam=None,
        camera_pose=None,
        stamp=None,
    ):
        width, height = image_size
        cluster_target_pc = self._projection_target_pc()
        if cluster_target_pc is None:
            self._publish_empty(original_frame, width, height, stamp=stamp)
            return

        cluster_overlay, cluster_mask = self.project_target_mask(
            body_to_world=body_to_world,
            original_frame=original_frame,
            current_points_cam=current_points_cam,
            camera_pose=camera_pose,
        )
        if cluster_overlay is None:
            self._publish_empty(original_frame, width, height, stamp=stamp)
            return

        cluster_overlay_msg = self.bridge.cv2_to_imgmsg(cluster_overlay, "rgb8")
        cluster_mask_msg = self.bridge.cv2_to_imgmsg(cluster_mask, "mono8")
        if stamp is not None:
            cluster_overlay_msg.header.stamp = stamp
            cluster_mask_msg.header.stamp = stamp
        self.cluster_projection_pub.publish(cluster_overlay_msg)
        self.cluster_mask_pub.publish(cluster_mask_msg)
        source_topic = (
            self.cluster_accumulated_target_pc_topic
            if self.projection_use_accumulated
            else self.cluster_target_pc_topic
        )
        rospy.loginfo(
            "Published cluster projection and mask from %s",
            source_topic,
        )

    def project_target_mask(
        self,
        body_to_world,
        original_frame,
        current_points_cam=None,
        camera_pose=None,
        return_metadata=False,
        prompt_point_count=4,
    ):
        cluster_target_pc = self._projection_target_pc()
        if cluster_target_pc is None:
            if return_metadata:
                return None, None, {"prompt_points": []}
            return None, None
        return self.cluster_postprocess.project_cluster_to_image(
            cluster_pc_msg=cluster_target_pc,
            body_to_world=body_to_world,
            original_frame=original_frame,
            current_points_cam=current_points_cam,
            camera_pose=camera_pose,
            return_metadata=return_metadata,
            prompt_point_count=prompt_point_count,
        )

    def target_stamp(self):
        cluster_target_pc = self._projection_target_pc()
        if cluster_target_pc is None:
            return None
        return cluster_target_pc.header.stamp

    def _publish_empty(self, original_frame, width, height, stamp=None):
        empty_mask = np.zeros((height, width), dtype=np.uint8)
        mask_msg = self.bridge.cv2_to_imgmsg(empty_mask, "mono8")
        overlay_msg = self.bridge.cv2_to_imgmsg(original_frame, "rgb8")
        if stamp is not None:
            mask_msg.header.stamp = stamp
            overlay_msg.header.stamp = stamp
        self.cluster_mask_pub.publish(mask_msg)
        self.cluster_projection_pub.publish(overlay_msg)
