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
import rospy
from sensor_msgs import point_cloud2
from dataclasses import dataclass
from scipy.spatial import ConvexHull


@dataclass(frozen=True)
class IncrementalClusterConfig:
    enabled: bool = False
    input_max_points: int = 1024
    voxel_size: float = 0.2
    sampling_seed: int = 20260517
    support_radius_scale: float = 1.75
    support_radius_margin: float = 1.0
    min_support_radius: float = 2.0
    max_centroid_jump: float = 8.0


class ClusterPostprocess:
    def __init__(
        self,
        camera_model,
        transform_pc_fn,
        static_rotation,
        static_transform,
        image_size,
        depth_threshold=0.3,
        use_hpr=True,
    ):
        self.camera_model = camera_model
        self.transform_pc_fn = transform_pc_fn
        self.static_rotation = static_rotation
        self.static_transform = static_transform
        self.image_size = image_size
        self.depth_threshold = depth_threshold
        self.use_hpr = bool(use_hpr)
        self._sampling_seed = 20260517

    def voxel_downsample(self, points, voxel_size):
        if len(points) == 0:
            return points

        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        voxel_dict = {}
        for i, idx in enumerate(voxel_indices):
            key = tuple(idx)
            voxel_dict.setdefault(key, []).append(points[i])

        return np.array([np.mean(pts, axis=0) for pts in voxel_dict.values()])

    def build_incremental_target(self, cluster_msg, accumulated_points, config):
        if len(cluster_msg.data) == 0:
            return None, accumulated_points

        points = point_cloud2.read_points(
            cluster_msg, field_names=("x", "y", "z"), skip_nans=True
        )
        new_points = np.array(list(points))
        if len(new_points) == 0:
            return None, accumulated_points

        if len(new_points) > config.input_max_points:
            stride = max(1, len(new_points) // config.input_max_points)
            new_points = new_points[::stride][: config.input_max_points]

        if len(accumulated_points) == 0:
            accumulated_points = new_points
        else:
            accumulated_points = self._merge_supported_points(
                accumulated_points=accumulated_points,
                new_points=new_points,
                config=config,
            )

        before_voxel = len(accumulated_points)
        accumulated_points = self.voxel_downsample(
            accumulated_points, config.voxel_size
        )
        after_voxel = len(accumulated_points)

        rospy.loginfo(
            "Cluster incremental: input %s, before voxel %s, after voxel %s",
            len(new_points),
            before_voxel,
            after_voxel,
        )
        return accumulated_points, accumulated_points

    def _merge_supported_points(self, accumulated_points, new_points, config):
        current_centroid = np.mean(new_points, axis=0)
        current_offsets = np.linalg.norm(new_points - current_centroid, axis=1)
        current_radius = float(np.percentile(current_offsets, 95))
        support_radius = max(
            config.min_support_radius,
            current_radius * config.support_radius_scale + config.support_radius_margin,
        )

        accumulated_centroid = np.mean(accumulated_points, axis=0)
        centroid_jump = float(np.linalg.norm(current_centroid - accumulated_centroid))
        if centroid_jump > config.max_centroid_jump:
            rospy.logwarn(
                "Cluster incremental centroid jump %.3f m exceeds %.3f m; keep merging without hard reset",
                centroid_jump,
                config.max_centroid_jump,
            )

        keep_mask = (
            np.linalg.norm(accumulated_points - current_centroid, axis=1)
            <= support_radius
        )
        kept_points = accumulated_points[keep_mask]
        dropped_count = int(len(accumulated_points) - len(kept_points))
        if dropped_count > 0:
            rospy.loginfo(
                "Cluster incremental pruned %s stale points outside support radius %.3f m",
                dropped_count,
                support_radius,
            )

        return np.vstack([kept_points, new_points])

    def project_cluster_to_image(
        self,
        cluster_pc_msg,
        body_to_world,
        original_frame,
        current_points_cam=None,
        camera_pose=None,
        return_metadata=False,
        prompt_point_count=4,
    ):
        if cluster_pc_msg is None:
            return self._empty_projection_result(return_metadata)

        points = point_cloud2.read_points(
            cluster_pc_msg, field_names=("x", "y", "z"), skip_nans=True
        )
        points_world = np.array(list(points), dtype=np.float32)
        if len(points_world) == 0:
            return self._empty_projection_result(return_metadata)

        camera_origin_world = self._camera_origin_world(
            body_to_world=body_to_world,
            camera_pose=camera_pose,
        )
        if self.use_hpr:
            visible_world, hidden_count = self._select_hpr_visible_points(
                points_world=points_world,
                camera_origin_world=camera_origin_world,
            )
        else:
            visible_world = points_world
            hidden_count = 0
        if len(visible_world) == 0:
            return self._empty_projection_result(return_metadata)

        if camera_pose is not None:
            rotation_wc = camera_pose[:3, :3]
            translation_wc = camera_pose[:3, 3]
            points_cam = (rotation_wc.transpose() @ (visible_world - translation_wc).T).T
        else:
            pc = visible_world - body_to_world[:3, 3]
            pc = np.linalg.inv(body_to_world[:3, :3]) @ (pc.transpose())
            pc = (np.linalg.inv(self.static_rotation) @ pc).transpose()

            points_cam = self.transform_pc_fn(pc, self.static_transform)
            points_cam = np.array(list(points_cam))
        if len(points_cam) == 0:
            return self._empty_projection_result(return_metadata)

        width, height = self.image_size
        valid_mask = points_cam[:, 2] > 0
        target_pts = points_cam[valid_mask]
        target_world = visible_world[valid_mask]
        if len(target_pts) == 0:
            return self._empty_projection_result(return_metadata)

        u_target, v_target, z_target, in_bounds = self._project_points(
            target_pts, width, height
        )
        if len(u_target) == 0:
            return self._empty_projection_result(return_metadata)
        target_world = target_world[in_bounds]

        u_target, v_target, z_target = self._keep_frontmost_target_pixels(
            u_target, v_target, z_target
        )
        u_visible = u_target
        v_visible = v_target
        prompt_points = self._sample_prompt_points_from_3d_cluster(
            target_world=target_world,
            u_target=u_target,
            v_target=v_target,
            max_points=prompt_point_count,
        )

        projection_mode = "HPR" if self.use_hpr else "direct"
        rospy.loginfo(
            "Cluster projection (%s): %s world-visible, %s hidden, %s projected pixels, %s prompt points",
            projection_mode,
            len(visible_world),
            hidden_count,
            len(u_visible),
            len(prompt_points),
        )

        projected_mask = np.zeros((height, width), dtype=np.uint8)
        overlay_image = original_frame.copy()
        for u, v in zip(u_visible, v_visible):
            cv2.circle(projected_mask, (u, v), radius=3, color=255, thickness=-1)
            cv2.circle(overlay_image, (u, v), radius=3, color=(0, 255, 0), thickness=-1)

        projected_mask = cv2.dilate(
            projected_mask, np.ones((5, 5), np.uint8), iterations=2
        )
        if not return_metadata:
            return overlay_image, projected_mask

        return (
            overlay_image,
            projected_mask,
            {
                "prompt_points": prompt_points,
                "projected_pixel_count": int(len(u_visible)),
                "visible_world_count": int(len(visible_world)),
                "hidden_world_count": int(hidden_count),
            },
        )

    def _empty_projection_result(self, return_metadata):
        if return_metadata:
            return None, None, {"prompt_points": []}
        return None, None

    def _camera_origin_world(self, body_to_world, camera_pose=None):
        if camera_pose is not None:
            return np.asarray(camera_pose[:3, 3], dtype=np.float32)
        return np.asarray(body_to_world[:3, 3], dtype=np.float32)

    def _select_hpr_visible_points(self, points_world, camera_origin_world):
        if len(points_world) <= 4:
            return points_world, 0

        translated = points_world - camera_origin_world.reshape(1, 3)
        distances = np.linalg.norm(translated, axis=1)
        valid_mask = distances > 1e-6
        if np.count_nonzero(valid_mask) < 4:
            valid_points = points_world[valid_mask]
            return valid_points, int(len(points_world) - len(valid_points))

        valid_points = points_world[valid_mask]
        valid_translated = translated[valid_mask]
        valid_distances = distances[valid_mask]
        radius = 3.0 * float(np.max(valid_distances))
        transformed = valid_translated + 2.0 * (
            radius - valid_distances
        ).reshape(-1, 1) * (
            valid_translated / valid_distances.reshape(-1, 1)
        )
        try:
            hull = ConvexHull(transformed, qhull_options="QJ")
            visible_valid_idx = np.unique(hull.vertices)
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0,
                "Cluster HPR fallback to valid points after ConvexHull failure: %s",
                exc,
            )
            visible_valid_idx = np.arange(len(valid_points))

        visible_world = valid_points[visible_valid_idx]
        hidden_count = int(len(points_world) - len(visible_world))
        return visible_world, hidden_count

    def _keep_frontmost_target_pixels(self, u, v, z):
        if len(u) == 0:
            return u, v, z

        sort_idx = np.argsort(z)
        u_sorted = u[sort_idx]
        v_sorted = v[sort_idx]
        z_sorted = z[sort_idx]

        pixel_pairs = np.stack((u_sorted, v_sorted), axis=1)
        _, unique_idx = np.unique(pixel_pairs, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        return u_sorted[unique_idx], v_sorted[unique_idx], z_sorted[unique_idx]

    def _sample_prompt_points_from_3d_cluster(
        self,
        target_world,
        u_target,
        v_target,
        max_points=4,
    ):
        if (
            max_points <= 0
            or len(target_world) == 0
            or len(u_target) == 0
            or len(v_target) == 0
        ):
            return []

        sample_count = min(int(max_points), len(target_world))
        sampled_indices = self._farthest_point_sample_indices(
            points=target_world,
            sample_count=sample_count,
        )
        prompt_points = []
        for idx in sampled_indices:
            if idx < 0 or idx >= len(u_target):
                continue
            point = [int(u_target[idx]), int(v_target[idx])]
            if point in prompt_points:
                continue
            prompt_points.append(point)
        return prompt_points

    def _farthest_point_sample_indices(self, points, sample_count):
        if sample_count <= 0 or len(points) == 0:
            return np.array([], dtype=np.int32)
        if sample_count >= len(points):
            return np.arange(len(points), dtype=np.int32)

        points = np.asarray(points, dtype=np.float32)
        centroid = np.mean(points, axis=0, keepdims=True)
        distances_to_centroid = np.linalg.norm(points - centroid, axis=1)
        selected = [int(np.argmax(distances_to_centroid))]
        min_distances = np.linalg.norm(points - points[selected[0]], axis=1)

        while len(selected) < sample_count:
            next_idx = int(np.argmax(min_distances))
            if next_idx in selected:
                break
            selected.append(next_idx)
            candidate_distances = np.linalg.norm(points - points[next_idx], axis=1)
            min_distances = np.minimum(min_distances, candidate_distances)

        return np.asarray(selected, dtype=np.int32)

    def _project_points(self, points_cam, width, height):
        fx = self.camera_model.fx()
        fy = self.camera_model.fy()
        cx = self.camera_model.cx()
        cy = self.camera_model.cy()

        z = points_cam[:, 2]
        u = (fx * points_cam[:, 0] / z + cx).astype(np.int32)
        v = (fy * points_cam[:, 1] / z + cy).astype(np.int32)
        in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        return u[in_bounds], v[in_bounds], z[in_bounds], in_bounds
