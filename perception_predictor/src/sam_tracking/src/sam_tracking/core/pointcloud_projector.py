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

import copy

import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
from sensor_msgs import point_cloud2


class PointCloudProjector:
    def __init__(
        self, camera_model, static_rotation, transform_pc_fn, filter_points_fn
    ):
        self.camera_model = camera_model
        self.static_rotation = static_rotation
        self.transform_pc = transform_pc_fn
        self.filter_points_below_height = filter_points_fn

    def project_mask_to_world_points(
        self,
        points_msg_list,
        mask_resized,
        body_to_world,
        static_transforms,
        image_size,
        camera_pose=None,
    ):
        if camera_pose is not None:
            return self._project_with_camera_pose(
                points_msg_list=points_msg_list,
                mask_resized=mask_resized,
                image_size=image_size,
                camera_pose=camera_pose,
            )

        width, height = image_size
        points_list = []

        for pc_msg in points_msg_list:
            pc = point_cloud2.read_points(
                pc_msg, field_names=("x", "y", "z"), skip_nans=True
            )
            static_transform = static_transforms["body2cam"]
            pc = np.array(list(pc))
            if len(pc) == 0:
                continue
            pc = pc - body_to_world[:3, 3]
            pc = np.linalg.inv(body_to_world[:3, :3]) @ (pc.transpose())
            pc = (np.linalg.inv(self.static_rotation) @ pc).transpose()
            points = self.transform_pc(pc, static_transform)
            points_list.extend(points)

        project_pc = np.zeros((height, width), dtype=np.uint8)
        # Visible surface selection should prefer the nearest point per pixel.
        points_list = sorted(points_list, key=lambda x: x[2])
        selected_points = []
        selected_source_points_camera = []

        for point in points_list:
            point_x, point_y, point_z = point[:3]
            if point_z <= 0:
                continue
            uv = self.camera_model.project3dToPixel((point_x, point_y, point_z))
            pixel_x, pixel_y = int(uv[0]), int(uv[1])
            if pixel_x < 0 or pixel_x >= width or pixel_y < 0 or pixel_y >= height:
                continue
            if project_pc[pixel_y, pixel_x] == 255:
                continue

            project_pc[pixel_y, pixel_x] = 255
            xy = self.camera_model.projectPixelTo3dRay([pixel_x, pixel_y])
            if mask_resized[pixel_y, pixel_x] > 0:
                selected_points.append(
                    [
                        xy[0] / xy[2] * point_z,
                        xy[1] / xy[2] * point_z,
                        point_z,
                    ]
                )
                selected_source_points_camera.append([point_x, point_y, point_z])

        if len(selected_points) == 0:
            return {
                "selected_points_world": np.empty((0, 3)),
                "points_list": points_list,
                "project_pc": project_pc,
                "direction_points": np.empty((0, 3)),
                "selected_points_camera": np.empty((0, 3)),
                "selected_source_points_camera": np.empty((0, 3)),
            }

        selected_points = np.array(selected_points)
        selected_source_points_camera = np.array(selected_source_points_camera)
        static_transform = static_transforms["cam2body"]
        point = self.transform_pc(selected_points, static_transform)
        selected_points_world = np.array(list(point))
        selected_points_world = self.static_rotation @ selected_points_world.transpose()
        selected_points_world = (
            body_to_world[:3, :3] @ selected_points_world
        ).transpose() + body_to_world[:3, 3]

        projected_points = pc2.create_cloud_xyz32(
            copy.deepcopy(points_msg_list[-1]).header, selected_points_world
        )
        direction_points, _ = self.convert_and_sample_points(projected_points)

        return {
            "selected_points_world": selected_points_world,
            "points_list": points_list,
            "project_pc": cv2.resize(project_pc, (width, height)),
            "direction_points": direction_points,
            "selected_points_camera": selected_points,
            "selected_source_points_camera": selected_source_points_camera,
        }

    def _project_with_camera_pose(
        self,
        points_msg_list,
        mask_resized,
        image_size,
        camera_pose,
    ):
        width, height = image_size
        world_points = []
        for pc_msg in points_msg_list:
            pc = point_cloud2.read_points(
                pc_msg, field_names=("x", "y", "z"), skip_nans=True
            )
            pc = np.array(list(pc))
            if len(pc) == 0:
                continue
            world_points.append(pc)

        if not world_points:
            return {
                "selected_points_world": np.empty((0, 3)),
                "points_list": np.empty((0, 3)),
                "project_pc": np.zeros((height, width), dtype=np.uint8),
                "direction_points": np.empty((0, 3)),
                "selected_points_camera": np.empty((0, 3)),
                "selected_source_points_camera": np.empty((0, 3)),
            }

        world_points = np.vstack(world_points)
        rotation_wc = camera_pose[:3, :3]
        translation_wc = camera_pose[:3, 3]
        points_cam = (rotation_wc.transpose() @ (world_points - translation_wc).T).T

        sort_idx = np.argsort(points_cam[:, 2])
        points_cam = points_cam[sort_idx]
        world_points = world_points[sort_idx]

        project_pc = np.zeros((height, width), dtype=np.uint8)
        selected_points_world = []
        selected_points_camera = []

        for point_world, point_cam in zip(world_points, points_cam):
            point_x, point_y, point_z = point_cam[:3]
            if point_z <= 0:
                continue
            uv = self.camera_model.project3dToPixel((point_x, point_y, point_z))
            pixel_x, pixel_y = int(uv[0]), int(uv[1])
            if pixel_x < 0 or pixel_x >= width or pixel_y < 0 or pixel_y >= height:
                continue
            if project_pc[pixel_y, pixel_x] == 255:
                continue

            project_pc[pixel_y, pixel_x] = 255
            if mask_resized[pixel_y, pixel_x] > 0:
                selected_points_world.append(point_world)
                selected_points_camera.append(point_cam)

        if len(selected_points_world) == 0:
            return {
                "selected_points_world": np.empty((0, 3)),
                "points_list": points_cam,
                "project_pc": project_pc,
                "direction_points": np.empty((0, 3)),
                "selected_points_camera": np.empty((0, 3)),
                "selected_source_points_camera": np.empty((0, 3)),
            }

        selected_points_world = np.array(selected_points_world)
        selected_points_camera = np.array(selected_points_camera)
        projected_points = pc2.create_cloud_xyz32(
            copy.deepcopy(points_msg_list[-1]).header, selected_points_world
        )
        direction_points, _ = self.convert_and_sample_points(projected_points)
        return {
            "selected_points_world": selected_points_world,
            "points_list": points_cam,
            "project_pc": project_pc,
            "direction_points": direction_points,
            "selected_points_camera": selected_points_camera,
            "selected_source_points_camera": selected_points_camera,
        }

    def convert_and_sample_points(self, projected_points):
        world_points = projected_points
        world_points = self.filter_points_below_height(world_points)
        points_generator = point_cloud2.read_points(
            world_points, field_names=("x", "y", "z"), skip_nans=True
        )
        points = np.array(list(points_generator))
        return points, world_points

    def create_world_cloud(self, points, stamp):
        header = std_msgs.msg.Header()
        header.stamp = stamp
        header.frame_id = "world"
        return pc2.create_cloud_xyz32(header, points)
