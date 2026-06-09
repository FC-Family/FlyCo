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

import math

import cv2
import numpy as np
import open3d as o3d
import scipy.ndimage as ndi
import torch
import tf.transformations as tft
from einops import rearrange
from PIL import Image
from pointnet2_ops import pointnet2_utils
from scipy.spatial import ConvexHull, cKDTree
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CompressedImage
from torchvision.utils import make_grid


LEGACY_ROS_ROTATION = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

POSE_ALIGNMENT_MODES = {
    "legacy_ros": LEGACY_ROS_ROTATION,
    "identity": np.eye(3),
}


def fps(data: torch.Tensor, number: int) -> torch.Tensor:
    """Furthest-point sample a batched point cloud using the packaged runtime op."""

    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = (
        pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx)
        .transpose(1, 2)
        .contiguous()
    )
    return fps_data


class PCSurfaceSampler:
    """Sample visible surface support points for mesh reconstruction."""

    def __init__(self, args=None):
        if args is None:
            self.sample_num_ = 1024
        else:
            self.sample_num_ = args.sample_num

    def _hpr(self, point_cloud, viewpoint, radius):
        translated_points = point_cloud - viewpoint
        distances = np.linalg.norm(translated_points, axis=1)
        transformed_points = translated_points + 2 * (radius - distances).reshape(
            -1, 1
        ) * (translated_points / distances.reshape(-1, 1))
        hull = ConvexHull(transformed_points)
        return hull.vertices

    def _generate_sphere(self, point_cloud):
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(point_cloud)
        )
        center = bbox.get_center()
        radius = bbox.get_max_bound() - center
        return center, radius

    def sample_surface(self, point_cloud, h_num=9, v_num=9):
        center, radius = self._generate_sphere(point_cloud)
        viewpoint_radius = np.linalg.norm(radius)
        visible_points = np.zeros((0, 3))
        viewpoints = []

        for h in range(-1 - h_num, h_num + 1):
            pitch = np.pi * h / h_num
            for i in range(v_num):
                angle = 2 * np.pi * i / v_num
                viewpoint = np.array(
                    [
                        center[0] + viewpoint_radius * np.cos(pitch) * np.cos(angle),
                        center[1] + viewpoint_radius * np.cos(pitch) * np.sin(angle),
                        center[2] + viewpoint_radius * np.sin(pitch),
                    ]
                )
                viewpoints.append(viewpoint)

                radius_hpr = 3 * np.max(np.linalg.norm(point_cloud - viewpoint, axis=1))
                point_indices = self._hpr(point_cloud, viewpoint, radius_hpr)
                visible_points = np.vstack((visible_points, point_cloud[point_indices]))

        for viewpoint in (
            np.array([center[0], center[1], center[2] + radius[2]]),
            np.array([center[0], center[1], center[2] - radius[2]]),
        ):
            radius_hpr = 5 * np.max(np.linalg.norm(point_cloud - viewpoint, axis=1))
            point_indices = self._hpr(point_cloud, viewpoint, radius_hpr)
            visible_points = np.vstack((visible_points, point_cloud[point_indices]))

        return visible_points, viewpoints


def reshape_pc(pc):
    bbox = np.array(
        [
            [pc[:, 0].min(), pc[:, 0].max()],
            [pc[:, 1].min(), pc[:, 1].max()],
            [pc[:, 2].min(), pc[:, 2].max()],
        ]
    )
    center = (
        np.array(
            [
                bbox[0][0] + bbox[0][1],
                bbox[1][0] + bbox[1][1],
                bbox[2][0] + bbox[2][1],
            ]
        )
        / 2.0
    )
    radius = math.sqrt(
        math.pow((bbox[0][1] - bbox[0][0]) / 2, 2)
        + math.pow((bbox[1][1] - bbox[1][0]) / 2, 2)
        + math.pow((bbox[2][1] - bbox[2][0]) / 2, 2)
    )
    pc = (pc - center) / radius
    return pc, center, radius


def filter_outlier(pc_np, nb_neighbors, std_ratio):
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pc_np)
    filtered_cloud, _ = pointcloud.remove_radius_outlier(
        nb_points=nb_neighbors, radius=std_ratio
    )
    return np.asarray(filtered_cloud.points)


def calculate_target_to_source_recall(target_points, source_points, distance_threshold):
    """Return the fraction of source points covered by target points."""

    if len(source_points) == 0:
        return 1.0
    if len(target_points) == 0:
        return 0.0

    tree = cKDTree(target_points)
    distances, _ = tree.query(source_points, k=1)
    return float(np.mean(distances <= distance_threshold))


def resample_pointcloud_exact(pc_np, target_points, device):
    """Resample a point cloud to the exact size expected by the PC backend."""

    if len(pc_np) == 0:
        raise ValueError("cannot resample an empty point cloud")
    if len(pc_np) == target_points:
        return pc_np.copy()
    if len(pc_np) > target_points:
        device_str = str(device)
        if device_str.startswith("cuda"):
            pc_th = torch.from_numpy(pc_np).float().unsqueeze(0).to(device).contiguous()
            return fps(pc_th, target_points).squeeze(0).detach().cpu().numpy()

        stride = max(1, len(pc_np) // target_points)
        sampled = pc_np[::stride][:target_points].copy()
        if len(sampled) < target_points:
            pad_count = target_points - len(sampled)
            sampled = np.concatenate([sampled, pc_np[:pad_count].copy()], axis=0)
        return sampled

    full_repeats = target_points // len(pc_np)
    remainder = target_points % len(pc_np)
    parts = [pc_np.copy() for _ in range(full_repeats)]
    if remainder > 0:
        stride = max(1, len(pc_np) // remainder)
        remainder_np = pc_np[::stride][:remainder].copy()
        if len(remainder_np) < remainder:
            pad_count = remainder - len(remainder_np)
            remainder_np = np.concatenate(
                [remainder_np, pc_np[:pad_count].copy()],
                axis=0,
            )
        parts.append(remainder_np)
    return np.concatenate(parts, axis=0)


def move_pc(pc_pred, pc_partial):
    pc_partial = filter_outlier(pc_partial, 10, 0.2)
    pc_partial_bbox = np.array(
        [
            [pc_partial[:, 0].min(), pc_partial[:, 0].max()],
            [pc_partial[:, 1].min(), pc_partial[:, 1].max()],
            [pc_partial[:, 2].min(), pc_partial[:, 2].max()],
        ]
    )
    pc_partial_height = pc_partial_bbox[2][1] - pc_partial_bbox[2][0]
    pc_pred_box = np.array(
        [
            [pc_pred[:, 0].min(), pc_pred[:, 0].max()],
            [pc_pred[:, 1].min(), pc_pred[:, 1].max()],
            [pc_pred[:, 2].min(), pc_pred[:, 2].max()],
        ]
    )
    pc_pred_height = pc_pred_box[2][1] - pc_pred_box[2][0]

    ratio = pc_partial_height / pc_pred_height
    pc_pred[:, 2] = pc_pred[:, 2] * ratio

    pc_pred_box = np.array(
        [
            [pc_pred[:, 0].min(), pc_pred[:, 0].max()],
            [pc_pred[:, 1].min(), pc_pred[:, 1].max()],
            [pc_pred[:, 2].min(), pc_pred[:, 2].max()],
        ]
    )
    pc_pred_mean = pc_pred_box.mean(axis=1)
    pc_partial_mean = 0.75 * pc_partial_bbox.mean(axis=1) + 0.25 * pc_pred.mean(axis=0)
    bias = np.array(
        [
            pc_partial_mean[0] - pc_pred_mean[0],
            pc_partial_mean[1] - pc_pred_mean[1],
            pc_partial_bbox[2][0] - pc_pred_box[2][0],
        ]
    )
    pc_pred += bias
    return pc_pred


def process_pointcloud(pc_np):
    centered = pc_np.copy()
    patial_mean = (
        np.array(
            [
                pc_np[:, 0].max() + pc_np[:, 0].min(),
                pc_np[:, 1].max() + pc_np[:, 1].min(),
                pc_np[:, 2].max() + pc_np[:, 2].min(),
            ]
        )
        / 2.0
    )
    centered = pc_np - patial_mean
    patial_l_max = np.max(np.sqrt(np.sum(np.abs(centered**2), axis=-1)))
    centered = centered / patial_l_max
    return centered, patial_mean, patial_l_max


def crop_with_alpha_ratio(image_np, cond_width, cond_height):
    alpha = image_np[..., 3]
    non_zero_coords = np.argwhere(alpha > 0)

    if non_zero_coords.size == 0:
        resized_image = np.zeros((cond_height, cond_width, 4), dtype=np.float32)
        return resized_image, {}

    (y_min, x_min), (y_max, x_max) = non_zero_coords.min(axis=0), non_zero_coords.max(
        axis=0
    )
    original_width = x_max - x_min + 1
    original_height = y_max - y_min + 1
    original_aspect = original_width / original_height

    target_aspect = cond_width / cond_height
    if original_aspect > target_aspect:
        crop_width = max(original_width, 128)
        crop_height = int(crop_width / target_aspect)
    else:
        crop_height = max(original_height, 128)
        crop_width = int(crop_height * target_aspect)

    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    x_min = max(0, x_center - crop_width // 2)
    x_max = min(image_np.shape[1], x_center + crop_width // 2)
    y_min = max(0, y_center - crop_height // 2)
    y_max = min(image_np.shape[0], y_center + crop_height // 2)

    crop_width = x_max - x_min
    crop_height = y_max - y_min
    if crop_width < 64 or crop_height < 64:
        raise ValueError("Effective masked image area is too small for runtime crop")

    scale = min(cond_width / crop_width, cond_height / crop_height)
    scaled_width = int(crop_width * scale)
    scaled_height = int(crop_height * scale)

    cropped_image = image_np[y_min:y_max, x_min:x_max]
    resized_image = (
        np.asarray(Image.fromarray(cropped_image).resize((scaled_width, scaled_height)))
        / 255.0
    )

    pad_left = (cond_width - scaled_width) // 2
    pad_top = (cond_height - scaled_height) // 2
    padded_image = np.zeros((cond_height, cond_width, 4), dtype=np.float32)
    padded_image[
        pad_top : pad_top + scaled_height, pad_left : pad_left + scaled_width
    ] = resized_image

    crop_info = {
        "x_min": x_min,
        "y_min": y_min,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "resize_width": scaled_width,
        "resize_height": scaled_height,
    }
    return padded_image, crop_info


def get_crop_intrinsics(intrinsic: torch.Tensor, crop_info: dict) -> torch.Tensor:
    x_min = crop_info["x_min"]
    y_min = crop_info["y_min"]
    crop_width = crop_info["crop_width"]
    crop_height = crop_info["crop_height"]
    resize_width = crop_info["resize_width"]
    resize_height = crop_info["resize_height"]

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    new_cx = cx - x_min
    new_cy = cy - y_min

    scale_x = resize_width / crop_width
    scale_y = resize_height / crop_height

    new_fx = fx * scale_x
    new_fy = fy * scale_y
    new_cx = new_cx * scale_x
    new_cy = new_cy * scale_y

    return torch.tensor(
        [[new_fx, 0, new_cx], [0, new_fy, new_cy], [0, 0, 1]],
        dtype=intrinsic.dtype,
        device=intrinsic.device,
    )


def build_masked_rgb_tensor(image_np: np.ndarray) -> torch.Tensor:
    img_th = torch.from_numpy(image_np).float()
    mask_cond = (img_th[:, :, :3] > 0.0001).any(dim=-1, keepdim=True).float()

    mask_np = mask_cond.squeeze(-1).numpy()
    filled_mask = ndi.binary_fill_holes(mask_np).astype(mask_np.dtype)
    labeled_mask, num_features = ndi.label(filled_mask)
    sizes = ndi.sum(filled_mask, labeled_mask, range(num_features + 1))
    size_threshold = 100
    mask_cleaned = sum(
        (labeled_mask == idx) for idx, size in enumerate(sizes) if size > size_threshold
    )
    mask_cond = torch.from_numpy(mask_cleaned).unsqueeze(-1).float()

    return img_th[:, :, :3] * mask_cond + torch.as_tensor([1.0, 1.0, 1.0])[
        None, None, :
    ] * (1 - mask_cond)


def prepare_data_metric(
    images,
    pc_patial,
    cw2s,
    intrinsic,
    image_size,
    patial_mean,
    patial_l_max,
    device,
):
    height, width = image_size

    intrinsic_th = torch.from_numpy(intrinsic).float()
    pc_patial_cond_np = (pc_patial - patial_mean) / patial_l_max
    pc_patial_cond_th = (
        torch.from_numpy(pc_patial_cond_np).float().unsqueeze(0).to(device).contiguous()
    )
    pc_patial_cond = fps(pc_patial_cond_th, 2048)
    pc_patial_cond_dense = fps(pc_patial_cond_th, 2048)

    imgs_th_list = []
    intrinsic_list = []
    for img in images:
        resized_image, crop_info = crop_with_alpha_ratio(img, 252, 252)
        rgb_cond = build_masked_rgb_tensor(resized_image)
        imgs_th_list.append(rgb_cond.unsqueeze(0))

        intrinsic_cond = get_crop_intrinsics(intrinsic_th.clone(), crop_info)
        intrinsic_normed_cond = intrinsic_cond.clone()
        intrinsic_normed_cond[..., 0, 2] /= width
        intrinsic_normed_cond[..., 1, 2] /= height
        intrinsic_normed_cond[..., 0, 0] /= width
        intrinsic_normed_cond[..., 1, 1] /= height
        intrinsic_list.append(intrinsic_normed_cond.unsqueeze(0))

    c2w_th_list = []
    for c2w in cw2s:
        c2w_copy = c2w.copy()
        c2w_copy[:3, 3] = (c2w_copy[:3, 3] - patial_mean) / patial_l_max
        c2w_th_list.append(torch.from_numpy(c2w_copy).float().unsqueeze(0))

    batch = {
        "c2w_cond": torch.cat(c2w_th_list, dim=0).unsqueeze(0),
        "rgb_cond": torch.cat(imgs_th_list, dim=0).unsqueeze(0),
        "pc_patial": pc_patial_cond,
        "intrinsic_normed_cond": torch.cat(intrinsic_list, dim=0).unsqueeze(0),
        "text_view_cond": torch.cat(imgs_th_list, dim=0).unsqueeze(0),
    }
    cond = {
        "pc_patial": pc_patial,
        "patial_mean": patial_mean,
        "patial_L_max": patial_l_max,
        "pc_patial_cond_dense": pc_patial_cond_dense,
    }
    return batch, cond


def prepare_data_metric_batch(images, pc_patial, cw2s, intrinsic, image_size, device):
    _, patial_mean, patial_l_max = process_pointcloud(pc_patial)
    return prepare_data_metric(
        images,
        pc_patial,
        cw2s,
        intrinsic,
        image_size,
        patial_mean,
        patial_l_max,
        device=device,
    )


def decompress_image(compressed_image_msg: CompressedImage):
    np_arr = np.frombuffer(compressed_image_msg.data, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if cv_image is None:
        raise ValueError("Failed to decode compressed image!")

    alpha_channel = np.where(cv_image.max(axis=2) > 20, 255, 0).astype(np.uint8)
    cv_image_rgba = np.dstack((cv_image, alpha_channel))
    return cv2.cvtColor(cv_image_rgba, cv2.COLOR_BGRA2RGBA)


def decompress_transform_to_matrix(pose_data):
    translation = pose_data[:3]
    rotation = pose_data[3:]
    transform = tft.translation_matrix(translation)
    rotation_matrix = tft.quaternion_matrix(rotation)
    return np.dot(transform, rotation_matrix)


def apply_pose_alignment(c2w: np.ndarray, alignment_mode: str) -> np.ndarray:
    try:
        alignment = POSE_ALIGNMENT_MODES[alignment_mode]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported pose alignment mode '{alignment_mode}'. "
            f"Expected one of: {sorted(POSE_ALIGNMENT_MODES)}"
        ) from exc

    aligned = c2w.copy()
    aligned[:3, :3] = aligned[:3, :3] @ alignment
    return aligned


def decode_compressed_mask_with_pose(msg, pose_alignment_mode: str = "legacy_ros"):
    img = decompress_image(msg.mask)
    c2w = decompress_transform_to_matrix(msg.pose)
    c2w = apply_pose_alignment(c2w, pose_alignment_mode)
    return {"img": img, "c2w": c2w}


def accumulate_recent_pointclouds(point_messages, input_pc_all, voxel_size=0.6):
    for msg in point_messages:
        pc = point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )
        points_array = np.array([point[:3] for point in pc])
        points_array = points_array[points_array[:, 2] < 100]

        temp = o3d.geometry.PointCloud()
        temp.points = o3d.utility.Vector3dVector(points_array)
        input_pc_all += temp

    return input_pc_all.voxel_down_sample(voxel_size=voxel_size)


def select_key_frames(image_pose_pairs, keyframe_num=1):
    if not image_pose_pairs:
        return [], []

    frame_indexes = [0][:keyframe_num]
    imgs = [image_pose_pairs[i]["img"].copy() for i in frame_indexes]
    c2w = [image_pose_pairs[i]["c2w"].copy() for i in frame_indexes]
    return imgs, c2w


def prepare_metric_batch_inputs(
    image_pose_pairs,
    camera_model,
    input_pc_all,
    text_prompt,
    device,
):
    imgs, c2w = select_key_frames(image_pose_pairs)
    intrinsic = np.array(camera_model.intrinsicMatrix()).reshape(3, 3).copy()
    image_size = [imgs[0].shape[1] * 2, imgs[0].shape[0] * 2]

    pc_input = np.asarray(input_pc_all.points).copy()
    pc_input = filter_outlier(pc_input, 10, 8.2)
    batch, cond = prepare_data_metric_batch(
        imgs,
        pc_input,
        c2w,
        intrinsic,
        image_size,
        device=device,
    )
    batch.update({"text_promt": text_prompt, "text_cond": text_prompt})
    return batch, cond


def build_debug_grid(rgb_cond):
    img_tensor = rgb_cond.squeeze(0)
    img_tensor = rearrange(img_tensor, "B H W C -> B C H W")
    grid = make_grid(img_tensor, nrow=2)
    return grid.cpu().numpy().transpose(1, 2, 0) * 255


def prepare_mesh_surface(filtered_pc_np, cond, pc_sampler, nksr_scale=0.3):
    pc_partial = cond["pc_patial_cond_dense"]
    pc_partial_sparse = pc_partial.copy()

    pred_pc = (filtered_pc_np - cond["patial_mean"]) / cond["patial_L_max"]

    pc_partial = filter_outlier(pc_partial, 10, 0.3)
    pred_stack_points_np = np.asarray(pred_pc)

    patial_min = np.min(pc_partial_sparse, axis=0)[2]
    mask = pred_stack_points_np[:, 2] >= patial_min
    selected_points = pred_stack_points_np[mask]

    mask2 = (pred_stack_points_np[:, 2] >= patial_min) & (
        pred_stack_points_np[:, 2] <= patial_min + 1.0 / cond["patial_L_max"]
    )
    selected_points2 = pred_stack_points_np[mask2]

    projected_points = np.zeros_like(selected_points)
    projected_points[:, :2] = selected_points[:, :2]
    projected_points[:, 2] = patial_min - 1.5 / cond["patial_L_max"]

    projected_points2 = np.zeros_like(selected_points2)
    projected_points2[:, :2] = selected_points2[:, :2]
    projected_points2[:, 2] = patial_min - 0.75 / cond["patial_L_max"]

    pred_stack_points_np = np.concatenate(
        [pred_stack_points_np, projected_points, projected_points2], axis=0
    )
    pred_stack_points_np = filter_outlier(pred_stack_points_np, 10, 0.2)

    pred_stack_points_np_o3d = o3d.geometry.PointCloud()
    pred_stack_points_np_o3d.points = o3d.utility.Vector3dVector(pred_stack_points_np)
    pred_stack_points_np = np.asarray(pred_stack_points_np_o3d.points)

    pc_final_surface, _ = pc_sampler.sample_surface(pred_stack_points_np)

    pc_final_surface_o3d = o3d.geometry.PointCloud()
    pc_final_surface_o3d.points = o3d.utility.Vector3dVector(pc_final_surface)
    pc_final_surface_o3d = pc_final_surface_o3d.voxel_down_sample(voxel_size=0.05)
    pc_final_surface = np.asarray(pc_final_surface_o3d.points)

    return {
        "pc_partial": pc_partial,
        "pc_final_surface": pc_final_surface,
        "pc_for_mesh": pc_final_surface * nksr_scale,
    }


def filter_mesh_faces_by_height(mesh_v, mesh_f, min_z=0.7):
    above_threshold = mesh_v[:, 2] > min_z
    indices_to_keep = np.where(above_threshold)[0]

    triangles_to_keep = []
    for triangle_index, triangle in enumerate(mesh_f):
        if all(vertex_index in indices_to_keep for vertex_index in triangle):
            triangles_to_keep.append(triangle_index)

    mesh_f_filtered = mesh_f[triangles_to_keep]
    if len(mesh_f_filtered) == 0:
        return (
            np.empty((0, 3), dtype=mesh_v.dtype),
            np.empty((0, 3), dtype=mesh_f.dtype),
        )

    unique_vertex_indices = np.unique(mesh_f_filtered)
    vertex_index_mapping = {
        old_index: new_index
        for new_index, old_index in enumerate(unique_vertex_indices)
    }

    mesh_f_remapped = np.array(
        [
            [vertex_index_mapping[old_index] for old_index in triangle]
            for triangle in mesh_f_filtered
        ]
    )
    mesh_v_filtered = mesh_v[unique_vertex_indices]
    return mesh_v_filtered, mesh_f_remapped


def keep_largest_mesh_component(mesh_v, mesh_f):
    """Keep the largest connected triangle component and drop floater islands."""

    if len(mesh_f) == 0 or len(mesh_v) == 0:
        return mesh_v, mesh_f

    vertex_to_faces = {}
    for face_idx, face in enumerate(mesh_f):
        for vertex_idx in face:
            vertex_to_faces.setdefault(int(vertex_idx), []).append(face_idx)

    parent = list(range(len(mesh_f)))
    rank = [0] * len(mesh_f)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if rank[root_left] < rank[root_right]:
            parent[root_left] = root_right
        elif rank[root_left] > rank[root_right]:
            parent[root_right] = root_left
        else:
            parent[root_right] = root_left
            rank[root_left] += 1

    for face_ids in vertex_to_faces.values():
        if len(face_ids) < 2:
            continue
        first_face = face_ids[0]
        for face_idx in face_ids[1:]:
            union(first_face, face_idx)

    components = {}
    for face_idx in range(len(mesh_f)):
        components.setdefault(find(face_idx), []).append(face_idx)

    if len(components) <= 1:
        return mesh_v, mesh_f

    largest_face_indices = max(components.values(), key=len)
    kept_faces = mesh_f[largest_face_indices]
    used_vertex_indices = np.unique(kept_faces)
    vertex_index_mapping = {
        int(old_index): new_index
        for new_index, old_index in enumerate(used_vertex_indices.tolist())
    }
    remapped_faces = np.array(
        [
            [vertex_index_mapping[int(old_index)] for old_index in face]
            for face in kept_faces
        ],
        dtype=np.int64,
    )
    filtered_vertices = mesh_v[used_vertex_indices]
    return filtered_vertices, remapped_faces
