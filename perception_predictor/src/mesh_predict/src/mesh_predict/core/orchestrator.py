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

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from image_geometry import PinholeCameraModel
from mesh_predict.pmp.utils.misc import todevice
from sensor_msgs.msg import CameraInfo
from torch.cuda.amp import autocast

from mesh_predict.backends.pc_mesh_predict import PCMeshPredictBackend
from mesh_predict.core.camera import load_camera_info
from mesh_predict.core.densification import DEFAULT_SPACING_THRESHOLD, needs_densification
from mesh_predict.core.processing import (
    PCSurfaceSampler,
    accumulate_recent_pointclouds,
    build_debug_grid,
    calculate_target_to_source_recall,
    decode_compressed_mask_with_pose,
    filter_mesh_faces_by_height,
    keep_largest_mesh_component,
    prepare_mesh_surface,
    prepare_metric_batch_inputs,
)
from mesh_predict.core.state import OrchestratorState, PredictionRecord
from mesh_predict.runtime.resources import OfficialRuntimeSpec


class MeshPredictOrchestrator:
    """Runtime orchestrator that bridges cached ROS inputs to backend inference."""

    def __init__(
        self,
        runtime_spec: OfficialRuntimeSpec,
        min_height: float = 0.1,
        mesh_min_z_offset: float = 0.3,
        keep_largest_component: bool = True,
        nksr_scale: float = 0.3,
        recall_threshold: float = 0.99,
        recall_distance_threshold: float = 0.1,
        pose_alignment_mode: str = "legacy_ros",
        debug_dump_dir: str = "",
        debug_dump_once: bool = False,
        device: str = "cuda",
    ) -> None:
        self.runtime_spec = runtime_spec
        self.min_height = min_height
        self.mesh_min_z_offset = mesh_min_z_offset
        self.keep_largest_component = keep_largest_component
        self.nksr_scale = nksr_scale
        self.recall_threshold = recall_threshold
        self.recall_distance_threshold = recall_distance_threshold
        self.pose_alignment_mode = pose_alignment_mode
        self.debug_dump_dir = Path(debug_dump_dir).expanduser() if debug_dump_dir else None
        self.debug_dump_once = debug_dump_once
        self.device = device
        self._debug_dump_done = False
        self._debug_dump_index = 0
        self.enable_mesh_densification = False
        self.mesh_densification_spacing_threshold = DEFAULT_SPACING_THRESHOLD

        self.state = OrchestratorState()
        self.backend = PCMeshPredictBackend(runtime_spec=runtime_spec, device=device)
        self.camera_model = PinholeCameraModel()
        self.camera_info_received = False
        self.camera_info: Optional[CameraInfo] = None
        self.pc_sampler = PCSurfaceSampler()

        if self.debug_dump_dir is not None:
            self.debug_dump_dir.mkdir(parents=True, exist_ok=True)

    def _dumpable(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return {k: self._dumpable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._dumpable(v) for v in value]
        return value

    def _maybe_dump_pointcloud_prediction_debug(
        self,
        batch,
        cond,
        out,
        predicted_pointcloud,
        pure_predicted_pointcloud,
        filtered_input_world,
    ) -> Optional[Path]:
        if self.debug_dump_dir is None:
            return None
        if self.debug_dump_once and self._debug_dump_done:
            return None

        self._debug_dump_index += 1
        dump_path = self.debug_dump_dir / (
            f"pc_pred_dump_{self._debug_dump_index:03d}.pt"
        )
        payload = {
            "runtime": {
                "runtime_config_path": self.runtime_spec.runtime_config_path,
                "model_config_path": self.runtime_spec.model_config_path,
                "mesh_pred_checkpoint": self.runtime_spec.mesh_pred_checkpoint,
                "text_prompt": self.state.text_prompt,
                "pose_alignment_mode": self.pose_alignment_mode,
            },
            "batch": self._dumpable(batch),
            "cond": self._dumpable(cond),
            "model_out": self._dumpable(
                {
                    "pc": out.get("pc"),
                    "down": out.get("down"),
                    "pc_token": out.get("pc_token"),
                    "img_cond": out.get("img_cond"),
                    "pc_upsampled": out.get("pc_upsampled"),
                    "up_results": out.get("up_results"),
                }
            ),
            "world": self._dumpable(
                {
                    "predicted_pointcloud": predicted_pointcloud,
                    "pure_predicted_pointcloud": pure_predicted_pointcloud,
                    "filtered_input_world": filtered_input_world,
                }
            ),
        }
        torch.save(payload, dump_path)
        self._debug_dump_done = True
        return dump_path

    def set_text_prompt(self, text_prompt: str) -> None:
        self.state.text_prompt = text_prompt

    def append_image_pose_message(self, msg) -> None:
        image_condition = decode_compressed_mask_with_pose(msg, self.pose_alignment_mode)
        if self.state.initial_image_condition is None:
            self.state.initial_image_condition = image_condition
        self.state.image_conditions.append(image_condition)
        if len(self.state.image_conditions) > self.state.max_image_conditions:
            self.state.image_conditions = self.state.image_conditions[
                -self.state.max_image_conditions :
            ]

    def append_filtered_pointcloud(self, pointcloud_np: np.ndarray) -> None:
        self.state.filtered_pointcloud_queue.append(pointcloud_np)

    def update_camera_info(self, msg: CameraInfo) -> None:
        self.camera_info = msg
        self.camera_model.fromCameraInfo(msg)
        self.camera_info_received = True

    def load_static_camera_info(self, camera_param_file: str) -> None:
        self.camera_info = load_camera_info(camera_param_file)
        self.camera_model.fromCameraInfo(self.camera_info)
        self.camera_info_received = True

    def append_pointcloud_messages(self, point_messages) -> None:
        self.state.input_pointcloud = accumulate_recent_pointclouds(
            point_messages, self.state.input_pointcloud
        )

    def is_ready_for_pc_prediction(self) -> bool:
        return (
            len(self.state.image_conditions) >= 3
            and len(self.state.input_pointcloud.points) >= 512
            and self.camera_info_received
        )

    def prepare_metric_batch(self):
        return prepare_metric_batch_inputs(
            [self.state.initial_image_condition],
            self.camera_model,
            self.state.input_pointcloud,
            self.state.text_prompt,
            self.backend.device,
        )

    def maybe_densify_mesh_input(
        self, pointcloud_np: np.ndarray, pred_state: PredictionRecord
    ) -> np.ndarray:
        if not self.enable_mesh_densification:
            return pointcloud_np

        if not needs_densification(
            pointcloud_np,
            spacing_threshold=self.mesh_densification_spacing_threshold,
        ):
            return pointcloud_np

        if pred_state.pc_upsampled is None:
            return pointcloud_np

        upsampled_world = (
            pred_state.pc_upsampled * pred_state.cond["patial_L_max"]
            + pred_state.cond["patial_mean"]
        )
        partial_min_z = np.min(pred_state.partial, axis=0)[2]
        pure_upsampled_world = upsampled_world[
            pred_state.pc_upsampled[:, 2] >= partial_min_z
        ]
        partial_world = (
            pred_state.pc_partial_dense * pred_state.cond["patial_L_max"]
            + pred_state.cond["patial_mean"]
        )
        return np.concatenate([partial_world, pure_upsampled_world], axis=0)

    def run_pointcloud_prediction(self) -> Optional[Dict[str, np.ndarray]]:
        if not self.is_ready_for_pc_prediction():
            return None

        t_start = time.time()
        batch, cond = self.prepare_metric_batch()
        t_batch_end = time.time()

        with torch.no_grad():
            batch = todevice(batch, self.backend.device)
            t_device_end = time.time()
            t_pred_start = time.time()
            # pointnet2_ops in the official PC backend expects float32 inputs.
            # AMP here can downcast the pointcloud branch and crash the runtime
            # on real bag playback, so keep the official pred_pc path in fp32.
            with autocast(enabled=False):
                out = self.backend.pred_pc(batch)
            t_pred_end = time.time()

            pc_pred = out["pc"].detach().squeeze(0).cpu().numpy()
            up_results = out.get("up_results") or []
            pc_upsampled = out.get("pc_upsampled") if len(up_results) > 0 else None
            if pc_upsampled is not None:
                pc_upsampled = pc_upsampled.detach().squeeze(0).cpu().numpy()
            partial = batch["pc_patial"].detach().squeeze(0).cpu().numpy()
            pc_partial_dense = (
                cond["pc_patial_cond_dense"].detach().squeeze(0).cpu().numpy()
            )
            debug_grid = build_debug_grid(batch["rgb_cond"]).astype(np.uint8)
            t_decode_end = time.time()

        cond = dict(cond)
        cond["pc_patial_cond_dense"] = pc_partial_dense

        self.state.predictions.append(
            PredictionRecord(
                pc=pc_pred,
                pc_upsampled=pc_upsampled,
                partial=partial,
                pc_partial_dense=pc_partial_dense,
                cond=cond,
                debug_grid=debug_grid,
            )
        )
        self.state.condition_queue.append(cond)

        patial_min = np.min(partial, axis=0)[2]
        pure_pred_world = pc_pred[pc_pred[:, 2] >= patial_min]
        filtered_input_world = partial * cond["patial_L_max"] + cond["patial_mean"]
        pure_pred_world = pure_pred_world * cond["patial_L_max"] + cond["patial_mean"]
        pc_final_surface = np.concatenate(
            [filtered_input_world, pure_pred_world.copy()], axis=0
        )
        dump_path = self._maybe_dump_pointcloud_prediction_debug(
            batch=batch,
            cond=cond,
            out=out,
            predicted_pointcloud=pc_final_surface,
            pure_predicted_pointcloud=pure_pred_world,
            filtered_input_world=filtered_input_world,
        )
        t_end = time.time()

        result = {
            "predicted_pointcloud": pc_final_surface,
            "pure_predicted_pointcloud": pure_pred_world,
            "debug_image": debug_grid,
            "timings": {
                "total": t_end - t_start,
                "batch": t_batch_end - t_start,
                "to_device": t_device_end - t_batch_end,
                "infer": t_pred_end - t_pred_start,
                "decode": t_decode_end - t_pred_end,
                "post": t_end - t_decode_end,
                "input_points": len(self.state.input_pointcloud.points),
                "partial_points": len(filtered_input_world),
                "predicted_points": len(pure_pred_world),
            },
        }
        if dump_path is not None:
            result["debug_dump_path"] = str(dump_path)
        return result

    def build_mesh_input_from_prediction(self, cond):
        if not self.state.predictions:
            raise RuntimeError(
                "mesh_predict fallback requires a prior pointcloud prediction result"
            )

        pred_state = self.state.predictions[-1]
        pred_pc_world = pred_state.pc * cond["patial_L_max"] + cond["patial_mean"]
        partial_world = (
            pred_state.pc_partial_dense * cond["patial_L_max"] + cond["patial_mean"]
        )
        return np.concatenate([partial_world, pred_pc_world], axis=0)

    def run_mesh_prediction(self) -> Optional[Dict[str, np.ndarray]]:
        if len(self.state.predictions) < 1:
            return None

        t_start = time.time()
        pred_state = self.state.predictions[-1]
        cond = pred_state.cond
        self.state.condition_queue.clear()
        if len(self.state.filtered_pointcloud_queue) > 0:
            filtered_pc_np = self.state.filtered_pointcloud_queue.pop()
            self.state.filtered_pointcloud_queue.clear()
            used_fallback = False
        else:
            filtered_pc_np = pred_state.pc * cond["patial_L_max"] + cond["patial_mean"]
            used_fallback = True
        t_input_end = time.time()

        if len(self.state.predictions) > self.state.stack_len:
            self.state.predictions = self.state.predictions[-1 * self.state.stack_len :]

        filtered_pc_np = self.maybe_densify_mesh_input(filtered_pc_np, pred_state)
        mesh_inputs = prepare_mesh_surface(
            filtered_pc_np, cond, self.pc_sampler, nksr_scale=self.nksr_scale
        )
        t_surface_end = time.time()
        pc_partial = mesh_inputs["pc_partial"]
        pc_final_surface = mesh_inputs["pc_final_surface"]
        pc_final_surface_test = mesh_inputs["pc_for_mesh"]

        surface_recall = None
        if self.state.last_mesh_result is not None:
            surface_recall = calculate_target_to_source_recall(
                pc_partial,
                pc_final_surface,
                distance_threshold=self.recall_distance_threshold,
            )
            if surface_recall > self.recall_threshold:
                cached = dict(self.state.last_mesh_result)
                cached["partial_pointcloud"] = (
                    pc_partial * cond["patial_L_max"] + cond["patial_mean"]
                )
                cached["used_fallback"] = used_fallback
                cached["used_cached_mesh"] = True
                cached["surface_recall"] = surface_recall
                cached["timings"] = {
                    "total": time.time() - t_start,
                    "input": t_input_end - t_start,
                    "surface": t_surface_end - t_input_end,
                    "recall": time.time() - t_surface_end,
                    "pred_mesh": 0.0,
                    "post": 0.0,
                    "filtered_input_points": len(filtered_pc_np),
                    "partial_points": len(pc_partial),
                    "surface_points": len(pc_final_surface),
                    "mesh_vertices": len(cached.get("mesh_vertices", [])),
                    "mesh_faces": len(cached.get("mesh_faces", [])),
                }
                return cached
        t_recall_end = time.time()

        pc_final_th = (
            torch.from_numpy(pc_final_surface_test).to(self.backend.device).float()
        )
        mesh_out = self.backend.pred_mesh(pc_final_th)
        t_mesh_end = time.time()
        mesh = mesh_out["mesh"]
        mesh_v = (
            mesh.v.detach().cpu().numpy()
            * (1.0 / self.nksr_scale)
            * cond["patial_L_max"]
            + cond["patial_mean"]
        )
        mesh_f = mesh.f.cpu().numpy()
        mesh_min_z = (
            np.min(cond["pc_patial_cond_dense"], axis=0)[2]
            * cond["patial_L_max"]
            + cond["patial_mean"][2]
            + self.mesh_min_z_offset
        )
        mesh_v, mesh_f = filter_mesh_faces_by_height(
            mesh_v, mesh_f, min_z=mesh_min_z
        )
        if (
            self.keep_largest_component
            and len(mesh_v) > 0
            and mesh_v[:, 2].max() <= 5.0
        ):
            mesh_v, mesh_f = keep_largest_mesh_component(mesh_v, mesh_f)

        pc_partial = pc_partial * cond["patial_L_max"] + cond["patial_mean"]
        pc_final_surface = pc_final_surface * cond["patial_L_max"] + cond["patial_mean"]
        t_end = time.time()

        result = {
            "mesh_vertices": mesh_v,
            "mesh_faces": mesh_f,
            "partial_pointcloud": pc_partial,
            "debug_pointcloud": pc_final_surface,
            "used_fallback": used_fallback,
            "used_cached_mesh": False,
            "surface_recall": surface_recall,
            "timings": {
                "total": t_end - t_start,
                "input": t_input_end - t_start,
                "surface": t_surface_end - t_input_end,
                "recall": t_recall_end - t_surface_end,
                "pred_mesh": t_mesh_end - t_recall_end,
                "post": t_end - t_mesh_end,
                "filtered_input_points": len(filtered_pc_np),
                "partial_points": len(pc_partial),
                "surface_points": len(pc_final_surface),
                "mesh_vertices": len(mesh_v),
                "mesh_faces": len(mesh_f),
            },
        }
        self.state.last_mesh_result = dict(result)
        return result


def format_waiting_reason(orchestrator: MeshPredictOrchestrator) -> str:
    if len(orchestrator.state.image_conditions) < 3:
        return "Waiting for enough image pose conditions..."
    if len(orchestrator.state.input_pointcloud.points) < 512:
        return "Waiting for enough accumulated point cloud samples..."
    if not orchestrator.camera_info_received:
        return "Waiting for camera parameters..."
    return "Waiting for prediction prerequisites..."
