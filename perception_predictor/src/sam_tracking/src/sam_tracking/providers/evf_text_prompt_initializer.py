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

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra


@dataclass(frozen=True)
class EVFTextInitResult:
    mask: np.ndarray
    bbox: np.ndarray
    points: np.ndarray
    labels: np.ndarray
    area: int


class EVFTextPromptInitializer:
    def __init__(
        self,
        evf_root: str,
        model_version: str,
        model_type: str,
        precision: str,
        device: str,
        image_size: int,
        semantic_level: bool,
        min_mask_area: int,
        max_init_points: int,
    ) -> None:
        self.evf_root = Path(evf_root).expanduser().resolve()
        self.model_version = str(model_version)
        self.model_type = str(model_type)
        self.precision = str(precision)
        self.device = self._resolve_device(device)
        self.image_size = int(image_size)
        self.semantic_level = bool(semantic_level)
        self.min_mask_area = int(min_mask_area)
        self.max_init_points = max(1, int(max_init_points))

        self._tokenizer = None
        self._model = None
        self._sam_preprocess = None
        self._beit3_preprocess = None

    def _ensure_ready(self) -> None:
        if self._model is not None:
            return

        if not self.evf_root.exists():
            raise FileNotFoundError(f"EVF-SAM root does not exist: {self.evf_root}")

        if str(self.evf_root) not in sys.path:
            sys.path.insert(0, str(self.evf_root))

        helpers_path = self.evf_root / "inference.py"
        spec = importlib.util.spec_from_file_location(
            "sam_tracking_evf_inference", helpers_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load EVF-SAM helpers from {helpers_path}")

        helpers_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers_module)
        self._sam_preprocess = helpers_module.sam_preprocess
        self._beit3_preprocess = helpers_module.beit3_preprocess

        from transformers import AutoTokenizer

        torch_dtype = torch.float32
        if self.device.type == "cpu":
            torch_dtype = torch.float32
        elif self.precision == "bf16":
            torch_dtype = torch.bfloat16
        elif self.precision == "fp16":
            torch_dtype = torch.float16

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_version,
            padding_side="right",
            use_fast=False,
        )

        self._prepare_hydra_for_evf_import()

        if self.model_type == "sam2":
            from model.evf_sam2 import EvfSam2Model

            self._ensure_evf_hydra_ready()

            self._model = EvfSam2Model.from_pretrained(
                self.model_version,
                low_cpu_mem_usage=True,
                torch_dtype=torch_dtype,
            )
            if hasattr(self._model.visual_model, "memory_encoder"):
                del self._model.visual_model.memory_encoder
            if hasattr(self._model.visual_model, "memory_attention"):
                del self._model.visual_model.memory_attention
        elif self.model_type == "ori":
            from model.evf_sam import EvfSamModel

            self._model = EvfSamModel.from_pretrained(
                self.model_version,
                low_cpu_mem_usage=True,
                torch_dtype=torch_dtype,
            )
        elif self.model_type == "effi":
            from model.evf_effisam import EvfEffiSamModel

            self._model = EvfEffiSamModel.from_pretrained(
                self.model_version,
                low_cpu_mem_usage=True,
                torch_dtype=torch_dtype,
            )
        else:
            raise ValueError(f"Unsupported EVF-SAM model_type={self.model_type!r}")

        self._model = self._model.to(self.device).eval()

    @staticmethod
    def _prepare_hydra_for_evf_import() -> None:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

    @staticmethod
    def _ensure_evf_hydra_ready() -> None:
        if GlobalHydra.instance().is_initialized():
            return
        initialize_config_module(
            config_module="model.segment_anything_2.sam2_configs",
            version_base="1.2",
        )

    def initialize_from_text(self, frame_rgb: np.ndarray, prompt_text: str):
        text = str(prompt_text or "").strip()
        if not text:
            return None

        self._ensure_ready()
        if self.semantic_level:
            text = "[semantic] " + text

        with torch.inference_mode():
            image_beit = self._beit3_preprocess(frame_rgb, self.image_size).to(
                dtype=self._model.dtype,
                device=self.device,
            )
            image_sam, resize_shape = self._sam_preprocess(
                frame_rgb,
                model_type=self.model_type,
            )
            image_sam = image_sam.to(dtype=self._model.dtype, device=self.device)
            input_ids = self._tokenizer(text, return_tensors="pt")["input_ids"].to(
                device=self.device
            )

            pred_mask = self._model.inference(
                image_sam.unsqueeze(0),
                image_beit.unsqueeze(0),
                input_ids,
                resize_list=[resize_shape],
                original_size_list=[frame_rgb.shape[:2]],
            )

        pred_mask = pred_mask.detach().float().cpu().numpy()[0] > 0
        binary_mask = pred_mask.astype(np.uint8)
        area = int(np.count_nonzero(binary_mask))
        if area < self.min_mask_area:
            return None

        bbox = self._mask_to_bbox(binary_mask)
        if bbox is None:
            return None

        points = self._sample_mask_points(binary_mask, self.max_init_points)
        labels = np.ones(points.shape[0], dtype=np.int32)
        return EVFTextInitResult(
            mask=binary_mask,
            bbox=bbox,
            points=points,
            labels=labels,
            area=area,
        )

    @staticmethod
    def _mask_to_bbox(mask: np.ndarray):
        coords = np.column_stack(np.where(mask > 0))
        if coords.size == 0:
            return None

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        if x1 <= x0 or y1 <= y0:
            return None

        return np.array([[x0, y0], [x1, y1]], dtype=np.float32)

    def _sample_mask_points(self, mask: np.ndarray, max_points: int):
        candidate_mask = self._build_inner_candidate_mask(mask)
        coords = np.column_stack(np.where(candidate_mask > 0))
        if coords.size == 0:
            coords = np.column_stack(np.where(mask > 0))
        if coords.size == 0:
            return np.zeros((0, 2), dtype=np.int32)

        if coords.shape[0] > 2048:
            step = max(1, coords.shape[0] // 2048)
            coords = coords[::step]

        centroid = coords.mean(axis=0)
        dists = np.sum((coords - centroid) ** 2, axis=1)
        selected = [int(np.argmin(dists))]

        while len(selected) < min(max_points, coords.shape[0]):
            selected_coords = coords[selected]
            pairwise = coords[:, None, :] - selected_coords[None, :, :]
            min_dists = np.sum(pairwise * pairwise, axis=2).min(axis=1)
            next_index = int(np.argmax(min_dists))
            if next_index in selected:
                break
            selected.append(next_index)

        sampled = coords[selected]
        points_xy = np.stack([sampled[:, 1], sampled[:, 0]], axis=1)
        return points_xy.astype(np.int32)

    @staticmethod
    def _build_inner_candidate_mask(mask: np.ndarray):
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        area = int(np.count_nonzero(mask_uint8))
        if area == 0:
            return mask_uint8

        erosion_steps = max(1, min(6, int(np.sqrt(area) / 18)))
        kernel = np.ones((5, 5), np.uint8)
        candidate = mask_uint8.copy()
        for _ in range(erosion_steps):
            eroded = cv2.erode(candidate, kernel, iterations=1)
            if np.count_nonzero(eroded) < max(8, area * 0.08):
                break
            candidate = eroded
        return candidate

    @staticmethod
    def _resolve_device(device: str):
        device = str(device or "cuda").strip().lower()
        if device.startswith("cuda") and torch.cuda.is_available():
            return torch.device(device)
        return torch.device("cpu")
