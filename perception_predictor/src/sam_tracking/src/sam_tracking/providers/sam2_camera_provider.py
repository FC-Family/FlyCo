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
import numpy as np
import sys
from pathlib import Path

import rospy

_THIS_FILE = Path(__file__).resolve()


def _candidate_package_roots():
    roots = []
    for parent in _THIS_FILE.parents:
        roots.append(parent)
        thirdparty = parent / "thirdparty"
        if thirdparty.exists():
            roots.append(thirdparty)
        share_thirdparty = parent / "share" / "sam_tracking" / "thirdparty"
        if share_thirdparty.exists():
            roots.append(share_thirdparty)
    return roots


def _extend_sys_path_for_thirdparty():
    for root in _candidate_package_roots():
        sam_tk_root = root / "sam_tk"
        if sam_tk_root.exists() and str(sam_tk_root) not in sys.path:
            sys.path.insert(0, str(sam_tk_root))


_extend_sys_path_for_thirdparty()

from sam_tk import tracker_factory
from .evf_text_prompt_initializer import EVFTextPromptInitializer


@dataclass(frozen=True)
class EVFTextInitConfig:
    enabled: bool = False
    evf_root: str = ""
    model_version: str = "YxZhang/evf-sam2"
    model_type: str = "sam2"
    precision: str = "fp16"
    device: str = "cuda"
    image_size: int = 224
    semantic_level: bool = False
    min_mask_area: int = 256
    max_init_points: int = 8


@dataclass(frozen=True)
class SAM2RuntimeConfig:
    samurai_mode: bool = True
    depth_score_weight: float = 0.7
    kf_score_weight: float = 0.15
    memory_bank_geo_score_threshold: float = 0.0


@dataclass(frozen=True)
class ProviderRuntimeStats:
    initialized: bool
    initialization_count: int
    reinitialization_count: int
    reset_count: int
    last_init_reason: str
    has_last_prompt: bool


class Sam2CameraProvider:
    def __init__(
        self,
        evf_text_init_config: EVFTextInitConfig = EVFTextInitConfig(),
        sam2_runtime_config: SAM2RuntimeConfig = SAM2RuntimeConfig(),
    ):
        self._initialization_count = 0
        self._reinitialization_count = 0
        self._reset_count = 0
        self._last_init_reason = ""
        self._last_prompt = None
        self._evf_text_init_config = evf_text_init_config
        self._sam2_runtime_config = sam2_runtime_config
        self._evf_text_initializer = None
        self._tracker = self._build_tracker()

    def _build_tracker(self):
        tracker_args = {
            "samurai_mode": self._sam2_runtime_config.samurai_mode,
            "depth_score_weight": self._sam2_runtime_config.depth_score_weight,
            "kf_score_weight": self._sam2_runtime_config.kf_score_weight,
            "memory_bank_geo_score_threshold": self._sam2_runtime_config.memory_bank_geo_score_threshold,
        }
        return tracker_factory(tracker_type="sam2camera", args=tracker_args)

    def _normalize_prompt(self, prompt):
        if prompt is None:
            raise ValueError("prompt must not be None")

        coords = np.asarray(prompt["coords"], dtype=np.int32)
        labels = np.asarray(prompt["labels"], dtype=np.int32).reshape(-1)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("prompt coords must have shape [N, 2]")
        if labels.shape[0] != coords.shape[0]:
            raise ValueError("prompt labels must have the same length as coords")

        bbox = prompt.get("bbox")
        if bbox is None and coords.shape[0] >= 2 and labels.shape[0] >= 2:
            if labels[0] == 2 and labels[1] == 3:
                x0, y0 = coords[0].tolist()
                x1, y1 = coords[1].tolist()
                bbox = np.array([[x0, y0], [x1, y1]], dtype=np.float32)
                coords = coords[2:]
                labels = labels[2:]
        return {
            "coords": coords,
            "labels": labels,
            "text": str(prompt.get("text", "")),
            "bbox": bbox,
        }

    def _build_evf_text_initializer(self):
        if not self._evf_text_init_config.enabled:
            return None
        if self._evf_text_initializer is None:
            self._evf_text_initializer = EVFTextPromptInitializer(
                evf_root=self._evf_text_init_config.evf_root,
                model_version=self._evf_text_init_config.model_version,
                model_type=self._evf_text_init_config.model_type,
                precision=self._evf_text_init_config.precision,
                device=self._evf_text_init_config.device,
                image_size=self._evf_text_init_config.image_size,
                semantic_level=self._evf_text_init_config.semantic_level,
                min_mask_area=self._evf_text_init_config.min_mask_area,
                max_init_points=self._evf_text_init_config.max_init_points,
            )
        return self._evf_text_initializer

    def _augment_prompt_from_text(self, frame, prompt):
        initializer = self._build_evf_text_initializer()
        if initializer is None:
            return prompt, None

        text = str(prompt.get("text", "")).strip()
        if not text:
            return prompt, None

        try:
            evf_result = initializer.initialize_from_text(frame_rgb=frame, prompt_text=text)
        except Exception as exc:
            rospy.logwarn("EVF-SAM text initialization failed: %s", exc)
            return prompt, None

        if evf_result is None:
            rospy.logwarn("EVF-SAM did not return a usable initialization mask")
            return prompt, None

        coords = prompt["coords"]
        labels = prompt["labels"]
        if coords.shape[0] == 0:
            merged_coords = evf_result.points
            merged_labels = evf_result.labels
        else:
            merged_coords = np.vstack([coords, evf_result.points])
            merged_labels = np.concatenate([labels, evf_result.labels])

        augmented_prompt = {
            "coords": merged_coords,
            "labels": merged_labels,
            "text": text,
            "bbox": evf_result.bbox,
        }
        return augmented_prompt, evf_result

    @property
    def initialized(self):
        return bool(getattr(self._tracker, "init_flag", False))

    @property
    def stats(self):
        return ProviderRuntimeStats(
            initialized=self.initialized,
            initialization_count=self._initialization_count,
            reinitialization_count=self._reinitialization_count,
            reset_count=self._reset_count,
            last_init_reason=self._last_init_reason,
            has_last_prompt=self._last_prompt is not None,
        )

    @property
    def last_prompt(self):
        return self._last_prompt

    def can_initialize_from_prompt(self, prompt):
        if prompt is None:
            return False
        coords = np.asarray(prompt.get("coords", []))
        text = str(prompt.get("text", "")).strip()
        return coords.shape[0] > 0 or (
            self._evf_text_init_config.enabled and bool(text)
        )

    def initialize(self, frame, prompt):
        prompt = self._normalize_prompt(prompt)
        prompt, evf_result = self._augment_prompt_from_text(frame=frame, prompt=prompt)
        bbox = prompt.get("bbox")
        if prompt["coords"].shape[0] == 0 and bbox is None:
            raise ValueError("prompt must provide points or a bbox")
        initial_mask = self._tracker.init_tracker(
            frame=frame,
            coords=prompt["coords"],
            labels=prompt["labels"],
            bbox=bbox,
        )
        self._last_prompt = prompt
        self._initialization_count += 1
        self._last_init_reason = "initialize"
        if evf_result is not None:
            rospy.loginfo(
                "Initialized SAM2 from EVF-SAM text mask: area=%s, bbox=%s, points=%s",
                evf_result.area,
                evf_result.bbox.tolist(),
                int(evf_result.points.shape[0]),
            )
        return initial_mask

    def reinitialize_from_prompt(self, frame, prompt, reason="reinitialize"):
        prompt = self._normalize_prompt(prompt)
        self.reset()
        initial_mask = self.initialize(frame=frame, prompt=prompt)
        self._reinitialization_count += 1
        self._last_init_reason = str(reason or "reinitialize")
        return initial_mask

    def track(self, frame, depth_mask=None):
        return self._tracker.track(
            frame=frame,
            depth_mask=depth_mask,
            update_memory=True,
            check_lost=False,
            iou_threshold=0.8,
            update_mask=True,
        )

    def reset(self):
        self._tracker = self._build_tracker()
        self._last_prompt = None
        self._reset_count += 1
        self._last_init_reason = "reset"
