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
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import open3d as o3d


@dataclass
class PredictionRecord:
    """Prediction cache entry used to feed later mesh reconstruction steps."""

    pc: np.ndarray
    pc_upsampled: Optional[np.ndarray]
    partial: np.ndarray
    pc_partial_dense: np.ndarray
    cond: Dict[str, Any]
    debug_grid: np.ndarray


@dataclass
class OrchestratorState:
    """Mutable runtime state shared across ROS callbacks and inference timers."""

    initial_image_condition: Optional[Dict[str, np.ndarray]] = None
    image_conditions: List[Dict[str, np.ndarray]] = field(default_factory=list)
    input_pointcloud: o3d.geometry.PointCloud = field(
        default_factory=o3d.geometry.PointCloud
    )
    predictions: List[PredictionRecord] = field(default_factory=list)
    filtered_pointcloud_queue: Deque[np.ndarray] = field(
        default_factory=lambda: deque(maxlen=10)
    )
    condition_queue: Deque[Dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=10)
    )
    last_mesh_result: Optional[Dict[str, Any]] = None
    text_prompt: str = "a tower"
    stack_len: int = 1
    max_image_conditions: int = 30
