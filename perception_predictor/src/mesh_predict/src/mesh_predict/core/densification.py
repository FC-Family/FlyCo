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
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from scipy.spatial import cKDTree


DEFAULT_SPACING_THRESHOLD = 0.05


@dataclass
class DensificationResult:
    points: Any
    spacing: float
    densified: bool
    reason: str


def _as_numpy(points: Any) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    points_np = np.asarray(points)
    if points_np.ndim == 3:
        if points_np.shape[0] != 1:
            raise ValueError(
                f"Expected a single point cloud batch, got shape {points_np.shape}"
            )
        points_np = points_np[0]
    if points_np.ndim != 2 or points_np.shape[-1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points_np.shape}")
    return points_np


def estimate_min_spacing(points: Any) -> float:
    """Return the nearest-neighbor spacing of a point cloud."""

    points_np = _as_numpy(points)
    if len(points_np) < 2:
        return float("inf")
    distances, _ = cKDTree(points_np).query(points_np, k=2)
    return float(np.min(distances[:, 1]))


def needs_densification(
    points: Any, spacing_threshold: float = DEFAULT_SPACING_THRESHOLD
) -> bool:
    """Return True when the point cloud is sparser than the target spacing."""

    return estimate_min_spacing(points) > spacing_threshold


def first_upsample_result(upsampler_output: Any) -> Any:
    """Extract the first upsampling level from common upsampler return formats."""

    if isinstance(upsampler_output, (list, tuple)):
        if not upsampler_output:
            return None
        return upsampler_output[0]
    if isinstance(upsampler_output, dict):
        for key in ("pc_upsampled", "points", "pc", "dense_points", "output"):
            if key in upsampler_output:
                return upsampler_output[key]
        return None
    return upsampler_output


def jitter_points(
    points: torch.Tensor,
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply a small Gaussian perturbation to point coordinates."""

    if sigma <= 0:
        return points
    noise = torch.randn(
        points.shape,
        dtype=points.dtype,
        device=points.device,
        generator=generator,
    )
    return points + noise * sigma


def repeated_first_level_upsampling(
    upsampler: Callable[[Dict[str, Any]], Any],
    upsampling_input: Dict[str, Any],
    repeat_levels: int = 1,
    jitter_sigma: float = 0.0,
) -> list:
    """Build linear density levels from one 2x upsampler and the base points.

    Every upsampler call is anchored on the original predicted points. Later
    calls jitter that same base cloud, instead of feeding the previous upsampled
    output back into the network. With a 2x upsampler, the returned levels are
    approximately 2N, 3N, 4N, ... points.
    """

    if repeat_levels <= 0:
        return []

    base_points = upsampling_input["points"]
    upsampled_chunks = []
    results = []
    for level in range(repeat_levels):
        current_input = dict(upsampling_input)
        current_input["points"] = (
            base_points
            if level == 0
            else jitter_points(base_points, sigma=jitter_sigma)
        )
        first_level = first_upsample_result(upsampler(current_input))
        if first_level is None:
            break
        upsampled_chunks.append(first_level)

        target_multiplier = level + 2
        if target_multiplier % 2 == 0:
            needed_chunks = target_multiplier // 2
            result = torch.cat(upsampled_chunks[:needed_chunks], dim=1)
        else:
            needed_chunks = (target_multiplier - 1) // 2
            result = torch.cat(
                [base_points, *upsampled_chunks[:needed_chunks]],
                dim=1,
            )
        results.append(result)

    return results


def densify_once_if_sparse(
    points: Any,
    upsampler: Optional[Callable[[Dict[str, Any]], Any]],
    upsampling_input: Optional[Dict[str, Any]] = None,
    spacing_threshold: float = DEFAULT_SPACING_THRESHOLD,
) -> DensificationResult:
    """Run at most one upsampling level when the input is too sparse.

    This is intentionally a one-step interface rather than a cascade. The runtime
    does not call it by default; callers that own an upsampler can opt in later.
    """

    spacing = estimate_min_spacing(points)
    if spacing <= spacing_threshold:
        return DensificationResult(
            points=points,
            spacing=spacing,
            densified=False,
            reason="spacing_within_threshold",
        )
    if upsampler is None:
        return DensificationResult(
            points=points,
            spacing=spacing,
            densified=False,
            reason="upsampler_unavailable",
        )

    payload = dict(upsampling_input or {})
    payload.setdefault("points", points)
    first_level = first_upsample_result(upsampler(payload))
    if first_level is None:
        return DensificationResult(
            points=points,
            spacing=spacing,
            densified=False,
            reason="upsampler_returned_no_points",
        )

    return DensificationResult(
        points=first_level,
        spacing=spacing,
        densified=True,
        reason="first_upsample_level",
    )
