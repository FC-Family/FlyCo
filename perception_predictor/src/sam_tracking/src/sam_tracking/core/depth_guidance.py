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

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthReinitDecision:
    should_reinitialize: bool
    reason: str = ""
    prompt: dict = None
    consistency: float = None
    depth_area: int = 0
    bbox: tuple = None


@dataclass(frozen=True)
class DepthGuidanceSnapshot:
    enabled: bool
    last_consistency: float = None
    last_depth_area: int = 0
    last_depth_bbox: tuple = None
    last_reinit_reason: str = ""
    reinit_count: int = 0
    empty_reinit_count: int = 0
    inconsistency_reinit_count: int = 0
    cooldown_remaining_frames: int = 0


class DepthGuidancePolicy:
    def __init__(
        self,
        enabled=False,
        consistency_threshold=0.2,
        empty_mask_streak_threshold=3,
        reinit_cooldown_frames=8,
        min_mask_area=256,
        bbox_padding=8,
        max_reinit_bbox_area_ratio=1.0,
        max_reinit_bbox_width_ratio=1.0,
        max_reinit_bbox_height_ratio=1.0,
        reinit_on_inconsistency=True,
        reinit_on_empty_streak=True,
    ):
        self.enabled = bool(enabled)
        self.consistency_threshold = float(consistency_threshold)
        self.empty_mask_streak_threshold = int(empty_mask_streak_threshold)
        self.reinit_cooldown_frames = int(reinit_cooldown_frames)
        self.min_mask_area = int(min_mask_area)
        self.bbox_padding = int(bbox_padding)
        self.max_reinit_bbox_area_ratio = float(max_reinit_bbox_area_ratio)
        self.max_reinit_bbox_width_ratio = float(max_reinit_bbox_width_ratio)
        self.max_reinit_bbox_height_ratio = float(max_reinit_bbox_height_ratio)
        self.reinit_on_inconsistency = bool(reinit_on_inconsistency)
        self.reinit_on_empty_streak = bool(reinit_on_empty_streak)
        self.last_depth_mask = None
        self.last_depth_prompt_points = []
        self.last_consistency = None
        self.last_depth_bbox = None
        self.last_depth_area = 0
        self.last_reinit_reason = ""
        self._frame_index = 0
        self._last_reinit_frame = -(10**9)
        self._last_track_area = 0
        self._reinit_count = 0
        self._empty_reinit_count = 0
        self._inconsistency_reinit_count = 0

    def reset(self):
        self.last_depth_mask = None
        self.last_depth_prompt_points = []
        self.last_consistency = None
        self.last_depth_bbox = None
        self.last_depth_area = 0
        self.last_reinit_reason = ""
        self._frame_index = 0
        self._last_reinit_frame = -(10**9)
        self._last_track_area = 0
        self._reinit_count = 0
        self._empty_reinit_count = 0
        self._inconsistency_reinit_count = 0

    def snapshot(self):
        return DepthGuidanceSnapshot(
            enabled=self.enabled,
            last_consistency=self.last_consistency,
            last_depth_area=self.last_depth_area,
            last_depth_bbox=self.last_depth_bbox,
            last_reinit_reason=self.last_reinit_reason,
            reinit_count=self._reinit_count,
            empty_reinit_count=self._empty_reinit_count,
            inconsistency_reinit_count=self._inconsistency_reinit_count,
            cooldown_remaining_frames=self._cooldown_remaining_frames(),
        )

    def preprocess(self, depth_mask, target_shape, positive_prompt_points=None):
        if not self.enabled or depth_mask is None:
            self.last_depth_mask = None
            self.last_depth_prompt_points = []
            self.last_depth_bbox = None
            self.last_depth_area = 0
            return None

        if depth_mask.ndim == 3:
            depth_mask = cv2.cvtColor(depth_mask, cv2.COLOR_BGR2GRAY)

        source_shape = depth_mask.shape[:2]
        height, width = target_shape
        resized = cv2.resize(
            depth_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        binary = (resized > 0).astype(np.uint8) * 255
        # Projected cluster masks are sparse. Use the same bridge operation as
        # the depth-guided demo before deriving SAM reinitialization prompts.
        kernel = np.ones((7, 7), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=3)
        binary = cv2.erode(binary, kernel, iterations=1)
        self.last_depth_mask = binary
        self.last_depth_prompt_points = self._resize_prompt_points(
            positive_prompt_points=positive_prompt_points,
            source_shape=source_shape,
            target_shape=target_shape,
            binary_mask=binary,
        )
        self.last_depth_bbox, self.last_depth_area = self._extract_combined_bbox(binary)
        return binary

    def evaluate_consistency(self, track_mask):
        if self.last_depth_mask is None or track_mask is None:
            self.last_consistency = None
            self._last_track_area = 0
            return None

        track_binary = (track_mask > 0).astype(np.uint8)
        depth_binary = (self.last_depth_mask > 0).astype(np.uint8)
        self._last_track_area = int(np.count_nonzero(track_binary))
        union = np.count_nonzero(track_binary | depth_binary)
        if union == 0:
            self.last_consistency = None
            return None
        intersection = np.count_nonzero(track_binary & depth_binary)
        self.last_consistency = float(intersection) / float(union)
        return self.last_consistency

    def note_reinitialized(self, reason):
        self._last_reinit_frame = self._frame_index
        self.last_reinit_reason = str(reason or "")
        self._reinit_count += 1
        if self.last_reinit_reason == "depth_empty_mask_reinit":
            self._empty_reinit_count += 1
        elif self.last_reinit_reason == "depth_low_consistency_reinit":
            self._inconsistency_reinit_count += 1

    def _cooldown_remaining_frames(self):
        elapsed = self._frame_index - self._last_reinit_frame
        remaining = self.reinit_cooldown_frames - elapsed
        return max(0, remaining)

    def _build_no_reinit_decision(self, consistency):
        return DepthReinitDecision(
            should_reinitialize=False,
            consistency=consistency,
            depth_area=self.last_depth_area,
            bbox=self.last_depth_bbox,
        )

    def _can_reinitialize_now(self):
        return self._cooldown_remaining_frames() == 0

    def _maybe_build_empty_streak_decision(
        self, empty_mask_streak, frame_shape, consistency
    ):
        if (
            not self.reinit_on_empty_streak
            or empty_mask_streak < self.empty_mask_streak_threshold
        ):
            return None
        prompt, bbox = self._build_depth_bbox_prompt(frame_shape, track_mask=None)
        if prompt is None:
            return None
        return DepthReinitDecision(
            should_reinitialize=True,
            reason="depth_empty_mask_reinit",
            prompt=prompt,
            consistency=consistency,
            depth_area=self.last_depth_area,
            bbox=bbox,
        )

    def _maybe_build_inconsistency_decision(
        self,
        consistency,
        frame_shape,
        track_mask=None,
    ):
        if not self.reinit_on_inconsistency:
            return None
        if consistency is None or consistency >= self.consistency_threshold:
            return None
        if self._last_track_area <= 0:
            return None
        prompt, bbox = self._build_depth_bbox_prompt(
            frame_shape,
            track_mask=track_mask,
        )
        if prompt is None:
            return None
        return DepthReinitDecision(
            should_reinitialize=True,
            reason="depth_low_consistency_reinit",
            prompt=prompt,
            consistency=consistency,
            depth_area=self.last_depth_area,
            bbox=bbox,
        )

    def make_reinit_decision(self, track_mask, empty_mask_streak, frame_shape):
        self._frame_index += 1
        consistency = self.evaluate_consistency(track_mask)

        if not self.enabled or self.last_depth_mask is None:
            return self._build_no_reinit_decision(consistency)

        if self.last_depth_area < self.min_mask_area:
            return self._build_no_reinit_decision(consistency)

        if not self._depth_prompt_is_reliable(frame_shape):
            return self._build_no_reinit_decision(consistency)

        if not self._can_reinitialize_now():
            return self._build_no_reinit_decision(consistency)

        track_is_empty = track_mask is None or np.count_nonzero(track_mask) == 0
        if self.reinit_on_empty_streak and track_is_empty:
            decision = self._maybe_build_empty_streak_decision(
                empty_mask_streak=empty_mask_streak,
                frame_shape=frame_shape,
                consistency=consistency,
            )
            if decision is not None:
                return decision

        decision = self._maybe_build_inconsistency_decision(
            consistency=consistency,
            frame_shape=frame_shape,
            track_mask=track_mask,
        )
        if decision is not None:
            return decision

        return self._build_no_reinit_decision(consistency)

    def _depth_prompt_is_reliable(self, frame_shape):
        if self.last_depth_bbox is None:
            return False

        frame_height, frame_width = frame_shape
        if frame_height <= 0 or frame_width <= 0:
            return False

        x0, y0, x1, y1 = self.last_depth_bbox
        bbox_width = max(0, x1 - x0 + 1)
        bbox_height = max(0, y1 - y0 + 1)
        bbox_area = bbox_width * bbox_height
        frame_area = frame_width * frame_height
        if bbox_area <= 0 or frame_area <= 0:
            return False

        if bbox_area / float(frame_area) > self.max_reinit_bbox_area_ratio:
            return False
        if bbox_width / float(frame_width) > self.max_reinit_bbox_width_ratio:
            return False
        if bbox_height / float(frame_height) > self.max_reinit_bbox_height_ratio:
            return False
        return True

    def _extract_combined_bbox(self, binary_mask):
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None, 0

        all_points = np.vstack(contours)
        x, y, width, height = cv2.boundingRect(all_points)
        area = int(np.count_nonzero(binary_mask))
        if area <= 0:
            return None, 0

        return (x, y, x + width - 1, y + height - 1), area

    def _build_depth_bbox_prompt(self, frame_shape, track_mask=None):
        bbox = self.last_depth_bbox
        if bbox is None:
            return None, None

        frame_height, frame_width = frame_shape
        x0, y0, x1, y1 = bbox
        x0 = max(0, x0 - self.bbox_padding)
        y0 = max(0, y0 - self.bbox_padding)
        x1 = min(frame_width - 1, x1 + self.bbox_padding)
        y1 = min(frame_height - 1, y1 + self.bbox_padding)

        if x1 <= x0 or y1 <= y0:
            return None, None

        sampled_points = self.last_depth_prompt_points or self._sample_prompt_points()
        negative_points = self._sample_negative_prompt_points(track_mask)
        coords = [[x0, y0], [x1, y1]]
        labels = [2, 3]
        if sampled_points:
            coords.extend(sampled_points)
            labels.extend([1] * len(sampled_points))
        if negative_points:
            coords.extend(negative_points)
            labels.extend([0] * len(negative_points))

        prompt = {
            "coords": np.array(coords, dtype=np.int32),
            "labels": np.array(labels, dtype=np.int32),
            "text": "",
        }
        return prompt, (x0, y0, x1, y1)

    def _resize_prompt_points(
        self,
        positive_prompt_points,
        source_shape,
        target_shape,
        binary_mask,
    ):
        if (
            positive_prompt_points is None
            or len(positive_prompt_points) == 0
            or source_shape[0] <= 0
            or source_shape[1] <= 0
        ):
            return []

        src_h, src_w = source_shape
        dst_h, dst_w = target_shape
        scale_x = float(dst_w) / float(src_w)
        scale_y = float(dst_h) / float(src_h)

        resized_points = []
        for point in positive_prompt_points:
            if len(point) < 2:
                continue
            x = int(np.clip(round(float(point[0]) * scale_x), 0, dst_w - 1))
            y = int(np.clip(round(float(point[1]) * scale_y), 0, dst_h - 1))
            if binary_mask[y, x] == 0:
                continue
            candidate = [x, y]
            if candidate in resized_points:
                continue
            resized_points.append(candidate)
        return resized_points

    def _sample_prompt_points(self, max_points=4):
        if self.last_depth_mask is None or max_points <= 0:
            return []

        binary = (self.last_depth_mask > 0).astype(np.uint8)
        if np.count_nonzero(binary) == 0:
            return []

        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        points = []
        suppression_radius = max(8, int(min(binary.shape[:2]) * 0.04))

        for _ in range(max_points):
            _, max_value, _, max_loc = cv2.minMaxLoc(distance)
            if max_value <= 0:
                break
            x, y = max_loc
            points.append([int(x), int(y)])
            cv2.circle(distance, (x, y), suppression_radius, 0, thickness=-1)

        return points

    def _sample_negative_prompt_points(self, track_mask, max_points=4):
        if (
            track_mask is None
            or self.last_depth_mask is None
            or max_points <= 0
        ):
            return []

        track_binary = (track_mask > 0).astype(np.uint8)
        if track_binary.shape != self.last_depth_mask.shape:
            track_binary = cv2.resize(
                track_binary,
                (self.last_depth_mask.shape[1], self.last_depth_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        depth_binary = (self.last_depth_mask > 0).astype(np.uint8)
        depth_guard = cv2.dilate(
            depth_binary,
            np.ones((15, 15), np.uint8),
            iterations=1,
        )
        candidate = ((track_binary > 0) & (depth_guard == 0)).astype(np.uint8)
        if np.count_nonzero(candidate) < 64:
            return []

        distance = cv2.distanceTransform(candidate, cv2.DIST_L2, 5)
        points = []
        suppression_radius = max(10, int(min(candidate.shape[:2]) * 0.05))

        for _ in range(max_points):
            _, max_value, _, max_loc = cv2.minMaxLoc(distance)
            if max_value <= 0:
                break
            x, y = max_loc
            points.append([int(x), int(y)])
            cv2.circle(distance, (x, y), suppression_radius, 0, thickness=-1)

        return points
