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


class TrackingPipeline:
    def __init__(
        self,
        provider,
        prompt_session,
        pointcloud_projector,
        bridge,
        depth_guidance=None,
    ):
        self.provider = provider
        self.prompt_session = prompt_session
        self.pointcloud_projector = pointcloud_projector
        self.bridge = bridge
        self.depth_guidance = depth_guidance
        self._empty_mask_streak = 0

    @property
    def empty_mask_streak(self):
        return self._empty_mask_streak

    def reset_runtime_state(self):
        self._empty_mask_streak = 0

    def _build_tracked_result(
        self,
        frame_rgb,
        track_mask,
        mask_with_transform,
        reinit_result=None,
    ):
        masked_rgb = cv2.bitwise_and(frame_rgb, frame_rgb, mask=track_mask)
        mask_rgba = cv2.cvtColor(masked_rgb, cv2.COLOR_BGR2RGBA)
        mask_rgba[:, :, 3] = track_mask

        if mask_with_transform is not None and self.bridge is not None:
            mask_with_transform.mask_image = self.bridge.cv2_to_imgmsg(mask_rgba)
        return {
            "state": "tracked",
            "track_mask": track_mask,
            "masked_rgb": masked_rgb,
            "mask_rgba": mask_rgba,
            "mask_with_transform_msg": mask_with_transform,
            "depth_mask_iou": (
                None if reinit_result is None else reinit_result["depth_mask_iou"]
            ),
            "reinit_reason": (
                None if reinit_result is None else reinit_result["reinit_reason"]
            ),
            "reinit_bbox": (
                None if reinit_result is None else reinit_result["reinit_bbox"]
            ),
            "depth_area": (
                None if reinit_result is None else reinit_result["depth_area"]
            ),
            "reinit_prompt": (
                None if reinit_result is None else reinit_result.get("reinit_prompt")
            ),
            "pre_reinit_mask": (
                None if reinit_result is None else reinit_result.get("pre_reinit_mask")
            ),
            "reinit_depth_mask": (
                None if reinit_result is None else reinit_result.get("reinit_depth_mask")
            ),
            "depth_guidance_snapshot": (
                None if self.depth_guidance is None else self.depth_guidance.snapshot()
            ),
            "provider_stats": self.provider.stats,
        }

    def build_tracked_result_from_mask(self, frame_rgb, track_mask, mask_with_transform):
        if track_mask is None or np.count_nonzero(track_mask) == 0:
            return None
        self._empty_mask_streak = 0
        return self._build_tracked_result(
            frame_rgb=frame_rgb,
            track_mask=track_mask,
            mask_with_transform=mask_with_transform,
        )

    def sync_provider_state(self, frame_rgb):
        prompt = self.prompt_session.as_provider_prompt()
        if not self.provider.initialized:
            if prompt is None:
                return False, "No query points", None, None
            if not self.provider.can_initialize_from_prompt(prompt):
                return (
                    False,
                    "Prompt is not actionable for the current provider",
                    None,
                    None,
                )

            initial_mask = self.provider.initialize(frame=frame_rgb, prompt=prompt)
            self.prompt_session.mark_provider_prompt_consumed()
            if self.depth_guidance is not None:
                self.depth_guidance.note_reinitialized("provider_initialize")
            self.reset_runtime_state()
            return True, None, "initialized", initial_mask

        if self.prompt_session.has_pending_provider_prompt_update:
            if prompt is None:
                return False, "No prompt payload", None, None
            if not self.provider.can_initialize_from_prompt(prompt):
                return (
                    False,
                    "Prompt update is not actionable for the current provider",
                    None,
                    None,
                )

            initial_mask = self.provider.reinitialize_from_prompt(
                frame=frame_rgb,
                prompt=prompt,
                reason="prompt_update",
            )
            self.prompt_session.mark_provider_prompt_consumed()
            if self.depth_guidance is not None:
                self.depth_guidance.note_reinitialized("prompt_update")
            self.reset_runtime_state()
            return True, None, "prompt_reinitialized", initial_mask

        return True, None, None, None

    def _maybe_depth_reinitialize(
        self,
        frame_rgb,
        track_mask,
        depth_mask=None,
    ):
        if self.depth_guidance is None:
            return None

        decision = self.depth_guidance.make_reinit_decision(
            track_mask=track_mask,
            empty_mask_streak=self._empty_mask_streak,
            frame_shape=frame_rgb.shape[:2],
        )
        if not decision.should_reinitialize:
            return {
                "state": "no_reinit",
                "depth_mask_iou": decision.consistency,
                "reinit_reason": None,
                "reinit_bbox": decision.bbox,
                "depth_area": decision.depth_area,
                "depth_guidance_snapshot": (
                    None
                    if self.depth_guidance is None
                    else self.depth_guidance.snapshot()
                ),
            }

        if decision.prompt is not None and not decision.prompt.get("text"):
            decision.prompt["text"] = self.prompt_session.query_text

        initial_mask = self.provider.reinitialize_from_prompt(
            frame=frame_rgb,
            prompt=decision.prompt,
            reason=decision.reason,
        )
        self.depth_guidance.note_reinitialized(decision.reason)
        self.reset_runtime_state()
        return {
            "state": "reinitialized",
            "depth_mask_iou": decision.consistency,
            "reinit_reason": decision.reason,
            "reinit_bbox": decision.bbox,
            "depth_area": decision.depth_area,
            "track_mask": initial_mask,
            "reinit_prompt": decision.prompt,
            "pre_reinit_mask": None if track_mask is None else track_mask.copy(),
            "reinit_depth_mask": None if depth_mask is None else depth_mask.copy(),
            "depth_guidance_snapshot": (
                None if self.depth_guidance is None else self.depth_guidance.snapshot()
            ),
        }

    def track_frame(
        self,
        frame_rgb,
        mask_with_transform,
        depth_mask=None,
        depth_prompt_points=None,
    ):
        processed_depth_mask = None
        if self.depth_guidance is not None:
            processed_depth_mask = self.depth_guidance.preprocess(
                depth_mask,
                frame_rgb.shape[:2],
                positive_prompt_points=depth_prompt_points,
            )

        track_mask = self.provider.track(
            frame=frame_rgb, depth_mask=processed_depth_mask
        )
        if track_mask is None:
            self._empty_mask_streak += 1
            reinit_result = self._maybe_depth_reinitialize(
                frame_rgb=frame_rgb,
                track_mask=None,
                depth_mask=processed_depth_mask,
            )
            if reinit_result is not None and reinit_result["state"] == "reinitialized":
                track_mask = reinit_result.get("track_mask")
                if track_mask is not None and np.count_nonzero(track_mask) > 0:
                    return self._build_tracked_result(
                        frame_rgb=frame_rgb,
                        track_mask=track_mask,
                        mask_with_transform=mask_with_transform,
                        reinit_result=reinit_result,
                    )
                return reinit_result
            return None

        if np.count_nonzero(track_mask) == 0:
            self._empty_mask_streak += 1
            reinit_result = self._maybe_depth_reinitialize(
                frame_rgb=frame_rgb,
                track_mask=track_mask,
                depth_mask=processed_depth_mask,
            )
            if reinit_result is not None and reinit_result["state"] == "reinitialized":
                track_mask = reinit_result.get("track_mask")
                if track_mask is not None and np.count_nonzero(track_mask) > 0:
                    return self._build_tracked_result(
                        frame_rgb=frame_rgb,
                        track_mask=track_mask,
                        mask_with_transform=mask_with_transform,
                        reinit_result=reinit_result,
                    )
                return reinit_result
            return None

        self._empty_mask_streak = 0
        reinit_result = self._maybe_depth_reinitialize(
            frame_rgb=frame_rgb,
            track_mask=track_mask,
            depth_mask=processed_depth_mask,
        )
        if reinit_result is not None and reinit_result["state"] == "reinitialized":
            track_mask = reinit_result.get("track_mask")
            if track_mask is not None and np.count_nonzero(track_mask) > 0:
                return self._build_tracked_result(
                    frame_rgb=frame_rgb,
                    track_mask=track_mask,
                    mask_with_transform=mask_with_transform,
                    reinit_result=reinit_result,
                )
            return reinit_result

        return self._build_tracked_result(
            frame_rgb=frame_rgb,
            track_mask=track_mask,
            mask_with_transform=mask_with_transform,
            reinit_result=reinit_result,
        )

    def project_target_points(
        self,
        pointcloud_msgs,
        mask_resized,
        body_to_world,
        static_transforms,
        image_size,
        camera_pose=None,
    ):
        return self.pointcloud_projector.project_mask_to_world_points(
            points_msg_list=pointcloud_msgs,
            mask_resized=mask_resized,
            body_to_world=body_to_world,
            static_transforms=static_transforms,
            image_size=image_size,
            camera_pose=camera_pose,
        )
