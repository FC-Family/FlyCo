import inspect
import logging
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2_camera_predictor

# from sam_tk.utils.segmentor import Segmentor
from sam_tk.utils.utils import *
from sam_tk.utils.Tracker import Tracker

LOGGER = logging.getLogger(__name__)


class SamTracker(Tracker):
    def __init__(self, args) -> None:
        """
        Initialize the official SAM2 camera tracker.
        """
        super().__init__(args)
        model_cfg = args.get("model_cfg")
        sam_checkpoint = args.get("sam_checkpoint")
        hydra_overrides_extra = self._build_hydra_overrides(args)
        if not model_cfg or not sam_checkpoint:
            raise ValueError("sam2camera requires model_cfg and sam_checkpoint")
        model_cfg_path = Path(model_cfg)
        if model_cfg_path.is_absolute():
            config_dir = str(model_cfg_path.parent.parent)
            config_name = str(model_cfg_path.relative_to(config_dir))
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                self.sam2_predict = build_sam2_camera_predictor(
                    config_name,
                    str(Path(sam_checkpoint)),
                    hydra_overrides_extra=hydra_overrides_extra,
                )
        else:
            self.sam2_predict = build_sam2_camera_predictor(
                str(model_cfg_path),
                str(Path(sam_checkpoint)),
                hydra_overrides_extra=hydra_overrides_extra,
            )

        self.everything_points = []
        self.everything_labels = []
        self.last_mask = None
        self.init_flag = False
        self._supports_depth_mask = "depth_mask" in inspect.signature(
            self.sam2_predict.track
        ).parameters
        self._depth_guidance_warning_emitted = False

    @staticmethod
    def _build_hydra_overrides(args):
        overrides = []
        samurai_mode = args.get("samurai_mode")
        if samurai_mode is not None:
            overrides.append(f"++model.samurai_mode={str(bool(samurai_mode)).lower()}")

        for arg_name in (
            "depth_score_weight",
            "kf_score_weight",
            "memory_bank_geo_score_threshold",
        ):
            value = args.get(arg_name)
            if value is not None:
                overrides.append(f"++model.{arg_name}={float(value)}")
        return overrides

    @staticmethod
    def _mask_logits_to_numpy(out_mask_logits):
        if len(out_mask_logits) == 0:
            return None

        pred_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0)
        return pred_mask.cpu().numpy().astype(np.uint8).squeeze(2)

    def _predictor_device(self):
        condition_state = getattr(self.sam2_predict, "condition_state", None) or {}
        device = condition_state.get("device")
        if device is not None:
            return device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_depth_mask(self, depth_mask):
        if depth_mask is None:
            return None

        if torch.is_tensor(depth_mask):
            depth_tensor = depth_mask.detach()
        else:
            depth_tensor = torch.from_numpy(np.asarray(depth_mask))

        if depth_tensor.ndim == 2:
            depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
        elif depth_tensor.ndim == 3:
            depth_tensor = depth_tensor.unsqueeze(0)
        elif depth_tensor.ndim != 4:
            raise ValueError(
                f"depth_mask must have 2, 3, or 4 dimensions, got {depth_tensor.ndim}"
            )

        return depth_tensor.float().to(self._predictor_device())

    def reinit(self, frame, mask):
        _ = mask
        self.sam2_predict.reset_state()
        self.sam2_predict.load_first_frame(frame)
        self.last_mask = None
        self.init_flag = False

    def if_init(self):
        return self.init_flag

    def init_tracker(self, frame, coords=None, labels=None, bbox=None):
        self.sam2_predict.load_first_frame(frame)

        if coords is None:
            coords = np.zeros((0, 2), dtype=np.float32)
        coords = np.asarray(coords)

        if labels is None:
            labels = np.ones((len(coords),), dtype=int)
        labels = np.asarray(labels)

        prompt_kwargs = {
            "frame_idx": 0,
            "obj_id": 1,
        }
        if bbox is not None:
            prompt_kwargs["bbox"] = np.asarray(bbox, dtype=float)
        if len(coords) != 0:
            prompt_kwargs["points"] = coords
            prompt_kwargs["labels"] = labels

        if "bbox" not in prompt_kwargs and "points" not in prompt_kwargs:
            raise ValueError("init_tracker requires bbox or points")

        _, out_obj_ids, out_mask_logits = self.sam2_predict.add_new_prompt(
            **prompt_kwargs
        )
        _ = out_obj_ids

        self.last_mask = self._mask_logits_to_numpy(out_mask_logits)
        self.init_flag = True
        return self.last_mask

    def seg_acc_click(self, frame, coords, modes, multimask=True):
        labels = np.ones((len(coords), 1)[0])
        self.sam2_predict.add_conditioning_frame(frame)

        _, out_obj_ids, out_mask_logits = self.sam2_predict.add_new_points(
            frame_idx=self.sam2_predict.condition_state["num_frames"] - 1,
            obj_id=(1),
            points=coords,
            labels=labels,
        )
        self.sam2_predict.frame_idx = self.sam2_predict.condition_state["num_frames"]
        self.sam2_predict.condition_state["tracking_has_started"] = False
        return out_mask_logits.cpu().numpy().astype(np.uint8) * 255, out_obj_ids

    def track(
        self,
        frame,
        depth_mask=None,
        update_memory=False,
        check_lost=False,
        iou_threshold=0.6,
        update_mask=False,
    ):
        """
        Track all known objects.
        Arguments:
            frame: numpy array (h,w,3)
        Return:
            origin_merged_mask: numpy array (h,w)
        """
        predictor_depth_mask = self._prepare_depth_mask(depth_mask)
        if self._supports_depth_mask:
            _, out_mask_logits = self.sam2_predict.track(
                frame,
                depth_mask=predictor_depth_mask,
            )
        else:
            if predictor_depth_mask is not None and not self._depth_guidance_warning_emitted:
                LOGGER.warning(
                    "Installed SAM2 predictor does not accept depth_mask; falling back to RGB-only tracking."
                )
                self._depth_guidance_warning_emitted = True
            _, out_mask_logits = self.sam2_predict.track(frame)
        pred_mask_np = self._mask_logits_to_numpy(out_mask_logits)
        if pred_mask_np is None:
            return None
        if check_lost and self.last_mask is not None:
            last_mask_np = self.last_mask
            iou = cal_iou_from_masks(last_mask_np, pred_mask_np)
            _ = iou

        self.last_mask = pred_mask_np

        return pred_mask_np

    def restart_tracker(self):
        self.sam2_predict.reset_state()
        self.everything_points = []
        self.everything_labels = []
        self.last_mask = None
        self.init_flag = False
