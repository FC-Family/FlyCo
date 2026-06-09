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

import os

from mesh_predict.pmp import find
from mesh_predict.pmp.utils.config import load_config

from mesh_predict.runtime.resources import OfficialRuntimeSpec


class PCMeshPredictBackend:
    """Thin adapter around the packaged PMP model implementation."""

    def __init__(self, runtime_spec: OfficialRuntimeSpec, device="cuda"):
        assert runtime_spec.model_config_path is not None and os.path.exists(
            runtime_spec.model_config_path
        )

        self.cfg = load_config(runtime_spec.model_config_path)
        self.cfg.setting.pretrain_path = runtime_spec.mesh_pred_checkpoint
        self.cfg.system.weights = runtime_spec.mesh_pred_checkpoint

        image_tokenizer = self.cfg.system.pointcloud_generator.image_tokenizer
        image_tokenizer.pretrained_model_name_or_path = runtime_spec.dino_model

        text_encoder = self.cfg.system.pointcloud_generator.text_encoder
        text_encoder.checkpoint = runtime_spec.siglip_model

        mesh_decoder = self.cfg.system.mesh_decoder
        if runtime_spec.nksr_checkpoint is not None:
            mesh_decoder.checkpoint_url = runtime_spec.nksr_checkpoint

        self.device = device
        self.model = find(self.cfg.system.model_cls)(self.cfg.system).to(self.device)
        self.model.eval()

    def pred_pc(self, batch):
        return self.model.pred_pc(batch)

    def pred_mesh(self, pointcloud):
        return self.model.pred_mesh(pointcloud)
