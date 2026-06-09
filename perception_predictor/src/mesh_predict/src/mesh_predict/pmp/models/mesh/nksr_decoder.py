# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from dataclasses import dataclass, field
from typing import Optional

import torch
from nksr import NKSRNetwork, SparseFeatureHierarchy
from nksr.configs import load_checkpoint_from_url
from nksr.fields import KernelField, LayerField, NeuralField

from mesh_predict.pmp.utils.base import BaseModule


class NKSRDecoder(BaseModule):

    @dataclass
    class Config(BaseModule.Config):

        checkpoint_url: Optional[str] = None

        # nskr_network: dict = field(default_factory=dict)
        voxel_size: float = 0.05
        tree_depth: int = 6
        adaptive_depth: int = 4
        feature: str = "none"
        geometry: str = "geometry"

        kernel_dim: int = 4
        unet: dict = field(default_factory=lambda: {"f_maps": 32})
        interpolator: dict = field(
            default_factory=lambda: {"n_hidden": 2, "hidden_dim": 16}
        )

        solver: dict = field(
            default_factory=lambda: {"normal_weight": 10000.0, "pos_weight": 10000.0}
        )
        udf: dict = field(default_factory=lambda: {"enabled": False})
        adaptive_policy: dict = field(default_factory=lambda: {"tau": False})
        density_range: str = "none"

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.network = NKSRNetwork(self.cfg)

        if self.cfg.checkpoint_url:
            ckpt_data = load_checkpoint_from_url(self.cfg.checkpoint_url)
            self.network.load_state_dict(ckpt_data["state_dict"])

    def forward(self, input_xyz, feat=None):

        out: dict = {}

        out["feat"] = feat

        enc_svh = SparseFeatureHierarchy(
            voxel_size=self.cfg.voxel_size,
            depth=self.cfg.tree_depth,
            device=self.device,
        )
        enc_svh.build_point_splatting(input_xyz)

        feat = self.network.encoder(input_xyz, feat, enc_svh, 0)
        feat, dec_svh, udf_svh = self.network.unet(
            feat,
            enc_svh,
            adaptive_depth=self.cfg.adaptive_depth,
            gt_decoder_svh=out.get("gt_svh", None),
        )

        out.update({"enc_svh": enc_svh, "dec_svh": dec_svh, "dec_tmp_svh": udf_svh})

        output_field = KernelField(
            svh=dec_svh,
            interpolator=self.network.interpolators,
            features=feat.basis_features,
            approx_kernel_grad=False,
        )

        normal_xyz = torch.cat(
            [dec_svh.get_voxel_centers(d) for d in range(self.cfg.adaptive_depth)]
        )
        normal_value = torch.cat(
            [feat.normal_features[d] for d in range(self.cfg.adaptive_depth)]
        )

        normal_weight = (
            self.cfg.solver.normal_weight
            / normal_xyz.size(0)
            * (self.cfg.voxel_size**2)
        )

        output_field.solve_non_fused(
            pos_xyz=input_xyz,
            normal_xyz=normal_xyz,
            normal_value=-normal_value,
            pos_weight=self.cfg.solver.pos_weight / input_xyz.size(0),
            normal_weight=normal_weight,
            reg_weight=1.0,
        )

        if self.cfg.udf.enabled:
            mask_field = NeuralField(
                svh=udf_svh,
                decoder=self.network.udf_decoder,
                features=feat.udf_features,
            )
            mask_field.set_level_set(2 * self.cfg.voxel_size)
        else:
            mask_field = LayerField(dec_svh, self.cfg.adaptive_depth)
        output_field.set_mask_field(mask_field)

        out.update(
            {
                "structure_features": feat.structure_features,
                "normal_features": feat.normal_features,
                "basis_features": feat.basis_features,
                "field": output_field,
                "mesh": output_field.extract_dual_mesh(mise_iter=1),
            }
        )

        return out
