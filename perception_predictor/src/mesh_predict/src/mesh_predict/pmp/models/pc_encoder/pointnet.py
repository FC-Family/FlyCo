from dataclasses import dataclass, field

import torch
import torch.nn as nn
from einops import rearrange
from mesh_predict.pmp.models.pc_encoder.pointnet2.pointnet2_modules import (
    PointnetFPModule,
    PointnetSAModuleMSG,
)
from mesh_predict.pmp.utils.base import BaseModule
from mesh_predict.pmp.utils.typing import *


RADIUS = [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]
NSAMPLE = [[16, 32], [16, 32], [16, 32], [16, 32]]
MLPS = [
    [[16, 16, 32], [32, 32, 64]],
    [[64, 64, 128], [64, 96, 128]],
    [[128, 196, 256], [128, 196, 256]],
    [[256, 256, 512], [256, 384, 512]],
]


class PointNet2(BaseModule):

    @dataclass
    class Config(BaseModule.Config):
        channels: int = 3
        pts_list: List[int] = field(default_factory=lambda: [2048, 1024, 256, 64])
        pt_dim: int = 512

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # self.cfg = cfg
        input_channels = self.cfg.channels
        self.h_size = self.cfg.pt_dim
        NPOINTS = self.cfg.pts_list

        FP_MLPS = [[self.h_size, self.h_size], [512, 512], [512, 512], [512, 512]]

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels - 3]  # xyz
        ft = False
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=NPOINTS[k],
                    radii=RADIUS[k],
                    nsamples=NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=True,
                    bn=True,
                    features=ft,
                )
            )
            ft = True
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(FP_MLPS.__len__()):
            pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            self.FP_modules.append(
                PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k])
            )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud: torch.Tensor, hidden_states=None):
        r"""
        :param pointcloud: input point cloud -> torch.Tensor, shape [B, N, 3 + input_channels]
        :return: feature: positional point feature -> torch.Tensor, shape [B, N, 3+C]
        """

        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        feature = rearrange(l_features[0], "b c n -> b n c")
        return feature
