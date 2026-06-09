from dataclasses import dataclass

import torch
import torch.nn as nn
from mesh_predict.pmp.utils.base import BaseModule
from mesh_predict.pmp.utils.typing import *


class PointLearnablePositionalEmbedding(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        num_pcl: int = 2048
        num_channels: int = 512

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.pcl_embeddings = nn.Embedding(self.cfg.num_pcl, self.cfg.num_channels)

    def forward(self, batch_size, cond_embeddings=None) -> Float[Tensor, "B Ct Nt"]:
        range_ = torch.arange(self.cfg.num_pcl, device=self.device)
        # print(range_)
        embeddings = self.pcl_embeddings(range_).unsqueeze(0).repeat((batch_size, 1, 1))

        if cond_embeddings is not None:
            embeddings = embeddings + cond_embeddings
        # print(embeddings)
        # print(torch.permute(embeddings, (0,2,1)).shape)
        return torch.permute(embeddings, (0, 2, 1))

    def detokenize(self, tokens: Float[Tensor, "B Ct Nt"]) -> Float[Tensor, "B Nt Ct"]:
        return torch.permute(tokens, (0, 2, 1))
