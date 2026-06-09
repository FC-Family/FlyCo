from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

from mesh_predict.pmp.utils.base import BaseModule
from mesh_predict.pmp.utils.ops import get_activation


class PointOutLayer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 1024
        out_channels: int = 3

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.point_layer = nn.Linear(self.cfg.in_channels, self.cfg.out_channels)
        nn.init.constant_(self.point_layer.weight, 0)
        nn.init.constant_(self.point_layer.bias, 0)

    def forward(self, x):
        return self.point_layer(x)


class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_neurons: int,
        n_hidden_layers: int,
        activation: str = "relu",
        output_activation: str = None,
        bias: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Linear(dim_in, n_neurons, bias=bias),
            self.make_activation(activation),
        ]
        for _ in range(n_hidden_layers - 1):
            layers += [
                nn.Linear(n_neurons, n_neurons, bias=bias),
                self.make_activation(activation),
            ]
        layers.append(nn.Linear(n_neurons, dim_out, bias=bias))
        self.layers = nn.Sequential(*layers)
        self.output_activation = get_activation(output_activation)

    def forward(self, x):
        return self.output_activation(self.layers(x))

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        if activation == "silu":
            return nn.SiLU(inplace=True)
        raise NotImplementedError
