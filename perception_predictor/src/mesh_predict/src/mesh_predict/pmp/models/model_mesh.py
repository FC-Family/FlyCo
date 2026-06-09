import torch
from dataclasses import dataclass, field
import warnings

from mesh_predict.pmp import find
from mesh_predict.pmp.utils.config import parse_structured
from mesh_predict.pmp.utils.misc import load_module_weights
from mesh_predict.pmp.utils.typing import *
from torch.cuda.amp import autocast as autocast


class PMPMesh(torch.nn.Module):
    @dataclass
    class Config:
        model_cls: str = "PMPMesh"

        weights: Optional[str] = None
        weights_ignore_modules: Optional[List[str]] = None

        pointcloud_generator_cls: str = ""
        pointcloud_generator: dict = field(default_factory=dict)

        mesh_decoder_cls: str = ""
        mesh_decoder: dict = field(default_factory=dict)
        mesh_input_key: str = "pc"

    cfg: Config

    def load_weights(self, weights: str, ignore_modules: Optional[List[str]] = None):
        state_dict = load_module_weights(
            weights, ignore_modules=ignore_modules, map_location="cpu"
        )
        incompatible = self.load_state_dict(state_dict, strict=False)
        warnings.warn(
            "Loaded PMP mesh checkpoint "
            f"{weights} with {len(state_dict)} tensors; "
            f"missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}; "
            f"missing_sample={incompatible.missing_keys[:8]} "
            f"unexpected_sample={incompatible.unexpected_keys[:8]}"
        )

    def __init__(self, cfg):
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)

        self.pointcloud_generator = find(self.cfg.pointcloud_generator_cls)(
            self.cfg.pointcloud_generator
        )

        self.mesh_decoder = find(self.cfg.mesh_decoder_cls)(self.cfg.mesh_decoder)

        if self.cfg.weights is not None:
            self.load_weights(self.cfg.weights, self.cfg.weights_ignore_modules)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        pc_out = self.pointcloud_generator(batch)
        mesh_points = pc_out.get(self.cfg.mesh_input_key)
        if mesh_points is None:
            raise KeyError(
                f"pointcloud_generator output has no `{self.cfg.mesh_input_key}` key"
            )
        with autocast(enabled=False):
            mesh_out = self.mesh_decoder(input_xyz=mesh_points[0].float(), feat=None)

        return {**pc_out, **mesh_out}

    def pred_mesh(self, pc):
        with autocast(enabled=False):
            mesh_out = self.mesh_decoder(input_xyz=pc.float(), feat=None)
        return {**mesh_out}

    def pred_pc(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        pc_out = self.pointcloud_generator(batch)
        return {**pc_out}
