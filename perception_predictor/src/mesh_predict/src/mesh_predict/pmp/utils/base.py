from dataclasses import dataclass

import torch.nn as nn
from mesh_predict.pmp.utils.config import parse_structured
from mesh_predict.pmp.utils.misc import get_device, load_module_weights
from mesh_predict.pmp.utils.typing import *


class BaseModule(nn.Module):
    @dataclass
    class Config:
        weights: Optional[str] = None
        freeze: Optional[bool] = False

    cfg: Config  # add this to every subclass of BaseModule to enable static type checking

    def __init__(
        self, cfg: Optional[Union[dict, DictConfig]] = None, *args, **kwargs
    ) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self.device = get_device()
        self._non_modules = {}
        self.configure(*args, **kwargs)
        if self.cfg.weights is not None:
            # format: path/to/weights:module_name
            weights_path, module_name = self.cfg.weights.split(":")
            state_dict = load_module_weights(
                weights_path, module_name=module_name, map_location="cpu"
            )
            self.load_state_dict(state_dict, strict=False)
        if self.cfg.freeze:
            for params in self.parameters():
                params.requires_grad = False

    def configure(self, *args, **kwargs) -> None:
        pass

    def register_non_module(self, name: str, module: nn.Module) -> None:
        # non-modules won't be treated as model parameters
        if name in self._non_modules:
            raise ValueError(f"Non-module {name} already exists!")
        self._non_modules[name] = module

    def non_module(self, name: str):
        return self._non_modules.get(name, None)
