from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from mesh_predict.pmp.models.tokenizers.dinov2 import Dinov2Model
from mesh_predict.pmp.models.utils.transformers import Modulation
from mesh_predict.pmp.utils.base import BaseModule
from mesh_predict.pmp.utils.typing import *


class DINOV2SingleImageTokenizer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "facebook/dinov2-base"
        width: int = 224
        height: int = 224
        modulation: bool = False
        modulation_zero_init: bool = False
        modulation_single_layer: bool = False
        modulation_cond_dim: int = 16
        freeze_backbone_params: bool = True
        enable_memory_efficient_attention: bool = False
        enable_gradient_checkpointing: bool = False
        freeze_camera_emb: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()
        model: Dinov2Model

        if self.cfg.freeze_backbone_params:
            # freeze dino backbone parameters
            self.register_non_module(
                "model",
                Dinov2Model.from_pretrained(self.cfg.pretrained_model_name_or_path).to(
                    self.device
                ),
            )
            # print("Freezing backbone parameters")

            model = self.non_module("model")
            for p in model.parameters():
                p.requires_grad_(False)
            model.eval()
        else:
            self.model = Dinov2Model.from_pretrained(
                self.cfg.pretrained_model_name_or_path
            ).to(self.device)
            model = self.model

        model.set_use_memory_efficient_attention_xformers(
            self.cfg.enable_memory_efficient_attention
        )
        model.set_gradient_checkpointing(self.cfg.enable_gradient_checkpointing)

        # add modulation
        if self.cfg.modulation:
            modulations = []
            for layer in model.encoder.layer:
                norm1_modulation = Modulation(
                    model.config.hidden_size,
                    self.cfg.modulation_cond_dim,
                    zero_init=self.cfg.modulation_zero_init,
                    single_layer=self.cfg.modulation_single_layer,
                )
                norm2_modulation = Modulation(
                    model.config.hidden_size,
                    self.cfg.modulation_cond_dim,
                    zero_init=self.cfg.modulation_zero_init,
                    single_layer=self.cfg.modulation_single_layer,
                )
                # print("registering modulation")
                layer.register_ada_norm_modulation(norm1_modulation, norm2_modulation)
                modulations += [norm1_modulation, norm2_modulation]
            if not self.cfg.freeze_camera_emb:
                for name, param in model.state_dict().items():

                    if "norm1_modulation" in name or "norm2_modulation" in name:
                        # print(name,"True")
                        param.requires_grad = True

                    # print(name,param.requires_grad)
                    # else:
                    # print(name,"False")

            self.modulations = nn.ModuleList(modulations)

        self.register_buffer(
            "image_mean",
            torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1),
            persistent=False,
        )

    def forward(
        self,
        images: Float[Tensor, "B *N C H W"],
        modulation_cond: Optional[Float[Tensor, "B *N Cc"]],
    ) -> Float[Tensor, "B *N Ct Nt"]:
        model: Dinov2Model
        if self.cfg.freeze_backbone_params:
            model = self.non_module("model")
        else:
            model = self.model

        packed = False
        if images.ndim == 4:
            packed = True
            images = images.unsqueeze(1)
            if modulation_cond is not None:
                assert modulation_cond.ndim == 2
                modulation_cond = modulation_cond.unsqueeze(1)

        batch_size, n_input_views = images.shape[:2]
        images = (images - self.image_mean) / self.image_std
        out = model(
            rearrange(images, "B N C H W -> (B N) C H W"),
            modulation_cond=(
                rearrange(modulation_cond, "B N Cc -> (B N) Cc")
                if modulation_cond is not None
                else None
            ),
        )

        # print("out ", out)
        # print("img",images.shape)
        # print("last_hidden_state ", out.last_hidden_state.shape)
        # print("pooler_output ", out.pooler_output.shape)
        # out = model(rearrange(images, "B N C H W -> (B N) C H W"))
        local_features, global_features = out.last_hidden_state, out.pooler_output
        local_features = local_features.permute(0, 2, 1)
        local_features = rearrange(
            local_features, "(B N) Ct Nt -> B N Ct Nt", B=batch_size
        )
        if packed:
            local_features = local_features.squeeze(1)

        # print("local_feature shape ",local_features.shape)
        # print("global feature ",global_features.shape)
        return local_features

    def detokenize(self, *args, **kwargs):
        raise NotImplementedError
