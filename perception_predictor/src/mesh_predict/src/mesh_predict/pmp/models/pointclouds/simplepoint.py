from dataclasses import dataclass, field

from mesh_predict.pmp import find
import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention
from einops import rearrange
from pointnet2_ops import pointnet2_utils

from mesh_predict.core.densification import repeated_first_level_upsampling
from mesh_predict.pmp.models.utils.transformers import MemoryEfficientAttentionMixin
from mesh_predict.pmp.utils.base import BaseModule
from mesh_predict.pmp.utils.typing import *


def fps(data, number):
    """Sample point-cloud anchors without research-only point encoders."""

    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = (
        pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx)
        .transpose(1, 2)
        .contiguous()
    )
    return fps_data


class FeatureProjectionMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = out_dim if hidden_dim is None else hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class AlternatingAttentionBlock(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, dropout: float = 0.0, mlp_ratio: int = 4
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"AlternatingAttentionBlock requires dim ({dim}) divisible by "
                f"num_heads ({num_heads})."
            )
        attention_head_dim = dim // num_heads
        self.geom_norm1 = nn.LayerNorm(dim)
        self.geom_attn = Attention(
            query_dim=dim,
            heads=num_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
        )
        self.geom_norm2 = nn.LayerNorm(dim)
        self.geom_mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

        self.semantic_norm1 = nn.LayerNorm(dim)
        self.semantic_attn = Attention(
            query_dim=dim,
            heads=num_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
        )
        self.semantic_norm2 = nn.LayerNorm(dim)
        self.semantic_mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_kv = nn.LayerNorm(dim)
        self.cross_attn = Attention(
            query_dim=dim,
            cross_attention_dim=dim,
            heads=num_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
        )
        self.cross_norm_ff = nn.LayerNorm(dim)
        self.cross_mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, geom_tokens, semantic_tokens):
        geom_residual = self.geom_norm1(geom_tokens)
        geom_tokens = geom_tokens + self.geom_attn(geom_residual)
        geom_tokens = geom_tokens + self.geom_mlp(self.geom_norm2(geom_tokens))

        semantic_residual = self.semantic_norm1(semantic_tokens)
        semantic_tokens = semantic_tokens + self.semantic_attn(semantic_residual)
        semantic_tokens = semantic_tokens + self.semantic_mlp(
            self.semantic_norm2(semantic_tokens)
        )

        geom_query = self.cross_norm_q(geom_tokens)
        semantic_context = self.cross_norm_kv(semantic_tokens)
        geom_tokens = geom_tokens + self.cross_attn(
            geom_query, encoder_hidden_states=semantic_context
        )
        geom_tokens = geom_tokens + self.cross_mlp(self.cross_norm_ff(geom_tokens))
        return geom_tokens, semantic_tokens


class AlternatingAttentionFusion(nn.Module, MemoryEfficientAttentionMixin):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        gradient_checkpointing: bool = False,
        enable_memory_efficient_attention: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                AlternatingAttentionBlock(
                    dim=dim, num_heads=num_heads, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.gradient_checkpointing = gradient_checkpointing
        self._enable_memory_efficient_attention = enable_memory_efficient_attention
        self._memory_efficient_attention_enabled = False
        self._maybe_enable_memory_efficient_attention()

    def _maybe_enable_memory_efficient_attention(self):
        if (
            self._enable_memory_efficient_attention
            and not self._memory_efficient_attention_enabled
            and torch.cuda.is_available()
        ):
            self.set_use_memory_efficient_attention_xformers(True)
            self._memory_efficient_attention_enabled = True

    def forward(self, geom_tokens, semantic_tokens):
        self._maybe_enable_memory_efficient_attention()
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                geom_tokens, semantic_tokens = torch.utils.checkpoint.checkpoint(
                    block,
                    geom_tokens,
                    semantic_tokens,
                    use_reentrant=False,
                )
            else:
                geom_tokens, semantic_tokens = block(geom_tokens, semantic_tokens)
        return geom_tokens, semantic_tokens

class SimplePointGenerator(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        camera_embedder_cls: str = ""
        camera_embedder: dict = field(default_factory=dict)

        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)


        pointcloud_embedding_cls: str = ""
        pointcloud_embedding: dict = field(default_factory=dict)

        pointcloud_encoder_cls: str = ""
        pointcloud_encoder: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)

        pointcloud_upsampling_cls: str = ""
        pointcloud_upsampling: dict = field(default_factory=dict)

        flip_c2w_cond: bool = True
        use_concat_self_attn: bool = False
        use_alternating_fusion: bool = False
        projection_dropout: float = 0.0
        alternating_gradient_checkpointing: bool = False
        alternating_enable_memory_efficient_attention: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.image_tokenizer = find(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )

        assert self.cfg.camera_embedder_cls == "mesh_predict.pmp.models.utils.networks.MLP"
        weights = (
            self.cfg.camera_embedder.pop("weights")
            if "weights" in self.cfg.camera_embedder
            else None
        )
        self.camera_embedder = find(self.cfg.camera_embedder_cls)(
            **self.cfg.camera_embedder
        )
        if weights:
            from mesh_predict.pmp.utils.misc import load_module_weights

            weights_path, module_name = weights.split(":")
            state_dict = load_module_weights(
                weights_path, module_name=module_name, map_location="cpu"
            )
            self.camera_embedder.load_state_dict(state_dict)

        self.pointcloud_encoder = find(self.cfg.pointcloud_encoder_cls)(
            self.cfg.pointcloud_encoder
        )

        self.pointcloud_embedding = find(self.cfg.pointcloud_embedding_cls)(
            self.cfg.pointcloud_embedding
        )
        # self.tokenizer=find(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.backbone = None
        if self.cfg.use_concat_self_attn or not self.cfg.use_alternating_fusion:
            self.backbone = find(self.cfg.backbone_cls)(self.cfg.backbone)
        """
            post_processor_cls: tgs.models.networks.PointOutLayer
            post_processor:
            in_channels: 512
            out_channels: 3
        """

        self.post_processor = find(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )


        # self.pointcloud_upsampling = find(self.cfg.pointcloud_upsampling_cls)(self.cfg.pointcloud_upsampling)

    def forward(self, batch, encoder_hidden_states=None, **kwargs):
        batch_size, n_input_views = batch["rgb_cond"].shape[:2]
        # print("rgb shape ",batch["rgb_cond"].shape)

        if encoder_hidden_states is None:
            # Camera modulation
            c2w_cond = batch["c2w_cond"].clone()
            if self.cfg.flip_c2w_cond:
                c2w_cond[..., :3, 1:3] *= -1
            camera_extri = c2w_cond.view(*c2w_cond.shape[:-2], -1)
            camera_intri = batch["intrinsic_normed_cond"].view(
                *batch["intrinsic_normed_cond"].shape[:-2], -1
            )

            camera_feats = torch.cat([camera_intri, camera_extri], dim=-1)
            # print("camera_feats.shape ", camera_feats.shape)
            # camera_feats = rearrange(camera_feats, 'B Nv C -> (B Nv) C')

            camera_feats = self.camera_embedder(camera_feats)

            # print("camera_feats.shape ", camera_feats.shape)

            # print("img shape ",batch["rgb_cond"].shape)
            encoder_hidden_states: Float[Tensor, "B Nv Cit Nit"] = self.image_tokenizer(
                rearrange(batch["rgb_cond"], "B Nv H W C -> B Nv C H W"),
                modulation_cond=camera_feats,
            )
            # print("img status",encoder_hidden_states.shape)
            encoder_hidden_states = rearrange(
                encoder_hidden_states,
                "B Nv Cit Nit -> B (Nv Nit) Cit",
                Nv=n_input_views,
            )
            # print("img status2",encoder_hidden_states.shape)
            # print("setting camera matrix")


        tokens = self.pointcloud_encoder(batch["pc_patial"].float())


        tokens=tokens
        tokens = self.pointcloud_embedding(batch_size, cond_embeddings=tokens)

        tokens = self.backbone(
            tokens,
            encoder_hidden_states=encoder_hidden_states,
            modulation_cond=None,
        )

        tokens = self.pointcloud_embedding.detokenize(tokens)

        pointclouds = self.post_processor(tokens)

        pointclouds_temp = fps(
            torch.cat((batch["pc_patial"].float(), pointclouds), dim=1), 2048
        )
        # print(pointclouds_temp.shape)
        # pointclouds = pc_down

        # upsampling_input = {
        #     "input_image_tokens": encoder_hidden_states.permute(0, 2, 1),
        #     "input_image_tokens_global": encoder_hidden_states[:, :1],
        #     "c2w_cond": c2w_cond,
        #     "rgb_cond": batch["rgb_cond"],
        #     "intrinsic_cond": batch["intrinsic_cond"],
        #     "intrinsic_normed_cond": batch["intrinsic_normed_cond"],
        #     "points": pointclouds.float()
        # }
        # up_results = self.pointcloud_upsampling(upsampling_input)
        # up_results.insert(0, pointclouds)
        # pointclouds = up_results[-1]
        ###WHERE ARE YOU
        out = {
            # "points": pointclouds,
            # "up_results": up_results,
            "img_cond": encoder_hidden_states,
            "pc_token": tokens,
            "pc": pointclouds,
            "down": pointclouds_temp,
        }
        return out


class SimplePointGenerator2(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        camera_embedder_cls: str = ""
        camera_embedder: dict = field(default_factory=dict)

        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

        text_encoder_cls: str = ""
        text_encoder: str = ""
        # tokenizer_cls: str = ""
        # tokenizer: dict = field(default_factory=dict)

        pointcloud_embedding_cls: str = ""
        pointcloud_embedding: dict = field(default_factory=dict)

        pointcloud_encoder_cls: str = ""
        pointcloud_encoder: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)

        pointcloud_upsampling_cls: str = ""
        pointcloud_upsampling: dict = field(default_factory=dict)

        flip_c2w_cond: bool = True

        # 消融实验：使用 concat + self-attention 代替 cross-attention
        use_concat_self_attn: bool = False
        use_alternating_fusion: bool = False
        projection_dropout: float = 0.0
        alternating_gradient_checkpointing: bool = False
        alternating_enable_memory_efficient_attention: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.image_tokenizer = find(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )

        assert self.cfg.camera_embedder_cls == "mesh_predict.pmp.models.utils.networks.MLP"
        weights = (
            self.cfg.camera_embedder.pop("weights")
            if "weights" in self.cfg.camera_embedder
            else None
        )
        self.camera_embedder = find(self.cfg.camera_embedder_cls)(
            **self.cfg.camera_embedder
        )
        if weights:
            from mesh_predict.pmp.utils.misc import load_module_weights

            weights_path, module_name = weights.split(":")
            state_dict = load_module_weights(
                weights_path, module_name=module_name, map_location="cpu"
            )
            self.camera_embedder.load_state_dict(state_dict)

        self.pointcloud_encoder = find(self.cfg.pointcloud_encoder_cls)(
            self.cfg.pointcloud_encoder
        )

        self.pointcloud_embedding = find(self.cfg.pointcloud_embedding_cls)(
            self.cfg.pointcloud_embedding
        )
        # self.tokenizer=find(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.backbone = None
        if self.cfg.use_concat_self_attn or not self.cfg.use_alternating_fusion:
            self.backbone = find(self.cfg.backbone_cls)(self.cfg.backbone)
        """
            post_processor_cls: tgs.models.networks.PointOutLayer
            post_processor:
            in_channels: 512
            out_channels: 3
        """
        self.text_encoder = find(self.cfg.text_encoder_cls)(self.cfg.text_encoder)

        self.post_processor = find(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )

        self.fc_lyaer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
        )

        adaptive_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.layer_norm = nn.LayerNorm([2048, 512])

        # self.pointcloud_upsampling = find(self.cfg.pointcloud_upsampling_cls)(self.cfg.pointcloud_upsampling)

    def forward(self, batch, encoder_hidden_states=None, **kwargs):

        batch_size, n_input_views = batch["rgb_cond"].shape[:2]
        # print("rgb shape ",batch["rgb_cond"].shape)

        if encoder_hidden_states is None:
            # Camera modulation
            c2w_cond = batch["c2w_cond"].clone()
            if self.cfg.flip_c2w_cond:
                c2w_cond[..., :3, 1:3] *= -1
            camera_extri = c2w_cond.view(*c2w_cond.shape[:-2], -1)
            camera_intri = batch["intrinsic_normed_cond"].view(
                *batch["intrinsic_normed_cond"].shape[:-2], -1
            )

            camera_feats = torch.cat([camera_intri, camera_extri], dim=-1)
            # print("camera_feats.shape ", camera_feats.shape)
            # camera_feats = rearrange(camera_feats, 'B Nv C -> (B Nv) C')

            camera_feats = self.camera_embedder(camera_feats)

            # print("camera_feats.shape ", camera_feats.shape)

            # print("img shape ",batch["rgb_cond"].shape)
            encoder_hidden_states: Float[Tensor, "B Nv Cit Nit"] = self.image_tokenizer(
                rearrange(batch["rgb_cond"], "B Nv H W C -> B Nv C H W"),
                modulation_cond=camera_feats,
            )
            # print("img status",encoder_hidden_states.shape)
            encoder_hidden_states = rearrange(
                encoder_hidden_states,
                "B Nv Cit Nit -> B (Nv Nit) Cit",
                Nv=n_input_views,
            )
            # print("img status2",encoder_hidden_states.shape)
            # print("setting camera matrix")

        text_hidden_states = self.text_encoder(
            image=rearrange(
                batch["rgb_cond"],
                "B Nv H W C -> (B Nv) C H W",
            )
        )

        # print(text_hidden_states.shape)

        text_hidden_states = rearrange(
            text_hidden_states,
            "(B Nv) D -> B  Nv D ",
            Nv=n_input_views,
        )

        text_hidden_states = text_hidden_states.mean(dim=1, keepdim=True)

        tokens = self.pointcloud_encoder(batch["pc_patial"].float())

        text_hidden_states = text_hidden_states.expand(-1, tokens.size(1), -1)

        # temp=self.layer_norm(self.fc_lyaer(torch.concat((tokens, text_hidden_states), dim=2)))

        tokens = self.layer_norm(
            tokens + self.fc_lyaer(torch.concat((tokens, text_hidden_states), dim=2))
        )

        tokens=tokens
        tokens = self.pointcloud_embedding(batch_size, cond_embeddings=tokens)

        # 消融实验：concat + self-attention vs cross-attention
        if self.cfg.use_concat_self_attn:
            # concat + self-attention 方案
            num_pc_tokens = tokens.shape[2]  # tokens: [B, C, N]
            img_tokens = encoder_hidden_states.permute(0, 2, 1)  # [B, N_img, C] -> [B, C, N_img]
            concat_tokens = torch.cat([tokens, img_tokens], dim=2)  # [B, C, N_pc + N_img]

            concat_tokens = self.backbone(
                concat_tokens,
                encoder_hidden_states=None,  # 不用 cross-attention
                modulation_cond=None,
            )
            tokens = concat_tokens[:, :, :num_pc_tokens]  # 只取点云部分
        else:
            # 原始 cross-attention 方案
            tokens = self.backbone(
                tokens,
                encoder_hidden_states=encoder_hidden_states,
                modulation_cond=None,
            )

        tokens = self.pointcloud_embedding.detokenize(tokens)

        pointclouds = self.post_processor(tokens)

        # pointclouds_temp = fps(
        #     torch.cat((batch["pc_patial"].float(), pointclouds), dim=1), 2048
        # )
        # print(pointclouds_temp.shape)
        # pointclouds = pc_down

        # upsampling_input = {
        #     "input_image_tokens": encoder_hidden_states.permute(0, 2, 1),
        #     "input_image_tokens_global": encoder_hidden_states[:, :1],
        #     "c2w_cond": c2w_cond,
        #     "rgb_cond": batch["rgb_cond"],
        #     "intrinsic_cond": batch["intrinsic_cond"],
        #     "intrinsic_normed_cond": batch["intrinsic_normed_cond"],
        #     "points": pointclouds.float()
        # }
        # up_results = self.pointcloud_upsampling(upsampling_input)
        # up_results.insert(0, pointclouds)
        # pointclouds = up_results[-1]
        ###WHERE ARE YOU
        out = {
            # "points": pointclouds,
            # "up_results": up_results,
            "img_cond": encoder_hidden_states,
            "pc_token": tokens,
            "pc": pointclouds,
            "down": pointclouds,
        }
        return out


class TextFusionPointGenerator(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        camera_embedder_cls: str = ""
        camera_embedder: dict = field(default_factory=dict)

        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

        text_encoder_cls: str = ""
        text_encoder: str = ""
        # tokenizer_cls: str = ""
        # tokenizer: dict = field(default_factory=dict)

        pointcloud_embedding_cls: str = ""
        pointcloud_embedding: dict = field(default_factory=dict)

        pointcloud_encoder_cls: str = ""
        pointcloud_encoder: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)

        pointcloud_upsampling_cls: str = ""
        pointcloud_upsampling: dict = field(default_factory=dict)

        flip_c2w_cond: bool = True

        # 消融实验：使用 concat + self-attention 代替 cross-attention
        use_concat_self_attn: bool = False
        use_alternating_fusion: bool = False
        projection_dropout: float = 0.0
        alternating_gradient_checkpointing: bool = False
        alternating_enable_memory_efficient_attention: bool = False
        enable_upsampling: bool = False
        detach_upsampling_input: bool = True
        upsampling_repeat_levels: int = 1
        upsampling_jitter_sigma: float = 0.0

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.image_tokenizer = find(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )

        assert self.cfg.camera_embedder_cls == "mesh_predict.pmp.models.utils.networks.MLP"
        weights = (
            self.cfg.camera_embedder.pop("weights")
            if "weights" in self.cfg.camera_embedder
            else None
        )
        self.camera_embedder = find(self.cfg.camera_embedder_cls)(
            **self.cfg.camera_embedder
        )
        if weights:
            from mesh_predict.pmp.utils.misc import load_module_weights

            weights_path, module_name = weights.split(":")
            state_dict = load_module_weights(
                weights_path, module_name=module_name, map_location="cpu"
            )
            self.camera_embedder.load_state_dict(state_dict)

        self.pointcloud_encoder = find(self.cfg.pointcloud_encoder_cls)(
            self.cfg.pointcloud_encoder
        )

        self.pointcloud_embedding = find(self.cfg.pointcloud_embedding_cls)(
            self.cfg.pointcloud_embedding
        )
        # self.tokenizer=find(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.backbone = None
        if self.cfg.use_concat_self_attn or not self.cfg.use_alternating_fusion:
            self.backbone = find(self.cfg.backbone_cls)(self.cfg.backbone)
        """
            post_processor_cls: tgs.models.networks.PointOutLayer
            post_processor:
            in_channels: 512
            out_channels: 3
        """
        self.text_encoder = find(self.cfg.text_encoder_cls)(self.cfg.text_encoder)

        self.post_processor = find(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )

        self.pointcloud_upsampling = None
        self.image_projector = None
        self.text_projector = None
        self.point_projector = None
        self.multimodal_fuser = None
        if self.cfg.pointcloud_upsampling_cls:
            self.pointcloud_upsampling = find(self.cfg.pointcloud_upsampling_cls)(
                self.cfg.pointcloud_upsampling
            )
        if self.cfg.use_alternating_fusion:
            hidden_dim = self.cfg.backbone.get("in_channels", 512)
            num_heads = self.cfg.backbone.get("num_attention_heads", 8)
            num_layers = self.cfg.backbone.get("num_layers", 10)
            dropout = self.cfg.backbone.get("dropout", 0.0)
            semantic_input_dim = self.cfg.backbone.get("cross_attention_dim")
            if semantic_input_dim is None:
                semantic_input_dim = self.cfg.backbone.get(
                    "gated_cross_attention_dim", hidden_dim
                )
            point_input_dim = self.cfg.pointcloud_encoder.get("pt_dim", hidden_dim)
            self.image_projector = FeatureProjectionMLP(
                semantic_input_dim,
                hidden_dim,
                dropout=self.cfg.projection_dropout,
            )
            self.text_projector = FeatureProjectionMLP(
                semantic_input_dim,
                hidden_dim,
                dropout=self.cfg.projection_dropout,
            )
            self.point_projector = FeatureProjectionMLP(
                point_input_dim,
                hidden_dim,
                dropout=self.cfg.projection_dropout,
            )
            self.multimodal_fuser = AlternatingAttentionFusion(
                dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                gradient_checkpointing=self.cfg.alternating_gradient_checkpointing,
                enable_memory_efficient_attention=(
                    self.cfg.alternating_enable_memory_efficient_attention
                ),
            )

    def _encode_image_tokens(self, batch, n_input_views, encoder_hidden_states=None):
        if encoder_hidden_states is not None:
            return encoder_hidden_states

        c2w_cond = batch["c2w_cond"].clone()
        if self.cfg.flip_c2w_cond:
            c2w_cond[..., :3, 1:3] *= -1
        camera_extri = c2w_cond.view(*c2w_cond.shape[:-2], -1)
        camera_intri = batch["intrinsic_normed_cond"].view(
            *batch["intrinsic_normed_cond"].shape[:-2], -1
        )
        camera_feats = torch.cat([camera_intri, camera_extri], dim=-1)
        camera_feats = self.camera_embedder(camera_feats)
        encoder_hidden_states = self.image_tokenizer(
            rearrange(batch["rgb_cond"], "B Nv H W C -> B Nv C H W"),
            modulation_cond=camera_feats,
        )
        return rearrange(
            encoder_hidden_states,
            "B Nv Cit Nit -> B (Nv Nit) Cit",
            Nv=n_input_views,
        )

    def _encode_text_tokens(self, batch):
        if "text_promt" in batch.keys():
            text_hidden_states = self.text_encoder(text=batch["text_promt"])
            n_text_input_views = 1
        elif "text_cond" in batch.keys():
            text_hidden_states = self.text_encoder(text=batch["text_cond"])
            n_text_input_views = 1
        elif "text_view_cond" in batch.keys():
            text_hidden_states = self.text_encoder(
                image=rearrange(
                    batch["text_view_cond"],
                    "B Nv H W C -> (B Nv) C H W",
                )
            )
            _, n_text_input_views = batch["text_view_cond"].shape[:2]
        else:
            raise KeyError(
                "TextFusionPointGenerator requires `text_promt`, `text_cond`, or "
                "`text_view_cond`."
            )

        text_hidden_states = rearrange(
            text_hidden_states,
            "(B Nv) D -> B Nv D ",
            Nv=n_text_input_views,
        )
        return text_hidden_states.mean(dim=1, keepdim=True)

    def forward(self, batch, encoder_hidden_states=None, **kwargs):
        batch_size, n_input_views = batch["rgb_cond"].shape[:2]
        raw_image_tokens = self._encode_image_tokens(
            batch,
            n_input_views=n_input_views,
            encoder_hidden_states=encoder_hidden_states,
        )
        text_hidden_states = self._encode_text_tokens(batch)
        raw_semantic_tokens = torch.cat((raw_image_tokens, text_hidden_states), dim=1)

        point_features = self.pointcloud_encoder(batch["pc_patial"].float())
        tokens = self.pointcloud_embedding(
            batch_size,
            cond_embeddings=(
                self.point_projector(point_features)
                if self.cfg.use_alternating_fusion
                else point_features
            ),
        )

        # 消融实验：concat + self-attention vs cross-attention
        if self.cfg.use_concat_self_attn:
            # concat + self-attention 方案
            num_pc_tokens = tokens.shape[2]  # tokens: [B, C, N]
            semantic_tokens_cf = raw_semantic_tokens.permute(0, 2, 1)
            concat_tokens = torch.cat([tokens, semantic_tokens_cf], dim=2)

            concat_tokens = self.backbone(
                concat_tokens,
                encoder_hidden_states=None,  # 不用 cross-attention
                modulation_cond=None,
            )
            tokens = concat_tokens[:, :, :num_pc_tokens]  # 只取点云部分
            semantic_tokens_out = raw_semantic_tokens
        elif self.cfg.use_alternating_fusion:
            image_tokens = self.image_projector(raw_image_tokens)
            text_tokens = self.text_projector(text_hidden_states)
            semantic_tokens = torch.cat((image_tokens, text_tokens), dim=1)
            geom_tokens = tokens.permute(0, 2, 1)
            geom_tokens, semantic_tokens_out = self.multimodal_fuser(
                geom_tokens, semantic_tokens
            )
            tokens = geom_tokens.permute(0, 2, 1)
        else:
            tokens = self.backbone(
                tokens,
                encoder_hidden_states=raw_semantic_tokens,
                modulation_cond=None,
            )
            semantic_tokens_out = raw_semantic_tokens

        tokens_op = self.pointcloud_embedding.detokenize(tokens)

        pointclouds = self.post_processor(tokens_op)

        pointclouds_temp = fps(
            torch.cat((batch["pc_patial"].float(), pointclouds), dim=1), 2048
        )
        
        # print(pointclouds_temp.shape)
        # pointclouds = pointclouds_temp

        up_points = pointclouds
        up_tokens = tokens
        up_text_tokens = text_hidden_states
        if self.cfg.detach_upsampling_input:
            up_points = up_points.detach()
            up_tokens = up_tokens.detach()
            up_text_tokens = up_text_tokens.detach()

        upsampling_input = {
            "input_text_tokens": up_text_tokens.float(),
            "points": up_points.float(),
            "pcl_token" :up_tokens.float()
        }
        up_results = []
        if self.cfg.enable_upsampling and self.pointcloud_upsampling is not None:
            up_results = repeated_first_level_upsampling(
                self.pointcloud_upsampling,
                upsampling_input,
                repeat_levels=self.cfg.upsampling_repeat_levels,
                jitter_sigma=self.cfg.upsampling_jitter_sigma,
            )

        # up_results.insert(0, pointclouds)
        # pointclouds = up_results[-1]
        ###WHERE ARE YOU
        out = {
            # "points": pointclouds,
            "up_results": up_results,
            "img_cond": semantic_tokens_out,
            "pc_token": tokens,
            "pc": pointclouds,
            "down": pointclouds,
            "pc_upsampled": up_results[-1] if len(up_results) > 0 else pointclouds,
            # "up_results" :up_results
        }
        return out


class PointGenerator(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        # camera_embedder_cls: str = ""
        # camera_embedder: dict = field(default_factory=dict)

        # image_tokenizer_cls: str = ""
        # image_tokenizer: dict = field(default_factory=dict)

        text_encoder_cls: str = ""
        text_encoder: str = ""
        # tokenizer_cls: str = ""
        # tokenizer: dict = field(default_factory=dict)

        pointcloud_embedding_cls: str = ""
        pointcloud_embedding: dict = field(default_factory=dict)

        pointcloud_encoder_cls: str = ""
        pointcloud_encoder: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)

        pointcloud_upsampling_cls: str = ""
        pointcloud_upsampling: dict = field(default_factory=dict)

        flip_c2w_cond: bool = True

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # self.image_tokenizer = find(self.cfg.image_tokenizer_cls)(
        #     self.cfg.image_tokenizer
        # )

        # assert self.cfg.camera_embedder_cls == "mesh_predict.pmp.models.utils.networks.MLP"
        # weights = (
        #     self.cfg.camera_embedder.pop("weights")
        #     if "weights" in self.cfg.camera_embedder
        #     else None
        # )
        # self.camera_embedder = find(self.cfg.camera_embedder_cls)(
        #     **self.cfg.camera_embedder
        # )
        # if weights:
        #     from mesh_predict.pmp.utils.misc import load_module_weights

        #     weights_path, module_name = weights.split(":")
        #     state_dict = load_module_weights(
        #         weights_path, module_name=module_name, map_location="cpu"
        #     )
        #     self.camera_embedder.load_state_dict(state_dict)

        self.pointcloud_encoder = find(self.cfg.pointcloud_encoder_cls)(
            self.cfg.pointcloud_encoder
        )

        self.pointcloud_embedding = find(self.cfg.pointcloud_embedding_cls)(
            self.cfg.pointcloud_embedding
        )
        # self.tokenizer=find(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.backbone = find(self.cfg.backbone_cls)(self.cfg.backbone)
        """
            post_processor_cls: tgs.models.networks.PointOutLayer
            post_processor:
            in_channels: 512
            out_channels: 3
        """
        self.text_encoder = find(self.cfg.text_encoder_cls)(self.cfg.text_encoder)

        self.post_processor = find(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )

        # self.pointcloud_upsampling = find(self.cfg.pointcloud_upsampling_cls)(self.cfg.pointcloud_upsampling)

    def forward(self, batch, encoder_hidden_states=None, **kwargs):
        batch_size, n_input_views = batch["rgb_cond"].shape[:2]

        
        # print("rgb shape ",batch["rgb_cond"].shape)

        # if encoder_hidden_states is None:
        #     # Camera modulation
        #     c2w_cond = batch["c2w_cond"].clone()
        #     if self.cfg.flip_c2w_cond:
        #         c2w_cond[..., :3, 1:3] *= -1
        #     camera_extri = c2w_cond.view(*c2w_cond.shape[:-2], -1)
        #     camera_intri = batch["intrinsic_normed_cond"].view(
        #         *batch["intrinsic_normed_cond"].shape[:-2], -1
        #     )

        #     camera_feats = torch.cat([camera_intri, camera_extri], dim=-1)
        #     # print("camera_feats.shape ", camera_feats.shape)
        #     # camera_feats = rearrange(camera_feats, 'B Nv C -> (B Nv) C')
        #     # print("camera_feats2.shape ", camera_feats.shape)
        #     camera_feats = self.camera_embedder(camera_feats)

        #     # print("camera_feats.shape ", camera_feats.shape)

        #     # print("img shape ",batch["rgb_cond"].shape)
        #     encoder_hidden_states: Float[Tensor, "B Nv Cit Nit"] = self.image_tokenizer(
        #         rearrange(batch["rgb_cond"], "B Nv H W C -> B Nv C H W"),
        #         modulation_cond=camera_feats,
        #     )
        #     # print("img status",encoder_hidden_states.shape)
        #     encoder_hidden_states = rearrange(
        #         encoder_hidden_states,
        #         "B Nv Cit Nit -> B (Nv Nit) Cit",
        #         Nv=n_input_views,
        #     )
        #     # print("img status2",encoder_hidden_states.shape)
        #     # print("setting camera matrix")
        if "text_view_cond" in batch.keys():

            text_hidden_states = self.text_encoder(
                image=rearrange(
                    batch["text_view_cond"],
                    "B Nv H W C -> (B Nv) C H W",
                )
            )
            _,n_text_input_views= batch["text_view_cond"].shape[:2]
        elif "text_cond" in batch.keys():

            text_hidden_states = self.text_encoder(
                text=batch["text_cond"])
            n_text_input_views=1


        # print(text_hidden_states.shape)

        text_hidden_states = rearrange(
            text_hidden_states,
            "(B Nv) D -> B  Nv D ",
            Nv=n_text_input_views,
        )

        text_hidden_states = text_hidden_states.mean(dim=1, keepdim=True)

        # print("text_hidden_states shape ",text_hidden_states.shape,"  encoder_hidden_states ",encoder_hidden_states.shape)
        # encoder_hidden_states = torch.concat((encoder_hidden_states,text_hidden_states),dim=1)


        tokens = self.pointcloud_encoder(batch["pc_patial"].float())
        tokens = self.pointcloud_embedding(batch_size, cond_embeddings=tokens)

        tokens = self.backbone(
            tokens,
            encoder_hidden_states=None,
            modulation_cond=None,
            # cross_attention_kwargs = cross_attention_kwargs_for_gligen
            gated_encoder_hidden_states =text_hidden_states
        )


        tokens_op = self.pointcloud_embedding.detokenize(tokens)

        pointclouds = self.post_processor(tokens_op)

        # pointclouds_temp = fps(
        #     torch.cat((batch["pc_patial"].float(), pointclouds), dim=1), 512
        # )
        # print(pointclouds_temp.shape)
        # pointclouds = pc_down

        up_results = None
        # upsampling_input = {
        #     "input_text_tokens": text_hidden_states.float(),
        #     "points": pointclouds.float(),
        #     "pcl_token" :tokens.float()
        # }
        # up_results = self.pointcloud_upsampling(upsampling_input)
        # print("up_reslut",)
        # up_results.insert(0, pointclouds)
        # pointclouds = up_results[-1]
        ###WHERE ARE YOU
        out = {
            # "points": pointclouds,
            "up_results": up_results,
            # "img_cond": encoder_hidden_states,
            "pc_token": tokens,
            "pc": pointclouds,
            "down": pointclouds,
            # "up_results" :up_results
        }
        return out
