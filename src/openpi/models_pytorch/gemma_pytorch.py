from typing import Literal

import pytest
import torch
from torch import Tensor
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma
from openpi.models.cross_view_config import CrossViewFusionConfig
from openpi.models_pytorch.layers.block import Block
from openpi.models_pytorch.layers.layer_scale import LayerScale
from openpi.models_pytorch.layers.mlp import Mlp
from openpi.models_pytorch.layers.prope import prepare_apply_fns as _prope_prepare_apply_fns
from openpi.models_pytorch.layers.rope import PositionGetter
from openpi.models_pytorch.layers.rope import RotaryPositionEmbedding2D
from openpi.models_pytorch.utils import get_pixel


class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B, C, H, W) -> (B, N, D).

    Cleaned-up port of pi3's DINOv2 PatchEmbed.
    """

    def __init__(
        self,
        patch_size: int | tuple[int, int] = 14,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: type[nn.Module] | None = None,
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        pH, pW = self.patch_size
        assert H % pH == 0, f"Input height {H} is not a multiple of patch height {pH}"
        assert W % pW == 0, f"Input width {W} is not a multiple of patch width {pW}"

        x = self.proj(x)  # (B, D, Hp, Wp)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        x = self.norm(x)
        return x


def se3_inverse(T: Tensor) -> Tensor:
    """Compute the inverse of a batch of SE(3) matrices exploiting R orthogonality."""
    R = T[..., :3, :3]
    t = T[..., :3, 3:]
    R_inv = R.transpose(-2, -1)
    t_inv = -torch.matmul(R_inv, t)
    T_inv = torch.zeros_like(T)
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3:] = t_inv
    T_inv[..., 3, 3] = 1.0
    return T_inv


class PRoPEAttention(nn.Module):
    """Self-attention with PRoPE-style Q/K/V transforms driven by camera extrinsics+intrinsics.

    Mirrors pi3's PRopeFlashAttention (AttentionRope subclass).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        qk_norm: bool = False,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        freq_base: float = 100.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.freq_base = freq_base

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(
        self,
        x: Tensor,
        extrinsics: Tensor,
        image_h: int,
        image_w: int,
        patch_h: int,
        patch_w: int,
        Ks: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)

        apply_q, apply_kv, apply_o = _prope_prepare_apply_fns(
            head_dim=self.head_dim,
            viewmats=extrinsics,
            Ks=Ks,
            patches_x=patch_w,
            patches_y=patch_h,
            image_width=image_w,
            image_height=image_h,
            freq_base=self.freq_base,
        )
        q = apply_q(q)
        k = apply_kv(k)
        v = apply_kv(v)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = apply_o(out)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class PoseInjectBlock(nn.Module):
    """Transformer block with PRoPE attention for camera-pose injection.

    Mirrors pi3's PoseInjectBlock. By default (connect=False), the block does NOT
    add a residual from the input — the output is purely from attention + FFN.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        qk_norm: bool = True,
        init_values: float | None = 0.01,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        freq_base: float = 100.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PRoPEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
            freq_base=freq_base,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden, act_layer=act_layer, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(
        self,
        x: Tensor,
        poses: Tensor,
        image_h: int,
        image_w: int,
        patch_h: int,
        patch_w: int,
        Ks: Tensor | None = None,
        connect: bool = False,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        # poses are world->camera; invert to get camera->world extrinsics
        extrinsics = se3_inverse(poses)
        import pdb; pdb.set_trace()

        def attn_residual(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), extrinsics, image_h, image_w, patch_h, patch_w, Ks, attn_mask))

        def ffn_residual(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if connect:
            return x + attn_residual(x) + ffn_residual(x)
        return attn_residual(x) + ffn_residual(x)


class CrossViewFusion(nn.Module):
    def __init__(
        self,
        aa_order: str = "fg",
        embed_dim: int = 1024,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        rope_freq: int = 100,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values: float = 0.01,
        prope_layer_idx: tuple[int, ...] = (),
        prope_freq_base: float = 100.0,
    ):
        super().__init__()
        self.aa_order = aa_order
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.prope_after_global = set(prope_layer_idx)

        self.frame_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for attn_type in aa_order
                if attn_type == "f"
            ]
        )
        self.global_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for attn_type in aa_order
                if attn_type == "g"
            ]
        )

        # PRoPE pose-injection blocks, one per entry in prope_layer_idx
        if self.prope_after_global:
            self.pose_inject_blocks = nn.ModuleList(
                [
                    PoseInjectBlock(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        freq_base=prope_freq_base,
                    )
                    for _ in prope_layer_idx
                ]
            )

    def forward(
        self,
        tokens,
        mask=None,
        pos=None,
        poses: Tensor | None = None,
        image_h: int = 0,
        image_w: int = 0,
        patch_h: int = 0,
        patch_w: int = 0,
        Ks: Tensor | None = None,
    ):  
        import pdb; pdb.set_trace()
        bsz, num_views, num_tokens, dim = tokens.shape

        frame_idx = 0
        global_idx = 0
        pose_inject_idx = 0
        for attn_type in self.aa_order:
            if attn_type == "f":
                tokens = self._process_frame_attention(tokens, bsz, num_views, num_tokens, dim, frame_idx, pos=pos, mask=mask)
                frame_idx += 1
            elif attn_type == "g":
                tokens = self._process_global_attention(tokens, bsz, num_views, num_tokens, dim, global_idx, pos=pos, mask=mask)
                # Inject PRoPE after this global block if configured
                if global_idx in self.prope_after_global and poses is not None:
                    tokens = tokens.reshape(bsz, num_views, num_tokens, dim)
                    pose_feat = self.pose_inject_blocks[pose_inject_idx](
                        tokens.reshape(bsz, num_views * num_tokens, dim),
                        poses, image_h, image_w, patch_h, patch_w, Ks=Ks,
                    )
                    tokens = tokens + pose_feat.reshape(bsz, num_views, num_tokens, dim)
                    pose_inject_idx += 1
                global_idx += 1
            else:
                raise ValueError(f"Unknown attention type: {attn_type}")

        return tokens.reshape(bsz, num_views, num_tokens, dim)

    def _process_frame_attention(self, tokens, bsz, num_views, num_tokens, dim, layer_idx, pos=None, mask=None):
        if tokens.shape != (bsz * num_views, num_tokens, dim):
            tokens = tokens.reshape(bsz, num_views, num_tokens, dim).reshape(bsz * num_views, num_tokens, dim)
        if pos is not None and pos.shape != (bsz * num_views, num_tokens, 2):
            pos = pos.reshape(bsz, num_views, num_tokens, 2).reshape(bsz * num_views, num_tokens, 2)

        frame_mask = ~mask.reshape(bsz * num_views, num_tokens) if mask is not None else None
        return self.frame_blocks[layer_idx](tokens, pos=pos, attn_mask=frame_mask)

    def _process_global_attention(self, tokens, bsz, num_views, num_tokens, dim, layer_idx, pos=None, mask=None):
        if tokens.shape != (bsz, num_views * num_tokens, dim):
            tokens = tokens.reshape(bsz, num_views, num_tokens, dim).reshape(bsz, num_views * num_tokens, dim)
        if pos is not None and pos.shape != (bsz, num_views * num_tokens, 2):
            pos = pos.reshape(bsz, num_views, num_tokens, 2).reshape(bsz, num_views * num_tokens, 2)

        global_mask = ~mask.reshape(bsz, num_views * num_tokens) if mask is not None else None
        return self.global_blocks[layer_idx](tokens, pos=pos, attn_mask=global_mask)


class SimpleCrossViewFusion(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()

        self.norm1 = modeling_gemma.GemmaRMSNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        self.norm2 = modeling_gemma.GemmaRMSNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x, mask=None, pos=None):
        del pos
        bsz, num_views, num_tokens, dim = x.shape
        x = x.reshape(bsz, num_views * num_tokens, dim)

        x_norm, _ = self.norm1(x)
        key_padding_mask = (mask == 0).reshape(bsz, num_views * num_tokens) if mask is not None else None
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = x + attn_out

        x_norm, _ = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x.reshape(bsz, num_views, num_tokens, dim)


class CamPoseEncoder(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, pose):
        # pose: (B, 9) or (B, V, 9)

        if pose.dim() == 3:
            B, V, D = pose.shape
            pose = pose.reshape(B * V, D)
            out = self.mlp(pose)
            return out.reshape(B, V, -1)

        return self.mlp(pose)


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        *,
        pose_enc_type: Literal["null", "relative_pose", "absolute_pose", "prope"] = "null",
        ray_enc_type: bool = False,
        view_enc_type: bool = False,
        cross_view_config: CrossViewFusionConfig = CrossViewFusionConfig(),
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)  # vlm (vision + gemma LLM)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)  # action expert
        self.gemma_expert.model.embed_tokens = None

        self.cross_view_config = cross_view_config
        self.pose_enc_type = pose_enc_type
        self.ray_enc_type = ray_enc_type
        self.view_enc_type = view_enc_type
        self.num_views = 3  # Pi-0: agent + wrist + wrist2

        if view_enc_type:
            self.view_embedding = nn.Embedding(
                num_embeddings=self.num_views,
                embedding_dim=vlm_config_hf.vision_config.hidden_size,  # must match token dim D
            )

        if self.cross_view_config.type != "none":
            self.position_getter = PositionGetter() if self.cross_view_config.rope_freq > 0 else None

            if self.cross_view_config.type == "standard":
                self.cross_view_fusion = CrossViewFusion(
                    embed_dim=vlm_config_hf.vision_config.hidden_size,
                    aa_order=self.cross_view_config.aa_order,
                    rope_freq=self.cross_view_config.rope_freq,
                    init_values=self.cross_view_config.init_values,
                    num_heads=self.cross_view_config.num_heads,
                    mlp_ratio=self.cross_view_config.mlp_ratio,
                    qk_norm=self.cross_view_config.qk_norm,
                    prope_layer_idx=self.cross_view_config.prope_layer_idx if pose_enc_type == "prope" else (),
                    prope_freq_base=self.cross_view_config.rope_freq,
                )
            elif self.cross_view_config.type == "simple":
                self.cross_view_fusion = SimpleCrossViewFusion(
                    embed_dim=vlm_config_hf.vision_config.hidden_size,
                    num_heads=self.cross_view_config.num_heads,
                    mlp_ratio=self.cross_view_config.mlp_ratio,
                )
            else:
                raise ValueError(f"Unsupported cross_view type: {self.cross_view_config.type}")
        if pose_enc_type in ("relative_pose", "absolute_pose"):
            self.cam_pose_encoder = CamPoseEncoder(vlm_config_hf.vision_config.hidden_size)

        if ray_enc_type:
            vision_patch_size = vlm_config_hf.vision_config.patch_size
            vision_dim = vlm_config_hf.vision_config.hidden_size
            self.ray_embed = PatchEmbed(patch_size=vision_patch_size, in_chans=2, embed_dim=vision_dim)
            nn.init.constant_(self.ray_embed.proj.weight, 0)
            nn.init.constant_(self.ray_embed.proj.bias, 0)

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(
        self,
        images: torch.Tensor,
        img_masks: torch.Tensor,
        cam_pos: dict | None = None,
        cam_intr: dict | None = None,
        cam_keys: list | None = None,
    ):
        B, V, C, H, W = images.shape
        device = images.device

        vision_tower = self.paligemma.model.vision_tower
        projector = self.paligemma.model.multi_modal_projector

        images = images.reshape(B * V, C, H, W)
        tokens = vision_tower(images).last_hidden_state  # [B*V, P, D]
        P, D = tokens.shape[1], tokens.shape[2]
        tokens = tokens.reshape(B, V, P, D)
        masks = img_masks[:, :, None].expand(B, V, P).clone()
        import pdb; pdb.set_trace()
        # -------- helpers --------
        def encode_pose(pose):
            R = pose[..., :3, :3]
            t = pose[..., :3, 3]
            rot6d = R[..., :, :2].reshape(*R.shape[:-2], -1)  # (B,6)
            return torch.cat([t, rot6d], dim=-1).to(device=tokens.device, dtype=tokens.dtype)  # (B,9)

        def make_null_token() -> torch.Tensor:
            return torch.zeros(B, 1, D, device=tokens.device, dtype=tokens.dtype)

        def compute_rel_pose(agent_T, wrist_T):
            return torch.linalg.inv(agent_T) @ wrist_T  # (B,4,4)

        def stack_per_cam(cam_map: dict | None, eye_dim: int) -> torch.Tensor | None:
            """Stack a {cam_key: (B, D, D) | None} dict to (B, V, D, D) along cam_keys,
            filling missing entries with identity. Returns None if cam_map is None."""
            if cam_map is None:
                return None
            out = []
            for cam_key in cam_keys:
                T = cam_map.get(cam_key)
                if T is None:
                    T = torch.eye(eye_dim, device=device, dtype=tokens.dtype)[None].expand(B, -1, -1)
                out.append(T)
            return torch.stack(out, dim=1)

        # -------- ray direction embedding (from intrinsics) --------
        if self.ray_enc_type and cam_intr is not None:
            # (B, V, 3, 3) — compute pixel grid, back-project, keep (x, y)
            cam_intr_stacked = stack_per_cam(cam_intr, 3)
            pix = torch.from_numpy(get_pixel(H, W).T.reshape(H, W, 3)).to(device=device, dtype=tokens.dtype)
            pix = pix[None].expand(B, -1, -1, -1)  # (B, H, W, 3)
            K_inv = torch.linalg.inv(cam_intr_stacked.float()).to(tokens.dtype)  # invert in fp32 for stability
            rays = torch.einsum("bvij, bhwj -> bvhwi", K_inv, pix)[..., :2]  # (B, V, H, W, 2)
            ray_emb = self.ray_embed(rays.reshape(B * V, H, W, 2).permute(0, 3, 1, 2))  # (B*V, P, D)
            ray_emb = ray_emb.reshape(B, V, P, D)
            tokens = tokens + ray_emb

        # --------- camera token injection ---------
        if self.pose_enc_type in ["relative_pose", "absolute_pose"]:
            cam_tokens = []

            if self.pose_enc_type == "relative_pose":
                agent_T = cam_pos["base"]  # (B,4,4)

                for cam_key in cam_keys:
                    if cam_key == "base" or cam_pos.get(cam_key) is None:
                        cam_tokens.append(make_null_token())
                    elif cam_pos.get(cam_key) is not None:
                        wrist_T = compute_rel_pose(agent_T, cam_pos[cam_key])  # (B,4,4)
                        cam_token = self.cam_pose_encoder(encode_pose(wrist_T)).unsqueeze(1)
                        cam_tokens.append(cam_token)

            elif self.pose_enc_type == "absolute_pose":
                for cam_key in cam_keys:
                    if cam_pos[cam_key] is not None:
                        cam_token = self.cam_pose_encoder(encode_pose(cam_pos[cam_key])).unsqueeze(1)
                        cam_tokens.append(cam_token)
                    else:
                        cam_tokens.append(make_null_token())
            cam_tokens = torch.stack(cam_tokens, dim=1)  # (B,V,1,D)
            tokens = torch.cat([cam_tokens, tokens], dim=2)  # [B, V, P+1, D]
            P = P + 1

            # ------ update the image masks ------
            pose_valid = img_masks[:, :, None]
            masks = torch.cat([pose_valid, masks], dim=-1)
            
        elif self.pose_enc_type == "prope" and self.cross_view_config.type != "none":
            prope_poses = stack_per_cam(cam_pos, 4)  # (B, V, 4, 4)

        # -------- view embedding --------
        if self.view_enc_type:
            view_ids = torch.arange(V, device=tokens.device)
            view_embed = self.view_embedding(view_ids)
            tokens = tokens + view_embed[None, :, None, :]

        # -------- cross-view fusion --------
        if self.cross_view_config.type != "none":
            pos = None
            if self.cross_view_config.rope_freq > 0:
                patch_size = vision_tower.config.patch_size
                pos = self.position_getter(B * V, H // patch_size, W // patch_size, device=tokens.device)
                num_special = 1 if self.pose_enc_type not in ("null", "prope") else 0
                if num_special >= 1:
                    pos = pos + 1
                    pos_special = torch.zeros(B * V, num_special, 2, device=tokens.device, dtype=pos.dtype)
                    pos = torch.cat([pos_special, pos], dim=1)

            # Pass PRoPE args when pose_enc_type == "prope" (extrinsics only)
            prope_kwargs = {}
            if self.pose_enc_type == "prope":
                patch_size = vision_tower.config.patch_size
                prope_kwargs = dict(
                    poses=prope_poses,
                    image_h=H,
                    image_w=W,
                    patch_h=H // patch_size,
                    patch_w=W // patch_size,
                )

            tokens = self.cross_view_fusion(tokens, mask=masks, pos=pos, **prope_kwargs)

        # Post-fusion, pre-projector patch tokens for downstream auxiliary heads.
        # Strip the leading cam_token (when pose_enc_type prepends one) so consumers
        # see exactly patch_h*patch_w tokens.
        num_special = 1 if self.pose_enc_type in ("relative_pose", "absolute_pose") else 0
        fused_tokens = tokens[:, :, num_special:, :]  # (B, V, patch_h*patch_w, D)

        # -------- projector --------
        tokens = tokens.reshape(B * V, P, D)
        tokens = projector(tokens)
        return tokens.reshape(B, V * P, -1), masks.reshape(B, V * P), fused_tokens

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        # --------- case 1: prefix only ---------
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        # --------- case 2: suffix only ---------
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        # --------- case 3: full joint forward ---------
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []

                # --------------- Step 1: Compute Q/K/V separately for prefix + suffix ---------------
                for i, hidden_states in enumerate(inputs_embeds):  # loop over 0: prefix, 1: suffix
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                # ----------------Step 2: Apply Rotary Positional Embedding (RoPE) to Q/K  ----------------
                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                # ----------------Step 3: Compute attention (BOTH prefix + suffix)  ----------------
                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # ----------------Step 4: Split attn outputs back into prefix/suffix tokens  ----------------
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # ---- first residual ----
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)
                    # ---- FFN MLP block  ----
                    out_emb = layer.mlp(out_emb)
                    # ---- second residual ----
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001

                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # ===================== Process all layers with gradient checkpointing if enabled =====================
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # ===================== final norm layer  =====================
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            # ----------- split output back to prefix/suffix -----------
            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
