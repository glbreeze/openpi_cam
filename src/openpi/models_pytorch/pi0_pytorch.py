import logging
import math
import os

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.layers.point_head import PointHead
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def _local_points_from_xy_logz(xy: Tensor, logz: Tensor) -> tuple[Tensor, Tensor]:
    logz = torch.nan_to_num(logz, nan=0.0, posinf=15.0, neginf=-30.0)
    depth = torch.exp(logz.clamp(max=15.0))
    return torch.cat([xy * depth, depth], dim=-1), depth


def _weighted_mean(
    values: Tensor,
    weights: Tensor,
    dim: tuple[int, ...],
    *,
    keepdim: bool,
    eps: float = 1e-6,
) -> Tensor:
    denom = weights.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
    return (values * weights).sum(dim=dim, keepdim=keepdim) / denom


def _subsample_alignment_inputs(
    pred_points: Tensor,
    target_points: Tensor,
    weights: Tensor,
    max_points: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if max_points <= 0 or pred_points.shape[1] <= max_points:
        return pred_points, target_points, weights

    pred_out, target_out, weight_out = [], [], []
    for pred_i, target_i, weight_i in zip(pred_points, target_points, weights, strict=True):
        valid_idx = torch.nonzero(weight_i > 0, as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            idx = torch.zeros(max_points, dtype=torch.long, device=weights.device)
            pred_out.append(pred_i[idx])
            target_out.append(target_i[idx])
            weight_out.append(torch.zeros(max_points, dtype=weight_i.dtype, device=weight_i.device))
            continue

        sample_idx = torch.div(
            torch.arange(max_points, device=weights.device) * valid_idx.numel(),
            max_points,
            rounding_mode="floor",
        ).clamp_max(valid_idx.numel() - 1)
        idx = valid_idx[sample_idx]
        pred_out.append(pred_i[idx])
        target_out.append(target_i[idx])
        weight_out.append(weight_i[idx])

    return torch.stack(pred_out, dim=0), torch.stack(target_out, dim=0), torch.stack(weight_out, dim=0)


@torch.no_grad()
def _align_points_scale_l1(pred_points: Tensor, target_points: Tensor, weights: Tensor, max_points: int) -> Tensor:
    """Pi3X-style positive scalar alignment for local pointmaps.

    Solves the same weighted L1 scale problem as Pi3X's `align_points_scale`, but
    on an optional deterministic subsample to keep full-resolution training cheap.
    """

    bsz = pred_points.shape[0]
    pred_points = pred_points.reshape(bsz, -1, 3)
    target_points = target_points.reshape(bsz, -1, 3)
    weights = weights.reshape(bsz, -1)
    pred_points, target_points, weights = _subsample_alignment_inputs(
        pred_points,
        target_points,
        weights,
        max_points,
    )

    x = pred_points.flatten(1)
    y = target_points.flatten(1)
    w = weights[..., None].expand_as(pred_points).flatten(1)
    finite = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(w) & (w > 0)
    w = torch.where(finite, w, torch.zeros_like(w))
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    eps = 1e-7
    sign = torch.sign(x)
    x = x * sign
    y = y * sign
    y_div_x = y / x.clamp_min(eps)
    y_div_x, argsort = y_div_x.sort(dim=-1)

    wx = torch.gather(x * w, dim=-1, index=argsort)
    derivatives = 2 * wx.cumsum(dim=-1) - wx.sum(dim=-1, keepdim=True)
    search = torch.searchsorted(
        derivatives,
        torch.zeros_like(derivatives[..., :1]),
        side="left",
    ).clamp_max(derivatives.shape[-1] - 1)
    scale = y_div_x.gather(dim=-1, index=search).squeeze(-1)

    has_weight = weights.sum(dim=-1) > eps
    return torch.where(has_weight, scale.abs().clamp_min(eps), torch.ones_like(scale))


def _pi3x_point_loss(
    xy_pred: Tensor,
    logz_pred: Tensor,
    xy_target: Tensor,
    logz_target: Tensor,
    conf_weights: Tensor,
    view_mask: Tensor,
    *,
    scale_align_num_points: int,
    depth_weight_min_frac: float,
    ray_loss_weight: float,
    depth_loss_weight: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    finite_target = torch.isfinite(xy_target).all(dim=-1, keepdim=True) & torch.isfinite(logz_target)
    xy_target = torch.nan_to_num(xy_target, nan=0.0, posinf=0.0, neginf=0.0)
    xy_pred = torch.nan_to_num(xy_pred, nan=0.0, posinf=0.0, neginf=0.0)

    pred_points, _ = _local_points_from_xy_logz(xy_pred, logz_pred)
    target_points, target_depth = _local_points_from_xy_logz(xy_target, logz_target)

    conf_weights = torch.where(finite_target, conf_weights, torch.zeros_like(conf_weights))
    valid_weights = (conf_weights > 0.5).to(dtype=xy_pred.dtype)

    ray_weights = view_mask[:, :, None, None].to(dtype=xy_pred.dtype) * finite_target.to(dtype=xy_pred.dtype)
    ray_denom = (ray_weights.sum() * xy_pred.shape[-1]).clamp_min(1.0)
    ray_loss = (torch.abs(xy_pred - xy_target) * ray_weights).sum() / ray_denom

    mean_depth = _weighted_mean(target_depth.detach(), valid_weights, dim=(2,), keepdim=True)
    min_depth = (depth_weight_min_frac * mean_depth).clamp_min(1e-6)
    depth_weights = 1.0 / target_depth.detach().clamp_min(min_depth).clamp_min(1e-6)
    point_weights = valid_weights * depth_weights

    scale = _align_points_scale_l1(
        pred_points,
        target_points,
        point_weights.squeeze(-1),
        max_points=scale_align_num_points,
    )
    aligned_pred_points = pred_points * scale.view(-1, 1, 1, 1)

    depth_denom = valid_weights.sum().clamp_min(1.0)
    depth_loss = (
        torch.abs(aligned_pred_points[..., 2:3] - target_points[..., 2:3]) * point_weights
    ).sum() / depth_denom

    total = ray_loss_weight * ray_loss + depth_loss_weight * depth_loss
    return total, ray_loss, depth_loss, scale


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.pose_enc_type = config.pose_enc_type
        self.cross_view_config = config.cross_view
        self.cross_view_fusion = self.cross_view_config.type != "none"

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
            pose_enc_type=config.pose_enc_type,
            ray_enc_type=config.ray_enc_type,
            view_enc_type=config.view_enc_type,
            cross_view_config=self.cross_view_config,
        )

        # Optional warm start of `ray_embed` from Pi3X's pretrained PatchEmbed.
        # Overrides the zero-init that PaliGemmaWithExpertModel just applied.
        if config.ray_enc_type and config.ray_embed_pi3x_init_path is not None:
            self._init_ray_embed_from_pi3x(
                config.ray_embed_pi3x_init_path, scale=config.ray_embed_pi3x_init_scale
            )

        self.aux_point_head = None
        if config.aux_point_head.enabled:
            ph_cfg = config.aux_point_head
            self.aux_point_head = PointHead(
                in_dim=ph_cfg.in_dim,
                hidden_dim=ph_cfg.hidden_dim,
                depth=ph_cfg.depth,
                num_heads=ph_cfg.num_heads,
                mlp_ratio=ph_cfg.mlp_ratio,
                rope_freq=ph_cfg.rope_freq,
                qk_norm=ph_cfg.qk_norm,
                init_values=ph_cfg.init_values,
                output_resolution=ph_cfg.output_resolution,
            )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")

        disable_torch_compile = os.getenv("OPENPI_DISABLE_TORCH_COMPILE", "").lower() in {"1", "true", "yes"}
        compile_reason = None
        if disable_torch_compile:
            compile_reason = "disabled via OPENPI_DISABLE_TORCH_COMPILE"
        elif self.cross_view_config.type != "none":
            # The cross-view path currently uses cached RoPE position tensors that can trip
            # TorchDynamo/CUDAGraphs during batched policy serving. Keep training untouched and
            # run inference eagerly for cross-view models until that path is made compile-safe.
            compile_reason = f"cross_view={self.cross_view_config.type}"
        elif not hasattr(torch, "compile"):
            compile_reason = "torch.compile unavailable"

        if compile_reason is None:
            self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")
            logging.info("Enabled torch.compile for sample_actions")
        else:
            logging.info("Skipping torch.compile for sample_actions (%s)", compile_reason)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False
        self.last_loss_breakdown: dict[str, float] = {}

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def _init_ray_embed_from_pi3x(self, path: str, *, scale: float):
        """Warm-start `paligemma_with_expert.ray_embed.proj` from Pi3X's PatchEmbed.

        Pi3X's `ray_embed.proj` has shape (1024, 2, 14, 14) / (1024,). openpi's is
        (1152, 2, 14, 14) / (1152,) because SigLIP-base/14 has hidden_dim=1152.
        First 1024 output channels copy from Pi3X (scaled by `scale`); the
        remaining 128 channels stay at zero (so they don't perturb the last 128
        dims of pretrained SigLIP patch tokens).

        Raises if the file is missing or has the wrong shape -- silent fallback
        would hide config errors.
        """
        import os

        if scale <= 0:
            logging.warning(
                "ray_embed_pi3x_init_path=%s but scale=%g -> skipping (no transfer).", path, scale
            )
            return
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ray_embed_pi3x_init_path={path!r} does not exist. "
                "Run scripts/dump_pi3x_ray_embed.py first."
            )

        state = torch.load(path, map_location="cpu", weights_only=True)
        w_pi3x = state["weight"]
        b_pi3x = state["bias"]

        ray_embed = self.paligemma_with_expert.ray_embed
        w_dst = ray_embed.proj.weight
        b_dst = ray_embed.proj.bias

        expected_pi3x = (1024, 2, 14, 14)
        if tuple(w_pi3x.shape) != expected_pi3x or tuple(b_pi3x.shape) != (1024,):
            raise ValueError(
                f"Unexpected Pi3X ray_embed shapes: weight={tuple(w_pi3x.shape)} bias={tuple(b_pi3x.shape)}; "
                f"expected weight={expected_pi3x} bias=(1024,)."
            )
        if w_dst.shape[1:] != w_pi3x.shape[1:]:
            raise ValueError(
                f"openpi ray_embed weight shape {tuple(w_dst.shape)} is incompatible with Pi3X "
                f"{tuple(w_pi3x.shape)} on the kernel dims."
            )
        if w_dst.shape[0] < w_pi3x.shape[0]:
            raise ValueError(
                f"openpi ray_embed has fewer output channels ({w_dst.shape[0]}) than Pi3X "
                f"({w_pi3x.shape[0]}); transfer requires openpi >= Pi3X."
            )

        with torch.no_grad():
            w_dst.zero_()
            b_dst.zero_()
            w_dst[:1024].copy_(w_pi3x.to(dtype=w_dst.dtype) * scale)
            b_dst[:1024].copy_(b_pi3x.to(dtype=b_dst.dtype) * scale)
        logging.info(
            "Initialized paligemma_with_expert.ray_embed.proj from Pi3X (%s, scale=%g): "
            "first 1024 of %d output channels populated; remaining zero.",
            path,
            scale,
            w_dst.shape[0],
        )

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(
            observation,
            train=train,
            disable_geometric_augs=getattr(self.config, "disable_geometric_augs", False),
        )
        return (
            observation,
            torch.stack(list(observation.images.values()), dim=1),
            torch.stack(list(observation.image_masks.values()), dim=1),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def auxiliary_head(self, fused_tokens):
        """Run auxiliary distillation heads on post-fusion, pre-projector vision tokens.

        fused_tokens: (B, V, patch_h*patch_w, D) in the backbone dtype.
        Returns a dict of predictions (empty when no aux heads are enabled). Dict keys:
            point: (xy, z) from the pi3x point head.
        """
        aux_out = {}
        if self.aux_point_head is not None:
            # Upcast — aux_point_head is float32 while fused_tokens are bfloat16.
            aux_out["point"] = self.aux_point_head(fused_tokens.float())
        return aux_out

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, obs):
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.

        Returns (embs, pad_masks, att_masks, fused_tokens) — fused_tokens are the
        post-cross-view-fusion, pre-projector patch tokens (B, V, patch_h*patch_w, D).
        """
        if isinstance(images, list):
            images = torch.stack(images, dim=1)
        if isinstance(img_masks, list):
            img_masks = torch.stack(img_masks, dim=1)

        embs = []
        pad_masks = []
        att_masks = []

        cam_keys = [name.split("_0_rgb")[0] for name in obs.images]

        # Per-camera extrinsics/intrinsics as dicts keyed by cam name. embed_image
        # stacks these to (B, V, 4, 4)/(B, V, 3, 3) with identity fallback as needed.
        cam_pos = None
        if self.pose_enc_type != "null":
            cam_pos = {"base": obs.agent_extrinsic, "left_wrist": obs.wrist_extrinsic, "right_wrist": None}

        cam_intr = None
        if obs.agent_intrinsic is not None or obs.wrist_intrinsic is not None:
            cam_intr = {"base": obs.agent_intrinsic, "left_wrist": obs.wrist_intrinsic, "right_wrist": None}

        # -------------- Process images (vision_tower + multi_modal_projector) --------------
        def image_embed_func(images, img_masks, cam_pos, cam_intr):
            return self.paligemma_with_expert.embed_image(
                images,
                img_masks,
                cam_pos=cam_pos,
                cam_intr=cam_intr,
                cam_keys=cam_keys,
            )

        img_emb, img_masks, fused_tokens = self._apply_checkpoint(
            image_embed_func, images, img_masks, cam_pos, cam_intr
        )  # [B, 256, 2048] here just img for one cam

        embs.append(img_emb)
        pad_masks.append(img_masks)
        att_masks += [0] * img_emb.shape[1]

        # -------------- Process language tokens --------------
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        # -------------- concat img language tokens --------------
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, fused_tokens

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # ----------------------- Embed state  -----------------------
        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # ----------------------- Embed time  -----------------------
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # ----------------------- Embed action  -----------------------
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        # -------------- Fuse timestep + action information using an MLP  --------------
        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        # ---------------------- concat state_emb + time_action_emb ----------------------
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        # observation: images{'base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb'}, image_masks
        # state [b, 32]  tokenized_prompt,tokenized_prompt_mask [32, 48],
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        observation, images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=True
        )

        # ----------------- get noisy actions  -----------------
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # ----------------- embed img + language  -----------------
        prefix_embs, prefix_pad_masks, prefix_att_masks, fused_tokens = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, obs=observation
        )

        # ----------------- embed state + time_action  -----------------
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)  # [batch_size, 1, 867, 867]

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        action_loss_raw = F.mse_loss(u_t, v_t, reduction="none")
        # Loss-weight curriculum knob: e.g. set to 0.1 in Stage 1 of the two-stage
        # Pi3X distillation recipe to let the aux loss dominate while action heads warm up.
        loss = action_loss_raw * self.config.action_loss_weight
        aux_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)
        aux_xy_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)
        aux_z_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)
        aux_scale_mean = torch.ones((), dtype=loss.dtype, device=loss.device)
        aux_gt_frac = torch.zeros((), dtype=loss.dtype, device=loss.device)
        aux_pi3x_frac = torch.zeros((), dtype=loss.dtype, device=loss.device)
        aux_gt_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)
        aux_pi3x_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)

        aux_pred = self.auxiliary_head(fused_tokens)
        if "point" in aux_pred:
            xy_pred, z_pred = aux_pred["point"]  # (B, V, P, 2), (B, V, P, 1)
            w = self.config.aux_point_head.loss_weight
            point_xy_tgt = getattr(observation, "point_target_xy", None)
            point_logz_tgt = getattr(observation, "point_target_logz", None)
            point_conf_tgt = getattr(observation, "point_target_conf", None)
            pi3x_xy_tgt = getattr(observation, "pi3x_target_xy", None)
            pi3x_logz_tgt = getattr(observation, "pi3x_target_logz", None)
            pi3x_conf_tgt = getattr(observation, "pi3x_target_conf", None)
            target_source = getattr(observation, "point_target_source", None)
            has_point_target = point_xy_tgt is not None and point_logz_tgt is not None and point_conf_tgt is not None
            has_pi3x_target = pi3x_xy_tgt is not None and pi3x_logz_tgt is not None and pi3x_conf_tgt is not None

            if not has_point_target and not has_pi3x_target:
                # No teacher targets in this batch — keep the head in the autograd graph
                # for DDP without contributing to the loss.
                loss = loss + w * 0.0 * (xy_pred.sum() + z_pred.sum())
            else:
                ph_cfg = self.config.aux_point_head

                def compute_target_loss(xy_tgt: Tensor, logz_tgt: Tensor, conf_tgt: Tensor):
                    # Targets come from the loader as (B, V_t, ph, pw, C). Flatten
                    # pixels/patches and slice the prediction to the supervised view
                    # count (V_t = 2 for LIBERO; right_wrist is padded).
                    v_tgt = xy_tgt.shape[1]
                    xy_tgt_f = xy_tgt.flatten(2, 3).to(xy_pred.dtype)
                    logz_tgt_f = logz_tgt.flatten(2, 3).to(z_pred.dtype)
                    conf_tgt_f = conf_tgt.flatten(2, 3).to(xy_pred.dtype)
                    xy_pred_v = xy_pred[:, :v_tgt]
                    z_pred_v = z_pred[:, :v_tgt]

                    conf_weights = torch.sigmoid(conf_tgt_f).to(xy_pred.dtype)
                    view_mask = img_masks[:, :v_tgt].to(xy_pred.dtype)
                    conf_weights = conf_weights * view_mask[:, :, None, None]

                    return _pi3x_point_loss(
                        xy_pred_v,
                        z_pred_v,
                        xy_tgt_f,
                        logz_tgt_f,
                        conf_weights,
                        view_mask,
                        scale_align_num_points=ph_cfg.scale_align_num_points,
                        depth_weight_min_frac=ph_cfg.depth_weight_min_frac,
                        ray_loss_weight=ph_cfg.ray_loss_weight,
                        depth_loss_weight=ph_cfg.depth_loss_weight,
                    )

                if has_point_target and has_pi3x_target:
                    source = (
                        target_source.to(dtype=loss.dtype, device=loss.device).flatten()
                        if target_source is not None
                        else torch.full((1,), 0.5, dtype=loss.dtype, device=loss.device)
                    )
                    gt_weight = source.mean().clamp(0.0, 1.0)
                    pi3x_weight = 1.0 - gt_weight

                    gt_raw, gt_ray, gt_depth, gt_scale = compute_target_loss(
                        point_xy_tgt,
                        point_logz_tgt,
                        point_conf_tgt,
                    )
                    pi3x_raw, pi3x_ray, pi3x_depth, pi3x_scale = compute_target_loss(
                        pi3x_xy_tgt,
                        pi3x_logz_tgt,
                        pi3x_conf_tgt,
                    )

                    raw_aux_loss = gt_weight * gt_raw + pi3x_weight * pi3x_raw
                    aux_xy_loss = gt_weight * gt_ray + pi3x_weight * pi3x_ray
                    aux_z_loss = gt_weight * gt_depth + pi3x_weight * pi3x_depth
                    aux_scale_mean = (
                        gt_weight * gt_scale.mean() + pi3x_weight * pi3x_scale.mean()
                    ).to(dtype=loss.dtype)
                    aux_loss = w * raw_aux_loss
                    aux_gt_loss = w * gt_weight * gt_raw
                    aux_pi3x_loss = w * pi3x_weight * pi3x_raw
                    aux_gt_frac = gt_weight
                    aux_pi3x_frac = pi3x_weight
                else:
                    if has_point_target:
                        xy_tgt, logz_tgt, conf_tgt = point_xy_tgt, point_logz_tgt, point_conf_tgt
                    else:
                        xy_tgt, logz_tgt, conf_tgt = pi3x_xy_tgt, pi3x_logz_tgt, pi3x_conf_tgt

                    raw_aux_loss, aux_xy_loss, aux_z_loss, aux_scale = compute_target_loss(
                        xy_tgt,
                        logz_tgt,
                        conf_tgt,
                    )
                    aux_scale_mean = aux_scale.mean().to(dtype=loss.dtype)
                    aux_loss = w * raw_aux_loss

                loss = loss + aux_loss
                if has_point_target and not has_pi3x_target and target_source is not None:
                    source = target_source.to(dtype=loss.dtype, device=loss.device).flatten()
                    aux_gt_frac = source.mean()
                    aux_pi3x_frac = 1.0 - aux_gt_frac
                elif has_pi3x_target and not has_point_target:
                    aux_gt_frac = torch.zeros((), dtype=loss.dtype, device=loss.device)
                    aux_pi3x_frac = torch.ones((), dtype=loss.dtype, device=loss.device)

        self.last_loss_breakdown = {
            "action_loss_raw": float(action_loss_raw.mean().detach()),
            "action_loss": float(loss.mean().detach() - aux_loss.detach()),
            "aux_loss": float(aux_loss.detach()),
            "aux_gt_loss": float(aux_gt_loss.detach()),
            "aux_pi3x_loss": float(aux_pi3x_loss.detach()),
            "aux_xy_loss": float(aux_xy_loss.detach()),
            "aux_z_loss": float(aux_z_loss.detach()),
            "aux_scale_mean": float(aux_scale_mean.detach()),
            "aux_gt_frac": float(aux_gt_frac.detach()),
            "aux_pi3x_frac": float(aux_pi3x_frac.detach()),
        }

        return loss

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        observation, images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=False
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks, _ = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, observation
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
