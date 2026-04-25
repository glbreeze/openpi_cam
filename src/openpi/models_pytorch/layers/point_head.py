"""Decoder head for (xy, log_z) prediction at either patch or full resolution.

Mirrors the structure of pi3x's `point_decoder` + `point_head`. Two output
modes selected via `output_resolution`:

* `16` (default): emits patch-level (xy, log_z) via a single zero-init Linear.
  Cheap. Pair with the avg-pooled patch cache.

* `224`: emits full-resolution (xy, log_z) via a Pi3X-style ConvHead upsampler
  (3 ConvTranspose2d stages 16->32->64->128, then bilinear to 224, then
  per-output Conv2d). Final Conv2d weights are zero-init so step-0 output is
  still zero. Pair with the full-res teacher cache.

Outputs per patch (or per pixel at full-res):
    xy: (..., 2)  — ray direction in camera frame (pre-normalization, matches
                   pi3x's pred['xy']).
    z:  (..., 1)  — log-depth (pre-exp). Consumer applies `exp(z.clamp(max=15))`
                   then `local_points = (xy * z, z)` to reconstruct camera-frame
                   3D points, matching pi3x's convention.
"""

import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch import nn

from openpi.models_pytorch.layers.block import Block
from openpi.models_pytorch.layers.rope import PositionGetter
from openpi.models_pytorch.layers.rope import RotaryPositionEmbedding2D


class _ConvHeadUpsampler(nn.Module):
    """Pi3X-ConvHead-style 16x16 -> 224x224 upsampler (no UV-conditioning).

    Mirrors pi3.models.layers.conv_head.ConvHead but stripped of UV-conditioning
    (pi0 LIBERO is square) and with separate xy / z output heads. Final
    per-output Conv2d weights are zero-init so the head contributes zero output
    at step 0, preserving the pretrained baseline.
    """

    def __init__(
        self,
        dim_in: int,
        target_hw: int = 224,
        dim_upsample: tuple[int, ...] = (256, 128, 64),
        last_conv_channels: int = 32,
    ):
        super().__init__()
        self.target_hw = target_hw

        in_chs = (dim_in, *dim_upsample[:-1])
        self.upsample_blocks = nn.ModuleList()
        for in_ch, out_ch in zip(in_chs, dim_upsample, strict=True):
            stage = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode="replicate"),
            )
            self.upsample_blocks.append(stage)

        last_dim = dim_upsample[-1]

        def _make_output_block(out_dim: int) -> nn.Sequential:
            block = nn.Sequential(
                nn.Conv2d(last_dim, last_conv_channels, kernel_size=3, padding=1, padding_mode="replicate"),
                nn.ReLU(inplace=True),
                nn.Conv2d(last_conv_channels, out_dim, kernel_size=1),
            )
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)
            return block

        self.xy_head = _make_output_block(2)
        self.z_head = _make_output_block(1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x: (BV, dim_in, 16, 16)
        for block in self.upsample_blocks:
            x = block(x)
        x = F.interpolate(x, size=(self.target_hw, self.target_hw), mode="bilinear", align_corners=False)
        return self.xy_head(x), self.z_head(x)


class PointHead(nn.Module):
    """(xy, z) prediction head for pi3x distillation.

    Input:
        tokens: (B, V, P, D) vision features after cross-view fusion, where
                P = patch_h * patch_w.

    Output:
        xy: (B, V, P_out, 2)   P_out = patch_h*patch_w when output_resolution=16,
                                else output_resolution**2.
        z:  (B, V, P_out, 1)   (log-depth)
    """

    def __init__(
        self,
        in_dim: int = 1152,
        hidden_dim: int = 512,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_h: int = 16,
        patch_w: int = 16,
        rope_freq: int = 100,
        qk_norm: bool = True,
        init_values: float | None = 0.01,
        output_resolution: int = 16,
    ):
        super().__init__()
        if hidden_dim % (num_heads * 4) != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads*4 ({num_heads * 4}) for 2D RoPE."
            )
        if output_resolution not in (16, 224):
            raise ValueError(f"output_resolution must be 16 or 224, got {output_resolution}.")

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.output_resolution = output_resolution

        self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        if output_resolution == 16:
            self.linear_out = nn.Linear(hidden_dim, 3)
            nn.init.zeros_(self.linear_out.weight)
            nn.init.zeros_(self.linear_out.bias)
        else:
            self.upsampler = _ConvHeadUpsampler(dim_in=hidden_dim, target_hw=output_resolution)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        B, V, P, D = tokens.shape
        expected_P = self.patch_h * self.patch_w
        if P != expected_P:
            raise ValueError(f"Expected P={expected_P} (patch_h*patch_w), got P={P}.")

        x = tokens.reshape(B * V, P, D)
        x = self.input_proj(x)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * V, self.patch_h, self.patch_w, device=x.device)

        for blk in self.blocks:
            x = blk(x, pos=pos)

        x = self.norm(x)

        if self.output_resolution == 16:
            out = self.linear_out(x)  # (BV, P, 3)
            xy = out[..., :2].reshape(B, V, P, 2)
            z = out[..., 2:3].reshape(B, V, P, 1)
            return xy, z

        # Full-res path: reshape patch tokens to a 2D feature map and upsample.
        x_2d = x.permute(0, 2, 1).reshape(B * V, -1, self.patch_h, self.patch_w)
        xy_full, z_full = self.upsampler(x_2d)  # (BV, 2, R, R), (BV, 1, R, R)
        r = self.output_resolution
        xy = xy_full.permute(0, 2, 3, 1).reshape(B, V, r * r, 2)
        z = z_full.permute(0, 2, 3, 1).reshape(B, V, r * r, 1)
        return xy, z
