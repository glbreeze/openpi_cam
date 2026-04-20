"""Lightweight decoder head for per-patch point prediction.

Mirrors the structure of pi3x's `point_decoder` + `point_head` but distills at
patch resolution — no ConvHead upsampling — because teacher targets are cached
at 16x16 patch grid, not full 224x224.

Outputs per patch:
    xy: (..., 2)  — ray direction in camera frame (pre-normalization, matches
                   pi3x's pred['xy']).
    z:  (..., 1)  — log-depth (pre-exp). Consumer applies `exp(z.clamp(max=15))`
                   then `local_points = (xy * z, z)` to reconstruct camera-frame
                   3D points, matching pi3x's convention.

The output linear layer is zero-initialized so the head contributes no gradient
signal until the distillation loss drives it away from zero — keeps the
pre-distillation behavior identical.
"""

from torch import Tensor
from torch import nn

from openpi.models_pytorch.layers.block import Block
from openpi.models_pytorch.layers.rope import PositionGetter
from openpi.models_pytorch.layers.rope import RotaryPositionEmbedding2D


class PointHead(nn.Module):
    """Per-patch (xy, z) prediction head for pi3x distillation.

    Input:
        tokens: (B, V, P, D) vision features after cross-view fusion, where
                P = patch_h * patch_w.

    Output:
        xy: (B, V, P, 2)
        z:  (B, V, P, 1)  (log-depth)
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
    ):
        super().__init__()
        if hidden_dim % (num_heads * 4) != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads*4 ({num_heads * 4}) for 2D RoPE."
            )

        self.patch_h = patch_h
        self.patch_w = patch_w

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
        self.linear_out = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)

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
        out = self.linear_out(x)  # (B*V, P, 3)
        xy = out[..., :2].reshape(B, V, P, 2)
        z = out[..., 2:3].reshape(B, V, P, 1)
        return xy, z
