"""PRoPE (Projective Positional Encoding) helpers.

Ported from pi3's `pi3/models/layers/prope.py`. Implements the pure-math
transforms needed to apply PRoPE-flavored RoPE to Q/K/V given per-view
extrinsics (camera<-world) and optional intrinsics.
"""

from functools import partial
from typing import Callable
from typing import Optional
from typing import Tuple

import torch


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device, dtype=Ks.dtype)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out


def _rope_precompute_coeffs(
    positions: torch.Tensor, freq_base: float, freq_scale: float, feat_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert positions.ndim == 1
    assert feat_dim % 2 == 0
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base ** (-torch.arange(num_freqs, device=positions.device)[None, None, None, :] / num_freqs)
    )
    angles = positions[None, None, :, None] * freqs
    return torch.cos(angles), torch.sin(angles)


def _rope_apply_coeffs(
    feats: torch.Tensor, coeffs: Tuple[torch.Tensor, torch.Tensor], inverse: bool = False
) -> torch.Tensor:
    cos, sin = coeffs
    if cos.shape[2] != feats.shape[2]:
        n_repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n_repeats, 1)
        sin = sin.repeat(1, 1, n_repeats, 1)
    cos = cos.to(feats.dtype)
    sin = sin.to(feats.dtype)
    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    if inverse:
        return torch.cat([cos * x_in - sin * y_in, sin * x_in + cos * y_in], dim=-1)
    return torch.cat([cos * x_in + sin * y_in, -sin * x_in + cos * y_in], dim=-1)


def _apply_tiled_projmat(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    # feats: (batch, num_heads, seqlen, feat_dim), seqlen = cameras * patches
    batch, num_heads, seqlen, feat_dim = feats.shape
    cameras = matrix.shape[1]
    D = matrix.shape[-1]
    assert seqlen % cameras == 0 and feat_dim % D == 0
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix.to(feats.dtype),
        feats.reshape(batch, num_heads, cameras, -1, feat_dim // D, D),
    ).reshape(feats.shape)


def _apply_block_diagonal(feats, func_size_pairs):
    funcs, block_sizes = zip(*func_size_pairs)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    return torch.cat([f(x) for f, x in zip(funcs, x_blocks)], dim=-1)


def prepare_apply_fns(
    head_dim: int,
    viewmats: torch.Tensor,
    Ks: Optional[torch.Tensor],
    patches_x: int,
    patches_y: int,
    image_width: int,
    image_height: int,
    freq_base: float = 100.0,
) -> Tuple[Callable, Callable, Callable]:
    """Build (apply_fn_q, apply_fn_kv, apply_fn_o) for PRoPE attention.

    viewmats: (B, cameras, 4, 4) camera<-world.
    Ks: (B, cameras, 3, 3) or None.
    """
    device = viewmats.device
    batch, cameras, _, _ = viewmats.shape

    if Ks is not None:
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
        Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
        Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
        Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
        Ks_norm[..., 2, 2] = 1.0
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2)
        P_inv = torch.einsum(
            "...ij,...jk->...ik", _invert_SE3(viewmats), _lift_K(_invert_K(Ks_norm))
        )
    else:
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)

    coeffs_x = _rope_precompute_coeffs(
        torch.tile(torch.arange(patches_x, device=device), (patches_y * cameras,)),
        freq_base=freq_base,
        freq_scale=1.0,
        feat_dim=head_dim // 4,
    )
    coeffs_y = _rope_precompute_coeffs(
        torch.tile(
            torch.repeat_interleave(torch.arange(patches_y, device=device), patches_x),
            (cameras,),
        ),
        freq_base=freq_base,
        freq_scale=1.0,
        feat_dim=head_dim // 4,
    )

    assert head_dim % 4 == 0
    transforms_q = [
        (partial(_apply_tiled_projmat, matrix=P_T), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_kv = [
        (partial(_apply_tiled_projmat, matrix=P_inv), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat, matrix=P), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x, inverse=True), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y, inverse=True), head_dim // 4),
    ]
    return (
        partial(_apply_block_diagonal, func_size_pairs=transforms_q),
        partial(_apply_block_diagonal, func_size_pairs=transforms_kv),
        partial(_apply_block_diagonal, func_size_pairs=transforms_o),
    )
