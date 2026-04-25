"""Cache Pi3X teacher targets at SigLIP patch resolution for openpi distillation.

For every (episode, frame, cam) in a converted LeRobot LIBERO dataset, this script
runs the Pi3X teacher and dumps per-frame patch-level targets that the openpi
`AuxPointHead` consumes:

    xy     : (T, 16, 16, 2)  — pre-z-multiplication direction (matches PointHead output)
    log_z  : (T, 16, 16, 1)  — log depth (matches PointHead output, pre-exp)
    conf   : (T, 16, 16, 1)  — conf logits (pre-sigmoid)

To keep teacher patch features pixel-aligned with openpi's SigLIP grid, the cache
mirrors openpi's preprocessing exactly:
  1. Decode 256x256 uint8 frames from parquet.
  2. Apply `[::-1, ::-1]` flip (matches `_preprocess_image`).
  3. Bilinear-resize 256 -> 224 (square->square, no padding).
  4. Scale K by 224/256 and apply the openpi `fx -> -fx` flip
     (`_adjust_K_for_openpi_image_flip`).

We pool Pi3X's full-res xy / log_z / conf with a 14x14 / stride-14 avg pool, which
exactly matches the 16x16 SigLIP patch grid (224 / 14 = 16).

Output layout:
    {output_root}/{cam}/episode_{NNNNNN}.npz   keys: xy, log_z, conf  (all fp16)

Usage:
    uv run scripts/cache_pi3x_targets.py \\
        --data-root ~/.cache/huggingface/lerobot/glbreeze/libero_object_cam_v2 \\
        --output-root ~/.cache/openpi/pi3x_targets/libero_object_cam_v2 \\
        --pi3x-repo ~/Research/Pi3X_Libero
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

logger = logging.getLogger("cache_pi3x_targets")

CAM_SPECS = (
    {"name": "agent", "image_col": "image", "intrinsic_col": "agent_intrinsic"},
    {"name": "wrist", "image_col": "wrist_image", "intrinsic_col": "wrist_intrinsic"},
)


def _decode_image_field(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.dtype != np.uint8:
            value = np.clip(value, 0, 255).astype(np.uint8)
        return value
    if isinstance(value, dict) and value.get("bytes") is not None:
        return np.array(Image.open(io.BytesIO(value["bytes"])).convert("RGB"))
    if isinstance(value, (bytes, bytearray)):
        return np.array(Image.open(io.BytesIO(value)).convert("RGB"))
    raise TypeError(f"Unsupported image field type: {type(value)}")


def _adjust_K_openpi(K: np.ndarray, scale: float) -> np.ndarray:
    """openpi's `_adjust_K_for_openpi_image_flip` (fx -> -fx) followed by isotropic K scaling."""
    K_out = np.asarray(K, dtype=np.float32).copy()
    K_out[0, 0] = -K_out[0, 0]
    K_out[0, 0] *= scale
    K_out[0, 2] *= scale
    K_out[1, 1] *= scale
    K_out[1, 2] *= scale
    return K_out


def _prep_images(uint8_imgs: np.ndarray, target_hw: int) -> torch.Tensor:
    """uint8 (T, H, W, 3) -> float [0,1] (T, 3, target_hw, target_hw), pre-flipped, resized."""
    flipped = uint8_imgs[:, ::-1, ::-1, :].copy()
    chw = torch.from_numpy(flipped).permute(0, 3, 1, 2).contiguous().float() / 255.0
    if chw.shape[-1] != target_hw or chw.shape[-2] != target_hw:
        chw = F.interpolate(chw, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
    return chw


@torch.no_grad()
def teacher_patch_forward(
    model, imgs: torch.Tensor, K: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run Pi3X (single-view) and return patch-resolution xy / log_z / conf.

    Args:
        imgs: (B, 3, H, W) in [0, 1].
        K:    (B, 3, 3) intrinsics already in the openpi-flipped, 224-scaled convention.

    Returns:
        xy_patch    : (B, 16, 16, 2) raw xy direction (matches Pi3X point_head output before exp).
        log_z_patch : (B, 16, 16, 1) raw log-z (matches Pi3X point_head output before exp).
        conf_patch  : (B, 16, 16, 1) conf logits (pre-sigmoid).
    """
    B, _, H, W = imgs.shape
    patch_h, patch_w = H // 14, W // 14

    imgs5 = imgs[:, None]  # (B, N=1, 3, H, W)
    K4 = K[:, None]  # (B, N=1, 3, 3)

    imgs_norm = (imgs5 - model.image_mean) / model.image_std

    hidden, poses_, _, use_pose_mask, _ = model.encode(
        imgs_norm,
        with_prior=True,
        depths=None,
        intrinsics=K4,
        poses=None,
        rays=None,
    )
    hidden = hidden.reshape(B, 1, -1, model.dec_embed_dim)
    hidden, pos = model.decode(hidden, 1, H, W, poses_, use_pose_mask)

    ret_point = model.point_decoder(hidden, xpos=pos)
    point_feat = ret_point[:, model.patch_start_idx :].float()
    with torch.amp.autocast(device_type="cuda", enabled=False):
        xy_full, log_z_full = model._chunked_conv_head(model.point_head, point_feat, patch_h, patch_w)

    ret_conf = model.conf_decoder(hidden, xpos=pos)
    conf_feat = ret_conf[:, model.patch_start_idx :].float()
    with torch.amp.autocast(device_type="cuda", enabled=False):
        conf_full = model._chunked_conv_head(model.conf_head, conf_feat, patch_h, patch_w)[0]

    # avg-pool 224 -> 16 (kernel=14, stride=14, no overlap)  -- exactly aligns to SigLIP patches
    xy_patch = F.avg_pool2d(xy_full, kernel_size=14, stride=14)  # (B, 2, 16, 16)
    log_z_patch = F.avg_pool2d(log_z_full, kernel_size=14, stride=14)  # (B, 1, 16, 16)
    conf_patch = F.avg_pool2d(conf_full, kernel_size=14, stride=14)  # (B, 1, 16, 16)

    xy_patch = xy_patch.permute(0, 2, 3, 1).contiguous()
    log_z_patch = log_z_patch.permute(0, 2, 3, 1).contiguous()
    conf_patch = conf_patch.permute(0, 2, 3, 1).contiguous()
    return xy_patch, log_z_patch, conf_patch


def _episode_outputs_exist(output_root: Path, episode_stem: str, cam_names: list[str]) -> bool:
    return all((output_root / cam / f"{episode_stem}.npz").exists() for cam in cam_names)


def _process_episode(
    parquet_path: Path,
    output_root: Path,
    model,
    device: torch.device,
    target_hw: int,
    src_hw: int,
    batch_size: int,
    autocast_dtype: torch.dtype | None,
):
    columns = ["image", "wrist_image", "agent_intrinsic", "wrist_intrinsic"]
    table = pq.read_table(parquet_path, columns=columns)
    rows = table.to_pylist()
    if not rows:
        logger.warning("Empty parquet: %s", parquet_path)
        return
    T = len(rows)
    scale = target_hw / src_hw

    for cam in CAM_SPECS:
        out_path = output_root / cam["name"] / f"{parquet_path.stem}.npz"
        if out_path.exists():
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)

        uint8_frames = np.stack([_decode_image_field(r[cam["image_col"]]) for r in rows], axis=0)
        if uint8_frames.shape[1] != src_hw or uint8_frames.shape[2] != src_hw:
            raise RuntimeError(
                f"{parquet_path.name}: expected {src_hw}x{src_hw} {cam['image_col']}, got {uint8_frames.shape[1:3]}"
            )
        K0 = np.asarray(rows[0][cam["intrinsic_col"]], dtype=np.float32)
        # All frames in libero share the same intrinsics; sanity-check on first/last only.
        K_last = np.asarray(rows[-1][cam["intrinsic_col"]], dtype=np.float32)
        if not np.allclose(K0, K_last, atol=1e-3):
            logger.warning("%s %s: intrinsics drift across frames; using row 0", parquet_path.name, cam["name"])
        K_adj = _adjust_K_openpi(K0, scale)
        K_tensor = torch.from_numpy(K_adj)[None].to(device).expand(batch_size, 3, 3)

        xy_buf = np.empty((T, 16, 16, 2), dtype=np.float16)
        logz_buf = np.empty((T, 16, 16, 1), dtype=np.float16)
        conf_buf = np.empty((T, 16, 16, 1), dtype=np.float16)

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            chunk = uint8_frames[start:end]
            imgs = _prep_images(chunk, target_hw).to(device)
            K_chunk = K_tensor[: imgs.shape[0]]

            ctx = (
                torch.amp.autocast("cuda", dtype=autocast_dtype)
                if autocast_dtype is not None and device.type == "cuda"
                else _nullcontext()
            )
            with ctx:
                xy_p, logz_p, conf_p = teacher_patch_forward(model, imgs, K_chunk)

            xy_buf[start:end] = xy_p.float().cpu().numpy().astype(np.float16)
            logz_buf[start:end] = logz_p.float().cpu().numpy().astype(np.float16)
            conf_buf[start:end] = conf_p.float().cpu().numpy().astype(np.float16)

        np.savez(out_path, xy=xy_buf, log_z=logz_buf, conf=conf_buf)
        logger.info("[done] %s/%s  T=%d -> %s", parquet_path.stem, cam["name"], T, out_path)


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _select_episodes(parquet_paths: list[Path], episode_range: str | None) -> list[Path]:
    if not episode_range:
        return parquet_paths
    if ":" in episode_range:
        lo, hi = episode_range.split(":")
        lo_i = int(lo) if lo else 0
        hi_i = int(hi) if hi else len(parquet_paths)
        return parquet_paths[lo_i:hi_i]
    wanted = {int(x) for x in episode_range.split(",") if x.strip()}
    out = []
    for p in parquet_paths:
        idx = int(p.stem.split("_")[-1])
        if idx in wanted:
            out.append(p)
    return out


def _build_pi3x(pi3x_repo: Path, ckpt: str | None, device: torch.device):
    if str(pi3x_repo) not in sys.path:
        sys.path.insert(0, str(pi3x_repo))
    from pi3.models.pi3x import Pi3X

    if ckpt:
        model = Pi3X(use_multimodal=True).eval()
        if ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file

            state = load_file(ckpt)
        else:
            state = torch.load(ckpt, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning("Pi3X load_state_dict: missing=%d unexpected=%d", len(missing), len(unexpected))
    else:
        model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("~/.cache/huggingface/lerobot/glbreeze/libero_object_cam_v2").expanduser(),
        help="LeRobot dataset root containing data/chunk-XXX/episode_*.parquet.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("~/.cache/openpi/pi3x_targets/libero_object_cam_v2").expanduser(),
        help="Where to write per-(cam, episode) .npz files.",
    )
    parser.add_argument("--pi3x-repo", type=Path, default=Path("~/Research/Pi3X_Libero").expanduser())
    parser.add_argument("--ckpt", type=str, default=None, help="Optional local Pi3X ckpt; defaults to HF yyfz233/Pi3X.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--src-hw", type=int, default=256, help="Stored image side; libero default is 256.")
    parser.add_argument("--target-hw", type=int, default=224, help="openpi SigLIP input side; must be 14*16.")
    parser.add_argument(
        "--episode-range",
        type=str,
        default=None,
        help="Either 'lo:hi' python slice or 'i,j,k' explicit ids. Default: all episodes.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-run episodes whose output .npz files already exist.",
    )
    parser.add_argument(
        "--no-autocast",
        action="store_true",
        help="Disable bfloat16/float16 autocast in Pi3X forward (slower, ~2x memory).",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.target_hw % 14 != 0 or args.target_hw // 14 != 16:
        raise SystemExit(f"--target-hw must be 14*16=224 to match SigLIP patch grid, got {args.target_hw}")

    device = torch.device(args.device)
    autocast_dtype = None
    if not args.no_autocast and device.type == "cuda":
        autocast_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    parquet_paths = sorted((args.data_root / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"No parquet files under {args.data_root}/data/chunk-*/")
    parquet_paths = _select_episodes(parquet_paths, args.episode_range)

    if not args.no_skip_existing:
        cam_names = [c["name"] for c in CAM_SPECS]
        before = len(parquet_paths)
        parquet_paths = [p for p in parquet_paths if not _episode_outputs_exist(args.output_root, p.stem, cam_names)]
        skipped = before - len(parquet_paths)
        if skipped:
            logger.info("Skipping %d episodes whose outputs already exist.", skipped)

    if not parquet_paths:
        logger.info("Nothing to do.")
        return

    logger.info("Building Pi3X (device=%s, autocast=%s)", device, autocast_dtype)
    model = _build_pi3x(args.pi3x_repo, args.ckpt, device)

    args.output_root.mkdir(parents=True, exist_ok=True)
    for cam in CAM_SPECS:
        (args.output_root / cam["name"]).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, parquet_path in enumerate(parquet_paths):
        _process_episode(
            parquet_path=parquet_path,
            output_root=args.output_root,
            model=model,
            device=device,
            target_hw=args.target_hw,
            src_hw=args.src_hw,
            batch_size=args.batch_size,
            autocast_dtype=autocast_dtype,
        )
        if (i + 1) % 10 == 0 or (i + 1) == len(parquet_paths):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(parquet_paths) - i - 1) / rate if rate > 0 else 0
            logger.info("Progress: %d/%d episodes (%.2f ep/s, eta %.1fm)", i + 1, len(parquet_paths), rate, eta / 60)


if __name__ == "__main__":
    main()
