"""Cache Pi3X teacher targets for the RoboCasa-v0 (24-task) LeRobot dataset.

For every (episode, frame, cam) in `robocasa24/all24_human_camaware`, runs the
Pi3X teacher on the agentview_left + eye_in_hand frames and dumps per-frame
patch-level targets that the openpi `AuxPointHead` consumes:

    xy     : (T, 16, 16, 2)  — pre-z-multiplication direction
    log_z  : (T, 16, 16, 1)  — log depth (pre-exp)
    conf   : (T, 16, 16, 1)  — conf logits (pre-sigmoid)

Format and preprocessing match `scripts/cache_pi3x_targets.py` (the LIBERO
variant) and the GT counterpart. The two caches share the npz schema so
`MixedPointTargetLoader` / `DualPointTargetLoader` can read either freely.

Differences vs the LIBERO Pi3X cache:
  - Reads frames via LeRobotDataset (our dataset stores PNG bytes inline in
    parquet; the LIBERO script's raw `pq.read_table` path assumes LIBERO's
    column names and is fragile across storage modes).
  - Source images are already 224×224 (Pi0/Pi3 input size); no resize needed,
    K is *not* rescaled — just the openpi `fx → -fx` flip.
  - Cam mapping mirrors `LeRobotRobocasaCamDataConfig.point_target_cams`:
        robot0_agentview_left  →  base   (model slot base_0_rgb)
        robot0_eye_in_hand     →  wrist  (model slot left_wrist_0_rgb)

Output layout (matches the existing robocasa365 / libero distill caches):
    {output_root}/{cam_short}/episode_NNNNNN.npz   keys: xy, log_z, conf  fp16

Typical invocation:

    uv run scripts/cache_robocasa24_pi3x_targets.py \\
        --pi3x-repo ~/Research/CamVLA/Pi3X_Libero \\
        --output-root ~/.cache/openpi/pi3x_targets_224/robocasa24_all24_human_camaware

Runs in the `openpi` venv. Does NOT need robosuite/robocasa/mujoco.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger("cache_robocasa24_pi3x_targets")

# Cam mapping. Left side is the LeRobot key suffix; right is the cache subdir
# (matches model-side slot used by `RobocasaCamInputs`).
CAM_SPECS = (
    {"lerobot_key": "robot0_agentview_left", "intr_key": "agentview_left_intrinsic", "cache_dir": "base"},
    {"lerobot_key": "robot0_eye_in_hand",   "intr_key": "eye_in_hand_intrinsic",   "cache_dir": "wrist"},
)
SRC_IMAGE_SIZE = 224       # robocasa24/all24_human_camaware frames are already 224
TARGET_IMAGE_SIZE = 224    # Pi0/Pi3 input — no resize needed; just the openpi flip


def _passthrough_K(K: np.ndarray, scale: float) -> np.ndarray:
    """Pass-through K (natural convention) with isotropic scale.

    RoboCasa's training pipeline does NOT apply the openpi `_preprocess_image`
    flip (verified in `src/openpi/policies/robocasa_policy.py` on py-torch),
    and `RobocasaCamInputs._passthrough_intrinsic` leaves K natural. So we
    must NOT apply `fx → -fx` here, unlike the LIBERO cache script.
    """
    K_out = np.asarray(K, dtype=np.float32).copy()
    if scale != 1.0:
        K_out[0, 0] *= scale
        K_out[0, 2] *= scale
        K_out[1, 1] *= scale
        K_out[1, 2] *= scale
    return K_out


def _prep_images(uint8_imgs: np.ndarray, target_hw: int) -> torch.Tensor:
    """(T, H, W, 3) uint8 → (T, 3, target_hw, target_hw) float in [0, 1].

    RoboCasa uses the natural y-down image convention end-to-end (see
    `robocasa_policy.py` header). NO `[::-1, ::-1]` flip is applied here.
    """
    chw = torch.from_numpy(uint8_imgs).permute(0, 3, 1, 2).contiguous().float() / 255.0
    if chw.shape[-1] != target_hw or chw.shape[-2] != target_hw:
        chw = F.interpolate(chw, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
    return chw


@torch.no_grad()
def _teacher_patch_forward(model, imgs: torch.Tensor, K: torch.Tensor,
                            output_resolution: int = 16):
    """Run Pi3X (single view) and pool to the SigLIP patch grid.

    Args:
        imgs: (B, 3, 224, 224) in [0, 1], pre-flipped to openpi orientation.
        K:    (B, 3, 3) intrinsics in the openpi-flipped convention.

    Returns:
        xy   : (B, R, R, 2)
        logz : (B, R, R, 1)
        conf : (B, R, R, 1)
        where R = output_resolution.
    """
    B, _, H, W = imgs.shape
    patch_h, patch_w = H // 14, W // 14

    imgs5 = imgs[:, None]      # (B, N=1, 3, H, W)
    K4 = K[:, None]            # (B, N=1, 3, 3)
    imgs_norm = (imgs5 - model.image_mean) / model.image_std

    hidden, poses_, _, use_pose_mask, _ = model.encode(
        imgs_norm, with_prior=True, depths=None, intrinsics=K4, poses=None, rays=None,
    )
    hidden = hidden.reshape(B, 1, -1, model.dec_embed_dim)
    hidden, pos = model.decode(hidden, 1, H, W, poses_, use_pose_mask)

    ret_point = model.point_decoder(hidden, xpos=pos)
    point_feat = ret_point[:, model.patch_start_idx:].float()
    with torch.amp.autocast(device_type="cuda", enabled=False):
        xy_full, log_z_full = model._chunked_conv_head(model.point_head, point_feat, patch_h, patch_w)

    ret_conf = model.conf_decoder(hidden, xpos=pos)
    conf_feat = ret_conf[:, model.patch_start_idx:].float()
    with torch.amp.autocast(device_type="cuda", enabled=False):
        conf_full = model._chunked_conv_head(model.conf_head, conf_feat, patch_h, patch_w)[0]

    if output_resolution == 16:
        xy_out = F.avg_pool2d(xy_full, kernel_size=14, stride=14)
        logz_out = F.avg_pool2d(log_z_full, kernel_size=14, stride=14)
        conf_out = F.avg_pool2d(conf_full, kernel_size=14, stride=14)
    elif output_resolution == 224:
        xy_out, logz_out, conf_out = xy_full, log_z_full, conf_full
    else:
        raise ValueError(f"output_resolution must be 16 or 224, got {output_resolution}")

    xy_out = xy_out.permute(0, 2, 3, 1).contiguous()
    logz_out = logz_out.permute(0, 2, 3, 1).contiguous()
    conf_out = conf_out.permute(0, 2, 3, 1).contiguous()
    return xy_out, logz_out, conf_out


def _frame_to_uint8_hwc(t) -> np.ndarray:
    """LeRobot returns frames as float32 CHW in [0,1] or uint8 HWC depending on
    storage mode. Normalize to uint8 (T, H, W, 3)."""
    arr = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    # (3, H, W) -> (H, W, 3)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    elif arr.ndim == 4 and arr.shape[1] == 3:
        arr = arr.transpose(0, 2, 3, 1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    return arr


def _load_episode_frames(ds: LeRobotDataset, episode_idx: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Returns:
        frames[lerobot_key]    : (T, 224, 224, 3) uint8
        intrinsics[intr_key]   : (3, 3) float32
        T
    """
    start = int(ds.episode_data_index["from"][episode_idx])
    end = int(ds.episode_data_index["to"][episode_idx])
    T = end - start
    frames: dict[str, list[np.ndarray]] = {c["lerobot_key"]: [] for c in CAM_SPECS}
    intr_by_key: dict[str, np.ndarray | None] = {c["intr_key"]: None for c in CAM_SPECS}

    for global_t in range(start, end):
        row = ds[global_t]
        for cam in CAM_SPECS:
            img = _frame_to_uint8_hwc(row[f"observation.images.{cam['lerobot_key']}"])
            frames[cam["lerobot_key"]].append(img)
            if intr_by_key[cam["intr_key"]] is None:
                intr_by_key[cam["intr_key"]] = np.asarray(
                    row[f"observation.{cam['intr_key']}"], dtype=np.float32
                )
    stacked = {k: np.stack(v, axis=0) for k, v in frames.items()}
    intr_final: dict[str, np.ndarray] = {k: np.asarray(v, dtype=np.float32) for k, v in intr_by_key.items()}
    return stacked, intr_final, T


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
            logger.warning("Pi3X load_state_dict: missing=%d unexpected=%d",
                           len(missing), len(unexpected))
    else:
        model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
    return model.to(device)


def _episode_outputs_exist(output_root: Path, episode_idx: int) -> bool:
    return all((output_root / cam["cache_dir"] / f"episode_{episode_idx:06d}.npz").exists()
               for cam in CAM_SPECS)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--repo-id", default="robocasa24/all24_human_camaware",
        help="LeRobot repo_id under $HF_LEROBOT_HOME.",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("~/.cache/openpi/pi3x_targets_224/robocasa24_all24_human_camaware").expanduser(),
        help="Where per-cam .npz files land. Matches the convention "
             "`LeRobotRobocasaCamDataConfig` expects via "
             "`pi3x_targets_root` + `point_target_cams`.",
    )
    parser.add_argument(
        "--pi3x-repo", type=Path,
        default=Path("~/Research/CamVLA/Pi3X_Libero").expanduser(),
        help="Path to the Pi3X_Libero source repo (provides pi3.models.pi3x.Pi3X).",
    )
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional local Pi3X ckpt; default = HF yyfz233/Pi3X.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-resolution", type=int, default=16, choices=[16, 224])
    parser.add_argument("--episode-range", type=str, default=None,
                        help="'lo:hi' python slice or 'i,j,k' explicit ids.")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Re-run episodes whose outputs already exist.")
    parser.add_argument("--no-autocast", action="store_true",
                        help="Disable bf16/fp16 autocast (~2x slower, ~2x memory).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    device = torch.device(args.device)
    autocast_dtype = None
    if not args.no_autocast and device.type == "cuda":
        autocast_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    logger.info("loading dataset %s ...", args.repo_id)
    ds = LeRobotDataset(repo_id=args.repo_id, video_backend="pyav")
    logger.info("  episodes=%d  frames=%d  fps=%d", ds.num_episodes, ds.num_frames, ds.fps)

    # Resolve episode index list
    total = ds.num_episodes
    if args.episode_range:
        if ":" in args.episode_range:
            lo, hi = args.episode_range.split(":", 1)
            ep_indices = list(range(int(lo) if lo else 0,
                                    min(int(hi) if hi else total, total)))
        else:
            ep_indices = [int(x) for x in args.episode_range.split(",") if x.strip()]
    else:
        ep_indices = list(range(total))

    if not args.no_skip_existing:
        before = len(ep_indices)
        ep_indices = [i for i in ep_indices if not _episode_outputs_exist(args.output_root, i)]
        skipped = before - len(ep_indices)
        if skipped:
            logger.info("skipping %d episodes whose outputs already exist", skipped)
    if not ep_indices:
        logger.info("nothing to do.")
        return

    logger.info("building Pi3X teacher (device=%s, autocast=%s)", device, autocast_dtype)
    model = _build_pi3x(args.pi3x_repo, args.ckpt, device)

    args.output_root.mkdir(parents=True, exist_ok=True)
    for cam in CAM_SPECS:
        (args.output_root / cam["cache_dir"]).mkdir(parents=True, exist_ok=True)

    R = args.output_resolution
    t0 = time.time()
    pbar = tqdm.tqdm(ep_indices, desc="episodes", unit="ep")
    total_frames = 0
    for ep_idx in pbar:
        frames, intr_by_key, T = _load_episode_frames(ds, ep_idx)
        for cam in CAM_SPECS:
            out_path = args.output_root / cam["cache_dir"] / f"episode_{ep_idx:06d}.npz"
            if out_path.exists() and not args.no_skip_existing:
                continue
            uint8 = frames[cam["lerobot_key"]]
            K_adj = _passthrough_K(intr_by_key[cam["intr_key"]], scale=1.0)
            K_tensor = torch.from_numpy(K_adj)[None].to(device).expand(args.batch_size, 3, 3)

            xy_buf = np.empty((T, R, R, 2), dtype=np.float16)
            logz_buf = np.empty((T, R, R, 1), dtype=np.float16)
            conf_buf = np.empty((T, R, R, 1), dtype=np.float16)

            for s in range(0, T, args.batch_size):
                e = min(s + args.batch_size, T)
                chunk = uint8[s:e]
                imgs = _prep_images(chunk, TARGET_IMAGE_SIZE).to(device, non_blocking=True)
                K_chunk = K_tensor[: imgs.shape[0]]
                if autocast_dtype is not None and device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=autocast_dtype):
                        xy_p, logz_p, conf_p = _teacher_patch_forward(
                            model, imgs, K_chunk, output_resolution=R
                        )
                else:
                    xy_p, logz_p, conf_p = _teacher_patch_forward(
                        model, imgs, K_chunk, output_resolution=R
                    )
                xy_buf[s:e] = xy_p.float().cpu().numpy().astype(np.float16)
                logz_buf[s:e] = logz_p.float().cpu().numpy().astype(np.float16)
                conf_buf[s:e] = conf_p.float().cpu().numpy().astype(np.float16)

            np.savez(out_path, xy=xy_buf, log_z=logz_buf, conf=conf_buf)
        total_frames += T
        elapsed = time.time() - t0
        rate = (pbar.n + 1) / max(elapsed, 1e-9)
        pbar.set_postfix_str(f"frames={total_frames} ep/s={rate:.2f}")
    pbar.close()
    logger.info("done — %d episodes / %d frames at %s",
                len(ep_indices), total_frames, args.output_root)


if __name__ == "__main__":
    main()
