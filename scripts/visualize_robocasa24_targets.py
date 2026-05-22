"""Rich-and-fun visualizations of the RoboCasa-v0 GT + Pi3X distillation caches.

Reads:
  - Raw RGB frames from the source HDF5 (128×128) for ground-truth comparison
  - GT cache npz files (`xy / log_z / conf` at 16×16, fp16)
  - Pi3X cache npz files (same schema)

Produces (under `--out-dir`, default `~/Research/robocasa24/viz_targets/`):

  episode_overview_ep<N>.png
      4-frame strip of one demo. Per frame, per cam, 6 panels:
          [RGB] [GT depth-map] [Pi3X depth-map] [Pi3X conf]
          [GT xy_overlay] [|GT − Pi3X| depth diff]
      Lets you eyeball: do the depth maps agree with the scene, does
      Pi3X confidence drop where it should, where do GT and Pi3X
      disagree.

  task_gallery.png
      One representative frame per task (24 tiles). Per tile shows
      RGB | GT depth | Pi3X depth side-by-side at 224×224 (Pi3X grid
      upsampled bilinear). Useful for spotting collapsed / weird
      teacher outputs by task family.

  depth_histograms.png
      Aggregate histograms of `log_z` across a sample of episodes,
      separately for GT and Pi3X, both cams. Tells you whether Pi3X is
      predicting a metric-similar depth range to the sim.

  conf_distribution.png
      Pi3X confidence (post-sigmoid) histogram across sampled episodes.
      GT conf is trivially +10 everywhere; shown only for completeness.

Defaults assume the local bundle layout at
`~/Research/robocasa-human50/`. Override with `--bundle` if you ran from
the original `.cache` paths.

Typical invocation:
    uv run scripts/visualize_robocasa24_targets.py
    # or: pick a specific demo and only generate the overview figure
    uv run scripts/visualize_robocasa24_targets.py --episode 700 --only overview
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

logger = logging.getLogger("visualize_robocasa24_targets")

TASKS_24 = (
    "PnPCounterToCab", "PnPCabToCounter", "PnPCounterToSink", "PnPSinkToCounter",
    "PnPCounterToMicrowave", "PnPMicrowaveToCounter", "PnPCounterToStove",
    "PnPStoveToCounter",
    "OpenSingleDoor", "CloseSingleDoor", "OpenDoubleDoor", "CloseDoubleDoor",
    "OpenDrawer", "CloseDrawer",
    "TurnOnSinkFaucet", "TurnOffSinkFaucet", "TurnSinkSpout",
    "TurnOnStove", "TurnOffStove",
    "CoffeeSetupMug", "CoffeeServeMug", "CoffeePressButton",
    "TurnOnMicrowave", "TurnOffMicrowave",
)

CAMS = ("base", "wrist")
CAM_TITLES = {"base": "agentview_left (base)", "wrist": "eye_in_hand (wrist)"}
HDF5_CAM_KEY = {"base": "robot0_agentview_left_image", "wrist": "robot0_eye_in_hand_image"}


# --------------------------------------------------------------------------
# planning: map episode_idx -> (task, demo_key, source hdf5 path) using the
# canonical TASKS_24 order. Same logic the GT cache uses for episode_offset.
# --------------------------------------------------------------------------
def _resolve_src_path(raw_root: Path, task: str) -> Path:
    candidates = sorted(
        (raw_root / "v0.1" / "single_stage").glob(
            f"*/{task}/*/demo_gentex_im128_randcams.hdf5"
        )
    )
    candidates = [p for p in candidates if "/mg/" not in str(p)]
    if not candidates:
        raise FileNotFoundError(f"no HDF5 for {task} under {raw_root}")
    return candidates[0]


def _build_episode_plan(raw_root: Path) -> list[tuple[int, str, str, Path]]:
    """Returns list of (episode_idx, task_name, demo_key, src_hdf5_path)."""
    plan = []
    cum = 0
    for t in TASKS_24:
        try:
            src = _resolve_src_path(raw_root, t)
        except FileNotFoundError:
            logger.warning("skip %s — source missing", t)
            continue
        with h5py.File(src, "r") as f:
            demos = sorted(
                (k for k in f["data"].keys() if k.startswith("demo_")),
                key=lambda s: int(s.split("_")[1]),
            )
        for d in demos:
            plan.append((cum, t, d, src))
            cum += 1
    return plan


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------
def _load_cache_episode(root: Path, cam: str, ep_idx: int) -> dict:
    p = root / cam / f"episode_{ep_idx:06d}.npz"
    with np.load(p) as f:
        return {"xy": f["xy"][...], "log_z": f["log_z"][...], "conf": f["conf"][...]}


def _load_rgb_episode(src_hdf5: Path, demo_key: str, cam: str) -> np.ndarray:
    """Returns (T, 128, 128, 3) uint8."""
    with h5py.File(src_hdf5, "r") as f:
        return np.asarray(f[f"data/{demo_key}/obs/{HDF5_CAM_KEY[cam]}"][...])


def _upsample_grid(grid_16: np.ndarray, target_hw: int = 224) -> np.ndarray:
    """(H, W, C) -> (target_hw, target_hw, C) bilinear; matches what the model sees."""
    t = torch.from_numpy(grid_16).permute(2, 0, 1).unsqueeze(0).float()
    up = F.interpolate(t, size=(target_hw, target_hw), mode="bilinear",
                       align_corners=False)
    return up.squeeze(0).permute(1, 2, 0).numpy()


def _depth_from_logz(log_z_grid: np.ndarray) -> np.ndarray:
    """(..., 1) log-depth -> (..., ) depth in meters (with reasonable clip)."""
    z = np.exp(np.asarray(log_z_grid, dtype=np.float32)[..., 0])
    return np.clip(z, 0.0, 5.0)  # ignore wild outliers for vis


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def fig_episode_overview(
    ep_idx: int, plan: list, gt_root: Path, pi3x_root: Path, out_path: Path,
    n_frames: int = 4,
):
    """Multi-panel figure for one episode, both cams, n_frames sampled evenly."""
    _, task, demo_key, src = plan[ep_idx]
    rgbs = {cam: _load_rgb_episode(src, demo_key, cam) for cam in CAMS}
    gts = {cam: _load_cache_episode(gt_root, cam, ep_idx) for cam in CAMS}
    pi3xs = {cam: _load_cache_episode(pi3x_root, cam, ep_idx) for cam in CAMS}
    with h5py.File(src, "r") as f:
        lang = json.loads(f[f"data/{demo_key}"].attrs["ep_meta"]).get("lang", "")

    T = rgbs[CAMS[0]].shape[0]
    frame_idxs = np.linspace(0, T - 1, n_frames, dtype=int)
    n_cols = 7  # RGB | RGB+GT_overlay | RGB+GT_FLIPPED_overlay (alignment proof)
                #     | GT depth | Pi3X depth | Pi3X conf | |Δ| depth
    n_rows = n_frames * len(CAMS)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2),
        constrained_layout=True,
    )
    title_color = "#222"
    for j, cam in enumerate(CAMS):
        for i, t in enumerate(frame_idxs):
            r = j * n_frames + i
            rgb = rgbs[cam][t]   # (128, 128, 3) uint8
            gt_depth = _depth_from_logz(gts[cam]["log_z"][t])           # (16,16)
            pi3x_depth = _depth_from_logz(pi3xs[cam]["log_z"][t])        # (16,16)
            pi3x_conf = 1.0 / (1.0 + np.exp(-pi3xs[cam]["conf"][t, ..., 0].astype(np.float32)))  # sigmoid
            gt_xy = gts[cam]["xy"][t]                                    # (16,16,2)

            # vmin/vmax depth: shared between GT and Pi3X for fair comparison
            depth_lo = float(min(gt_depth.min(), pi3x_depth.min()))
            depth_hi = float(max(gt_depth.max(), pi3x_depth.max()))

            # RGB → 224×224 for clean overlay arithmetic with the upsampled grid
            from PIL import Image as _PIL
            rgb_224 = np.asarray(_PIL.fromarray(rgb).resize((224, 224)))

            # Depth heatmap upsampled to 224×224. Use the natural-frame GT.
            depth_grid = _upsample_grid(gts[cam]["log_z"][t])[..., 0]   # (224, 224)
            depth_grid_flipped = depth_grid[::-1, ::-1]                  # simulate "broken" pre-fix view

            # Map log_z → RGB colors with viridis, then alpha-blend on RGB.
            cm = plt.get_cmap("viridis")
            normed = (depth_grid - depth_grid.min()) / max(depth_grid.max() - depth_grid.min(), 1e-6)
            depth_color = (cm(normed)[..., :3] * 255).astype(np.uint8)
            overlay = (0.55 * rgb_224 + 0.45 * depth_color).astype(np.uint8)

            normed_f = (depth_grid_flipped - depth_grid_flipped.min()) / max(depth_grid_flipped.max() - depth_grid_flipped.min(), 1e-6)
            depth_color_f = (cm(normed_f)[..., :3] * 255).astype(np.uint8)
            overlay_flipped = (0.55 * rgb_224 + 0.45 * depth_color_f).astype(np.uint8)

            axes[r, 0].imshow(rgb); axes[r, 0].set_title(f"{cam}  t={t}", color=title_color, fontsize=8)
            axes[r, 1].imshow(overlay); axes[r, 1].set_title("RGB + GT depth\n(should align)", fontsize=7)
            axes[r, 2].imshow(overlay_flipped); axes[r, 2].set_title("RGB + flipped GT\n(pre-fix look)", fontsize=7)
            axes[r, 3].imshow(_upsample_grid(gts[cam]["log_z"][t]), cmap="viridis", vmin=np.log(max(depth_lo, 1e-3)), vmax=np.log(max(depth_hi, 1e-3)))
            axes[r, 3].set_title(f"GT log_z\n[{depth_lo:.2f}, {depth_hi:.2f}] m", fontsize=7)
            axes[r, 4].imshow(_upsample_grid(pi3xs[cam]["log_z"][t]), cmap="viridis", vmin=np.log(max(depth_lo, 1e-3)), vmax=np.log(max(depth_hi, 1e-3)))
            axes[r, 4].set_title("Pi3X log_z (same scale)", fontsize=7)
            axes[r, 5].imshow(_upsample_grid(pi3x_conf[..., None]), cmap="RdYlGn", vmin=0, vmax=1)
            axes[r, 5].set_title(f"Pi3X conf σ\nmean={pi3x_conf.mean():.2f}", fontsize=7)
            diff = np.abs(_upsample_grid(gts[cam]["log_z"][t])
                          - _upsample_grid(pi3xs[cam]["log_z"][t]))
            axes[r, 6].imshow(diff[..., 0], cmap="hot")
            axes[r, 6].set_title(f"|Δ log_z|\nmean={float(diff.mean()):.3f}", fontsize=7)
            for c in range(n_cols):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    suptitle = f"episode {ep_idx} — {task} / {demo_key}  ·  language: “{lang}”  ·  T={T}"
    fig.suptitle(suptitle, fontsize=10, y=1.005)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def fig_task_gallery(
    plan: list, gt_root: Path, pi3x_root: Path, out_path: Path,
    frame_at_fraction: float = 0.5,
):
    """One frame per task. Each tile: RGB | GT depth | Pi3X depth (224×224)."""
    by_task: dict[str, tuple[int, str, str, Path]] = {}
    for ep_idx, task, demo, src in plan:
        # first occurrence per task = demo_1 of that task
        if task not in by_task:
            by_task[task] = (ep_idx, task, demo, src)
        if len(by_task) == len(TASKS_24):
            break

    task_list = [t for t in TASKS_24 if t in by_task]
    n = len(task_list)
    cols = 3                   # RGB | GT | Pi3X
    n_rows = n
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 2.4, n_rows * 1.7),
                             constrained_layout=True)
    if n_rows == 1:
        axes = axes[None, :]

    for r, task in enumerate(tqdm.tqdm(task_list, desc="gallery")):
        ep_idx, _, demo, src = by_task[task]
        cam = "base"  # agentview gives the most informative scene shot
        rgb_seq = _load_rgb_episode(src, demo, cam)
        T = rgb_seq.shape[0]
        t = int(T * frame_at_fraction)
        rgb = rgb_seq[t]
        gt = _load_cache_episode(gt_root, cam, ep_idx)
        pi3x = _load_cache_episode(pi3x_root, cam, ep_idx)
        gt_depth = _depth_from_logz(gt["log_z"][t])
        pi3x_depth = _depth_from_logz(pi3x["log_z"][t])
        depth_lo = float(min(gt_depth.min(), pi3x_depth.min()))
        depth_hi = float(max(gt_depth.max(), pi3x_depth.max()))

        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"{task}  ep{ep_idx} t={t}", fontsize=7)
        axes[r, 1].imshow(_upsample_grid(gt["log_z"][t]), cmap="viridis",
                          vmin=np.log(max(depth_lo, 1e-3)), vmax=np.log(max(depth_hi, 1e-3)))
        axes[r, 1].set_title(f"GT  z∈[{depth_lo:.2f},{depth_hi:.2f}]m", fontsize=7)
        axes[r, 2].imshow(_upsample_grid(pi3x["log_z"][t]), cmap="viridis",
                          vmin=np.log(max(depth_lo, 1e-3)), vmax=np.log(max(depth_hi, 1e-3)))
        axes[r, 2].set_title("Pi3X (same scale)", fontsize=7)
        for c in range(cols):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    fig.suptitle("Task gallery — agentview_left, mid-episode frame", fontsize=11, y=1.001)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def fig_depth_histograms(
    plan: list, gt_root: Path, pi3x_root: Path, out_path: Path,
    n_sample_episodes: int = 60,
):
    """Pooled log_z distribution across sampled episodes, GT vs Pi3X, per cam."""
    rng = np.random.default_rng(0)
    sample_idxs = rng.choice(len(plan), size=min(n_sample_episodes, len(plan)),
                             replace=False)
    pools: dict[tuple[str, str], list[np.ndarray]] = {
        (kind, cam): [] for kind in ("GT", "Pi3X") for cam in CAMS
    }
    for ep_idx in tqdm.tqdm(sample_idxs, desc="hist sample"):
        for cam in CAMS:
            pools[("GT", cam)].append(
                _load_cache_episode(gt_root, cam, int(ep_idx))["log_z"].ravel().astype(np.float32)
            )
            pools[("Pi3X", cam)].append(
                _load_cache_episode(pi3x_root, cam, int(ep_idx))["log_z"].ravel().astype(np.float32)
            )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, cam in zip(axes, CAMS):
        gt_all = np.concatenate(pools[("GT", cam)])
        pi3x_all = np.concatenate(pools[("Pi3X", cam)])
        bins = np.linspace(min(gt_all.min(), pi3x_all.min()),
                           max(gt_all.max(), pi3x_all.max()), 80)
        ax.hist(gt_all, bins=bins, alpha=0.55, label=f"GT  μ={gt_all.mean():.2f}", color="#1f77b4")
        ax.hist(pi3x_all, bins=bins, alpha=0.55, label=f"Pi3X μ={pi3x_all.mean():.2f}", color="#d62728")
        ax.set_title(f"log_z distribution — {CAM_TITLES[cam]}  ({len(sample_idxs)} eps)")
        ax.set_xlabel("log_z"); ax.set_ylabel("count")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def fig_conf_distribution(
    plan: list, pi3x_root: Path, out_path: Path, n_sample_episodes: int = 60,
):
    rng = np.random.default_rng(1)
    sample_idxs = rng.choice(len(plan), size=min(n_sample_episodes, len(plan)),
                             replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, cam in zip(axes, CAMS):
        confs = []
        for ep_idx in sample_idxs:
            data = _load_cache_episode(pi3x_root, cam, int(ep_idx))
            confs.append(data["conf"].ravel().astype(np.float32))
        all_conf_logits = np.concatenate(confs)
        sig = 1.0 / (1.0 + np.exp(-all_conf_logits))
        bins = np.linspace(0, 1, 60)
        ax.hist(sig, bins=bins, color="#2ca02c", alpha=0.7)
        ax.set_title(f"Pi3X conf (sigmoid) — {CAM_TITLES[cam]}\n"
                     f"mean={sig.mean():.2f}  median={np.median(sig):.2f}")
        ax.set_xlabel("σ(logit)"); ax.set_ylabel("count")
        ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--bundle", type=Path,
        default=Path("~/Research/robocasa-human50").expanduser(),
        help="Bundle root containing raw_human_im/, gt_point_targets/, pi3x_targets/.",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("~/Research/robocasa24/viz_targets").expanduser(),
    )
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index for the per-episode overview figure.")
    parser.add_argument("--n-frames", type=int, default=4,
                        help="How many frames to sample for the overview.")
    parser.add_argument(
        "--n-sample-episodes", type=int, default=60,
        help="How many random episodes to pool for the histograms.",
    )
    parser.add_argument(
        "--only", choices=["overview", "gallery", "depth-hist", "conf-hist", "all"],
        default="all",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    raw_root = args.bundle / "raw_human_im"
    gt_root = args.bundle / "gt_point_targets"
    pi3x_root = args.bundle / "pi3x_targets"
    for p, name in [(raw_root, "raw_human_im"), (gt_root, "gt_point_targets"),
                    (pi3x_root, "pi3x_targets")]:
        if not p.exists():
            raise SystemExit(f"missing {name} at {p}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("building episode plan...")
    plan = _build_episode_plan(raw_root)
    logger.info("plan has %d episodes across %d tasks", len(plan), len(TASKS_24))

    if args.only in ("overview", "all"):
        if args.episode < 0 or args.episode >= len(plan):
            raise SystemExit(f"--episode out of range [0, {len(plan)})")
        fig_episode_overview(
            args.episode, plan, gt_root, pi3x_root,
            args.out_dir / f"episode_overview_ep{args.episode:06d}.png",
            n_frames=args.n_frames,
        )
    if args.only in ("gallery", "all"):
        fig_task_gallery(plan, gt_root, pi3x_root,
                         args.out_dir / "task_gallery.png")
    if args.only in ("depth-hist", "all"):
        fig_depth_histograms(plan, gt_root, pi3x_root,
                             args.out_dir / "depth_histograms.png",
                             n_sample_episodes=args.n_sample_episodes)
    if args.only in ("conf-hist", "all"):
        fig_conf_distribution(plan, pi3x_root,
                              args.out_dir / "conf_distribution.png",
                              n_sample_episodes=args.n_sample_episodes)
    logger.info("done — figures in %s", args.out_dir)


if __name__ == "__main__":
    main()
