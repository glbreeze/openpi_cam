"""Visualize the newly regenerated (camfix) RoboCasa-v0 GT 224 depth cache.

Reads directly from the two full caches on disk (no MuJoCo re-render):
  OLD (buggy, default cam):   ~/.cache/openpi/gt_point_targets_grid224/robocasa24_all24_human_camaware
  NEW (camfix, ep_meta cam):  ~/.cache/openpi/gt_point_targets_grid224_camfix/robocasa24_all24_human_camaware
plus source RGB from the raw HDF5s (episode plan in canonical TASKS_24 order).

Produces under --out-dir:
  camfix_episode_ep<N>.png  per cam x frame: [RGB] [RGB+OLD] [RGB+NEW]
                            [arm contour: OLD(red)/NEW(lime)] [|Δ OLD-NEW| log_z]
  camfix_gallery.png        one mid frame per task (agent view): RGB | RGB+NEW depth

Run: /home/asus/Research/openpi_cam/.venv/bin/python scripts/debug/viz_robocasa24_camfix_cache.py
"""
from __future__ import annotations
import argparse, json, logging, os
from pathlib import Path
os.environ.setdefault("MPLBACKEND", "Agg")
import h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("viz_camfix")

TASKS_24 = (
    "PnPCounterToCab", "PnPCabToCounter", "PnPCounterToSink", "PnPSinkToCounter",
    "PnPCounterToMicrowave", "PnPMicrowaveToCounter", "PnPCounterToStove", "PnPStoveToCounter",
    "OpenSingleDoor", "CloseSingleDoor", "OpenDoubleDoor", "CloseDoubleDoor",
    "OpenDrawer", "CloseDrawer",
    "TurnOnSinkFaucet", "TurnOffSinkFaucet", "TurnSinkSpout", "TurnOnStove", "TurnOffStove",
    "CoffeeSetupMug", "CoffeeServeMug", "CoffeePressButton", "TurnOnMicrowave", "TurnOffMicrowave",
)
CAMS = ("base", "wrist")
HDF5_CAM_KEY = {"base": "robot0_agentview_left_image", "wrist": "robot0_eye_in_hand_image"}


def resolve(raw, task):
    c = sorted((raw / "v0.1" / "single_stage").glob(f"*/{task}/*/demo_gentex_im128_randcams.hdf5"))
    c = [p for p in c if "/mg/" not in str(p)]
    return c[0] if c else None


def build_plan(raw):
    plan, cum = [], 0
    for t in TASKS_24:
        src = resolve(raw, t)
        if src is None:
            continue
        with h5py.File(src, "r") as f:
            demos = sorted((k for k in f["data"].keys() if k.startswith("demo_")),
                           key=lambda s: int(s.split("_")[1]))
        for d in demos:
            plan.append((cum, t, d, src)); cum += 1
    return plan


def load_lz(root, cam, ep):
    with np.load(root / cam / f"episode_{ep:06d}.npz") as f:
        return f["log_z"][...].astype(np.float32)[..., 0]   # (T,224,224)


def load_rgb(src, demo, cam):
    with h5py.File(src, "r") as f:
        return np.asarray(f[f"data/{demo}/obs/{HDF5_CAM_KEY[cam]}"][...])  # (T,128,128,3)


def overlay(rgb224, lz, lo, hi):
    cm = plt.get_cmap("viridis")
    n = np.clip((lz - lo) / max(hi - lo, 1e-6), 0, 1)
    dc = (cm(n)[..., :3] * 255).astype(np.uint8)
    return (0.5 * rgb224 + 0.5 * dc).astype(np.uint8)


def arm_mask_rgb(rgb224):
    r, g, b = rgb224[..., 0] / 255., rgb224[..., 1] / 255., rgb224[..., 2] / 255.
    mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
    return (mx > 0.62) & ((mx - mn) < 0.12)


def fig_episode(ep, plan, old_root, new_root, out, n_frames=3):
    _, task, demo, src = plan[ep]
    with h5py.File(src, "r") as f:
        lang = json.loads(f[f"data/{demo}"].attrs["ep_meta"]).get("lang", "")
    rgbs = {c: load_rgb(src, demo, c) for c in CAMS}
    old = {c: load_lz(old_root, c, ep) for c in CAMS}
    new = {c: load_lz(new_root, c, ep) for c in CAMS}
    T = rgbs["base"].shape[0]
    fidx = np.linspace(0, T - 1, n_frames, dtype=int)
    nrow = n_frames * len(CAMS)
    fig, ax = plt.subplots(nrow, 5, figsize=(5 * 2.5, nrow * 2.5), constrained_layout=True)
    for j, cam in enumerate(CAMS):
        for i, t in enumerate(fidx):
            r = j * n_frames + i
            rgb224 = np.asarray(Image.fromarray(rgbs[cam][t]).resize((224, 224)))
            lzo, lzn = old[cam][t], new[cam][t]
            lo, hi = float(lzn.min()), float(np.percentile(lzn, 99))
            mR = arm_mask_rgb(rgb224); af = float(mR.mean())
            mo = lzo < np.quantile(lzo, af); mn = lzn < np.quantile(lzn, af)
            ax[r, 0].imshow(rgb224); ax[r, 0].set_title(f"{cam} t={t}", fontsize=8)
            ax[r, 1].imshow(overlay(rgb224, lzo, lo, hi)); ax[r, 1].set_title("RGB + OLD depth", fontsize=8, color="#b00")
            ax[r, 2].imshow(overlay(rgb224, lzn, lo, hi)); ax[r, 2].set_title("RGB + NEW (camfix)", fontsize=8, color="#070")
            ax[r, 3].imshow(rgb224)
            ax[r, 3].contour(mo.astype(float), levels=[0.5], colors=["red"], linewidths=1.2)
            ax[r, 3].contour(mn.astype(float), levels=[0.5], colors=["lime"], linewidths=1.2)
            ax[r, 3].set_title("near-mask OLD(r)/NEW(g)", fontsize=7.5)
            d = np.abs(lzo - lzn)
            ax[r, 4].imshow(d, cmap="hot"); ax[r, 4].set_title(f"|Δ log_z| mean={d.mean():.2f}", fontsize=8)
            for c in range(5):
                ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    fig.suptitle(f"ep{ep} — {task}/{demo} · “{lang}” · OLD vs NEW(camfix) GT224  "
                 f"(wrist rows should be ~identical; base should realign)", fontsize=10, y=1.003)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig); log.info("wrote %s", out)


def fig_gallery(plan, new_root, out, frac=0.5):
    by_task = {}
    for ep, task, demo, src in plan:
        by_task.setdefault(task, (ep, demo, src))
    tasks = [t for t in TASKS_24 if t in by_task]
    n = len(tasks)
    fig, ax = plt.subplots(n, 2, figsize=(2 * 2.6, n * 1.7), constrained_layout=True)
    for r, task in enumerate(tasks):
        ep, demo, src = by_task[task]
        rgb = load_rgb(src, demo, "base"); t = int(rgb.shape[0] * frac)
        rgb224 = np.asarray(Image.fromarray(rgb[t]).resize((224, 224)))
        lz = load_lz(new_root, "base", ep)[t]
        lo, hi = float(lz.min()), float(np.percentile(lz, 99))
        ax[r, 0].imshow(rgb224); ax[r, 0].set_title(f"{task} ep{ep}", fontsize=7)
        ax[r, 1].imshow(overlay(rgb224, lz, lo, hi)); ax[r, 1].set_title("RGB + NEW depth", fontsize=7, color="#070")
        for c in range(2):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    fig.suptitle("camfix GT224 — agent-view depth overlay, one frame per task", fontsize=11, y=1.002)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig); log.info("wrote %s", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("~/Research/robocasa-human50/raw_human_im").expanduser())
    p.add_argument("--old", type=Path, default=Path("~/.cache/openpi/gt_point_targets_grid224/robocasa24_all24_human_camaware").expanduser())
    p.add_argument("--new", type=Path, default=Path("~/.cache/openpi/gt_point_targets_grid224_camfix/robocasa24_all24_human_camaware").expanduser())
    p.add_argument("--out-dir", type=Path, default=Path("~/Research/robocasa24/viz_camfix").expanduser())
    p.add_argument("--episodes", type=int, nargs="+", default=[0, 700, 1100])
    p.add_argument("--only", choices=["episode", "gallery", "all"], default="all")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args.raw)
    log.info("plan: %d episodes", len(plan))
    if args.only in ("episode", "all"):
        for ep in args.episodes:
            if 0 <= ep < len(plan):
                fig_episode(ep, plan, args.old, args.new, args.out_dir / f"camfix_episode_ep{ep:06d}.png")
    if args.only in ("gallery", "all"):
        fig_gallery(plan, args.new, args.out_dir / "camfix_gallery.png")
    log.info("done -> %s", args.out_dir)


if __name__ == "__main__":
    main()
