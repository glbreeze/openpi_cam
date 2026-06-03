"""Visualize old (buggy) vs fixed agentview depth overlaid on the dataset RGB.

Reads the npz dumps from render_robocasa24_depth_camfix.py and produces, per
episode, a grid: rows = sampled frames, cols =
  [RGB] [RGB + OLD depth] [RGB + NEW depth] [OLD log_z] [NEW log_z]

Run: /home/asus/Research/openpi_cam/.venv/bin/python <this>
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REN = Path("~/Research/robocasa24/viz_targets_224/camfix_render").expanduser()
OUT = Path("~/Research/robocasa24/viz_targets_224").expanduser()


def rgb_arm_mask(rgb224):
    """White robot arm: bright + low saturation."""
    r, g, b = rgb224[..., 0] / 255., rgb224[..., 1] / 255., rgb224[..., 2] / 255.
    mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
    return (mx > 0.62) & ((mx - mn) < 0.12)


def depth_arm_mask(depth224, area_frac):
    """Nearest `area_frac` of pixels = foreground arm."""
    thr = np.quantile(depth224, area_frac)
    return depth224 < thr


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def main():
    files = sorted(REN.glob("ep*.npz"))
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        rgb = z["rgb"]                       # (F,128,128,3)
        zo = z["depth_old"]; zn = z["depth_new"]               # (F,224,224) meters
        do = np.log(np.clip(zo, 1e-3, None)); dn = np.log(np.clip(zn, 1e-3, None))
        fidx = z["frame_idxs"]; task = str(z["task"]); demo = str(z["demo"]); lang = str(z["lang"])
        F = rgb.shape[0]
        fig, axes = plt.subplots(F, 4, figsize=(4 * 2.6, F * 2.6), constrained_layout=True)
        if F == 1:
            axes = axes[None, :]
        for r in range(F):
            rgb224 = np.asarray(Image.fromarray(rgb[r]).resize((224, 224)))
            mR = rgb_arm_mask(rgb224)
            af = float(mR.mean())
            mo = depth_arm_mask(zo[r], af); mn_ = depth_arm_mask(zn[r], af)
            iou_old, iou_new = iou(mR, mo), iou(mR, mn_)
            lo, hi = float(dn[r].min()), float(np.percentile(dn[r], 99))

            axes[r, 0].imshow(rgb[r]); axes[r, 0].set_title(f"RGB arm mask  t={fidx[r]}", fontsize=8)
            # contour overlay: RGB + OLD(red) + NEW(lime) arm silhouettes
            axes[r, 1].imshow(rgb224)
            axes[r, 1].contour(mo.astype(float), levels=[0.5], colors=["red"], linewidths=1.4)
            axes[r, 1].contour(mn_.astype(float), levels=[0.5], colors=["lime"], linewidths=1.4)
            axes[r, 1].set_title(f"arm contour on RGB\nOLD(red) IoU={iou_old:.2f}  "
                                 f"NEW(lime) IoU={iou_new:.2f}", fontsize=7.5)
            axes[r, 2].imshow(do[r], cmap="viridis", vmin=lo, vmax=hi)
            axes[r, 2].set_title("OLD log_z (default cam)", fontsize=8, color="#b00")
            axes[r, 3].imshow(dn[r], cmap="viridis", vmin=lo, vmax=hi)
            axes[r, 3].set_title("NEW log_z (ep_meta cam)", fontsize=8, color="#070")
            for c in range(4):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        fig.suptitle(f"{task}/{demo}  ·  “{lang}”  ·  agentview depth arm-silhouette: OLD vs FIXED",
                     fontsize=11, y=1.004)
        out = OUT / f"camfix_overlay_{fp.stem}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
