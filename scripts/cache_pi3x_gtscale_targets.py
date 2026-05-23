"""Produce a GT-scale-aligned Pi3X point-target cache.

Pi3X's depth is scale-ambiguous: its `log_z` is a *raw* PointHead output whose
metric value is `raw_log_z + log(scale)` (see `cache_pi3x_targets.py`). The
ground-truth (sim/depth) cache, by contrast, stores metric `log_z = log(depth)`.
Because the auxiliary point head is trained with a direct MSE on `log_z`
(`legacy_conf_mse`), the raw Pi3X scale offset — and its frame-to-frame jitter —
is an inconsistent supervisory signal.

The `xy` rays are normalized direction vectors (scale-invariant) and `conf` is a
mask/weight, so *all* of Pi3X's scale ambiguity lives in a single additive
constant in `log_z`. This script estimates that constant per (camera, frame) by
matching Pi3X's depth to the metric GT depth, and writes a new cache with
`log_z := log_z + delta`. `xy` and `conf` are copied through unchanged.

    delta[cam, t] = robust_median over jointly-valid pixels of (logz_gt - logz_pi3x)

It also emits a verification report: how much of the Pi3X-vs-GT depth-target
error is pure global scale (which alignment removes) vs. residual within-frame
relative-depth error (which a single scalar cannot fix). That residual spread is
an offline oracle for the downstream eval — small residual => aligned-Pi3X should
behave like the GT teacher; large residual => alignment buys little.

Layout (input and output identical):
    {root}/{cam_subdir}/episode_{NNNNNN}.npz   keys: xy, log_z, conf  (fp16)

Example (LIBERO 4-suite caches on the cluster):
    python scripts/cache_pi3x_gtscale_targets.py \
        --pi3x-root  $GEO_ROOT/.cache/openpi/pi3x_targets_224/libero_cam_v2 \
        --gt-root    $GEO_ROOT/.cache/openpi/gt_point_targets_224/libero_cam_v2_aligned \
        --out-root   $GEO_ROOT/.cache/openpi/pi3x_targets_224/libero_cam_v2_gtscale \
        --cams agent wrist

Dry run (no writes, just the verification report on the first N episodes):
    python scripts/cache_pi3x_gtscale_targets.py ... --report-only --max-episodes 50
"""

import argparse
import json
import pathlib

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _episode_stems(pi3x_cam_dir: pathlib.Path, gt_cam_dir: pathlib.Path) -> list[str]:
    pi3x = {p.name for p in pi3x_cam_dir.glob("episode_*.npz")}
    gt = {p.name for p in gt_cam_dir.glob("episode_*.npz")}
    common = sorted(pi3x & gt)
    only_pi3x, only_gt = pi3x - gt, gt - pi3x
    if only_pi3x or only_gt:
        print(
            f"  [warn] episode mismatch in {pi3x_cam_dir.parent.name}: "
            f"{len(only_pi3x)} pi3x-only, {len(only_gt)} gt-only, {len(common)} shared"
        )
    return common


def _estimate_episode(
    logz_pi3x: np.ndarray,  # (T, R, R, 1) float32
    conf_pi3x: np.ndarray,  # (T, R, R, 1) float32 logits
    logz_gt: np.ndarray,  # (T, R, R, 1) float32
    conf_gt: np.ndarray,  # (T, R, R, 1) float32 logits
    *,
    pi3x_conf_threshold: float,
    min_valid_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    """Estimate per-frame log-z offsets for one episode (no global fallback yet).

    Returns:
        delta_frame   : (T,) per-frame nanmedian of (logz_gt - logz_pi3x), NaN where
                        fewer than `min_valid_pixels` valid pixels.
        episode_delta : scalar nanmedian over the whole episode (NaN if empty).
        valid         : (T, R, R, 1) jointly-valid mask.
        corr_gap      : median |per-pixel-diff delta - distribution-match delta|, an
                        orientation/correspondence sanity number (NaN if no data).
        counts        : dict of frame counts by resolution level.
    """
    valid = (
        (conf_gt > 0.0)
        & (_sigmoid(conf_pi3x) >= pi3x_conf_threshold)
        & np.isfinite(logz_gt)
        & np.isfinite(logz_pi3x)
    )
    diff = np.where(valid, logz_gt - logz_pi3x, np.nan)  # (T,R,R,1)
    n_valid = valid.reshape(valid.shape[0], -1).sum(axis=1)  # (T,)

    with np.errstate(invalid="ignore"):
        frame_med = np.nanmedian(diff.reshape(diff.shape[0], -1), axis=1)  # (T,)
        episode_delta = float(np.nanmedian(diff))  # scalar over whole episode
    delta_frame = np.where(n_valid >= min_valid_pixels, frame_med, np.nan).astype(np.float32)

    # Cross-check estimator that does NOT need pixel correspondence: match the
    # medians of the two depth distributions per frame. Large disagreement vs the
    # per-pixel-diff delta signals a spatial (orientation/flip) misalignment.
    with np.errstate(invalid="ignore"):
        logz_gt_v = np.nanmedian(np.where(valid, logz_gt, np.nan).reshape(diff.shape[0], -1), axis=1)
        logz_pi3x_v = np.nanmedian(np.where(valid, logz_pi3x, np.nan).reshape(diff.shape[0], -1), axis=1)
    corr = np.abs(frame_med - (logz_gt_v - logz_pi3x_v))
    corr_gap = float(np.nanmedian(corr)) if np.isfinite(corr).any() else float("nan")

    counts = {
        "n_frames": int(diff.shape[0]),
        "n_frame_level": int(np.isfinite(delta_frame).sum()),
        "n_episode_fallback": int((~np.isfinite(delta_frame)).sum() if np.isfinite(episode_delta) else 0),
        "n_global_fallback": int((~np.isfinite(delta_frame)).sum() if not np.isfinite(episode_delta) else 0),
    }
    return delta_frame, episode_delta, valid, corr_gap, counts


def _resolve_delta(delta_frame: np.ndarray, episode_delta: float, global_delta: float) -> np.ndarray:
    """Apply the frame -> episode -> camera-global fallback ladder (never 0)."""
    ep = episode_delta if np.isfinite(episode_delta) else global_delta
    delta = np.where(np.isfinite(delta_frame), delta_frame, ep)
    return np.where(np.isfinite(delta), delta, global_delta).astype(np.float32)


def _accumulate_residuals(
    logz_pi3x: np.ndarray,
    logz_gt: np.ndarray,
    delta: np.ndarray,
    valid: np.ndarray,
    acc: dict,
) -> None:
    """Accumulate |raw| and |aligned| log-z errors over valid pixels."""
    aligned = logz_pi3x + delta[:, None, None, None]
    v = valid.reshape(-1)
    raw_err = np.abs(logz_pi3x - logz_gt).reshape(-1)[v]
    aln_err = np.abs(aligned - logz_gt).reshape(-1)[v]
    signed = (aligned - logz_gt).reshape(-1)[v]
    # Reservoir-free: keep running quantile material cheaply via sampling.
    acc["raw"].append(raw_err.astype(np.float32))
    acc["aligned"].append(aln_err.astype(np.float32))
    acc["signed_aligned"].append(signed.astype(np.float32))
    acc["delta"].append(delta.astype(np.float32))


def _summarize(acc: dict) -> dict:
    raw = np.concatenate(acc["raw"]) if acc["raw"] else np.array([np.nan], np.float32)
    aln = np.concatenate(acc["aligned"]) if acc["aligned"] else np.array([np.nan], np.float32)
    sgn = np.concatenate(acc["signed_aligned"]) if acc["signed_aligned"] else np.array([np.nan], np.float32)
    delta = np.concatenate(acc["delta"]) if acc["delta"] else np.array([np.nan], np.float32)
    med_raw = float(np.median(raw))
    med_aln = float(np.median(aln))
    return {
        "median_abs_logz_err_raw": med_raw,
        "median_abs_logz_err_aligned": med_aln,
        "error_reduction_frac": float((med_raw - med_aln) / med_raw) if med_raw > 0 else float("nan"),
        "aligned_residual_std": float(np.std(sgn)),
        "aligned_residual_iqr": float(np.subtract(*np.percentile(sgn, [75, 25]))),
        "delta_mean": float(np.mean(delta)),
        "delta_std": float(np.std(delta)),
        "delta_min": float(np.min(delta)),
        "delta_max": float(np.max(delta)),
        "n_valid_pixels_sampled": int(raw.size),
    }


def _load_episode(pi3x_dir, gt_dir, stem):
    with np.load(pi3x_dir / stem) as fp:
        xy = fp["xy"]
        logz_pi3x = fp["log_z"].astype(np.float32)
        conf_pi3x = fp["conf"].astype(np.float32)
    with np.load(gt_dir / stem) as fg:
        logz_gt = fg["log_z"].astype(np.float32)
        conf_gt = fg["conf"].astype(np.float32)
    if logz_pi3x.shape != logz_gt.shape:
        raise ValueError(f"{stem}: shape mismatch pi3x {logz_pi3x.shape} vs gt {logz_gt.shape}")
    return xy, logz_pi3x, conf_pi3x, logz_gt, conf_gt


def process_cam(
    pi3x_root: pathlib.Path,
    gt_root: pathlib.Path,
    out_root: pathlib.Path | None,
    cam_subdir: str,
    *,
    pi3x_conf_threshold: float,
    min_valid_pixels: int,
    max_episodes: int | None,
    report_only: bool,
) -> dict:
    pi3x_dir = pi3x_root / cam_subdir
    gt_dir = gt_root / cam_subdir
    if not pi3x_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError(f"missing cam subdir: {pi3x_dir} or {gt_dir}")

    stems = _episode_stems(pi3x_dir, gt_dir)
    if max_episodes is not None:
        stems = stems[:max_episodes]

    # ---- Pass 1: estimate per-frame / per-episode deltas; derive camera-global delta ----
    est = {}  # stem -> (delta_frame, episode_delta)
    all_frame_deltas, corr_gaps, counts = [], [], {"n_frames": 0, "n_frame_level": 0, "n_episode_fallback": 0, "n_global_fallback": 0}
    for i, stem in enumerate(stems):
        _, logz_pi3x, conf_pi3x, logz_gt, conf_gt = _load_episode(pi3x_dir, gt_dir, stem)
        delta_frame, episode_delta, _, corr_gap, c = _estimate_episode(
            logz_pi3x, conf_pi3x, logz_gt, conf_gt,
            pi3x_conf_threshold=pi3x_conf_threshold, min_valid_pixels=min_valid_pixels,
        )
        est[stem] = (delta_frame, episode_delta)
        finite = delta_frame[np.isfinite(delta_frame)]
        if finite.size:
            all_frame_deltas.append(finite)
        if np.isfinite(corr_gap):
            corr_gaps.append(corr_gap)
        for k in counts:
            counts[k] += c[k]
        if (i + 1) % 400 == 0:
            print(f"    [pass1] {cam_subdir}: {i + 1}/{len(stems)}")

    global_delta = float(np.median(np.concatenate(all_frame_deltas))) if all_frame_deltas else 0.0

    # ---- Pass 2: resolve fallback ladder, accumulate residuals, write ----
    acc = {"raw": [], "aligned": [], "signed_aligned": [], "delta": []}
    if not report_only and out_root is not None:
        (out_root / cam_subdir).mkdir(parents=True, exist_ok=True)
    for i, stem in enumerate(stems):
        xy, logz_pi3x, conf_pi3x, logz_gt, conf_gt = _load_episode(pi3x_dir, gt_dir, stem)
        delta_frame, episode_delta = est[stem]
        delta = _resolve_delta(delta_frame, episode_delta, global_delta)
        valid = (
            (conf_gt > 0.0) & (_sigmoid(conf_pi3x) >= pi3x_conf_threshold)
            & np.isfinite(logz_gt) & np.isfinite(logz_pi3x)
        )
        _accumulate_residuals(logz_pi3x, logz_gt, delta, valid, acc)
        if not report_only and out_root is not None:
            aligned_logz = (logz_pi3x + delta[:, None, None, None]).astype(np.float16)
            np.savez(
                out_root / cam_subdir / stem,
                xy=xy,  # unchanged (already fp16)
                log_z=aligned_logz,
                conf=conf_pi3x.astype(np.float16),  # unchanged
            )
        if (i + 1) % 400 == 0:
            print(f"    [pass2] {cam_subdir}: {i + 1}/{len(stems)}")

    summary = _summarize(acc)
    n_frames = max(counts["n_frames"], 1)
    summary.update(
        {
            "cam": cam_subdir,
            "n_episodes": len(stems),
            "n_frames": counts["n_frames"],
            "global_delta": global_delta,
            "frac_frame_level": counts["n_frame_level"] / n_frames,
            "frac_episode_fallback": counts["n_episode_fallback"] / n_frames,
            "frac_global_fallback": counts["n_global_fallback"] / n_frames,
            "corr_check_median_abs_gap": float(np.median(corr_gaps)) if corr_gaps else float("nan"),
        }
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pi3x-root", required=True, type=pathlib.Path)
    ap.add_argument("--gt-root", required=True, type=pathlib.Path)
    ap.add_argument("--out-root", type=pathlib.Path, default=None)
    ap.add_argument("--cams", nargs="+", default=["agent", "wrist"], help="cache subdir names")
    ap.add_argument("--pi3x-conf-threshold", type=float, default=0.5, help="sigmoid(conf) gate for pi3x pixels")
    ap.add_argument("--min-valid-pixels", type=int, default=16, help="per-frame min before episode fallback")
    ap.add_argument("--max-episodes", type=int, default=None, help="limit episodes (smoke test)")
    ap.add_argument("--report-only", action="store_true", help="compute + report deltas, write nothing")
    args = ap.parse_args()

    if not args.report_only and args.out_root is None:
        ap.error("--out-root is required unless --report-only")
    if args.out_root is not None:
        out_res = args.out_root.resolve()
        if out_res == args.pi3x_root.resolve() or out_res == args.gt_root.resolve():
            ap.error("--out-root must differ from --pi3x-root and --gt-root")

    print(f"pi3x: {args.pi3x_root}")
    print(f"gt:   {args.gt_root}")
    print(f"out:  {args.out_root if not args.report_only else '(report-only)'}")
    print(f"cams: {args.cams}  conf_thr={args.pi3x_conf_threshold}\n")

    summaries = []
    for cam in args.cams:
        print(f"[{cam}] processing...")
        s = process_cam(
            args.pi3x_root,
            args.gt_root,
            args.out_root,
            cam,
            pi3x_conf_threshold=args.pi3x_conf_threshold,
            min_valid_pixels=args.min_valid_pixels,
            max_episodes=args.max_episodes,
            report_only=args.report_only,
        )
        summaries.append(s)
        print(
            f"  delta: mean={s['delta_mean']:+.3f} std={s['delta_std']:.3f} "
            f"[{s['delta_min']:+.3f}, {s['delta_max']:+.3f}]  global_fallback_delta={s['global_delta']:+.3f}"
        )
        print(
            f"  |log_z err| median: raw={s['median_abs_logz_err_raw']:.4f} -> "
            f"aligned={s['median_abs_logz_err_aligned']:.4f} "
            f"({100 * s['error_reduction_frac']:.1f}% reduction)"
        )
        print(
            f"  aligned residual: std={s['aligned_residual_std']:.4f} iqr={s['aligned_residual_iqr']:.4f}  "
            f"(this is the within-frame relative-depth error scale cannot fix)"
        )
        print(
            f"  delta source: frame={100 * s['frac_frame_level']:.1f}% "
            f"episode={100 * s['frac_episode_fallback']:.1f}% global={100 * s['frac_global_fallback']:.1f}%  "
            f"corr-check gap (orientation sanity)={s['corr_check_median_abs_gap']:.4f}\n"
        )

    print("=== VERIFICATION SUMMARY ===")
    print(json.dumps({s["cam"]: s for s in summaries}, indent=2))
    print(
        "\nReading the oracle: a large `error_reduction_frac` with small "
        "`aligned_residual_std` means the Pi3X<->GT gap is mostly global scale, so "
        "GT-scale-aligned Pi3X should behave like the GT teacher downstream. A small "
        "reduction or large residual std means relative-depth error dominates and "
        "alignment will buy little."
    )


if __name__ == "__main__":
    main()
