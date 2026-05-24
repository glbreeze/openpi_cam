"""Cache simulator-GT dense point targets for RoboCasa-v0 distillation.

Produces the same `(xy, log_z, conf)` 16×16 patch-grid cache that
`MixedPointTargetLoader` / `DualPointTargetLoader` consume in
`LeRobotRobocasaCamDataConfig`, so GT and Pi3X targets can be mixed freely:

    {output_root}/{cam_short}/episode_NNNNNN.npz   keys: xy, log_z, conf  fp16

Output episode indexing mirrors `convert_robocasa24_to_camaware_lerobot.py`:
tasks iterated in the canonical TASKS_24 order, demos sorted by integer
suffix; running counter starts at 0 — same one-to-one alignment as the
LeRobot dataset at `robocasa24/all24_human_camaware`.

What this script does per demo:
  1. Build `robocasa.utils.env_utils.create_env(robots='PandaMobile',
     camera_names=[agentview_left, eye_in_hand], camera_widths=224,
     camera_heights=224, camera_depths=True)` once per task.
  2. `reset_to(env, {"model": @model_file, "ep_meta": @ep_meta,
     "states": states[0]})` — same pattern as stage 1.
  3. For each step t: `env.sim.set_state_from_flattened(states[t])`,
     `env.sim.forward()`, then render depth (224×224) for both cams.
  4. `get_real_depth_map(env.sim, depth_raw)` → metric meters.
  5. Apply the openpi `[::-1, ::-1]` flip; build `xy = ((u-cx)/(-fx),
     (v-cy)/fy)` using the same `fx → -fx` flip openpi training applies
     to K (mirrors `cache_libero_gt_point_targets.py`).
  6. Pool (224, 224) → (16, 16) via 14×14 stride-14 avg-pool (matches
     SigLIP patch grid).

Output convention (cam_short → cache subdir → model slot):
    robot0_agentview_left  →  base   (model slot base_0_rgb)
    robot0_eye_in_hand     →  wrist  (model slot left_wrist_0_rgb)

Run in the `robocasa24` conda env (needs robosuite 1.5 + robocasa 0.2 +
mujoco 3.2.6). Does NOT need lerobot.

Wall-time estimate on this machine: depth render in MuJoCo is the
bottleneck (~10-30 ms/frame/cam). 1293 demos × ~276 frames × 2 cams ≈
several hours single-process. Use --tasks to shard across processes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

# IMPORTANT: must be set BEFORE robosuite/mujoco import, otherwise MuJoCo picks
# the slow osmesa CPU rasterizer. EGL is ~50× faster (0.6 ms vs 30 ms per render
# at 224×224 on RTX 4090) — the single biggest speedup available here.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import h5py
import numpy as np
import torch
import torch.nn.functional as functional
import tqdm

logger = logging.getLogger("cache_robocasa24_gt_point_targets")

# Identical to the converter / stage-1 cache to guarantee a 1:1 episode_idx
# alignment with the LeRobot dataset under robocasa24/all24_human_camaware.
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

CAM_AGENT = "robot0_agentview_left"
CAM_WRIST = "robot0_eye_in_hand"
CAM_TO_CACHE = {CAM_AGENT: "base", CAM_WRIST: "wrist"}

VALID_CONF_LOGIT = 10.0
INVALID_CONF_LOGIT = -10.0


def _resolve_src_path(src_root: Path, task: str, source_type: str) -> Path:
    """Same logic as stage 1's _resolve_src_path; duplicated for self-containedness."""
    if source_type == "human_im":
        pattern = f"*/{task}/*/demo_gentex_im128_randcams.hdf5"
        candidates = [p for p in (src_root / "v0.1" / "single_stage").glob(pattern)
                      if "/mg/" not in str(p)]
    elif source_type == "mg_im":
        pattern = f"*/{task}/mg/*/demo_gentex_im128_randcams.hdf5"
        candidates = list((src_root / "v0.1" / "single_stage").glob(pattern))
    else:
        raise ValueError(f"unknown source_type {source_type!r}")
    candidates = sorted(candidates)
    if not candidates:
        raise FileNotFoundError(
            f"No {source_type} HDF5 found for {task} under {src_root}."
        )
    return candidates[0]


def _scale_k(K: np.ndarray, src_hw: int, target_hw: int) -> np.ndarray:
    """Pass-through K (natural convention) with isotropic scale.

    RoboCasa uses the natural y-down image convention end-to-end (see
    `src/openpi/policies/robocasa_policy.py` header on py-torch). Do NOT
    apply the `fx → -fx` flip the LIBERO cache uses.
    """
    scale = float(target_hw) / float(src_hw)
    out = np.asarray(K, dtype=np.float32).copy()
    out[0, 0] *= scale
    out[0, 2] *= scale
    out[1, 1] *= scale
    out[1, 2] *= scale
    return out


def _depth_to_targets_gpu(
    depth_hw: np.ndarray, K: np.ndarray, *, target_hw: int, output_hw: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """depth_hw: (T, H, W) metric meters at H=W=target_hw (already flipped
    + K-adjusted to openpi convention). Returns xy/log_z/conf as
    (T, output_hw, output_hw, *) fp16."""
    T_, H, W = depth_hw.shape
    assert H == W == target_hw
    depth = torch.from_numpy(depth_hw).to(device)                          # (T, H, W)
    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])
    v_grid, u_grid = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    xy_single = torch.stack([(u_grid - cx) / fx, (v_grid - cy) / fy], dim=-1)  # (H, W, 2)
    xy = xy_single.unsqueeze(0).expand(T_, -1, -1, -1).contiguous()             # (T, H, W, 2)

    finite = torch.isfinite(depth) & (depth > 1e-5)
    safe_depth = torch.where(finite, depth, torch.ones_like(depth))
    log_z = torch.log(safe_depth).unsqueeze(-1)                                  # (T, H, W, 1)
    conf = torch.where(
        finite.unsqueeze(-1),
        torch.full_like(log_z, VALID_CONF_LOGIT),
        torch.full_like(log_z, INVALID_CONF_LOGIT),
    )

    if output_hw != target_hw:
        if target_hw % output_hw != 0:
            raise ValueError(f"Cannot pool {target_hw} → {output_hw}")
        k = target_hw // output_hw

        def _pool(t: torch.Tensor) -> torch.Tensor:
            return functional.avg_pool2d(
                t.permute(0, 3, 1, 2), kernel_size=k, stride=k
            ).permute(0, 2, 3, 1)
        xy = _pool(xy); log_z = _pool(log_z); conf = _pool(conf)

    return (
        xy.to(torch.float16).cpu().numpy(),
        log_z.to(torch.float16).cpu().numpy(),
        conf.to(torch.float16).cpu().numpy(),
    )


def _build_env(env_name: str, robots: str, render_hw: int):
    import robosuite_models  # noqa: F401
    import robocasa  # noqa: F401
    from robocasa.utils.env_utils import create_env
    # camera_depths=True triggers offscreen depth render in robosuite.
    env = create_env(
        env_name=env_name,
        robots=robots,
        camera_names=[CAM_AGENT, CAM_WRIST],
        camera_widths=render_hw,
        camera_heights=render_hw,
        render_onscreen=False,
        seed=0,
    )
    return env


def _count_demos(src_path: Path, max_demos: int | None) -> int:
    with h5py.File(src_path, "r") as f:
        n = sum(1 for k in f["data"].keys() if k.startswith("demo_"))
    if max_demos is not None:
        n = min(n, max_demos)
    return n


def _convert_task(
    src_path: Path,
    out_root: Path,
    *,
    episode_offset: int,
    target_hw: int,
    output_hw: int,
    device: torch.device,
    max_demos: int | None,
    resume: bool,
    pbar: "tqdm.tqdm | None" = None,
) -> int:
    """Returns the next episode_offset after this task's demos. If `pbar` is
    supplied, advance it once per demo with task name as postfix."""
    from robocasa.scripts.playback_dataset import reset_to
    from robosuite.utils.camera_utils import get_real_depth_map

    with h5py.File(src_path, "r") as src:
        env_args = json.loads(src["data"].attrs["env_args"])
        task_name = env_args["env_name"]
        robots = env_args["env_kwargs"]["robots"]
        demos = sorted((k for k in src["data"].keys() if k.startswith("demo_")),
                       key=lambda s: int(s.split("_")[1]))
        if max_demos is not None:
            demos = demos[:max_demos]
        logger.info("[%s] %d demos (episode_offset=%d)", task_name, len(demos), episode_offset)

        env = _build_env(task_name, robots, target_hw)
        try:
            for i, demo_key in enumerate(demos):
                ep_idx = episode_offset + i
                if pbar is not None:
                    pbar.set_postfix_str(f"task={task_name} ep={ep_idx}")
                out_paths = {
                    cache: out_root / cache / f"episode_{ep_idx:06d}.npz"
                    for cache in CAM_TO_CACHE.values()
                }
                if resume and all(p.exists() for p in out_paths.values()):
                    continue
                for p in out_paths.values():
                    p.parent.mkdir(parents=True, exist_ok=True)

                demo = src[f"data/{demo_key}"]
                model_file = demo.attrs["model_file"]
                ep_meta_json = demo.attrs["ep_meta"]
                states = np.asarray(demo["states"][:])
                T = states.shape[0]

                reset_to(env, {"model": model_file,
                               "ep_meta": ep_meta_json,
                               "states": states[0]})

                # K is constant per episode (fovy + image size); pull once,
                # then apply the openpi fx flip (no scale: render is at 224).
                fovy_agent = float(env.sim.model.cam_fovy[env.sim.model.camera_name2id(CAM_AGENT)])
                fovy_wrist = float(env.sim.model.cam_fovy[env.sim.model.camera_name2id(CAM_WRIST)])
                f_agent = (target_hw / 2.0) / np.tan(np.deg2rad(fovy_agent) / 2.0)
                f_wrist = (target_hw / 2.0) / np.tan(np.deg2rad(fovy_wrist) / 2.0)
                K_agent_raw = np.array([[f_agent, 0, target_hw / 2.0],
                                        [0, f_agent, target_hw / 2.0],
                                        [0, 0, 1]], dtype=np.float32)
                K_wrist_raw = np.array([[f_wrist, 0, target_hw / 2.0],
                                        [0, f_wrist, target_hw / 2.0],
                                        [0, 0, 1]], dtype=np.float32)
                K_agent_adj = _scale_k(K_agent_raw, target_hw, target_hw)
                K_wrist_adj = _scale_k(K_wrist_raw, target_hw, target_hw)

                depth_agent_seq = np.empty((T, target_hw, target_hw), dtype=np.float32)
                depth_wrist_seq = np.empty((T, target_hw, target_hw), dtype=np.float32)

                for t in range(T):
                    env.sim.set_state_from_flattened(states[t])
                    env.sim.forward()
                    _, d_a = env.sim.render(camera_name=CAM_AGENT, width=target_hw, height=target_hw, depth=True)
                    _, d_w = env.sim.render(camera_name=CAM_WRIST, width=target_hw, height=target_hw, depth=True)
                    # Convert MuJoCo's normalized depth buffer to metric meters,
                    # then apply openpi's [::-1, ::-1] image flip (in-plane).
                    da = get_real_depth_map(env.sim, d_a)
                    dw = get_real_depth_map(env.sim, d_w)
                    # get_real_depth_map can return (H, W, 1); squeeze if so.
                    if da.ndim == 3:
                        da = da[..., 0]
                    if dw.ndim == 3:
                        dw = dw[..., 0]
                    # env.sim.render returns raw y-up OpenGL. Apply flipud
                    # ([::-1, :]) to convert to natural y-down (matches LeRobot
                    # stored RGB). DO NOT also apply fliplr — that was the
                    # original bug we corrected.
                    depth_agent_seq[t] = da[::-1, :].copy()
                    depth_wrist_seq[t] = dw[::-1, :].copy()

                xy_a, lz_a, c_a = _depth_to_targets_gpu(
                    depth_agent_seq, K_agent_adj,
                    target_hw=target_hw, output_hw=output_hw, device=device,
                )
                xy_w, lz_w, c_w = _depth_to_targets_gpu(
                    depth_wrist_seq, K_wrist_adj,
                    target_hw=target_hw, output_hw=output_hw, device=device,
                )
                np.savez_compressed(out_paths["base"], xy=xy_a, log_z=lz_a, conf=c_a)
                np.savez_compressed(out_paths["wrist"], xy=xy_w, log_z=lz_w, conf=c_w)
                if pbar is not None:
                    pbar.update(1)
        finally:
            try:
                env.close()
            except Exception:
                pass
    return episode_offset + len(demos)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--src-root", type=Path,
        default=Path("/home/asus/Research/robocasa24/src/robocasa/datasets"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("~/.cache/openpi/gt_point_targets_224/robocasa24_all24_human_camaware").expanduser(),
        help="Where per-cam .npz files land; matches the convention "
             "`LeRobotRobocasaCamDataConfig.gt_point_targets_root` expects.",
    )
    parser.add_argument(
        "--source-type", choices=["human_im", "mg_im"], default="human_im",
        help="Must match what stage 1 / stage 2 used so episode_idx aligns.",
    )
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Subset of task names. Default: all 24. NOTE: using a "
                             "subset will skip episode_offsets for missing tasks → "
                             "indices will NOT align with the LeRobot dataset.")
    parser.add_argument("--max-demos-per-task", type=int, default=None)
    parser.add_argument("--render-resolution", type=int, default=224,
                        help="Depth render H=W. Match the dataset's image_size.")
    parser.add_argument("--output-resolution", type=int, default=16, choices=(16, 224))
    parser.add_argument("--device", default="cuda",
                        help="Device for the depth→xy/log_z/conf math + pooling. "
                             "Depth rendering itself is CPU-bound MuJoCo regardless.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip episodes whose .npz already exist.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda unavailable; using cpu")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    tasks = args.tasks if args.tasks else list(TASKS_24)
    unknown = [t for t in tasks if t not in TASKS_24]
    if unknown:
        raise SystemExit(f"Unknown task(s): {unknown}")
    if args.tasks is not None:
        logger.warning("Running on a task subset — episode_idx will NOT align with "
                       "the full LeRobot dataset. Only use for smoke testing.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    for cache_dir in CAM_TO_CACHE.values():
        (args.output_root / cache_dir).mkdir(parents=True, exist_ok=True)

    # Resolve the full task plan up-front so we can compute total episode
    # count + pre-skip resumed tasks before opening any env. Iterating
    # TASKS_24 in order keeps episode_idx aligned with the LeRobot dataset.
    plan: list[dict] = []
    cumulative = 0
    for task in TASKS_24:
        try:
            src = _resolve_src_path(args.src_root, task, args.source_type)
        except FileNotFoundError as exc:
            logger.warning("skip %s: %s", task, exc)
            continue
        n = _count_demos(src, args.max_demos_per_task)
        plan.append({
            "task": task, "src": src, "n": n,
            "episode_offset": cumulative,
            "do_convert": task in tasks,
        })
        cumulative += n
    total_episodes = cumulative
    selected_episodes = sum(p["n"] for p in plan if p["do_convert"])
    logger.info("plan: %d tasks present, %d selected, total %d episodes "
                "(%d to process in this run)",
                len(plan), sum(1 for p in plan if p["do_convert"]),
                total_episodes, selected_episodes)

    # One global "episodes" progress bar — total = selected episodes only.
    # `set_postfix_str(task=..., ep=...)` updates as we walk through.
    pbar = tqdm.tqdm(total=selected_episodes, desc="episodes", unit="ep",
                     dynamic_ncols=True)
    for p in plan:
        if not p["do_convert"]:
            logger.info("[%s] skipping (not in --tasks selection)", p["task"])
            continue
        _convert_task(
            p["src"], args.output_root,
            episode_offset=p["episode_offset"],
            target_hw=args.render_resolution,
            output_hw=args.output_resolution,
            device=device,
            max_demos=args.max_demos_per_task,
            resume=args.resume,
            pbar=pbar,
        )
    pbar.close()
    logger.info("done — wrote up to episode_%06d under %s",
                total_episodes - 1, args.output_root)


if __name__ == "__main__":
    main()
