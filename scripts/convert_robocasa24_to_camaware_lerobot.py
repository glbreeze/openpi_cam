#!/usr/bin/env python3
"""Stage 2 of the RoboCasa v0 → cam-aware LeRobot conversion pipeline.

Reads:
  - 24 source mg_im HDF5s under
    <src_root>/v0.1/single_stage/<env>/<Task>/mg/<date>/demo_gentex_im128_randcams.hdf5
    (for the rendered 128x128 RGB frames — left untouched, only read here)
  - 24 stage-1 cam-matrix caches under
    <cache-dir>/cam_cache_<Task>.h5
    (for per-frame K, T_wc, the LeRobot-permuted action, the 16-d state, lang)

Writes:
  - A single multi-task LeRobot v2.1 dataset at
    $HF_LEROBOT_HOME/<repo-id>/
    with 2 cam channels (agent + wrist), 4 cam-aware channels per cam, the
    standard 16-d state / 12-d action, and `annotation.human.task_description`
    (== ep_meta["lang"]) per row. ~72k episodes for the full 24 tasks.

Run in any env that has `lerobot` installed (typical: openpi `.venv`). Does
NOT need robosuite/robocasa/mujoco — that's stage 1.

Image handling:
  - Source frames are HWC uint8 at 128x128.
  - We resize bilinearly to (image_size, image_size), default 224 to match Pi3
    input, and store as CHW uint8 video frames (LeRobot default).
  - K from stage 1 is at SRC_IMAGE_SIZE=128; we scale it to image_size here
    in the standard pinhole way.

Right-wrist cam (`right_wrist_0_rgb`) is NOT written — the policy-side
`RobocasaCamInputs` will pad it with mask=False (mirrors LIBERO single-arm).

Permutation note: the stage-1 cache already stores action in the LeRobot
canonical layout `[base_motion(4), control_mode(1), eef_pos(3), eef_rot(3),
gripper(1)]`. State is `[base_pos(3), base_quat(4), base_to_eef_pos(3),
base_to_eef_quat(4), gripper_qpos(2)]`. No further re-ordering here.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
import re

import h5py
import numpy as np
import torch
import tqdm
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ACTION_DIM = 12
STATE_DIM = 16
SRC_IMAGE_SIZE = 128

# LeRobot row keys — must match what `LeRobotRobocasaCamDataConfig.create()`
# repacks in src/openpi/training/config.py:541-552.
CAM_AGENT_LEROBOT = "robot0_agentview_left"
CAM_WRIST_LEROBOT = "robot0_eye_in_hand"
CAM_AGENT_SHORT = "agentview_left"   # used in observation.<short>_intrinsic etc.
CAM_WRIST_SHORT = "eye_in_hand"

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

logger = logging.getLogger("convert_robocasa24_to_camaware_lerobot")
EPISODE_RE = re.compile(r"episode_(\d{6})\.parquet$")


def _resize_image_chw(img_hwc_uint8: np.ndarray, dst_size: int) -> np.ndarray:
    """HWC uint8 → CHW uint8 bilinear resize (CPU)."""
    chw = torch.from_numpy(img_hwc_uint8).permute(2, 0, 1).float() / 255.0
    chw = torch.nn.functional.interpolate(
        chw[None], size=(dst_size, dst_size), mode="bilinear", align_corners=False
    )[0]
    return (chw * 255.0).clamp(0, 255).to(torch.uint8).numpy()


def _resize_image_seq_chw_gpu(seq_hwc_uint8: np.ndarray, dst_size: int, device: torch.device) -> np.ndarray:
    """Batched (T, H, W, 3) uint8 → (T, 3, dst, dst) uint8 bilinear on GPU.

    Ship the whole episode's frames in one batch; PCIe overhead amortized
    across T frames. ~5–10× faster than per-frame CPU resize for our
    typical T ∈ [200, 700].
    """
    if seq_hwc_uint8.ndim != 4 or seq_hwc_uint8.shape[-1] != 3:
        raise ValueError(f"expected (T, H, W, 3); got {seq_hwc_uint8.shape}")
    t = torch.from_numpy(seq_hwc_uint8).to(device, non_blocking=True)
    t = t.permute(0, 3, 1, 2).float().div_(255.0)
    t = torch.nn.functional.interpolate(t, size=(dst_size, dst_size), mode="bilinear", align_corners=False)
    t = t.mul_(255.0).clamp_(0, 255).to(torch.uint8)
    return t.cpu().numpy()


def _scale_K(K: np.ndarray, src_size: int, dst_size: int) -> np.ndarray:
    s = float(dst_size) / float(src_size)
    out = np.asarray(K, dtype=np.float32).copy()
    out[0, 0] *= s
    out[1, 1] *= s
    out[0, 2] *= s
    out[1, 2] *= s
    return out


def _resolve_src_path(src_root: Path, task: str, source_type: str) -> Path:
    """Locate a task's source HDF5. Patterns differ by source type:
      human_im:  v0.1/single_stage/<env>/<Task>/<date>/demo_gentex_im128_randcams.hdf5
      mg_im:     v0.1/single_stage/<env>/<Task>/mg/<date>/demo_gentex_im128_randcams.hdf5
    """
    if source_type == "human_im":
        pattern = f"*/{task}/*/demo_gentex_im128_randcams.hdf5"
        candidates = [p for p in (src_root / "v0.1" / "single_stage").glob(pattern)
                      if "/mg/" not in str(p)]
    elif source_type == "mg_im":
        pattern = f"*/{task}/mg/*/demo_gentex_im128_randcams.hdf5"
        candidates = list((src_root / "v0.1" / "single_stage").glob(pattern))
    else:
        raise ValueError(f"unknown source_type {source_type!r}; expected human_im or mg_im")
    candidates = sorted(candidates)
    if not candidates:
        raise FileNotFoundError(
            f"No {source_type} HDF5 found for {task} under {src_root}. "
            f"Expected pattern: v0.1/single_stage/<env>/{task}/"
            f"{'mg/' if source_type == 'mg_im' else ''}<date>/demo_gentex_im128_randcams.hdf5"
        )
    if len(candidates) > 1:
        logger.warning("Multiple %s candidates for %s; picking %s", source_type, task, candidates[0])
    return candidates[0]


def _create_dataset(
    repo_id: str,
    image_size: int,
    use_videos: bool = True,
    image_writer_threads: int = 8,
    resume: bool = False,
) -> LeRobotDataset:
    """Build a fresh LeRobot v2.1 dataset writer with the cam-aware schema.

    Wipes any existing dataset at the same repo_id, so a re-run starts clean.
    With `use_videos=False`, frames are stored as image files (PNG) per
    frame instead of as MP4 video — ~10× larger on disk but skips the
    SVT-AV1 encode that dominates wall time, and is bit-exact to the
    source uint8 frames.
    """
    cams = [CAM_AGENT_LEROBOT, CAM_WRIST_LEROBOT]
    cams_short = [CAM_AGENT_SHORT, CAM_WRIST_SHORT]
    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32", "shape": (STATE_DIM,),
            "names": [
                "base_pos.x", "base_pos.y", "base_pos.z",
                "base_quat.w", "base_quat.x", "base_quat.y", "base_quat.z",
                "base_to_eef_pos.x", "base_to_eef_pos.y", "base_to_eef_pos.z",
                "base_to_eef_quat.w", "base_to_eef_quat.x", "base_to_eef_quat.y",
                "base_to_eef_quat.z",
                "gripper_qpos.0", "gripper_qpos.1",
            ],
        },
        "action": {
            "dtype": "float32", "shape": (ACTION_DIM,),
            "names": [
                "base_motion.0", "base_motion.1", "base_motion.2", "base_motion.3",
                "control_mode",
                "eef_pos.x", "eef_pos.y", "eef_pos.z",
                "eef_rot.0", "eef_rot.1", "eef_rot.2",
                "gripper",
            ],
        },
    }
    image_dtype = "video" if use_videos else "image"
    for cam in cams:
        features[f"observation.images.{cam}"] = {
            "dtype": image_dtype,
            "shape": (3, image_size, image_size),
            "names": ["channels", "height", "width"],
        }
    for cam_short in cams_short:
        features[f"observation.{cam_short}_extrinsic"] = {
            "dtype": "float32", "shape": (4, 4),
            "names": [["rows"], ["cols"]],
        }
        features[f"observation.{cam_short}_intrinsic"] = {
            "dtype": "float32", "shape": (3, 3),
            "names": [["rows"], ["cols"]],
        }

    out_root = HF_LEROBOT_HOME / repo_id
    if resume and out_root.exists():
        logger.info("resuming existing dataset at %s", out_root)
        dataset = LeRobotDataset(repo_id=repo_id, root=out_root)
        if image_writer_threads > 0:
            dataset.start_image_writer(num_threads=image_writer_threads)
        _cleanup_incomplete_resume_state(dataset)
        return dataset

    if out_root.exists():
        logger.warning("removing existing dataset at %s", out_root)
        shutil.rmtree(out_root)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=20,                # RoboCasa v0 sim runs at 20 Hz
        robot_type="panda_mobile",
        features=features,
        use_videos=use_videos,
        # Background PNG writes so add_frame doesn't block on disk I/O.
        # Big speedup when use_videos=False since every frame round-trips
        # through PNG.
        image_writer_threads=image_writer_threads,
    )


def _cleanup_incomplete_resume_state(dataset: LeRobotDataset) -> None:
    """Delete artifacts from a timed-out final episode so append can restart cleanly.

    LeRobot writes the parquet file before updating metadata. If a job dies in
    between, we can end up with an "orphan" parquet at episode_index ==
    meta.total_episodes. Resume mode treats the metadata as source of truth and
    removes any files beyond that boundary before appending.
    """
    expected_episodes = dataset.meta.total_episodes

    for parquet_path in (dataset.root / "data").rglob("episode_*.parquet"):
        match = EPISODE_RE.search(parquet_path.name)
        if match and int(match.group(1)) >= expected_episodes:
            logger.warning("removing orphan parquet from interrupted run: %s", parquet_path)
            parquet_path.unlink()

    if dataset.root.joinpath("images").is_dir():
        logger.warning("removing staged images from interrupted run: %s", dataset.root / "images")
        shutil.rmtree(dataset.root / "images")


def _convert_demo(
    dataset: LeRobotDataset,
    src_demo: h5py.Group,
    cache_demo: h5py.Group,
    *,
    image_size: int,
    device: torch.device,
) -> int:
    """Write one demo as one LeRobot episode. Returns frame count."""
    K_agent = np.asarray(cache_demo["K_agent"][:], dtype=np.float32)
    K_wrist = np.asarray(cache_demo["K_wrist"][:], dtype=np.float32)
    T_wc_agent = np.asarray(cache_demo["T_wc_agent"][:], dtype=np.float32)
    T_wc_wrist = np.asarray(cache_demo["T_wc_wrist"][:], dtype=np.float32)
    action = np.asarray(cache_demo["action"][:], dtype=np.float32)
    state = np.asarray(cache_demo["state"][:], dtype=np.float32)
    language = str(cache_demo.attrs["language"])

    T = int(action.shape[0])
    assert state.shape == (T, STATE_DIM)
    assert T_wc_agent.shape == (T, 4, 4)
    assert T_wc_wrist.shape == (T, 4, 4)

    if not language:
        # Defensive: the v0 ep_meta should always have a lang string, but
        # don't poison the dataset with empty prompts.
        language = "(missing language)"

    # K is constant per episode in stage 1 (one (3,3) array per cam). Scale
    # it once here and broadcast per-frame for the LeRobot row schema.
    K_agent_scaled = _scale_K(K_agent, SRC_IMAGE_SIZE, image_size)
    K_wrist_scaled = _scale_K(K_wrist, SRC_IMAGE_SIZE, image_size)

    # Source frames live in the original HDF5 (the cache doesn't duplicate
    # them). Both cams have (T, 128, 128, 3) uint8.
    agent_seq = np.asarray(src_demo["obs/robot0_agentview_left_image"][:])
    wrist_seq = np.asarray(src_demo["obs/robot0_eye_in_hand_image"][:])
    assert agent_seq.shape[0] == T, f"agent frames ({agent_seq.shape[0]}) != T ({T})"
    assert wrist_seq.shape[0] == T, f"wrist frames ({wrist_seq.shape[0]}) != T ({T})"

    # Batched resize. On CUDA this ships the whole episode in one PCIe round
    # trip; on CPU it falls back to a vectorized torch interpolate.
    if device.type == "cuda":
        agent_chw_seq = _resize_image_seq_chw_gpu(agent_seq, image_size, device)
        wrist_chw_seq = _resize_image_seq_chw_gpu(wrist_seq, image_size, device)
    else:
        # CPU: still batched via interpolate (much faster than per-frame loop).
        a = torch.from_numpy(agent_seq).permute(0, 3, 1, 2).float().div_(255.0)
        w = torch.from_numpy(wrist_seq).permute(0, 3, 1, 2).float().div_(255.0)
        a = torch.nn.functional.interpolate(a, size=(image_size, image_size),
                                            mode="bilinear", align_corners=False)
        w = torch.nn.functional.interpolate(w, size=(image_size, image_size),
                                            mode="bilinear", align_corners=False)
        agent_chw_seq = a.mul_(255.0).clamp_(0, 255).to(torch.uint8).numpy()
        wrist_chw_seq = w.mul_(255.0).clamp_(0, 255).to(torch.uint8).numpy()

    for t in range(T):
        frame = {
            "observation.state": torch.from_numpy(state[t]),
            "action": torch.from_numpy(action[t]),
            "task": language,
            f"observation.images.{CAM_AGENT_LEROBOT}": agent_chw_seq[t],
            f"observation.images.{CAM_WRIST_LEROBOT}": wrist_chw_seq[t],
            f"observation.{CAM_AGENT_SHORT}_intrinsic": K_agent_scaled,
            f"observation.{CAM_AGENT_SHORT}_extrinsic": T_wc_agent[t],
            f"observation.{CAM_WRIST_SHORT}_intrinsic": K_wrist_scaled,
            f"observation.{CAM_WRIST_SHORT}_extrinsic": T_wc_wrist[t],
        }
        dataset.add_frame(frame)
    dataset.save_episode()
    return T


def _convert_task(
    dataset: LeRobotDataset,
    src_path: Path,
    cache_path: Path,
    *,
    image_size: int,
    max_demos: int | None,
    skip_first_n: int,
    device: torch.device,
    outer_pbar: "tqdm.tqdm | None" = None,
) -> tuple[int, int]:
    """Returns (n_demos_written, total_frames_written) for this task."""
    n_demos = 0
    n_frames = 0
    with h5py.File(src_path, "r") as src, h5py.File(cache_path, "r") as cache:
        task_name = str(cache.attrs["task_name"])
        cache_demos = sorted(cache["demos"].keys(), key=lambda s: int(s.split("_")[1]))
        if max_demos is not None:
            cache_demos = cache_demos[:max_demos]
        if skip_first_n:
            logger.info("[%s] resuming: skipping first %d completed demos", task_name, skip_first_n)
            cache_demos = cache_demos[skip_first_n:]
        logger.info("[%s] converting %d demos", task_name, len(cache_demos))

        inner = tqdm.tqdm(cache_demos, desc=f"  {task_name}", leave=False, unit="demo")
        for demo_key in inner:
            if demo_key not in src["data"]:
                logger.warning("[%s] %s in cache but missing in source HDF5; skipping",
                               task_name, demo_key)
                continue
            T = _convert_demo(
                dataset,
                src["data"][demo_key],
                cache["demos"][demo_key],
                image_size=image_size,
                device=device,
            )
            n_demos += 1
            n_frames += T
            inner.set_postfix_str(f"frames={n_frames}")
            if outer_pbar is not None:
                outer_pbar.set_postfix_str(f"task={task_name} demos={n_demos} frames={n_frames}")
        inner.close()
    return n_demos, n_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src-root", type=Path,
        default=Path("/home/asus/Research/robocasa24/src/robocasa/datasets"),
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path("/home/asus/Research/robocasa24/cam_matrix_cache"),
    )
    parser.add_argument(
        "--repo-id", default="robocasa24/all24_human_camaware",
        help="LeRobot repo id; output lands at $HF_LEROBOT_HOME/<repo-id>/. "
             "Convention: use 'all24_human_camaware' for human_im source, "
             "'all24_mg_camaware' if you later re-pull mg_im.",
    )
    parser.add_argument(
        "--source-type",
        choices=["human_im", "mg_im"],
        default="human_im",
        help="Must match the --source-type used in stage 1.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--videos",
        action="store_true",
        help="Store frames as MP4 video (lerobot default). Off by default "
             "because the SVT-AV1 encode + PNG round-trip dominates wall "
             "time. With videos off, frames live as per-frame PNGs in the "
             "dataset — ~10× larger on disk but ~3-5× faster total and "
             "bit-exact to source.",
    )
    parser.add_argument(
        "--image-writer-threads", type=int, default=8,
        help="Background threads for lerobot's PNG image writer. >0 makes "
             "add_frame non-blocking on disk I/O — big speedup with --no-"
             "videos (default).",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "auto"], default="auto",
        help="Where to run the image resize (128 → image-size). 'auto' = "
             "cuda if available. Image resize is a small fraction of total "
             "time, so the speedup is modest; main GPU value is freeing "
             "CPU for PNG encoding.",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Subset of task names. Default: all 24.",
    )
    parser.add_argument(
        "--max-demos-per-task", type=int, default=None,
        help="Cap demos per task for smoke testing.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previously interrupted run by preserving completed episodes "
             "and continuing from the first missing one.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    tasks = args.tasks if args.tasks else list(TASKS_24)
    unknown = [t for t in tasks if t not in TASKS_24]
    if unknown:
        raise SystemExit(f"Unknown task(s): {unknown}. Valid: {TASKS_24}")

    # Resolve paths up front so we fail fast if anything is missing.
    pairs: list[tuple[str, Path, Path]] = []
    for task in tasks:
        try:
            src = _resolve_src_path(args.src_root, task, args.source_type)
        except FileNotFoundError as exc:
            logger.warning("skip %s: %s", task, exc)
            continue
        cache = args.cache_dir / f"cam_cache_{task}.h5"
        if not cache.exists():
            logger.warning("skip %s: stage-1 cache missing at %s", task, cache)
            continue
        pairs.append((task, src, cache))

    if not pairs:
        raise SystemExit("No (source, cache) pairs to convert.")

    # Device selection. Resize is CPU-bound otherwise; CUDA is a modest win
    # (a few ms saved per demo) but useful for keeping CPU available for
    # PNG encoding done by lerobot's image writer threads.
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    use_videos = bool(args.videos)
    logger.info("creating LeRobot dataset at %s (videos=%s, device=%s, "
                "image_writer_threads=%d)",
                HF_LEROBOT_HOME / args.repo_id, use_videos, device,
                args.image_writer_threads)
    dataset = _create_dataset(
        args.repo_id, args.image_size, use_videos=use_videos,
        image_writer_threads=args.image_writer_threads,
        resume=args.resume,
    )
    remaining_skip = dataset.meta.total_episodes if args.resume else 0
    if args.resume:
        logger.info("resume mode: preserving %d completed episodes", remaining_skip)

    total_demos = 0
    total_frames = 0
    outer = tqdm.tqdm(pairs, desc="tasks", unit="task", position=0)
    for task, src, cache in outer:
        skip_first_n = 0
        with h5py.File(cache, "r") as cache_h5:
            cache_demos = sorted(cache_h5["demos"].keys(), key=lambda s: int(s.split("_")[1]))
            if args.max_demos_per_task is not None:
                cache_demos = cache_demos[:args.max_demos_per_task]
            if remaining_skip > 0:
                skip_first_n = min(remaining_skip, len(cache_demos))
                remaining_skip -= skip_first_n
        outer.set_description(f"task {task}")
        n_d, n_f = _convert_task(
            dataset, src, cache,
            image_size=args.image_size,
            max_demos=args.max_demos_per_task,
            skip_first_n=skip_first_n,
            device=device,
            outer_pbar=outer,
        )
        total_demos += n_d
        total_frames += n_f
        outer.set_postfix_str(f"done={total_demos} demos / {total_frames} frames")
        logger.info("[%s] +%d demos / +%d frames (totals: %d / %d)",
                    task, n_d, n_f, total_demos, total_frames)
    outer.close()

    logger.info("done — %d demos, %d frames at %s",
                total_demos, total_frames, HF_LEROBOT_HOME / args.repo_id)


if __name__ == "__main__":
    main()
