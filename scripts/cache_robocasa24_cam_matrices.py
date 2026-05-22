#!/usr/bin/env python3
"""Stage 1 of the RoboCasa v0 → cam-aware LeRobot conversion pipeline.

This script handles the only piece that requires MuJoCo: replaying each demo
through the simulator to extract per-frame camera intrinsics (K) and
camera-to-world extrinsics (T_wc) for {robot0_agentview_left,
robot0_eye_in_hand}. It also does the v0-HDF5 → LeRobot action permutation and
assembles the 16-d state from the obs dict. Output is a small HDF5 cache per
task (~200 MB per 3000-demo task); the heavy 28 GB source HDF5s are left
untouched and only re-read in stage 2.

Run in the `robocasa24` conda env (needs robosuite 1.5.1, robocasa 0.2,
mujoco 3.2.6). Does NOT need lerobot — that's stage 2.

Output layout:
    <output-dir>/cam_cache_<TaskName>.h5
      attrs/{task_name, src_path, image_size}
      /demos/demo_<n>/
        K_agent           (3, 3)        float32  — natural OpenGL K (positive fx)
        K_wrist           (3, 3)        float32
        T_wc_agent        (T, 4, 4)     float32  — OpenCV camera-to-world
        T_wc_wrist        (T, 4, 4)     float32
        action            (T, 12)       float32  — permuted to LeRobot layout
        state             (T, 16)       float32  — [base_pos(3), base_quat(4),
                                                     base_to_eef_pos(3),
                                                     base_to_eef_quat(4),
                                                     gripper_qpos(2)]
        attrs/{language, num_samples}

Coordinate conventions:
    - MuJoCo `cam_xmat` is camera-to-world rotation in OpenGL convention
      (x-right, y-up, z-back). We negate the y and z basis vectors to convert
      to OpenCV (x-right, y-down, z-forward), matching what `RobocasaCamInputs`
      expects and what `libero_policy._mujoco_to_opencv_extrinsic` does.
    - K stored is at the SOURCE image size (128). Stage 2 rescales K to 224
      when it also resizes the images.

Action permutation (verified empirically against OpenDrawer/demo_1):
    v0 HDF5 layout:    [eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)]
    LeRobot layout:    [base_motion(4), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)]
    Index permutation: target[i] = source[ACTION_PERM[i]]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import tqdm

# Source v0 12-d action → LeRobot 12-d action permutation.
# target[i] = source[ACTION_PERM[i]]
ACTION_PERM = np.array([7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
ACTION_DIM = 12
STATE_DIM = 16
SRC_IMAGE_SIZE = 128

CAM_AGENT_NAME = "robot0_agentview_left"
CAM_WRIST_NAME = "robot0_eye_in_hand"

# The 24 manipulation tasks (single_stage minus NavigateKitchen). Used both to
# default `--tasks` and to validate user-supplied task names.
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

logger = logging.getLogger("cache_robocasa24_cam_matrices")


def _mujoco_to_opencv_extrinsic(T_wc_mj: np.ndarray) -> np.ndarray:
    """Convert MuJoCo (OpenGL) camera-to-world to OpenCV camera-to-world.

    OpenGL camera frame: x-right, y-up, z-back   (looking down -z)
    OpenCV camera frame: x-right, y-down, z-fwd  (looking down +z)
    """
    T = np.asarray(T_wc_mj, dtype=np.float32).copy()
    T[:3, 1] = -T[:3, 1]
    T[:3, 2] = -T[:3, 2]
    return T


def _K_from_fovy_deg(fovy_deg: float, image_h: int, image_w: int) -> np.ndarray:
    """Pinhole K from MuJoCo vertical FOV (degrees) + image size."""
    fovy_rad = float(np.deg2rad(fovy_deg))
    focal = (image_h / 2.0) / np.tan(fovy_rad / 2.0)
    return np.array(
        [[focal, 0.0, image_w / 2.0],
         [0.0, focal, image_h / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _build_state(obs_grp: h5py.Group, T: int) -> np.ndarray:
    """Concat the 5 obs channels into the 16-d LeRobot state."""
    parts = [
        np.asarray(obs_grp["robot0_base_pos"][:], dtype=np.float32),         # (T, 3)
        np.asarray(obs_grp["robot0_base_quat"][:], dtype=np.float32),        # (T, 4)
        np.asarray(obs_grp["robot0_base_to_eef_pos"][:], dtype=np.float32),  # (T, 3)
        np.asarray(obs_grp["robot0_base_to_eef_quat"][:], dtype=np.float32), # (T, 4)
        np.asarray(obs_grp["robot0_gripper_qpos"][:], dtype=np.float32),     # (T, 2)
    ]
    state = np.concatenate(parts, axis=1)
    assert state.shape == (T, STATE_DIM), (
        f"state shape {state.shape} != expected ({T}, {STATE_DIM})"
    )
    return state


def _find_camera_id(env, name: str) -> int:
    """Resolve a camera name → MuJoCo body index, version-tolerantly."""
    # robosuite 1.5: env.sim.model.camera_name2id(name)
    model = env.sim.model
    if hasattr(model, "camera_name2id"):
        return int(model.camera_name2id(name))
    # mujoco bindings direct fallback
    import mujoco  # noqa: WPS433  (env-local import)
    return int(mujoco.mj_name2id(model._model, mujoco.mjtObj.mjOBJ_CAMERA, name))


def _cam_fovy(env, cam_id: int) -> float:
    """Read MuJoCo vertical FOV (degrees) for a camera by id."""
    model = env.sim.model
    if hasattr(model, "cam_fovy"):
        return float(model.cam_fovy[cam_id])
    return float(model._model.cam_fovy[cam_id])


def _read_T_wc(env, cam_id: int) -> np.ndarray:
    """Build a 4x4 OpenGL camera-to-world from MuJoCo's cam_xmat / cam_xpos."""
    data = env.sim.data
    R = np.asarray(data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)
    p = np.asarray(data.cam_xpos[cam_id], dtype=np.float32).reshape(3)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _build_env(env_name: str, robots: str):
    """Construct the robosuite env once per task. Cameras are required so that
    cam_xmat / cam_xpos are populated; we don't use the rendered images (saved
    in the source HDF5 already)."""
    # Lazy imports — these are only valid inside the robocasa24 conda env.
    import robosuite_models  # noqa: F401  (registers PandaMobile etc.)
    import robocasa  # noqa: F401  (registers kitchen envs)
    from robocasa.utils.env_utils import create_env

    return create_env(
        env_name=env_name,
        robots=robots,
        camera_names=[CAM_AGENT_NAME, CAM_WRIST_NAME],
        camera_widths=SRC_IMAGE_SIZE,
        camera_heights=SRC_IMAGE_SIZE,
        render_onscreen=False,
        seed=0,
    )


def _convert_task(
    src_path: Path,
    out_path: Path,
    *,
    max_demos: int | None,
    overwrite: bool,
) -> None:
    if out_path.exists() and not overwrite:
        logger.info("skip %s: cache already exists (use --overwrite to regenerate)", out_path)
        return

    # robocasa's playback_dataset.reset_to handles model swap + ep_meta init.
    from robocasa.scripts.playback_dataset import reset_to

    with h5py.File(src_path, "r") as src:
        env_args = json.loads(src["data"].attrs["env_args"])
        task_name = env_args["env_name"]
        robots = env_args["env_kwargs"]["robots"]
        demos = sorted((k for k in src["data"].keys() if k.startswith("demo_")),
                       key=lambda s: int(s.split("_")[1]))
        if max_demos is not None:
            demos = demos[:max_demos]
        logger.info("[%s] %d demos from %s", task_name, len(demos), src_path)

        env = _build_env(task_name, robots)
        try:
            cam_id_agent = _find_camera_id(env, CAM_AGENT_NAME)
            cam_id_wrist = _find_camera_id(env, CAM_WRIST_NAME)
        except KeyError as exc:
            raise RuntimeError(
                f"camera not found in env {task_name}: {exc}. Check robocasa "
                f"v0.2 default CAM_CONFIGS includes {CAM_AGENT_NAME} and {CAM_WRIST_NAME}."
            ) from exc

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp path and rename on success so a partial cache is never
        # accepted by --skip-if-exists logic on the next run.
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        with h5py.File(tmp_path, "w") as out:
            out.attrs["task_name"] = task_name
            out.attrs["src_path"] = str(src_path)
            out.attrs["image_size"] = SRC_IMAGE_SIZE
            out.attrs["cam_agent_name"] = CAM_AGENT_NAME
            out.attrs["cam_wrist_name"] = CAM_WRIST_NAME
            grp = out.create_group("demos")

            for demo_key in tqdm.tqdm(demos, desc=task_name):
                demo = src[f"data/{demo_key}"]
                model_file = demo.attrs["model_file"]
                ep_meta_json = demo.attrs["ep_meta"]
                ep_meta = json.loads(ep_meta_json)
                language = str(ep_meta.get("lang", "")).strip()
                states = np.asarray(demo["states"][:])
                actions_raw = np.asarray(demo["actions"][:], dtype=np.float32)
                T = int(actions_raw.shape[0])
                assert states.shape[0] == T, (
                    f"{demo_key}: states ({states.shape[0]}) vs actions ({T}) mismatch"
                )

                # Construct sim with this demo's MJCF + ep_meta (sets layout,
                # style, fixtures so kinematics resolve correctly).
                reset_to(env, {"model": model_file,
                               "ep_meta": ep_meta_json,
                               "states": states[0]})

                # FOV is constant per episode for both cams; K depends only on
                # fovy + image size, so we compute one K per cam per episode.
                K_agent = _K_from_fovy_deg(_cam_fovy(env, cam_id_agent),
                                           SRC_IMAGE_SIZE, SRC_IMAGE_SIZE)
                K_wrist = _K_from_fovy_deg(_cam_fovy(env, cam_id_wrist),
                                           SRC_IMAGE_SIZE, SRC_IMAGE_SIZE)

                T_wc_agent = np.empty((T, 4, 4), dtype=np.float32)
                T_wc_wrist = np.empty((T, 4, 4), dtype=np.float32)

                for t in range(T):
                    env.sim.set_state_from_flattened(states[t])
                    env.sim.forward()
                    # Store RAW MuJoCo (OpenGL) cam-to-world. The training-time
                    # `RobocasaCamInputs._mujoco_to_opencv_extrinsic` (in py-torch
                    # branch) is responsible for the OpenGL→OpenCV swap. Doing
                    # it here would double-negate.
                    T_wc_agent[t] = _read_T_wc(env, cam_id_agent)
                    T_wc_wrist[t] = _read_T_wc(env, cam_id_wrist)

                action = actions_raw[:, ACTION_PERM]
                state = _build_state(demo["obs"], T)

                d = grp.create_group(demo_key)
                d.attrs["language"] = language
                d.attrs["num_samples"] = T
                d.create_dataset("K_agent", data=K_agent, dtype="float32")
                d.create_dataset("K_wrist", data=K_wrist, dtype="float32")
                d.create_dataset("T_wc_agent", data=T_wc_agent, dtype="float32",
                                 compression="gzip", compression_opts=4)
                d.create_dataset("T_wc_wrist", data=T_wc_wrist, dtype="float32",
                                 compression="gzip", compression_opts=4)
                d.create_dataset("action", data=action, dtype="float32",
                                 compression="gzip", compression_opts=4)
                d.create_dataset("state", data=state, dtype="float32",
                                 compression="gzip", compression_opts=4)

        tmp_path.rename(out_path)
        logger.info("[%s] wrote %s", task_name, out_path)


def _resolve_src_path(src_root: Path, task: str, source_type: str) -> Path:
    """Find a task's HDF5 under the robocasa registry layout.

    Path patterns (per robocasa.utils.dataset_registry):
      human_im:  <src_root>/v0.1/single_stage/<env_folder>/<TaskName>/<date>/demo_gentex_im128_randcams.hdf5
      mg_im:     <src_root>/v0.1/single_stage/<env_folder>/<TaskName>/mg/<date>/demo_gentex_im128_randcams.hdf5
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
            f"No {source_type} HDF5 found for task {task} under {src_root}. "
            f"Did the download finish? Expected pattern: "
            f"v0.1/single_stage/<env>/{task}/"
            f"{'mg/' if source_type == 'mg_im' else ''}<date>/demo_gentex_im128_randcams.hdf5"
        )
    if len(candidates) > 1:
        logger.warning("Multiple %s candidates for %s; picking %s", source_type, task, candidates[0])
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("/home/asus/Research/robocasa24/src/robocasa/datasets"),
        help="Directory containing v0.1/single_stage/<env>/<Task>/mg/<date>/*.hdf5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/asus/Research/robocasa24/cam_matrix_cache"),
        help="Where to write per-task cache HDF5s.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Subset of task names. Default: all 24 manipulation tasks.",
    )
    parser.add_argument(
        "--source-type",
        choices=["human_im", "mg_im"],
        default="human_im",
        help="Which v0 source to read. human_im = ~50 human teleop demos/task "
             "(small, fast); mg_im = ~3000 MimicGen demos/task (paper-baseline "
             "numbers, but ~28 GB per task).",
    )
    parser.add_argument(
        "--max-demos-per-task",
        type=int,
        default=None,
        help="Cap demos per task for smoke testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate caches even if they already exist.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    tasks = args.tasks if args.tasks else list(TASKS_24)
    unknown = [t for t in tasks if t not in TASKS_24]
    if unknown:
        raise SystemExit(f"Unknown task(s): {unknown}. Valid: {TASKS_24}")

    outer = tqdm.tqdm(tasks, desc="tasks", unit="task", position=0)
    for task in outer:
        outer.set_description(f"task {task}")
        try:
            src = _resolve_src_path(args.src_root, task, args.source_type)
        except FileNotFoundError as exc:
            logger.warning("skip %s: %s", task, exc)
            continue

        out_path = args.output_dir / f"cam_cache_{task}.h5"
        _convert_task(src, out_path,
                      max_demos=args.max_demos_per_task,
                      overwrite=args.overwrite)
    outer.close()


if __name__ == "__main__":
    main()
