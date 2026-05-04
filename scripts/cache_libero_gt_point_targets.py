"""Cache simulator-GT dense point targets for LIBERO point-head distillation.

This produces the same cache layout consumed by `MixedPointTargetLoader` and
`Pi3xLiberoTargetLoader`:

    {output_root}/{agent,wrist}/episode_{NNNNNN}.npz
        xy    : (T, R, R, 2)
        log_z : (T, R, R, 1)
        conf  : (T, R, R, 1)

The script requires LIBERO-Camera HDF5 files generated with `--use-depth`.
It reads stored depth for `agentview` and `robot0_eye_in_hand`, converts it to
the OpenPI-flipped camera convention, and writes fp16 point-map targets.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import torch
import torch.nn.functional as functional

logger = logging.getLogger("cache_libero_gt_point_targets")


def _as_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _resolve_libero_repo(libero_repo: str | None) -> Path | None:
    if libero_repo:
        return Path(libero_repo).expanduser().resolve()
    return None


def _ensure_libero_imports(libero_repo: Path | None):
    if libero_repo is not None:
        sys.path.insert(0, str(libero_repo))
        sys.path.insert(0, str(libero_repo / "scripts"))
    from libero.libero.envs import TASK_MAPPING
    import libero.libero.utils.utils as libero_utils
    from robosuite.utils import camera_utils

    return libero_utils, TASK_MAPPING, camera_utils


def _postprocess_model_xml(model_xml, libero_utils, libero_repo: Path | None):
    xml_str = libero_utils.postprocess_model_xml(_as_text(model_xml), {})
    root = ET.fromstring(xml_str)
    local_assets_root = (
        libero_repo / "libero" / "libero" / "assets"
        if libero_repo is not None
        else Path(os.getcwd()) / "libero" / "libero" / "assets"
    )
    for tag in ("mesh", "texture"):
        for elem in root.iter(tag):
            file_path = elem.get("file")
            if not file_path:
                continue
            normalized = file_path.replace("\\", "/")
            marker = "/assets/"
            if marker not in normalized:
                continue
            suffix = normalized.split(marker, 1)[1]
            candidate = local_assets_root / suffix
            if candidate.exists():
                elem.set("file", str(candidate))
    return ET.tostring(root, encoding="utf8").decode("utf8")


def _get_camera_intrinsic(sim, camera_name: str, image_h: int, image_w: int) -> np.ndarray:
    cam_id = sim.model.camera_name2id(camera_name)
    fovy_rad = np.deg2rad(float(sim.model.cam_fovy[cam_id]))
    fy = (image_h / 2.0) / np.tan(fovy_rad / 2.0)
    return np.array(
        [[fy, 0.0, image_w / 2.0], [0.0, fy, image_h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _adjust_and_scale_k(K: np.ndarray, src_hw: int, target_hw: int) -> np.ndarray:
    scale = float(target_hw) / float(src_hw)
    out = np.asarray(K, dtype=np.float32).copy()
    out[0, 0] = -out[0, 0] * scale
    out[0, 2] *= scale
    out[1, 1] *= scale
    out[1, 2] *= scale
    return out


def _resize_depth(depth: np.ndarray, target_hw: int) -> np.ndarray:
    depth_t = torch.from_numpy(depth[None, None].astype(np.float32))
    if depth_t.shape[-1] != target_hw or depth_t.shape[-2] != target_hw:
        depth_t = functional.interpolate(depth_t, size=(target_hw, target_hw), mode="nearest")
    return depth_t[0, 0].numpy()


def _depth_to_targets(depth: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    xy = np.stack([(u - cx) / fx, (v - cy) / fy], axis=-1).astype(np.float32)
    valid = np.isfinite(depth) & (depth > 1e-5)
    safe_depth = np.where(valid, depth, 1.0).astype(np.float32)
    log_z = np.log(safe_depth)[..., None].astype(np.float32)
    conf = np.where(valid[..., None], 10.0, -10.0).astype(np.float32)
    return xy, log_z, conf


def _pool_targets(
    xy: np.ndarray, log_z: np.ndarray, conf: np.ndarray, output_resolution: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if output_resolution == xy.shape[0]:
        return xy, log_z, conf
    if xy.shape[0] % output_resolution != 0:
        raise ValueError(f"Cannot pool {xy.shape[0]} to {output_resolution}")
    k = xy.shape[0] // output_resolution

    def pool(arr):
        t = torch.from_numpy(arr).permute(2, 0, 1)[None]
        return functional.avg_pool2d(t, kernel_size=k, stride=k)[0].permute(1, 2, 0).numpy()

    return pool(xy), pool(log_z), pool(conf)


def _append_target_from_depth(
    per_cam: dict,
    out_name: str,
    depth: np.ndarray,
    K: np.ndarray,
    *,
    src_hw: int,
    target_resolution: int,
    output_resolution: int,
):
    depth = np.asarray(depth, dtype=np.float32)[::-1, ::-1].copy()
    depth = _resize_depth(depth, target_resolution)
    K = _adjust_and_scale_k(K, src_hw, target_resolution)
    xy, log_z, conf = _depth_to_targets(depth, K)
    xy, log_z, conf = _pool_targets(xy, log_z, conf, output_resolution)
    per_cam[out_name]["xy"].append(xy.astype(np.float16))
    per_cam[out_name]["log_z"].append(log_z.astype(np.float16))
    per_cam[out_name]["conf"].append(conf.astype(np.float16))


def _init_env(h5_file, render_resolution: int, libero_utils, TASK_MAPPING):
    env_name = h5_file["data"].attrs.get("env", h5_file["data"].attrs.get("env_name"))
    env_args_raw = h5_file["data"].attrs.get("env_args")
    env_info_raw = h5_file["data"].attrs.get("env_info")
    if env_info_raw not in (None, ""):
        env_kwargs = json.loads(_as_text(env_info_raw))
    else:
        env_args_json = json.loads(_as_text(env_args_raw))
        env_kwargs = env_args_json["env_kwargs"]
        if env_name is None:
            env_name = env_args_json.get("env_name")

    problem_info = json.loads(_as_text(h5_file["data"].attrs["problem_info"]))
    problem_name = problem_info["problem_name"]
    bddl_file_name = _as_text(h5_file["data"].attrs["bddl_file_name"])
    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_depths=True,
        camera_names=["robot0_eye_in_hand", "agentview"],
        reward_shaping=True,
        control_freq=20,
        camera_heights=render_resolution,
        camera_widths=render_resolution,
        camera_segmentations=None,
    )
    return TASK_MAPPING[problem_name](**env_kwargs)


def _iter_hdf5_files(dataset_root: Path) -> list[Path]:
    return sorted(dataset_root.rglob("*.hdf5"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--libero-repo", default=None)
    parser.add_argument("--render-resolution", type=int, default=256)
    parser.add_argument("--target-resolution", type=int, default=224)
    parser.add_argument("--output-resolution", type=int, default=224, choices=(16, 224))
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-episodes-per-file", type=int, default=0)
    parser.add_argument("--start-episode-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    libero_repo = _resolve_libero_repo(args.libero_repo)
    libero_utils, TASK_MAPPING, camera_utils = _ensure_libero_imports(libero_repo)

    files = _iter_hdf5_files(Path(args.dataset_root).expanduser())
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {args.dataset_root}")

    output_root = Path(args.output_root).expanduser()
    episode_out_idx = int(args.start_episode_index)
    for path in files:
        logger.info("processing %s", path)
        with h5py.File(path, "r") as f:
            env = None
            episode_names = sorted(f["data"].keys())
            if args.max_episodes_per_file > 0:
                episode_names = episode_names[: args.max_episodes_per_file]

            for ep in episode_names:
                out_paths = {
                    "agent": output_root / "agent" / f"episode_{episode_out_idx:06d}.npz",
                    "wrist": output_root / "wrist" / f"episode_{episode_out_idx:06d}.npz",
                }
                if args.resume and all(p.exists() for p in out_paths.values()):
                    episode_out_idx += 1
                    continue
                for p in out_paths.values():
                    p.parent.mkdir(parents=True, exist_ok=True)

                ep_group = f[f"data/{ep}"]
                obs_group = ep_group["obs"]
                missing_depth = [
                    key for key in ("agentview_depth", "eye_in_hand_depth") if key not in obs_group
                ]
                if missing_depth:
                    raise ValueError(
                        f"{path}/{ep} is missing stored depth datasets {missing_depth}. "
                        "Regenerate LIBERO-Camera HDF5 with --use-camera-obs --use-depth before caching GT targets."
                    )

                if env is None:
                    env = _init_env(f, args.render_resolution, libero_utils, TASK_MAPPING)

                model_xml = _postprocess_model_xml(ep_group.attrs["model_file"], libero_utils, libero_repo)
                per_cam = {name: {"xy": [], "log_z": [], "conf": []} for name in out_paths}

                env.reset()
                env.reset_from_xml_string(model_xml)
                env.sim.reset()

                agent_depths = obs_group["agentview_depth"][()]
                wrist_depths = obs_group["eye_in_hand_depth"][()]
                agent_hw = np.asarray(obs_group.attrs.get("agent_image_size", agent_depths.shape[1:3]))
                wrist_hw = np.asarray(obs_group.attrs.get("wrist_image_size", wrist_depths.shape[1:3]))
                agent_K = np.asarray(
                    obs_group.attrs.get(
                        "agent_intrinsic",
                        _get_camera_intrinsic(env.sim, "agentview", int(agent_hw[0]), int(agent_hw[1])),
                    ),
                    dtype=np.float32,
                )
                wrist_K = np.asarray(
                    obs_group.attrs.get(
                        "wrist_intrinsic",
                        _get_camera_intrinsic(env.sim, "robot0_eye_in_hand", int(wrist_hw[0]), int(wrist_hw[1])),
                    ),
                    dtype=np.float32,
                )
                for agent_depth, wrist_depth in zip(agent_depths, wrist_depths, strict=True):
                    _append_target_from_depth(
                        per_cam,
                        "agent",
                        camera_utils.get_real_depth_map(env.sim, agent_depth),
                        agent_K,
                        src_hw=int(agent_hw[0]),
                        target_resolution=args.target_resolution,
                        output_resolution=args.output_resolution,
                    )
                    _append_target_from_depth(
                        per_cam,
                        "wrist",
                        camera_utils.get_real_depth_map(env.sim, wrist_depth),
                        wrist_K,
                        src_hw=int(wrist_hw[0]),
                        target_resolution=args.target_resolution,
                        output_resolution=args.output_resolution,
                    )

                for out_name, out_path in out_paths.items():
                    np.savez_compressed(
                        out_path,
                        xy=np.stack(per_cam[out_name]["xy"], axis=0),
                        log_z=np.stack(per_cam[out_name]["log_z"], axis=0),
                        conf=np.stack(per_cam[out_name]["conf"], axis=0),
                    )
                logger.info("wrote episode_%06d from %s/%s", episode_out_idx, path.name, ep)
                episode_out_idx += 1
            if env is not None:
                env.close()


if __name__ == "__main__":
    main()
