"""Replay LIBERO raw demos from a fixed manifest and store aligned depth HDF5.

This is a manifest-driven variant of LIBERO-Camera's `scripts/create_dataset.py`.
The manifest fixes the exact LeRobot episode order, so the regenerated depth
HDF5 can be consumed by `cache_libero_gt_point_targets.py` without any later
episode re-indexing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np


def _load_libero_camera(libero_camera_root: Path):
    sys.path.insert(0, str(libero_camera_root))
    sys.path.insert(0, str(libero_camera_root / "scripts"))
    import robosuite.macros as macros
    import robosuite.utils.transform_utils as transform_utils
    import libero.libero.utils.utils as libero_utils
    from libero.libero.envs import TASK_MAPPING

    return macros, transform_utils, libero_utils, TASK_MAPPING


def _as_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _is_noop(action: np.ndarray, prev_action: np.ndarray | None = None, threshold: float = 1e-4) -> bool:
    action = np.asarray(action)
    if prev_action is None:
        return float(np.linalg.norm(action[:-1])) < threshold
    return (
        float(np.linalg.norm(action[:-1])) < threshold
        and float(action[-1]) == float(np.asarray(prev_action)[-1])
    )


def _filter_noops(actions: np.ndarray) -> tuple[np.ndarray, list[int]]:
    kept: list[np.ndarray] = []
    kept_indices: list[int] = []
    prev_kept_action = None
    for idx, action in enumerate(np.asarray(actions, dtype=np.float32)):
        if _is_noop(action, prev_kept_action):
            continue
        kept.append(action)
        kept_indices.append(idx)
        prev_kept_action = action
    if not kept:
        return np.zeros((0, np.asarray(actions).shape[-1]), dtype=np.float32), kept_indices
    return np.stack(kept, axis=0).astype(np.float32), kept_indices


def _fingerprint_actions(actions: np.ndarray, decimals: int) -> tuple[int, str]:
    actions = np.asarray(actions, dtype=np.float32)
    rounded = np.round(actions, decimals=decimals).astype(np.float32, copy=False)
    return int(actions.shape[0]), hashlib.sha1(rounded.tobytes()).hexdigest()


def _postprocess_model_xml_for_dataset(model_xml, cameras_dict, libero_camera_root: Path, libero_utils):
    xml_str = libero_utils.postprocess_model_xml(_as_text(model_xml), cameras_dict)
    root = ET.fromstring(xml_str)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("autolimits", "true")

    local_assets_root = libero_camera_root / "libero" / "libero" / "assets"
    for tag in ("mesh", "texture"):
        for elem in root.iter(tag):
            if tag == "texture":
                elem.attrib.pop("colorspace", None)
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


def _get_camera_extrinsic(sim, camera_name: str) -> np.ndarray:
    cam_id = sim.model.camera_name2id(camera_name)
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = np.asarray(sim.data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)
    extrinsic[:3, 3] = np.asarray(sim.data.cam_xpos[cam_id], dtype=np.float32)
    return extrinsic


def _get_camera_intrinsic(sim, camera_name: str, image_h: int, image_w: int) -> np.ndarray:
    cam_id = sim.model.camera_name2id(camera_name)
    fovy_rad = np.deg2rad(float(sim.model.cam_fovy[cam_id]))
    fy = (image_h / 2.0) / np.tan(fovy_rad / 2.0)
    return np.array(
        [[fy, 0.0, image_w / 2.0], [0.0, fy, image_h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    for idx, row in enumerate(rows):
        if int(row["episode_index"]) != idx:
            raise ValueError(f"Manifest is not contiguous at row {idx}: {row}")
    return rows


def _resolve_bddl_path(bddl_file_name: str, libero_camera_root: Path) -> str:
    path = Path(bddl_file_name)
    if path.is_absolute():
        return str(path)
    candidate = libero_camera_root / path
    if candidate.exists():
        return str(candidate)
    normalized = bddl_file_name.replace("\\", "/")
    marker = "bddl_files/"
    if marker in normalized:
        candidate = libero_camera_root / "libero" / "libero" / "bddl_files" / normalized.split(marker, 1)[1]
        if candidate.exists():
            return str(candidate)
    return bddl_file_name


def _load_env_args(
    h5_file: h5py.File,
    render_resolution: int,
    libero_camera_root: Path,
    libero_utils,
) -> tuple[str, dict, dict]:
    env_name = h5_file["data"].attrs.get("env", h5_file["data"].attrs.get("env_name"))
    env_info_raw = h5_file["data"].attrs.get("env_info")
    if env_info_raw not in (None, ""):
        env_kwargs = json.loads(_as_text(env_info_raw))
    else:
        env_args_raw = h5_file["data"].attrs.get("env_args")
        if env_args_raw in (None, ""):
            raise ValueError("HDF5 missing both env_info and env_args; cannot build env")
        env_args_json = json.loads(_as_text(env_args_raw))
        env_kwargs = env_args_json["env_kwargs"]
        if env_name is None:
            env_name = env_args_json.get("env_name")

    problem_info = json.loads(_as_text(h5_file["data"].attrs["problem_info"]))
    problem_name = problem_info["problem_name"]
    bddl_file_name = _resolve_bddl_path(_as_text(h5_file["data"].attrs["bddl_file_name"]), libero_camera_root)
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
    env_args = {
        "type": 1,
        "env_name": env_name,
        "problem_name": problem_name,
        "bddl_file": bddl_file_name,
        "env_kwargs": env_kwargs,
    }
    return problem_name, problem_info, env_args


def _copy_data_attrs(src_data: h5py.Group, dst_data: h5py.Group, *, env_args: dict, macros):
    for key, value in src_data.attrs.items():
        dst_data.attrs[key] = value
    dst_data.attrs["env_args"] = json.dumps(env_args)
    dst_data.attrs["env_name"] = env_args["env_name"]
    dst_data.attrs["macros_image_convention"] = macros.IMAGE_CONVENTION
    dst_data.attrs["manifest_aligned"] = True


def _write_episode_file(
    output_path: Path,
    *,
    source_h5: h5py.File,
    source_episode: str,
    target_episode_index: int,
    kept_indices: list[int],
    env,
    env_args: dict,
    macros,
    transform_utils,
    libero_utils,
    libero_camera_root: Path,
    settle_steps: int,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    source_ep = source_h5[f"data/{source_episode}"]
    model_xml = source_ep.attrs["model_file"]
    states_all = source_ep["states"][()]
    actions_all = np.asarray(source_ep["actions"], dtype=np.float32)

    dummy_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    model_xml = _postprocess_model_xml_for_dataset(model_xml, {}, libero_camera_root, libero_utils)

    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except Exception:
            continue
    env.reset_from_xml_string(model_xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(states_all[0])
    env.sim.forward()
    model_xml = env.sim.model.get_xml()

    env._post_process()
    env._update_observables(force=True)
    obs = env._get_observations()
    for _ in range(max(0, int(settle_steps))):
        obs, _, _, _ = env.step(dummy_action)

    kept_set = set(int(i) for i in kept_indices)
    ee_states = []
    gripper_states = []
    joint_states = []
    robot_states = []
    agentview_images = []
    eye_in_hand_images = []
    agentview_depths = []
    eye_in_hand_depths = []
    agentview_extrinsics = []
    eye_in_hand_extrinsics = []
    last_done = False

    for idx, action in enumerate(actions_all):
        if idx not in kept_set:
            continue

        gripper_states.append(obs["robot0_gripper_qpos"])
        joint_states.append(obs["robot0_joint_pos"])
        ee_states.append(
            np.hstack((obs["robot0_eef_pos"], transform_utils.quat2axisangle(obs["robot0_eef_quat"])))
        )
        robot_states.append(env.get_robot_state_vector(obs))
        agentview_images.append(obs["agentview_image"])
        eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])
        agentview_depths.append(obs["agentview_depth"])
        eye_in_hand_depths.append(obs["robot0_eye_in_hand_depth"])
        agentview_extrinsics.append(_get_camera_extrinsic(env.sim, "agentview"))
        eye_in_hand_extrinsics.append(_get_camera_extrinsic(env.sim, "robot0_eye_in_hand"))

        obs, _, done, _ = env.step(action)
        last_done = bool(done)

    if len(agentview_images) != len(kept_indices):
        raise RuntimeError(
            f"Replay kept {len(agentview_images)} frames but manifest expected {len(kept_indices)} "
            f"for target episode {target_episode_index}"
        )
    if not agentview_images:
        raise RuntimeError(f"Target episode {target_episode_index} produced no frames")

    kept_actions = actions_all[kept_indices]
    kept_states = states_all[kept_indices]
    dones = np.zeros(len(kept_actions), dtype=np.uint8)
    dones[-1] = 1
    rewards = np.zeros(len(kept_actions), dtype=np.uint8)
    rewards[-1] = 1

    with h5py.File(output_path, "w") as out_h5:
        data_group = out_h5.create_group("data")
        _copy_data_attrs(source_h5["data"], data_group, env_args=env_args, macros=macros)
        data_group.attrs["manifest_target_episode"] = int(target_episode_index)
        data_group.attrs["manifest_source_episode"] = source_episode

        ep_group = data_group.create_group("demo_0")
        obs_group = ep_group.create_group("obs")
        obs_group.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
        obs_group.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
        obs_group.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
        obs_group.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
        obs_group.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
        obs_group.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
        obs_group.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
        obs_group.create_dataset("agentview_depth", data=np.stack(agentview_depths, axis=0))
        obs_group.create_dataset("eye_in_hand_depth", data=np.stack(eye_in_hand_depths, axis=0))
        obs_group.create_dataset("agent_extrinsic", data=np.stack(agentview_extrinsics, axis=0))
        obs_group.create_dataset("wrist_extrinsic", data=np.stack(eye_in_hand_extrinsics, axis=0))

        agent_h, agent_w = agentview_images[0].shape[:2]
        wrist_h, wrist_w = eye_in_hand_images[0].shape[:2]
        obs_group.attrs["agent_intrinsic"] = _get_camera_intrinsic(env.sim, "agentview", agent_h, agent_w)
        obs_group.attrs["wrist_intrinsic"] = _get_camera_intrinsic(
            env.sim, "robot0_eye_in_hand", wrist_h, wrist_w
        )
        obs_group.attrs["agent_image_size"] = np.array([agent_h, agent_w], dtype=np.int32)
        obs_group.attrs["wrist_image_size"] = np.array([wrist_h, wrist_w], dtype=np.int32)

        ep_group.create_dataset("actions", data=kept_actions)
        ep_group.create_dataset("states", data=kept_states)
        ep_group.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
        ep_group.create_dataset("rewards", data=rewards)
        ep_group.create_dataset("dones", data=dones)
        ep_group.attrs["num_samples"] = len(agentview_images)
        ep_group.attrs["model_file"] = model_xml
        ep_group.attrs["init_state"] = states_all[0]
        ep_group.attrs["source_episode"] = source_episode
        ep_group.attrs["replay_success"] = bool(last_done)

        data_group.attrs["num_demos"] = 1
        data_group.attrs["total"] = len(agentview_images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--libero-camera-root", required=True, type=Path)
    parser.add_argument("--render-resolution", type=int, default=256)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--action-decimals", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    libero_camera_root = args.libero_camera_root.expanduser().resolve()
    macros, transform_utils, libero_utils, TASK_MAPPING = _load_libero_camera(libero_camera_root)
    manifest_rows = _load_manifest(args.manifest.expanduser())
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    rows_by_file: dict[str, list[dict]] = {}
    for row in manifest_rows:
        rows_by_file.setdefault(row["raw_file"], []).append(row)

    wrote = 0
    skipped = 0
    for raw_file, rows in sorted(rows_by_file.items()):
        raw_path = raw_root / raw_file
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw HDF5: {raw_path}")
        with h5py.File(raw_path, "r") as source_h5:
            problem_name, _, env_args = _load_env_args(
                source_h5, args.render_resolution, libero_camera_root, libero_utils
            )
            env = TASK_MAPPING[problem_name](**env_args["env_kwargs"])
            try:
                for row in sorted(rows, key=lambda item: int(item["episode_index"])):
                    target_idx = int(row["episode_index"])
                    output_path = output_root / "libero_object" / f"episode_{target_idx:06d}.hdf5"
                    if args.resume and output_path.exists():
                        skipped += 1
                        continue

                    source_episode = row["raw_episode"]
                    actions = np.asarray(source_h5[f"data/{source_episode}/actions"], dtype=np.float32)
                    filtered_actions, kept_indices = _filter_noops(actions)
                    length, digest = _fingerprint_actions(filtered_actions, args.action_decimals)
                    if length != int(row["length"]) or digest != row["action_sha1"]:
                        raise ValueError(
                            f"Manifest action mismatch for target episode {target_idx}: "
                            f"got length={length} sha1={digest}, expected length={row['length']} sha1={row['action_sha1']}"
                        )
                    _write_episode_file(
                        output_path,
                        source_h5=source_h5,
                        source_episode=source_episode,
                        target_episode_index=target_idx,
                        kept_indices=kept_indices,
                        env=env,
                        env_args=env_args,
                        macros=macros,
                        transform_utils=transform_utils,
                        libero_utils=libero_utils,
                        libero_camera_root=libero_camera_root,
                        settle_steps=args.settle_steps,
                    )
                    wrote += 1
                    print(
                        f"[aligned-depth] episode_{target_idx:06d} {raw_file}/{source_episode} "
                        f"T={length} -> {output_path}",
                        flush=True,
                    )
            finally:
                env.close()

    print(f"aligned_depth_complete wrote={wrote} skipped={skipped} total={len(manifest_rows)}")


if __name__ == "__main__":
    main()
