"""Check RoboCasa v0 demo replay and eval-environment state conventions."""

from __future__ import annotations

import argparse
import copy
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import h5py
import numpy as np
import robocasa  # noqa: F401
import robosuite
from PIL import Image

from robocasa.scripts.playback_dataset import get_env_metadata_from_dataset


# Source v0 action -> LeRobot action permutation used by cache_robocasa24_cam_matrices.py.
ACTION_PERM = np.array([7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6], dtype=np.int64)


def _patch_robocasa_temp_mjcf_xml() -> None:
    """Write generated RoboCasa object XML files to scratch instead of asset dirs."""
    import time
    import xml.etree.ElementTree as ET

    import robocasa.environments.kitchen.kitchen as kitchen_mod
    import robocasa.models.objects.objects as objects_mod
    from robosuite.models.objects import MujocoXMLObject

    if getattr(kitchen_mod.MJCFObject, "_openpi_tmp_xml_patch", False):
        return

    original_cls = objects_mod.MJCFObject

    class ScratchMJCFObject(original_cls):
        _openpi_tmp_xml_patch = True

        def __init__(
            self,
            name,
            mjcf_path,
            scale=1.0,
            solimp=(0.998, 0.998, 0.001),
            solref=(0.001, 1),
            density=100,
            friction=(0.95, 0.3, 0.1),
            margin=None,
            rgba=None,
            priority=None,
        ):
            if isinstance(scale, float):
                scale = [scale, scale, scale]
            elif isinstance(scale, tuple) or isinstance(scale, list):
                assert len(scale) == 3
                scale = tuple(scale)
            else:
                raise TypeError(f"got invalid scale: {scale}")
            scale = np.array(scale)

            self.solimp = solimp
            self.solref = solref
            self.density = density
            self.friction = friction
            self.margin = margin
            self.priority = priority
            self.rgba = rgba

            source_xml_dir = os.path.dirname(mjcf_path)
            root = ET.parse(mjcf_path).getroot()
            xml_str = ET.tostring(root, encoding="utf8").decode("utf8")
            xml_str = self.postprocess_model_xml(xml_str)
            xml_root = ET.fromstring(xml_str)
            asset = xml_root.find("asset")
            if asset is not None:
                for elem in asset.findall("mesh") + asset.findall("texture"):
                    asset_file = elem.get("file")
                    if asset_file is not None and not os.path.isabs(asset_file):
                        elem.set("file", os.path.normpath(os.path.join(source_xml_dir, asset_file)))
            xml_str = ET.tostring(xml_root, encoding="utf8").decode("utf8")

            tmp_root = Path(
                os.environ.get(
                    "ROBOCASA_TMP_XML_DIR",
                    os.path.join(os.environ.get("TMPDIR", "/tmp"), "robocasa_mjcf_xml"),
                )
            )
            tmp_root.mkdir(parents=True, exist_ok=True)
            new_xml_path = tmp_root / f"{time.time()}_{os.getpid()}.xml"
            new_xml_path.write_text(xml_str)
            try:
                MujocoXMLObject.__init__(
                    self,
                    fname=str(new_xml_path),
                    name=name,
                    joints=[dict(type="free", damping="0.0005")],
                    obj_type="all",
                    duplicate_collision_geoms=False,
                    scale=scale,
                )
            finally:
                new_xml_path.unlink(missing_ok=True)

    kitchen_mod.MJCFObject = ScratchMJCFObject


def _obs_to_state(obs: dict[str, Any]) -> np.ndarray:
    keys = (
        "robot0_base_pos",
        "robot0_base_quat",
        "robot0_base_to_eef_pos",
        "robot0_base_to_eef_quat",
        "robot0_gripper_qpos",
    )
    return np.concatenate([np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in keys], axis=0)


def _make_env_from_dataset(dataset_path: Path, *, use_camera_obs: bool = False, image_size: int = 224):
    env_meta = get_env_metadata_from_dataset(str(dataset_path))
    env_kwargs = copy.deepcopy(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    # The saved demo XML already contains the exact materials needed for
    # reset_to(...). Avoid requiring the external generative-texture pack just
    # to construct the temporary shell env.
    env_kwargs["generative_textures"] = None
    if use_camera_obs:
        env_kwargs["camera_names"] = [
            "robot0_agentview_left",
            "robot0_eye_in_hand",
        ]
        env_kwargs["camera_widths"] = image_size
        env_kwargs["camera_heights"] = image_size
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = bool(use_camera_obs)
    env_kwargs["use_camera_obs"] = bool(use_camera_obs)
    env_kwargs["ignore_done"] = True
    return robosuite.make(**env_kwargs), env_meta


def _make_current_eval_env(task_name: str, seed: int):
    from robocasa.utils.env_utils import create_env

    return create_env(
        env_name=task_name,
        robots="PandaOmron",
        camera_names=["robot0_agentview_left", "robot0_eye_in_hand"],
        camera_widths=224,
        camera_heights=224,
        seed=seed,
        render_onscreen=False,
        randomize_cameras=False,
    )


def _make_official_eval_env(task_name: str, seed: int):
    from robocasa.utils.eval_utils import create_eval_env

    return create_eval_env(
        env_name=task_name,
        seed=seed,
        camera_names=["robot0_agentview_left", "robot0_eye_in_hand"],
        camera_widths=224,
        camera_heights=224,
        randomize_cameras=False,
    )


def _check_success(env, info: dict[str, Any]) -> bool:
    for key in ("success", "is_success", "task_success", "eval_success"):
        if key in info:
            return bool(info[key])
    for method_name in ("_check_success", "check_success", "_check_successes"):
        method = getattr(env, method_name, None)
        if method is None:
            continue
        try:
            value = method()
        except TypeError:
            continue
        if isinstance(value, dict):
            return bool(all(value.values()))
        return bool(value)
    return False


def _door_state(env) -> dict[str, float] | None:
    drawer = getattr(env, "drawer", None)
    if drawer is None or not hasattr(drawer, "get_door_state"):
        return None
    try:
        return {str(k): float(v) for k, v in drawer.get_door_state(env=env).items()}
    except Exception:
        return None


def _lerobot_to_v0_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    return np.concatenate(
        (
            action[5:8],
            action[8:11],
            action[11:12],
            action[0:4],
            action[4:5],
        ),
        axis=0,
    ).astype(np.float32)


def _resize_hwc(image: np.ndarray, image_size: int) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8)
    if arr.shape[0] == image_size and arr.shape[1] == image_size:
        return arr
    return np.asarray(Image.fromarray(arr).resize((image_size, image_size), Image.BILINEAR), dtype=np.uint8)


def _image_orientation_mse(source: np.ndarray, rendered: np.ndarray) -> dict[str, float]:
    src = _resize_hwc(source, int(np.asarray(rendered).shape[0])).astype(np.float32)
    obs = np.asarray(rendered, dtype=np.float32)
    modes = {
        "as_is": obs,
        "vflip": obs[::-1, :, :],
        "hflip": obs[:, ::-1, :],
        "rot180": obs[::-1, ::-1, :],
        "bgr": obs[..., ::-1],
        "vflip_bgr": obs[::-1, :, ::-1],
        "hflip_bgr": obs[:, ::-1, ::-1],
        "rot180_bgr": obs[::-1, ::-1, ::-1],
    }
    return {name: float(np.mean((src - candidate) ** 2)) for name, candidate in modes.items()}


def _save_image_check_dump(
    dump_dir: Path | None,
    demo_name: str,
    mode: str,
    camera_name: str,
    source: np.ndarray,
    rendered: np.ndarray,
) -> dict[str, str]:
    if dump_dir is None:
        return {}

    dump_dir.mkdir(parents=True, exist_ok=True)
    prefix = dump_dir / f"{demo_name}_{mode}_{camera_name}"
    src = _resize_hwc(source, int(np.asarray(rendered).shape[0]))
    render = np.asarray(rendered, dtype=np.uint8)
    policy = np.asarray(rendered, dtype=np.uint8)[::-1, :, :]
    diff = np.clip(np.abs(src.astype(np.int16) - policy.astype(np.int16)) * 2, 0, 255).astype(np.uint8)
    canvas = np.concatenate([src, render, policy, diff], axis=1)
    paths = {
        "source": f"{prefix}_source.png",
        "render_raw": f"{prefix}_render_raw.png",
        "render_policy_flip": f"{prefix}_render_policy_flip.png",
        "policy_diff_x2": f"{prefix}_policy_diff_x2.png",
        "panel": f"{prefix}_panel.png",
    }
    Image.fromarray(src).save(paths["source"])
    Image.fromarray(render).save(paths["render_raw"])
    Image.fromarray(policy).save(paths["render_policy_flip"])
    Image.fromarray(diff).save(paths["policy_diff_x2"])
    Image.fromarray(canvas).save(paths["panel"])
    return paths


def _camera_id(model, name: str) -> int:
    if hasattr(model, "camera_name2id"):
        return int(model.camera_name2id(name))
    import mujoco

    raw_model = getattr(model, "_model", model)
    return int(mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_CAMERA, name))


def _camera_pose(sim, camera_name: str) -> np.ndarray:
    cam_id = _camera_id(sim.model, camera_name)
    T_wc = np.eye(4, dtype=np.float32)
    T_wc[:3, :3] = np.asarray(sim.data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)
    T_wc[:3, 3] = np.asarray(sim.data.cam_xpos[cam_id], dtype=np.float32)
    return T_wc


def _camera_fovy(sim, camera_name: str) -> float:
    cam_id = _camera_id(sim.model, camera_name)
    model = sim.model
    if hasattr(model, "cam_fovy"):
        return float(model.cam_fovy[cam_id])
    return float(model._model.cam_fovy[cam_id])


def _intrinsic_from_fovy(fovy_deg: float, height: int, width: int) -> np.ndarray:
    focal = (float(height) / 2.0) / np.tan(np.deg2rad(float(fovy_deg)) / 2.0)
    return np.asarray(
        [[focal, 0.0, float(width) / 2.0], [0.0, focal, float(height) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _initial_camera_matrices(env, initial_obs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for camera_name in ("robot0_agentview_left", "robot0_eye_in_hand"):
        obs_key = f"{camera_name}_image"
        if obs_key not in initial_obs:
            continue
        image = np.asarray(initial_obs[obs_key])
        height, width = int(image.shape[0]), int(image.shape[1])
        try:
            out[camera_name] = {
                "extrinsic": _camera_pose(env.sim, camera_name).tolist(),
                "intrinsic": _intrinsic_from_fovy(_camera_fovy(env.sim, camera_name), height, width).tolist(),
            }
        except Exception as exc:
            out[camera_name] = {"error": repr(exc)}
    return out


def _xml_array(value: Any) -> str:
    return " ".join(f"{float(x):.17g}" for x in np.asarray(value).reshape(-1))


def _find_xml_camera(root: ET.Element, name: str) -> ET.Element | None:
    for camera in root.iter("camera"):
        if camera.get("name") == name:
            return camera
    return None


def _find_xml_body(root: ET.Element, name: str | None) -> ET.Element | None:
    if not name:
        return None
    for candidate in (name, name.replace("base0_", "mobilebase0_"), name.replace("mobilebase0_", "base0_")):
        for body in root.iter("body"):
            if body.get("name") == candidate:
                return body
    return None


def _remove_xml_cameras(root: ET.Element, name: str) -> None:
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "camera" and child.get("name") == name:
                parent.remove(child)


def _apply_ep_meta_camera_configs_to_xml(xml: str, ep_meta: dict[str, Any]) -> str:
    cam_configs = ep_meta.get("cam_configs") or {}
    if not cam_configs:
        return xml

    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    if worldbody is None:
        return xml

    for cam_name, cam_config in cam_configs.items():
        old_camera = _find_xml_camera(root, cam_name)
        old_attribs = dict(old_camera.attrib) if old_camera is not None else {}
        _remove_xml_cameras(root, cam_name)
        parent = _find_xml_body(worldbody, cam_config.get("parent_body")) or worldbody
        camera = ET.Element("camera")
        for key, value in old_attribs.items():
            if key not in {"name", "pos", "quat"}:
                camera.set(key, value)
        if camera.get("mode") is None:
            camera.set("mode", "fixed")
        camera.set("name", cam_name)
        parent.append(camera)

        camera.set("pos", _xml_array(cam_config["pos"]))
        camera.set("quat", _xml_array(cam_config["quat"]))
        for key, value in (cam_config.get("camera_attribs") or {}).items():
            camera.set(key, str(value))

    return ET.tostring(root, encoding="unicode")


def _reset_to_with_ep_meta_cameras(env, demo: h5py.Group, states: np.ndarray) -> None:
    ep_meta = json.loads(demo.attrs["ep_meta"])
    if hasattr(env, "set_attrs_from_ep_meta"):
        env.set_attrs_from_ep_meta(ep_meta)
    elif hasattr(env, "set_ep_meta"):
        env.set_ep_meta(ep_meta)

    env.reset()
    robosuite_version_id = int(robosuite.__version__.split(".")[1])
    if robosuite_version_id <= 3:
        from robosuite.utils.mjcf_utils import postprocess_model_xml

        xml = postprocess_model_xml(demo.attrs["model_file"])
    else:
        xml = env.edit_model_xml(demo.attrs["model_file"])
    xml = _apply_ep_meta_camera_configs_to_xml(xml, ep_meta)

    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()


def _initial_image_checks(
    demo: h5py.Group,
    initial_obs: dict[str, Any],
    *,
    demo_name: str,
    mode: str,
    image_dump_dir: Path | None,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name in ("robot0_agentview_left", "robot0_eye_in_hand"):
        obs_key = f"{name}_image"
        src_key = f"obs/{obs_key}"
        if src_key not in demo or obs_key not in initial_obs:
            continue
        mse = _image_orientation_mse(demo[src_key][0], initial_obs[obs_key])
        policy_mse = _image_orientation_mse(demo[src_key][0], np.asarray(initial_obs[obs_key])[::-1, :, :])
        checks[name] = {
            "render_shape": list(np.asarray(initial_obs[obs_key]).shape),
            "source_shape": list(np.asarray(demo[src_key][0]).shape),
            "mse": mse,
            "best": min(mse, key=mse.get),
            "policy_flipped_mse": policy_mse,
            "policy_flipped_best": min(policy_mse, key=policy_mse.get),
            "dump_paths": _save_image_check_dump(
                image_dump_dir,
                demo_name,
                mode,
                name,
                demo[src_key][0],
                initial_obs[obs_key],
            ),
        }
    return checks


def _replay_one(
    env,
    demo: h5py.Group,
    *,
    demo_name: str,
    mode: str,
    max_steps: int,
    image_dump_dir: Path | None,
) -> dict[str, Any]:
    states = np.asarray(demo["states"][:], dtype=np.float64)
    raw_actions = np.asarray(demo["actions"][:], dtype=np.float32)
    if mode == "raw":
        actions = raw_actions
    elif mode == "lerobot_roundtrip":
        lerobot_actions = raw_actions[:, ACTION_PERM]
        actions = np.stack([_lerobot_to_v0_action(a) for a in lerobot_actions], axis=0)
    else:
        raise ValueError(mode)

    _reset_to_with_ep_meta_cameras(env, demo, states)
    initial_obs = env._get_observations() if hasattr(env, "_get_observations") else {}
    initial_state16 = _obs_to_state(initial_obs) if initial_obs else None
    image_checks = (
        _initial_image_checks(
            demo,
            initial_obs,
            demo_name=demo_name,
            mode=mode,
            image_dump_dir=image_dump_dir,
        )
        if initial_obs
        else {}
    )
    camera_matrices = _initial_camera_matrices(env, initial_obs) if initial_obs else {}

    max_state_err = 0.0
    final_state_err = 0.0
    success = False
    first_success_step = None
    last_info: dict[str, Any] = {}
    steps = min(int(max_steps), int(actions.shape[0]))
    for i in range(steps):
        _, _, _, info = env.step(actions[i])
        last_info = dict(info)
        success_now = _check_success(env, last_info)
        if success_now and first_success_step is None:
            first_success_step = i
        success = success or success_now
        if i < states.shape[0] - 1:
            sim_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
            err = float(np.linalg.norm(states[i + 1] - sim_state))
            max_state_err = max(max_state_err, err)
            final_state_err = err

    return {
        "mode": mode,
        "steps": steps,
        "success": bool(success),
        "first_success_step": first_success_step,
        "max_state_err": max_state_err,
        "final_state_err": final_state_err,
        "initial_state16": None if initial_state16 is None else initial_state16.tolist(),
        "initial_image_checks": image_checks,
        "initial_camera_matrices": camera_matrices,
        "final_door_state": _door_state(env),
        "last_info_keys": sorted(last_info.keys()),
    }


def _env_snapshot(env) -> dict[str, Any]:
    obs = env.reset()
    return {
        "layout_id": int(getattr(env, "layout_id", -1)),
        "style_id": int(getattr(env, "style_id", -1)),
        "obj_instance_split": getattr(env, "obj_instance_split", None),
        "robot_names": [type(robot.robot_model).__name__ for robot in getattr(env, "robots", [])],
        "state16": _obs_to_state(obs).tolist(),
        "door_state": _door_state(env),
        "action_spec_low": np.asarray(env.action_spec[0], dtype=np.float32).tolist(),
        "action_spec_high": np.asarray(env.action_spec[1], dtype=np.float32).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/"
            "kitchen_drawer/OpenDrawer/2024-05-03/demo_gentex_im128_randcams.hdf5"
        ),
    )
    parser.add_argument("--demos", default="demo_1,demo_2")
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-dump-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    _patch_robocasa_temp_mjcf_xml()
    env, env_meta = _make_env_from_dataset(args.dataset, use_camera_obs=True, image_size=args.image_size)
    task_name = str(env_meta["env_name"])
    demo_names = [part.strip() for part in args.demos.replace(":", ",").split(",") if part.strip()]

    out: dict[str, Any] = {
        "dataset": str(args.dataset),
        "env_meta_env_name": env_meta["env_name"],
        "env_meta_robots": env_meta["env_kwargs"].get("robots"),
        "env_meta_controller_type": env_meta["env_kwargs"].get("controller_configs", {}).get("type"),
        "demos": {},
        "eval_env_snapshots": {},
    }

    with h5py.File(args.dataset, "r") as f:
        for demo_name in demo_names:
            demo = f[f"data/{demo_name}"]
            ep_meta = json.loads(demo.attrs["ep_meta"])
            demo_out = {
                "ep_meta": {
                    "lang": ep_meta.get("lang"),
                    "layout_id": ep_meta.get("layout_id"),
                    "style_id": ep_meta.get("style_id"),
                },
                "num_samples": int(demo.attrs["num_samples"]),
                "source_initial_state16": _obs_to_state({k: demo[f"obs/{k}"][0] for k in (
                    "robot0_base_pos",
                    "robot0_base_quat",
                    "robot0_base_to_eef_pos",
                    "robot0_base_to_eef_quat",
                    "robot0_gripper_qpos",
                )}).tolist(),
                "raw_action0": np.asarray(demo["actions"][0], dtype=np.float32).tolist(),
                "lerobot_action0": np.asarray(demo["actions"][0], dtype=np.float32)[ACTION_PERM].tolist(),
                "replay": [],
            }
            for mode in ("raw", "lerobot_roundtrip"):
                demo_out["replay"].append(
                    _replay_one(
                        env,
                        demo,
                        demo_name=demo_name,
                        mode=mode,
                        max_steps=args.max_steps,
                        image_dump_dir=args.image_dump_dir,
                    )
                )
            out["demos"][demo_name] = demo_out

    for name, maker in (
        ("current_eval_env_utils", _make_current_eval_env),
        ("official_eval_utils", _make_official_eval_env),
    ):
        try:
            snap_env = maker(task_name, args.seed)
            try:
                out["eval_env_snapshots"][name] = _env_snapshot(snap_env)
            finally:
                close_fn = getattr(snap_env, "close", None)
                if close_fn is not None:
                    close_fn()
        except Exception as exc:
            out["eval_env_snapshots"][name] = {"error": repr(exc)}

    close_fn = getattr(env, "close", None)
    if close_fn is not None:
        close_fn()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
