import json
import math
import os
import pathlib
import xml.etree.ElementTree as ET

import numpy as np

from libero.libero.utils import utils as libero_utils

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LIBERO_ASSETS_ROOT = REPO_ROOT / "third_party" / "libero" / "libero" / "libero" / "assets"


def _as_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _float_list_to_str(values) -> str:
    return " ".join(f"{float(v):.10f}" for v in values)


def _get_env_observations(env):
    if hasattr(env, "_get_observations"):
        return env._get_observations()
    inner_env = getattr(env, "env", None)
    if inner_env is not None and hasattr(inner_env, "_get_observations"):
        return inner_env._get_observations()
    raise AttributeError(f"Environment of type {type(env).__name__} does not expose _get_observations")


def _quat_normalize_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        raise ValueError("Quaternion norm is too small")
    return quat / norm


def _rotmat_to_quat_wxyz(R):
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return _quat_normalize_wxyz(np.array([w, x, y, z], dtype=np.float64))


def _lookat_quat_wxyz(camera_pos, target_pos, up_ref):
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    up_ref = np.asarray(up_ref, dtype=np.float64)

    forward = target_pos - camera_pos
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-9:
        raise ValueError("camera_pos and target_pos are identical")
    forward = forward / forward_norm

    z_axis = -forward
    x_axis = np.cross(up_ref, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-9:
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = np.cross(fallback_up, z_axis)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-9:
            fallback_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            x_axis = np.cross(fallback_up, z_axis)
            x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-9:
            raise ValueError("Cannot construct look-at camera basis")
    x_axis = x_axis / x_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    return _rotmat_to_quat_wxyz(R)


def _parse_schedule(spec, count, name):
    if isinstance(spec, (int, float)):
        return np.full((count,), float(spec), dtype=np.float64)
    if isinstance(spec, list):
        if len(spec) != count:
            raise ValueError(f"{name} list length ({len(spec)}) must equal count ({count})")
        return np.asarray(spec, dtype=np.float64)
    if isinstance(spec, dict):
        schedule_type = spec.get("type", "linspace")
        if schedule_type != "linspace":
            raise ValueError(f"Unsupported {name} schedule type: {schedule_type}")
        start = float(spec["start"])
        stop = float(spec["stop"])
        endpoint = bool(spec.get("endpoint", True))
        return np.linspace(start, stop, count, endpoint=endpoint, dtype=np.float64)
    raise ValueError(f"Unsupported schedule format for {name}: {type(spec)}")


def load_camera_variation_config(path: str | None) -> dict | None:
    if not path:
        return None
    with open(path, "r") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("camera variation config must be a JSON object")
    return cfg


def get_effective_count(cli_count: int, cfg: dict | None) -> int:
    if cfg is None:
        return int(cli_count)
    cfg_count = cfg.get("count")
    if cfg_count is None and cfg.get("strategy") == "manual_poses":
        cfg_count = len(cfg.get("manual_poses", []))
    if cfg_count is None:
        return int(cli_count)
    cfg_count = int(cfg_count)
    if cli_count not in (0, cfg_count):
        raise ValueError(f"CLI camera variation count ({cli_count}) conflicts with config count ({cfg_count})")
    return cfg_count


def needs_target_pos(cfg: dict | None) -> bool:
    if not cfg:
        return False
    return cfg.get("strategy", "random_local") == "orbit_lookat"


def get_target_request(cfg: dict | None) -> dict:
    target_cfg = (cfg or {}).get("target", {})
    source = target_cfg.get("source", "eef_pos")
    offset = np.asarray(target_cfg.get("offset", [0.0, 0.0, 0.0]), dtype=np.float64)
    state_index = int(target_cfg.get("state_index", 0))
    position = target_cfg.get("position")
    if position is not None:
        position = np.asarray(position, dtype=np.float64)
    return {
        "source": source,
        "offset": offset,
        "state_index": state_index,
        "position": position,
    }


def extract_camera_pose_from_xml(xml_str, camera_name: str) -> dict:
    root = ET.fromstring(_as_text(xml_str))
    for camera in root.iter("camera"):
        if camera.get("name") != camera_name:
            continue
        pos = np.fromstring(camera.get("pos", ""), sep=" ", dtype=np.float64)
        quat = np.fromstring(camera.get("quat", ""), sep=" ", dtype=np.float64)
        if pos.shape[0] != 3 or quat.shape[0] != 4:
            raise ValueError(f"Invalid pose for camera '{camera_name}' in XML")
        return {"pos": pos, "quat": _quat_normalize_wxyz(quat)}
    raise ValueError(f"Camera '{camera_name}' not found in XML")


def postprocess_model_xml_for_eval(model_xml, cameras_dict):
    xml_str = libero_utils.postprocess_model_xml(_as_text(model_xml), cameras_dict)
    root = ET.fromstring(xml_str)
    local_assets_root = str(LIBERO_ASSETS_ROOT)

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
            candidate = os.path.join(local_assets_root, suffix)
            if os.path.exists(candidate):
                elem.set("file", candidate)

    return ET.tostring(root, encoding="utf8").decode("utf8")


def reset_env_with_camera_pose(env, model_xml, state, cameras_dict):
    xml_override = postprocess_model_xml_for_eval(model_xml, cameras_dict)
    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except Exception:
            continue
    env.reset_from_xml_string(xml_override)
    env.sim.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    return _get_env_observations(env)


def resolve_target_pos_for_eval(env, model_xml, state, target_request: dict):
    source = target_request["source"]
    offset = np.asarray(target_request["offset"], dtype=np.float64)

    if source == "fixed_world":
        if target_request["position"] is None:
            raise ValueError("camera variation config target.position is required for fixed_world")
        return np.asarray(target_request["position"], dtype=np.float64) + offset
    if source != "eef_pos":
        raise ValueError(f"Unsupported target source: {source}")

    obs = reset_env_with_camera_pose(env, model_xml, state, {})
    if "robot0_eef_pos" not in obs:
        raise ValueError("Cannot resolve eef target position from environment observations")
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float64) + offset


def generate_camera_variation_poses(base_pos, count: int, cfg: dict, target_pos):
    if cfg.get("strategy", "random_local") != "orbit_lookat":
        raise ValueError(f"Unsupported camera variation strategy for eval: {cfg.get('strategy')}")

    orbit_cfg = cfg.get("orbit", {})
    up_ref = np.asarray(orbit_cfg.get("up_ref", [0.0, 0.0, 1.0]), dtype=np.float64)
    relative = bool(orbit_cfg.get("angles_relative_to_base", True))

    yaw_vals_deg = _parse_schedule(
        orbit_cfg.get("yaw_deg", {"type": "linspace", "start": -35, "stop": 35}),
        count,
        "yaw_deg",
    )
    pitch_vals_deg = _parse_schedule(orbit_cfg.get("pitch_deg", -15.0), count, "pitch_deg")
    radius_scale_vals = _parse_schedule(orbit_cfg.get("radius_scale", 1.0), count, "radius_scale")
    radius_offset_vals = _parse_schedule(orbit_cfg.get("radius_offset", 0.0), count, "radius_offset")
    radius_offset_per_abs_yaw_deg = float(orbit_cfg.get("radius_offset_per_abs_yaw_deg", 0.0))
    radius_offset_per_abs_pitch_deg = float(orbit_cfg.get("radius_offset_per_abs_pitch_deg", 0.0))

    base_vec = np.asarray(base_pos, dtype=np.float64) - np.asarray(target_pos, dtype=np.float64)
    base_r = np.linalg.norm(base_vec)
    if base_r < 1e-9:
        raise ValueError("Base camera is too close to target for orbit_lookat")
    base_yaw = math.degrees(math.atan2(base_vec[1], base_vec[0]))
    base_pitch = math.degrees(math.atan2(base_vec[2], np.linalg.norm(base_vec[:2])))

    poses = []
    for variation_id in range(count):
        yaw_deg = float(yaw_vals_deg[variation_id])
        pitch_deg = float(pitch_vals_deg[variation_id])
        if relative:
            yaw_world_deg = base_yaw + yaw_deg
            pitch_world_deg = base_pitch + pitch_deg
        else:
            yaw_world_deg = yaw_deg
            pitch_world_deg = pitch_deg

        radius = float(base_r * radius_scale_vals[variation_id] + radius_offset_vals[variation_id])
        radius += radius_offset_per_abs_yaw_deg * abs(yaw_deg)
        radius += radius_offset_per_abs_pitch_deg * abs(pitch_deg)
        if radius <= 1e-6:
            raise ValueError(f"Invalid orbit radius for variation {variation_id}: {radius}")

        yaw_rad = math.radians(yaw_world_deg)
        pitch_rad = math.radians(pitch_world_deg)
        xy = radius * math.cos(pitch_rad)
        offset = np.array(
            [
                xy * math.cos(yaw_rad),
                xy * math.sin(yaw_rad),
                radius * math.sin(pitch_rad),
            ],
            dtype=np.float64,
        )
        pos = np.asarray(target_pos, dtype=np.float64) + offset
        quat = _lookat_quat_wxyz(pos, target_pos, up_ref)

        poses.append(
            {
                "variation_id": variation_id,
                "applied_pos": pos,
                "applied_quat": quat,
                "delta_pos": pos - np.asarray(base_pos, dtype=np.float64),
                "delta_rpy_deg": np.array([0.0, pitch_deg, yaw_deg], dtype=np.float64),
                "strategy": "orbit_lookat",
            }
        )
    return poses


def build_camera_variation_specs(
    *,
    env,
    model_xml,
    initial_state,
    config_path: str | None,
    count: int,
    seed: int,
    include_original: bool,
    target_camera: str = "agentview",
):
    cfg = load_camera_variation_config(config_path)
    effective_count = get_effective_count(count, cfg)
    base_pose = extract_camera_pose_from_xml(model_xml, target_camera)

    specs = []
    if include_original:
        specs.append(
            {
                "label": "original",
                "variation_id": None,
                "cameras_dict": {},
                "pos": base_pose["pos"].tolist(),
                "quat": base_pose["quat"].tolist(),
                "delta_pos": [0.0, 0.0, 0.0],
                "delta_rpy_deg": [0.0, 0.0, 0.0],
            }
        )

    if effective_count <= 0:
        return specs
    if cfg is None:
        raise ValueError("camera variation count > 0 requires a camera variation config")

    target_pos = None
    if needs_target_pos(cfg):
        target_request = get_target_request(cfg)
        target_pos = resolve_target_pos_for_eval(env, model_xml, initial_state, target_request)

    poses = generate_camera_variation_poses(
        base_pos=base_pose["pos"],
        count=effective_count,
        cfg=cfg,
        target_pos=target_pos,
    )
    for pose in poses:
        pos = np.asarray(pose["applied_pos"], dtype=np.float64)
        quat = np.asarray(pose["applied_quat"], dtype=np.float64)
        specs.append(
            {
                "label": f"camvar_{int(pose['variation_id']):02d}",
                "variation_id": int(pose["variation_id"]),
                "cameras_dict": {
                    target_camera: {
                        "pos": _float_list_to_str(pos),
                        "quat": _float_list_to_str(quat),
                    }
                },
                "pos": pos.tolist(),
                "quat": quat.tolist(),
                "delta_pos": np.asarray(pose["delta_pos"], dtype=np.float64).tolist(),
                "delta_rpy_deg": np.asarray(pose["delta_rpy_deg"], dtype=np.float64).tolist(),
            }
        )
    return specs
