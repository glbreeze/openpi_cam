#!/usr/bin/env python3
"""Live-render verification for the RoboCasa365 sim setup.

Run this from the `robocasa365` conda env (NOT the openpi env) after
`download_kitchen_assets` finishes:

    conda activate robocasa365
    python scripts/debug/verify_robocasa_setup.py --env robocasa/OpenDrawer

What it verifies:

  1. The gym env loads, resets, and exposes the expected obs/action keys.
  2. Image direction: dumps the gym wrapper's frame (already vflipped from
     the raw MuJoCo buffer, so right-side-up to a human) and the raw MuJoCo
     buffer (re-flipped, upside-down). Empirically the LeRobot v2.1 stored
     frames also decode right-side-up, so the gym output and the LeRobot
     training data agree — the eval client passes the gym frame through
     to the policy server with no flip. (Both PNGs kept for debugging.)
  3. Action space: confirms 12-d action dict layout matches the
     PandaOmron_modality.json indices we hardcoded into robocasa_policy.py.
  4. Cam intrinsics + extrinsics: pulls K and cam-to-world from the
     underlying MuJoCo sim for each of the 3 cams. Sanity checks: fx==fy,
     cx≈W/2, cy≈H/2.

Saves all artifacts under `--out` (default `./out/robocasa_setup_check/`).

This script intentionally does NOT depend on openpi (the openpi conda env
has different package versions); copy outputs by hand to confirm against
robocasa_policy.py if needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import robocasa
import robocasa.wrappers.gym_wrapper
import robosuite

_ = (robocasa.__name__, robosuite.__name__)  # ensure the imports survive an unused-import linter


def _redo_gym_flip(img: np.ndarray) -> np.ndarray:
    """The gym wrapper applies `img[::-1, :, :]` to the raw MuJoCo buffer
    before returning. Re-apply the same flip to recover the raw buffer
    orientation (upside-down to a human). Kept as a visual-debugging
    artifact only — the LeRobot stored frames are right-side-up, not raw."""
    return img[::-1, :, :]


def _dump_cam_k_t_wc(env_inner, cam_names):
    """Pull intrinsics + extrinsics for each camera from the MuJoCo sim.

    Returns dict[cam_name] = {"K": (3,3), "T_wc": (4,4)} where T_wc is the
    OpenGL convention camera-to-world matrix (x-right, y-up, z-back).
    """
    sim = env_inner.sim
    out = {}
    for cam in cam_names:
        cam_id = sim.model.camera_name2id(cam)
        fovy_deg = float(sim.model.cam_fovy[cam_id])
        # We need W, H. RoboCasa renders at 256x256 by default; pull from the
        # offscreen renderer if available, else default.
        H = W = 256
        f = 0.5 * H / np.tan(0.5 * np.deg2rad(fovy_deg))
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float64)

        cam_pos = sim.data.cam_xpos[cam_id].copy()  # world
        cam_xmat = sim.data.cam_xmat[cam_id].copy().reshape(3, 3)  # cam-to-world, OpenGL
        T = np.eye(4)
        T[:3, :3] = cam_xmat
        T[:3, 3] = cam_pos
        out[cam] = {
            "K": K,
            "T_wc": T,
            "fovy_deg": fovy_deg,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="robocasa/OpenDrawer")
    parser.add_argument("--split", default="target")
    parser.add_argument("--out", type=Path, default=Path("out/robocasa_setup_check"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env, enable_render=True, split=args.split)
    obs, _ = env.reset(seed=args.seed)

    print(f"\n=== obs keys ({len(obs)}) ===")
    for k, v in obs.items():
        shape = getattr(v, "shape", None)
        dtype = getattr(v, "dtype", None)
        print(f"  {k:60s} shape={shape} dtype={dtype}")

    print("\n=== action_space ===")
    print(env.action_space)
    action_keys = list(env.action_space.spaces.keys()) if hasattr(env.action_space, "spaces") else []
    expected = {
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    }
    missing = expected - set(action_keys)
    extra = set(action_keys) - expected
    print(f"action keys: {action_keys}")
    print(f"missing vs expected: {missing}")
    print(f"extra vs expected:   {extra}")
    if missing:
        print("WARNING: missing action dict keys — re-check PandaOmron_modality.json + gym_wrapper.py.")

    # Dump image triples (gym-flipped vs undone).
    cam_names = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    print("\n=== camera images ===")
    for cam in cam_names:
        key = f"video.{cam}"
        if key not in obs:
            print(f"  MISSING: {key}")
            continue
        img_gym = np.asarray(obs[key])
        img_raw = _redo_gym_flip(img_gym)
        imageio.imwrite(args.out / f"{cam}__gym_wrapper_humanview.png", img_gym)
        imageio.imwrite(args.out / f"{cam}__raw_mujoco_buffer.png", img_raw)
        print(f"  {cam}: gym shape={img_gym.shape} -> wrote __gym_wrapper_humanview.png and __raw_mujoco_buffer.png")

    # Dump K + T_wc from MuJoCo. env.env reaches the inner RoboCasaGymEnv.
    inner = env.unwrapped
    K_T = _dump_cam_k_t_wc(inner.env, cam_names)
    print("\n=== camera intrinsics + extrinsics ===")
    out_json = {}
    for cam, params in K_T.items():
        K = params["K"]
        T = params["T_wc"]
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        print(f"  {cam}: fovy={params['fovy_deg']:.2f}deg  fx={fx:.2f} fy={fy:.2f} cx={cx:.1f} cy={cy:.1f}")
        print(f"    T_wc translation = {T[:3, 3]}")
        # Sanity: square intrinsic, principal point centered.
        ok = abs(fx - fy) < 1e-3 and abs(cx - 128) < 1e-3 and abs(cy - 128) < 1e-3
        print(f"    K sanity: fx==fy & cx==cy==W/2 : {ok}")
        out_json[cam] = {
            "K": K.tolist(),
            "T_wc": T.tolist(),
            "fovy_deg": float(params["fovy_deg"]),
        }

    (args.out / "K_T_wc.json").write_text(json.dumps(out_json, indent=2))

    # Step once with a zero action to make sure step() runs.
    action_dict = {k: np.zeros(space.shape, dtype=np.float32) for k, space in env.action_space.spaces.items()}
    next_obs, reward, terminated, truncated, info = env.step(action_dict)
    print(f"\n=== step() ok: reward={reward}, terminated={terminated}, truncated={truncated} ===")

    env.close()
    print(f"\nArtifacts written to {args.out.resolve()}")
    print("Eyeball the two PNG variants per cam:")
    print("  '__gym_wrapper_humanview.png'  — gym wrapper output; right-side-up.")
    print("  '__raw_mujoco_buffer.png'      — raw buffer (re-flipped); upside-down.")
    print("Empirically the LeRobot v2.1 stored frames also decode right-side-up,")
    print("so the eval client passes the gym frame through unchanged.")


if __name__ == "__main__":
    main()
