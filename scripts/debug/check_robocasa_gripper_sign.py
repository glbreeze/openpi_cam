#!/usr/bin/env python3
"""Minimal RoboCasa gripper control sanity check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import robocasa  # noqa: F401
import robosuite
from robosuite.controllers import load_composite_controller_config

from scripts.debug.eval_robocasa365_remote import _patch_robocasa_temp_mjcf_xml


def _make_env(env_name: str, seed: int):
    _patch_robocasa_temp_mjcf_xml()
    controller_config = load_composite_controller_config(controller=None, robot="PandaOmron")
    return robosuite.make(
        env_name=env_name,
        robots="PandaOmron",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        ignore_done=True,
        camera_depths=False,
        seed=seed,
        randomize_cameras=False,
        translucent_robot=False,
    )


def _qpos(obs: dict) -> list[float]:
    return [float(x) for x in np.asarray(obs["robot0_gripper_qpos"]).reshape(-1)]


def _run_sequence(env, gripper_value: float, steps: int) -> dict:
    obs = env.reset()
    low, high = env.action_spec
    action = np.zeros_like(low, dtype=np.float32)
    action[-1] = -1.0
    action[6] = np.float32(gripper_value)
    qs = [_qpos(obs)]
    for _ in range(steps):
        clipped = np.clip(action, low, high).astype(np.float32)
        obs, reward, done, info = env.step(clipped)
        qs.append(_qpos(obs))
    return {
        "gripper_command": float(gripper_value),
        "steps": steps,
        "qpos_start": qs[0],
        "qpos_end": qs[-1],
        "qpos_trace": qs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="OpenDrawer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    env = _make_env(args.env, args.seed)
    try:
        result = {
            "env": args.env,
            "seed": args.seed,
            "plus_one": _run_sequence(env, 1.0, args.steps),
            "minus_one": _run_sequence(env, -1.0, args.steps),
        }
    finally:
        close_fn = getattr(env, "close", None)
        if close_fn is not None:
            close_fn()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
