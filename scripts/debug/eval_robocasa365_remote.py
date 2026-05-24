"""Run RoboCasa365 rollouts against an openpi websocket policy server."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import gymnasium as gym
import imageio
import numpy as np
import robocasa  # noqa: F401
import robocasa.wrappers.gym_wrapper as robocasa_gym_wrapper
import robosuite  # noqa: F401

from openpi_client import websocket_client_policy


CAM_KEYS = {
    "agentview_left": "video.robot0_agentview_left",
    "agentview_right": "video.robot0_agentview_right",
    "eye_in_hand": "video.robot0_eye_in_hand",
}

STATE_KEYS = (
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
)

GRIPPER_CLOSE_THRESHOLD = 0.0
BASE_MOTION_DEADZONE = 0.02


def _patch_eval_action_unmap() -> None:
    """Eval-only patch for RoboCasa PandaOmron action thresholding."""

    def unmap_action(cls, input_action):
        return {
            "robot0_right_gripper": (
                -1.0
                if input_action["action.gripper_close"] < GRIPPER_CLOSE_THRESHOLD
                else 1.0
            ),
            "robot0_right": np.concatenate(
                (
                    input_action["action.end_effector_position"],
                    input_action["action.end_effector_rotation"],
                ),
                axis=-1,
            ),
            "robot0_base": input_action["action.base_motion"][..., 0:3],
            "robot0_torso": input_action["action.base_motion"][..., 3:4],
            "robot0_base_mode": -1.0,
        }

    robocasa_gym_wrapper.PandaOmronKeyConverter.unmap_action = classmethod(unmap_action)


def _resize_chw(image: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {arr.shape}")
    if arr.shape[0] == size and arr.shape[1] == size:
        return np.ascontiguousarray(arr.transpose(2, 0, 1))

    from PIL import Image

    resized = Image.fromarray(arr.astype(np.uint8)).resize((size, size), Image.BILINEAR)
    return np.ascontiguousarray(np.asarray(resized).transpose(2, 0, 1))


def _prompt_for_obs(obs: dict[str, Any], prompt: str) -> str:
    if prompt and prompt.lower() != "auto":
        return prompt
    obs_prompt = obs.get("annotation.human.task_description")
    if obs_prompt is None:
        return "OpenDrawer"
    if isinstance(obs_prompt, np.ndarray):
        obs_prompt = obs_prompt.item() if obs_prompt.shape == () else obs_prompt.tolist()
    if isinstance(obs_prompt, (list, tuple)):
        obs_prompt = obs_prompt[0] if obs_prompt else ""
    return str(obs_prompt)


def _obs_to_policy_input(obs: dict[str, Any], prompt: str, image_size: int) -> dict[str, Any]:
    images = {}
    for policy_key, obs_key in CAM_KEYS.items():
        # RoboCasa LeRobot videos and gym-wrapper observations are both
        # right-side-up, so eval should send wrapper frames unchanged.
        images[policy_key] = _resize_chw(np.asarray(obs[obs_key]), image_size)

    return {
        "state": _obs_to_state(obs),
        "images": images,
        "prompt": _prompt_for_obs(obs, prompt),
    }


def _obs_to_state(obs: dict[str, Any]) -> np.ndarray:
    if "observation.state" in obs:
        return np.asarray(obs["observation.state"], dtype=np.float32)

    try:
        parts = [np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in STATE_KEYS]
    except KeyError as exc:
        raise KeyError(f"Missing RoboCasa state key {exc.args[0]!r}; available keys: {sorted(obs)}") from exc
    return np.concatenate(parts, axis=0).astype(np.float32)


def _action_to_gym(action: np.ndarray, action_space) -> dict[str, np.ndarray]:
    action = np.asarray(action, dtype=np.float32)
    base_motion = action[0:4].copy()
    base_motion[np.abs(base_motion) < BASE_MOTION_DEADZONE] = 0.0
    action_dict = {
        "action.base_motion": base_motion,
        "action.control_mode": np.full((1,), -1.0, dtype=np.float32),
        "action.end_effector_position": action[5:8],
        "action.end_effector_rotation": action[8:11],
        "action.gripper_close": action[11:12],
    }

    clipped = {}
    for key, value in action_dict.items():
        space = action_space.spaces[key]
        arr = np.asarray(value, dtype=np.float32).reshape(space.shape)
        clipped[key] = np.clip(arr, space.low, space.high).astype(np.float32)
    return clipped


def _apply_eval_overrides(
    action: np.ndarray,
    *,
    step: int,
    force_gripper_after: int | None,
    force_gripper_value: float,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).copy()
    if force_gripper_after is not None and step >= force_gripper_after:
        action[11] = np.float32(force_gripper_value)
    return action


def _maybe_success_value(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if not value:
            return None
        return bool(all(bool(v) for v in value.values()))
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        return bool(np.all(arr))
    return bool(value)


def _check_success(env, info: dict[str, Any]) -> bool:
    for key in ("success", "is_success", "task_success", "eval_success"):
        value = _maybe_success_value(info.get(key))
        if value is not None:
            return value

    for target in (getattr(env, "unwrapped", None), getattr(getattr(env, "unwrapped", None), "env", None)):
        if target is None:
            continue
        for method_name in ("_check_success", "check_success", "_check_successes"):
            method = getattr(target, method_name, None)
            if method is None:
                continue
            try:
                value = _maybe_success_value(method())
            except TypeError:
                continue
            if value is not None:
                return value
        for attr_name in ("success", "eval_success"):
            if hasattr(target, attr_name):
                value = _maybe_success_value(getattr(target, attr_name))
                if value is not None:
                    return value
    return False


def _task_debug_metrics(env) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    target = getattr(getattr(env, "unwrapped", None), "env", None)
    if target is None:
        return metrics

    drawer = getattr(target, "drawer", None)
    if drawer is not None and hasattr(drawer, "get_door_state"):
        try:
            metrics["drawer_door_state"] = {
                str(k): float(v) for k, v in drawer.get_door_state(env=target).items()
            }
        except Exception as exc:  # pragma: no cover - debug best effort
            metrics["drawer_door_state_error"] = repr(exc)

    try:
        site_ids = target.robots[0].eef_site_id
        if isinstance(site_ids, dict) and "right" in site_ids:
            metrics["right_eef_site_pos"] = target.sim.data.site_xpos[site_ids["right"]].tolist()
    except Exception as exc:  # pragma: no cover - debug best effort
        metrics["right_eef_site_pos_error"] = repr(exc)

    return metrics


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--env", default="robocasa/OpenDrawer")
    parser.add_argument("--split", default="target")
    parser.add_argument("--prompt", default="auto")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-every", type=int, default=10)
    parser.add_argument("--debug-out", type=Path, default=None)
    parser.add_argument("--force-gripper-after", type=int, default=None)
    parser.add_argument("--force-gripper-value", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    _patch_eval_action_unmap()
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    env = gym.make(args.env, enable_render=True, split=args.split)

    episodes = []
    debug_records = []
    try:
        for ep in range(args.episodes):
            seed = args.seed + ep
            obs, _ = env.reset(seed=seed)
            episode_prompt = _prompt_for_obs(obs, args.prompt)
            frames = []
            done = False
            success = False
            steps = 0
            policy_calls = 0
            total_reward = 0.0
            last_info: dict[str, Any] = {}

            save_video = args.video_dir is not None and args.video_every > 0 and (ep % args.video_every == 0)
            while not done and steps < args.max_steps:
                state_before = _obs_to_state(obs)
                task_debug_before = _task_debug_metrics(env)
                result = policy.infer(_obs_to_policy_input(obs, args.prompt, args.image_size))
                actions = np.asarray(result["actions"], dtype=np.float32)[: args.chunk_size]
                executed_actions = []
                policy_calls += 1
                chunk_start_step = steps

                for action in actions:
                    executed_action = _apply_eval_overrides(
                        action,
                        step=steps,
                        force_gripper_after=args.force_gripper_after,
                        force_gripper_value=args.force_gripper_value,
                    )
                    executed_actions.append(executed_action)
                    if save_video and CAM_KEYS["agentview_left"] in obs:
                        frames.append(np.asarray(obs[CAM_KEYS["agentview_left"]]))
                    obs, reward, terminated, truncated, info = env.step(_action_to_gym(executed_action, env.action_space))
                    steps += 1
                    total_reward += float(reward)
                    last_info = dict(info)
                    success = _check_success(env, last_info)
                    done = bool(terminated or truncated or success or steps >= args.max_steps)
                    if done:
                        break
                if args.debug_out is not None:
                    state_after = _obs_to_state(obs)
                    task_debug_after = _task_debug_metrics(env)
                    executed_actions_array = np.asarray(executed_actions, dtype=np.float32)
                    debug_records.append(
                        {
                            "episode": ep,
                            "policy_call": policy_calls,
                            "chunk_start_step": chunk_start_step,
                            "chunk_end_step": steps,
                            "prompt": episode_prompt,
                            "state_before": state_before.tolist(),
                            "state_after": state_after.tolist(),
                            "state_delta": (state_after - state_before).tolist(),
                            "task_debug_before": task_debug_before,
                            "task_debug_after": task_debug_after,
                            "action_first": actions[0].tolist(),
                            "action_mean": actions.mean(axis=0).tolist(),
                            "action_std": actions.std(axis=0).tolist(),
                            "action_min": actions.min(axis=0).tolist(),
                            "action_max": actions.max(axis=0).tolist(),
                            "executed_action_first": executed_actions_array[0].tolist(),
                            "executed_action_mean": executed_actions_array.mean(axis=0).tolist(),
                            "executed_action_std": executed_actions_array.std(axis=0).tolist(),
                            "executed_action_min": executed_actions_array.min(axis=0).tolist(),
                            "executed_action_max": executed_actions_array.max(axis=0).tolist(),
                            "success_after_chunk": bool(success),
                        }
                    )
                    args.debug_out.parent.mkdir(parents=True, exist_ok=True)
                    args.debug_out.write_text(json.dumps(debug_records, indent=2, sort_keys=True) + "\n")

            if save_video and frames:
                video_path = args.video_dir / f"episode_{ep:04d}_success_{int(success)}.mp4"
                try:
                    _write_video(video_path, frames, fps=20)
                except Exception:
                    logging.exception("Failed to write video %s; continuing rollout eval.", video_path)

            episode = {
                "episode": ep,
                "seed": seed,
                "prompt": episode_prompt,
                "success": bool(success),
                "steps": steps,
                "policy_calls": policy_calls,
                "total_reward": total_reward,
                "last_info_keys": sorted(last_info.keys()),
            }
            episodes.append(episode)
            success_rate = float(np.mean([e["success"] for e in episodes]))
            print(json.dumps({**episode, "running_success_rate": success_rate}), flush=True)

            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(
                    {
                        "env": args.env,
                        "split": args.split,
                        "prompt": args.prompt,
                        "episodes_requested": args.episodes,
                        "episodes_completed": len(episodes),
                        "success_rate": success_rate,
                        "episodes": episodes,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
