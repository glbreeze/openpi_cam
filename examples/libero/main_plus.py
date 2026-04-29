import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import sys

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GEO_ROOT = REPO_ROOT.parent
LIBERO_SRC_ROOT = pathlib.Path(os.environ.get("OPENPI_LIBERO_ROOT", REPO_ROOT / "third_party" / "libero"))
LIBERO_BENCHMARK_ROOT = pathlib.Path(os.environ.get("OPENPI_LIBERO_BENCHMARK_ROOT", LIBERO_SRC_ROOT / "libero"))

if str(LIBERO_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_SRC_ROOT))

if "LIBERO_CONFIG_PATH" not in os.environ:
    default_libero_config = pathlib.Path.home() / ".libero_openpi_cam_plus"
    os.environ["LIBERO_CONFIG_PATH"] = str(default_libero_config)
else:
    default_libero_config = pathlib.Path(os.environ["LIBERO_CONFIG_PATH"])

default_libero_config.mkdir(parents=True, exist_ok=True)
libero_config_file = default_libero_config / "config.yaml"
if not libero_config_file.exists():
    libero_config_file.write_text(
        "\n".join(
            [
                f"benchmark_root: {LIBERO_BENCHMARK_ROOT}",
                f"bddl_files: {LIBERO_BENCHMARK_ROOT / 'bddl_files'}",
                f"init_states: {LIBERO_BENCHMARK_ROOT / 'init_files'}",
                f"datasets: {GEO_ROOT / 'libero_cam_rlds'}",
                f"assets: {LIBERO_BENCHMARK_ROOT / 'assets'}",
            ]
        )
        + "\n"
    )

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    model_name: str = ""
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_object"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    task_id_start: int = 0
    task_id_end: int = -1

    video_out_path: str = "data/libero/videos"
    summary_out_path: str | None = None

    seed: int = 7


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info("Task suite: %s", args.task_suite_name)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    task_id_start = max(0, args.task_id_start)
    task_id_end = num_tasks_in_suite if args.task_id_end < 0 else min(args.task_id_end, num_tasks_in_suite)
    if task_id_start >= task_id_end:
        raise ValueError(
            f"Invalid task range [{task_id_start}, {task_id_end}) for suite with {num_tasks_in_suite} tasks"
        )
    logging.info("Task range: [%d, %d) out of %d", task_id_start, task_id_end, num_tasks_in_suite)

    total_episodes, total_successes = 0, 0
    records: list[dict] = []
    for task_id in tqdm.tqdm(range(task_id_start, task_id_end)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info("\nTask: %s", task_description)
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            replay_images = []

            logging.info("Starting episode %d...", task_episodes + 1)
            done = False
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "observation/agent_extrinsic": _get_camera_extrinsic(env, "agentview"),
                            "observation/wrist_extrinsic": _get_camera_extrinsic(env, "robot0_eye_in_hand"),
                            "observation/agent_intrinsic": _get_camera_intrinsic(
                                env, "agentview", args.resize_size, args.resize_size
                            ),
                            "observation/wrist_intrinsic": _get_camera_intrinsic(
                                env, "robot0_eye_in_hand", args.resize_size, args.resize_size
                            ),
                            "prompt": str(task_description),
                        }

                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    logging.error("Caught exception: %s", e)
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            logging.info("Success: %s", done)
            logging.info("# episodes completed so far: %d", total_episodes)
            logging.info("# successes: %d (%.1f%%)", total_successes, total_successes / total_episodes * 100.0)

        task_success_rate = float(task_successes) / float(task_episodes)
        total_success_rate = float(total_successes) / float(total_episodes)
        logging.info("Current task success rate: %s", task_success_rate)
        logging.info("Current total success rate: %s", total_success_rate)

        records.append(
            {
                "task_id": task_id,
                "task_description": task_description,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": task_success_rate,
            }
        )

    summary = {
        "model_name": args.model_name or None,
        "task_suite_name": args.task_suite_name,
        "task_range": [task_id_start, task_id_end],
        "num_trials_per_task": args.num_trials_per_task,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": float(total_successes) / float(total_episodes),
        "records": records,
    }

    logging.info("Total success rate: %s", summary["total_success_rate"])
    logging.info("Total episodes: %d", total_episodes)

    if args.summary_out_path:
        summary_path = pathlib.Path(args.summary_out_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _get_camera_extrinsic(env, camera_name):
    cam_id = env.sim.model.camera_name2id(camera_name)
    cam_pos = np.asarray(env.sim.data.cam_xpos[cam_id], dtype=np.float32)
    cam_rot = np.asarray(env.sim.data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)

    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = cam_rot
    extrinsic[:3, 3] = cam_pos
    return extrinsic


def _get_camera_intrinsic(env, camera_name, image_h, image_w):
    cam_id = env.sim.model.camera_name2id(camera_name)
    fovy_rad = np.deg2rad(float(env.sim.model.cam_fovy[cam_id]))
    fy = (image_h / 2.0) / np.tan(fovy_rad / 2.0)
    fx = fy
    cx = image_w / 2.0
    cy = image_h / 2.0
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
