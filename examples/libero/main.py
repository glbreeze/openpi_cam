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
import torch
import tqdm
import tyro

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GEO_ROOT = REPO_ROOT.parent
THIRD_PARTY_LIBERO = REPO_ROOT / "third_party" / "libero"
THIRD_PARTY_LIBERO_ROOT = THIRD_PARTY_LIBERO / "libero" / "libero"

if str(THIRD_PARTY_LIBERO) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_LIBERO))

if "LIBERO_CONFIG_PATH" not in os.environ:
    default_libero_config = pathlib.Path.home() / ".libero_openpi_cam"
    os.environ["LIBERO_CONFIG_PATH"] = str(default_libero_config)
else:
    default_libero_config = pathlib.Path(os.environ["LIBERO_CONFIG_PATH"])

default_libero_config.mkdir(parents=True, exist_ok=True)
libero_config_file = default_libero_config / "config.yaml"
if not libero_config_file.exists():
    libero_config_file.write_text(
        "\n".join(
            [
                f"benchmark_root: {THIRD_PARTY_LIBERO_ROOT}",
                f"bddl_files: {THIRD_PARTY_LIBERO_ROOT / 'bddl_files'}",
                f"init_states: {THIRD_PARTY_LIBERO_ROOT / 'init_files'}",
                f"datasets: {GEO_ROOT / 'libero_cam_rlds'}",
                f"assets: {THIRD_PARTY_LIBERO_ROOT / 'assets'}",
            ]
        )
        + "\n"
    )

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    model_name: str = ""
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    task_id_start: int = 0  # Inclusive task index to start from
    task_id_end: int = -1  # Exclusive task index to stop at; -1 means run through the end

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    summary_out_path: str | None = None  # Optional JSON summary path

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    task_id_start = max(0, args.task_id_start)
    task_id_end = num_tasks_in_suite if args.task_id_end < 0 else min(args.task_id_end, num_tasks_in_suite)
    if task_id_start >= task_id_end:
        raise ValueError(
            f"Invalid task range [{task_id_start}, {task_id_end}) for suite with {num_tasks_in_suite} tasks"
        )
    logging.info(f"Task range: [{task_id_start}, {task_id_end}) out of {num_tasks_in_suite}")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    records: list[dict] = []
    for task_id in tqdm.tqdm(range(task_id_start, task_id_end)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = _load_task_init_states(task)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
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
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        task_success_rate = float(task_successes) / float(task_episodes)
        total_success_rate = float(total_successes) / float(total_episodes)
        logging.info(f"Current task success rate: {task_success_rate}")
        logging.info(f"Current total success rate: {total_success_rate}")

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

    logging.info(f"Total success rate: {summary['total_success_rate']}")
    logging.info(f"Total episodes: {total_episodes}")

    if args.summary_out_path:
        summary_path = pathlib.Path(args.summary_out_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _load_task_init_states(task):
    init_states_path = pathlib.Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return torch.load(init_states_path, weights_only=False)


def _get_camera_extrinsic(env, camera_name):
    cam_id = env.sim.model.camera_name2id(camera_name)
    cam_pos = np.asarray(env.sim.data.cam_xpos[cam_id], dtype=np.float32)
    cam_rot = np.asarray(env.sim.data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)

    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = cam_rot
    extrinsic[:3, 3] = cam_pos
    return extrinsic


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
