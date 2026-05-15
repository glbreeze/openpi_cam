#!/usr/bin/env python
import argparse
import importlib
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


def _add_robotwin_paths(robotwin_root: str) -> None:
    for rel in ("", "policy", "description/utils", "script"):
        path = os.path.join(robotwin_root, rel)
        if path not in sys.path:
            sys.path.insert(0, path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception as exc:
        raise SystemExit("No Task") from exc
    return env_instance


def eval_function_decorator(policy_name, model_name):
    policy_model = importlib.import_module(policy_name)
    return getattr(policy_model, model_name)


def get_camera_config(robotwin_root: str, camera_type):
    camera_config_path = os.path.join(robotwin_root, "task_config/_camera_config.yml")
    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _load_eval_args(robotwin_root: str, usr_args: dict) -> dict:
    from envs import CONFIGS_PATH

    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    with open(os.path.join(robotwin_root, "task_config", f"{task_config}.yml"), "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = usr_args["ckpt_setting"]

    embodiment_type = args.get("embodiment")
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(name):
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise RuntimeError("No embodiment files")
        return robot_file

    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
        embodiment_name = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["_embodiment_name"] = embodiment_name
    return args


def eval_policy_debug(robotwin_root, task_name, task_env, args, model, usr_args, st_seed, save_dir):
    from envs.utils.create_actor import UnStableError
    from generate_episode_instructions import generate_episode_descriptions
    from PIL import Image

    policy_name = args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    test_num = int(usr_args.get("test_num", 1))
    max_seed_attempts = int(usr_args.get("max_seed_attempts", max(50, test_num * 20)))
    expert_check = bool(usr_args.get("expert_check", True))
    debug_log_every_seed = int(usr_args.get("debug_log_every_seed", 1))
    debug_save_frames = bool(int(usr_args.get("debug_save_frames", 0)))
    debug_frame_every = max(1, int(usr_args.get("debug_frame_every", 50)))
    frame_dir = Path(save_dir) / "frames"
    if debug_save_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)

    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    camera_config = get_camera_config(robotwin_root, args["camera"]["head_camera_type"])
    video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
    if args["eval_video_log"]:
        args["eval_video_save_dir"] = save_dir

    print(f"[debug] Task Name: {args['task_name']}", flush=True)
    print(f"[debug] Policy Name: {args['policy_name']}", flush=True)
    print(f"[debug] test_num={test_num} max_seed_attempts={max_seed_attempts} expert_check={expert_check}", flush=True)

    task_env.suc = 0
    task_env.test_num = 0
    now_id = 0
    succ_seed = 0
    now_seed = st_seed
    attempts = 0
    clear_cache_freq = args["clear_cache_freq"]
    args["eval_mode"] = True

    while succ_seed < test_num and attempts < max_seed_attempts:
        attempts += 1
        render_freq = args["render_freq"]
        args["render_freq"] = 0
        episode_info = None

        if attempts == 1 or attempts % debug_log_every_seed == 0:
            print(f"[debug] expert-check attempt={attempts} now_id={now_id} seed={now_seed}", flush=True)

        if expert_check:
            try:
                print(f"[debug] setup_demo(expert) seed={now_seed}", flush=True)
                task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                print(f"[debug] play_once(expert) seed={now_seed}", flush=True)
                episode_info = task_env.play_once()
                print(
                    f"[debug] expert done seed={now_seed} plan_success={getattr(task_env, 'plan_success', None)} "
                    f"check_success={task_env.check_success()}",
                    flush=True,
                )
                task_env.close_env()
            except UnStableError as exc:
                print(f"[debug] UnStableError seed={now_seed}: {exc}", flush=True)
                task_env.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception:
                print(f"[debug] expert exception seed={now_seed}", flush=True)
                print(traceback.format_exc(), flush=True)
                task_env.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue

        if (not expert_check) or (task_env.plan_success and task_env.check_success()):
            succ_seed += 1
            print(f"[debug] accepted seed={now_seed} succ_seed={succ_seed}/{test_num}", flush=True)
        else:
            print(f"[debug] rejected seed={now_seed}", flush=True)
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq
        print(f"[debug] setup_demo(eval) seed={now_seed}", flush=True)
        task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]] if episode_info is not None else [{"info": {}}]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        task_env.set_instruction(instruction=instruction)
        print(f"[debug] instruction={instruction}", flush=True)

        ffmpeg = None
        if task_env.eval_video_path is not None:
            if shutil.which("ffmpeg") is None:
                print("[debug] ffmpeg not found; skipping mp4 video and saving PNG frames instead", flush=True)
                task_env.eval_video_path = None
            else:
                ffmpeg = subprocess.Popen(
                    [
                        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                        "-pixel_format", "rgb24", "-video_size", video_size,
                        "-framerate", "10", "-i", "-", "-pix_fmt", "yuv420p",
                        "-vcodec", "libx264", "-crf", "23",
                        f"{task_env.eval_video_path}/episode{task_env.test_num}.mp4",
                    ],
                    stdin=subprocess.PIPE,
                )
                task_env._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        step_log_every = int(usr_args.get("debug_step_log_every", 25))
        while task_env.take_action_cnt < task_env.step_lim:
            if task_env.take_action_cnt == 0 or task_env.take_action_cnt % step_log_every == 0:
                print(f"[debug] rollout step={task_env.take_action_cnt}/{task_env.step_lim}", flush=True)
            observation = task_env.get_obs()
            if debug_save_frames and (
                task_env.take_action_cnt == 0
                or task_env.take_action_cnt % debug_frame_every == 0
                or task_env.take_action_cnt == task_env.step_lim - 1
            ):
                for cam_name in ("head_camera", "left_camera", "right_camera"):
                    arr = np.asarray(observation["observation"][cam_name]["rgb"])
                    if np.issubdtype(arr.dtype, np.floating):
                        arr = (255 * arr).clip(0, 255).astype(np.uint8)
                    Image.fromarray(arr.astype(np.uint8)).save(
                        frame_dir / f"episode{task_env.test_num:03d}_step{task_env.take_action_cnt:04d}_{cam_name}.png"
                    )
            eval_func(task_env, model, observation)
            if task_env.eval_success:
                succ = True
                break

        if task_env.eval_video_path is not None:
            task_env._del_eval_video_ffmpeg()

        if succ:
            task_env.suc += 1
            print("[debug] Success!", flush=True)
        else:
            print("[debug] Fail!", flush=True)

        now_id += 1
        task_env.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        if task_env.render_freq:
            task_env.viewer.close()
        task_env.test_num += 1
        print(f"[debug] success_rate={task_env.suc}/{task_env.test_num} current_seed={now_seed}", flush=True)
        now_seed += 1

    result_path = os.path.join(save_dir, "_result.txt")
    with open(result_path, "w", encoding="utf-8") as file:
        file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write(str(task_env.suc / max(1, task_env.test_num)))
    print(f"[debug] Data has been saved to {result_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", default="/scratch/yp2841/geometry-vla/RoboTwin")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def parse_override_pairs(pairs):
    override_dict = {}
    for i in range(0, len(pairs), 2):
        key = pairs[i].lstrip("--")
        value = pairs[i + 1]
        try:
            value = eval(value)
        except Exception:
            pass
        override_dict[key] = value
    return override_dict


def main():
    cli_args = parse_args()
    robotwin_root = os.path.abspath(cli_args.robotwin_root)
    os.chdir(robotwin_root)
    _add_robotwin_paths(robotwin_root)

    from test_render import Sapien_TEST

    Sapien_TEST()
    with open(cli_args.config, "r", encoding="utf-8") as f:
        usr_args = yaml.safe_load(f)
    if cli_args.overrides:
        usr_args.update(parse_override_pairs(cli_args.overrides))

    args = _load_eval_args(robotwin_root, usr_args)
    for key in ("eval_video_log", "render_freq"):
        if key in usr_args:
            args[key] = usr_args[key]
    task_name = usr_args["task_name"]
    policy_name = usr_args["policy_name"]
    args["policy_name"] = policy_name

    save_dir = Path(
        f"eval_result/{task_name}/{policy_name}/{usr_args['task_config']}/"
        f"{usr_args['ckpt_setting']}/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    task_env = class_decorator(task_name)
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])
    model = eval_function_decorator(policy_name, "get_model")(usr_args)

    st_seed = 100000 * (1 + int(usr_args["seed"]))
    eval_policy_debug(robotwin_root, task_name, task_env, args, model, usr_args, st_seed, save_dir)


if __name__ == "__main__":
    main()
