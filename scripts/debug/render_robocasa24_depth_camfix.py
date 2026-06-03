"""Render agentview depth for a few episodes with BOTH the buggy reset_to and
the cam-config-injection fix, so we can visualize the alignment difference.

Saves per-episode npz to OUT/ with: rgb128, depth_old, depth_new (224, meters),
frame indices, task/demo labels. Visualized by the companion viz step.

Run: MUJOCO_GL=egl /home/asus/miniconda3/envs/robocasa24/bin/python <this>
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import json
from pathlib import Path
import numpy as np
import h5py

RAW = Path("~/Research/robocasa-human50/raw_human_im").expanduser()
OUT = Path("~/Research/robocasa24/viz_targets_224/camfix_render").expanduser()
CAM = "robot0_agentview_left"
RES = 224
N_FRAMES = 3

TASKS_24 = (
    "PnPCounterToCab", "PnPCabToCounter", "PnPCounterToSink", "PnPSinkToCounter",
    "PnPCounterToMicrowave", "PnPMicrowaveToCounter", "PnPCounterToStove",
    "PnPStoveToCounter",
    "OpenSingleDoor", "CloseSingleDoor", "OpenDoubleDoor", "CloseDoubleDoor",
    "OpenDrawer", "CloseDrawer",
    "TurnOnSinkFaucet", "TurnOffSinkFaucet", "TurnSinkSpout",
    "TurnOnStove", "TurnOffStove",
    "CoffeeSetupMug", "CoffeeServeMug", "CoffeePressButton",
    "TurnOnMicrowave", "TurnOffMicrowave",
)
TARGET_EPISODES = [0, 1, 700]  # PnPCounterToCab d1, d2; CloseDrawer


def resolve(task):
    c = sorted((RAW / "v0.1" / "single_stage").glob(f"*/{task}/*/demo_gentex_im128_randcams.hdf5"))
    c = [p for p in c if "/mg/" not in str(p)]
    return c[0] if c else None


def build_plan():
    """episode_idx -> (task, demo_key, src) in canonical order, matching the cache."""
    plan, cum = {}, 0
    want = set(TARGET_EPISODES)
    for t in TASKS_24:
        src = resolve(t)
        if src is None:
            continue
        with h5py.File(src, "r") as f:
            demos = sorted((k for k in f["data"].keys() if k.startswith("demo_")),
                           key=lambda s: int(s.split("_")[1]))
        for d in demos:
            if cum in want:
                plan[cum] = (t, d, src)
            cum += 1
        if len(plan) == len(want):
            break
    return plan


def reset_to_buggy(env, model_file, ep_meta_json, state0):
    from robocasa.scripts.playback_dataset import reset_to
    reset_to(env, {"model": model_file, "ep_meta": ep_meta_json, "states": state0})


def reset_to_fixed(env, model_file, ep_meta_json, state0):
    ep_meta = json.loads(ep_meta_json)
    if hasattr(env, "set_ep_meta"):
        env.set_ep_meta(ep_meta)
    elif hasattr(env, "set_attrs_from_ep_meta"):
        env.set_attrs_from_ep_meta(ep_meta)
    env.reset()
    env._cam_configs = ep_meta["cam_configs"]  # KEY FIX
    xml = env.edit_model_xml(model_file)
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(state0)
    env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()
    elif hasattr(env, "update_sites"):
        env.update_sites()


def render_depth_seq(env, states, frame_idxs):
    from robosuite.utils.camera_utils import get_real_depth_map
    out = np.empty((len(frame_idxs), RES, RES), np.float32)
    for i, t in enumerate(frame_idxs):
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()
        _, d = env.sim.render(camera_name=CAM, width=RES, height=RES, depth=True)
        d = get_real_depth_map(env.sim, d)
        if d.ndim == 3:
            d = d[..., 0]
        out[i] = d[::-1, :].copy()  # flipud -> y-down, same as GT cache
    return out


def main():
    import robosuite_models  # noqa
    import robocasa  # noqa
    from robocasa.utils.env_utils import create_env

    OUT.mkdir(parents=True, exist_ok=True)
    plan = build_plan()
    print("plan:", {k: (v[0], v[1]) for k, v in plan.items()})

    # group episodes by task so we build one env per task
    by_task = {}
    for ep, (t, d, src) in plan.items():
        by_task.setdefault((t, src), []).append((ep, d))

    for (task, src), eps in by_task.items():
        with h5py.File(src, "r") as f:
            env_args = json.loads(f["data"].attrs["env_args"])
            robots = env_args["env_kwargs"]["robots"]
            env = create_env(env_name=task, robots=robots,
                             camera_names=[CAM, "robot0_eye_in_hand"],
                             camera_widths=RES, camera_heights=RES,
                             render_onscreen=False, seed=0)
            for ep, dk in eps:
                model_file = f[f"data/{dk}"].attrs["model_file"]
                ep_meta_json = f[f"data/{dk}"].attrs["ep_meta"]
                states = np.asarray(f[f"data/{dk}/states"][:])
                rgb = np.asarray(f[f"data/{dk}/obs/robot0_agentview_left_image"][:])
                lang = json.loads(ep_meta_json).get("lang", "")
                T = states.shape[0]
                fidx = np.linspace(0, T - 1, N_FRAMES, dtype=int)

                reset_to_buggy(env, model_file, ep_meta_json, states[0])
                depth_old = render_depth_seq(env, states, fidx)
                reset_to_fixed(env, model_file, ep_meta_json, states[0])
                depth_new = render_depth_seq(env, states, fidx)

                np.savez_compressed(
                    OUT / f"ep{ep:06d}.npz",
                    rgb=rgb[fidx], depth_old=depth_old, depth_new=depth_new,
                    frame_idxs=fidx, task=task, demo=dk, lang=lang, T=T,
                )
                print(f"  ep{ep} {task}/{dk}  T={T}  frames={list(fidx)}  saved")
            env.close()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
