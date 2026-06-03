"""Test the fix: re-inject ep_meta['cam_configs'] before edit_model_xml so the
agentview camera matches the pose the stored RGB was rendered with.

Run: MUJOCO_GL=egl /home/asus/miniconda3/envs/robocasa24/bin/python <this>
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import json
from pathlib import Path
import numpy as np
import h5py
from PIL import Image

RAW = Path("~/Research/robocasa-human50/raw_human_im").expanduser()
OUT = Path("~/Research/robocasa24/viz_targets_224").expanduser()
CAM = "robot0_agentview_left"


def resolve(task):
    c = sorted((RAW / "v0.1" / "single_stage").glob(f"*/{task}/*/demo_gentex_im128_randcams.hdf5"))
    return [p for p in c if "/mg/" not in str(p)][0]


def reset_to_with_cam(env, model_file, ep_meta_json, state0):
    """Like robocasa.playback_dataset.reset_to, but restores the recorded
    cam_configs so randomized cameras match the source render."""
    import robosuite
    ep_meta = json.loads(ep_meta_json)
    if hasattr(env, "set_ep_meta"):
        env.set_ep_meta(ep_meta)
    elif hasattr(env, "set_attrs_from_ep_meta"):
        env.set_attrs_from_ep_meta(ep_meta)
    env.reset()  # this re-randomizes env._cam_configs
    # KEY FIX: overwrite with the recorded cam_configs before XML edit.
    env._cam_configs = ep_meta["cam_configs"]
    xml = env.edit_model_xml(model_file)
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(state0)
    env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()
    elif hasattr(env, "update_sites"):
        env.update_sites()


def main():
    import robosuite_models  # noqa
    import robocasa  # noqa
    from robocasa.utils.env_utils import create_env

    src = resolve("PnPCounterToCab")
    with h5py.File(src, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        task = env_args["env_name"]
        robots = env_args["env_kwargs"]["robots"]
        demos = sorted((k for k in f["data"].keys() if k.startswith("demo_")),
                       key=lambda s: int(s.split("_")[1]))
        dk = demos[0]
        model_file = f[f"data/{dk}"].attrs["model_file"]
        ep_meta_json = f[f"data/{dk}"].attrs["ep_meta"]
        states = np.asarray(f[f"data/{dk}/states"][:])
        stored_rgb = np.asarray(f[f"data/{dk}/obs/robot0_agentview_left_image"][0])

    cc = json.loads(ep_meta_json)["cam_configs"][CAM]
    env = create_env(env_name=task, robots=robots,
                     camera_names=[CAM, "robot0_eye_in_hand"],
                     camera_widths=128, camera_heights=128,
                     render_onscreen=False, seed=0)
    reset_to_with_cam(env, model_file, ep_meta_json, states[0])

    model = env.sim.model
    cid = model.camera_name2id(CAM)
    live_pos = np.array(model.cam_pos[cid])
    dpos = np.linalg.norm(live_pos - np.array(cc["pos"]))
    print(f"ep_meta pos = {np.array(cc['pos'])}")
    print(f"live    pos = {live_pos}")
    print(f"||Δ|| = {dpos*100:.3f} cm  ->", "FIXED" if dpos < 1e-3 else "still off")

    rgb = env.sim.render(camera_name=CAM, width=128, height=128)[::-1, :]
    side = np.concatenate([stored_rgb, np.full((128, 6, 3), 255, np.uint8), rgb], axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(side).save(OUT / "_camfix_stored_vs_rerender.png")
    diff = np.abs(stored_rgb.astype(int) - rgb.astype(int)).mean()
    print(f"stored vs re-render mean|Δpix| = {diff:.1f} / 255 (was 84.8)")
    print(f"saved {OUT/'_camfix_stored_vs_rerender.png'}")
    env.close()


if __name__ == "__main__":
    main()
