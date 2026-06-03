"""Verify whether the cache's reset_to path reproduces the randomized agentview
camera that the stored RGB was rendered with.

For PnPCounterToCab/demo_1:
  1. Build env + reset_to(model, ep_meta, states[0]) exactly like the cache.
  2. Read live mujoco model.cam_pos/cam_quat for robot0_agentview_left (these are
     relative to parent_body base0_support, == what ep_meta cam_configs stores).
  3. Compare to ep_meta["cam_configs"]["robot0_agentview_left"].
  4. Render agentview RGB at 128 and dump next to the stored HDF5 RGB.

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


def main():
    import robosuite_models  # noqa
    import robocasa  # noqa
    from robocasa.utils.env_utils import create_env
    from robocasa.scripts.playback_dataset import reset_to

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
        stored_rgb = np.asarray(f[f"data/{dk}/obs/robot0_agentview_left_image"][0])  # (128,128,3)

    ep_meta = json.loads(ep_meta_json)
    cc = ep_meta["cam_configs"][CAM]
    print(f"ep_meta cam_configs[{CAM}]:")
    print(f"   pos  = {np.array(cc['pos'])}")
    print(f"   quat = {np.array(cc['quat'])}")

    env = create_env(env_name=task, robots=robots,
                     camera_names=[CAM, "robot0_eye_in_hand"],
                     camera_widths=128, camera_heights=128,
                     render_onscreen=False, seed=0)
    reset_to(env, {"model": model_file, "ep_meta": ep_meta_json, "states": states[0]})

    model = env.sim.model
    cid = model.camera_name2id(CAM)
    live_pos = np.array(model.cam_pos[cid])
    live_quat = np.array(model.cam_quat[cid])  # mujoco wxyz
    print(f"\nlive mujoco model.cam_pos[{CAM}]  = {live_pos}")
    print(f"live mujoco model.cam_quat[{CAM}] = {live_quat}  (mujoco wxyz)")

    dpos = np.linalg.norm(live_pos - np.array(cc["pos"]))
    print(f"\n||live_pos - ep_meta_pos|| = {dpos*100:.2f} cm")
    if dpos > 1e-3:
        print("  => MISMATCH: reset_to did NOT restore the recorded randomized agentview pose.")
    else:
        print("  => match.")

    # render agentview and compare to stored
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()
    rgb = env.sim.render(camera_name=CAM, width=128, height=128)[::-1, :]  # y-up -> y-down
    OUT.mkdir(parents=True, exist_ok=True)
    side = np.concatenate([stored_rgb, np.full((128, 6, 3), 255, np.uint8), rgb], axis=1)
    Image.fromarray(side).save(OUT / "_camrand_stored_vs_rerender.png")
    diff = np.abs(stored_rgb.astype(int) - rgb.astype(int)).mean()
    print(f"\nstored vs re-render mean|Δpix| = {diff:.1f} / 255")
    print(f"saved {OUT/'_camrand_stored_vs_rerender.png'}  (left=stored RGB, right=re-render)")
    env.close()


if __name__ == "__main__":
    main()
