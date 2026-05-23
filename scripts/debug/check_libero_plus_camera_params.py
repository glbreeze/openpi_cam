#!/usr/bin/env python3
"""Dump live MuJoCo camera parameters for selected LIBERO-plus tasks."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GEO_ROOT = REPO_ROOT.parent
LIBERO_PLUS_ROOT = pathlib.Path(os.environ.get("LIBERO_PLUS_ROOT", "/scratch/yp2841/LIBERO-plus"))
LIBERO_PLUS_BENCHMARK_ROOT = pathlib.Path(
    os.environ.get("LIBERO_PLUS_BENCHMARK_ROOT", LIBERO_PLUS_ROOT / "libero" / "libero")
)
LIBERO_PLUS_DATASETS_ROOT = pathlib.Path(
    os.environ.get("LIBERO_PLUS_DATASETS_ROOT", LIBERO_PLUS_ROOT / "libero" / "datasets")
)
LIBERO_PLUS_ASSETS_ROOT = pathlib.Path(
    os.environ.get("LIBERO_PLUS_ASSETS_ROOT", LIBERO_PLUS_BENCHMARK_ROOT / "assets")
)


def _prepare_env() -> None:
    config_root = pathlib.Path(
        os.environ.get("LIBERO_CONFIG_PATH", f"/scratch/{os.environ.get('USER', 'yp2841')}/tmp/libero_plus_cam_check")
    )
    config_root.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    (config_root / "config.yaml").write_text(
        "\n".join(
            [
                f"benchmark_root: {LIBERO_PLUS_BENCHMARK_ROOT}",
                f"bddl_files: {LIBERO_PLUS_BENCHMARK_ROOT / 'bddl_files'}",
                f"init_states: {LIBERO_PLUS_BENCHMARK_ROOT / 'init_files'}",
                f"datasets: {LIBERO_PLUS_DATASETS_ROOT}",
                f"assets: {LIBERO_PLUS_ASSETS_ROOT}",
            ]
        )
        + "\n"
    )
    os.environ.setdefault("OPENPI_LIBERO_ROOT", str(LIBERO_PLUS_ROOT))
    os.environ.setdefault("OPENPI_LIBERO_BENCHMARK_ROOT", str(LIBERO_PLUS_BENCHMARK_ROOT))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    for path in (REPO_ROOT, REPO_ROOT / "src", LIBERO_PLUS_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _parse_task_ids(args: argparse.Namespace) -> list[int]:
    ids: list[int] = []
    if args.task_id_list:
        for raw in pathlib.Path(args.task_id_list).read_text().splitlines():
            raw = raw.strip()
            if raw and not raw.startswith("#"):
                ids.append(int(raw))
    ids.extend(args.task_ids)
    if args.limit is not None:
        ids = ids[: args.limit]
    return ids


def _rotation_delta_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    rel = r0.T @ r1
    cos_theta = (np.trace(rel) - 1.0) / 2.0
    return float(math.degrees(math.acos(float(np.clip(cos_theta, -1.0, 1.0)))))


def _round_array(array: np.ndarray, digits: int = 6):
    return np.round(np.asarray(array, dtype=np.float64), digits).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-ids", type=int, nargs="*", default=[])
    parser.add_argument("--task-id-list", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    _prepare_env()

    from libero.libero import benchmark
    from examples.libero import main_plus

    task_ids = _parse_task_ids(args)
    if not task_ids:
        raise ValueError("Pass --task-ids or --task-id-list")

    suite = benchmark.get_benchmark_dict()[args.suite]()
    records = []
    base_by_camera: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        env, description = main_plus._get_libero_env(task, main_plus.LIBERO_ENV_RESOLUTION, args.seed)
        try:
            env.reset()
            if len(initial_states):
                env.set_init_state(initial_states[0])

            cameras = {}
            for camera_name in ("agentview", "robot0_eye_in_hand"):
                extrinsic = main_plus._get_camera_extrinsic(env, camera_name)
                intrinsic = main_plus._get_camera_intrinsic(env, camera_name, args.resize_size, args.resize_size)
                cam_id = env.sim.model.camera_name2id(camera_name)
                fovy = float(env.sim.model.cam_fovy[cam_id])

                base = base_by_camera.setdefault(camera_name, (extrinsic[:3, 3].copy(), extrinsic[:3, :3].copy()))
                base_pos, base_rot = base
                cameras[camera_name] = {
                    "pos": _round_array(extrinsic[:3, 3]),
                    "rot": _round_array(extrinsic[:3, :3]),
                    "fovy": round(fovy, 6),
                    "K": _round_array(intrinsic),
                    "delta_pos_norm_from_first": round(float(np.linalg.norm(extrinsic[:3, 3] - base_pos)), 6),
                    "delta_rot_deg_from_first": round(_rotation_delta_deg(base_rot, extrinsic[:3, :3]), 6),
                }

            records.append(
                {
                    "task_id": task_id,
                    "description": description,
                    "stripped_description": main_plus._strip_libero_plus_task_suffix(description),
                    "cameras": cameras,
                }
            )
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()

    print(json.dumps({"suite": args.suite, "task_ids": task_ids, "records": records}, indent=2))


if __name__ == "__main__":
    main()
