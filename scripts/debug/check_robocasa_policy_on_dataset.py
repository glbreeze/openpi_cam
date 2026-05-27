"""Compare a RoboCasa policy server against LeRobot training frames.

This is a lightweight offline diagnostic: it sends real dataset observations to
an already-running policy server and compares the first predicted action, plus
optionally the predicted chunk, against the LeRobot action labels.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from openpi_client import websocket_client_policy


CAMERA_COLUMNS = {
    "agentview_left": "observation.images.robot0_agentview_left",
    "eye_in_hand": "observation.images.robot0_eye_in_hand",
}
MATRIX_COLUMNS = {
    "agentview_left": (
        "observation.agentview_left_intrinsic",
        "observation.agentview_left_extrinsic",
    ),
    "eye_in_hand": (
        "observation.eye_in_hand_intrinsic",
        "observation.eye_in_hand_extrinsic",
    ),
}
BASE_COLUMNS = [
    "episode_index",
    "frame_index",
    "task_index",
    "observation.state",
    "action",
]
ACTION_SLICES = {
    "base": slice(0, 4),
    "control": slice(4, 5),
    "eef_pos": slice(5, 8),
    "eef_rot": slice(8, 11),
    "gripper": slice(11, 12),
}


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.replace(":", ",").split(",") if part.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.replace(":", ",").split(",") if part.strip()]


def _load_tasks(dataset_root: Path) -> dict[int, str]:
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    tasks: dict[int, str] = {}
    with tasks_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            tasks[int(item["task_index"])] = str(item["task"])
    return tasks


def _episode_parquet(dataset_root: Path, episode: int) -> Path:
    chunk = episode // 1000
    return dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"


def _apply_image_mode(image_hwc: np.ndarray, image_mode: str) -> np.ndarray:
    if image_mode == "as_is":
        return image_hwc
    if image_mode == "vflip":
        return image_hwc[::-1, :, :]
    if image_mode == "hflip":
        return image_hwc[:, ::-1, :]
    if image_mode == "rot180":
        return image_hwc[::-1, ::-1, :]
    raise ValueError(f"Unknown image mode: {image_mode}")


def _decode_lerobot_image(image_item: Any, dataset_root: Path, image_size: int, image_mode: str) -> np.ndarray:
    if isinstance(image_item, dict):
        raw_bytes = image_item.get("bytes")
        rel_path = image_item.get("path")
    else:
        raw_bytes = None
        rel_path = None

    if raw_bytes:
        image = Image.open(io.BytesIO(raw_bytes))
    elif rel_path:
        image = Image.open(dataset_root / rel_path)
    else:
        raise ValueError(f"Cannot decode LeRobot image item: {type(image_item)!r}")

    image = image.convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.BILINEAR)
    image_hwc = _apply_image_mode(np.asarray(image, dtype=np.uint8), image_mode)
    return np.ascontiguousarray(image_hwc.transpose(2, 0, 1))


def _flip_opencv_mujoco_columns(T: Any) -> np.ndarray:
    out = np.asarray(T, dtype=np.float32).copy()
    out[:3, 1:3] *= -1.0
    return out


def _policy_input(
    row: dict[str, Any],
    dataset_root: Path,
    tasks: dict[int, str],
    image_size: int,
    extrinsics_mode: str,
    image_mode: str,
) -> dict[str, Any]:
    obs: dict[str, Any] = {
        "state": np.asarray(row["observation.state"], dtype=np.float32),
        "images": {
            policy_cam: _decode_lerobot_image(row[column], dataset_root, image_size, image_mode)
            for policy_cam, column in CAMERA_COLUMNS.items()
        },
        "prompt": tasks[int(row["task_index"])],
    }

    for policy_cam, (intr_col, ext_col) in MATRIX_COLUMNS.items():
        obs[f"observation.{policy_cam}_intrinsic"] = np.asarray(row[intr_col], dtype=np.float32)
        T = np.asarray(row[ext_col], dtype=np.float32)
        if extrinsics_mode == "unflip":
            T = _flip_opencv_mujoco_columns(T)
        elif extrinsics_mode != "dataset":
            raise ValueError(f"Unknown extrinsics mode: {extrinsics_mode}")
        obs[f"observation.{policy_cam}_extrinsic"] = T

    return obs


def _action_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float32)[:12]
    gt = np.asarray(gt, dtype=np.float32)[:12]
    diff = pred - gt
    out: dict[str, Any] = {
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "pred": pred.tolist(),
        "gt": gt.tolist(),
        "diff": diff.tolist(),
    }
    for name, slc in ACTION_SLICES.items():
        out[f"{name}_mae"] = float(np.mean(np.abs(diff[slc])))
    out["gripper_sign_match"] = bool((pred[11] >= 0.0) == (gt[11] >= 0.0))
    return out


def _chunk_metrics(pred_chunk: np.ndarray, gt_actions: np.ndarray) -> dict[str, Any]:
    n = min(len(pred_chunk), len(gt_actions))
    if n <= 0:
        return {}
    pred = np.asarray(pred_chunk[:n, :12], dtype=np.float32)
    gt = np.asarray(gt_actions[:n, :12], dtype=np.float32)
    diff = pred - gt
    return {
        "chunk_len": int(n),
        "chunk_mae": float(np.mean(np.abs(diff))),
        "chunk_eef_pos_mae": float(np.mean(np.abs(diff[:, ACTION_SLICES["eef_pos"]]))),
        "chunk_eef_rot_mae": float(np.mean(np.abs(diff[:, ACTION_SLICES["eef_rot"]]))),
        "chunk_gripper_mae": float(np.mean(np.abs(diff[:, ACTION_SLICES["gripper"]]))),
        "chunk_gripper_sign_acc": float(np.mean((pred[:, 11] >= 0.0) == (gt[:, 11] >= 0.0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("/scratch/yz11445/robocasa24/all24_human_camaware"))
    parser.add_argument("--episodes", default="646,647,655,692,699")
    parser.add_argument("--frames", default="0,25,50,75,100,125,150,175")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument(
        "--samples-per-frame",
        type=int,
        default=3,
        help="Repeat stochastic policy inference this many times per frame and extrinsics mode.",
    )
    parser.add_argument(
        "--extrinsics-modes",
        default="dataset,unflip",
        help="'dataset' sends LeRobot matrices as stored; 'unflip' pre-flips columns 1:3 before server transforms.",
    )
    parser.add_argument(
        "--image-modes",
        default="as_is",
        help="Comma-separated image transforms before sending to policy: as_is,vflip,hflip,rot180.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    tasks = _load_tasks(dataset_root)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    episodes = _parse_csv_ints(args.episodes)
    requested_frames = _parse_csv_ints(args.frames)
    extrinsics_modes = _parse_csv_strings(args.extrinsics_modes)
    image_modes = _parse_csv_strings(args.image_modes)

    records: list[dict[str, Any]] = []
    columns = (
        BASE_COLUMNS
        + list(CAMERA_COLUMNS.values())
        + [col for pair in MATRIX_COLUMNS.values() for col in pair]
    )
    for episode in episodes:
        parquet_path = _episode_parquet(dataset_root, episode)
        table = pq.read_table(parquet_path, columns=columns)
        rows = table.to_pylist()
        actions = np.asarray([row["action"] for row in rows], dtype=np.float32)
        frame_to_row = {int(row["frame_index"]): idx for idx, row in enumerate(rows)}
        for frame in requested_frames:
            if frame not in frame_to_row:
                continue
            row_idx = frame_to_row[frame]
            row = rows[row_idx]
            for mode in extrinsics_modes:
                for image_mode in image_modes:
                    for sample_idx in range(args.samples_per_frame):
                        result = policy.infer(
                            _policy_input(row, dataset_root, tasks, args.image_size, mode, image_mode)
                        )
                        pred_actions = np.asarray(result["actions"], dtype=np.float32)
                        pred_chunk = pred_actions[: args.chunk_size, :12]
                        gt_chunk = actions[row_idx : row_idx + args.chunk_size, :12]
                        metrics = _action_metrics(pred_chunk[0], np.asarray(row["action"], dtype=np.float32))
                        metrics.update(_chunk_metrics(pred_chunk, gt_chunk))
                        records.append(
                            {
                                "episode": int(episode),
                                "frame": int(frame),
                                "sample": int(sample_idx),
                                "task_index": int(row["task_index"]),
                                "prompt": tasks[int(row["task_index"])],
                                "extrinsics_mode": mode,
                                "image_mode": image_mode,
                                **metrics,
                            }
                        )

    summary: dict[str, Any] = {"num_records": len(records), "by_mode": {}}
    for mode in extrinsics_modes:
        for image_mode in image_modes:
            subset = [
                r for r in records
                if r["extrinsics_mode"] == mode and r["image_mode"] == image_mode
            ]
            if not subset:
                continue
            mode_key = f"{mode}/{image_mode}"
            summary["by_mode"][mode_key] = {
                "mae": float(np.mean([r["mae"] for r in subset])),
                "eef_pos_mae": float(np.mean([r["eef_pos_mae"] for r in subset])),
                "eef_rot_mae": float(np.mean([r["eef_rot_mae"] for r in subset])),
                "gripper_mae": float(np.mean([r["gripper_mae"] for r in subset])),
                "gripper_sign_acc": float(np.mean([r["gripper_sign_match"] for r in subset])),
                "chunk_mae": float(np.mean([r["chunk_mae"] for r in subset if "chunk_mae" in r])),
                "chunk_gripper_sign_acc": float(
                    np.mean([r["chunk_gripper_sign_acc"] for r in subset if "chunk_gripper_sign_acc" in r])
                ),
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
