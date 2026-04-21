import argparse
import io
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from scipy.spatial.transform import Rotation


DEFAULT_WORLD_ROOT = Path("/scratch/yp2841/geometry-vla/glbreeze/libero_object_cam_train_o00_02_03_05_06_09_10")
DEFAULT_AGENTCAM_ROOT = Path(
    "/scratch/yp2841/geometry-vla/glbreeze/libero_object_cam_train_o00_02_03_05_06_09_10_agentcam_action"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/camera_frame_alignment")
IMAGE_ROT_180 = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    leading_shape = rotvec.shape[:-1]
    return Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix().reshape(*leading_shape, 3, 3)


def _matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    leading_shape = matrix.shape[:-2]
    return Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_rotvec().reshape(*leading_shape, 3)


def _load_episode_table(dataset_root: Path, episode_index: int, columns: list[str]):
    chunk_index = episode_index // 1000
    parquet_path = dataset_root / "data" / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet episode file: {parquet_path}")
    return pq.read_table(parquet_path, columns=columns)


def _row_to_example(table, row_index: int) -> dict:
    return {key: values[0] for key, values in table.slice(row_index, 1).to_pydict().items()}


def _decode_image(image_struct: dict) -> Image.Image:
    return Image.open(io.BytesIO(image_struct["bytes"])).convert("RGB")


def _actions_world_to_agent_camera(actions: np.ndarray, agent_extrinsic: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    agent_extrinsic = np.asarray(agent_extrinsic, dtype=np.float32)
    rotation_wc = agent_extrinsic[:3, :3]
    rotation_cw = rotation_wc.T

    transformed = actions.copy()
    transformed[:3] = rotation_cw @ actions[:3]
    rotation_delta_w = _rotvec_to_matrix(actions[3:6])
    rotation_delta_c = rotation_cw @ rotation_delta_w @ rotation_wc
    transformed[3:6] = _matrix_to_rotvec(rotation_delta_c)
    return transformed.astype(np.float32)


def _state_world_to_agent_camera(state: np.ndarray, agent_extrinsic: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    agent_extrinsic = np.asarray(agent_extrinsic, dtype=np.float32)
    rotation_wc = agent_extrinsic[:3, :3]
    rotation_cw = rotation_wc.T
    translation_wc = agent_extrinsic[:3, 3]

    transformed = state.copy()
    transformed[:3] = rotation_cw @ (state[:3] - translation_wc)
    rotation_we = _rotvec_to_matrix(state[3:6])
    rotation_ce = rotation_cw @ rotation_we
    transformed[3:6] = _matrix_to_rotvec(rotation_ce)
    return transformed.astype(np.float32)


def _apply_image_rotation_to_camera_action(action_cam: np.ndarray) -> np.ndarray:
    action_cam = np.asarray(action_cam, dtype=np.float32)
    transformed = action_cam.copy()
    transformed[:3] = IMAGE_ROT_180 @ action_cam[:3]
    delta_c = _rotvec_to_matrix(action_cam[3:6])
    delta_img = IMAGE_ROT_180 @ delta_c @ IMAGE_ROT_180.T
    transformed[3:6] = _matrix_to_rotvec(delta_img)
    return transformed.astype(np.float32)


def _apply_image_rotation_to_camera_state(state_cam: np.ndarray) -> np.ndarray:
    state_cam = np.asarray(state_cam, dtype=np.float32)
    transformed = state_cam.copy()
    transformed[:3] = IMAGE_ROT_180 @ state_cam[:3]
    rot_c = _rotvec_to_matrix(state_cam[3:6])
    rot_img = IMAGE_ROT_180 @ rot_c
    transformed[3:6] = _matrix_to_rotvec(rot_img)
    return transformed.astype(np.float32)


def _round_list(x: np.ndarray, digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in np.asarray(x).tolist()]


def _draw_arrow(draw: ImageDraw.ImageDraw, origin: tuple[float, float], vec_xy: np.ndarray, color: str, label: str) -> None:
    ox, oy = origin
    dx, dy = float(vec_xy[0]), float(vec_xy[1])
    end = (ox + dx, oy + dy)
    draw.line((ox, oy, end[0], end[1]), fill=color, width=5)

    angle = math.atan2(dy, dx)
    head_len = 12.0
    left = (
        end[0] - head_len * math.cos(angle - math.pi / 6),
        end[1] - head_len * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - head_len * math.cos(angle + math.pi / 6),
        end[1] - head_len * math.sin(angle + math.pi / 6),
    )
    draw.polygon([end, left, right], fill=color)
    draw.text((end[0] + 8, end[1] - 10), label, fill=color)


def _build_visualization(
    image: Image.Image,
    title: str,
    current_action_cam: np.ndarray,
    corrected_action_cam: np.ndarray,
    summary: dict,
) -> Image.Image:
    font = ImageFont.load_default()
    panel = image.copy()
    draw = ImageDraw.Draw(panel)
    w, h = panel.size
    origin = (w / 2, h / 2)

    # Use image-plane convention: +x right, +y up.
    current_vec = np.array([current_action_cam[0], -current_action_cam[1]], dtype=np.float32)
    corrected_vec = np.array([corrected_action_cam[0], -corrected_action_cam[1]], dtype=np.float32)

    max_norm = max(np.linalg.norm(current_vec), np.linalg.norm(corrected_vec), 1e-6)
    scale = min(w, h) * 0.28 / max_norm

    draw.line((origin[0] - 35, origin[1], origin[0] + 35, origin[1]), fill="gray", width=2)
    draw.line((origin[0], origin[1] - 35, origin[0], origin[1] + 35), fill="gray", width=2)
    draw.ellipse((origin[0] - 4, origin[1] - 4, origin[0] + 4, origin[1] + 4), fill="white", outline="black")
    draw.text((origin[0] + 40, origin[1] + 5), "+u", fill="gray")
    draw.text((origin[0] + 5, origin[1] - 50), "+v(up)", fill="gray")

    _draw_arrow(draw, origin, current_vec * scale, "#d7263d", "current label")
    _draw_arrow(draw, origin, corrected_vec * scale, "#1b9e77", "image-aligned")

    canvas_w = w + 640
    canvas_h = max(h, 420)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(panel, (20, 80))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 20), title, fill="black", font=font)

    lines = [
        "What this checks:",
        "Stored image is already rotated 180 deg, but raw camera extrinsic was kept.",
        "Red = current camera-frame translation label interpreted on stored image.",
        "Green = label after compensating that fixed 180 deg image-plane rotation.",
        "",
        f"world action xyz: {summary['world_action_xyz']}",
        f"current cam xyz: {summary['current_camera_action_xyz']}",
        f"stored agentcam xyz: {summary.get('stored_agentcam_action_xyz')}",
        f"image-aligned cam xyz: {summary['corrected_camera_action_xyz']}",
        "",
        f"world rotvec: {summary['world_action_rotvec']}",
        f"current cam rotvec: {summary['current_camera_action_rotvec']}",
        f"stored agentcam rotvec: {summary.get('stored_agentcam_action_rotvec')}",
        f"image-aligned cam rotvec: {summary['corrected_camera_action_rotvec']}",
        "",
        f"current vs stored max abs diff: {summary.get('current_vs_stored_max_abs_diff')}",
        f"x/y sign flip after correction: {summary['xy_sign_flip']}",
        f"chosen score (||cam xy||): {summary['selection_score']}",
    ]

    text_x = w + 40
    text_y = 80
    line_h = 20
    for i, line in enumerate(lines):
        draw.text((text_x, text_y + i * line_h), line, fill="black", font=font)
    return canvas


def _choose_frame(world_table) -> tuple[int, float]:
    data = world_table.to_pydict()
    best_index = 0
    best_score = -1.0
    for i, (action, extrinsic) in enumerate(zip(data["actions"], data["agent_extrinsic"], strict=True)):
        action_cam = _actions_world_to_agent_camera(np.asarray(action, dtype=np.float32), np.asarray(extrinsic, dtype=np.float32))
        score = float(np.linalg.norm(action_cam[:2]))
        if score > best_score:
            best_index = i
            best_score = score
    return best_index, best_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-dataset-root", type=Path, default=DEFAULT_WORLD_ROOT)
    parser.add_argument("--agentcam-dataset-root", type=Path, default=DEFAULT_AGENTCAM_ROOT)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=-1, help="-1 means auto-pick the strongest xy action in episode")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    columns = ["image", "actions", "state", "agent_extrinsic", "frame_index", "episode_index", "task_index"]
    world_table = _load_episode_table(args.world_dataset_root, args.episode_index, columns)
    agentcam_table = _load_episode_table(args.agentcam_dataset_root, args.episode_index, columns)

    if args.frame_index < 0:
        frame_index, selection_score = _choose_frame(world_table)
    else:
        frame_index = args.frame_index
        row = _row_to_example(world_table, frame_index)
        selection_score = float(
            np.linalg.norm(
                _actions_world_to_agent_camera(
                    np.asarray(row["actions"], dtype=np.float32),
                    np.asarray(row["agent_extrinsic"], dtype=np.float32),
                )[:2]
            )
        )

    world_row = _row_to_example(world_table, frame_index)
    agentcam_row = _row_to_example(agentcam_table, frame_index)

    image = _decode_image(world_row["image"])
    world_action = np.asarray(world_row["actions"], dtype=np.float32)
    world_state = np.asarray(world_row["state"], dtype=np.float32)
    agent_extrinsic = np.asarray(world_row["agent_extrinsic"], dtype=np.float32)
    stored_agentcam_action = np.asarray(agentcam_row["actions"], dtype=np.float32)

    current_action_cam = _actions_world_to_agent_camera(world_action, agent_extrinsic)
    current_state_cam = _state_world_to_agent_camera(world_state, agent_extrinsic)
    corrected_action_cam = _apply_image_rotation_to_camera_action(current_action_cam)
    corrected_state_cam = _apply_image_rotation_to_camera_state(current_state_cam)

    summary = {
        "episode_index": int(world_row["episode_index"]),
        "task_index": int(world_row["task_index"]),
        "frame_index": int(world_row["frame_index"]),
        "selection_score": round(selection_score, 6),
        "world_action_xyz": _round_list(world_action[:3]),
        "world_action_rotvec": _round_list(world_action[3:6]),
        "current_camera_action_xyz": _round_list(current_action_cam[:3]),
        "current_camera_action_rotvec": _round_list(current_action_cam[3:6]),
        "stored_agentcam_action_xyz": _round_list(stored_agentcam_action[:3]),
        "stored_agentcam_action_rotvec": _round_list(stored_agentcam_action[3:6]),
        "corrected_camera_action_xyz": _round_list(corrected_action_cam[:3]),
        "corrected_camera_action_rotvec": _round_list(corrected_action_cam[3:6]),
        "current_camera_state_xyz": _round_list(current_state_cam[:3]),
        "corrected_camera_state_xyz": _round_list(corrected_state_cam[:3]),
        "current_vs_stored_max_abs_diff": round(float(np.max(np.abs(current_action_cam - stored_agentcam_action))), 8),
        "xy_sign_flip": {
            "x": bool(np.sign(current_action_cam[0]) == -np.sign(corrected_action_cam[0]) or abs(current_action_cam[0]) < 1e-9),
            "y": bool(np.sign(current_action_cam[1]) == -np.sign(corrected_action_cam[1]) or abs(current_action_cam[1]) < 1e-9),
        },
    }

    title = (
        f"Camera-frame sanity check | episode {summary['episode_index']} | "
        f"frame {summary['frame_index']} | task {summary['task_index']}"
    )
    viz = _build_visualization(image, title, current_action_cam, corrected_action_cam, summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{summary['episode_index']:06d}_frame_{summary['frame_index']:04d}"
    png_path = args.output_dir / f"{stem}.png"
    json_path = args.output_dir / f"{stem}.json"
    viz.save(png_path)
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps({"png": str(png_path), "json": str(json_path), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
