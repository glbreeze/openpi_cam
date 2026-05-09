"""In-place 180-degree flip of images inside a converted RoboTwin LeRobot v2.1
dataset, to bring it onto the openpi cam-aware convention (LIBERO-style flipped
parquet image + parquet K_orig + RobotwinCamInputs applies fx->-fx at training).

This is a one-shot fix for datasets that were converted with the original
(buggy) convert_robotwin_cam_to_lerobot.py — newer conversions already include
the flip in `_resize_image`. Only the image columns are touched; K and
extrinsic columns are not modified.
"""

from __future__ import annotations
import argparse
import io
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


IMAGE_COLS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _flip_image_struct(value: dict) -> dict:
    """value is a {'bytes': ..., 'path': ...} struct (LeRobot image storage)."""
    raw = value.get("bytes")
    img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    flipped = np.ascontiguousarray(img[::-1, ::-1])
    buf = io.BytesIO()
    Image.fromarray(flipped).save(buf, format="PNG")
    return {"bytes": buf.getvalue(), "path": value.get("path")}


def flip_parquet(path: Path):
    table = pq.read_table(path)
    rows = table.to_pylist()
    n_flipped = 0
    for row in rows:
        for col in IMAGE_COLS:
            if col in row and isinstance(row[col], dict):
                row[col] = _flip_image_struct(row[col])
                n_flipped += 1
    new_table = pa.Table.from_pylist(rows, schema=table.schema)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(new_table, tmp)
    shutil.move(str(tmp), str(path))
    return n_flipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, help="LeRobot repo dir, e.g. .../robotwin/<task>_demo_clean_camaware_50")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    parquets = sorted(repo_root.rglob("episode_*.parquet"))
    if not parquets:
        raise SystemExit(f"No parquet files under {repo_root}")
    total = 0
    for p in parquets:
        n = flip_parquet(p)
        total += n
        print(f"  {p.name}: flipped {n} cells")
    print(f"DONE: {total} image cells flipped across {len(parquets)} parquet files in {repo_root}")


if __name__ == "__main__":
    main()
