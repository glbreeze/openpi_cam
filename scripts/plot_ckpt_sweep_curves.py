#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


PREFERRED_SUMMARY_RE = re.compile(r"^(?:aggregate|summary)_(\d+)(?:_.*)?\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot checkpoint sweep success-rate curves from JSON summaries.")
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        help="Series spec in the form name=/path/to/summary_root_or_json_dir",
    )
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--csv-out", default="", help="Optional CSV output path")
    parser.add_argument("--title", default="Checkpoint Sweep Curve")
    parser.add_argument("--ylabel", default="Success Rate")
    parser.add_argument("--min-step", type=int, default=None, help="Optional minimum checkpoint step to include")
    parser.add_argument("--max-step", type=int, default=None, help="Optional maximum checkpoint step to include")
    return parser.parse_args()


def discover_points(root: Path) -> list[tuple[int, float, Path]]:
    candidates = []
    if root.is_file() and root.suffix == ".json":
        candidates = [root]
    elif root.is_dir():
        all_candidates = sorted(root.rglob("*.json"))
        preferred = [path for path in all_candidates if PREFERRED_SUMMARY_RE.match(path.name)]
        candidates = preferred or all_candidates

    points: list[tuple[int, float, Path]] = []
    for path in candidates:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue

        if "total_success_rate" not in data:
            continue

        step = infer_step(path)
        if step is None:
            continue

        points.append((step, float(data["total_success_rate"]), path))

    dedup: dict[int, tuple[float, Path]] = {}
    for step, rate, path in sorted(points, key=lambda item: (item[0], str(item[2]))):
        dedup[step] = (rate, path)

    return [(step, rate, path) for step, (rate, path) in sorted(dedup.items())]


def infer_step(path: Path) -> int | None:
    match = PREFERRED_SUMMARY_RE.match(path.name)
    if match:
        return int(match.group(1))

    for part in reversed(path.parts):
        if part.isdigit():
            return int(part)

    match = re.search(r"_(\d+)\.json$", path.name)
    if match:
        return int(match.group(1))

    return None


def main() -> None:
    args = parse_args()

    series_points: list[tuple[str, list[tuple[int, float, Path]]]] = []
    for item in args.series:
        if "=" not in item:
            raise SystemExit(f"Invalid --series spec: {item}")
        name, raw_path = item.split("=", 1)
        root = Path(raw_path)
        points = discover_points(root)
        if args.min_step is not None:
            points = [item for item in points if item[0] >= args.min_step]
        if args.max_step is not None:
            points = [item for item in points if item[0] <= args.max_step]
        if not points:
            raise SystemExit(f"No summary JSONs with total_success_rate found under {root}")
        series_points.append((name, points))

    plt.figure(figsize=(7.5, 4.5))
    for name, points in series_points:
        xs = [step for step, _, _ in points]
        ys = [rate * 100.0 for _, rate, _ in points]
        plt.plot(xs, ys, marker="o", linewidth=2, label=name)

    plt.xlabel("Checkpoint Step")
    plt.ylabel(args.ylabel)
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)

    if args.csv_out:
        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["series", "step", "success_rate", "source_json"])
            for name, points in series_points:
                for step, rate, source in points:
                    writer.writerow([name, step, rate, str(source)])


if __name__ == "__main__":
    main()
