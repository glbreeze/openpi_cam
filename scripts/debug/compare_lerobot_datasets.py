"""Compare two LeRobot-format datasets for structural and value consistency.

Example:
    uv run scripts/compare_lerobot_datasets.py \\
        /home/asus/Research/datasets/huggingface/lerobot/glbreeze/libero \\
        /home/asus/Research/datasets/huggingface/lerobot/glbreeze/libero_cam \\
        --sample-buckets 40
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import tyro


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def _ep_path(root: pathlib.Path, idx: int) -> pathlib.Path:
    return root / f"data/chunk-{idx // 1000:03d}/episode_{idx:06d}.parquet"


def _to_mat(x) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype == object:
        return np.array([np.asarray(r, dtype=np.float32) for r in a], dtype=np.float32)
    return a.astype(np.float32)


def compare(root_a: pathlib.Path, root_b: pathlib.Path, sample_buckets: int = 40, seed: int = 0) -> None:
    info_a = json.load(open(root_a / "meta/info.json"))
    info_b = json.load(open(root_b / "meta/info.json"))

    print(f"== A: {root_a}")
    print(f"== B: {root_b}\n")

    print("[info.json]")
    print(f"  A: {info_a['total_episodes']} eps, {info_a['total_frames']} frames")
    print(f"  B: {info_b['total_episodes']} eps, {info_b['total_frames']} frames")
    a_feats, b_feats = set(info_a["features"]), set(info_b["features"])
    print(f"  features only in A: {sorted(a_feats - b_feats)}")
    print(f"  features only in B: {sorted(b_feats - a_feats)}\n")

    a_eps = _load_jsonl(root_a / "meta/episodes.jsonl")
    b_eps = _load_jsonl(root_b / "meta/episodes.jsonl")
    a_tasks = _load_jsonl(root_a / "meta/tasks.jsonl")
    b_tasks = _load_jsonl(root_b / "meta/tasks.jsonl")

    print("[meta]")
    print(f"  A tasks: {len(a_tasks)}  B tasks: {len(b_tasks)}  same list: {a_tasks == b_tasks}")

    def _bucket_by_task(eps):
        d = defaultdict(list)
        for e in eps:
            d[e["tasks"][0]].append((e["episode_index"], e["length"]))
        return d

    a_by_task = _bucket_by_task(a_eps)
    b_by_task = _bucket_by_task(b_eps)
    print("\n[per-task differences]")
    any_diff = False
    for t in sorted(set(a_by_task) | set(b_by_task)):
        na, nb = len(a_by_task.get(t, [])), len(b_by_task.get(t, []))
        la = sum(x[1] for x in a_by_task.get(t, []))
        lb = sum(x[1] for x in b_by_task.get(t, []))
        if na != nb or la != lb:
            any_diff = True
            print(f"  {t[:60]!r:62s}  eps {na}->{nb}  frames {la}->{lb}")
    if not any_diff:
        print("  (none)")

    a_sig = Counter((e["tasks"][0], e["length"]) for e in a_eps)
    b_sig = Counter((e["tasks"][0], e["length"]) for e in b_eps)
    common = sum((a_sig & b_sig).values())
    print(f"\n  episodes with matching (task, length): {common} / min({len(a_eps)}, {len(b_eps)})")

    print("\n[value check on (task, length)-matched episodes]")

    def _bucket_idx(eps):
        d = defaultdict(list)
        for e in eps:
            d[(e["tasks"][0], e["length"])].append(e["episode_index"])
        return d

    A, B = _bucket_idx(a_eps), _bucket_idx(b_eps)
    shared_keys = list(set(A) & set(B))
    rng = np.random.default_rng(seed)
    rng.shuffle(shared_keys)

    match = mismatch = no_state_match = 0
    img_match = img_diff = 0
    checked = 0
    common_cols: set[str] | None = None

    for key in shared_keys[:sample_buckets]:
        a_idxs, b_idxs = A[key], B[key]
        a_dfs = {i: pd.read_parquet(_ep_path(root_a, i)) for i in a_idxs}
        for bi in b_idxs:
            bdf = pd.read_parquet(_ep_path(root_b, bi))
            if common_cols is None:
                common_cols = set(bdf.columns) & set(next(iter(a_dfs.values())).columns)
            sig = bdf["state"].iloc[0].tobytes() + bdf["state"].iloc[-1].tobytes()
            found = None
            for ai, adf in a_dfs.items():
                if adf["state"].iloc[0].tobytes() + adf["state"].iloc[-1].tobytes() == sig:
                    found = ai
                    break
            checked += 1
            if found is None:
                no_state_match += 1
                continue
            adf = a_dfs[found]
            s_ok = np.allclose(np.stack(adf["state"]), np.stack(bdf["state"]))
            ac_ok = np.allclose(np.stack(adf["actions"]), np.stack(bdf["actions"]))
            if s_ok and ac_ok:
                match += 1
            else:
                mismatch += 1
                print(f"  MISMATCH len={key[1]} a_ep={found} b_ep={bi} state_ok={s_ok} action_ok={ac_ok}")
            ia, ib = adf["image"].iloc[0], bdf["image"].iloc[0]
            im_ok = bytes(ia) == bytes(ib) if isinstance(ia, (bytes, bytearray)) else np.array_equal(ia, ib)
            if im_ok:
                img_match += 1
            else:
                img_diff += 1

    print(f"  sampled={checked} (from {min(sample_buckets, len(shared_keys))} shared buckets)")
    print(f"  state+action equal: {match}  mismatched: {mismatch}  no state match: {no_state_match}")
    print(f"  frame-0 image: equal={img_match}  differ={img_diff}")

    extras_b = sorted(set(info_b["features"]) - set(info_a["features"]))
    if extras_b:
        print(f"\n[B-only feature sanity] inspecting ep 0 of B for: {extras_b}")
        ep0 = pd.read_parquet(_ep_path(root_b, b_eps[0]["episode_index"]))
        for col in extras_b:
            if col not in ep0.columns:
                print(f"  {col}: declared in info.json but missing from parquet")
                continue
            try:
                arr = np.stack([_to_mat(x) for x in ep0[col]])
                print(
                    f"  {col}: shape={arr.shape} dtype={arr.dtype} "
                    f"constant_within_ep={np.allclose(arr, arr[0])}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  {col}: could not parse ({exc})")


if __name__ == "__main__":
    tyro.cli(compare)
