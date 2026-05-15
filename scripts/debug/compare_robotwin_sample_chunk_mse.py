#!/usr/bin/env python3
"""Compare sampled action chunk MSE for two RoboTwin pi0 checkpoints."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import jax
import numpy as np
import torch

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _make_config(
    config_name: str,
    *,
    repo_id: str,
    asset_id: str,
    assets_dir: str,
    batch_size: int,
    num_workers: int,
    include_cam_extrinsics: bool,
) -> _config.TrainConfig:
    cfg = _config.get_config(config_name)
    data = dataclasses.replace(
        cfg.data,
        repo_id=repo_id,
        assets=_config.AssetsConfig(assets_dir=assets_dir, asset_id=asset_id),
        base_config=_config.DataConfig(prompt_from_task=True),
        include_cam_extrinsics=include_cam_extrinsics,
        pi3x_targets_root=None,
        gt_point_targets_root=None,
    )
    return dataclasses.replace(cfg, data=data, batch_size=batch_size, num_workers=num_workers)


def _to_device(tree, device: str):
    return jax.tree.map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, tree)


def _load_model(cfg: _config.TrainConfig, checkpoint_dir: Path, device: str):
    weight_path = checkpoint_dir / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing model weights: {weight_path}")
    model = cfg.model.load_pytorch(cfg, str(weight_path))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.to(device)
    model.eval()
    return model


def _batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, np.ndarray]:
    diff2 = (pred - target).float().pow(2)
    return {
        "all32_all50": diff2.mean(dim=(1, 2)).detach().cpu().numpy(),
        "first14_all50": diff2[:, :, :14].mean(dim=(1, 2)).detach().cpu().numpy(),
        "first14_t0": diff2[:, 0, :14].mean(dim=1).detach().cpu().numpy(),
        "first14_t0_4": diff2[:, :5, :14].mean(dim=(1, 2)).detach().cpu().numpy(),
        "first14_t0_9": diff2[:, :10, :14].mean(dim=(1, 2)).detach().cpu().numpy(),
        "joints_t0_9": diff2[:, :10, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]
        .mean(dim=(1, 2))
        .detach()
        .cpu()
        .numpy(),
        "gripper_t0_9": diff2[:, :10, [6, 13]].mean(dim=(1, 2)).detach().cpu().numpy(),
    }


def _summarize(values: dict[str, list[np.ndarray]]) -> dict[str, dict[str, float]]:
    out = {}
    for key, chunks in values.items():
        arr = np.concatenate(chunks, axis=0)
        out[key] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "std": float(arr.std()),
        }
    return out


@torch.inference_mode()
def _run_one(
    *,
    label: str,
    cfg: _config.TrainConfig,
    checkpoint_dir: Path,
    noises: list[torch.Tensor],
    device: str,
    num_steps: int,
    num_batches: int,
) -> tuple[dict[str, dict[str, float]], list[np.ndarray]]:
    loader = _data_loader.create_data_loader(
        cfg,
        shuffle=False,
        num_batches=num_batches,
        framework="pytorch",
    )
    model = _load_model(cfg, checkpoint_dir, device)
    values: dict[str, list[np.ndarray]] = {}
    targets_for_check: list[np.ndarray] = []

    for batch_idx, (obs, actions) in enumerate(loader):
        obs = _to_device(obs, device)
        actions = actions.to(device)
        noise = noises[batch_idx].to(device)
        pred = model.sample_actions(device, obs, noise=noise, num_steps=num_steps)
        metrics = _batch_metrics(pred, actions)
        for key, arr in metrics.items():
            values.setdefault(key, []).append(arr)
        targets_for_check.append(actions[:, :, :14].detach().cpu().numpy())
        print(f"[{label}] batch {batch_idx + 1}/{num_batches} first14_all50={metrics['first14_all50'].mean():.6g}", flush=True)

    del model
    torch.cuda.empty_cache()
    return _summarize(values), targets_for_check


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", required=True, type=Path)
    parser.add_argument("--cam-checkpoint", required=True, type=Path)
    parser.add_argument("--repo-id", default="robotwin/handover_block_demo_clean_camaware_50")
    parser.add_argument("--asset-id", default="robotwin/handover_block_demo_clean_camaware_50")
    parser.add_argument("--assets-dir", default="/scratch/yp2841/geometry-vla/pi0_libero")
    parser.add_argument("--baseline-config", default="pi0_robotwin_cam_baseline")
    parser.add_argument("--cam-config", default="pi0_robotwin_cam_prope_ray_view_distill_fullres_stage2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("log/robotwin_action_mse/sample_chunk_mse.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("OPENPI_DISABLE_TORCH_COMPILE", "1")

    baseline_cfg = _make_config(
        args.baseline_config,
        repo_id=args.repo_id,
        asset_id=args.asset_id,
        assets_dir=args.assets_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_cam_extrinsics=False,
    )
    cam_cfg = _make_config(
        args.cam_config,
        repo_id=args.repo_id,
        asset_id=args.asset_id,
        assets_dir=args.assets_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_cam_extrinsics=True,
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    shape = (args.batch_size, cam_cfg.model.action_horizon, cam_cfg.model.action_dim)
    noises = [torch.randn(shape, generator=generator) for _ in range(args.num_batches)]

    baseline_summary, baseline_targets = _run_one(
        label="baseline",
        cfg=baseline_cfg,
        checkpoint_dir=args.baseline_checkpoint,
        noises=noises,
        device=args.device,
        num_steps=args.num_steps,
        num_batches=args.num_batches,
    )
    cam_summary, cam_targets = _run_one(
        label="cam",
        cfg=cam_cfg,
        checkpoint_dir=args.cam_checkpoint,
        noises=noises,
        device=args.device,
        num_steps=args.num_steps,
        num_batches=args.num_batches,
    )

    target_max_abs_diff = max(
        float(np.max(np.abs(a - b))) for a, b in zip(baseline_targets, cam_targets, strict=True)
    )
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    result = {
        "settings": settings | {
            "baseline_checkpoint": str(args.baseline_checkpoint),
            "cam_checkpoint": str(args.cam_checkpoint),
            "target_max_abs_diff_first14": target_max_abs_diff,
            "num_samples": args.batch_size * args.num_batches,
        },
        "baseline": baseline_summary,
        "cam": cam_summary,
        "cam_minus_baseline_mean": {
            key: cam_summary[key]["mean"] - baseline_summary[key]["mean"]
            for key in baseline_summary
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
