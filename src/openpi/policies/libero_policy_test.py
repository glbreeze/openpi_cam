import numpy as np

from openpi.policies import libero_policy


def _write_target(root, episode_index, value):
    for cam in ("agent", "wrist"):
        cam_dir = root / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            cam_dir / f"episode_{episode_index:06d}.npz",
            xy=np.full((2, 4, 4, 2), value, dtype=np.float32),
            log_z=np.full((2, 4, 4, 1), value + 1, dtype=np.float32),
            conf=np.full((2, 4, 4, 1), value + 2, dtype=np.float32),
        )


def test_mixed_point_target_loader_deterministic(tmp_path):
    pi3x_root = tmp_path / "pi3x"
    gt_root = tmp_path / "gt"
    _write_target(pi3x_root, episode_index=3, value=0.0)
    _write_target(gt_root, episode_index=3, value=10.0)

    loader = libero_policy.MixedPointTargetLoader(
        pi3x_root=str(pi3x_root),
        gt_root=str(gt_root),
        gt_ratio=0.5,
        seed=123,
    )
    data = {"episode_index": np.asarray(3), "frame_index": np.asarray(1)}

    out_a = loader(dict(data))
    out_b = loader(dict(data))

    assert out_a["point_target_source"] == out_b["point_target_source"]
    assert out_a["point_target_xy"].shape == (2, 4, 4, 2)
    assert out_a["point_target_logz"].shape == (2, 4, 4, 1)
    assert out_a["point_target_conf"].shape == (2, 4, 4, 1)
    assert np.allclose(out_a["point_target_xy"], out_b["point_target_xy"])


def test_mixed_point_target_loader_ratio_extremes(tmp_path):
    pi3x_root = tmp_path / "pi3x"
    gt_root = tmp_path / "gt"
    _write_target(pi3x_root, episode_index=0, value=1.0)
    _write_target(gt_root, episode_index=0, value=9.0)
    data = {"episode_index": np.asarray(0), "frame_index": np.asarray(0)}

    pi3x_only = libero_policy.MixedPointTargetLoader(str(pi3x_root), str(gt_root), gt_ratio=0.0)
    gt_only = libero_policy.MixedPointTargetLoader(str(pi3x_root), str(gt_root), gt_ratio=1.0)

    pi3x_out = pi3x_only(dict(data))
    gt_out = gt_only(dict(data))

    assert float(pi3x_out["point_target_source"]) == 0.0
    assert float(gt_out["point_target_source"]) == 1.0
    assert np.allclose(pi3x_out["point_target_xy"], 1.0)
    assert np.allclose(gt_out["point_target_xy"], 9.0)
