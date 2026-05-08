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


def test_dual_point_target_loader_keeps_targets_separate(tmp_path):
    pi3x_root = tmp_path / "pi3x"
    gt_root = tmp_path / "gt"
    _write_target(pi3x_root, episode_index=2, value=3.0)
    _write_target(gt_root, episode_index=2, value=11.0)
    data = {"episode_index": np.asarray(2), "frame_index": np.asarray(0)}

    loader = libero_policy.DualPointTargetLoader(str(pi3x_root), str(gt_root), gt_weight=0.6)
    out = loader(dict(data))

    assert np.isclose(float(out["point_target_source"]), 0.6)
    assert np.allclose(out["point_target_xy"], 11.0)
    assert np.allclose(out["point_target_logz"], 12.0)
    assert np.allclose(out["point_target_conf"], 13.0)
    assert np.allclose(out["pi3x_target_xy"], 3.0)
    assert np.allclose(out["pi3x_target_logz"], 4.0)
    assert np.allclose(out["pi3x_target_conf"], 5.0)


def _write_corner_marked_target(root, episode_index):
    """Write a Pi3X-style cache where (0, 0) and (H-1, W-1) carry distinct sentinels.

    Per cam, frame 0 has a 4x4 xy grid with xy[0, 0, 0] = 100 and xy[3, 3, 0] = 200.
    """
    for cam in ("agent", "wrist"):
        cam_dir = root / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        xy = np.zeros((1, 4, 4, 2), dtype=np.float32)
        xy[0, 0, 0, 0] = 100.0
        xy[0, 3, 3, 0] = 200.0
        log_z = np.zeros((1, 4, 4, 1), dtype=np.float32)
        log_z[0, 0, 0, 0] = 1.0
        log_z[0, 3, 3, 0] = 2.0
        conf = np.full((1, 4, 4, 1), -10.0, dtype=np.float32)
        conf[0, 0, 0, 0] = 5.0
        np.savez(cam_dir / f"episode_{episode_index:06d}.npz", xy=xy, log_z=log_z, conf=conf)


def test_pi3x_legacy_flip_reverses_loaded_arrays(tmp_path):
    pi3x_root = tmp_path / "pi3x"
    _write_corner_marked_target(pi3x_root, episode_index=0)
    data = {"episode_index": np.asarray(0), "frame_index": np.asarray(0)}

    no_flip = libero_policy.Pi3xLiberoTargetLoader(str(pi3x_root))(dict(data))
    flipped = libero_policy.Pi3xLiberoTargetLoader(str(pi3x_root), pi3x_legacy_flip=True)(dict(data))

    # Without the flag the loader is a passthrough.
    assert no_flip["pi3x_target_xy"][0, 0, 0, 0] == 100.0
    assert no_flip["pi3x_target_xy"][0, 3, 3, 0] == 200.0
    # With the flag the (0, 0) sentinel moves to (H-1, W-1) and vice versa.
    assert flipped["pi3x_target_xy"][0, 0, 0, 0] == 200.0
    assert flipped["pi3x_target_xy"][0, 3, 3, 0] == 100.0
    # log_z and conf flip together with xy.
    assert flipped["pi3x_target_logz"][0, 0, 0, 0] == 2.0
    assert flipped["pi3x_target_logz"][0, 3, 3, 0] == 1.0
    assert flipped["pi3x_target_conf"][0, 3, 3, 0] == 5.0


def test_dual_loader_legacy_flip_only_flips_pi3x(tmp_path):
    pi3x_root = tmp_path / "pi3x"
    gt_root = tmp_path / "gt"
    _write_corner_marked_target(pi3x_root, episode_index=0)
    _write_corner_marked_target(gt_root, episode_index=0)
    data = {"episode_index": np.asarray(0), "frame_index": np.asarray(0)}

    out = libero_policy.DualPointTargetLoader(
        pi3x_root=str(pi3x_root),
        gt_root=str(gt_root),
        gt_weight=0.5,
        pi3x_legacy_flip=True,
    )(dict(data))

    # Pi3X branch is flipped — (0, 0) carries the (H-1, W-1) sentinel of the cache.
    assert out["pi3x_target_xy"][0, 0, 0, 0] == 200.0
    # GT branch is left alone — (0, 0) keeps the cached (0, 0) sentinel.
    assert out["point_target_xy"][0, 0, 0, 0] == 100.0


def test_mixed_loader_legacy_flip_only_when_source_is_pi3x(tmp_path):
    pi3x_root = tmp_path / "pi3x"
    gt_root = tmp_path / "gt"
    _write_corner_marked_target(pi3x_root, episode_index=0)
    _write_corner_marked_target(gt_root, episode_index=0)
    data = {"episode_index": np.asarray(0), "frame_index": np.asarray(0)}

    pi3x_only = libero_policy.MixedPointTargetLoader(
        str(pi3x_root), str(gt_root), gt_ratio=0.0, pi3x_legacy_flip=True
    )
    gt_only = libero_policy.MixedPointTargetLoader(
        str(pi3x_root), str(gt_root), gt_ratio=1.0, pi3x_legacy_flip=True
    )
    # Pi3X-source sample: corner sentinels flipped.
    out_pi3x = pi3x_only(dict(data))
    assert float(out_pi3x["point_target_source"]) == 0.0
    assert out_pi3x["point_target_xy"][0, 0, 0, 0] == 200.0
    # GT-source sample: corner sentinels untouched even though the flag is on.
    out_gt = gt_only(dict(data))
    assert float(out_gt["point_target_source"]) == 1.0
    assert out_gt["point_target_xy"][0, 0, 0, 0] == 100.0
