from types import SimpleNamespace

import torch

from openpi.models_pytorch import preprocessing_pytorch as preprocessing


def _make_observation(*, base_shape: tuple[int, int], left_wrist_shape: tuple[int, int] = (224, 224)):
    base_h, base_w = base_shape
    wrist_h, wrist_w = left_wrist_shape
    eye4 = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    return SimpleNamespace(
        images={
            "base_0_rgb": torch.zeros((1, 3, base_h, base_w), dtype=torch.float32),
            "left_wrist_0_rgb": torch.zeros((1, 3, wrist_h, wrist_w), dtype=torch.float32),
            "right_wrist_0_rgb": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
        },
        image_masks={},
        state=torch.zeros((1, 32), dtype=torch.float32),
        agent_extrinsic=eye4,
        wrist_extrinsic=eye4.clone(),
        agent_intrinsic=torch.tensor(
            [[[100.0, 0.0, 50.0], [0.0, 100.0, 20.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ),
        wrist_intrinsic=torch.tensor(
            [[[120.0, 0.0, 80.0], [0.0, 110.0, 60.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ),
        tokenized_prompt=torch.zeros((1, 8), dtype=torch.int32),
        tokenized_prompt_mask=torch.ones((1, 8), dtype=torch.bool),
        token_ar_mask=None,
        token_loss_mask=None,
    )


def _rotation_forward_transform(height: int, width: int, angle_degrees: float) -> torch.Tensor:
    angle_rad = torch.tensor(angle_degrees * torch.pi / 180.0, dtype=torch.float32)
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    pixel_to_norm = torch.tensor(
        [
            [2.0 / width, 0.0, (1.0 / width) - 1.0],
            [0.0, 2.0 / height, (1.0 / height) - 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    norm_to_pixel = torch.tensor(
        [
            [width / 2.0, 0.0, (width / 2.0) - 0.5],
            [0.0, height / 2.0, (height / 2.0) - 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    rotation_inv = torch.tensor(
        [
            [cos_a, sin_a, 0.0],
            [-sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return norm_to_pixel @ rotation_inv @ pixel_to_norm


def test_preprocess_updates_intrinsics_for_resize_with_pad():
    observation = _make_observation(base_shape=(100, 200))

    processed = preprocessing.preprocess_observation_pytorch(observation, train=False)

    expected_transform = torch.tensor(
        [
            [224.0 / 200.0, 0.0, 0.0],
            [0.0, 112.0 / 100.0, 56.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    torch.testing.assert_close(
        processed.agent_intrinsic,
        expected_transform @ observation.agent_intrinsic,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(processed.agent_extrinsic, observation.agent_extrinsic)
    assert processed.images["base_0_rgb"].shape == (1, 3, 224, 224)


def test_preprocess_updates_intrinsics_for_crop_and_rotation(monkeypatch):
    observation = _make_observation(base_shape=(224, 224))

    randint_values = iter((torch.tensor([10]), torch.tensor([20])))
    rand_values = iter(
        (
            torch.tensor([0.75]),  # angle -> +2.5 degrees
            torch.tensor([0.5]),   # brightness -> 1.0
            torch.tensor([0.5]),   # contrast -> 1.0
            torch.tensor([0.5]),   # saturation -> 1.0
            torch.tensor([0.5]),   # wrist brightness -> 1.0
            torch.tensor([0.5]),   # wrist contrast -> 1.0
            torch.tensor([0.5]),   # wrist saturation -> 1.0
            torch.tensor([0.5]),   # right wrist brightness -> 1.0
            torch.tensor([0.5]),   # right wrist contrast -> 1.0
            torch.tensor([0.5]),   # right wrist saturation -> 1.0
        )
    )

    def fake_randint(low, high, size, device=None):
        del low, high, size
        return next(randint_values).to(device=device)

    def fake_rand(*size, device=None):
        del size
        return next(rand_values).to(device=device)

    monkeypatch.setattr(preprocessing.torch, "randint", fake_randint)
    monkeypatch.setattr(preprocessing.torch, "rand", fake_rand)

    processed = preprocessing.preprocess_observation_pytorch(observation, train=True)

    crop_transform = torch.tensor(
        [
            [224.0 / 212.0, 0.0, -(224.0 / 212.0) * 20.0],
            [0.0, 224.0 / 212.0, -(224.0 / 212.0) * 10.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    rotation_transform = _rotation_forward_transform(224, 224, 2.5).unsqueeze(0)
    expected_intrinsic = rotation_transform @ crop_transform @ observation.agent_intrinsic

    torch.testing.assert_close(processed.agent_intrinsic, expected_intrinsic, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(processed.wrist_intrinsic, observation.wrist_intrinsic)
    torch.testing.assert_close(processed.agent_extrinsic, observation.agent_extrinsic)
