import numpy as np

from openpi.models import model as _model
from openpi.policies import robotwin_policy
from openpi.training import config as _config


def test_robotwin_inputs_from_flat_lerobot_keys():
    transform = robotwin_policy.RobotwinInputs(model_type=_model.ModelType.PI0, adapt_to_pi=False)
    data = {
        "observation.images.head_camera": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "observation.images.left_camera": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "observation.images.right_camera": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "observation.state": np.arange(14, dtype=np.float32),
        "action": np.ones((50, 14), dtype=np.float32),
        "prompt": "click the bell",
    }

    result = transform(data)

    assert result["state"].shape == (14,)
    assert result["actions"].shape == (50, 14)
    assert result["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert result["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)
    assert result["image"]["right_wrist_0_rgb"].shape == (224, 224, 3)
    assert result["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.True_,
    }
    assert result["prompt"] == "click the bell"


def test_robotwin_inputs_from_official_robotwin_converted_keys():
    transform = robotwin_policy.RobotwinInputs(model_type=_model.ModelType.PI0, adapt_to_pi=False)
    data = {
        "observation.images.cam_high": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.cam_left_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.cam_right_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.state": np.arange(14, dtype=np.float32),
        "action": np.ones((50, 14), dtype=np.float32),
        "task": "beat block hammer",
    }

    result = transform(data)

    assert result["state"].shape == (14,)
    assert result["actions"].shape == (50, 14)
    assert set(result["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert result["prompt"] == "beat block hammer"


def test_robotwin_inputs_from_simple_inference_keys():
    transform = robotwin_policy.RobotwinInputs(model_type=_model.ModelType.PI0, adapt_to_pi=False)
    data = robotwin_policy.make_robotwin_example() | {"actions": np.zeros((50, 14), dtype=np.float32)}

    result = transform(data)

    assert result["state"].shape == (14,)
    assert result["actions"].shape == (50, 14)
    assert set(result["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}


def test_robotwin_outputs_return_14d_actions():
    transform = robotwin_policy.RobotwinOutputs(adapt_to_pi=False)
    data = {"actions": np.ones((50, 32), dtype=np.float32)}

    result = transform(data)

    assert result["actions"].shape == (50, 14)


def test_robotwin_configs_registered():
    smoke = _config.get_config("pi0_robotwin_smoke")
    full = _config.get_config("pi0_robotwin")

    assert smoke.data.repo_id == "lerobot/robotwin_unified"
    assert full.data.repo_id == "lerobot/robotwin_unified"
