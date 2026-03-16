from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _stack_batch(values: Sequence[Any]) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {key: _stack_batch([value[key] for value in values]) for key in first}
    return np.stack([np.asarray(value) for value in values], axis=0)


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        if noise is None:
            noises = None
        else:
            noise = np.asarray(noise)
            if noise.ndim == 3:
                if noise.shape[0] != 1:
                    raise ValueError(f"Single-observation infer expected noise batch of size 1, got {noise.shape[0]}")
                noise = noise[0]
            noises = [noise]
        return self.infer_batch([obs], noises=noises)[0]

    def infer_batch(
        self,
        obs_batch: Sequence[dict],
        *,
        noises: Sequence[np.ndarray | None] | None = None,
    ) -> list[dict]:
        if not obs_batch:
            return []

        total_start = time.monotonic()

        copy_start = time.monotonic()
        # Make a copy since transformations may modify the inputs in place.
        inputs_batch = [jax.tree.map(lambda x: x, obs) for obs in obs_batch]
        copy_time = time.monotonic() - copy_start

        input_transform_start = time.monotonic()
        inputs_batch = [self._input_transform(inputs) for inputs in inputs_batch]
        input_transform_time = time.monotonic() - input_transform_start

        batch_convert_start = time.monotonic()
        stacked_inputs = _stack_batch(inputs_batch)
        if not self._is_pytorch_model:
            inputs = jax.tree.map(jnp.asarray, stacked_inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device.
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x)).to(self._pytorch_device), stacked_inputs)
            sample_rng_or_pytorch_device = self._pytorch_device
        batch_convert_time = time.monotonic() - batch_convert_start

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noises is not None and any(noise is not None for noise in noises):
            if len(noises) != len(obs_batch):
                raise ValueError(f"Expected {len(obs_batch)} noises, got {len(noises)}")
            if any(noise is None for noise in noises):
                raise ValueError("Either provide noise for every observation or for none of them.")
            noise_batch = np.stack([np.asarray(noise) for noise in noises if noise is not None], axis=0)
            if self._is_pytorch_model:
                sample_kwargs["noise"] = torch.from_numpy(noise_batch).to(self._pytorch_device)
            else:
                sample_kwargs["noise"] = jnp.asarray(noise_batch)

        observation_start = time.monotonic()
        observation = _model.Observation.from_dict(inputs)
        observation_time = time.monotonic() - observation_start

        model_start = time.monotonic()
        if self._is_pytorch_model:
            with torch.inference_mode():
                actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        else:
            actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        model_time = time.monotonic() - model_start

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        postprocess_start = time.monotonic()
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x.detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(np.asarray, outputs)
        postprocess_time = time.monotonic() - postprocess_start
        total_time = time.monotonic() - total_start

        logging.info(
            "Policy.infer_batch timings batch_size=%d copy=%.2fms input_transform=%.2fms batch_convert=%.2fms observation=%.2fms model=%.2fms postprocess=%.2fms total=%.2fms",
            len(obs_batch),
            copy_time * 1000,
            input_transform_time * 1000,
            batch_convert_time * 1000,
            observation_time * 1000,
            model_time * 1000,
            postprocess_time * 1000,
            total_time * 1000,
        )

        results = []
        for batch_idx in range(len(obs_batch)):
            result = jax.tree.map(lambda x: np.asarray(x[batch_idx, ...]), outputs)
            result = self._output_transform(result)
            result["policy_timing"] = {
                "infer_ms": model_time * 1000,
                "total_ms": total_time * 1000,
                "batch_size": len(obs_batch),
            }
            results.append(result)
        return results

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
