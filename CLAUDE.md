# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

openpi is Physical Intelligence's open-source VLA (vision-language-action) robotics stack. It ships three model families — π₀ (flow-matching), π₀-FAST (autoregressive with FAST tokenizer), and π₀.₅ — with both a primary JAX/Flax implementation and a secondary PyTorch implementation. The two backends share the same configs, data pipeline, policy wrappers, and serving code; only the model internals differ.

## Environment & common commands

Dependencies are managed with `uv` (Python ≥3.11). Install with `GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .`.

- Lint / format: `uv run ruff check .` and `uv run ruff format .` (line length 120, configured in `pyproject.toml`)
- Tests: `uv run pytest` (testpaths: `src`, `scripts`, `packages`; marker `manual` is opt-in). Single test: `uv run pytest src/openpi/models/pi0_test.py::TestName -k expr`.
- Compute norm stats (required before training a config): `uv run scripts/compute_norm_stats.py --config-name <name>`
- JAX training: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name <run> [--overwrite] [--fsdp-devices N]`
- PyTorch training: `uv run scripts/train_pytorch.py <config_name> --exp_name <run>` (single GPU) or `uv run torchrun --standalone --nnodes=1 --nproc_per_node=<N> scripts/train_pytorch.py <config_name> --exp_name <run>` (DDP). Add `--resume` to continue from the latest checkpoint.
- Serve a policy: `uv run scripts/serve_policy.py policy:checkpoint --policy.config=<name> --policy.dir=<checkpoint_dir>` (websocket server on port 8000).
- Convert JAX → PyTorch checkpoint: `uv run examples/convert_jax_model_to_pytorch.py --config_name <name> --checkpoint_dir <jax> --output_path <pt>`.

Checkpoints auto-download from `gs://openpi-assets` into `~/.cache/openpi` (override via `OPENPI_DATA_HOME`).

## Architecture

The code under `src/openpi/` is organized by concern, not by model:

- `models/` — JAX/Flax model code. `model.py` defines the shared `BaseModel` / `Observation` / `Actions` interfaces; `pi0.py`, `pi0_fast.py` are the model heads; `gemma.py` + `siglip.py` are the LLM/vision backbones; `pi0_config.py` and `cross_view_config.py` hold model-shape configs; `lora.py` implements LoRA adapters; `tokenizer.py` wraps the action/text tokenizers including FAST.
- `models_pytorch/` — PyTorch mirrors of the above (`pi0_pytorch.py`, `gemma_pytorch.py`, `preprocessing_pytorch.py`). Requires applying patches from `models_pytorch/transformers_replace/` on top of the installed `transformers==4.53.2` (see README "PyTorch Support"). PyTorch lacks FSDP, LoRA, EMA, mixed precision, and π₀-FAST.
- `policies/` — input/output transforms that map a specific robot/dataset (`aloha_policy.py`, `droid_policy.py`, `libero_policy.py`) to the model's observation/action format. `policy.py` wraps a trained model for inference; `policy_config.create_trained_policy` is the main entry point and autodetects JAX vs. PyTorch checkpoints.
- `training/` — `config.py` is the central registry: it defines `TrainConfig`, `DataConfig`, `AssetsConfig`, and the `_CONFIGS` list of named recipes (e.g. `pi05_libero`, `pi0_aloha_sim`, `debug`). `data_loader.py` builds LeRobot-backed loaders; `weight_loaders.py` loads base-model weights (including `pytorch_weight_path` for PyTorch); `optimizer.py`, `sharding.py` (FSDP mesh), `checkpoints.py`, and `droid_rlds_dataset.py` are used by both trainers. `misc/` holds environment-specific overrides.
- `shared/` — `download.py` (GCS checkpoint fetch), `normalize.py` (norm-stats math, must match `scripts/compute_norm_stats.py`), `image_tools.py`, `array_typing.py`.
- `serving/websocket_policy_server.py` — the remote-inference server used by `scripts/serve_policy.py`; paired with `packages/openpi-client` (a thin client workspace package installable on robots without the full training stack).
- `transforms.py` — generic data transforms composed by each policy config.

`scripts/train.py` (JAX) and `scripts/train_pytorch.py` (PyTorch) are the two training entry points; both take a positional config name that is looked up in `training.config._CONFIGS`. Adding a new dataset/robot typically means: (1) add an Inputs/Outputs pair under `policies/`, (2) add a `DataConfig` + `TrainConfig` entry in `training/config.py`, (3) run `compute_norm_stats.py`, (4) train.

## Conventions

- Configs are dataclasses selected by string name; don't pass hyperparameters via CLI flags that aren't already in the config — add them to the dataclass.
- Norm stats live under `assets/<asset_id>/norm_stats.json` and are copied into each checkpoint. See `docs/norm_stats.md` for reloading stats from a pre-training checkpoint.
- The `debug` config is the fastest way to smoke-test a code path end-to-end.
- Run `pre-commit install` once; CI expects `ruff check .` and `ruff format .` to pass.
- `third_party/`, `docker/`, and `src/openpi/models_pytorch/transformers_replace/*` are excluded from ruff.
