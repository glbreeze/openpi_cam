#!/bin/bash
# Inner runner for openpi PyTorch single-stage training.
# Must be invoked from a wrapper sbatch via `exec bash` — assumes the job is
# already inside a slurm allocation. Not standalone-submittable.
#
# Required from the wrapper:
#   REPO_ROOT, CONFIG_NAME, DATASET_REPO_ID, NORM_ASSET_ID, EXP_NAME
# Optional (with defaults shown below):
#   NUM_GPUS, NUM_WORKERS, BATCH_SIZE, NUM_TRAIN_STEPS, SAVE_INTERVAL, KEEP_PERIOD,
#   CHECKPOINT_BASE_DIR, RESUME, WANDB_ENABLED, WANDB_PROJECT, WANDB_API_KEY_OVERRIDE,
#   PI3X_TARGETS_ROOT_OVERRIDE, GT_POINT_TARGETS_ROOT_OVERRIDE, PYTORCH_WEIGHT_PATH_OVERRIDE.

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT must be set by the wrapper}"
: "${CONFIG_NAME:?}" "${DATASET_REPO_ID:?}" "${NORM_ASSET_ID:?}" "${EXP_NAME:?}"

NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}
GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints}
WEIGHT_PATH=${PYTORCH_WEIGHT_PATH_OVERRIDE:-${GEO_ROOT}/pi0_base}
NORM_STATS_ROOT=${REPO_ROOT}/assets/pi0_libero

# --- per-job scratch (isolated TMP / HOME / XDG so concurrent jobs can't collide) ---
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp/$USER}/openpi_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp" TMP="${JOB_TMP_ROOT}/tmp" TEMP="${JOB_TMP_ROOT}/tmp"
export DATASETS_TMPDIR="${JOB_TMP_ROOT}/datasets_tmp"
export HOME="${JOB_TMP_ROOT}/home"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"
mkdir -p "${TMPDIR}" "${DATASETS_TMPDIR}" "${HOME}" \
         "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}"
chmod 700 "${TMPDIR}" "${DATASETS_TMPDIR}" "${HOME}" \
          "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" || true

# --- python env + persistent caches (HF / openpi assets under GEO_ROOT) ---
source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/lg154/conda-envs/openpi
cd "${REPO_ROOT}"
export OPENPI_GEO_ROOT="${GEO_ROOT}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${GEO_ROOT}}"
export HF_HOME="${GEO_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export OPENPI_DATA_HOME="${GEO_ROOT}/.cache/openpi"
unset TRANSFORMERS_CACHE
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" \
         "${OPENPI_DATA_HOME}" "${CHECKPOINT_BASE_DIR}"

# --- compute / runtime knobs ---
THREADS_PER_PROC=$(( ${SLURM_CPUS_PER_TASK:-${NUM_GPUS}} / NUM_GPUS ))
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( THREADS_PER_PROC > 0 ? THREADS_PER_PROC : 1 ))}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# --- wandb (per-job dirs so concurrent runs don't collide) ---
if [[ "${WANDB_ENABLED:-true}" == "true" ]]; then
  [[ -n "${WANDB_API_KEY_OVERRIDE:-}" ]] && export WANDB_API_KEY="${WANDB_API_KEY_OVERRIDE}"
  export WANDB_DIR="${JOB_TMP_ROOT}/wandb_root"
  export WANDB_CACHE_DIR="${JOB_TMP_ROOT}/wandb_cache"
  export WANDB_CONFIG_DIR="${JOB_TMP_ROOT}/wandb_config"
  export WANDB_DATA_DIR="${JOB_TMP_ROOT}/wandb_data"
  mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${WANDB_DATA_DIR}"
  WANDB_FLAG=--wandb-enabled
else
  export WANDB_MODE=disabled
  WANDB_FLAG=--no-wandb-enabled
fi

# --- pre-flight: required artifacts exist on disk ---
[[ -f "${HF_LEROBOT_HOME}/${DATASET_REPO_ID}/meta/info.json" ]] \
  || { echo "Missing LeRobot dataset: ${HF_LEROBOT_HOME}/${DATASET_REPO_ID}" >&2; exit 1; }
[[ -f "${WEIGHT_PATH}/model.safetensors" ]] \
  && { [[ -f "${WEIGHT_PATH}/config.json" ]] || [[ -f "${WEIGHT_PATH}/metadata.pt" ]]; } \
  || { echo "Missing PyTorch weights at: ${WEIGHT_PATH} (need model.safetensors + config.json or metadata.pt)" >&2; exit 1; }
[[ -f "${NORM_STATS_ROOT}/${NORM_ASSET_ID}/norm_stats.json" ]] \
  || { echo "Missing norm stats: ${NORM_STATS_ROOT}/${NORM_ASSET_ID}/norm_stats.json" >&2; exit 1; }

# --- assemble training argv ---
TRAIN_ARGS=(
  "${CONFIG_NAME}"
  --exp-name="${EXP_NAME}"
  --project-name="${WANDB_PROJECT:-openpi}"
  --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}"
  --batch-size="${BATCH_SIZE:-32}"
  --num-workers="${NUM_WORKERS:-16}"
  --num-train-steps="${NUM_TRAIN_STEPS:-60000}"
  --save-interval="${SAVE_INTERVAL:-1000}"
  --keep-period="${KEEP_PERIOD:-5000}"
  "$([[ "${RESUME:-false}" == "true" ]] && echo --resume || echo --overwrite)"
  --pytorch_weight_path "${WEIGHT_PATH}"
  --data.repo_id "${DATASET_REPO_ID}"
  --data.assets.assets_dir "${NORM_STATS_ROOT}"
  --data.assets.asset_id "${NORM_ASSET_ID}"
  "${WANDB_FLAG}"
)
[[ -n "${PI3X_TARGETS_ROOT_OVERRIDE:-}" ]] \
  && TRAIN_ARGS+=(--data.pi3x_targets_root "${PI3X_TARGETS_ROOT_OVERRIDE}")
[[ -n "${GT_POINT_TARGETS_ROOT_OVERRIDE:-}" ]] \
  && TRAIN_ARGS+=(--data.gt_point_targets_root "${GT_POINT_TARGETS_ROOT_OVERRIDE}")
# Free-form tyro overrides from the wrapper (e.g. `--model.aux_point_head.X Y`).
# Word-split on whitespace; quote-aware splitting isn't supported, so values must
# not contain spaces.
if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  read -r -a _extra_train_args <<< "${EXTRA_TRAIN_ARGS}"
  TRAIN_ARGS+=("${_extra_train_args[@]}")
fi

echo "===== TRAIN ENV CHECK ====="
echo "config:      ${CONFIG_NAME}  exp: ${EXP_NAME}"
echo "dataset:     ${HF_LEROBOT_HOME}/${DATASET_REPO_ID}"
echo "weight:      ${WEIGHT_PATH}"
echo "norm stats:  ${NORM_STATS_ROOT}/${NORM_ASSET_ID}"
echo "checkpoints: ${CHECKPOINT_BASE_DIR}"
echo "wandb:       ${WANDB_FLAG} (project=${WANDB_PROJECT:-openpi} entity=${WANDB_ENTITY:-<default>})"
echo "gpus=${NUM_GPUS} bs=${BATCH_SIZE:-32} steps=${NUM_TRAIN_STEPS:-60000} pi3x=${PI3X_TARGETS_ROOT_OVERRIDE:-<none>}"
which python; python -V; hostname; nvidia-smi -L || true

srun --export=ALL --ntasks=1 --ntasks-per-node=1 \
     --cpus-per-task="${SLURM_CPUS_PER_TASK}" --kill-on-bad-exit=1 \
  python -m torch.distributed.run --standalone --nnodes=1 \
         --nproc_per_node="${NUM_GPUS}" \
    scripts/train_pytorch.py "${TRAIN_ARGS[@]}"
