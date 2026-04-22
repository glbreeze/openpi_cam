#!/bin/bash
#SBATCH --job-name=pi0_libero_object_ft
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_libero_object_ft/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_libero_object_ft/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --account=torch_pr_637_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:4
#SBATCH --export=ALL

set -euo pipefail

resolve_repo_root() {
  local candidate
  local script_root=""

  if script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd); then
    :
  else
    script_root=""
  fi

  for candidate in "${REPO_ROOT:-}" "${OPENPI_CAM_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "${script_root}"; do
    [[ -n "${candidate}" ]] || continue
    while [[ "${candidate}" != "/" ]]; do
      if [[ -f "${candidate}/scripts/env/activate_env.sh" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
      candidate=$(dirname -- "${candidate}")
    done
  done

  return 1
}

if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(resolve_repo_root || true)"
fi
if [[ -z "${REPO_ROOT}" && -d "/scratch/yp2841/geometry-vla/openpi_cam" ]]; then
  REPO_ROOT="/scratch/yp2841/geometry-vla/openpi_cam"
fi
if [[ -z "${REPO_ROOT}" ]]; then
  echo "Unable to locate repo root. Set REPO_ROOT or submit from inside the repo." >&2
  exit 1
fi
GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}

USE_CAM=${USE_CAM:-false}
RESUME=${RESUME:-false}
WANDB_ENABLED=${WANDB_ENABLED:-true}
WANDB_PROJECT=${WANDB_PROJECT:-openpi_cam_libero_object}
NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}
NUM_WORKERS=${NUM_WORKERS:-16}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-60000}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
KEEP_PERIOD=${KEEP_PERIOD:-5000}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints}
CACHE_ROOT=${CACHE_ROOT:-${GEO_ROOT}/.cache/huggingface}
OPENPI_CACHE_ROOT=${OPENPI_CACHE_ROOT:-${GEO_ROOT}/.cache/openpi}
MODEL_POSE_ENC_TYPE=${MODEL_POSE_ENC_TYPE:-absolute_pose}
CROSS_VIEW_TYPE=${CROSS_VIEW_TYPE:-}
TRAIN_RECIPE=${TRAIN_RECIPE:-auto}
SIMPLE_STAGE1_STEPS=${SIMPLE_STAGE1_STEPS:-5000}
SIMPLE_STAGE2_STEPS=${SIMPLE_STAGE2_STEPS:-55000}
SIMPLE_STAGE1_WARMUP_STEPS=${SIMPLE_STAGE1_WARMUP_STEPS:-500}
SIMPLE_STAGE1_PEAK_LR=${SIMPLE_STAGE1_PEAK_LR:-1e-4}
SIMPLE_STAGE1_DECAY_STEPS=${SIMPLE_STAGE1_DECAY_STEPS:-5000}
SIMPLE_STAGE1_END_LR=${SIMPLE_STAGE1_END_LR:-1e-5}
SIMPLE_STAGE2_WARMUP_STEPS=${SIMPLE_STAGE2_WARMUP_STEPS:-1000}
SIMPLE_STAGE2_PEAK_LR=${SIMPLE_STAGE2_PEAK_LR:-2.5e-5}
SIMPLE_STAGE2_DECAY_STEPS=${SIMPLE_STAGE2_DECAY_STEPS:-55000}
SIMPLE_STAGE2_END_LR=${SIMPLE_STAGE2_END_LR:-2.5e-6}
SIMPLE_NEW_MODULE_PREFIXES=${SIMPLE_NEW_MODULE_PREFIXES:-paligemma_with_expert.cam_pose_encoder,paligemma_with_expert.view_embedding,paligemma_with_expert.cross_view_fusion,paligemma_with_expert.ray_embed}
SIMPLE_STAGE2_NEW_MODULE_LR_MULTIPLIER=${SIMPLE_STAGE2_NEW_MODULE_LR_MULTIPLIER:-4.0}

if [[ "${USE_CAM}" == "true" ]]; then
  CONFIG_NAME=${CONFIG_NAME:-pi0_libero_cam_pytorch_full_finetune}
  DATASET_REPO_ID=${DATASET_REPO_ID:-glbreeze/libero_object_cam}
  NORM_ASSET_ID=${NORM_ASSET_ID:-glbreeze/libero_object_cam}
  EXP_NAME=${EXP_NAME:-pi0_libero_object_cam_${SLURM_JOB_ID:-manual}}

  if [[ -z "${CROSS_VIEW_TYPE}" ]]; then
    case "${CONFIG_NAME}" in
      pi0_libero_cam_pytorch_full_finetune)
        CROSS_VIEW_TYPE="simple"
        ;;
      pi0_libero_cam_pytorch_full_finetune_standard)
        CROSS_VIEW_TYPE="standard"
        ;;
    esac
  fi
else
  CONFIG_NAME=${CONFIG_NAME:-pi0_libero_pytorch_full_finetune}
  DATASET_REPO_ID=${DATASET_REPO_ID:-glbreeze/libero_object}
  NORM_ASSET_ID=${NORM_ASSET_ID:-glbreeze/libero_object}
  EXP_NAME=${EXP_NAME:-pi0_libero_object_${SLURM_JOB_ID:-manual}}
fi

if [[ "${TRAIN_RECIPE}" == "auto" ]]; then
  if [[ "${USE_CAM}" == "true" && "${CONFIG_NAME}" == "pi0_libero_cam_pytorch_full_finetune" && "${CROSS_VIEW_TYPE}" == "simple" ]]; then
    TRAIN_RECIPE="simple_two_stage"
  else
    TRAIN_RECIPE="single_stage"
  fi
fi

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/openpi_cam_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export DATASETS_TMPDIR="${JOB_TMP_ROOT}/datasets_tmp"
export HOME="${JOB_TMP_ROOT}/home"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"

mkdir -p "${TMPDIR}" "${DATASETS_TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}"
chmod 700 "${TMPDIR}" "${DATASETS_TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" || true

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

export HF_HOME="${CACHE_ROOT}"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export HF_HUB_CACHE="${CACHE_ROOT}/hub"
export OPENPI_CACHE_DIR="${OPENPI_CACHE_ROOT}"
export OPENPI_DATA_HOME="${OPENPI_CACHE_ROOT}"
unset TRANSFORMERS_CACHE

mkdir -p \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_HUB_CACHE}" \
  "${OPENPI_CACHE_DIR}" \
  "${CHECKPOINT_BASE_DIR}"

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "NUM_GPUS must be >= 1, got ${NUM_GPUS}" >&2
  exit 1
fi

THREADS_PER_PROC=$(( SLURM_CPUS_PER_TASK / NUM_GPUS ))
if [[ "${THREADS_PER_PROC}" -lt 1 ]]; then
  THREADS_PER_PROC=1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${THREADS_PER_PROC}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "${WANDB_ENABLED}" == "true" ]]; then
  unset WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR WANDB_DATA_DIR
  export WANDB_API_KEY="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"
  export WANDB_DIR="${JOB_TMP_ROOT}/wandb_root"
  export WANDB_CACHE_DIR="${JOB_TMP_ROOT}/wandb_cache"
  export WANDB_CONFIG_DIR="${JOB_TMP_ROOT}/wandb_config"
  export WANDB_DATA_DIR="${JOB_TMP_ROOT}/wandb_data"
  mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${WANDB_DATA_DIR}"
else
  export WANDB_MODE=disabled
fi

DATASET_DIR="${HF_LEROBOT_HOME}/${DATASET_REPO_ID}"
BASE_MODEL_DIR="${OPENPI_PI0_BASE_DIR}"
NORM_STATS_ROOT="${OPENPI_PI0_LIBERO_NORM_DIR}"
NORM_STATS_DIR="${NORM_STATS_ROOT}/${NORM_ASSET_ID}"

if [[ ! -d "${DATASET_DIR}/data" || ! -f "${DATASET_DIR}/meta/info.json" ]]; then
  echo "Missing local LeRobot dataset at ${DATASET_DIR}" >&2
  exit 1
fi

if [[ ! -f "${BASE_MODEL_DIR}/model.safetensors" || ! -f "${BASE_MODEL_DIR}/config.json" ]]; then
  echo "Missing local PyTorch base model at ${BASE_MODEL_DIR}" >&2
  exit 1
fi

if [[ ! -f "${NORM_STATS_DIR}/norm_stats.json" ]]; then
  echo "Missing local norm stats at ${NORM_STATS_DIR}/norm_stats.json" >&2
  exit 1
fi

build_train_args() {
  local -n out_args=$1
  local exp_name=$2
  local num_steps=$3
  local weight_path=$4
  local wandb_flag=$5
  local resume_flag=$6
  local warmup_steps=${7:-}
  local peak_lr=${8:-}
  local decay_steps=${9:-}
  local end_lr=${10:-}

  local wand_arg="--no-wandb-enabled"
  if [[ "${wandb_flag}" == "true" ]]; then
    wand_arg="--wandb-enabled"
  fi

  local resume_or_overwrite_arg="--overwrite"
  if [[ "${resume_flag}" == "true" ]]; then
    resume_or_overwrite_arg="--resume"
  fi

  out_args=(
    "${CONFIG_NAME}"
    --exp-name="${exp_name}"
    --project-name="${WANDB_PROJECT}"
    --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}"
    --batch-size="${BATCH_SIZE}"
    --num-workers="${NUM_WORKERS}"
    --num-train-steps="${num_steps}"
    --save-interval="${SAVE_INTERVAL}"
    --keep-period="${KEEP_PERIOD}"
    "${resume_or_overwrite_arg}"
    --pytorch_weight_path "${weight_path}"
    --data.repo_id "${DATASET_REPO_ID}"
    --data.assets.assets_dir "${NORM_STATS_ROOT}"
    --data.assets.asset_id "${NORM_ASSET_ID}"
    "${wand_arg}"
  )

  if [[ "${USE_CAM}" == "true" ]]; then
    out_args+=(
      --model.pose_enc_type "${MODEL_POSE_ENC_TYPE}"
    )
    if [[ -n "${CROSS_VIEW_TYPE}" ]]; then
      out_args+=(
        --model.cross_view.type "${CROSS_VIEW_TYPE}"
      )
    fi
  fi

  if [[ -n "${warmup_steps}" ]]; then
    out_args+=(--lr_schedule.warmup_steps "${warmup_steps}")
  fi
  if [[ -n "${peak_lr}" ]]; then
    out_args+=(--lr_schedule.peak_lr "${peak_lr}")
  fi
  if [[ -n "${decay_steps}" ]]; then
    out_args+=(--lr_schedule.decay_steps "${decay_steps}")
  fi
  if [[ -n "${end_lr}" ]]; then
    out_args+=(--lr_schedule.decay_lr "${end_lr}")
  fi
}

build_simple_lr_multiplier_spec() {
  local prefix
  local spec=""
  IFS=',' read -r -a simple_prefixes <<< "${SIMPLE_NEW_MODULE_PREFIXES}"
  for prefix in "${simple_prefixes[@]}"; do
    [[ -n "${prefix}" ]] || continue
    if [[ -n "${spec}" ]]; then
      spec+=","
    fi
    spec+="${prefix}=${SIMPLE_STAGE2_NEW_MODULE_LR_MULTIPLIER}"
  done
  printf '%s\n' "${spec}"
}

get_latest_checkpoint_step() {
  local checkpoint_root=$1
  local path
  local latest=""

  [[ -d "${checkpoint_root}" ]] || return 0

  shopt -s nullglob
  for path in "${checkpoint_root}"/*; do
    [[ -d "${path}" ]] || continue
    local step
    step=$(basename -- "${path}")
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    if [[ -z "${latest}" || "${step}" -gt "${latest}" ]]; then
      latest="${step}"
    fi
  done
  shopt -u nullglob

  if [[ -n "${latest}" ]]; then
    printf '%s\n' "${latest}"
  fi
}

run_training_command() {
  local phase_name=$1
  local phase_wandb=$2
  local trainable_prefixes=$3
  local lr_multiplier_spec=$4
  shift 4
  local -a phase_args=("$@")

  local old_wandb_mode=${WANDB_MODE-__unset__}
  local old_trainable_prefixes=${OPENPI_TRAINABLE_PREFIXES-__unset__}
  local old_lr_multipliers=${OPENPI_LR_MULTIPLIERS-__unset__}

  if [[ "${phase_wandb}" == "true" ]]; then
    unset WANDB_MODE
  else
    export WANDB_MODE=disabled
  fi

  if [[ -n "${trainable_prefixes}" ]]; then
    export OPENPI_TRAINABLE_PREFIXES="${trainable_prefixes}"
  else
    unset OPENPI_TRAINABLE_PREFIXES
  fi

  if [[ -n "${lr_multiplier_spec}" ]]; then
    export OPENPI_LR_MULTIPLIERS="${lr_multiplier_spec}"
  else
    unset OPENPI_LR_MULTIPLIERS
  fi

  echo "===== ${phase_name} ====="
  echo "phase wandb: ${phase_wandb}"
  echo "trainable prefixes: ${OPENPI_TRAINABLE_PREFIXES:-<all>}"
  echo "lr multipliers: ${OPENPI_LR_MULTIPLIERS:-<default>}"

  srun --export=ALL --ntasks=1 --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --kill-on-bad-exit=1 \
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
      scripts/train_pytorch.py "${phase_args[@]}"

  if [[ "${old_wandb_mode}" == "__unset__" ]]; then
    unset WANDB_MODE
  else
    export WANDB_MODE="${old_wandb_mode}"
  fi

  if [[ "${old_trainable_prefixes}" == "__unset__" ]]; then
    unset OPENPI_TRAINABLE_PREFIXES
  else
    export OPENPI_TRAINABLE_PREFIXES="${old_trainable_prefixes}"
  fi

  if [[ "${old_lr_multipliers}" == "__unset__" ]]; then
    unset OPENPI_LR_MULTIPLIERS
  else
    export OPENPI_LR_MULTIPLIERS="${old_lr_multipliers}"
  fi
}

echo "===== TRAIN ENV CHECK ====="
echo "repo root: ${REPO_ROOT}"
echo "use_cam: ${USE_CAM}"
echo "resume: ${RESUME}"
echo "config: ${CONFIG_NAME}"
echo "exp: ${EXP_NAME}"
echo "train_recipe: ${TRAIN_RECIPE}"
echo "pose_enc_type: ${MODEL_POSE_ENC_TYPE}"
echo "cross_view_type: ${CROSS_VIEW_TYPE:-config_default}"
echo "dataset: ${DATASET_DIR}"
echo "base model: ${BASE_MODEL_DIR}"
echo "norm stats: ${NORM_STATS_DIR}"
echo "checkpoints: ${CHECKPOINT_BASE_DIR}"
echo "wandb root: ${WANDB_DIR:-disabled}"
echo "num_gpus: ${NUM_GPUS}"
echo "batch_size: ${BATCH_SIZE}"
echo "num_train_steps: ${NUM_TRAIN_STEPS}"
echo "save_interval: ${SAVE_INTERVAL}"
echo "keep_period: ${KEEP_PERIOD}"
which python
python -V
hostname
nvidia-smi -L || true

if [[ "${TRAIN_RECIPE}" == "simple_two_stage" ]]; then
  SIMPLE_STAGE1_EXP_NAME="${EXP_NAME}_warmup"
  SIMPLE_STAGE1_CKPT_ROOT="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${SIMPLE_STAGE1_EXP_NAME}"
  SIMPLE_STAGE1_CKPT_DIR="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${SIMPLE_STAGE1_EXP_NAME}/${SIMPLE_STAGE1_STEPS}"
  SIMPLE_STAGE2_CKPT_ROOT="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"
  SIMPLE_STAGE2_LR_MULTIPLIERS=$(build_simple_lr_multiplier_spec)
  SIMPLE_STAGE2_LATEST_STEP=$(get_latest_checkpoint_step "${SIMPLE_STAGE2_CKPT_ROOT}")

  echo "simple stage1 steps: ${SIMPLE_STAGE1_STEPS}"
  echo "simple stage2 steps: ${SIMPLE_STAGE2_STEPS}"
  echo "simple stage1 peak lr: ${SIMPLE_STAGE1_PEAK_LR}"
  echo "simple stage2 peak lr: ${SIMPLE_STAGE2_PEAK_LR}"
  echo "simple new module prefixes: ${SIMPLE_NEW_MODULE_PREFIXES}"
  echo "simple stage2 lr multipliers: ${SIMPLE_STAGE2_LR_MULTIPLIERS}"
  echo "simple stage2 latest step: ${SIMPLE_STAGE2_LATEST_STEP:-<none>}"

  if [[ ! -f "${SIMPLE_STAGE1_CKPT_DIR}/model.safetensors" ]]; then
    build_train_args SIMPLE_STAGE1_ARGS \
      "${SIMPLE_STAGE1_EXP_NAME}" \
      "${SIMPLE_STAGE1_STEPS}" \
      "${BASE_MODEL_DIR}" \
      "false" \
      "false" \
      "${SIMPLE_STAGE1_WARMUP_STEPS}" \
      "${SIMPLE_STAGE1_PEAK_LR}" \
      "${SIMPLE_STAGE1_DECAY_STEPS}" \
      "${SIMPLE_STAGE1_END_LR}"
    run_training_command "simple_stage1_warmup" "false" "${SIMPLE_NEW_MODULE_PREFIXES}" "" "${SIMPLE_STAGE1_ARGS[@]}"
  else
    echo "Skipping stage1; found existing checkpoint at ${SIMPLE_STAGE1_CKPT_DIR}"
  fi

  if [[ ! -f "${SIMPLE_STAGE1_CKPT_DIR}/model.safetensors" ]]; then
    echo "Expected stage1 checkpoint at ${SIMPLE_STAGE1_CKPT_DIR}/model.safetensors" >&2
    exit 1
  fi

  if [[ -n "${SIMPLE_STAGE2_LATEST_STEP}" && "${SIMPLE_STAGE2_LATEST_STEP}" -ge "${SIMPLE_STAGE2_STEPS}" ]]; then
    echo "Stage2 already complete at step ${SIMPLE_STAGE2_LATEST_STEP}; nothing to do."
    exit 0
  fi

  if [[ -n "${SIMPLE_STAGE2_LATEST_STEP}" ]]; then
    build_train_args SIMPLE_STAGE2_ARGS \
      "${EXP_NAME}" \
      "${SIMPLE_STAGE2_STEPS}" \
      "${SIMPLE_STAGE1_CKPT_DIR}" \
      "${WANDB_ENABLED}" \
      "true" \
      "${SIMPLE_STAGE2_WARMUP_STEPS}" \
      "${SIMPLE_STAGE2_PEAK_LR}" \
      "${SIMPLE_STAGE2_DECAY_STEPS}" \
      "${SIMPLE_STAGE2_END_LR}"
    run_training_command "simple_stage2_resume" "${WANDB_ENABLED}" "" "${SIMPLE_STAGE2_LR_MULTIPLIERS}" "${SIMPLE_STAGE2_ARGS[@]}"
  else
    build_train_args SIMPLE_STAGE2_ARGS \
      "${EXP_NAME}" \
      "${SIMPLE_STAGE2_STEPS}" \
      "${SIMPLE_STAGE1_CKPT_DIR}" \
      "${WANDB_ENABLED}" \
      "false" \
      "${SIMPLE_STAGE2_WARMUP_STEPS}" \
      "${SIMPLE_STAGE2_PEAK_LR}" \
      "${SIMPLE_STAGE2_DECAY_STEPS}" \
      "${SIMPLE_STAGE2_END_LR}"
    run_training_command "simple_stage2_joint" "${WANDB_ENABLED}" "" "${SIMPLE_STAGE2_LR_MULTIPLIERS}" "${SIMPLE_STAGE2_ARGS[@]}"
  fi
else
  build_train_args TRAIN_ARGS \
    "${EXP_NAME}" \
    "${NUM_TRAIN_STEPS}" \
    "${BASE_MODEL_DIR}" \
    "${WANDB_ENABLED}" \
    "${RESUME}" \
    "" \
    "" \
    "" \
    ""
  run_training_command "single_stage_train" "${WANDB_ENABLED}" "" "" "${TRAIN_ARGS[@]}"
fi
