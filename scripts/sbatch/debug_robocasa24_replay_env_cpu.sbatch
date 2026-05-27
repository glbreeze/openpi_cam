#!/bin/bash
#SBATCH --job-name=pi0_rc24_replay
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/robocasa24_infer/replay_env-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/robocasa24_infer/replay_env-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cs
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

  for candidate in "${REPO_ROOT:-}" "${OPENPI_CAM_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "${PWD:-}" "${script_root}"; do
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

REPO_ROOT=${REPO_ROOT:-$(resolve_repo_root)} || {
  echo "Unable to locate repo root. Set REPO_ROOT or submit from inside the repo." >&2
  exit 1
}
GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}
cd "${REPO_ROOT}"

JOB_TMP_BASE="${SLURM_TMPDIR:-/scratch/${USER}/tmp}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/robocasa24_replay_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export ROBOCASA_HOME="${ROBOCASA_HOME:-/scratch/${USER}/.cache/robocasa24_home}"
export HOME="${ROBOCASA_HOME}"
export XDG_CACHE_HOME="${GEO_ROOT}/.cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"
export MPLCONFIGDIR="${JOB_TMP_ROOT}/mplconfig"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONNOUSERSITE=1
export ROBOCASA_TMP_XML_DIR="${ROBOCASA_TMP_XML_DIR:-${JOB_TMP_ROOT}/robocasa_mjcf_xml}"
mkdir -p "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" "${MPLCONFIGDIR}" log/robocasa24_infer

ROBOCASA_ENV=${ROBOCASA_ENV:-/scratch/${USER}/.venvs/robocasa24}
if [[ ! -x "${ROBOCASA_ENV}/bin/python" && -x /scratch/yz11445/.venvs/robocasa24/bin/python ]]; then
  ROBOCASA_ENV=/scratch/yz11445/.venvs/robocasa24
fi
ROBOCASA_PYTHON="${ROBOCASA_ENV}/bin/python"
ROBOCASA24_SOURCE_ROOT=${ROBOCASA24_SOURCE_ROOT:-/scratch/${USER}/src/robocasa24_infer/robocasa_v02}
ROBOSUITE151_SOURCE_ROOT=${ROBOSUITE151_SOURCE_ROOT:-/scratch/${USER}/src/robocasa24_infer/robosuite_v151}
if [[ ! -d "${ROBOCASA24_SOURCE_ROOT}" ]]; then
  ROBOCASA24_SOURCE_ROOT=/scratch/yz11445/openpi_cam/third_party/robocasa_v02
fi
if [[ ! -d "${ROBOSUITE151_SOURCE_ROOT}" ]]; then
  ROBOSUITE151_SOURCE_ROOT=/scratch/yz11445/openpi_cam/third_party/robosuite_v151
fi

DATASET=${DATASET:-/scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_drawer/OpenDrawer/2024-05-03/demo_gentex_im128_randcams.hdf5}
DEMOS=${DEMOS:-demo_1,demo_2}
MAX_STEPS=${MAX_STEPS:-260}
IMAGE_SIZE=${IMAGE_SIZE:-128}
RESULT_JSON=${RESULT_JSON:-${REPO_ROOT}/log/robocasa24_infer/replay_env_OpenDrawer_${SLURM_JOB_ID:-manual}.json}
IMAGE_DUMP_DIR=${IMAGE_DUMP_DIR:-}
RESULT_JSON=${RESULT_JSON//%j/${SLURM_JOB_ID:-manual}}
RESULT_JSON=${RESULT_JSON//%u/${USER:-user}}
IMAGE_DUMP_DIR=${IMAGE_DUMP_DIR//%j/${SLURM_JOB_ID:-manual}}
IMAGE_DUMP_DIR=${IMAGE_DUMP_DIR//%u/${USER:-user}}

[[ -x "${ROBOCASA_PYTHON}" ]] || { echo "Missing RoboCasa env python: ${ROBOCASA_PYTHON}" >&2; exit 1; }
[[ -f "${DATASET}" ]] || { echo "Missing dataset: ${DATASET}" >&2; exit 1; }

echo "===== ROBOCASA24 REPLAY / EVAL ENV CHECK ====="
echo "repo root      : ${REPO_ROOT}"
echo "robocasa env   : ${ROBOCASA_ENV}"
echo "robocasa src   : ${ROBOCASA24_SOURCE_ROOT}"
echo "robosuite src  : ${ROBOSUITE151_SOURCE_ROOT}"
echo "dataset        : ${DATASET}"
echo "demos          : ${DEMOS}"
echo "max steps      : ${MAX_STEPS}"
echo "image size     : ${IMAGE_SIZE}"
echo "image dump dir : ${IMAGE_DUMP_DIR:-none}"
echo "result json    : ${RESULT_JSON}"
hostname
date

export PATH="${ROBOCASA_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${ROBOCASA24_SOURCE_ROOT}:${ROBOSUITE151_SOURCE_ROOT}:${PYTHONPATH:-}"

extra_args=()
if [[ -n "${IMAGE_DUMP_DIR}" ]]; then
  extra_args+=(--image-dump-dir "${IMAGE_DUMP_DIR}")
fi

"${ROBOCASA_PYTHON}" scripts/debug/check_robocasa_replay_and_eval_env.py \
  --dataset "${DATASET}" \
  --demos "${DEMOS}" \
  --max-steps "${MAX_STEPS}" \
  --image-size "${IMAGE_SIZE}" \
  --out "${RESULT_JSON}" \
  "${extra_args[@]}"
