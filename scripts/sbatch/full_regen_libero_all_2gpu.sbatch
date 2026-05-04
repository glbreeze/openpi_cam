#!/bin/bash
#SBATCH --job-name=regen_libero_all_2gpu
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_regen_full/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_regen_full/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:2
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yp2841/geometry-vla/openpi_cam}
GEO_ROOT=${GEO_ROOT:-/scratch/yp2841/geometry-vla}
RAW_SRC_BASE=${RAW_SRC_BASE:-${GEO_ROOT}/libero_raw_datasets}
LIBERO_CAMERA_REPO=${LIBERO_CAMERA_REPO:-${GEO_ROOT}/LIBERO-Camera}
OPENPI_REPO=${OPENPI_REPO:-${REPO_ROOT}}
LIBERO_CONDA_ENV=${LIBERO_CONDA_ENV:-/scratch/yp2841/.conda-envs/openvla-oft}
CONDA_SH=${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}
OPENPI_PYTHON=${OPENPI_PYTHON:-${OPENPI_REPO}/.venv/bin/python}
WORK_DIR=${WORK_DIR:-/scratch/yp2841/data/libero_regen_work}
OUT_HDF5_ROOT=${OUT_HDF5_ROOT:-/scratch/yp2841/data/libero_cam_v2}
LIBERO_CFG_DIR=${LIBERO_CFG_DIR:-/scratch/yp2841/data/libero_cam_v2_cfg}
HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/scratch/yp2841/cache/lerobot}
REPO_ID=${REPO_ID:-glbreeze/libero_cam_v2}
RENDER_RES=${RENDER_RES:-256}
SETTLE_STEPS=${SETTLE_STEPS:-10}

WAVE1=(libero_object libero_spatial)
WAVE2=(libero_goal libero_10)

JOB_TMP_BASE="${SLURM_TMPDIR:-/scratch/yp2841/tmp}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/libero_regen_full_2gpu_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export MPLCONFIGDIR="${JOB_TMP_ROOT}/mpl"
export NUMBA_CACHE_DIR="${JOB_TMP_ROOT}/numba"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"

mkdir -p \
  "${TMPDIR}" \
  "${MPLCONFIGDIR}" \
  "${NUMBA_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${XDG_STATE_HOME}" \
  "${WORK_DIR}" \
  "${OUT_HDF5_ROOT}" \
  "${LIBERO_CFG_DIR}" \
  "${HF_LEROBOT_HOME}" \
  /scratch/yp2841/geometry-vla/openpi_cam/log/libero_regen_full

export SRC_BASE="${RAW_SRC_BASE}"
export LIBERO_CAMERA_REPO
export OPENPI_REPO
export LIBERO_CONDA_ENV
export CONDA_SH
export OPENPI_PYTHON
export WORK_DIR
export OUT_HDF5_ROOT
export LIBERO_CFG_DIR
export HF_LEROBOT_HOME
export REPO_ID
export RENDER_RES
export SETTLE_STEPS

if [[ -e "${HF_LEROBOT_HOME}/${REPO_ID}" ]]; then
  echo "Target LeRobot repo already exists: ${HF_LEROBOT_HOME}/${REPO_ID}" >&2
  exit 1
fi

echo "===== FULL REGEN ENV (2 GPU) ====="
echo "repo root: ${REPO_ROOT}"
echo "geo root: ${GEO_ROOT}"
echo "src base: ${SRC_BASE}"
echo "libero repo: ${LIBERO_CAMERA_REPO}"
echo "openpi repo: ${OPENPI_REPO}"
echo "libero conda env: ${LIBERO_CONDA_ENV}"
echo "openpi python: ${OPENPI_PYTHON}"
echo "work dir: ${WORK_DIR}"
echo "out hdf5 root: ${OUT_HDF5_ROOT}"
echo "libero cfg dir: ${LIBERO_CFG_DIR}"
echo "hf lerobot home: ${HF_LEROBOT_HOME}"
echo "repo id: ${REPO_ID}"
hostname
date

run_suite() {
  local gpu_idx="$1"
  local suite="$2"
  local log_path="${WORK_DIR}/regen_${suite}.log"
  echo "Launching ${suite} on GPU ${gpu_idx}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu_idx}" \
    SUITES="${suite}" \
    SKIP_PHASE2=1 \
    bash "${OPENPI_REPO}/scripts/regen_libero_all.sh" \
    > "${log_path}" 2>&1
}

run_wave() {
  local -n suites_ref="$1"
  declare -a pids=()
  local wave_failed=0

  for gpu_idx in 0 1; do
    run_suite "${gpu_idx}" "${suites_ref[$gpu_idx]}" &
    pids+=("$!")
  done

  for idx in 0 1; do
    if ! wait "${pids[$idx]}"; then
      echo "Phase 1 failed for ${suites_ref[$idx]} (pid=${pids[$idx]})." >&2
      wave_failed=1
    fi
  done

  return "${wave_failed}"
}

echo "Starting wave 1: ${WAVE1[*]}"
if ! run_wave WAVE1; then
  echo "Wave 1 failed. Check ${WORK_DIR}/regen_*.log" >&2
  exit 1
fi

echo "Starting wave 2: ${WAVE2[*]}"
if ! run_wave WAVE2; then
  echo "Wave 2 failed. Check ${WORK_DIR}/regen_*.log" >&2
  exit 1
fi

echo "Phase 1 complete for all suites."

SKIP_PHASE1=1 bash "${OPENPI_REPO}/scripts/regen_libero_all.sh" \
  > "${WORK_DIR}/convert_all.log" 2>&1

"${OPENPI_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

repo_root = Path(os.environ["HF_LEROBOT_HOME"]) / os.environ["REPO_ID"]
info_path = repo_root / "meta" / "info.json"
if not info_path.is_file():
    raise SystemExit(f"Missing info.json at {info_path}")

info = json.loads(info_path.read_text())
print("repo:", repo_root)
print("episodes:", info.get("total_episodes"))
print("frames:", info.get("total_frames"))
print("tasks:", info.get("total_tasks"))
assert info.get("codebase_version") == "v2.1"
assert info["features"]["image"]["shape"] == [256, 256, 3]
assert info["features"]["agent_extrinsic"]["shape"] == [4, 4]
assert "joint_state" in info["features"]
print("full-run metadata OK")
PY

echo "Full regen complete."
