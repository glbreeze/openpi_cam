#!/bin/bash
#SBATCH --job-name=convert_libero_all_phase2
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_regen_full/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_regen_full/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cs
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yp2841/geometry-vla/openpi_cam}
GEO_ROOT=${GEO_ROOT:-/scratch/yp2841/geometry-vla}
OPENPI_REPO=${OPENPI_REPO:-${REPO_ROOT}}
OPENPI_PYTHON=${OPENPI_PYTHON:-${OPENPI_REPO}/.venv/bin/python}
OUT_HDF5_ROOT=${OUT_HDF5_ROOT:-/scratch/yp2841/data/libero_cam_v2}
HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/scratch/yp2841/cache/lerobot}
REPO_ID=${REPO_ID:-glbreeze/libero_cam_v2}
LEROBOT_MODE=${LEROBOT_MODE:-original_only}
RENDER_RES=${RENDER_RES:-256}
IMAGE_WRITER_THREADS=${IMAGE_WRITER_THREADS:-12}
IMAGE_WRITER_PROCESSES=${IMAGE_WRITER_PROCESSES:-6}

JOB_TMP_BASE="${SLURM_TMPDIR:-/scratch/yp2841/tmp}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/libero_phase2_${SLURM_JOB_ID:-$$}"
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
  /scratch/yp2841/geometry-vla/openpi_cam/log/libero_regen_full

if [[ ! -x "${OPENPI_PYTHON}" ]]; then
  echo "Missing OPENPI_PYTHON: ${OPENPI_PYTHON}" >&2
  exit 1
fi

if [[ ! -d "${OUT_HDF5_ROOT}" ]]; then
  echo "Missing OUT_HDF5_ROOT: ${OUT_HDF5_ROOT}" >&2
  exit 1
fi

for suite in libero_object libero_spatial libero_goal libero_10; do
  if [[ ! -d "${OUT_HDF5_ROOT}/${suite}" ]]; then
    echo "Missing suite output dir: ${OUT_HDF5_ROOT}/${suite}" >&2
    exit 1
  fi
done

TARGET_REPO_DIR="${HF_LEROBOT_HOME}/${REPO_ID}"
if [[ -e "${TARGET_REPO_DIR}" ]]; then
  backup_suffix="$(date +%Y%m%d_%H%M%S)"
  backup_repo_dir="${TARGET_REPO_DIR}_partial_${backup_suffix}"
  mkdir -p "$(dirname "${backup_repo_dir}")"
  mv "${TARGET_REPO_DIR}" "${backup_repo_dir}"
  echo "Backed up incomplete repo to: ${backup_repo_dir}"
fi

echo "===== PHASE 2 CONVERT ENV ====="
echo "repo root: ${REPO_ROOT}"
echo "openpi repo: ${OPENPI_REPO}"
echo "openpi python: ${OPENPI_PYTHON}"
echo "dataset root: ${OUT_HDF5_ROOT}"
echo "hf lerobot home: ${HF_LEROBOT_HOME}"
echo "repo id: ${REPO_ID}"
echo "mode: ${LEROBOT_MODE}"
echo "image size: ${RENDER_RES}"
echo "image writer threads: ${IMAGE_WRITER_THREADS}"
echo "image writer processes: ${IMAGE_WRITER_PROCESSES}"
hostname
date

export HF_LEROBOT_HOME

cd "${OPENPI_REPO}"
"${OPENPI_PYTHON}" examples/libero/convert_libero_hdf5_to_lerobot.py \
  --dataset-root "${OUT_HDF5_ROOT}" \
  --mode "${LEROBOT_MODE}" \
  --repo-id "${REPO_ID}" \
  --image-size "${RENDER_RES}" \
  --image-writer-threads "${IMAGE_WRITER_THREADS}" \
  --image-writer-processes "${IMAGE_WRITER_PROCESSES}"

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
print("phase2 metadata OK")
PY

echo "Phase 2 conversion complete."
