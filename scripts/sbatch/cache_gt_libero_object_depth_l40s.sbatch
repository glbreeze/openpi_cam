#!/bin/bash
#SBATCH --job-name=gt_cache_libero_object
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_gt_cache/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_gt_cache/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_637_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yp2841/geometry-vla/openpi_cam}
GEO_ROOT=${GEO_ROOT:-/scratch/yp2841/geometry-vla}
SRC_DIR=${SRC_DIR:-${GEO_ROOT}/libero_raw_datasets/libero_object}
LIBERO_CAMERA_REPO=${LIBERO_CAMERA_REPO:-${GEO_ROOT}/LIBERO-Camera}
LIBERO_CONDA_ENV=${LIBERO_CONDA_ENV:-/scratch/yp2841/.conda-envs/openvla-oft}
CONDA_SH=${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}
OPENPI_PYTHON=${OPENPI_PYTHON:-${REPO_ROOT}/.venv/bin/python}

DEPTH_HDF5_ROOT=${DEPTH_HDF5_ROOT:-/scratch/yp2841/data/libero_object_depth_hdf5}
LIBERO_CFG_DIR=${LIBERO_CFG_DIR:-/scratch/yp2841/data/libero_object_depth_cfg}
GT_CACHE_ROOT=${GT_CACHE_ROOT:-${GEO_ROOT}/.cache/openpi/gt_point_targets_224/libero_object_cam_v3}
RENDER_RES=${RENDER_RES:-256}
TARGET_RES=${TARGET_RES:-224}
OUTPUT_RES=${OUTPUT_RES:-224}
SETTLE_STEPS=${SETTLE_STEPS:-10}
MAX_FILES=${MAX_FILES:-0}
MAX_EPISODES_PER_FILE=${MAX_EPISODES_PER_FILE:-0}

export REPO_ROOT
export GEO_ROOT
export SRC_DIR
export LIBERO_CAMERA_REPO
export DEPTH_HDF5_ROOT
export GT_CACHE_ROOT

JOB_TMP_BASE="${SLURM_TMPDIR:-/scratch/yp2841/tmp}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/gt_cache_libero_object_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/scratch/yp2841/.cache}
export HF_HOME=${HF_HOME:-/scratch/yp2841/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/scratch/yp2841/.cache/torch}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/scratch/yp2841/.cache/pip}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONUNBUFFERED=1

mkdir -p \
  "${TMPDIR}" \
  "${DEPTH_HDF5_ROOT}/libero_object" \
  "${LIBERO_CFG_DIR}" \
  "${GT_CACHE_ROOT}" \
  "${REPO_ROOT}/log/libero_gt_cache"

cat > "${LIBERO_CFG_DIR}/config.yaml" <<EOF
assets: ${LIBERO_CAMERA_REPO}/libero/libero/assets
bddl_files: ${LIBERO_CAMERA_REPO}/libero/libero/bddl_files
benchmark_root: ${LIBERO_CAMERA_REPO}/libero/libero
datasets: ${DEPTH_HDF5_ROOT}
init_states: ${LIBERO_CAMERA_REPO}/libero/libero/init_files
EOF

echo "===== LIBERO Object GT cache job ====="
echo "job id: ${SLURM_JOB_ID:-manual}"
echo "host: $(hostname)"
echo "repo root: ${REPO_ROOT}"
echo "source hdf5: ${SRC_DIR}"
echo "depth hdf5 root: ${DEPTH_HDF5_ROOT}/libero_object"
echo "gt cache root: ${GT_CACHE_ROOT}"
echo "libero repo: ${LIBERO_CAMERA_REPO}"
echo "render res: ${RENDER_RES}"
echo "target/output res: ${TARGET_RES}/${OUTPUT_RES}"
echo "max files: ${MAX_FILES}"
echo "max episodes per file: ${MAX_EPISODES_PER_FILE}"
date

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "Missing source HDF5 dir: ${SRC_DIR}" >&2
  exit 1
fi
if [[ ! -x "${OPENPI_PYTHON}" ]]; then
  echo "Missing OPENPI_PYTHON: ${OPENPI_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Missing CONDA_SH: ${CONDA_SH}" >&2
  exit 1
fi

mapfile -t SRC_FILES < <(find "${SRC_DIR}" -maxdepth 1 -name '*_demo.hdf5' | sort)
if [[ "${#SRC_FILES[@]}" -lt 1 ]]; then
  echo "No *_demo.hdf5 files under ${SRC_DIR}" >&2
  exit 1
fi
if [[ "${MAX_FILES}" -gt 0 && "${#SRC_FILES[@]}" -gt "${MAX_FILES}" ]]; then
  SRC_FILES=("${SRC_FILES[@]:0:${MAX_FILES}}")
fi

echo "Phase 1: regenerate depth HDF5 for ${#SRC_FILES[@]} source files"
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${LIBERO_CONDA_ENV}"
pushd "${LIBERO_CAMERA_REPO}" > /dev/null
for src in "${SRC_FILES[@]}"; do
  echo
  echo "[depth regen] $(date +%F_%T) $(basename "${src}")"
  eps_args=()
  if [[ "${MAX_EPISODES_PER_FILE}" -gt 0 ]]; then
    eps_args=(--max-episodes "${MAX_EPISODES_PER_FILE}")
  fi
  LIBERO_CONFIG_PATH="${LIBERO_CFG_DIR}" python scripts/create_dataset.py \
    --demo-file "${src}" \
    --use-camera-obs \
    --use-depth \
    --render-resolution "${RENDER_RES}" \
    --filter-noops \
    --settle-steps "${SETTLE_STEPS}" \
    --resume \
    "${eps_args[@]}"
done
popd > /dev/null
conda deactivate

echo
echo "Phase 1 validation: checking stored depth datasets"
"${OPENPI_PYTHON}" - <<'PY'
import os
from pathlib import Path
import h5py

root = Path(os.environ["DEPTH_HDF5_ROOT"]) / "libero_object"
files = sorted(root.glob("*.hdf5"))
if not files:
    raise SystemExit(f"No regenerated HDF5 files found under {root}")
episodes = 0
frames = 0
for path in files:
    with h5py.File(path, "r") as f:
        for ep in sorted(f["data"].keys()):
            obs = f[f"data/{ep}/obs"]
            missing = [k for k in ("agentview_depth", "eye_in_hand_depth") if k not in obs]
            if missing:
                raise SystemExit(f"{path}/{ep} missing depth datasets: {missing}")
            n = int(obs["agentview_depth"].shape[0])
            episodes += 1
            frames += n
print(f"depth hdf5 OK files={len(files)} episodes={episodes} frames={frames}")
PY

echo
echo "Phase 2: build GT point target cache"
pushd "${REPO_ROOT}" > /dev/null
PYTHONPATH=src "${OPENPI_PYTHON}" scripts/cache_libero_gt_point_targets.py \
  --dataset-root "${DEPTH_HDF5_ROOT}/libero_object" \
  --output-root "${GT_CACHE_ROOT}" \
  --libero-repo "${LIBERO_CAMERA_REPO}" \
  --render-resolution "${RENDER_RES}" \
  --target-resolution "${TARGET_RES}" \
  --output-resolution "${OUTPUT_RES}" \
  --resume
popd > /dev/null

echo
echo "Phase 2 validation: checking GT cache layout"
"${OPENPI_PYTHON}" - <<'PY'
import os
from pathlib import Path
import numpy as np

root = Path(os.environ["GT_CACHE_ROOT"])
agent = sorted((root / "agent").glob("episode_*.npz"))
wrist = sorted((root / "wrist").glob("episode_*.npz"))
if not agent or not wrist:
    raise SystemExit(f"Missing GT cache files under {root}")
if len(agent) != len(wrist):
    raise SystemExit(f"Mismatched agent/wrist cache counts: {len(agent)} vs {len(wrist)}")
sample = np.load(agent[0])
expected = {"xy", "log_z", "conf"}
if set(sample.files) != expected:
    raise SystemExit(f"Unexpected keys in {agent[0]}: {sample.files}")
print(f"gt cache OK root={root} episodes={len(agent)} sample_xy={sample['xy'].shape} sample_log_z={sample['log_z'].shape} sample_conf={sample['conf'].shape}")
PY

date
echo "GT cache job complete."
