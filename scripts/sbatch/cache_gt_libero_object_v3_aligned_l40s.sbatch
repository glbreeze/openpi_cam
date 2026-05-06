#!/bin/bash
#SBATCH --job-name=gt_cache_obj_v3_align
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_gt_cache/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_gt_cache/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_637_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yp2841/geometry-vla/openpi_cam}
GEO_ROOT=${GEO_ROOT:-/scratch/yp2841/geometry-vla}
OPENPI_PYTHON=${OPENPI_PYTHON:-${REPO_ROOT}/.venv/bin/python}
LIBERO_CAMERA_REPO=${LIBERO_CAMERA_REPO:-${GEO_ROOT}/LIBERO-Camera}
LIBERO_CONDA_ENV=${LIBERO_CONDA_ENV:-/scratch/yp2841/.conda-envs/openvla-oft}
CONDA_SH=${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}

LEROBOT_ROOT=${LEROBOT_ROOT:-/scratch/yp2841/.huggingface/lerobot/glbreeze/libero_object_cam_v3}
RAW_HDF5_ROOT=${RAW_HDF5_ROOT:-${GEO_ROOT}/libero_raw_datasets/libero_object}
MANIFEST_PATH=${MANIFEST_PATH:-/scratch/yp2841/data/libero_object_cam_v3_gt_manifest.jsonl}
DEPTH_HDF5_ROOT=${DEPTH_HDF5_ROOT:-/scratch/yp2841/data/libero_object_cam_v3_depth_aligned}
LIBERO_CFG_DIR=${LIBERO_CFG_DIR:-/scratch/yp2841/data/libero_object_cam_v3_depth_aligned_cfg}
GT_CACHE_ROOT=${GT_CACHE_ROOT:-${GEO_ROOT}/.cache/openpi/gt_point_targets_224/libero_object_cam_v3_aligned}

RENDER_RES=${RENDER_RES:-256}
TARGET_RES=${TARGET_RES:-224}
OUTPUT_RES=${OUTPUT_RES:-224}
SETTLE_STEPS=${SETTLE_STEPS:-10}
ACTION_DECIMALS=${ACTION_DECIMALS:-6}
RESUME=${RESUME:-0}

JOB_TMP_BASE="${SLURM_TMPDIR:-/scratch/yp2841/tmp}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/gt_cache_obj_v3_align_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/scratch/yp2841/.cache}
export HF_HOME=${HF_HOME:-/scratch/yp2841/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/scratch/yp2841/.cache/torch}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/scratch/yp2841/.cache/pip}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONUNBUFFERED=1
export LIBERO_CONFIG_PATH="${LIBERO_CFG_DIR}"

mkdir -p \
  "${TMPDIR}" \
  "$(dirname "${MANIFEST_PATH}")" \
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

echo "===== LIBERO Object v3 aligned GT cache job ====="
echo "job id: ${SLURM_JOB_ID:-manual}"
echo "host: $(hostname)"
echo "repo root: ${REPO_ROOT}"
echo "lerobot root: ${LEROBOT_ROOT}"
echo "raw hdf5 root: ${RAW_HDF5_ROOT}"
echo "manifest: ${MANIFEST_PATH}"
echo "aligned depth root: ${DEPTH_HDF5_ROOT}/libero_object"
echo "aligned gt cache root: ${GT_CACHE_ROOT}"
echo "render/target/output res: ${RENDER_RES}/${TARGET_RES}/${OUTPUT_RES}"
date

if [[ ! -x "${OPENPI_PYTHON}" ]]; then
  echo "Missing OPENPI_PYTHON: ${OPENPI_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Missing CONDA_SH: ${CONDA_SH}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

echo
echo "Phase 0: build exact LeRobot-v3 manifest"
"${OPENPI_PYTHON}" scripts/build_libero_v3_gt_manifest.py \
  --dataset-root "${LEROBOT_ROOT}" \
  --raw-root "${RAW_HDF5_ROOT}" \
  --output "${MANIFEST_PATH}" \
  --action-decimals "${ACTION_DECIMALS}"

resume_flag=()
if [[ "${RESUME}" == "1" || "${RESUME,,}" == "true" ]]; then
  resume_flag=(--resume)
fi
cache_resume_flag=()
if [[ "${RESUME}" == "1" || "${RESUME,,}" == "true" ]]; then
  cache_resume_flag=(--resume)
fi

echo
echo "Phase 1: replay aligned depth HDF5 from manifest"
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${LIBERO_CONDA_ENV}"
pushd "${LIBERO_CAMERA_REPO}" > /dev/null
python "${REPO_ROOT}/scripts/replay_libero_depth_from_manifest.py" \
  --manifest "${MANIFEST_PATH}" \
  --raw-root "${RAW_HDF5_ROOT}" \
  --output-root "${DEPTH_HDF5_ROOT}" \
  --libero-camera-root "${LIBERO_CAMERA_REPO}" \
  --render-resolution "${RENDER_RES}" \
  --settle-steps "${SETTLE_STEPS}" \
  --action-decimals "${ACTION_DECIMALS}" \
  "${resume_flag[@]}"
popd > /dev/null
conda deactivate

echo
echo "Phase 2: build aligned GT point target cache"
PYTHONPATH=src "${OPENPI_PYTHON}" scripts/cache_libero_gt_point_targets.py \
  --dataset-root "${DEPTH_HDF5_ROOT}/libero_object" \
  --output-root "${GT_CACHE_ROOT}" \
  --libero-repo "${LIBERO_CAMERA_REPO}" \
  --render-resolution "${RENDER_RES}" \
  --target-resolution "${TARGET_RES}" \
  --output-resolution "${OUTPUT_RES}" \
  "${cache_resume_flag[@]}"

echo
echo "Phase 3: validate aligned GT cache"
"${OPENPI_PYTHON}" scripts/validate_libero_gt_cache_alignment.py \
  --dataset-root "${LEROBOT_ROOT}" \
  --manifest "${MANIFEST_PATH}" \
  --depth-root "${DEPTH_HDF5_ROOT}" \
  --cache-root "${GT_CACHE_ROOT}" \
  --action-decimals "${ACTION_DECIMALS}"

date
echo "LIBERO Object v3 aligned GT cache job complete."
