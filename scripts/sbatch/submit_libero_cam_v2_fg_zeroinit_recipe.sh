#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/scratch/yp2841/cache/lerobot}
REPO_ID=${REPO_ID:-glbreeze/libero_cam_v2}

echo "Submitting fg + zero-init ray-embed recipe for ${REPO_ID}"
echo "repo root: ${REPO_ROOT}"
echo "hf lerobot home: ${HF_LEROBOT_HOME}"

norm_job=$(
  sbatch --parsable \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",HF_LEROBOT_HOME="${HF_LEROBOT_HOME}",REPO_ID="${REPO_ID}" \
    "${SCRIPT_DIR}/compute_norm_stats_libero_cam_v2.sbatch"
)

cache_job=$(
  sbatch --parsable \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",HF_LEROBOT_HOME="${HF_LEROBOT_HOME}",REPO_ID="${REPO_ID}" \
    "${SCRIPT_DIR}/cache_pi3x_libero_cam_v2_fullres.sbatch"
)

stage1_job=$(
  sbatch --parsable \
    --dependency="afterok:${norm_job}:${cache_job}" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",HF_LEROBOT_HOME="${HF_LEROBOT_HOME}" \
    "${SCRIPT_DIR}/train_pi0_libero_cam_v2_prope_ray_view_distill_fullres_stage1_l40s.sbatch"
)

stage2_job=$(
  sbatch --parsable \
    --dependency="afterok:${stage1_job}" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",HF_LEROBOT_HOME="${HF_LEROBOT_HOME}" \
    "${SCRIPT_DIR}/train_pi0_libero_cam_v2_prope_ray_view_distill_fullres_stage2_zeroinit_l40s.sbatch"
)

printf 'norm=%s\ncache=%s\nstage1=%s\nstage2=%s\n' "${norm_job}" "${cache_job}" "${stage1_job}" "${stage2_job}"
