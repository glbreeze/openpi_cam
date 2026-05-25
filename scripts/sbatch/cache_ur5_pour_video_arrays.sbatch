#!/bin/bash
#SBATCH --job-name=cache_ur5_pour_arrays
#SBATCH --output=/scratch/%u/openpi_cam/log/ur5_video_array_cache/slurm-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/ur5_video_array_cache/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_69_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
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

if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(resolve_repo_root || true)"
fi
if [[ -z "${REPO_ROOT}" && -d "/scratch/${USER}/openpi_cam" ]]; then
  REPO_ROOT="/scratch/${USER}/openpi_cam"
fi
if [[ -z "${REPO_ROOT}" ]]; then
  echo "Unable to locate repo root." >&2
  exit 1
fi

export OPENPI_GEO_ROOT="${OPENPI_GEO_ROOT:-/scratch/${USER}}"
export DATASET_DIR="${DATASET_DIR:-${OPENPI_GEO_ROOT}/real_robot_data/ur5_place_and_pour_nuts_camera_shifts}"
export ARRAY_CACHE_ROOT="${ARRAY_CACHE_ROOT:-${DATASET_DIR}_array_cache}"
export WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"

mkdir -p "/scratch/${USER}/openpi_cam/log/ur5_video_array_cache" "${ARRAY_CACHE_ROOT}"

if [[ ! -f "${DATASET_DIR}/meta/info.json" || ! -d "${DATASET_DIR}/videos" ]]; then
  echo "Missing LeRobot pour dataset under ${DATASET_DIR}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

echo "===== UR5 POUR VIDEO ARRAY CACHE ====="
echo "repo root: ${REPO_ROOT}"
echo "dataset: ${DATASET_DIR}"
echo "array cache: ${ARRAY_CACHE_ROOT}"
echo "workers: ${WORKERS}"
echo "views: observation.images.context_left_rgb observation.images.wrist_right_rgb"
hostname
python -V

python scripts/cache_lerobot_video_arrays.py \
  --dataset-root "${DATASET_DIR}" \
  --cache-root "${ARRAY_CACHE_ROOT}" \
  --video-key observation.images.context_left_rgb \
  --video-key observation.images.wrist_right_rgb \
  --workers "${WORKERS}"

python - <<'PY'
import os
from pathlib import Path

dataset_dir = Path(os.environ["DATASET_DIR"])
cache_root = Path(os.environ["ARRAY_CACHE_ROOT"])
episode_count = sum(1 for line in (dataset_dir / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip())
views = (
    "observation.images.context_left_rgb",
    "observation.images.wrist_right_rgb",
)
for view in views:
    files = sorted((cache_root / "videos/chunk-000" / view).glob("episode_*.npy"))
    if len(files) != episode_count:
        raise SystemExit(f"{view}: expected {episode_count} npy files, found {len(files)} under {cache_root}")
print(f"array cache OK root={cache_root} episodes={episode_count} views={views}")
PY

echo "UR5 pour video array cache complete: ${ARRAY_CACHE_ROOT}"
