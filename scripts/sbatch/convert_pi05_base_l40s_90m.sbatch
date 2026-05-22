#!/bin/bash
#SBATCH --job-name=pi05_base_conv
#SBATCH --output=/scratch/%u/openpi_cam/log/pi05_base_convert/slurm-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/pi05_base_convert/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --account=torch_pr_69_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
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

source "${REPO_ROOT}/scripts/env/activate_env.sh"

export OPENPI_GEO_ROOT="${OPENPI_GEO_ROOT:-/scratch/${USER}}"
export PI05_CACHE_ROOT="${PI05_CACHE_ROOT:-/scratch/${USER}/.cache/openpi/openpi-assets/checkpoints}"
export PI05_JAX_CHECKPOINT_DIR="${PI05_JAX_CHECKPOINT_DIR:-${PI05_CACHE_ROOT}/pi05_base}"
export PI05_PYTORCH_OUTPUT_DIR="${PI05_PYTORCH_OUTPUT_DIR:-${OPENPI_GEO_ROOT}/pi05_base}"
export PI05_CONFIG_NAME="${PI05_CONFIG_NAME:-pi05_libero}"

export GSUTIL_STATE_DIR="${GSUTIL_STATE_DIR:-/scratch/${USER}/.gsutil}"
export TMPDIR="${TMPDIR:-${GSUTIL_STATE_DIR}/tmp}"
export BOTO_CONFIG="${BOTO_CONFIG:-/scratch/${USER}/gsutil-scratch.boto}"

mkdir -p \
  "/scratch/${USER}/openpi_cam/log/pi05_base_convert" \
  "${PI05_CACHE_ROOT}" \
  "${GSUTIL_STATE_DIR}/tracker-files" \
  "${TMPDIR}"

cat > "${BOTO_CONFIG}" <<EOF
[GSUtil]
state_dir = ${GSUTIL_STATE_DIR}
parallel_process_count = 1
parallel_thread_count = 1
check_hashes = if_fast_else_skip
EOF

echo "repo root: ${REPO_ROOT}"
echo "pi05 cache root: ${PI05_CACHE_ROOT}"
echo "pi05 jax checkpoint dir: ${PI05_JAX_CHECKPOINT_DIR}"
echo "pi05 pytorch output dir: ${PI05_PYTORCH_OUTPUT_DIR}"
echo "config name: ${PI05_CONFIG_NAME}"
echo "tmpdir: ${TMPDIR}"
echo "boto config: ${BOTO_CONFIG}"

if [[ ! -f "${PI05_JAX_CHECKPOINT_DIR}/params/_METADATA" ]]; then
  echo "Downloading pi05_base JAX checkpoint to scratch..."
  gsutil cp -r gs://openpi-assets/checkpoints/pi05_base "${PI05_CACHE_ROOT}/"
fi

echo "Checkpoint layout:"
find "${PI05_JAX_CHECKPOINT_DIR}" -maxdepth 3 -printf '%y %p\n' | sort

rm -rf "${PI05_PYTORCH_OUTPUT_DIR}"
mkdir -p "${PI05_PYTORCH_OUTPUT_DIR}"

echo "Starting JAX -> PyTorch conversion..."
time uv run python "${REPO_ROOT}/examples/convert_jax_model_to_pytorch.py" \
  --config_name "${PI05_CONFIG_NAME}" \
  --checkpoint_dir "${PI05_JAX_CHECKPOINT_DIR}" \
  --output_path "${PI05_PYTORCH_OUTPUT_DIR}"

echo "Converted checkpoint layout:"
find "${PI05_PYTORCH_OUTPUT_DIR}" -maxdepth 2 -printf '%y %p\n' | sort
