#!/bin/bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Please source this script: source scripts/env/activate_env.sh" >&2
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
GEO_ROOT=$(cd -- "${REPO_ROOT}/.." && pwd)
LOCAL_VENV="${REPO_ROOT}/.venv"
WORKSPACE_ROOT=$(cd -- "${GEO_ROOT}/.." && pwd)

find_venv_to_use() {
  local candidate
  local -a candidates=("${LOCAL_VENV}")

  if [[ -n "${OPENPI_PARENT_VENV:-}" ]]; then
    candidates+=("${OPENPI_PARENT_VENV}")
  fi

  candidates+=(
    "${WORKSPACE_ROOT}/.venv"
    "${WORKSPACE_ROOT}/TFP/.venv"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  printf '%s\n' "${candidates[@]}" >&2
  return 1
}

if ! VENV_TO_USE=$(find_venv_to_use); then
  echo "Missing virtualenv. Checked the paths above." >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${VENV_TO_USE}/bin/activate"

export OPENPI_CAM_ROOT="${REPO_ROOT}"
export OPENPI_GEO_ROOT="${GEO_ROOT}"
export OPENPI_PI0_BASE_DIR="${GEO_ROOT}/pi0_base"
export OPENPI_PI0_LIBERO_NORM_DIR="${GEO_ROOT}/pi0_libero"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${GEO_ROOT}}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_ALIAS_DIR="${HF_LEROBOT_HOME}/glbreeze"
DATASET_ALIAS_PATH="${DATASET_ALIAS_DIR}/libero"
DATASET_SOURCE_PATH="${GEO_ROOT}/libero"

mkdir -p "${DATASET_ALIAS_DIR}"
if [[ -L "${DATASET_ALIAS_PATH}" ]]; then
  CURRENT_TARGET=$(readlink "${DATASET_ALIAS_PATH}")
  if [[ "${CURRENT_TARGET}" != "${DATASET_SOURCE_PATH}" ]]; then
    rm "${DATASET_ALIAS_PATH}"
    ln -s "${DATASET_SOURCE_PATH}" "${DATASET_ALIAS_PATH}"
  fi
elif [[ ! -e "${DATASET_ALIAS_PATH}" ]]; then
  ln -s "${DATASET_SOURCE_PATH}" "${DATASET_ALIAS_PATH}"
elif [[ "${DATASET_ALIAS_PATH}" != "${DATASET_SOURCE_PATH}" ]]; then
  echo "Dataset alias path already exists and is not a symlink: ${DATASET_ALIAS_PATH}" >&2
  return 1 2>/dev/null || exit 1
fi
