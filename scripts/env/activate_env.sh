#!/bin/bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Please source this script: source scripts/env/activate_env.sh" >&2
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
DEFAULT_GEO_ROOT=$(cd -- "${REPO_ROOT}/.." && pwd)
GEO_ROOT="${OPENPI_GEO_ROOT:-${DEFAULT_GEO_ROOT}}"
LOCAL_VENV="${REPO_ROOT}/.venv"
WORKSPACE_ROOT=$(cd -- "${DEFAULT_GEO_ROOT}/.." && pwd)

find_venv_to_use() {
  local candidate
  local -a candidates=()

  if [[ -n "${OPENPI_PARENT_VENV:-}" ]]; then
    candidates+=("${OPENPI_PARENT_VENV}")
  fi

  candidates+=("${LOCAL_VENV}")

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
export OPENPI_PI0_BASE_DIR="${OPENPI_PI0_BASE_DIR:-${GEO_ROOT}/pi0_base}"
export OPENPI_PI0_LIBERO_NORM_DIR="${OPENPI_PI0_LIBERO_NORM_DIR:-${GEO_ROOT}/pi0_libero}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${GEO_ROOT}/.huggingface/lerobot}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"

ensure_symlink_target() {
  local link_path=$1
  local target_path=$2

  if [[ -L "${link_path}" ]]; then
    local current_target
    current_target=$(readlink -f "${link_path}")
    local desired_target
    desired_target=$(readlink -f "${target_path}")
    if [[ "${current_target}" != "${desired_target}" ]]; then
      rm "${link_path}"
      ln -s "${target_path}" "${link_path}"
    fi
  elif [[ ! -e "${link_path}" ]]; then
    ln -s "${target_path}" "${link_path}"
  elif [[ "$(readlink -f "${link_path}")" != "$(readlink -f "${target_path}")" ]]; then
    echo "Path already exists and is not the expected symlink target: ${link_path}" >&2
    return 1
  fi
}

DATASET_ALIAS_DIR="${HF_LEROBOT_HOME}/glbreeze"
DATASET_ALIAS_PATH="${DATASET_ALIAS_DIR}/libero"
DATASET_SOURCE_PATH="${GEO_ROOT}/libero"

mkdir -p "${DATASET_ALIAS_DIR}"
if ! ensure_symlink_target "${DATASET_ALIAS_PATH}" "${DATASET_SOURCE_PATH}"; then
  return 1 2>/dev/null || exit 1
fi

if [[ -n "${DATASET_DIR:-}" && -n "${DATASET_REPO_ID:-}" ]]; then
  LOCAL_DATASET_PATH="${HF_LEROBOT_HOME}/${DATASET_REPO_ID}"
  mkdir -p "$(dirname "${LOCAL_DATASET_PATH}")"
  if ! ensure_symlink_target "${LOCAL_DATASET_PATH}" "${DATASET_DIR}"; then
    return 1 2>/dev/null || exit 1
  fi
fi
