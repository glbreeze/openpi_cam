#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
VENV_DIR="${REPO_ROOT}/.venv"
WORKSPACE_ROOT=$(cd -- "${REPO_ROOT}/../.." && pwd)
CACHE_ROOT="${REPO_ROOT}/.cache"
XDG_CACHE_ROOT="${CACHE_ROOT}/xdg"
UV_CACHE_ROOT="${CACHE_ROOT}/uv"
TMP_ROOT="${CACHE_ROOT}/tmp"
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
SETUP_MODE=${SETUP_MODE:-parent-site-packages}

find_parent_venv() {
  local candidate
  local -a candidates=()

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

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed or not on PATH." >&2
  exit 1
fi

install_local_transformers_overlay() {
  local local_python local_site_packages source_transformers_dir local_transformers_dir parent_site_packages_file
  local parent_site_packages
  local -a py_info dist_info_sources

  local_python="${VENV_DIR}/bin/python"
  if [[ ! -x "${local_python}" ]]; then
    echo "Local virtualenv python not found at ${local_python}" >&2
    return 1
  fi

  mapfile -t py_info < <(
    "${local_python}" - <<'PY'
import importlib.util
import pathlib
import sysconfig

site_packages = pathlib.Path(sysconfig.get_paths()["purelib"]).resolve()
spec = importlib.util.find_spec("transformers")
if spec is None or spec.origin is None:
    raise SystemExit("Could not resolve transformers in this environment.")

source_dir = pathlib.Path(spec.origin).resolve().parent
print(site_packages)
print(source_dir)
for dist_info in sorted(source_dir.parent.glob("transformers-*.dist-info")):
    print(dist_info.resolve())
PY
  )

  local_site_packages=${py_info[0]:-}
  source_transformers_dir=${py_info[1]:-}
  local_transformers_dir="${local_site_packages}/transformers"
  parent_site_packages_file="${local_site_packages}/99_parent_site_packages.pth"

  if [[ -z "${local_site_packages}" || -z "${source_transformers_dir}" ]]; then
    echo "Failed to resolve local site-packages or transformers directory." >&2
    return 1
  fi

  dist_info_sources=("${py_info[@]:2}")
  if [[ -f "${parent_site_packages_file}" ]]; then
    parent_site_packages=$(head -n 1 "${parent_site_packages_file}")
    if [[ -d "${parent_site_packages}/transformers" ]]; then
      source_transformers_dir="${parent_site_packages}/transformers"
      dist_info_sources=()
      shopt -s nullglob
      for dist_info in "${parent_site_packages}"/transformers-*.dist-info; do
        dist_info_sources+=("${dist_info}")
      done
      shopt -u nullglob
    fi
  fi

  if [[ -e "${local_transformers_dir}" ]]; then
    chmod -R u+rwX "${local_transformers_dir}" || true
  fi

  shopt -s nullglob
  for local_dist_info in "${local_site_packages}"/transformers-*.dist-info; do
    chmod -R u+rwX "${local_dist_info}" || true
  done
  shopt -u nullglob

  export REPO_ROOT
  export LOCAL_SITE_PACKAGES="${local_site_packages}"
  export SOURCE_TRANSFORMERS_DIR="${source_transformers_dir}"
  export LOCAL_TRANSFORMERS_DIR="${local_transformers_dir}"

  "${local_python}" - "${dist_info_sources[@]}" <<'PY'
import os
import pathlib
import shutil
import sys

repo_root = pathlib.Path(os.environ["REPO_ROOT"]).resolve()
local_site_packages = pathlib.Path(os.environ["LOCAL_SITE_PACKAGES"]).resolve()
source_transformers_dir = pathlib.Path(os.environ["SOURCE_TRANSFORMERS_DIR"]).resolve()
local_transformers_dir = pathlib.Path(os.environ["LOCAL_TRANSFORMERS_DIR"]).resolve()
replace_root = repo_root / "src/openpi/models_pytorch/transformers_replace"
ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")


def delete_path(path: pathlib.Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path, ignore_errors=True)


if source_transformers_dir != local_transformers_dir:
    if local_transformers_dir.exists():
        delete_path(local_transformers_dir)
    shutil.copytree(source_transformers_dir, local_transformers_dir, dirs_exist_ok=True, ignore=ignore)

for dist_info in sys.argv[1:]:
    src = pathlib.Path(dist_info).resolve()
    dst = local_site_packages / src.name
    if src != dst:
        if dst.exists():
            delete_path(dst)
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)

shutil.copytree(replace_root, local_transformers_dir, dirs_exist_ok=True, ignore=ignore)
PY

  "${local_python}" - <<'PY'
import pathlib
import transformers
from transformers.models.siglip import check

transformers_path = pathlib.Path(transformers.__file__).resolve()
if not check.check_whether_transformers_replace_is_installed_correctly():
    raise SystemExit("transformers_replace validation failed after setup.")

print(f"transformers overlay ready at {transformers_path.parent}")
PY
}

setup_parent_site_packages_fallback() {
  local local_python site_packages python_tag parent_site_packages parent_venv

  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Local virtualenv was not created, cannot enable fallback mode." >&2
    return 1
  fi

  if ! parent_venv=$(find_parent_venv); then
    echo "Parent virtualenv not found. Checked the paths above." >&2
    return 1
  fi

  local_python="${VENV_DIR}/bin/python"
  python_tag=$("${local_python}" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
  site_packages="${VENV_DIR}/lib/${python_tag}/site-packages"
  parent_site_packages="${parent_venv}/lib/${python_tag}/site-packages"

  if [[ ! -d "${parent_site_packages}" ]]; then
    echo "Parent site-packages not found at ${parent_site_packages}" >&2
    return 1
  fi

  mkdir -p "${site_packages}"
  printf '%s\n' "${REPO_ROOT}/src" > "${site_packages}/00_openpi_local_src.pth"
  printf '%s\n' "${REPO_ROOT}/packages/openpi-client/src" > "${site_packages}/01_openpi_client_local_src.pth"
  printf '%s\n' "${parent_site_packages}" > "${site_packages}/99_parent_site_packages.pth"

  echo
  echo "uv sync hit the evdev kernel-header build issue."
  echo "Configured local venv to reuse parent site-packages from ${parent_site_packages}"
  echo "This is enough for training in this repo, while keeping a repo-local .venv entrypoint."
}

ensure_local_venv_exists() {
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    return 0
  fi

  uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
}

echo "Creating or updating local virtualenv at ${VENV_DIR}"
cd "${REPO_ROOT}"

export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
export GIT_LFS_SKIP_SMUDGE=1
export XDG_CACHE_HOME="${XDG_CACHE_ROOT}"
export UV_CACHE_DIR="${UV_CACHE_ROOT}"
export TMPDIR="${TMP_ROOT}"

mkdir -p "${XDG_CACHE_HOME}" "${UV_CACHE_DIR}" "${TMPDIR}"

ensure_local_venv_exists

if [[ "${SETUP_MODE}" == "parent-site-packages" ]]; then
  setup_parent_site_packages_fallback
  install_local_transformers_overlay
elif uv sync --python "${PYTHON_VERSION}"; then
  install_local_transformers_overlay
else
  setup_parent_site_packages_fallback
  install_local_transformers_overlay
fi

echo
echo "Local venv is ready."
echo "Use it with:"
echo "  source ${REPO_ROOT}/scripts/env/activate_env.sh"
