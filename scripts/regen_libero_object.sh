#!/usr/bin/env bash
#
# End-to-end regen for the LIBERO-Object suite using the new pipeline:
#   Phase 1 (libero conda env):
#       source HDF5  →  regenerated HDF5 with extrinsics/intrinsics, no-ops
#                       filtered, success-gated, 256-rendered, (obs_t, action_t).
#   Phase 2 (uv-managed openpi env):
#       regenerated HDF5  →  LeRobot v2.1 dataset under $HF_LEROBOT_HOME.
#
# Companion to LIBERO-Camera/DATASET_CALIBRATION.md.
#
# Override any setting via env var, e.g.:
#   REPO_ID=glbreeze/libero_object_cam_v2 OUT_HDF5_ROOT=/data/libero_v2 \
#       bash scripts/regen_libero_object.sh
#
# Refuses to overwrite an existing OUT_HDF5_ROOT/<suite>/ tree or an existing
# LeRobot repo at $HF_LEROBOT_HOME/$REPO_ID. Clear those out first if you
# really want to re-run.

set -euo pipefail

# ---------------------------------------------------------------------------- #
# Configuration                                                                 #
# ---------------------------------------------------------------------------- #
SUITE="${SUITE:-libero_object}"
SRC_DIR="${SRC_DIR:-/home/asus/Research/datasets/libero_datasets/${SUITE}}"
OUT_HDF5_ROOT="${OUT_HDF5_ROOT:-/home/asus/Research/datasets/libero_cam_v2}"
LIBERO_CFG_DIR="${LIBERO_CFG_DIR:-/home/asus/Research/datasets/libero_cam_v2_cfg}"
REPO_ID="${REPO_ID:-glbreeze/libero_object_cam_v2}"
RENDER_RES="${RENDER_RES:-256}"
SETTLE_STEPS="${SETTLE_STEPS:-10}"
MAX_EPISODES="${MAX_EPISODES:-0}"     # 0 = all demos per source HDF5
LIBERO_CAMERA_REPO="${LIBERO_CAMERA_REPO:-/home/asus/Research/LIBERO-Camera}"
OPENPI_REPO="${OPENPI_REPO:-/home/asus/Research/openpi}"
LIBERO_CONDA_ENV="${LIBERO_CONDA_ENV:-libero}"
CONDA_SH="${CONDA_SH:-}"
OPENPI_PYTHON="${OPENPI_PYTHON:-}"
SKIP_PHASE1="${SKIP_PHASE1:-0}"       # set to 1 to only run conversion
SKIP_PHASE2="${SKIP_PHASE2:-0}"       # set to 1 to only run regen
LEROBOT_MODE="${LEROBOT_MODE:-original_only}"   # or all_views (camvar episodes)

resolve_conda_sh() {
    if [[ -n "$CONDA_SH" ]]; then
        if [[ -f "$CONDA_SH" ]]; then
            printf '%s\n' "$CONDA_SH"
            return 0
        fi
        echo "ERROR: CONDA_SH is set but does not exist: $CONDA_SH" >&2
        return 1
    fi

    if command -v conda >/dev/null 2>&1; then
        local conda_base
        conda_base="$(conda info --base 2>/dev/null || true)"
        if [[ -n "$conda_base" && -f "$conda_base/etc/profile.d/conda.sh" ]]; then
            printf '%s\n' "$conda_base/etc/profile.d/conda.sh"
            return 0
        fi
    fi

    local candidate
    for candidate in \
        /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh \
        /share/apps/anaconda3/etc/profile.d/conda.sh \
        /home/asus/miniconda3/etc/profile.d/conda.sh
    do
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    echo "ERROR: could not find conda.sh. Set CONDA_SH=/path/to/conda.sh." >&2
    return 1
}

resolve_openpi_python() {
    if [[ -n "$OPENPI_PYTHON" ]]; then
        if [[ -x "$OPENPI_PYTHON" ]]; then
            printf '%s\n' "$OPENPI_PYTHON"
            return 0
        fi
        echo "ERROR: OPENPI_PYTHON is set but not executable: $OPENPI_PYTHON" >&2
        return 1
    fi

    if [[ -x "$OPENPI_REPO/.venv/bin/python" ]]; then
        printf '%s\n' "$OPENPI_REPO/.venv/bin/python"
        return 0
    fi

    return 1
}

run_openpi_python() {
    if [[ -n "${OPENPI_PYTHON_BIN:-}" ]]; then
        "$OPENPI_PYTHON_BIN" "$@"
        return 0
    fi
    if command -v uv >/dev/null 2>&1; then
        uv run --project "$OPENPI_REPO" python "$@"
        return 0
    fi

    echo "ERROR: neither OPENPI_PYTHON/.venv nor uv is available for openpi." >&2
    return 1
}

echo "================================================================="
echo " LIBERO-Object regeneration"
echo "================================================================="
echo " SUITE              = $SUITE"
echo " SRC_DIR            = $SRC_DIR"
echo " OUT_HDF5_ROOT      = $OUT_HDF5_ROOT"
echo " LIBERO_CFG_DIR     = $LIBERO_CFG_DIR"
echo " REPO_ID            = $REPO_ID"
echo " RENDER_RES         = $RENDER_RES"
echo " SETTLE_STEPS       = $SETTLE_STEPS"
echo " MAX_EPISODES       = $MAX_EPISODES (0 = all)"
echo " LIBERO_CAMERA_REPO = $LIBERO_CAMERA_REPO"
echo " OPENPI_REPO        = $OPENPI_REPO"
echo " CONDA_SH           = ${CONDA_SH:-<auto>}"
echo " OPENPI_PYTHON      = ${OPENPI_PYTHON:-<auto>}"
echo " SKIP_PHASE1        = $SKIP_PHASE1"
echo " SKIP_PHASE2        = $SKIP_PHASE2"
echo " LEROBOT_MODE       = $LEROBOT_MODE"
echo "================================================================="

# ---------------------------------------------------------------------------- #
# Sanity / safety                                                               #
# ---------------------------------------------------------------------------- #
if [[ ! -d "$SRC_DIR" ]]; then
    echo "ERROR: source dir does not exist: $SRC_DIR" >&2
    exit 1
fi
if ! ls "$SRC_DIR"/*_demo.hdf5 >/dev/null 2>&1; then
    echo "ERROR: no *_demo.hdf5 files under $SRC_DIR" >&2
    exit 1
fi

OUT_SUITE_DIR="$OUT_HDF5_ROOT/$SUITE"
if [[ "$SKIP_PHASE1" -eq 0 ]]; then
    if [[ -d "$OUT_SUITE_DIR" ]] && [[ -n "$(ls -A "$OUT_SUITE_DIR" 2>/dev/null || true)" ]]; then
        echo "WARNING: $OUT_SUITE_DIR is non-empty."
        echo "         create_dataset.py is invoked with --resume so existing files"
        echo "         will be SKIPPED (not overwritten). To force a fresh build,"
        echo "         remove $OUT_SUITE_DIR first."
        echo
    fi
fi

# Refuse if LeRobot repo already exists; the converter would silently rmtree it.
if [[ "$SKIP_PHASE2" -eq 0 ]]; then
    if [[ -n "${HF_LEROBOT_HOME:-}" ]]; then
        LR_HOME="$HF_LEROBOT_HOME"
    else
        OPENPI_PYTHON_BIN="$(resolve_openpi_python || true)"
        LR_HOME="$(run_openpi_python -c \
            "from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME; print(HF_LEROBOT_HOME)" \
            2>/dev/null | tail -1)"
        if [[ -z "$LR_HOME" ]]; then
            echo "ERROR: could not resolve HF_LEROBOT_HOME. Set HF_LEROBOT_HOME explicitly or fix the openpi env." >&2
            exit 1
        fi
    fi
    LR_REPO_DIR="$LR_HOME/$REPO_ID"
    if [[ -e "$LR_REPO_DIR" ]]; then
        echo "ERROR: target LeRobot repo already exists: $LR_REPO_DIR" >&2
        echo "       The converter would rmtree it. Pick a different REPO_ID, or" >&2
        echo "       remove that path explicitly first." >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------- #
# Phase 1: Regenerate HDF5                                                      #
# ---------------------------------------------------------------------------- #
if [[ "$SKIP_PHASE1" -eq 0 ]]; then
    echo
    echo "----- Phase 1: regen HDF5 -----"
    mkdir -p "$OUT_SUITE_DIR" "$LIBERO_CFG_DIR"
    cat > "$LIBERO_CFG_DIR/config.yaml" <<EOF
assets: $LIBERO_CAMERA_REPO/libero/libero/assets
bddl_files: $LIBERO_CAMERA_REPO/libero/libero/bddl_files
benchmark_root: $LIBERO_CAMERA_REPO/libero/libero
datasets: $OUT_HDF5_ROOT
init_states: $LIBERO_CAMERA_REPO/libero/libero/init_files
EOF

    CONDA_SH_PATH="$(resolve_conda_sh)"
    # shellcheck disable=SC1091
    source "$CONDA_SH_PATH"
    conda activate "$LIBERO_CONDA_ENV"

    EPS_ARGS=()
    if [[ "$MAX_EPISODES" -gt 0 ]]; then
        EPS_ARGS=(--max-episodes "$MAX_EPISODES")
    fi

    PHASE1_START=$(date +%s)
    NUM_FILES=0
    pushd "$LIBERO_CAMERA_REPO" > /dev/null
    for SRC in "$SRC_DIR"/*_demo.hdf5; do
        NUM_FILES=$((NUM_FILES + 1))
        echo
        echo "[regen $(date +%H:%M:%S)] file $NUM_FILES: $(basename "$SRC")"
        LIBERO_CONFIG_PATH="$LIBERO_CFG_DIR" MUJOCO_GL=egl python scripts/create_dataset.py \
            --demo-file "$SRC" \
            --use-camera-obs \
            --render-resolution "$RENDER_RES" \
            --filter-noops \
            --settle-steps "$SETTLE_STEPS" \
            --resume \
            "${EPS_ARGS[@]}"
    done
    popd > /dev/null

    conda deactivate
    PHASE1_SEC=$(( $(date +%s) - PHASE1_START ))
    echo
    echo "[regen done] processed $NUM_FILES source files in ${PHASE1_SEC}s"
    echo "[regen done] outputs under: $OUT_SUITE_DIR"
fi

# ---------------------------------------------------------------------------- #
# Phase 2: HDF5 → LeRobot                                                       #
# ---------------------------------------------------------------------------- #
if [[ "$SKIP_PHASE2" -eq 0 ]]; then
    echo
    echo "----- Phase 2: HDF5 → LeRobot -----"
    PHASE2_START=$(date +%s)
    pushd "$OPENPI_REPO" > /dev/null
    OPENPI_PYTHON_BIN="${OPENPI_PYTHON_BIN:-$(resolve_openpi_python || true)}"
    run_openpi_python examples/libero/convert_libero_hdf5_to_lerobot.py \
        --dataset-root "$OUT_HDF5_ROOT" \
        --mode "$LEROBOT_MODE" \
        --repo-id "$REPO_ID" \
        --image-size "$RENDER_RES"
    popd > /dev/null
    PHASE2_SEC=$(( $(date +%s) - PHASE2_START ))
    echo "[convert done] LeRobot repo at: $LR_REPO_DIR  (${PHASE2_SEC}s)"
fi

# ---------------------------------------------------------------------------- #
# Summary                                                                       #
# ---------------------------------------------------------------------------- #
echo
echo "================================================================="
echo " DONE"
echo "================================================================="
if [[ "$SKIP_PHASE2" -eq 0 ]]; then
    echo " LeRobot repo: $LR_REPO_DIR"
    echo
    echo " Next step (compute norm stats for the training config):"
    echo "   cd $OPENPI_REPO"
    echo "   # First, edit src/openpi/training/config.py so"
    echo "   # pi0_libero_cam_pytorch_prope_ray_view's repo_id and asset_id"
    echo "   # both equal '$REPO_ID' (or your chosen REPO_ID)."
    if [[ -x "$OPENPI_REPO/.venv/bin/python" ]]; then
        echo "   $OPENPI_REPO/.venv/bin/python scripts/compute_norm_stats.py \\"
    else
        echo "   uv run scripts/compute_norm_stats.py \\"
    fi
    echo "       --config-name pi0_libero_cam_pytorch_prope_ray_view"
fi
