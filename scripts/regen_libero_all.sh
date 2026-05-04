#!/usr/bin/env bash
#
# All-suite regen pipeline: libero_object + libero_spatial + libero_goal +
# libero_10. Phase 1 regenerates HDF5 for each task in each suite (libero
# conda env); Phase 2 walks the combined root and writes ONE LeRobot v2.1
# repo containing every kept demo across all suites.
#
# Companion to scripts/regen_libero_object.sh; same conventions, same flags.
# Override any setting via env var, e.g.:
#
#   SUITES="libero_object libero_goal" REPO_ID=glbreeze/libero_partial_v2 \
#       bash scripts/regen_libero_all.sh
#
# Refuses to overwrite the target LeRobot repo. Phase 1 uses --resume so
# already-finished HDF5 files are skipped.

set -euo pipefail

# ---------------------------------------------------------------------------- #
# Config                                                                        #
# ---------------------------------------------------------------------------- #
SUITES="${SUITES:-libero_object libero_spatial libero_goal libero_10}"
SRC_BASE="${SRC_BASE:-/home/asus/Research/datasets/libero_datasets}"
OUT_HDF5_ROOT="${OUT_HDF5_ROOT:-/home/asus/Research/datasets/libero_cam_v2}"
LIBERO_CFG_DIR="${LIBERO_CFG_DIR:-/home/asus/Research/datasets/libero_cam_v2_cfg}"
REPO_ID="${REPO_ID:-glbreeze/libero_cam_v2}"
RENDER_RES="${RENDER_RES:-256}"
SETTLE_STEPS="${SETTLE_STEPS:-10}"
MAX_EPISODES="${MAX_EPISODES:-0}"
LIBERO_CAMERA_REPO="${LIBERO_CAMERA_REPO:-/home/asus/Research/LIBERO-Camera}"
OPENPI_REPO="${OPENPI_REPO:-/home/asus/Research/openpi}"
LIBERO_CONDA_ENV="${LIBERO_CONDA_ENV:-libero}"
CONDA_SH="${CONDA_SH:-}"
OPENPI_PYTHON="${OPENPI_PYTHON:-}"
SKIP_PHASE1="${SKIP_PHASE1:-0}"
SKIP_PHASE2="${SKIP_PHASE2:-0}"
LEROBOT_MODE="${LEROBOT_MODE:-original_only}"

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
echo " LIBERO all-suite regeneration"
echo "================================================================="
echo " SUITES             = $SUITES"
echo " SRC_BASE           = $SRC_BASE"
echo " OUT_HDF5_ROOT      = $OUT_HDF5_ROOT"
echo " LIBERO_CFG_DIR     = $LIBERO_CFG_DIR"
echo " REPO_ID            = $REPO_ID"
echo " RENDER_RES         = $RENDER_RES"
echo " SETTLE_STEPS       = $SETTLE_STEPS"
echo " MAX_EPISODES       = $MAX_EPISODES (0 = all)"
echo " CONDA_SH           = ${CONDA_SH:-<auto>}"
echo " OPENPI_PYTHON      = ${OPENPI_PYTHON:-<auto>}"
echo " SKIP_PHASE1        = $SKIP_PHASE1"
echo " SKIP_PHASE2        = $SKIP_PHASE2"
echo " LEROBOT_MODE       = $LEROBOT_MODE"
echo "================================================================="

# ---------------------------------------------------------------------------- #
# Sanity / safety                                                               #
# ---------------------------------------------------------------------------- #
for SUITE in $SUITES; do
    if [[ ! -d "$SRC_BASE/$SUITE" ]]; then
        echo "ERROR: source dir does not exist: $SRC_BASE/$SUITE" >&2
        exit 1
    fi
    if ! ls "$SRC_BASE/$SUITE"/*_demo.hdf5 >/dev/null 2>&1; then
        echo "ERROR: no *_demo.hdf5 files under $SRC_BASE/$SUITE" >&2
        exit 1
    fi
done

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

for SUITE in $SUITES; do
    OUT_SUITE_DIR="$OUT_HDF5_ROOT/$SUITE"
    if [[ "$SKIP_PHASE1" -eq 0 ]] && [[ -d "$OUT_SUITE_DIR" ]] && [[ -n "$(ls -A "$OUT_SUITE_DIR" 2>/dev/null || true)" ]]; then
        echo "WARNING: $OUT_SUITE_DIR is non-empty. --resume will skip existing files."
    fi
done

# ---------------------------------------------------------------------------- #
# Phase 1: Regenerate HDF5 for every suite                                      #
# ---------------------------------------------------------------------------- #
if [[ "$SKIP_PHASE1" -eq 0 ]]; then
    echo
    echo "----- Phase 1: regen HDF5 across $SUITES -----"
    mkdir -p "$LIBERO_CFG_DIR"
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
    pushd "$LIBERO_CAMERA_REPO" > /dev/null
    for SUITE in $SUITES; do
        SUITE_START=$(date +%s)
        OUT_SUITE_DIR="$OUT_HDF5_ROOT/$SUITE"
        mkdir -p "$OUT_SUITE_DIR"
        NUM_FILES=0
        echo
        echo "========== suite: $SUITE =========="
        for SRC in "$SRC_BASE/$SUITE"/*_demo.hdf5; do
            NUM_FILES=$((NUM_FILES + 1))
            echo
            echo "[regen $(date +%H:%M:%S)] $SUITE / file $NUM_FILES: $(basename "$SRC")"
            LIBERO_CONFIG_PATH="$LIBERO_CFG_DIR" MUJOCO_GL=egl python scripts/create_dataset.py \
                --demo-file "$SRC" \
                --use-camera-obs \
                --render-resolution "$RENDER_RES" \
                --filter-noops \
                --settle-steps "$SETTLE_STEPS" \
                --resume \
                "${EPS_ARGS[@]}"
        done
        SUITE_SEC=$(( $(date +%s) - SUITE_START ))
        echo
        echo "[regen done] $SUITE: $NUM_FILES files in ${SUITE_SEC}s"
    done
    popd > /dev/null

    conda deactivate
    PHASE1_SEC=$(( $(date +%s) - PHASE1_START ))
    echo
    echo "[regen all-suite done] total ${PHASE1_SEC}s"
    echo "[regen all-suite done] outputs under: $OUT_HDF5_ROOT/{$(echo $SUITES | tr ' ' ',')}"
fi

# ---------------------------------------------------------------------------- #
# Phase 2: One LeRobot conversion pass over the combined root                   #
# ---------------------------------------------------------------------------- #
# convert_libero_hdf5_to_lerobot.py uses dataset_root.rglob("*.hdf5"), so
# pointing at $OUT_HDF5_ROOT (parent of all suite subdirs) picks up every
# kept demo across every suite into one LeRobot repo.
if [[ "$SKIP_PHASE2" -eq 0 ]]; then
    echo
    echo "----- Phase 2: HDF5 (combined) → LeRobot -----"
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
    echo " Next: point a training config at REPO_ID='$REPO_ID' and refresh"
    echo " norm stats. The closest existing config is 'pi0_libero_cam'"
    echo " (which uses 'glbreeze/libero_cam'). Either:"
    echo "   (a) edit pi0_libero_cam to use '$REPO_ID' (and matching asset_id), or"
    echo "   (b) clone that block under a new name, then:"
    echo "       cd $OPENPI_REPO"
    if [[ -x "$OPENPI_REPO/.venv/bin/python" ]]; then
        echo "       $OPENPI_REPO/.venv/bin/python scripts/compute_norm_stats.py --config-name <new-or-edited-name>"
    else
        echo "       uv run scripts/compute_norm_stats.py --config-name <new-or-edited-name>"
    fi
fi
