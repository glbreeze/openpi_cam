#!/usr/bin/env bash
# Eval E1: 4-suite LIBERO eval for the architecture-only ablation checkpoint
# `pi0_libero_cam_v2_prope_ray_view` (modules R + P + X, no distillation).
#
# Tests whether the ray + PRoPE + cross-view-fusion stack alone (no Pi3X
# aux head) helps or hurts vs. vanilla pi0_libero_baseline (87.1% avg).
#
# Usage:
#   bash scripts/sbatch_lg/submit_eval_geom_only.sh
#
# Outputs land under logs/eval_libero/geom_only/<suite>/summary/.

set -euo pipefail

export CONFIG_NAME=pi0_libero_cam_pytorch_prope_ray_view
export EXP_NAME=pi0_libero_cam_v2_prope_ray_view
export STEP=${STEP:-30000}
export MODEL_NAME=${MODEL_NAME:-geom_only}

exec bash "$(dirname "$0")/submit_eval_4suites.sh"
