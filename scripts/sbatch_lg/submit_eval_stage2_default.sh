#!/usr/bin/env bash
# Eval E2: 4-suite LIBERO eval for the default-MSE stage2 distill checkpoint
# `pi0_libero_cam_v2_prope_ray_view_distill_fullres_stage2_h100_2gpu_b16_lg`
# (loss_type="legacy_conf_mse", legacy_conf_loss_order=2 = MSE).
#
# Counterfactual to the existing stage2_l1 eval: same architecture and aux
# weight, only the legacy loss order differs (MSE vs L1). The MSE variant has
# the lower training action_loss of all your distill runs (~0.046 @30k) but
# hasn't been evaluated yet.
#
# Usage:
#   bash scripts/sbatch_lg/submit_eval_stage2_default.sh
#
# Outputs land under logs/eval_libero/stage2_mse/<suite>/summary/.

set -euo pipefail

export CONFIG_NAME=pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2
export EXP_NAME=pi0_libero_cam_v2_prope_ray_view_distill_fullres_stage2_h100_2gpu_b16_lg
export STEP=${STEP:-30000}
export MODEL_NAME=${MODEL_NAME:-stage2_mse}

exec bash "$(dirname "$0")/submit_eval_4suites.sh"
