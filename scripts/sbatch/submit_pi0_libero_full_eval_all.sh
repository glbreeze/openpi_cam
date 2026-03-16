#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SBATCH_SCRIPT="${SCRIPT_DIR}/infer_pi0_libero_full_eval.sbatch"
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}

for suite in ${SUITES}; do
  echo "Submitting full eval for ${suite}"
  sbatch --export=ALL,TASK_SUITE_NAME="${suite}" "${SBATCH_SCRIPT}"
done
