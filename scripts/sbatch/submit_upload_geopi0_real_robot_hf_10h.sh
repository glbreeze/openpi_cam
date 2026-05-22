#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

SBATCH_TIME=10:00:00 bash "${SCRIPT_DIR}/submit_upload_geopi0_real_robot_hf.sh"
