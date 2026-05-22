#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

echo "Parallel time variants are disabled for upload-large-folder."
echo "Submit exactly one variant per release update to avoid corrupting Hugging Face upload metadata."
echo "Recommended:"
echo "  bash ${SCRIPT_DIR}/submit_upload_geopi0_real_robot_hf_30m.sh"
exit 1
