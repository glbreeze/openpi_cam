#!/bin/bash
#SBATCH --job-name=pi0_rc24_all24
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/robocasa24_eval_all24/slurm-%A_%a.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/robocasa24_eval_all24/slurm-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --account=torch_pr_676_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --array=0-23%4
#SBATCH --export=ALL

set -euo pipefail

resolve_repo_root() {
  local candidate
  local script_root=""

  if script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd); then
    :
  else
    script_root=""
  fi

  for candidate in "${REPO_ROOT:-}" "${OPENPI_CAM_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "${PWD:-}" "${script_root}"; do
    [[ -n "${candidate}" ]] || continue
    while [[ "${candidate}" != "/" ]]; do
      if [[ -f "${candidate}/scripts/env/activate_env.sh" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
      candidate=$(dirname -- "${candidate}")
    done
  done

  return 1
}

join_demo_keys() {
  local count="$1"
  local keys=""
  local i
  for ((i = 1; i <= count; i++)); do
    if [[ -n "${keys}" ]]; then
      keys+=":"
    fi
    keys+="demo_${i}"
  done
  printf '%s\n' "${keys}"
}

REPO_ROOT=${REPO_ROOT:-$(resolve_repo_root)} || {
  echo "Unable to locate repo root. Set REPO_ROOT or submit from inside the repo." >&2
  exit 1
}
cd "${REPO_ROOT}"

TASKS=(
  PnPCounterToCab
  PnPCabToCounter
  PnPCounterToSink
  PnPSinkToCounter
  PnPCounterToMicrowave
  PnPMicrowaveToCounter
  PnPCounterToStove
  PnPStoveToCounter
  OpenSingleDoor
  CloseSingleDoor
  OpenDoubleDoor
  CloseDoubleDoor
  OpenDrawer
  CloseDrawer
  TurnOnSinkFaucet
  TurnOffSinkFaucet
  TurnSinkSpout
  TurnOnStove
  TurnOffStove
  CoffeeSetupMug
  CoffeeServeMug
  CoffeePressButton
  TurnOnMicrowave
  TurnOffMicrowave
)

DATASETS=(
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-04-26/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_doors/OpenDoubleDoor/2024-04-26/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_doors/CloseDoubleDoor/2024-04-29/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_drawer/OpenDrawer/2024-05-03/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/2024-04-25/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_sink/TurnOffSinkFaucet/2024-04-25/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_sink/TurnSinkSpout/2024-04-29/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_stove/TurnOnStove/2024-05-02/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_stove/TurnOffStove/2024-05-02/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_coffee/CoffeeSetupMug/2024-04-25/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_coffee/CoffeePressButton/2024-04-25/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/2024-04-25/demo_gentex_im128_randcams.hdf5
  /scratch/yz11445/robocasa-human50/raw_human_im/v0.1/single_stage/kitchen_microwave/TurnOffMicrowave/2024-04-25/demo_gentex_im128_randcams.hdf5
)

task_index=${SLURM_ARRAY_TASK_ID:-${TASK_INDEX:-0}}
if (( task_index < 0 || task_index >= ${#TASKS[@]} )); then
  echo "Invalid task index ${task_index}; expected 0..$((${#TASKS[@]} - 1))" >&2
  exit 1
fi

variant=${VARIANT:-gtonly}
case "${variant}" in
  gtonly|gt-only|gt)
    variant="gtonly"
    export CONFIG_NAME=${CONFIG_NAME:-pi0_robocasa24_all24_cam_prope_ray_view_distill_fullres_stage2_gtonly}
    export EXP_NAME=${EXP_NAME:-stage2_all24_gtonly_224}
    ;;
  pi3x)
    export CONFIG_NAME=${CONFIG_NAME:-pi0_robocasa24_all24_cam_prope_ray_view_distill_fullres_stage2}
    export EXP_NAME=${EXP_NAME:-stage2_all24_libero4suite_pi3x224}
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}; expected gtonly or pi3x" >&2
    exit 1
    ;;
esac

export CHECKPOINT_STEP=${CHECKPOINT_STEP:-30000}
export EVAL_BACKEND=${EVAL_BACKEND:-robocasa-v0}
export EVAL_TASK=${EVAL_TASK:-${TASKS[${task_index}]}}
export DEMO_DATASET=${DEMO_DATASET:-${DATASETS[${task_index}]}}
export DEMO_COUNT=${DEMO_COUNT:-20}
export NUM_EPISODES=${NUM_EPISODES:-${DEMO_COUNT}}
export DEMO_KEYS=${DEMO_KEYS:-$(join_demo_keys "${DEMO_COUNT}")}
export MAX_STEPS=${MAX_STEPS:-500}
export VIDEO_EVERY=${VIDEO_EVERY:-0}
export RENDER_IMAGE_SIZE=${RENDER_IMAGE_SIZE:-128}
export SEND_OPENCV_EXTRINSICS=${SEND_OPENCV_EXTRINSICS:-1}

array_job_id=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
array_task_id=${SLURM_ARRAY_TASK_ID:-${task_index}}
run_tag=${RUN_TAG:-all24_20demo_ocv_${variant}_step${CHECKPOINT_STEP}_${array_job_id}}
result_root=${RESULT_ROOT:-${REPO_ROOT}/log/robocasa24_eval_all24/${run_tag}/${variant}}
mkdir -p "${result_root}/debug" "${result_root}/videos"

export RESULT_JSON=${RESULT_JSON:-${result_root}/${EVAL_TASK}_${DEMO_COUNT}demo_${array_job_id}_${array_task_id}.json}
export DEBUG_JSON=${DEBUG_JSON:-${result_root}/debug/${EVAL_TASK}_${DEMO_COUNT}demo_${array_job_id}_${array_task_id}.json}
export VIDEO_DIR=${VIDEO_DIR:-${result_root}/videos/${EVAL_TASK}_${DEMO_COUNT}demo_${array_job_id}_${array_task_id}}

echo "===== ROBOCASA24 ALL24 ARRAY ====="
echo "variant       : ${variant}"
echo "task index    : ${task_index}"
echo "task          : ${EVAL_TASK}"
echo "demo dataset  : ${DEMO_DATASET}"
echo "demo keys     : ${DEMO_KEYS}"
echo "result root   : ${result_root}"

exec bash "${REPO_ROOT}/scripts/sbatch/infer_pi0_robocasa24_cam_rollout_l40s.sbatch"
