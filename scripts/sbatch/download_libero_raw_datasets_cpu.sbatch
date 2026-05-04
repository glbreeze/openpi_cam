#!/bin/bash
#SBATCH --job-name=dl_libero_raw
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_dataset_download/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_dataset_download/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yp2841/geometry-vla/openpi_cam}
LIBERO_REPO=${LIBERO_REPO:-${REPO_ROOT}/third_party/libero}
DOWNLOAD_DIR=${DOWNLOAD_DIR:-/scratch/yp2841/geometry-vla/libero_raw_datasets}
CONDA_SH=${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}
LIBERO_CONDA_ENV=${LIBERO_CONDA_ENV:-/scratch/yp2841/.conda-envs/openvla-oft}
export REPO_ROOT LIBERO_REPO DOWNLOAD_DIR CONDA_SH LIBERO_CONDA_ENV

mkdir -p "${DOWNLOAD_DIR}" /scratch/yp2841/geometry-vla/openpi_cam/log/libero_dataset_download

count_hdf5() {
    local name="$1"
    local path="${DOWNLOAD_DIR}/${name}"
    if [[ ! -d "${path}" ]]; then
        echo 0
        return 0
    fi
    find "${path}" -maxdepth 1 -name '*_demo.hdf5' | wc -l
}

download_zip() {
    local url="$1"
    local out_zip="$2"
    echo "Downloading ${url}"
    if wget --no-check-certificate -O "${out_zip}" "${url}"; then
        return 0
    fi
    echo "wget failed; retrying with curl -k"
    curl -L -k -o "${out_zip}" "${url}"
}

download_and_extract() {
    local dataset="$1"
    local url="$2"
    local out_zip="${DOWNLOAD_DIR}/${dataset}.zip"

    download_zip "${url}" "${out_zip}"
    unzip -oq "${out_zip}" -d "${DOWNLOAD_DIR}"
    rm -f "${out_zip}"
}

echo "===== BEFORE DOWNLOAD ====="
echo "libero_object: $(count_hdf5 libero_object)"
echo "libero_goal: $(count_hdf5 libero_goal)"
echo "libero_spatial: $(count_hdf5 libero_spatial)"
echo "libero_10: $(count_hdf5 libero_10)"
echo "libero_90: $(count_hdf5 libero_90)"

declare -a downloads=()
if [[ "$(count_hdf5 libero_goal)" -lt 10 ]]; then
    downloads+=("libero_goal")
fi
if [[ "$(count_hdf5 libero_spatial)" -lt 10 ]]; then
    downloads+=("libero_spatial")
fi
if [[ "$(count_hdf5 libero_10)" -lt 10 || "$(count_hdf5 libero_90)" -lt 90 ]]; then
    downloads+=("libero_100")
fi

if [[ "${#downloads[@]}" -eq 0 ]]; then
    echo "All requested raw datasets are already present; nothing to download."
    exit 0
fi

for dataset in "${downloads[@]}"; do
    case "${dataset}" in
        libero_goal)
            download_and_extract "${dataset}" "https://utexas.box.com/shared/static/iv5e4dos8yy2b212pkzkpxu9wbdgjfeg.zip"
            ;;
        libero_spatial)
            download_and_extract "${dataset}" "https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip"
            ;;
        libero_100)
            download_and_extract "${dataset}" "https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip"
            ;;
        *)
            echo "Unknown dataset key: ${dataset}" >&2
            exit 1
            ;;
    esac
done

echo "===== AFTER DOWNLOAD ====="
echo "libero_object: $(count_hdf5 libero_object)"
echo "libero_goal: $(count_hdf5 libero_goal)"
echo "libero_spatial: $(count_hdf5 libero_spatial)"
echo "libero_10: $(count_hdf5 libero_10)"
echo "libero_90: $(count_hdf5 libero_90)"

errors=()
if [[ "$(count_hdf5 libero_goal)" -lt 10 ]]; then
    errors+=("libero_goal incomplete")
fi
if [[ "$(count_hdf5 libero_spatial)" -lt 10 ]]; then
    errors+=("libero_spatial incomplete")
fi
if [[ "$(count_hdf5 libero_10)" -lt 10 ]]; then
    errors+=("libero_10 incomplete")
fi
if [[ "$(count_hdf5 libero_90)" -lt 90 ]]; then
    errors+=("libero_90 incomplete")
fi

if [[ "${#errors[@]}" -gt 0 ]]; then
    printf '%s\n' "${errors[@]}" >&2
    exit 1
fi

echo "Download complete: ${DOWNLOAD_DIR}"
