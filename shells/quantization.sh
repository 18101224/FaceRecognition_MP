#!/usr/bin/env bash

set -euo pipefail

PROFILE="${BRECQ_PROFILE:-stable}"

DATASET_NAME="${DATASET_NAME:-RAF-DB}"
DATASET_PATH="${DATASET_PATH:-../data/RAF-DB_balanced}"
MODEL_TYPE="${MODEL_TYPE:-ir50}"
CKPT_PATH="${CKPT_PATH:-checkpoint/raf-ir50/best_acc.pth}"
NUM_WORKERS="${NUM_WORKERS:-16}"
CALIB_RATIO="${CALIB_RATIO:-0.1}"

case "${PROFILE}" in
    legacy)
        W_BITS_DEFAULT=4
        A_BITS_DEFAULT=8
        N_ITERS_DEFAULT=20000
        USE_FISHER_DEFAULT=1
        PRECOMPUTE_DEFAULT=1
        LAM_DEFAULT=1e-2
        REG_REDUCTION_DEFAULT=sum
        OPT_TARGET_DEFAULT=both
        ACT_INIT_MODE_DEFAULT=lsq
        ACT_INIT_PERCENTILE_DEFAULT=0.999
        ACT_INIT_SAMPLES_DEFAULT=64
        QUANT_OUTPUT_DEFAULT=checkpoint/ir50_w4a8_brecq_legacy.pth
        ;;
    stable)
        W_BITS_DEFAULT=4
        A_BITS_DEFAULT=8
        N_ITERS_DEFAULT=20000
        USE_FISHER_DEFAULT=1
        PRECOMPUTE_DEFAULT=1
        LAM_DEFAULT=1e-4
        REG_REDUCTION_DEFAULT=mean
        OPT_TARGET_DEFAULT=both
        ACT_INIT_MODE_DEFAULT=percentile
        ACT_INIT_PERCENTILE_DEFAULT=0.999
        ACT_INIT_SAMPLES_DEFAULT=256
        QUANT_OUTPUT_DEFAULT=checkpoint/ir50_w4a8_brecq_stable.pth
        ;;
    *)
        echo "Unsupported BRECQ_PROFILE: ${PROFILE}" >&2
        exit 1
        ;;
esac

W_BITS="${W_BITS:-${W_BITS_DEFAULT}}"
A_BITS="${A_BITS:-${A_BITS_DEFAULT}}"
N_ITERS="${N_ITERS:-${N_ITERS_DEFAULT}}"
USE_FISHER="${USE_FISHER:-${USE_FISHER_DEFAULT}}"
PRECOMPUTE="${PRECOMPUTE:-${PRECOMPUTE_DEFAULT}}"
LAM="${LAM:-${LAM_DEFAULT}}"
REG_REDUCTION="${REG_REDUCTION:-${REG_REDUCTION_DEFAULT}}"
OPT_TARGET="${OPT_TARGET:-${OPT_TARGET_DEFAULT}}"
ACT_INIT_MODE="${ACT_INIT_MODE:-${ACT_INIT_MODE_DEFAULT}}"
ACT_INIT_PERCENTILE="${ACT_INIT_PERCENTILE:-${ACT_INIT_PERCENTILE_DEFAULT}}"
ACT_INIT_SAMPLES="${ACT_INIT_SAMPLES:-${ACT_INIT_SAMPLES_DEFAULT}}"
QUANT_OUTPUT="${QUANT_OUTPUT:-${QUANT_OUTPUT_DEFAULT}}"

LOG_DIR="logs/quantization"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/ir50_brecq_${PROFILE}_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"
echo "Saving log to ${LOG_FILE}"

cmd=(
    python -m quantization.run_brecq
    --checkpoint "${CKPT_PATH}"
    --dataset-name "${DATASET_NAME}"
    --dataset-path "${DATASET_PATH}"
    --model-type "${MODEL_TYPE}"
    --num-workers "${NUM_WORKERS}"
    --w-bits "${W_BITS}"
    --a-bits "${A_BITS}"
    --n-iters "${N_ITERS}"
    --lam "${LAM}"
    --reg-reduction "${REG_REDUCTION}"
    --opt-target "${OPT_TARGET}"
    --act-init-mode "${ACT_INIT_MODE}"
    --act-init-percentile "${ACT_INIT_PERCENTILE}"
    --act-init-samples "${ACT_INIT_SAMPLES}"
    --calib-ratio "${CALIB_RATIO}"
    --output "${QUANT_OUTPUT}"
)

if [[ "${USE_FISHER}" == "1" ]]; then
    cmd+=(--use_fisher)
fi

if [[ "${PRECOMPUTE}" == "1" ]]; then
    cmd+=(--precompute)
fi

{
    echo "Profile: ${PROFILE}"
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n\n'
    "${cmd[@]}"
} 2>&1 | tee "${LOG_FILE}"
