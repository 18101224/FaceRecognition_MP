#!/usr/bin/env bash

set -u -o pipefail

DATASET_NAME="${DATASET_NAME:-RAF-DB}"
DATASET_PATH="${DATASET_PATH:-../data/RAF-DB_balanced}"
MODEL_TYPE="${MODEL_TYPE:-ir50}"
CKPT_PATH="${CKPT_PATH:-checkpoint/73c41da61-49bc-44f7-a8ac-7129bb49b694/best.pth}"
NUM_WORKERS="${NUM_WORKERS:-16}"

# Keep the default sweep tractable. Override, e.g.:
#   N_ITERS=20000 CALIB_RATIO=0.1 bash shells/quantization_sweep.sh
N_ITERS="${N_ITERS:-2000}"
CALIB_RATIO="${CALIB_RATIO:-0.1}"
PRECOMPUTE="${PRECOMPUTE:-1}"
LAM="${LAM:-1e-2}"
REG_REDUCTION="${REG_REDUCTION:-sum}"
OPT_TARGET="${OPT_TARGET:-both}"
ACT_INIT_MODE="${ACT_INIT_MODE:-lsq}"
ACT_INIT_PERCENTILE="${ACT_INIT_PERCENTILE:-0.999}"
ACT_INIT_SAMPLES="${ACT_INIT_SAMPLES:-64}"

RUN_TS="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="logs/quantization_sweep/${RUN_TS}"
LOG_DIR="${RUN_DIR}/logs"
OUT_DIR="${RUN_DIR}/checkpoints"
SUMMARY_FILE="${RUN_DIR}/summary.txt"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

cat <<EOF | tee "${SUMMARY_FILE}"
Quantization sweep started at ${RUN_TS}
dataset_name=${DATASET_NAME}
dataset_path=${DATASET_PATH}
model_type=${MODEL_TYPE}
ckpt_path=${CKPT_PATH}
num_workers=${NUM_WORKERS}
n_iters=${N_ITERS}
calib_ratio=${CALIB_RATIO}
precompute=${PRECOMPUTE}
lam=${LAM}
reg_reduction=${REG_REDUCTION}
opt_target=${OPT_TARGET}
act_init_mode=${ACT_INIT_MODE}
act_init_percentile=${ACT_INIT_PERCENTILE}
act_init_samples=${ACT_INIT_SAMPLES}
run_dir=${RUN_DIR}

Cases:
  1. w8a8_fisher_on
  2. w4a8_fisher_on
  3. w4a8_fisher_off
  4. w4a4_fisher_off
  5. w4a4_fisher_on

EOF

run_case() {
    local name="$1"
    local w_bits="$2"
    local a_bits="$3"
    local fisher="$4"

    local log_file="${LOG_DIR}/${name}.log"
    local quant_output="${OUT_DIR}/${name}.pth"

    local -a cmd=(
        python FER_CL.py
        --dataset_name "${DATASET_NAME}"
        --dataset_path "${DATASET_PATH}"
        --model_type "${MODEL_TYPE}"
        --ckpt_path "${CKPT_PATH}"
        --num_workers "${NUM_WORKERS}"
        --quantize
        --w_bits "${w_bits}"
        --a_bits "${a_bits}"
        --n_iters "${N_ITERS}"
        --lam "${LAM}"
        --reg_reduction "${REG_REDUCTION}"
        --opt_target "${OPT_TARGET}"
        --act_init_mode "${ACT_INIT_MODE}"
        --act_init_percentile "${ACT_INIT_PERCENTILE}"
        --act_init_samples "${ACT_INIT_SAMPLES}"
        --calib_ratio "${CALIB_RATIO}"
        --quant_output "${quant_output}"
    )

    if [[ "${PRECOMPUTE}" == "1" ]]; then
        cmd+=(--precompute)
    fi

    if [[ "${fisher}" == "1" ]]; then
        cmd+=(--use_fisher)
    fi

    {
        echo "============================================================"
        echo "CASE ${name}"
        echo "============================================================"
        printf 'Command:'
        printf ' %q' "${cmd[@]}"
        printf '\n\n'
        "${cmd[@]}"
    } 2>&1 | tee "${log_file}"

    local status="${PIPESTATUS[0]}"

    {
        echo "============================================================"
        echo "RESULT ${name}"
        echo "============================================================"
        echo "status=${status}"
        echo "log_file=${log_file}"
        echo "quant_output=${quant_output}"
        grep -E "original_fp32_acc|folded_fp32_acc|pre_brecq_quant_acc|post_brecq_quant_acc" "${log_file}" || true
        echo
    } | tee -a "${SUMMARY_FILE}"

    return 0
}

run_case "w8a8_fisher_on" 8 8 1
run_case "w4a8_fisher_on" 4 8 1
run_case "w4a8_fisher_off" 4 8 0
run_case "w4a4_fisher_off" 4 4 0
run_case "w4a4_fisher_on" 4 4 1

echo "Sweep complete. Summary: ${SUMMARY_FILE}" | tee -a "${SUMMARY_FILE}"
