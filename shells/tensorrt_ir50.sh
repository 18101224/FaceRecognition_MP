#!/usr/bin/env bash

set -euo pipefail

CKPT_PATH="${CKPT_PATH:-checkpoint/raf-ir50/best_acc.pth}"
QUANT_STATE="${QUANT_STATE:-checkpoint/ir50_w8a8_brecq_legacy_sweep.pth}"
DATASET_NAME="${DATASET_NAME:-RAF-DB}"
DATASET_PATH="${DATASET_PATH:-../data/RAF-DB_balanced}"
IMG_SIZE="${IMG_SIZE:-112}"
BENCH_BATCH="${BENCH_BATCH:-128}"
MIN_BATCH="${MIN_BATCH:-1}"
OPT_BATCH="${OPT_BATCH:-128}"
MAX_BATCH="${MAX_BATCH:-128}"
NUM_WORKERS="${NUM_WORKERS:-16}"
OUT_DIR="${OUT_DIR:-checkpoint/deploy/w8a8_legacy}"
FP_BASELINE_PRECISION="${FP_BASELINE_PRECISION:-fp32}"

mkdir -p "${OUT_DIR}"

python -m quantization.run_tensorrt export-onnx \
    --checkpoint "${CKPT_PATH}" \
    --variant fp32 \
    --batch-size 1 \
    --img-size "${IMG_SIZE}" \
    --dynamic-batch \
    --onnx "${OUT_DIR}/ir50_fp32.onnx"

python -m quantization.run_tensorrt export-onnx \
    --checkpoint "${CKPT_PATH}" \
    --variant qdq_int8 \
    --quant-state "${QUANT_STATE}" \
    --batch-size 1 \
    --img-size "${IMG_SIZE}" \
    --dynamic-batch \
    --quant-report "${OUT_DIR}/ir50_w8a8_qdq_report.json" \
    --onnx "${OUT_DIR}/ir50_w8a8_qdq.onnx"

python -m quantization.run_tensorrt build-engine \
    --onnx "${OUT_DIR}/ir50_fp32.onnx" \
    --engine "${OUT_DIR}/ir50_fp32.engine" \
    --precision "${FP_BASELINE_PRECISION}" \
    --img-size "${IMG_SIZE}" \
    --min-batch "${MIN_BATCH}" \
    --opt-batch "${OPT_BATCH}" \
    --max-batch "${MAX_BATCH}" \
    --use-cuda-graph

python -m quantization.run_tensorrt build-engine \
    --onnx "${OUT_DIR}/ir50_w8a8_qdq.onnx" \
    --engine "${OUT_DIR}/ir50_w8a8_qdq.engine" \
    --img-size "${IMG_SIZE}" \
    --min-batch "${MIN_BATCH}" \
    --opt-batch "${OPT_BATCH}" \
    --max-batch "${MAX_BATCH}" \
    --explicit-quantization \
    --use-cuda-graph

python -m quantization.run_tensorrt eval-trt \
    --checkpoint "${CKPT_PATH}" \
    --engine "${OUT_DIR}/ir50_fp32.engine" \
    --dataset-name "${DATASET_NAME}" \
    --dataset-path "${DATASET_PATH}" \
    --batch-size "${BENCH_BATCH}" \
    --num-workers "${NUM_WORKERS}" \
    --img-size "${IMG_SIZE}"

python -m quantization.run_tensorrt eval-trt \
    --checkpoint "${CKPT_PATH}" \
    --engine "${OUT_DIR}/ir50_w8a8_qdq.engine" \
    --dataset-name "${DATASET_NAME}" \
    --dataset-path "${DATASET_PATH}" \
    --batch-size "${BENCH_BATCH}" \
    --num-workers "${NUM_WORKERS}" \
    --img-size "${IMG_SIZE}"

python -m quantization.run_tensorrt benchmark-trt \
    --engine "${OUT_DIR}/ir50_fp32.engine" \
    --batch-size "${BENCH_BATCH}" \
    --img-size "${IMG_SIZE}" \
    --use-cuda-graph

python -m quantization.run_tensorrt benchmark-trt \
    --engine "${OUT_DIR}/ir50_w8a8_qdq.engine" \
    --batch-size "${BENCH_BATCH}" \
    --img-size "${IMG_SIZE}" \
    --use-cuda-graph
