#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

CKPT_DIR="${1:-${ROOT_DIR}/checkpoint/a893f2a0-0}"
CHECKPOINT_TAG="${2:-best}"
EVAL_ROOT="${3:-/data/mj/facerec_val}"
ALIGNER_CKPT="${4:-${ROOT_DIR}/checkpoint/adaface_vit_base_kprpe_webface12m}"
BATCH_SIZE="${5:-256}"
NUM_WORKERS="${6:-4}"
DEVICE="${7:-cuda:0}"

shift $(( $# > 7 ? 7 : $# )) || true
DATASETS=( "$@" )
if [ "${#DATASETS[@]}" -eq 0 ]; then
  DATASETS=(lfw agedb_30 cfp_fp cplfw calfw)
fi

EXPORT_PATH="${CKPT_DIR}/${CHECKPOINT_TAG}/model.exported.pt"

if [ ! -f "${EXPORT_PATH}" ]; then
  python "${ROOT_DIR}/tools/export_eval_model.py" \
    --checkpoint_dir "${CKPT_DIR}" \
    --checkpoint_tag "${CHECKPOINT_TAG}" \
    --output_path "${EXPORT_PATH}" \
    --device "${DEVICE}" \
    --overwrite false
else
  echo "[EXPORT] reuse existing ${EXPORT_PATH}"
fi

python "${ROOT_DIR}/eval_verification.py" \
  --checkpoint_dir "${CKPT_DIR}" \
  --checkpoint_tag "${CHECKPOINT_TAG}" \
  --model_path "${EXPORT_PATH}" \
  --aligner_ckpt "${ALIGNER_CKPT}" \
  --eval_root "${EVAL_ROOT}" \
  --datasets "${DATASETS[@]}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --mixed_precision no \
  --use_flash_attn false \
  --device "${DEVICE}"
