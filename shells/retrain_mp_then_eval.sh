#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

RUN_ID="${1:-mp-bf16-fixed}"
NO_MP_CKPT_DIR="${2:-${ROOT_DIR}/checkpoint/a893f2a0-0}"
EVAL_TAG="${3:-best}"

export RUN_ID

echo "============================================================"
echo "[TRAIN] retrain MP checkpoint with fixed full-state save"
echo "run_id=${RUN_ID}"
echo "checkpoint_dir=${ROOT_DIR}/checkpoint/${RUN_ID}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${MP_VISIBLE_DEVICES:-2,3}" torchrun --nproc_per_node=2 --standalone "${ROOT_DIR}/train.py" \
  --dataset_name casia \
  --dataset_root /data/mj/casia-webface-aligned \
  --aligner_ckpt "${ROOT_DIR}/checkpoint/adaface_vit_base_kprpe_webface12m" \
  --architecture kprpe_small \
  --embedding_dim 512 \
  --classifier fc \
  --batch_size 256 \
  --n_epochs 100 \
  --learning_rate 1e-3 \
  --weight_decay 0.05 \
  --h 0.333 \
  --mixed_precision bf16 \
  --use_accelerator true \
  --use_flash_attn false \
  --rpe_impl extension

echo
echo "============================================================"
echo "[EVAL] compare retrained MP vs existing no-MP"
echo "mp_checkpoint=${ROOT_DIR}/checkpoint/${RUN_ID}"
echo "no_mp_checkpoint=${NO_MP_CKPT_DIR}"
echo "============================================================"

bash "${ROOT_DIR}/shells/eval_mp_vs_nompi.sh" \
  "${ROOT_DIR}/checkpoint/${RUN_ID}" \
  "${NO_MP_CKPT_DIR}" \
  "${EVAL_TAG}"
