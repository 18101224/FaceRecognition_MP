#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

MP_CKPT_DIR="${1:-${ROOT_DIR}/checkpoint/c96d4787-8}"
NO_MP_CKPT_DIR="${2:-${ROOT_DIR}/checkpoint/a893f2a0-0}"
CHECKPOINT_TAG="${3:-best}"
EVAL_ROOT="${4:-/data/mj/facerec_val}"
ALIGNER_CKPT="${5:-${ROOT_DIR}/checkpoint/adaface_vit_base_kprpe_webface12m}"
BATCH_SIZE="${6:-256}"
NUM_WORKERS="${7:-4}"
FLIP_TEST="${8:-true}"

shift $(( $# > 8 ? 8 : $# )) || true
DATASETS=( "$@" )
if [ "${#DATASETS[@]}" -eq 0 ]; then
  DATASETS=(lfw agedb_30 cfp_fp cplfw calfw)
fi

NO_MP_VISIBLE_DEVICES="${NO_MP_VISIBLE_DEVICES:-0,1}"
MP_VISIBLE_DEVICES="${MP_VISIBLE_DEVICES:-2,3}"
NO_MP_DEVICE="${NO_MP_DEVICE:-cuda:0}"
MP_DEVICE="${MP_DEVICE:-cuda:0}"

read_checkpoint_arg() {
  local checkpoint_dir="$1"
  local checkpoint_tag="$2"
  local key="$3"
  python - "$checkpoint_dir" "$checkpoint_tag" "$key" <<'PY'
import sys
from pathlib import Path
import torch

checkpoint_root = Path(sys.argv[1]).expanduser().resolve()
checkpoint_tag = sys.argv[2]
key = sys.argv[3]

checkpoint_dir = checkpoint_root / checkpoint_tag if (checkpoint_root / checkpoint_tag).is_dir() else checkpoint_root
rank_state = torch.load(checkpoint_dir / "train_state.r0.pt", map_location="cpu", weights_only=False)
value = rank_state.get("args", {}).get(key)
if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("")
else:
    print(value)
PY
}

export_checkpoint() {
  local name="$1"
  local checkpoint_dir="$2"
  local visible_devices="$3"
  local device="$4"

  local use_accelerator
  local world_size
  local exported_path

  use_accelerator="$(read_checkpoint_arg "${checkpoint_dir}" "${CHECKPOINT_TAG}" "use_accelerator")"
  world_size="$(read_checkpoint_arg "${checkpoint_dir}" "${CHECKPOINT_TAG}" "world_size")"
  exported_path="${checkpoint_dir}/${CHECKPOINT_TAG}/model.exported.pt"

  echo
  echo "============================================================"
  echo "[EXPORT] ${name}"
  echo "checkpoint_dir=${checkpoint_dir}"
  echo "checkpoint_tag=${CHECKPOINT_TAG}"
  echo "use_accelerator=${use_accelerator}"
  echo "world_size=${world_size}"
  echo "visible_devices=${visible_devices}"
  echo "============================================================"

  if [ -f "${exported_path}" ]; then
    echo "[EXPORT] reuse existing ${exported_path}"
  else
    CUDA_VISIBLE_DEVICES="${visible_devices}" python "${ROOT_DIR}/tools/export_eval_model.py" \
      --checkpoint_dir "${checkpoint_dir}" \
      --checkpoint_tag "${CHECKPOINT_TAG}" \
      --output_path "${exported_path}" \
      --device "${device}" \
      --overwrite false
  fi
  EXPORTED_MODEL_PATH="${exported_path}"
}

run_eval() {
  local name="$1"
  local checkpoint_dir="$2"
  local model_path="$3"
  local visible_devices="$4"
  local device="$5"

  echo
  echo "============================================================"
  echo "[EVAL] ${name}"
  echo "checkpoint_dir=${checkpoint_dir}"
  echo "model_path=${model_path}"
  echo "checkpoint_tag=${CHECKPOINT_TAG}"
  echo "visible_devices=${visible_devices}"
  echo "device=${device}"
  echo "datasets=${DATASETS[*]}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${visible_devices}" python "${ROOT_DIR}/eval_verification.py" \
    --checkpoint_dir "${checkpoint_dir}" \
    --checkpoint_tag "${CHECKPOINT_TAG}" \
    --model_path "${model_path}" \
    --aligner_ckpt "${ALIGNER_CKPT}" \
    --eval_root "${EVAL_ROOT}" \
    --datasets "${DATASETS[@]}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --mixed_precision no \
    --use_flash_attn false \
    --flip_test "${FLIP_TEST}" \
    --device "${device}"
}

export_checkpoint "w.o._MP" "${NO_MP_CKPT_DIR}" "${NO_MP_VISIBLE_DEVICES}" "${NO_MP_DEVICE}"
NO_MP_MODEL_PATH="${EXPORTED_MODEL_PATH}"
run_eval "w.o._MP" "${NO_MP_CKPT_DIR}" "${NO_MP_MODEL_PATH}" "${NO_MP_VISIBLE_DEVICES}" "${NO_MP_DEVICE}"

export_checkpoint "w_MP" "${MP_CKPT_DIR}" "${MP_VISIBLE_DEVICES}" "${MP_DEVICE}"
MP_MODEL_PATH="${EXPORTED_MODEL_PATH}"
run_eval "w_MP" "${MP_CKPT_DIR}" "${MP_MODEL_PATH}" "${MP_VISIBLE_DEVICES}" "${MP_DEVICE}"
