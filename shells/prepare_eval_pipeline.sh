#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

ROOT="${1:-/data/mj}"
DEVICE="${2:-cuda:0}"
BATCH_SIZE="${3:-256}"
NUM_WORKERS="${4:-4}"

FACEREC_VAL="${ROOT}/facerec_val"
EVAL_BINS="${ROOT}/eval_bins"
TINYFACE_ROOT="${ROOT}/TinyFace"
IJB_ROOT="${ROOT}/IJB_release"
IJBS_ROOT_DEFAULT="${IJB_ROOT}/IJBS"

mkdir -p "${FACEREC_VAL}" "${EVAL_BINS}"

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1"; exit 1; }; }
need_cmd python

echo "[1/5] Install minimal deps for eval prep"
python -m pip install -U datasets pyarrow >/dev/null

echo "[2/5] Convert verification bins -> run_v1 eval format"
python "${PROJECT_ROOT}/tools/prepare_verification_eval.py" \
  --bin_root "${EVAL_BINS}" \
  --out_root "${FACEREC_VAL}" \
  --names lfw agedb_30 cfp_fp cplfw calfw || true

echo "[3/5] TinyFace preprocess (expects local raw data)"
if [ -d "${TINYFACE_ROOT}" ]; then
  python "${PROJECT_ROOT}/tools/prepare_tinyface_eval.py" \
    --tinyface_root "${TINYFACE_ROOT}" \
    --out_path "${FACEREC_VAL}/tinyface_aligned_pad_0.1" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --align \
    --overwrite || true
else
  echo "[WARN] TinyFace root not found: ${TINYFACE_ROOT}"
fi

echo "[4/5] IJB-C preprocess (expects local raw data)"
if [ -d "${IJB_ROOT}" ]; then
  python "${PROJECT_ROOT}/tools/prepare_ijbc_eval.py" \
    --ijb_root "${IJB_ROOT}" \
    --subset ijbc \
    --out_root "${FACEREC_VAL}" \
    --device "${DEVICE}" \
    --align \
    --overwrite || true
else
  echo "[WARN] IJB root not found: ${IJB_ROOT}"
fi

echo "[5/5] IJB-S aligned data prep (expects local raw data)"
if [ -d "${IJBS_ROOT_DEFAULT}" ]; then
  python "${PROJECT_ROOT}/tools/prepare_ijbs_aligned.py" \
    --ijbs_root "${IJBS_ROOT_DEFAULT}" \
    --out_root "${ROOT}/ijbs_aligned_112" \
    --device "${DEVICE}" \
    --overwrite || true
else
  echo "[WARN] IJB-S root not found (${IJBS_ROOT_DEFAULT}). Skip IJB-S preprocessing."
fi
echo
echo "[INFO] Final readiness check"
python "${PROJECT_ROOT}/tools/check_eval_ready.py" --root "${ROOT}"
