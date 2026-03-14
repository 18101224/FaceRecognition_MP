#!/usr/bin/env bash

set -euo pipefail

# Exact reproduction of the earlier sweep case:
#   w8a8_fisher_on
# from `shells/quantization_sweep.sh` defaults.
BRECQ_PROFILE=legacy \
W_BITS=8 \
A_BITS=8 \
N_ITERS=2000 \
USE_FISHER=1 \
PRECOMPUTE=1 \
CALIB_RATIO=0.1 \
LAM=1e-2 \
REG_REDUCTION=sum \
OPT_TARGET=both \
ACT_INIT_MODE=lsq \
ACT_INIT_PERCENTILE=0.999 \
ACT_INIT_SAMPLES=1024 \
QUANT_OUTPUT=checkpoint/ir50_w8a8_brecq_legacy_sweep.pth \
bash "$(dirname "$0")/quantization.sh"
