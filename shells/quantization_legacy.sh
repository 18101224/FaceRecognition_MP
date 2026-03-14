#!/usr/bin/env bash

set -euo pipefail

# Legacy single-run profile:
#   W4A8, 20000 iters, fisher on, old LSQ-style activation init.
# This is NOT the same as the earlier sweep case `w8a8_fisher_on`.
BRECQ_PROFILE=legacy bash "$(dirname "$0")/quantization.sh"
