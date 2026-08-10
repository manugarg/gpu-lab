#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../env/env.sh"
source "$VLLM_VENV/bin/activate"

exec vllm serve "$MODEL" $SERVE_ARGS "$@"
