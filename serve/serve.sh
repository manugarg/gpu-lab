#!/usr/bin/env bash
set -euo pipefail

curl -sf localhost:8000/health >/dev/null && { echo "server already on :8000"; exit 1; }

source "$(dirname "$0")/../env/env.sh"
source "$VLLM_VENV/bin/activate"

exec vllm serve "$MODEL" $SERVE_ARGS "$@"
