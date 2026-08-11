#!/usr/bin/env bash
set -euo pipefail

# Exit if server is already running at port 8000
curl -sf localhost:8000/health >/dev/null && { echo "server already on :8000"; exit 1; }

source "$(dirname "$0")/../env/env.sh"
source "$VLLM_VENV/bin/activate"

LABEL="${1:?usage: capture.sh <label>}"
TRACE_DIR="/tmp/vllm_traces/$LABEL"
mkdir -p "$TRACE_DIR"

vllm serve "$MODEL" $SERVE_ARGS \
  --profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$TRACE_DIR\"}" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

until curl -sf localhost:8000/health >/dev/null; do
  kill -0 $SERVER_PID 2>/dev/null || { echo "server died"; exit 1; }
  sleep 5
done

vllm bench serve --model "$MODEL" --dataset-name random \
  --random-input-len 1024 --random-output-len 256 \
  --request-rate inf --num-prompts 20 --profile

echo "traces in $TRACE_DIR"

