#!/usr/bin/env bash
set -euo pipefail

# Exit if server is already running at port 8000
curl -sf localhost:8000/health >/dev/null && { echo "server already on :8000"; exit 1; }

source "$(dirname "$0")/../env/env.sh"
source "$VLLM_VENV/bin/activate"

LABEL="${1:?usage: capture.sh <label> [input_len] [output_len]}"
INPUT_LEN="${2:-1024}"
OUTPUT_LEN="${3:-256}"
TRACE_DIR="$(dirname "$0")/results/$LABEL"
mkdir -p "$TRACE_DIR"
echo "MODEL=$MODEL" > "$TRACE_DIR/meta.txt"
echo "$INPUT_LEN/$OUTPUT_LEN $(date -Iseconds)" >> "$TRACE_DIR/meta.txt"

vllm serve "$MODEL" $SERVE_ARGS \
  --profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$TRACE_DIR\"}" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

"$(dirname "$0")/../serve/wait-ready.sh" "$SERVER_PID"

vllm bench serve --model "$MODEL" --dataset-name random \
  --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
  --request-rate inf --num-prompts 20 --profile \
  --save-result --result-dir "$TRACE_DIR" --label "$LABEL"

echo "traces in $TRACE_DIR"

