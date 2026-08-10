#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../env/env.sh"
source "$VLLM_VENV/bin/activate"

LABEL="${1:?usage: run.sh <label>}"
OUT="$(dirname "$0")/results"
mkdir -p "$OUT"

common=(--model "$MODEL" --dataset-name random
        --random-input-len 1024 --random-output-len 256
        --save-result --result-dir "$OUT")

"$(dirname "$0")/../serve/wait-ready.sh"

vllm bench serve "${common[@]}" --request-rate 4 --num-prompts 1000 \
  --label "${LABEL}-rate4"

vllm bench serve "${common[@]}" --request-rate inf --num-prompts 200 \
  --label "${LABEL}-saturated"

