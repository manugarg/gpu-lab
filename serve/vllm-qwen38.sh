#!/usr/bin/env bash
# vLLM serving Qwen3.8-27B. Thin wrapper over serve.sh that pins the flags
# this model actually needs — several of which fail in non-obvious ways if
# omitted. See serve/README.md for why each one is here.
set -euo pipefail

# MTP speculative decoding: ~2x raw decode (63.8 -> 126.4 tok/s, 71%
# acceptance on real workloads), ~1.25x on wall clock. Costs KV budget, so
# the context ceiling drops. VLLM_SPEC=0 to disable and get the full window.
SPEC="${VLLM_SPEC:-1}"
if [ "$SPEC" = "0" ]; then
  CTX="${VLLM_CTX:-131072}"
  spec_args=""
else
  CTX="${VLLM_CTX:-81920}"   # 131072 + MTP fails: needs 5.03 GiB KV, doesn't fit
  spec_args="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":3}"
fi

export MODEL="${MODEL:-unsloth/Qwen3.8-27B-NVFP4}"
export SERVE_ARGS="${SERVE_ARGS:-\
--served-model-name qwen3.8-27b qwen3.8-27b-medium qwen3.8-27b-low \
--attention-backend flashinfer \
--gpu-memory-utilization 0.92 \
--max-model-len $CTX \
--kv-cache-dtype fp8_e4m3 \
--max-num-seqs 4 \
--enable-prefix-caching \
--reasoning-parser qwen3 \
--enable-auto-tool-choice \
--tool-call-parser qwen3_xml \
$spec_args}"

exec "$(dirname "$0")/serve.sh" "$@"
