#!/usr/bin/env bash
# vLLM serving Qwen3.8-27B. Thin wrapper over serve.sh that pins the flags
# this model actually needs — several of which fail in non-obvious ways if
# omitted. See serve/README.md for why each one is here.
set -euo pipefail

# MTP speculative decoding: ~2x raw decode (63.8 -> 126.4 tok/s, 71%
# acceptance on real workloads), ~1.25x on wall clock. Costs KV budget, so
# the context ceiling drops. VLLM_SPEC=0 to disable and get the full window.
SPEC="${VLLM_SPEC:-1}"
CTX="${VLLM_CTX:-131072}"
if [ "$SPEC" = "0" ]; then
  UTIL="${VLLM_UTIL:-0.92}"
  spec_args=""
else
  # n=2, not 3: the draft model's weights dominate its KV cost (5.03 GiB at
  # n=3 vs 4.88 at n=2), so dropping a token barely helps - but combined with
  # util 0.95 it's exactly enough to fit the full 131072 window. Costs ~3%
  # decode (136 -> 132 tok/s) and buys 60% more context, which matters more:
  # at 82K, opencode was compacting, and compaction forces a full re-prefill.
  UTIL="${VLLM_UTIL:-0.95}"
  spec_args="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":2}"
fi

export MODEL="${MODEL:-unsloth/Qwen3.8-27B-NVFP4}"
export SERVE_ARGS="${SERVE_ARGS:-\
--served-model-name qwen3.8-27b qwen3.8-27b-medium qwen3.8-27b-low \
--attention-backend flashinfer \
--gpu-memory-utilization $UTIL \
--max-model-len $CTX \
--kv-cache-dtype fp8_e4m3 \
--max-num-seqs 4 \
--enable-prefix-caching \
--reasoning-parser qwen3 \
--enable-auto-tool-choice \
--tool-call-parser qwen3_xml \
$spec_args}"

exec "$(dirname "$0")/serve.sh" "$@"
