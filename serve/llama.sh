#!/usr/bin/env bash
# Long-running llama.cpp server. Flags are explained in serve/README.md.
# Runs in the foreground and execs, so systemd/tmux can supervise it.
set -euo pipefail

source "$(dirname "$0")/../env/env.sh"

PORT="${LLAMA_PORT:-8080}"
HOST="${LLAMA_HOST:-127.0.0.1}"
CTX="${LLAMA_CTX:-262144}"

[ -x "$LLAMA_BIN" ] || { echo "no llama-server at $LLAMA_BIN (build it: see serve/README.md)" >&2; exit 1; }

# Resolve the GGUF through the HF cache rather than hardcoding a snapshot
# hash, which changes whenever the repo is re-downloaded.
# `|| true`: under `set -e` a failing ls here aborts the script with ls's
# exit code before the check below can print anything useful.
GGUF="${LLAMA_GGUF:-$(ls -1 "$HF_HOME"/hub/"$LLAMA_GGUF_REPO"/snapshots/*/"$LLAMA_GGUF_FILE" 2>/dev/null | head -1 || true)}"
[ -n "$GGUF" ] && [ -f "$GGUF" ] || {
  echo "GGUF not found: $LLAMA_GGUF_REPO/$LLAMA_GGUF_FILE under $HF_HOME/hub" >&2
  echo "download it with: hf download unsloth/Qwen3.8-27B-GGUF $LLAMA_GGUF_FILE" >&2
  exit 1
}

curl -sf "localhost:$PORT/health" >/dev/null && { echo "server already on :$PORT"; exit 1; }

# Both engines want the whole GPU; a running vLLM will starve this one and
# the OOM it produces points at the KV cache rather than the real cause.
used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$used_mib" -lt 1000 ] || {
  echo "GPU already has ${used_mib} MiB in use - stop the other server first" >&2
  exit 1
}

exec "$LLAMA_BIN" \
  -m "$GGUF" \
  --alias "${LLAMA_ALIAS:-qwen3.8-27b}" \
  -ngl 99 \
  -c "$CTX" \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  --host "$HOST" --port "$PORT" \
  "$@"
