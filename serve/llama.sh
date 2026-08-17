#!/usr/bin/env bash
# Long-running llama.cpp server. Flags are explained in serve/README.md.
# Runs in the foreground and execs, so systemd/tmux can supervise it.
set -euo pipefail

source "$(dirname "$0")/../env/env.sh"

PORT="${LLAMA_PORT:-8080}"
HOST="${LLAMA_HOST:-127.0.0.1}"
# Default off. MTP measured 2.4x on short prompts with thinking disabled,
# but that benchmark doesn't resemble agentic coding, where prefill is
# ~63% of wall time and MTP only touches decode - while costing 3.3 GiB.
# LLAMA_SPEC=draft-mtp to enable.
SPEC="${LLAMA_SPEC:-none}"

# Context default follows SPEC, because MTP's draft context needs ~3 GiB
# and 262144 + MTP OOMs at load. Turning MTP on shouldn't hand you an
# OOM for a setting you didn't touch. Explicit LLAMA_CTX always wins.
# Bigger isn't free regardless: prefill slows as the context fills
# (see notes/llamacpp-vs-vllm.md).
if [ "$SPEC" = "none" ]; then
  CTX="${LLAMA_CTX:-262144}"   # full native window
else
  CTX="${LLAMA_CTX:-229376}"   # leaves room for the draft context
fi

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

# A server answering /health is genuinely serving: fail fast rather than
# waiting for something that isn't going to go away.
curl -sf "localhost:$PORT/health" >/dev/null && { echo "server already on :$PORT"; exit 1; }

port_bound() { ss -tln 2>/dev/null | grep -q ":$PORT "; }
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | grep -E '^[0-9]+$' || echo 0; }
gpu_busy() { [ "$(gpu_mib)" -ge 1000 ]; }

# Restarting races a shutting-down instance: after SIGTERM the port stays
# bound and 21 GB of weights take several seconds to release. Failing on
# that transient state once left the box with no server at all, so wait
# for it to clear instead. Both engines want the whole GPU, so a busy GPU
# is also blocking - if it's a *running* vLLM this just times out with a
# clear message rather than starting and dying on a confusing KV-cache OOM.
WAIT="${LLAMA_WAIT:-60}"
deadline=$(( $(date +%s) + WAIT ))
waited=0
while port_bound || gpu_busy; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    if port_bound; then echo "port $PORT still bound after ${WAIT}s" >&2; fi
    if gpu_busy; then echo "GPU still has $(gpu_mib) MiB in use after ${WAIT}s - stop the other server first" >&2; fi
    exit 1
  fi
  [ "$waited" -eq 0 ] && echo "waiting up to ${WAIT}s for port :$PORT and GPU to free..."
  waited=1
  sleep 2
done

# TLS is opt-in: set both LLAMA_SSL_CERT and LLAMA_SSL_KEY. Requires a
# build with OpenSSL (see serve/README.md) — without it llama-server
# doesn't expose these flags at all and would fail on unknown argument.
tls=()
if [ -n "${LLAMA_SSL_CERT:-}" ] || [ -n "${LLAMA_SSL_KEY:-}" ]; then
  [ -n "${LLAMA_SSL_CERT:-}" ] && [ -n "${LLAMA_SSL_KEY:-}" ] || {
    echo "set both LLAMA_SSL_CERT and LLAMA_SSL_KEY, or neither" >&2; exit 1; }
  for f in "$LLAMA_SSL_CERT" "$LLAMA_SSL_KEY"; do
    [ -r "$f" ] || { echo "cannot read $f" >&2; exit 1; }
  done
  tls=(--ssl-cert-file "$LLAMA_SSL_CERT" --ssl-key-file "$LLAMA_SSL_KEY")
fi

# Speculative decoding via the model's built-in MTP head. Lossless -
# verification guarantees the same output the target model would produce -
# so this is pure speed. LLAMA_SPEC=none to disable.
spec=()
[ "$SPEC" != "none" ] && spec=(--spec-type "$SPEC")

# Prometheus endpoint at /metrics. Off in llama-server by default, which
# makes /metrics return 501 rather than anything explanatory. On here so
# it's available for scraping; LLAMA_METRICS=0 to disable. Note this
# publishes usage stats on whatever LLAMA_HOST binds to.
metrics=()
[ "${LLAMA_METRICS:-1}" != "0" ] && metrics=(--metrics)

exec "$LLAMA_BIN" \
  -m "$GGUF" \
  --alias "${LLAMA_ALIAS:-qwen3.8-27b}" \
  -ngl 99 \
  -c "$CTX" \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  "${spec[@]}" \
  "${metrics[@]}" \
  --host "$HOST" --port "$PORT" \
  "${tls[@]}" \
  "$@"
