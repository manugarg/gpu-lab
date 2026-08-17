# Serving

Two engines, one GPU. They can't run at the same time — 27B weights plus
KV cache fills the card either way, so `llama.sh` refuses to start if the
GPU already has memory in use.

- `serve.sh` — vLLM, port 8000. Faster prefill (~3.8x), much better under
  concurrency. Use for benchmarking and anything multi-user.
- `llama.sh` — llama.cpp, port 8080. ~154 tok/s single-stream with MTP
  speculative decoding, 224K context, lower footprint, single binary.
  Use for private/interactive single-user work. (Can reach the full 262K
  instead, at 2.4x lower speed — see `--spec-type` below.)

Measured comparison behind those claims: `notes/llamacpp-vs-vllm.md`.

## `llama.sh` flags

Everything below was checked against `llama-server --help` on the build
in `~/tools/llama.cpp`; llama.cpp renames and deprecates flags fairly
often, so re-check rather than copying from blog posts.

| flag | why |
|---|---|
| `-m <gguf>` | The model file. Resolved through the HF cache by repo/filename so a re-download (new snapshot hash) doesn't break the script. |
| `--alias qwen3.8-27b` | The model name the API reports. Without it clients see the full filesystem path. |
| `-ngl 99` | Offload *all* layers to GPU. "99" is the idiom for "more layers than the model has" — anything left on CPU tanks generation speed. |
| `-c 229376` | Context size — 224K, not the full 262144. MTP below needs ~1 GiB for its draft context and 262144 + MTP OOMs. Measured trade: 2.4x decode speed for 13% less context. `LLAMA_SPEC=none LLAMA_CTX=262144` reverses it. |
| `--spec-type draft-mtp` | Speculative decoding using the model's built-in MTP head — **measured 64.4 -> 154.1 tok/s (2.4x)** with 86-88% draft acceptance. Lossless: verification guarantees the same output the model would have produced, so this is pure speed, not a quality trade. No separate draft model needed; the head ships in the GGUF (`qwen35.nextn_predict_layers`). `LLAMA_SPEC=none` disables. |
| `-fa on` | Flash attention. Required for quantized V cache; also cuts attention memory. Default is `auto`, which won't reliably give you it. |
| `-ctk q8_0 -ctv q8_0` | Quantize the KV cache to 8-bit. **Not optional at 262K.** llama.cpp's default f16 KV is 64 KB/token, so 262K wants a 16 GiB allocation and OOMs outright. q8_0 halves that to ~8 GiB and fits (28.7 GiB of 32.1 GiB total). |
| `--host` | `0.0.0.0` by default here (see "Remote access"), so LAN machines can reach it. `LLAMA_HOST=127.0.0.1` for local-only. |
| `--port 8080` | llama.cpp's default; deliberately not vLLM's 8000, so the port check distinguishes the two. |

Anything extra is passed straight through, e.g.:

```
./serve/llama.sh --api-key "$(cat ~/.llama-key)" --metrics
```

### Flags deliberately *not* set

- `--jinja` — already enabled by default on this build. Passing it is
  harmless but implies it's doing something it isn't.
- `--no-mmap` / `--mlock` — **deprecated** in favour of `--load-mode`.
  Advice to set `--no-mmap` for this model (on the theory that recurrent
  layers interact badly with mmap) is wrong twice over: the flag is
  deprecated, and mmap governs weight loading, not the recurrent state.
- `-np` / `--parallel` — defaults to auto. The server allocates unified
  KV across slots, so a single user gets the whole context regardless.
- `-t` / `--threads` — irrelevant at `-ngl 99`; generation isn't on CPU.

## It's a reasoning model — budget for thinking tokens

Qwen3.8 thinks by default, and the thinking is returned separately from
the answer. This surprises you as an *empty* reply rather than an error:

```
max_tokens: 24  -> content: ''      (all 24 tokens went to thinking)
max_tokens: 200 -> content: 'ok'    reasoning_content: 'We need to respond...'
                                    completion_tokens: 28
```

So `max_tokens` has to cover thinking *plus* the answer, and clients that
only read `message.content` will silently show nothing when the budget is
too small. The thinking is in `message.reasoning_content`.

To turn thinking off per request (verified — 2 completion tokens instead
of 28):

```json
{"messages": [...], "chat_template_kwargs": {"enable_thinking": false}}
```

Server-wide equivalents exist too — `--reasoning off`, or
`--reasoning-budget N` to cap thinking length — see `llama-server --help`.
Left unset in `llama.sh` so callers decide per request.

### Reasoning *effort* is separate from reasoning on/off

The chat template defaults to the **highest** setting:

```jinja
{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
```

Override per request (verified against `/apply-template` — the rendered
prompt really does change):

```json
{"chat_template_kwargs": {"reasoning_effort": "low"}}
```

This template accepts **only** `xhigh`, `medium`, `low`. It maps `high`
-> `xhigh` and raises on anything else, so `llama-server --help`'s
advertised `minimal` / `max` will error with this model. `medium` emits
no effort instruction at all (the model's natural behaviour).

Note for API clients: a `"reasoning": true` capability flag (opencode's,
for instance) only says "this model emits thinking, parse it separately".
It does not set effort — effort is decided model-side by the template
from `reasoning_effort`.

## Seeing what a client actually sends

`serve/logproxy.py` is a logging reverse proxy for when a client's docs
don't tell you which parameters it emits:

```
python3 serve/logproxy.py          # :8081 -> :8080
```

Point the client's baseURL at `http://127.0.0.1:8081/v1`, send one
message, and it prints the request body (messages elided) plus whether
`chat_template_kwargs` / `reasoning*` appear. Streaming passes through
unbuffered, so the client keeps working while you watch.

**What it settled for opencode** (v1.18.18): per-model `options` *are*
forwarded verbatim into the request body, flattened at top level — not
dropped, not nested under `providerOptions`. Its docs don't say this;
the proxy showed it:

```json
{"model": "qwen3.8-27b-low", "max_tokens": 16384, "temperature": 0.55,
 "chat_template_kwargs": {"reasoning_effort": "low"}, "stream": true,
 "tools": [...]}
```

So reasoning effort can be exposed as *model presets* — one entry per
effort level, all pointing at the same server, switched from opencode's
model picker:

```json
"models": {
  "qwen3.8-27b":        { "name": "Qwen3.8-27B (xhigh, default)", ... },
  "qwen3.8-27b-medium": { ..., "options": {"chat_template_kwargs": {"reasoning_effort": "medium"}} },
  "qwen3.8-27b-low":    { ..., "options": {"chat_template_kwargs": {"reasoning_effort": "low"}} }
}
```

The made-up model IDs are fine: llama-server ignores the requested model
name and serves whatever is loaded (verified — returns 200 and echoes
back `"model": "qwen3.8-27b"`). Set `limit.context` to the server's
actual `-c` value, not the model's theoretical maximum.

## Remote access

Binds to `0.0.0.0` by default (set in `env/env.sh`), so other machines
reach it directly over plain HTTP — no tunnel, no cert. Verified working
from both addresses:

```
http://192.168.86.15:8080/v1/chat/completions   # LAN
http://100.80.215.7:8080/v1/chat/completions    # tailnet (gandalf.tail07156.ts.net)
```

It's an OpenAI-compatible API, so most clients just need the base URL
`http://192.168.86.15:8080/v1` and any non-empty API key. Model name is
`qwen3.8-27b` (the `--alias`).

This is deliberate for a LAN that isn't reachable from outside. What it
means in practice: anything on the network can use the GPU and read
prompts/responses in the clear. `LLAMA_HOST=127.0.0.1` reverts to
local-only; the tunnel options below and TLS are there if the threat
model ever changes.

<details>
<summary>Tunnelled alternatives (not used here)</summary>

Tailscale terminating real TLS, server staying on localhost:
```
LLAMA_HOST=127.0.0.1 ./serve/llama.sh
tailscale serve --bg 8080   # -> https://gandalf.tail07156.ts.net/
```
SSH tunnel, for anything not on the tailnet:
```
ssh -N -L 8080:localhost:8080 manugarg@gandalf.tail07156.ts.net
```
</details>

### TLS (optional)

Requires a build with OpenSSL. The original build silently produced a
no-HTTPS binary (`OpenSSL not found, HTTPS support disabled` at configure
time, `OPENSSL_CRYPTO_LIBRARY-NOTFOUND` in CMakeCache) — the
`--ssl-cert-file`/`--ssl-key-file` flags simply don't exist on such a
binary. Fixed by installing headers and reconfiguring:
```
sudo apt install -y libssl-dev     # matches the system libssl3t64
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc -DLLAMA_OPENSSL=ON
cmake --build build --config Release -j $(nproc)
```
Verify it took rather than trusting the flag — `LLAMA_OPENSSL=ON` was
already set on the broken build:
```
ldd build/bin/llama-server | grep -E 'ssl|crypto'   # expect libssl.so.3
```

Then set both vars (the script rejects setting only one):
```
LLAMA_SSL_CERT=~/certs/gandalf.crt LLAMA_SSL_KEY=~/certs/gandalf.key ./serve/llama.sh
```
The server logs `listening on https://...` and plain HTTP stops working.

For a **real** cert rather than self-signed, Tailscale will issue one for
the tailnet name (needs HTTPS enabled in the tailnet admin console):
```
tailscale cert gandalf.tail07156.ts.net
```

## Running it continuously

The script `exec`s in the foreground, which is what a supervisor wants.
Pick one:

**tmux** (simplest, dies with the machine):
```
tmux new -s llama -d './serve/llama.sh'
tmux attach -t llama
```

**systemd user unit** (survives logout, restarts on crash and at boot).
Write `~/.config/systemd/user/llama.service`:
```ini
[Unit]
Description=llama.cpp server (Qwen3.8-27B)
After=network.target

[Service]
ExecStart=%h/gpu-lab/serve/llama.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```
Then:
```
systemctl --user daemon-reload
systemctl --user enable --now llama
systemctl --user status llama
journalctl --user -u llama -f
```
`loginctl enable-linger $USER` if you want it up without being logged in.

Note `Restart=on-failure` will retry forever if the GPU is occupied by
vLLM, since the guard exits non-zero. That's intentional — it recovers on
its own once the GPU frees up — but it's why `RestartSec` is 10 rather
than 1.

## Overrides

All read from `env/env.sh`, all overridable per-invocation:

| var | default |
|---|---|
| `LLAMA_BIN` | `~/tools/llama.cpp/build/bin/llama-server` |
| `LLAMA_GGUF_REPO` | `models--unsloth--Qwen3.8-27B-GGUF` |
| `LLAMA_GGUF_FILE` | `Qwen3.8-27B-UD-Q5_K_XL.gguf` |
| `LLAMA_GGUF` | (resolved from the two above; set to bypass) |
| `LLAMA_CTX` | `229376` (224K; see `--spec-type`) |
| `LLAMA_HOST` / `LLAMA_PORT` | `0.0.0.0` / `8080` |
| `LLAMA_ALIAS` | `qwen3.8-27b` |
| `LLAMA_SPEC` | `draft-mtp` (`none` disables speculative decoding) |
| `LLAMA_SSL_CERT` / `LLAMA_SSL_KEY` | unset (TLS off; set both or neither) |

Smaller context to free VRAM for something else:
```
LLAMA_CTX=32768 ./serve/llama.sh
```

## Building llama.cpp

See `notes/llamacpp-vs-vllm.md`. The one thing you cannot skip is
`-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc` — see CLAUDE.md's
"Build hazards" for why the default pick fails in a misleading way.
