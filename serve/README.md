# Serving

Two engines, one GPU. They can't run at the same time — 27B weights plus
KV cache fills the card either way, so `llama.sh` refuses to start if the
GPU already has memory in use.

- `serve.sh` — vLLM, port 8000. Faster prefill (~3.8x), much better under
  concurrency. Use for benchmarking and anything multi-user.
- `llama.sh` — llama.cpp, port 8080. Reaches the full 262K context with
  this GGUF, lower footprint, single binary. Use for private/interactive
  single-user work.

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
| `-c 262144` | Context size. This model's full native window. Verified to fit **only** with quantized KV below. |
| `-fa on` | Flash attention. Required for quantized V cache; also cuts attention memory. Default is `auto`, which won't reliably give you it. |
| `-ctk q8_0 -ctv q8_0` | Quantize the KV cache to 8-bit. **Not optional at 262K.** llama.cpp's default f16 KV is 64 KB/token, so 262K wants a 16 GiB allocation and OOMs outright. q8_0 halves that to ~8 GiB and fits (28.7 GiB of 32.1 GiB total). |
| `--host 127.0.0.1` | Localhost only. This is already llama.cpp's default, but it's set explicitly because it's a security property worth being able to see. Change it to `0.0.0.0` **only** if you actually want other machines to reach it — that exposes an unauthenticated LLM to your network. Pair with `--api-key` if you do. |
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
| `LLAMA_CTX` | `262144` |
| `LLAMA_HOST` / `LLAMA_PORT` | `127.0.0.1` / `8080` |
| `LLAMA_ALIAS` | `qwen3.8-27b` |

Smaller context to free VRAM for something else:
```
LLAMA_CTX=32768 ./serve/llama.sh
```

## Building llama.cpp

See `notes/llamacpp-vs-vllm.md`. The one thing you cannot skip is
`-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc` — see CLAUDE.md's
"Build hazards" for why the default pick fails in a misleading way.
