# Serving

Two engines, one GPU. They can't run at the same time — 27B weights plus
KV cache fills the card either way, so `llama.sh` refuses to start if the
GPU already has memory in use.

- `serve.sh` — vLLM, port 8000. **Prefills 15-40x faster than llama.cpp
  at long context and barely degrades** (8,333 tok/s at 16K, 6,713 at
  50K), and holds decode nearly flat. For agentic coding — long contexts,
  prefill-dominated — this is the one to use.
- `llama.sh` — llama.cpp, port 8080. Reaches 262K context (vs vLLM's
  189K here) and is operationally simpler — one binary, no venv. But its
  prefill collapses with context (567 tok/s at 16K, ~165 at 50K), so a
  68K-token prefill costs ~7 minutes against vLLM's ~10 seconds. Prefer
  it only for short contexts or when you need the extra context ceiling.

Measured comparison behind those claims: `notes/llamacpp-vs-vllm.md`.

## vLLM for Qwen3.8-27B (`vllm-qwen38.sh`)

```
./serve/vllm-qwen38.sh            # MTP on, 82K context
VLLM_SPEC=0 ./serve/vllm-qwen38.sh   # MTP off, 131K context
```

Every flag below was needed because leaving it out fails in a way that
does *not* look like a missing flag. That's the reason this is a script
and not a line in a README.

| flag | what breaks without it |
|---|---|
| `--reasoning-parser qwen3` | Thinking is emitted into `content`. Your first reply is literally `"We need answer user's simple math..."` — it looks like the model is broken, not like a config gap. |
| `--enable-auto-tool-choice` + `--tool-call-parser qwen3_xml` | Any request with `tools` returns **400**: *"auto tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set"*. The parser must be `qwen3_xml`, **not** the more common `hermes` — this model's template emits `<function=name><parameter=key>` XML, not hermes-style JSON. Check the template, don't guess. |
| `--served-model-name qwen3.8-27b …` | vLLM **validates** model names and 404s on unknown ones, where llama.cpp silently serves whatever is loaded. Without aliases the opencode effort presets (`-low`, `-medium`) break. |
| `--enable-prefix-caching` | It was **off** in our first runs. With it, a real session hit **80-93%** cache and mean TTFT of 0.4-1.3s; without it every turn re-prefills the whole history. Single biggest win for agentic use. |
| `--max-num-seqs 4` | vLLM's default drives CUDA-graph capture to a batch size that OOMs at load on 32 GB. The error points at the KV cache, not at concurrency. |
| `--kv-cache-dtype fp8_e4m3` | Halves KV, which matters because decode cost scales with KV read. |
| `--max-model-len` | See the MTP tradeoff below. |

### The MTP tradeoff

`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` uses
the model's built-in draft head:

| | MTP off | MTP on |
|---|---|---|
| decode | 63.8 tok/s | **126.4 tok/s** (2x) |
| draft acceptance | — | 83% synthetic, **71% real workload** |
| max context | **131,072** | 81,920 |
| PR review wall clock | 5 min | **4 min** (1.25x) |
| tool calling | works | works |

Raw decode doubles but wall clock only improves ~1.25x, because decode
isn't all of the wall time once tool execution, prefill and idle between
turns are counted. Don't expect the 2x to show up end-to-end.

**131072 + MTP does not start** — the draft model needs ~5 GiB of the KV
budget and vLLM refuses with a `ValueError` naming the shortfall. Hence
the context default following `VLLM_SPEC`.

Note vLLM **rejects** a request that exceeds `--max-model-len` rather
than truncating it, so 82K is a hard ceiling, not a soft one.

Also logged at startup and worth knowing: `CUDAGraphMode.FULL_AND_PIECEWISE
is not supported with spec-decode for FlashInferBackend`, so it falls back
to a narrower graph mode. There may be more performance available with a
different attention backend — untested.

### Field-name difference from llama.cpp

vLLM returns reasoning in **`reasoning`**; llama.cpp uses
**`reasoning_content`**. `content` is clean on both, so nothing breaks,
but a client that reads one name won't render the other's thinking.

## `llama.sh` flags

Everything below was checked against `llama-server --help` on the build
in `~/tools/llama.cpp`; llama.cpp renames and deprecates flags fairly
often, so re-check rather than copying from blog posts.

| flag | why |
|---|---|
| `-m <gguf>` | The model file. Resolved through the HF cache by repo/filename so a re-download (new snapshot hash) doesn't break the script. |
| `--alias qwen3.8-27b` | The model name the API reports. Without it clients see the full filesystem path. |
| `-ngl 99` | Offload *all* layers to GPU. "99" is the idiom for "more layers than the model has" — anything left on CPU tanks generation speed. |
| `-c 229376` | Context size. With MTP off this could go to 262144; left here so enabling MTP doesn't OOM. Note prefill gets slower the deeper the context, so bigger isn't free — see the prefill section in `notes/llamacpp-vs-vllm.md`. |
| `--spec-type draft-mtp` | Speculative decoding via the model's built-in MTP head. Lossless, so speed-only. **Its benefit decays with context**: measured 2.07x at ~26 tokens, 1.31x at 16K, 1.06x at 50K. Worth enabling below ~16K, not above ~30K — at long context KV attention dominates and there's little left to amortise. Costs 3.3 GiB and caps context at 229376, so it defaults to `none`; `LLAMA_SPEC=draft-mtp` enables it. |
| `-fa on` | Flash attention. Required for quantized V cache; also cuts attention memory. Default is `auto`, which won't reliably give you it. |
| `-ctk q8_0 -ctv q8_0` | Quantize the KV cache to 8-bit. **Not optional**, for two reasons. Memory: f16 KV is 64 KB/token, so 262K wants a 16 GiB allocation and OOMs. Speed: f16 is **2.3x slower at 16K and 3.7x slower at 50K** (decode is bandwidth-bound, KV size dominates). `q4_0` was measured too — it buys ~1 GiB but no speed over q8_0, so q8_0 is the right stop. |
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

## Watching it run

`serve/monitor.sh` — live view, read-only (HTTP GETs + `nvidia-smi`, it
never touches the running server):

```
./serve/monitor.sh
LLAMA_LOG=/path/to/server.log ./serve/monitor.sh
```

```
server:  UP
slots:   1/4 processing   n_ctx=229,376  speculative=True
gpu:     98% util  30.7/31.8 GiB  216.80W  54°C  throttle=none
prefill: [##########....................]  34%  23,000/68,000 tok
         92 tok/s   elapsed 250s   eta ~489s
decode:  14.9 tok/s overall   14.3 tok/s last-3s   (1,970 generated)
mtp:     79% accepted  (71/90)  mean draft len 3.37
```

Where each number comes from:

| panel | source | notes |
|---|---|---|
| slots | `/slots` | `is_processing` is the authoritative "is the GPU busy" flag. There is no `state` field — guessing at one is how I once concluded the GPU was idle while opencode was hammering it. |
| gpu | `nvidia-smi` | Watch **power**: prefill should be compute-bound and pull high wattage. ~150W on a ~575W card means the GPU is stalling, not computing. |
| prefill / decode / mtp | server **stdout** | Only present if stdout was captured. Under systemd: `journalctl --user -u llama -f > /tmp/llama.log &` then point `LLAMA_LOG` at it. |

`tg` vs `tg_3s` is the useful pair — lifetime average versus the last
three seconds, so you can watch decode degrade *within* one response.

Low MTP acceptance is called out explicitly: below ~50% speculation is
likely costing more than it saves, and `LLAMA_SPEC=none` is worth trying.

`/metrics` (Prometheus) is enabled by default via `LLAMA_METRICS`, for
scraping into Grafana later. It returns **501** on a server started
without it — that's "not enabled", not "broken".

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

### Restarting

`kill` then immediately re-run does **not** work naively: after SIGTERM
the port stays bound and 21 GB of weights take several seconds to
release. The script waits up to `LLAMA_WAIT` (60s) for both to clear
before starting, so a restart is just:

```
pkill -f '[l]lama-server'; ./serve/llama.sh
```

(Use the `[l]` bracket form — a plain `pkill -f llama-server` also
matches the shell running it and kills your own command.)

The wait only applies to a *transient* state. A server actually
answering `/health` still fails immediately with `server already on
:PORT`, rather than waiting 60s for something that isn't going away.

This exists because the naive version bit us: a restart raced the
shutdown, the guard correctly refused to double-start, the old server
then finished exiting, and the box was left with nothing serving.

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
| `LLAMA_WAIT` | `60` — seconds to wait for the port/GPU to free on restart |
| `LLAMA_METRICS` | `1` — Prometheus `/metrics` endpoint; `0` disables |
| `LLAMA_SSL_CERT` / `LLAMA_SSL_KEY` | unset (TLS off; set both or neither) |

Smaller context to free VRAM for something else:
```
LLAMA_CTX=32768 ./serve/llama.sh
```

## Building llama.cpp

See `notes/llamacpp-vs-vllm.md`. The one thing you cannot skip is
`-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc` — see CLAUDE.md's
"Build hazards" for why the default pick fails in a misleading way.
