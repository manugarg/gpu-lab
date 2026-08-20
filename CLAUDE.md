# GPU inference lab

## Hardware
RTX 5090, sm_120 (Blackwell consumer), 32GB VRAM.
Core Ultra 9 285K, 64GB DDR5. Ubuntu 26.04.

## Environment
- CUDA toolkit 13.3 at /usr/local/cuda-13.3. NOT 13.1 — its math
  headers conflict with 26.04 glibc (rsqrt exception spec).
- Requires TORCH_CUDA_ARCH_LIST="12.0". FlashInfer's arch check reads
  torch's build flags, not the device, and fails misleadingly without it.
- torch 2.11+cu130. NVIDIA driver 595, open kernel modules (required
  for Blackwell — proprietary has no RTX 50 support).
- Two venvs: ~/venvs/vllm is frozen and benchmark-only.
  ~/venvs/vllm-dev holds the editable checkout.

## Invocation
Don't invoke `vllm serve` / `vllm bench serve` directly or hand-set
CUDA_HOME/TORCH_CUDA_ARCH_LIST/MODEL/SERVE_ARGS — use these scripts,
which all source env/env.sh:
- `env/setup.sh` — run first. Checks nvcc is 13.3, sm_120/compute_120
  is present, torch reports cap (12, 0), and flashinfer-python/-cubin/
  -jit-cache versions match.
- `serve/serve.sh [extra vllm args]` — serves MODEL with SERVE_ARGS,
  execs so it stays in the foreground.
- `bench/run.sh <label>` — rate-4 and saturated benchmark passes
  against MODEL, results under bench/results. Use instead of ad hoc
  `vllm bench serve` calls so runs stay comparable to the baseline
  below.
- `profile/capture.sh <label> [input_len] [output_len]` — serves MODEL
  with the torch profiler attached and runs a short saturated pass;
  traces land in profile/results/<label>.
- `report.py <label>` — the normal way to read results back: prints
  bench stats vs. the baseline below (if `bench/run.sh` was run with
  this label) plus top kernels, prefill/decode split, and the
  attention/mlp/norm/quant split (if `profile/capture.sh` was). See
  profile/README.md for what each piece does individually.
- `serve/vllm-qwen38.sh` — vLLM serving Qwen3.8-27B with the flags that
  model actually needs. This is the current daily driver; see
  serve/README.md, where each flag is documented by what breaks without
  it.
- `serve/llama.sh` — llama.cpp equivalent, port 8080.
- `serve/monitor.sh` — read-only live view of throughput and GPU state.
- `serve/logproxy.py` — logs what a client actually sends, for when a
  client's docs don't say.

## Build hazards
- There are seven nvcc binaries on this box, including /usr/bin/nvcc
  (distro CUDA 12.4) and /usr/local/cuda-13.1 (the one the Environment
  note above warns against). CMake picks /usr/bin/nvcc even when
  /usr/local/cuda-13.3/bin is first on PATH, then pairs its output with
  13.3's ptxas. The mixed toolchain fails as
  "ptxas fatal : Value 'sm_52' is not defined for option 'gpu-name'",
  which reads like missing Blackwell support but is not — CUDA 13
  dropped Maxwell and the 12.4 frontend still defaults to it.
  ALWAYS pass -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc.
  PATH alone does not protect you. Confirm which one a failing build
  actually used by looking for __CUDACC_VER_MAJOR__ in its output.
- -DCMAKE_CUDA_COMPILER is NOT sufficient. It fixes the *compiler*;
  CMake still resolves the CUDA *runtime* separately and picks the
  distro's /usr/lib/x86_64-linux-gnu/libcudart.so (12.4). cudaDeviceProp
  changed layout between CUDA 12 and 13, so a 13.3-compiled binary reads
  that struct at the wrong offsets and silently gets garbage —
  multiProcessorCount comes back as **1** instead of 170.
  Nothing errors. Kernels launch with grids sized for a 1-SM GPU.
  This cost llama.cpp 5.8x prefill at 16K and 16.7x at 50K, and very
  nearly cost a blog post its thesis (2026-08-19).
  ALWAYS also pass:
    -DCUDAToolkit_ROOT=/usr/local/cuda-13.3
    -DCUDA_cudart_LIBRARY=/usr/local/cuda-13.3/lib64/libcudart.so
  VERIFY after every build — this is the only reliable check:
    ldd <built>.so | grep cudart     # must say libcudart.so.13, not .12
  CMakeCache.txt lies by omission here: CUDAToolkit_BIN_DIR can read
  /usr/local/cuda-13.3/bin while CUDA_cudart_LIBRARY points at the
  distro 12.4 one. Check both lines, not just the first.
- cmake is not installed system-wide. `uv tool install cmake` gets it
  in isolation; do NOT pip-install build tools into either vllm venv.
- A global git config rewrites https://github.com/ to git@github.com:
  (`url.git@github.com:.insteadof`), so even https clones need the SSH
  agent. This shell does not inherit a working SSH_AUTH_SOCK and
  ~/.bashrc returns early for non-interactive shells, so sourcing it
  does nothing — set SSH_AUTH_SOCK="$HOME/.ssh/ssh-agent.socket".

## Packaging rules
- ALWAYS use --no-deps for repairs. Re-resolution is what breaks this
  environment; it has cost hours twice.
- flashinfer-python, -cubin, -jit-cache must be the same version.
  -cubin comes from https://flashinfer.ai/whl (no CUDA suffix),
  -jit-cache from https://flashinfer.ai/whl/cu130. Neither is on PyPI.
- Never suggest FLASHINFER_DISABLE_VERSION_CHECK=1.

## Known sm_120 gaps
(AWQ/GPTQ/W4A16/Marlin/NVFP4 terms below: see notes/quantization.md)
- DeepGEMM asserts "Unknown SF transformation" on some weight-loading
  paths. VLLM_USE_DEEP_GEMM=0 falls back to CUTLASS — a few percent
  slower, fully working.
- NVFP4 dense linear layers do NOT fall back to Marlin on sm_120 —
  verified 2026-08-14 against vllm-dev (v0.26.0) + flashinfer
  0.6.16.post3 by reading vllm/model_executor/kernels/linear/
  __init__.py's `_POSSIBLE_NVFP4_KERNELS[CUDA]` priority list and
  running the actual capability checks on this GPU:
  - FlashInferCuteDsl: needs capability *family* 100 (exact match,
    datacenter Blackwell only) — fails on sm_120.
  - FlashInferB12x: the native SM120 kernel. Present in our FlashInfer
    build (`Sm120B12xBlockScaledDenseGemmKernel` exists) but hardcoded
    out of auto-selection in vLLM pending an upstream CUTLASS SM121
    MMA-op-guard bug. Opt in explicitly with
    `--linear-backend flashinfer_b12x` if benchmarking it.
  - FlashInferCutlass: needs `has_device_capability(100)`, which is a
    `>=` check (120 passes) — `cutlass_scaled_mm_supports_fp4()`
    returns True on this GPU. **This is the kernel that actually gets
    picked.** A real native FP4 GEMM path, not an emulation.
  - Marlin is 4th in the list and never reached for dense NVFP4.
  This was previously believed to fall back to Marlin; that was
  wrong, or became wrong. NVFP4 MoE is a separate, still-messier
  story (open vLLM issue #31085 re: SM120 MoE backend selection,
  assorted FlashInfer SM120 MoE correctness bugs) — irrelevant to
  dense models but don't generalize the dense finding to MoE.

## Current deployment (2026-08-19)

Qwen3.8-27B served by **llama.cpp** on :8080 via `serve/llama.sh`, under
the `llama-qwen38.service` user unit, driving opencode. MTP on,
**229376** context, q8_0 KV.

Switched from vLLM 2026-08-19 for the context window. Matched
measurement at 50K — both engines, MTP on, realistic fixture from
`bench/make_fixture.py`:

| @50K | llama.cpp | vLLM |
|---|---|---|
| prefill | 2,659 tok/s | **5,962 tok/s** (2.2x) |
| decode | **91.8 tok/s** | 89.8 tok/s (tie) |
| context | **229,376 deployed** (262,144 max) | 131,072 |
| perplexity (matched protocol) | 2.2542 +/- 0.0876 | 2.2832 |

Decode is a tie and quality is inside the error bar, so the trade is
vLLM's 2.2x prefill against 1.75x the context. Prefix caching absorbs
most prefill after a session's first turn, which is why context won.

**262144 + MTP does not fit** — it OOMs on the KV cache allocation at
load. 229376 leaves ~1.1 GiB headroom of 32 GiB. The draft head costs
~3 GiB; without MTP the full 262144 fits.

To switch back: `systemctl --user disable --now llama-qwen38.service &&
systemctl --user enable --now vllm-qwen38.service`. The units
`Conflicts=` each other, so they will not both run. Then in
`opencode.json` change **all four** engine-dependent fields:

| field | llama.cpp | vLLM |
|---|---|---|
| `options.baseURL` | `http://127.0.0.1:8080/v1` | `http://127.0.0.1:8000/v1` |
| `limit.context` (x3) | 229376 | 131072 |
| `name` (display label) | `llama.cpp (local)` | `vLLM (local)` |

The provider *key* is `llama-cpp-local` regardless — renaming it would
break existing opencode sessions, so it is not an engine indicator. The
`name` field is what the UI shows, and it silently went stale across the
last switch.

Numbers from before 2026-08-19 came from a build that read
`multiProcessorCount` as 1 (see "Build hazards") and are retracted: the
old "~20 min vs 4 min" PR review, "vLLM prefills 15-40x faster", and the
2026-08-17 vLLM set (8,333 / 6,713 / 69.8 / 63.8). The 69.8 decode is
contradicted by Prometheus, which has vLLM at median 75.5, p90 111.2,
peak 127.1 tok/s. Don't quote any of them.

Things that cost real time to learn, all in notes/ and serve/README.md:

- vLLM's remaining prefill advantage is **the GEMM**, not attention and
  not Gated DeltaNet. Profiled on the fixed build at 50K: `mul_mat_q`
  58.3%, flash-attn 21.0%, GDN 10.4%. NVFP4 through CUTLASS on native
  FP4 units beats Q5_K through int8 IMMA with k-quant unpacking. The
  older claim that GDN's unchunked kernel dominated was measured on the
  broken build and is wrong.
- **Benchmark fixtures can inflate speculative decoding.** The old
  `dec-ctx50k.json` / `vd-50k.json` are random words; the model echoes
  the prompt back, draft acceptance hits ~100%, and decode throughput
  measures nothing (llama.cpp read 147 tok/s emitting a copy of its own
  input). Use `bench/make_fixture.py`. Sanity check: real acceptance is
  ~49%, and 100% means the number is junk.
- **Always state the context length a number was taken at** — but note
  the two examples this rule used to cite were themselves artifacts of
  the broken build and are now retracted. On a correct build, MTP holds
  ~2x at every context (not 2.07x -> 1.06x), and KV dtype is a ~2%
  spread with f16 marginally *fastest* at 16K and 50K (not "f16 is 3.7x
  slower at 50K"). Re-measured 2026-08-19; see notes/llamacpp-vs-vllm.md.
- Prefix caching dominates agentic workloads (93% hit rate, TTFT 0.38s).
  It was **off** by default in vLLM; llama.cpp enables it by default.
- Restarting a server wipes its KV cache, so the next turn re-prefills
  everything. Don't restart mid-session and then time the next turn.

Keep `opencode.json`'s `limit.context` equal to the server's actual
context (currently **229376**, set in `llama-qwen38.service`). Both
engines reject over-length requests rather than truncating. opencode
also needs `baseURL` on the right port: **:8080** for llama.cpp, :8000
for vLLM.

llama.cpp **ignores the `model` field**, so the `qwen3.8-27b-medium` /
`-low` entries in `opencode.json` keep working even though `/v1/models`
lists only `qwen3.8-27b`. Their `chat_template_kwargs.reasoning_effort`
is honored: measured 1,607 / 2,412 / 6,341 chars of reasoning for
low / medium / high on the same prompt.

MTP is worth ~2x decode on both engines at every context measured on a
correct build. On llama.cpp it is `LLAMA_SPEC=draft-mtp`; on vLLM it is
`num_speculative_tokens: 2` with `--gpu-memory-utilization 0.95` (n=3
needs 5.03 GiB of KV and forces context to 82K, where opencode starts
compacting — and compaction rewrites the prefix, a full cache miss).

Draft acceptance is the health check for MTP: ~49% on realistic work.
100% means the fixture is letting the model copy its input.

## Baseline (Qwen3-14B-FP8)
Qwen3-14B-FP8, 1024/256, rate 4: 1011 tok/s out, TPOT 15.9ms,
TTFT 110ms mean. Any change regresses against this.

This is the *14B* baseline, used by `bench/run.sh` and `report.py`.
Qwen3.8-27B is a different model with its own numbers — see
"Current deployment" above and notes/llamacpp-vs-vllm.md. Don't compare
across the two.
