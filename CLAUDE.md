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

## Current deployment (2026-08-17)

Qwen3.8-27B served by **vLLM** on :8000 via `serve/vllm-qwen38.sh`,
driving opencode. Chosen on measurement, not preference:

| | llama.cpp | vLLM |
|---|---|---|
| same PR review, wall clock | ~20 min | **4 min** |
| prefill @50K context | ~165 tok/s | **6,713 tok/s** |
| perplexity (matched protocol) | 2.2542 +/- 0.0876 | 2.2832 |

The quality difference is +1.29% against a +/-3.9% error bar — nothing.
The speed difference is ~4x end to end.

Things that cost real time to learn, all in notes/ and serve/README.md:

- vLLM's prefill advantage on *this* model is mostly one unchunked
  llama.cpp kernel (`gated_delta_net.cu:180` says so in a TODO) covering
  48 of 64 layers. Don't generalise it to all-full-attention models.
- **Measurement context invalidates most published numbers**, including
  several of mine. MTP is 2.07x at 26 tokens and 1.06x at 50K; f16 KV is
  the fastest choice at trivial context and 3.7x the slowest at 50K.
  Always state the context length a number was taken at.
- Prefix caching dominates agentic workloads (93% hit rate, TTFT 0.38s)
  and was **off** by default in vLLM.
- Restarting a server wipes its KV cache, so the next turn re-prefills
  everything. Don't restart mid-session and then time the next turn.

Keep `opencode.json`'s `limit.context` equal to the server's actual
`--max-model-len` (currently **131072**). vLLM **rejects** over-length
requests rather than truncating.

MTP runs at `num_speculative_tokens: 2` and
`--gpu-memory-utilization 0.95`, which fits the full window *and* keeps
~97% of the speedup (132 vs 136 tok/s). n=3 needs 5.03 GiB of KV and
forces context down to 82K, where opencode starts compacting — and
compaction rewrites the prefix, which is a full cache miss.

## Baseline (Qwen3-14B-FP8)
Qwen3-14B-FP8, 1024/256, rate 4: 1011 tok/s out, TPOT 15.9ms,
TTFT 110ms mean. Any change regresses against this.

This is the *14B* baseline, used by `bench/run.sh` and `report.py`.
Qwen3.8-27B is a different model with its own numbers — see
"Current deployment" above and notes/llamacpp-vs-vllm.md. Don't compare
across the two.
