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

## Baseline
Qwen3-14B-FP8, 1024/256, rate 4: 1011 tok/s out, TPOT 15.9ms,
TTFT 110ms mean. Any change regresses against this.
