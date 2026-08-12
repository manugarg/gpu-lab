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
- DeepGEMM asserts "Unknown SF transformation" on some weight-loading
  paths. VLLM_USE_DEEP_GEMM=0 falls back to CUTLASS — a few percent
  slower, fully working.
- NVFP4 checkpoints fall back to Marlin W4A16 on sm_120. Prefer FP8/AWQ.

## Baseline
Qwen3-14B-FP8, 1024/256, rate 4: 1011 tok/s out, TPOT 15.9ms,
TTFT 110ms mean. Any change regresses against this.
