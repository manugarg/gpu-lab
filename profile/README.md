# Profiling workflow

1. `../env/setup.sh` — confirm the environment is sane before capturing.
2. `./capture.sh <label> [input_len] [output_len]` — serves MODEL with
   the torch profiler attached, runs a short saturated pass
   (`--request-rate inf --num-prompts 20`), then exits. Defaults to
   1024/256 (the baseline shape in ../CLAUDE.md) if lengths are
   omitted. Fails fast if something's already listening on :8000.
3. Traces and a `meta.txt` (shape + timestamp) land in
   `results/<label>/` — gitignored, local only.
4. `../bench/run.sh <label>` (optional) — same label, adds throughput
   numbers to the report below.
5. `python3 ../report.py <label>` — prints everything in one shot:
   bench stats (if `bench/run.sh` was run with this label) vs. the
   ../CLAUDE.md baseline, top kernels, prefill/decode split, and the
   attention/mlp/norm/quant component split. This is the normal way
   to look at a capture — the scripts below are what it calls, useful
   standalone if you only want one piece.

## What `report.py` runs

- `analyze.py <trace>` — top 25 kernels by total duration and % of
  tracked time.
- `phases.py <trace>` — splits GPU time into prefill / decode / mixed
  using the `execute_context_<n>(<ctx_tok>)_generation_<n>(<gen_tok>)`
  annotations vLLM emits per forward call (see `gpu_worker.py`'s
  `annotate_context_manager` call), plus the idle/scheduling gap
  between calls. "Mixed" steps interleave both in one call (continuous
  batching) and can only be split approximately, by token share.
- `components.py <trace>` — splits GPU time by component: attention
  (core/kv-cache), MLP, attention-proj, norm, quant, sampling. The
  attention/mlp GEMM split can't be read off the call stack
  (torch.compile fuses per-layer Python calls into opaque compiled
  graphs), so it's inferred from each GEMM kernel's (N, K) shape
  against the projection shapes computed from the MODEL's cached HF
  config.json — exact for this model, but will silently show as
  "gemm (unrecognized shape ...)" for an architecture this script
  doesn't know how to derive shapes for (e.g. MoE). deep_gemm's
  split-k reduce kernels carry no shape of their own; they inherit
  the category of the GEMM immediately before them on the same CUDA
  stream.
- `../bench/summarize.py <label>` — reads `bench/results/<label>*.json`
  (written by `bench/run.sh`) and diffs throughput/TTFT/TPOT against
  the ../CLAUDE.md baseline for the `rate4` shape.

## Picking a label/shape

Use input/output lengths that isolate the phase you care about:
- `decode_dominated`: short input, long output (e.g. `32 1024`)
- `prefill_dominated`: long input, minimal output (e.g. `1024 1`)
- default (1024/256) matches the baseline benchmark shape

Example:

```
./capture.sh decode_dominated 32 1024
python3 ../report.py decode_dominated
```

Re-running the same label overwrites its `results/<label>/` — rename
the label if you want to keep both traces side by side.
