# Profiling workflow

1. `../env/setup.sh` — confirm the environment is sane before capturing.
2. `./capture.sh <label> [input_len] [output_len]` — serves MODEL with
   the torch profiler attached, runs a short saturated pass
   (`--request-rate inf --num-prompts 20`), then exits. Defaults to
   1024/256 (the baseline shape in ../CLAUDE.md) if lengths are
   omitted. Fails fast if something's already listening on :8000.
3. Traces and a `meta.txt` (shape + timestamp) land in
   `results/<label>/` — gitignored, local only.
4. `python3 analyze.py results/<label>/trace.pt.trace.json.gz` —
   prints the top 25 kernels by total duration and % of tracked time.
5. `python3 phases.py results/<label>/trace.pt.trace.json.gz` —
   splits GPU time into prefill / decode / mixed using the
   `execute_context_<n>(<ctx_tok>)_generation_<n>(<gen_tok>)`
   annotations vLLM emits per forward call (see
   `gpu_worker.py`'s `annotate_context_manager` call), plus the
   idle/scheduling gap between calls. "Mixed" steps interleave both
   in one call (continuous batching) and can only be split
   approximately, by token share.

## Picking a label/shape

Use input/output lengths that isolate the phase you care about:
- `decode_dominated`: short input, long output (e.g. `32 1024`)
- `prefill_dominated`: long input, minimal output (e.g. `1024 1`)
- default (1024/256) matches the baseline benchmark shape

Example:

```
./capture.sh decode_dominated 32 1024
python3 analyze.py results/decode_dominated/trace.pt.trace.json.gz
```

Re-running the same label overwrites its `results/<label>/` — rename
the label if you want to keep both traces side by side.
