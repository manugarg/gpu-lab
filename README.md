# gpu-lab

LLM inference experiments on a single RTX 5090 (32 GB, sm_120 / consumer
Blackwell). Everything recorded here is measured on this machine — no
published numbers are repeated without being re-checked, and several
turned out not to reproduce.

`CLAUDE.md` has the environment, build hazards, and what's currently
deployed.

## Currently deployed

Qwen3.8-27B served by vLLM on :8000, driving opencode:

```
./serve/vllm-qwen38.sh          # vLLM  + MTP, 82K context   (daily driver)
./serve/llama.sh                # llama.cpp alternative, :8080
./serve/monitor.sh              # live throughput + GPU view
```

The same PR review takes **~20 min on llama.cpp, 4 min on vLLM**, with no
measurable quality difference between the two quantizations
(perplexity 2.2542 vs 2.2832, inside the error bar).

## Scripts

| path | what it does |
|---|---|
| `serve/vllm-qwen38.sh` | vLLM for Qwen3.8-27B, flags this model needs |
| `serve/llama.sh` | llama.cpp server, port 8080 |
| `serve/serve.sh` | generic vLLM launcher (uses `env/env.sh`) |
| `serve/monitor.sh` | read-only live view: slots, GPU, prefill/decode |
| `serve/logproxy.py` | logs what a client actually sends |
| `bench/run.sh` · `bench/summarize.py` | throughput benchmark vs the 14B baseline |
| `profile/capture.sh` · `report.py` | torch-profiler capture and analysis |
| `env/setup.sh` | verifies the CUDA/FlashInfer toolchain |

`serve/README.md` documents every serving flag by *what breaks without
it* — that's where the time went.

## Notes

Explanatory, in rough reading order:

| note | about |
|---|---|
| `notes/inference-anatomy.md` | prefill vs decode, why they behave differently |
| `notes/prefill-explained.md` | prefill in detail |
| `notes/qkv-and-attention.md` | QKV projection vs attention |
| `notes/hybrid-attention-and-kv-cache.md` | why a 27B hybrid model needs *less* KV cache than a 14B dense one |
| `notes/quantization.md` | AWQ, GPTQ, W4A16, Marlin, NVFP4 — which are algorithms, formats, or kernels |
| `notes/perplexity.md` | how to compare quantization quality, and the traps |
| `notes/qwen3.8-27b-options.md` | checkpoint survey and VRAM budgets |
| `notes/llamacpp-vs-vllm.md` | the measurements, including the ones that killed my own hypotheses |

## Three things worth knowing before trusting any benchmark

**Measurement context invalidates most published numbers.** MTP is 2.07x
at 26 tokens of context and 1.06x at 50K. f16 KV cache is the *fastest*
choice at trivial context and 3.7x the *slowest* at 50K. Both readings
are correct; neither generalises. Always state the context length.

**Big engine gaps often trace to one kernel, not engine quality.** vLLM
prefills 15-40x faster than llama.cpp *on this model* largely because
llama.cpp's Gated DeltaNet kernel is unchunked — its own source says
`//TODO: Add chunked kernel for even faster pre-fill` — and 48 of this
model's 64 layers use it. Don't generalise to dense models.

**Prefix caching dominates agentic workloads** (93% hit rate, 0.38s mean
TTFT) and was off by default. Restarting a server wipes it, so don't
restart mid-session and then time the next turn.
