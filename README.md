# gpu-lab

LLM inference experiments on a single RTX 5090 (32 GB, sm_120 / consumer
Blackwell). Everything recorded here is measured on this machine — no
published numbers are repeated without being re-checked, and several
turned out not to reproduce.

`CLAUDE.md` has the environment, build hazards, and what's currently
deployed.

## Currently deployed

Qwen3.8-27B served by llama.cpp on :8080, driving opencode. Runs under
the `llama-qwen38.service` user unit, so it survives logout:

```
./serve/llama.sh                # llama.cpp + MTP, 229K context  (daily driver)
./serve/vllm-qwen38.sh          # vLLM alternative, :8000
./serve/monitor.sh              # live throughput + GPU view
```

Matched measurement at 50K context — both engines, MTP on, realistic
fixture:

| @50K | llama.cpp | vLLM |
|---|---|---|
| prefill | 2,659 tok/s | **5,962 tok/s** (2.2x) |
| decode | **91.8 tok/s** | 89.8 tok/s (tie) |
| context | **229,376 deployed** | 131,072 |

Deployed on llama.cpp for the context window: decode is a tie, quality
is inside the error bar (perplexity 2.2542 vs 2.2832), and prefix caching
absorbs most of vLLM's prefill advantage after a session's first turn.
229,376 is the ceiling with MTP enabled — 262,144 OOMs on the KV cache.

## Scripts

| path | what it does |
|---|---|
| `serve/vllm-qwen38.sh` | vLLM for Qwen3.8-27B, flags this model needs |
| `serve/llama.sh` | llama.cpp server, port 8080 |
| `serve/serve.sh` | generic vLLM launcher (uses `env/env.sh`) |
| `serve/monitor.sh` | read-only live view: slots, GPU, prefill/decode |
| `serve/logproxy.py` | logs what a client actually sends |
| `bench/run.sh` · `bench/summarize.py` | throughput benchmark vs the 14B baseline |
| `bench/make_fixture.py` | build long-context fixtures speculation can't cheat |
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

## Four things worth knowing before trusting any benchmark

**A slow number can be your own machine.** This repo spent three weeks on
an engine comparison showing vLLM prefilling 15-40x faster than
llama.cpp. The real figure is 2.2x. The build had been compiled against
CUDA 13 headers and linked against the distro's CUDA 12 `libcudart`;
`cudaDeviceProp` changed layout between those versions, so llama.cpp read
`multiProcessorCount` as **1** instead of 170 and launched attention on 2
of 170 SMs. No error, no warning — just consistent, reproducible, wrong
numbers. Check anything you build with `ldd <lib>.so | grep cudart`, and
see `CLAUDE.md` "Build hazards".

**Watch power, not utilization.** `nvidia-smi` reported 100% utilization
the whole time the card was drawing 160 W of a 575 W budget. Utilization
means a kernel is resident, not that the silicon is busy. The fixed build
draws 570 W on the same work — and uses *half* the total energy, because
it finishes 5.9x sooner.

**Benchmark fixtures can inflate speculative decoding.** The old
long-context fixtures were random words; asked to continue, the model
echoed the prompt back, draft acceptance hit ~100%, and decode throughput
measured nothing (llama.cpp read 147 tok/s while emitting a copy of its
own input). Use `bench/make_fixture.py`; real acceptance is ~49%, and
100% means the number is junk.

**Prefix caching dominates agentic workloads** (93% hit rate, 0.38s mean
TTFT) and was off by default. Restarting a server wipes it, so don't
restart mid-session and then time the next turn.

Always state the context length a number was taken at. Note that the two
examples this README previously used for that rule — MTP decaying from
2.07x to 1.06x, and f16 KV being 3.7x slower at 50K — were both artifacts
of the broken build and are retracted. On a correct build MTP holds ~2x
throughout and KV dtype is a ~2% spread.
