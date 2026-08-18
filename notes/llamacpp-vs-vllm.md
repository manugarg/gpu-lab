# llama.cpp vs vLLM on this rig (Qwen3.8-27B)

Measured 2026-08-14 on the RTX 5090. Companion to
`qwen3.8-27b-options.md`. Everything here is from runs on this machine,
not from anyone's published numbers.

---

## Building llama.cpp here

`cmake` was not installed. Installed it as an isolated uv tool rather
than into either vllm venv — CLAUDE.md's packaging rules exist because
re-resolving those venvs has cost hours, and a build tool has no business
in them:

```
uv tool install cmake        # lands in ~/.local/bin
```

Then, with `SSH_AUTH_SOCK` set (see CLAUDE.md — a global
`url.git@github.com:.insteadof https://github.com/` rewrites even https
clones through SSH):

```
sudo apt install -y libssl-dev   # else you silently get a no-HTTPS binary
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/tools/llama.cpp
cd ~/tools/llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc -DLLAMA_OPENSSL=ON
cmake --build build --config Release -j $(nproc)
```

`libssl-dev` matters more than it looks: without the headers, configure
prints one `OpenSSL not found, HTTPS support disabled` warning amid
hundreds of lines and builds a binary with **no** `--ssl-cert-file` /
`--ssl-key-file` flags at all. `LLAMA_OPENSSL=ON` was already set in that
build — the flag being on proves nothing. Check the artifact instead:
`ldd build/bin/llama-server | grep ssl` should show `libssl.so.3`.

**The `-DCMAKE_CUDA_COMPILER` pin is mandatory** — see CLAUDE.md's
"Build hazards". Without it the build dies with a `sm_52` ptxas error
that looks like missing Blackwell support but is actually a mixed
CUDA 12.4/13.3 toolchain.

Do *not* pass `-DCMAKE_CUDA_ARCHITECTURES=120`. llama.cpp's own
`ggml/src/ggml-cuda/CMakeLists.txt` appends `120a-real` for
CUDA >= 12.8 already; overriding it just fights its logic.

Architecture support confirmed present before building (worth checking
first for any new model family): `GGML_OP_GATED_DELTA_NET` in
`ggml/include/ggml.h`, a CUDA kernel at
`ggml/src/ggml-cuda/gated_delta_net.cu`, and `LLM_ARCH_QWEN35` in
`src/llama-arch.cpp`.

## Single-stream throughput, 1024 in / 256 out

| | llama.cpp (UD-Q5_K_XL) | vLLM (NVFP4) |
|---|---|---|
| decode | 69.98 ± 0.32 tok/s (**154.1 with MTP** — see below) | ~68.1 tok/s (TPOT 14.69 ms) |
| prefill | 2211.69 ± 0.93 tok/s (~463 ms) | ~8390 tok/s (TTFT 122 ms) |
| weights on GPU | 18.82 GiB | 21.34 GiB |

**Decode is a tie** (~3% apart). Both beat the ~60 tok/s figures being
reported publicly for this model on a 5090.

**Prefill is not close — vLLM ~3.8x faster.** This is the difference
you'd actually feel on long prompts or a large system prompt, and it
widens with prompt length.

Commands:
```
# llama.cpp
./build/bin/llama-bench -m <gguf> -ngl 99 -p 1024 -n 256

# vLLM (single stream: --max-concurrency 1)
vllm bench serve --model <model> --dataset-name random \
  --random-input-len 1024 --random-output-len 256 \
  --max-concurrency 1 --num-prompts 5
```

Note vLLM's headline "Output token throughput" (66.17 tok/s) folds TTFT
into wall time; the decode row above uses TPOT so it's comparable to
llama-bench's pure `tg256`.

## MTP speculative decoding: 2.4x, measured 2026-08-16

Prompted by simonwillison.net/2026/Aug/16/qwen-38-27b/, which reported
~72% from MTP. We were running with none (`--spec-type` defaults to
`none`). Measured here on two code-generation prompts, thinking disabled
so output length is stable and this measures decode speed rather than
reasoning variance:

| | tok/s |
|---|---|
| baseline (no speculation) | 64.4 |
| `--spec-type draft-mtp` | **154.1** |

**2.4x**, better than the post's ~72%. The reason is visible in the
server's own stats: **86-88% draft acceptance, mean 3.6 tokens accepted
per draft.**

Why it works: decode is memory-bandwidth-bound (see
`inference-anatomy.md`) — producing one token means reading ~19 GB of
weights while the arithmetic units idle. Speculation drafts N tokens
cheaply and verifies all N in one forward pass; since the weight read
dominates, verifying 4 costs barely more than verifying 1. MTP supplies
the draft from a head trained into the model itself
(`mtp_num_hidden_layers: 1` against 64 real layers, sharing embeddings)
rather than a separate draft model, so there's no second set of weights
to load. It is **lossless** — verification guarantees the same output
the target model would have produced, so this is speed with no quality
trade, unlike quantization.

Cost: the draft context needs ~1 GiB, and **262144 + MTP OOMs**. The
largest context that fits alongside it is **229376** (31.4 GiB of
32.1 GiB), which is what `serve/llama.sh` now defaults to. Speed is
unchanged at that context (154.1 tok/s), so the trade is 13% context for
2.4x throughput.

## Prefill is the bottleneck for agentic coding (2026-08-17)

A real opencode PR review took ~20 minutes and *felt* slow. Measured from
the server's own timings, with the GPU to itself:

```
prefill  63,374 tok in 413s = 153 tok/s   <- 63% of wall time
decode    9,082 tok in 243s =  37 tok/s   <- 37%
```

Decode was never the problem. Three experiments, two of which killed my
own hypotheses:

**Micro-batch size does essentially nothing.** Same 25K prompt, restarted
between configs so no cache carried over:

| `-b` / `-ub` | prefill | peak power | VRAM |
|---|---|---|---|
| 2048 / 512 (default) | 393.6 tok/s | 344 W | 23.8 GiB |
| 4096 / 2048 | 403.1 tok/s | 383 W | 24.4 GiB |
| 8192 / 4096 | 402.6 tok/s | 347 W | 25.2 GiB |

~2%. The theory was that at `-ub 512` each micro-batch re-streams all
19 GB of weights, making prefill memory-bound — which the ~160 W draw
during prefill seemed to support. Wrong: raising it changes nothing, so
the low power is not micro-batch weight streaming.

**Prompt caching is worth ~350-1400x and dwarfs everything else:**

```
identical 25K prompt, 1st time: cached=0       63,513 ms
identical 25K prompt, 2nd time: cached=24,975     183 ms
```

**Prefill rate depends on where in the context you are, not how many
tokens you send.** This is the finding that actually explains the
experience:

```
25,000 tok fresh          393 tok/s
67,907 tok fresh (resume) 156 tok/s
19,051 tok appended deep   93 tok/s   <- fewer tokens, half the rate
```

19K tokens appended at depth is *slower per token* than 67.9K from
scratch. Cost scales with the KV already present, so **every turn in a
long agentic session prefills slower than the last**, even for identical
new content. That's the compounding cost, and no llama.cpp flag touches
it — it's the 16 full-attention layers doing O(n²).

**Disproved along the way:** small interleaved requests (opencode's
auxiliary calls) do *not* evict the big conversation's cache — with 4
slots the decoy lands elsewhere and the big prompt still hits. Requests
also stayed on one slot (292 of 297), so cache thrashing from slot
bouncing isn't happening either.

### MTP's benefit decays with context length (2026-08-17)

The measurement I should have taken before setting a default. Same
prompts, MTP on vs off, decode only (prefix warmed first):

| context | MTP off | MTP on | speedup | draft acceptance |
|---|---|---|---|---|
| ~26 tokens | 66.9 tok/s | 138.6 tok/s | **2.07x** | 68% |
| 16K | 51.9 tok/s | 67.8 tok/s | **1.31x** | 66% |
| 50K | 35.2 tok/s | 37.4 tok/s | **1.06x** | 100%* |

\* small sample — few draft rounds in 300 generated tokens.

The original 2.4x was not wrong, it was unrepresentative: reproduced as
2.07x here at ~26 tokens of context. By 50K it's 6%, which is noise
against the costs (3.3 GiB, and context capped at 229376 instead of
262144).

Why: MTP amortises the *weight* read across several tokens per forward
pass. Attention over the KV cache scales with context and isn't
amortised the same way, so as context grows KV attention dominates and
there is progressively less for speculation to save.

**Rule of thumb:** worth enabling below ~16K context, not worth it above
~30K. For agentic coding at 40-68K, leave it off — which is the current
default, now for a measured reason rather than an assumed one.

### KV cache dtype and draft depth (2026-08-17)

Prompted by huggingface.co/Qwen/Qwen3.8-27B/discussions/112 — 45 llama.cpp
configs on the same GPU. Their MTP numbers match ours closely
(acceptance 0.674 vs our 68%, 3.11 tokens/step vs our 3.19), which is
good independent corroboration of the mechanism.

**KV cache dtype.** They reported f16 -> q4_0 as +9%. Measured here
(MTP off, so the KV variable is isolated):

| KV | VRAM | 16K decode | 50K decode |
|---|---|---|---|
| f16 | 23.0 GiB | 22.1 tok/s | 9.4 tok/s |
| **q8_0** (our default) | 21.4 GiB | **51.9 tok/s** | **35.2 tok/s** |
| q4_0 | 20.4 GiB | 51.1 tok/s | 34.2 tok/s |

The effect is far larger than +9% — f16 is **2.3x slower at 16K and 3.7x
slower at 50K** — but it sits entirely in f16 -> q8_0. **q4_0 buys
nothing over q8_0** (51.1 vs 51.9, 34.2 vs 35.2, both marginally slower
and within noise); it only saves ~1 GiB. Stay on q8_0: same speed, less
aggressive quantization. Never run f16 KV at real context lengths.

**The effect inverts at short context**, which reconciles the two
results. Measured under their conditions (MTP on, short prompt):

| KV | short | 16K | 50K |
|---|---|---|---|
| f16 | **140.4** | 22.1 | 9.4 |
| q8_0 | 139.5 | **51.9** | **35.2** |
| q4_0 | 138.3 | 51.1 | 34.2 |

At ~26 tokens there is almost no KV to read, so dtype is irrelevant
(1.5% spread) and f16 is even marginally fastest — no dequantisation
overhead. Their absolute figures (125.5-136.7) land right on ours
(138-140), so their "32K context" was the `-c` *allocation*, not a
filled window; llama-bench generates from a tiny prompt. Their +9% is
what noise looks like at a context length where KV barely exists.

Their Q4_K_M vs our UD-Q5_K_XL accounts for only ~1.18x (17.1 vs
20.2 GB, and decode is bandwidth-bound) — a minor term next to the
measurement-context difference.

**Draft depth (`--spec-draft-n-max`, default 3).**

| n | short | 16K | 50K |
|---|---|---|---|
| 2 | 134.8 (82% acc) | 65.7 (80%) | 30.5 |
| **3 (default)** | 139.2 (68%) | **67.9** (66%) | 37.4 |
| 5 | **141.8** (56%) | 62.6 (55%) | 42.9 |
| 8 | 104.3 (40%) | 37.2 (35%) | 35.0 |
| 12 | 91.7 (26%) | 34.5 (25%) | 41.7 |

Acceptance decays smoothly with depth (82% -> 26%) as expected —
drafting further ahead is harder. Throughput peaks at n=3-5 and
collapses by n=8. **The default of 3 is a good choice**; the thread's
recommended n=2 is slightly worse here, and anything >=8 is bad.

**The 50K column is not trustworthy** and should not be used: it's
non-monotonic (30.5, 37.4, 42.9, 35.0, 41.7) with implausible 86-100%
acceptance versus 55-66% at 16K. The likely cause is the synthetic 50K
prompt — a wall of random words — producing repetitive output that's
unusually easy to draft. A reading of "1.22x at 50K with n=5" came out
of this and is noise. Re-measure against a real long conversation
before believing any long-context MTP number.

### What the cache does and doesn't rescue (2026-08-17)

Measured on a 25K prompt, same slot each time:

| case | cached | prefill |
|---|---|---|
| cold | 0% | 63.5 s |
| identical prompt again | 100% | 0.1 s |
| **text appended at the end** | **98%** | **2.6 s** |
| **multi-turn (what opencode sends)** | **98%** | **2.7 s** |
| something changed *mid*-prompt | 0% | 63.7 s |

So the ordinary agentic pattern — append a turn, resend the history —
is already nearly free. What costs a full re-prefill is anything that
**rewrites earlier context**: history compaction, re-summarisation, a
tool result being rewritten, a changed system prompt.

**`--cache-reuse` does not rescue that.** Tested at 0 vs 256, with both
q8_0 and f16 KV, on a prompt with a ~12K-token identical prefix, a
changed middle, and an identical tail: **0% cached in all three**. Either
it doesn't do what its help text suggested, or it needs conditions I
failed to trigger. Not recommended on this evidence.

**The other cache killer is restarting the server** — it wipes all slot
KV, so the next request re-prefills from zero. Several of the very large
re-prefills in the session logs (16K, 19K, 67.9K) followed restarts done
*while benchmarking*, not anything opencode did. Don't restart mid-session
and then draw conclusions from the next turn's timing.

### vLLM vs llama.cpp at long context (2026-08-17) — the decisive result

Same prompts, same GPU, both cold:

| context | llama.cpp prefill | vLLM prefill | llama.cpp decode | vLLM decode |
|---|---|---|---|---|
| 16K | 567 tok/s | **8,333 tok/s** | 51.9 tok/s | **69.8 tok/s** |
| 50K | ~165 tok/s | **6,713 tok/s** | 35.2 tok/s | **63.8 tok/s** |

**vLLM prefills 15-40x faster and barely degrades with context**
(8,333 -> 6,552 tok/s, -21% from 16K to 53K) while llama.cpp collapses
(567 -> 165, -71%). It also holds decode nearly flat (68 -> 70 -> 64)
where llama.cpp falls off (66.9 -> 51.9 -> 35.2).

Verified not to be a caching artifact: vLLM ran with
`enable_prefix_caching=False`, and a completely fresh prompt (different
RNG seed, zero shared prefix) reproduced it — 53,143 tokens in 8.1s =
6,552 tok/s. vLLM decode is derived as wall-time minus measured prefill,
since its response folds both together.

**Concretely:** the 67,907-token prefill that took **7.25 minutes** on
llama.cpp would take roughly **10 seconds** on vLLM.

**Power draw makes the mechanism visible** (measured 2026-08-18, 50K-token
prefill, sampled at 0.2s):

```
                 peak     sustained
llama.cpp        344 W       ~160 W
vLLM             596 W        540 W
```

Same GPU, same phase, ~3.4x the power. On a ~575 W card, llama.cpp
prefill runs at ~28% of budget while nvidia-smi reports 100%
"utilization" — that field only means a kernel is resident. This is the
clearest single signal that one engine is on the tensor cores and the
other is stalling on memory.

**Why — and this bounds the claim.** llama.cpp's CUDA kernel for Gated
DeltaNet says so itself, at `ggml/src/ggml-cuda/gated_delta_net.cu:180`:

```c
//TODO: Add chunked kernel for even faster pre-fill
```

**48 of this model's 64 layers are Gated DeltaNet**, and that kernel is
unchunked — it walks prefill sequentially where a chunked implementation
would go parallel. So this is not "vLLM is faster than llama.cpp"; it is
"llama.cpp has not optimised prefill for hybrid linear-attention models
yet, and this model is 75% such layers." Expect the gap to shrink when
that TODO lands, and **do not generalise this result to dense models** —
for a conventional architecture llama.cpp's prefill is far more
competitive.

A secondary term: prefill is compute-bound, so GEMM kernel quality also
matters. llama.cpp runs Q5_K through dequant-heavy kernels; vLLM runs
NVFP4 through CUTLASS tensor-core GEMMs (confirmed earlier —
`FlashInferCutlassNvFp4LinearKernel` is what executes on this card).
That's the part visible at 1024 tokens as 3.8x, before the linear-
attention layers dominate.

**Quality settled (2026-08-17):** perplexity on 4,608 tokens of real
source, using llama-perplexity's protocol on both sides — llama.cpp
`UD-Q5_K_XL` **2.2542 ± 0.0876** vs vLLM `NVFP4` **2.2832**. That's
+1.29% against a ±3.9% error bar: statistically indistinguishable. The
speed win costs nothing measurable in quality. See `perplexity.md`.

**Costs:** vLLM tops out at 189,728 context with this NVFP4 checkpoint
vs llama.cpp's 262,144, it's a different quantization (so not
output-identical), and it needs `--max-num-seqs` capped or CUDA-graph
profiling OOMs at load.

**Recommendation reversed.** Earlier notes here favoured llama.cpp for
single-user work on the basis of decode parity and max context. That
weighed the wrong things: for agentic coding, prefill is ~63% of wall
time and vLLM wins it by more than an order of magnitude, while also
winning decode at the context lengths that actually occur.


## End-to-end result on a real workload (2026-08-17)

The same PR review, run three ways. This is the number that matters —
everything above is mechanism.

| | wall clock | why |
|---|---|---|
| llama.cpp (Q5_K_XL, MTP off) | **~20 min** | prefill 63% of wall time, collapsing with context |
| vLLM (NVFP4) | **5 min** | prefill ~3%; 93% prefix cache hit, mean TTFT 0.38s |
| vLLM + MTP | **4 min** | decode 63.8 -> 126.4 tok/s, 71% draft acceptance |

Measured from Prometheus scraping vLLM's `/metrics`, not stopwatch:

```
session: 900,744 prompt tokens seen, 837,312 served from cache (93%)
         23,878 generated, mean TTFT 0.38s, decode 63.8 tok/s
```

**The engine switch is the big win (4x); MTP is a 1.25x on top.** Raw
decode doubles with MTP but wall clock doesn't, because tool execution,
prefill and idle between turns are all outside decode.

A caution on reading burst timings: a run that looked like "40 seconds"
was actually two bursts — a 240s review and a 40s follow-up command,
separated by a 70s gap. `rate(vllm:generation_tokens_total[30s])` over a
range query separates them; a single cumulative number cannot.

## Maximum context

| | max context | KV dtype |
|---|---|---|
| llama.cpp (UD-Q5_K_XL) | **262,144** (full native) | q8_0 |
| vLLM (NVFP4) | **189,728** | fp8_e4m3 |

llama.cpp reaches the model's full native 262K; vLLM tops out around
190K. vLLM reports its own ceiling precisely rather than guessing:

```
Available KV cache memory: 5.95 GiB
ValueError: To serve at least one request with the model's max seq len
(262144), 8.18 GiB KV cache is needed ... the estimated maximum model
length is 189728.
```

That vLLM figure is with settings tuned *in its favour* —
`--gpu-memory-utilization 0.95 --max-num-seqs 1` — so ~190K is close to
its real ceiling with this checkpoint, not an artifact of a conservative
config.

**This is a quantization difference, not an engine-efficiency
difference.** The gap traces almost entirely to weights: 2.52 GiB more
free x 32 KB/token ~= 82,575 extra tokens predicted vs 72,416 observed
(remainder goes to compute buffers and linear-attention state). Point
vLLM at a smaller checkpoint and most of the gap closes.

**Both need quantized KV to get there.** llama.cpp's default f16 KV is
64 KB/token; at 262K that's a 16 GiB allocation and it OOMs outright:

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 16384.00 MiB on
device 0: cudaMalloc failed: out of memory
```

`-fa on -ctk q8_0 -ctv q8_0` is what makes it fit (28.7 GiB used of
32.1 GiB).

## Gotcha: llama.cpp mislabels the UD quant

`llama-bench` reports `UD-Q5_K_XL` as `qwen35 27B Q4_K - Small`.
Unsloth's "UD" dynamic quants mix tensor types per layer, and llama.cpp
names the file by its dominant type. The size (18.82 GiB) confirms it is
the Q5_K_XL file. Don't read that label as "we downloaded the wrong
quant".

## Reading this

- Decode parity means raw generation speed is *not* a reason to pick
  either one for interactive single-user use.
- vLLM wins prefill decisively, and wins concurrency by construction
  (735 tok/s aggregate at 20 concurrent, vs llama.cpp being oriented at
  single-stream).
- llama.cpp wins max context here (262K vs 190K), driven by the smaller
  quant, plus ~2.5 GiB lower footprint and a simpler operational story
  (one binary, no venv).
- Caveat worth re-checking over time: llama.cpp's Gated DeltaNet is
  still, per its own merge PR, "a basic vector implementation, not the
  chunking implementation" — so its decode number may improve, while
  vLLM's path is the more mature one today.
