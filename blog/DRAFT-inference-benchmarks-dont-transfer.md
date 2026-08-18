# Same model, same GPU, 5x different: what I learned benchmarking Qwen3.8-27B

*Draft. All numbers measured on one RTX 5090 (32 GB, sm_120) running
Qwen3.8-27B. Nothing here is quoted from someone else's benchmark
without being re-run.*

---

A PR review through my local coding agent took **20 minutes**. After
switching inference engines and fixing two defaults, the same review
takes **4 minutes**. Nothing about the model changed, and — I checked —
neither did output quality.

That's the useful headline. But the more interesting finding is *why I
had to measure it myself*: almost every performance number published
about this model, including several I produced during this exercise, is
correct and doesn't transfer.

## The result first

Same PR review, same machine, same model weights family:

| setup | wall clock |
|---|---|
| llama.cpp, `UD-Q5_K_XL` | **~20 min** |
| vLLM, `NVFP4` | **5 min** |
| vLLM + MTP speculative decoding | **4 min** |

The gap is almost entirely **prefill** — the phase where the model reads
your prompt before generating anything:

| context | llama.cpp prefill | vLLM prefill |
|---|---|---|
| 16K tokens | 567 tok/s | **8,333 tok/s** |
| 50K tokens | ~165 tok/s | **6,713 tok/s** |

For an agentic coding workload this is the whole ballgame. Prefill was
**63% of wall time** on llama.cpp and about **3%** on vLLM. A 68K-token
session resume cost **7 minutes 15 seconds** on one engine and roughly
**10 seconds** on the other.

Before you conclude "vLLM is 40x faster," keep reading. That number is
real and it is not what it looks like.

## Why the numbers don't transfer

### Exhibit A: the same flag is the best and worst choice

llama.cpp lets you quantize the KV cache. Conventional wisdom, and a
well-run public benchmark I found, says `q4_0` beats `f16` by about 9%.

Here's what I measured, varying only context length:

| KV dtype | ~26 tokens | 16K | 50K |
|---|---|---|---|
| `f16` | **140.4 tok/s** | 22.1 | 9.4 |
| `q8_0` | 139.5 | **51.9** | **35.2** |
| `q4_0` | 138.3 | 51.1 | 34.2 |

At trivial context, `f16` is the *fastest* option — nothing to read, no
dequantization overhead. At 50K it is **3.7x the slowest**. Both
readings are correct. Neither generalizes.

The public benchmark reporting +9% wasn't wrong either. Its absolute
numbers (125–137 tok/s) land right on mine at short context (138–140),
which told me its "32K context" was the *allocation*, not a filled
window — `llama-bench` generates from a tiny prompt. At the context
length it actually measured, KV dtype barely matters, and +9% is what
noise looks like.

### Exhibit B: my own 2.4x evaporated

Qwen3.8 ships a Multi-Token Prediction head — a small draft model
trained into the network, used for speculative decoding. I benchmarked
it and got **2.4x** on decode. I set it as a default.

Then I measured it at the context lengths my actual work uses:

| context | MTP off | MTP on | speedup |
|---|---|---|---|
| ~26 tokens | 66.9 tok/s | 138.6 | **2.07x** |
| 16K | 51.9 | 67.8 | **1.31x** |
| 50K | 35.2 | 37.4 | **1.06x** |

My 2.4x was never wrong. It was taken with short prompts and thinking
disabled — conditions that don't resemble agentic coding at all. By 50K
the benefit is 6%, which is noise.

Speculation amortizes the *weight* read across several tokens per
forward pass. Attention over the KV cache scales with context and isn't
amortized the same way, so as context grows there's progressively less
left to save.

### Exhibit C: cost depends on where you are, not how much you send

The finding that actually explained my slow sessions:

```
25,000 tokens, fresh          393 tok/s
67,907 tokens, fresh          156 tok/s
19,051 tokens, appended deep   93 tok/s   <- fewer tokens, half the rate
```

19K tokens appended deep into a conversation prefills *slower per token*
than 68K from scratch. Cost scales with the KV already present. **Every
turn in a long session is more expensive than the last**, even for
identical new content.

Decode does the same thing: 66.9 tok/s at 26 tokens, 51.9 at 16K, 35.2
at 50K. Both phases degrade with context, and they compound.

So "tokens per second" is not a property of a model-plus-GPU. It's a
property of a model, a GPU, *and where you are in the conversation* —
and that third variable is missing from most published figures.

## The 40x is one kernel, not an engine verdict

The vLLM prefill advantage looked implausible, so I went looking for a
mechanism. It's in llama.cpp's own source, at
`ggml/src/ggml-cuda/gated_delta_net.cu:180`:

```c
//TODO: Add chunked kernel for even faster pre-fill
```

Qwen3.8-27B is a hybrid-attention model: **48 of its 64 layers** are
Gated DeltaNet (linear attention), and llama.cpp's CUDA kernel for them
is unchunked — it walks prefill sequentially where a chunked
implementation goes parallel.

So this isn't "vLLM is 40x faster than llama.cpp." It's "llama.cpp
hasn't optimized prefill for hybrid linear-attention models yet, and
this model is 75% such layers." It shouldn't be generalized to dense
models, and it has an expiry date — the day that TODO lands.

This is the shape of most large engine gaps, in my experience: not
craftsmanship, but one unfinished code path that happens to sit on your
critical path.

## Does the faster engine cost quality?

The two engines run different quantizations: llama.cpp `UD-Q5_K_XL`
(18.82 GiB) versus vLLM `NVFP4` (21.34 GiB), which allocate their bits
differently — NVFP4 keeps attention and the output head at FP8 while
pushing most MLP layers to 4-bit.

Perplexity over 4,608 tokens of real source code:

| | perplexity |
|---|---|
| llama.cpp `UD-Q5_K_XL` | **2.2542** ± 0.0876 |
| vLLM `NVFP4` | **2.2832** |

**+1.29% against a ±3.9% error bar.** Statistically indistinguishable.
The speed is free.

My first attempt at this said **11%**, which would have read as a real
regression. That was pure methodology: `llama-perplexity` scores only the
second half of each chunk, using the first half as context, while I'd
scored every token — including early ones with no context, which are
much harder to predict. Same model, same text, different protocol,
different answer.

Matching it meant pulling token IDs from vLLM's `/tokenize` and passing
them directly as the prompt, so both engines scored byte-identical
inputs, then averaging only the second half of each 512-token chunk.

## Things that were wrong along the way

I want to be specific about this, because a post that only lists
conclusions is less useful than one that shows which plausible ideas
died.

**"Prefill is memory-bound because micro-batches re-stream the weights."**
The GPU drew ~160 W of a ~575 W budget during prefill, which fit the
theory perfectly. Raising `-ub` from 512 to 4096 changed throughput by
**2%**. Wrong.

**"Small auxiliary requests are evicting the big conversation's cache."**
Tested by interleaving a decoy request between two large ones. The decoy
landed on a different slot; the big prompt still hit cache. Wrong.

**"MTP recovers at 50K with more draft tokens."** I saw 42.9 tok/s at
n=5 and reported a 1.22x recovery. Extending the sweep to n=8 and n=12
showed the 50K column was non-monotonic — 30.5, 37.4, 42.9, 35.0, 41.7 —
with implausible 86–100% acceptance. My synthetic 50K prompt was a wall
of random words that drafts far too easily. Noise, not signal.

**A benchmark I ran while the user's session was live.** I then read the
resulting slowdown as a 10x server degradation and started theorizing
about VRAM fragmentation. It was contention. I'd also mis-read an idle
GPU from a JSON field that didn't exist, and treated the failed lookup
as a result.

The pattern: every one of these was a *plausible mechanism* supported by
a *real observation*. What separated the true ones from the false ones
was always the same thing — changing one variable and measuring again.

## What actually mattered, in order

1. **Prefix caching.** 93% hit rate on a real session, mean time-to-first-
   token of 0.38s. It was **off by default** in my vLLM config. Appending
   a turn reuses ~98% of the cache; the ordinary agentic pattern is
   nearly free. Anything that rewrites earlier context — history
   compaction, a rewritten tool result — is a full re-prefill.
2. **The engine**, for this specific architecture, for the reason above.
3. **KV cache dtype**, but only the `f16` → `q8_0` step. `q4_0` buys
   nothing further.
4. **Speculative decoding**, sized correctly. Which brings me to the
   last thing.

## A footgun worth naming

MTP needs KV budget for its draft model. At `num_speculative_tokens: 3`
it needed 5.03 GiB and forced my context down from 131K to 82K — at
which point my agent started compacting history, and compaction is
exactly the operation that destroys the prefix cache.

`num_speculative_tokens: 2` needs **4.88 GiB**. The draft model's
*weights* dominate that cost, not the token count, so dropping a token
barely helps — but 0.15 GiB plus raising `--gpu-memory-utilization` to
0.95 was exactly enough to fit the full window:

| | n=3 | n=2 |
|---|---|---|
| context | 81,920 | **131,072** |
| decode | 136 tok/s | 132 tok/s |
| draft acceptance | ~71% | **84%** |

60% more context for 3% less decode. I'd assumed the context sacrifice
was inherent to speculative decoding and never tested a smaller `n`.

## The config, if you just want that

```bash
vllm serve unsloth/Qwen3.8-27B-NVFP4 \
  --served-model-name qwen3.8-27b \
  --attention-backend flashinfer \
  --gpu-memory-utilization 0.95 \
  --max-model-len 131072 \
  --kv-cache-dtype fp8_e4m3 \
  --max-num-seqs 4 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

Four of those fail in ways that don't look like a missing flag:

- Without `--reasoning-parser`, thinking is emitted into `content`. Your
  first reply is literally `"We need answer user's simple math..."` — it
  reads as a broken model.
- Without the tool-call flags, any request with `tools` returns a 400.
  The parser must be `qwen3_xml`, **not** the more common `hermes` —
  this model's template emits `<function=name><parameter=key>` XML, not
  hermes-style JSON. Check the template; don't guess.
- Without `--max-num-seqs`, CUDA-graph capture OOMs at load, and the
  error points at the KV cache rather than at concurrency.
- Without `--enable-prefix-caching`, every turn re-prefills the entire
  history.

## The takeaway

If you take one thing from this: **a throughput number without a context
length is not a measurement.** It's an anecdote about someone's test
harness.

That applies to the numbers in this post too. They're from one GPU, one
model, one workload, on one week's builds of two fast-moving projects.
The mechanisms — prefill dominance in agentic work, cost scaling with
existing KV, speculation decaying as attention grows — should outlive
the specific figures. The figures themselves probably won't survive the
next release of either engine.

Measure your own workload. It took me an afternoon and found a 5x.
