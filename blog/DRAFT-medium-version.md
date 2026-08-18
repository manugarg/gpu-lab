# Qwen3.8-27B on one RTX 5090: same model, same GPU, 5x different

*Medium version — tables are code blocks. All numbers measured on one RTX 5090 (32 GB, consumer Blackwell,
sm_120), working with an AI assistant that ran the experiments. Nothing
here is quoted from someone else's benchmark without being re-run.*

---

A PR review through my local coding agent — Qwen3.8-27B on a single
RTX 5090 — took **20 minutes**. After switching inference engines and
fixing two defaults, the same review takes **4 minutes**. Nothing about
the model changed, and — I checked — neither did output quality.

That's the useful headline. But the more interesting finding is *why I
had to measure it myself*: almost every performance number published
about this model, including several I produced during this exercise, is
correct and doesn't transfer.

## The result first

Same PR review, same machine, same model weights family:

```
setup                 wall clock
--------------------------------
llama.cpp UD-Q5_K_XL     ~20 min
vLLM NVFP4                 5 min
vLLM + MTP                 4 min
```

The gap is almost entirely **prefill** — the phase where the model reads
your prompt before generating anything (tok/s):

```
context  llama.cpp   vLLM
-------------------------
16K            567  8,333
50K           ~165  6,713
```

For an agentic coding workload this is the whole ballgame. Prefill was
**63% of wall time** on llama.cpp and about **3%** on vLLM. A 68K-token
session resume cost **7 minutes 15 seconds** on one engine and roughly
**10 seconds** on the other.

Before you conclude "vLLM is 40x faster," keep reading. That number is
real and it is not what it looks like.

## Why the numbers don't transfer

### Exhibit A: the same flag is the best and worst choice

llama.cpp lets you quantize the KV cache. Conventional wisdom, and a
[public benchmark of 45 llama.cpp configs][hf112] on the same GPU, says
`q4_0` beats `f16` by about 9%.

[hf112]: https://huggingface.co/Qwen/Qwen3.8-27B/discussions/112

Here's what I measured (decode, tok/s), varying only context length:

<!-- UPLOAD: blog/kv-cache-inversion.png — drop it here, then delete this
     comment. Medium caption (paste under the image):
     Same setting, same GPU, same model — measured at three context
     lengths. Published benchmarks rarely say which one they used. -->

```
KV dtype  ~26 tok   16K   50K
-----------------------------
f16         140.4  22.1   9.4
q8_0        139.5  51.9  35.2
q4_0        138.3  51.1  34.2
```

At trivial context, `f16` is the *fastest* option — nothing to read, no
dequantization overhead. At 50K it is **3.7x the slowest**. Both
readings are correct. Neither generalizes.

That benchmark isn't wrong either. Its absolute numbers (125–137 tok/s)
land right on mine at short context (138–140), which suggests its "32K
context" was the `-c` *allocation* rather than a filled window —
`llama-bench` generates from a tiny prompt. At the context length it
appears to have measured, KV dtype barely matters, and +9% is close to
what noise looks like. Its MTP figures independently match mine closely
(0.674 draft acceptance vs my 68%, 3.11 tokens/step vs 3.19), which is
what made me trust the setup enough to go looking for why the KV numbers
diverged.

### Exhibit B: my own 2.4x evaporated

Qwen3.8 ships a Multi-Token Prediction head — a small draft model
trained into the network, used for speculative decoding. I benchmarked
it and got **2.4x** on decode. I set it as a default.

Then I measured it at the context lengths my actual work uses (tok/s):

```
context  MTP off  MTP on  speedup
---------------------------------
~26 tok     66.9   138.6    2.07x
16K         51.9    67.8    1.31x
50K         35.2    37.4    1.06x
```

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
prefill        tokens   tok/s
-----------------------------
fresh          25,000     393
fresh          67,907     156
appended deep  19,051      93
```

Read the last two rows together: **fewer tokens, less than two-thirds
the rate.**

19K tokens appended deep into a conversation prefills *slower per token*
than 68K from scratch. Cost scales with the KV already present. **Every
turn in a long session is more expensive than the last**, even for
identical new content.

Decode does the same thing: 66.9 tok/s at 26 tokens, 51.9 at 16K, 35.2
at 50K. Both phases degrade with context, and they compound.

So "tokens per second" is not a property of a model-plus-GPU. It's a
property of a model, a GPU, *and where you are in the conversation* —
and that third variable is missing from most published figures.

## The 40x is two mechanisms, not an engine verdict

A 40x gap looked implausible, so I went looking for causes. There are
two, and they behave very differently — one is specific to this model
and will expire, the other is general.

**Cause 1: vLLM is using the tensor cores; llama.cpp isn't.**

Prefill is compute-bound — that's what makes it fast when it's fast — so
the quality of the matrix-multiply kernel dominates. The two engines take
very different paths there:

- llama.cpp runs **Q5_K**, a k-quant that must be dequantized into
  fairly generic CUDA kernels before the multiply.
- vLLM runs **NVFP4** through **CUTLASS tensor-core GEMMs**. I confirmed
  this rather than assuming it — vLLM picks
  `FlashInferCutlassNvFp4LinearKernel` on this card, and in the profiler
  trace mangled `cutlass::device_kernel` symbols account for ~73% of GPU
  time.

You can see the difference in the power meter. Same GPU, same phase,
same model:

```
prefill      peak   sustained
----------------------------
llama.cpp   344 W      ~160 W
vLLM        596 W       540 W
```

On a card rated ~575 W, llama.cpp's prefill draws about **28% of budget**
while reporting "100% GPU utilization" — that metric only means a kernel
is resident, not that the silicon is busy. vLLM pulls **3.4x the power**
to do the same work faster, which is what actually using the tensor cores
looks like.

Power draw turned out to be the single most useful diagnostic in this
whole exercise. Utilization percentages lie; watts don't.

This cause is **general**. It has nothing to do with hybrid attention and
would apply to a dense model too. It's the part visible at 1024 tokens as
a 3.8x gap, before anything else dominates.

**Cause 2: one unfinished kernel, which is where the other 10x lives.**

The rest is in llama.cpp's own source, at
`ggml/src/ggml-cuda/gated_delta_net.cu:180`:

```c
//TODO: Add chunked kernel for even faster pre-fill
```

Qwen3.8-27B is a hybrid-attention model: **48 of its 64 layers** are
Gated DeltaNet (linear attention), and llama.cpp's CUDA kernel for them
is unchunked — it walks prefill sequentially where a chunked
implementation goes parallel.

That's why the gap is 3.8x at 1024 tokens and 40x at 50K. The kernel
quality difference is roughly constant; the unchunked linear-attention
path gets worse the more tokens you feed it.

So: **not** "vLLM is 40x faster than llama.cpp." A few times faster
because of tensor-core GEMM kernels, which generalizes — and then a
further order of magnitude because 75% of this particular model's layers
hit an explicitly unfinished code path, which does not generalize and has
an expiry date.

Worth separating those before quoting either. The first is a reason to
prefer vLLM for quantized inference on Blackwell generally. The second is
a reason to re-benchmark after llama.cpp's next release.

## Does the faster engine cost quality?

The two engines run different quantizations: llama.cpp `UD-Q5_K_XL`
(18.82 GiB) versus vLLM `NVFP4` (21.34 GiB), which allocate their bits
differently — NVFP4 keeps attention and the output head at FP8 while
pushing most MLP layers to 4-bit.

Perplexity over 4,608 tokens of real source code:

```
                           perplexity
-------------------------------------
llama.cpp UD-Q5_K_XL  2.2542 ± 0.0876
vLLM NVFP4                     2.2832
```

**+1.29% against a ±3.9% error bar.** Statistically indistinguishable.
The speed is free.

Caveat worth stating plainly: that's **one 4,608-token sample of code**.
It's enough to rule out a large regression, not enough to detect a small
one, and perplexity measures next-token prediction rather than whether
the model writes working code. If you're choosing between these two
quantizations for something where quality is critical, run your own eval
on your own text — treat this as "no red flag," not as "proven
equivalent." 

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

Most of these were the assistant's hypotheses. All of them were
plausible, several were mine, and the only thing that ever separated the
true ones from the false ones was changing a variable and measuring
again.

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

**A benchmark run against a live session.** The resulting slowdown got
read as a 10x server degradation, complete with a theory about VRAM
fragmentation. It was GPU contention between the benchmark and the work
it was measuring. The "idle GPU" that seemed to rule that out came from
a JSON field that didn't exist — a failed lookup treated as a result.

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

```
                     n=3      n=2
---------------------------------
context           81,920  131,072
decode tok/s         136      132
draft acceptance    ~71%      84%
```

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
