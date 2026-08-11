# Prefill, in detail

Third in the series with `inference-anatomy.md` and `qkv-and-attention.md`.
Numbers are Qwen3-14B on RTX 5090, and the benchmark config we measured:
1024-token prompts, 256 output tokens.

---

## The one sentence that unlocks it

**Prefill runs 1024 tokens through 40 layers and produces exactly one token.**

The other 1023 positions do all that work purely to deposit their K and V in the
cache. Their predictions are computed and thrown away.

Once that lands, everything else follows.

---

## 1. Prefill and decode are the same operations

There is no separate "prefill code path" doing different math. Same 40 layers,
same attention, same MLP, same weights. The only difference is **how many rows
are in the matrix**:

| | Rows per forward pass | Passes |
|---|---|---|
| Prefill | 1024 | 1 |
| Decode | 1 | 256 |

That is the entire distinction. Everything else in this document is a consequence
of it.

A common confusion: "the MLP is the heavyweight decode stage." It is not. The MLP
runs in every layer in both phases. It is heavyweight in both. What changes is
how many tokens share the cost of reading its weights.

---

## 2. Step by step

### Input

1024 token IDs. The embedding lookup turns each into a 5120-dim vector.

You now have a **1024 x 5120 matrix**. One row per token. This shape persists
through all 40 layers.

### Inside each layer

Identical to decode, with 1024 rows instead of 1:

```
QKV projection   [1024 x 5120] x [5120 x 7168]   -> 1024 x 7168
                 write all 1024 K and V into this layer's cache
attention        (see below)
o_proj           [1024 x 5120] x [5120 x 5120]
RMSNorm
gate + up        [1024 x 5120] x [5120 x 34816]
SwiGLU
down_proj        [1024 x 17408] x [17408 x 5120]
```

Output: another 1024 x 5120 matrix. Feed to the next layer. Forty times.

### Attention, where prefill genuinely differs

Every token attends to every previous token — **simultaneously**.

- Token 1 sees 1 token
- Token 500 sees 500 tokens
- Token 1024 sees 1024 tokens

That is a **1024 x 1024 score matrix per head**. A causal mask zeroes the upper
triangle so no token can see the future. Roughly half the matrix is computed and
discarded.

In decode this constraint enforces itself — token 501 cannot see token 502
because it does not exist yet. In prefill all tokens exist at once, so the mask
is explicit.

### At the end

You hold a 1024 x 5120 matrix.

**Take row 1024 only.** Those 5120 numbers go through the LM head to produce
151,936 scores, and you sample token #1025.

Rows 1 through 1023 are discarded.

---

## 3. Why the other 1023 predictions are discarded

During **training**, every position predicts the next token. Position 12 predicts
token 13, position 500 predicts token 501, and so on. That is how a model learns
from an entire sequence in a single pass — it is enormously efficient.

At **inference**, you already know tokens 2 through 1024. You typed them.
Predicting them is worthless.

So the same forward pass that trains on 1024 examples yields exactly one useful
prediction at inference. Their real contribution was writing K and V at all 40
layers.

---

## 4. What prefill leaves behind

The KV cache, at 1024 tokens x 40 layers:

```
2 (K and V) x 8 kv_heads x 128 dims x 2 bytes x 40 layers = 160 KB per token
1024 tokens x 160 KB = ~160 MB
```

Decode then reads this instead of recomputing anything. That is the whole
economics: without the cache, generating token 1500 would mean re-running 1500
tokens through 40 layers — quadratic in output length.

---

## 5. Why prefill is fast and decode is slow

This is the payoff, and it is worth doing the arithmetic yourself once.

**Work done, in token-layer traversals:**

```
Prefill:  1024 tokens x 40 layers = 40,960
Decode:    256 tokens x 40 layers = 10,240
```

Prefill does **4x more math**.

**Time measured:**

```
Prefill (TTFT):     ~110 ms
Decode (256 x 15.9): ~4,070 ms
```

Decode takes **37x longer while doing a quarter of the work**. Roughly a 150x
efficiency gap.

### The reason

Every pass through a layer must read that layer's weights from VRAM.

- **Prefill**: read the 178M-parameter gate/up matrix once, apply it to 1024
  rows. Memory cost amortises across 1024 tokens.
- **Decode**: read that same 178M-parameter matrix, apply it to **one row**,
  discard, move to the next layer. Same bytes moved, 1/1024th the useful work.

Prefill is **compute-bound** — the tensor cores are the limit.
Decode is **memory-bandwidth-bound** — fetching weights is the limit.

Decode's floor: 15 GB of weights / 1792 GB/s = **8.4 ms per token**, no matter
how good the kernels are.

---

## 6. Consequences visible in our profiles

### No split-K in prefill

1024 rows give the tensor cores abundant parallel work, so DeepGEMM has no reason
to split the K dimension. Our prefill profile showed **zero** split-K.

Decode has one row, cannot fill 170 SMs, so DeepGEMM splits K to manufacture
parallelism — and pays a reduction pass for it. Our decode profile showed
**26.3%** in `sm120_split_k_reduce_impl`.

Same kernels, opposite problem.

### Attention share differs

Prefill attention was 7.3% of GPU time, decode 5.2%. But these scale differently:

- **Prefill attention is quadratic** in prompt length. Double the prompt,
  quadruple the attention cost, while the GEMMs only double. At 16k prompts it
  would dominate.
- **Decode attention is linear** in context length, but grows with batch size
  because every sequence has its own cache and nothing amortises.

Both of our measurements were taken at a friendly operating point. Do not
generalise them.

---

## 7. Prefix caching

If a prefix is already cached — a shared system prompt across requests, or an
earlier chunk of the same long prompt — prefill **skips those tokens entirely**
and processes only the new suffix, attending against the cached prefix KV.

This is a direct cut to TTFT and it is on by default in vLLM V1.

It is also why our `--dataset-name random` benchmark shows none of it: random
token IDs share no prefixes. **We measured the worst case.** Real traffic with
shared system prompts would show materially better TTFT.

Related: **chunked prefill** splits a long prompt into pieces so a huge prefill
does not monopolise the GPU and stall other requests' decode steps. Chunk 2
attends against chunk 1's already-cached KV. This is why prefill and decode work
end up interleaved in the same forward pass, and why our traces show a single
ragged attention kernel handling both rather than separate prefill and decode
kernels.

---

## Summary

Prefill takes the whole prompt as one wide matrix, runs it through every layer
once, writes K and V everywhere, and harvests a single token from the final row.
It is compute-bound, efficient, and quadratic in prompt length.

Decode then takes that one token, runs it alone through every layer, harvests the
next token, and repeats. It is bandwidth-bound, deeply inefficient per unit of
work, and it is where nearly all wall-clock time goes.

TTFT measures prefill. TPOT measures decode. They are different problems and they
want different optimisations.

