# QKV and attention: a closer look

Companion to `inference-anatomy.md`. Numbers are for Qwen3-14B:
40 layers, 40 query heads, 8 KV heads, head dimension 128, hidden size 5120.

---

## 1. Two distinct operations

These get lumped together as "attention" but they are different in kind, and they
show up as different kernel families in a profile.

### QKV projection — activations times *weights*

Multiply the token's 5120-dim hidden state by a learned 5120 x 7168 matrix.
The weights come from the checkpoint and never change at inference time.

- Produces Q (5120), K (1024), V (1024) for **one token**
- Looks at no other token
- Cost is fixed per token, independent of context length
- Runs as a GEMM — DeepGEMM or CUTLASS in our traces

### Attention — activations times *activations*

Compare this token's Q against every previous token's K, normalise, and take a
weighted sum of their V vectors.

- **No learned weights are involved in this step at all**
- Every input was produced by the projection step
- This is where tokens actually interact
- Cost grows with context length
- Runs as FlashInfer kernels in our traces

---

## 2. The "locally computed" property

Q, K, and V for a token depend **only on that token's own hidden state**. No
other token influences them.

Two consequences follow directly:

- **Prefill parallelises.** All 1024 prompt tokens can be projected at once,
  because none of them needs any other one's result.
- **K and V are cacheable.** Once computed, they are fixed. Token 12's K never
  needs recomputing, no matter how long the sequence grows.

### But: per layer, not per token

This is the easy thing to get wrong.

Each of the 40 layers has its own QKV weight matrix, and each operates on the
hidden state *as it exists at that depth* — which every earlier layer has been
modifying. So:

```
token 12, layer 3  -> K = W_k(layer 3)  x  hidden_state(token 12, after 2 layers)
token 12, layer 20 -> K = W_k(layer 20) x  hidden_state(token 12, after 19 layers)
```

Different vectors. The KV cache is really **40 separate caches**, one per layer.
That is where the per-token cache cost comes from:

```
2 (K and V) x 8 heads x 128 dims x 2 bytes x 40 layers = 160 KB per token
```

---

## 3. One head, one decode step, end to end

Setup: token 501 has just been projected. The cache holds 500 previous tokens.
Take query head 0, which reads KV group 0.

### Step 1 — score

Dot-product Q0 (128 numbers) against each of the 500 cached K0 vectors. A dot
product of two 128-dim vectors is a single number measuring alignment — large
when they point the same direction.

Output: 500 raw scores.

### Step 2 — scale

Divide each score by sqrt(128) ≈ 11.3.

Without this, dot products of high-dimensional vectors grow large enough that the
softmax collapses to nearly one-hot — the token attends to exactly one
predecessor and ignores everything else. The scaling keeps the distribution
usable.

### Step 3 — softmax

Exponentiate and normalise so the 500 scores become 500 weights summing to 1.

This is now a probability distribution over "how much does token 501 care about
each previous token."

### Step 4 — weighted sum

For **each of the 128 dimensions independently**, multiply that dimension across
all 500 cached V0 vectors by their weights and add.

```
output[0]   = w1 * V0_token1[0]   + w2 * V0_token2[0]   + ... + w500 * V0_token500[0]
output[1]   = w1 * V0_token1[1]   + w2 * V0_token2[1]   + ... + w500 * V0_token500[1]
...
output[127] = w1 * V0_token1[127] + w2 * V0_token2[127] + ... + w500 * V0_token500[127]
```

Concretely, if the weights were 0.7 for token 12 and 0.3 for token 340 with the
rest near zero, the output is `0.7 x V0[token 12] + 0.3 x V0[token 340]`,
computed elementwise.

Output: 128 numbers.

Then repeat for all 40 heads, concatenate the 40 x 128 outputs back to 5120, and
pass through `o_proj`.

---

## 4. What that output *is* (and isn't)

The 128-dim result is **not** token 501's V0. Token 501 has its own V0 — produced
by the projection, already written to the cache, and just one of the ~500 vectors
being blended.

What you computed is the **attention output** for head 0: a new vector that
belongs to no single token, but is a mixture of many.

The three things have different fates:

| Vector | Fate |
|---|---|
| **Q0** | Used once for scoring, then discarded. Never cached. |
| **K0, V0** | Written to cache. Read by every future token at this layer. |
| **attention output** | Transient. Concatenated, projected, added to residual, gone. |

A useful framing: the scores decided *who to listen to*; the V vectors are *what
those tokens actually say*. So the output means roughly "here is the information
from earlier in the sequence that this token needs right now, mixed in proportion
to relevance."

That mixture is then **added** to the token's running representation via the
residual connection — which is why attention enriches a token rather than
replacing it.

---

## 5. Where the learning actually lives

It is true that steps 1-4 contain no learned parameters. But attention is
sandwiched between two weight multiplies, and that is where the intelligence sits:

- The **QKV projection** learned how to turn a token into a query and a key.
  That is what makes the dot product meaningful rather than arbitrary — the model
  learned to place "looking for a subject" queries near "I am a subject" keys.
- **`o_proj`** learned how to interpret the blended result.

The comparison itself is just arithmetic. The semantics were installed on either
side of it.

---

## 6. Grouped-query attention, concretely

Query heads 0-4 all read KV group 0. Heads 5-9 read group 1. And so on — 40
query heads across 8 groups, 5 each.

Each query head has its own Q vector, so they compute **different score
distributions**. But within a group they score against the same keys and blend
the same values.

So the model keeps 40 distinct opinions about relevance, drawing on 8 distinct
sets of content.

Cache saving at 16384 context:

```
GQA  (8 kv heads):  160 KB/token  ->  2.6 GB
MHA (40 kv heads):  800 KB/token  -> 13.1 GB
```

With 15 GB of weights on a 32 GB card, the MHA version leaves essentially no room.

---

## 7. Why attention behaves badly on hardware

Count the traffic for one head at 500 tokens of context:

- Read 500 K vectors and 500 V vectors, 128 dims, 2 bytes = **~256 KB**
- Arithmetic: roughly **128,000 multiply-adds**

That is about **1 FLOP per byte read**. The 5090 can sustain hundreds of FLOPs
per byte of bandwidth, so attention leaves the tensor cores almost entirely idle.

Attention is a memory-movement problem wearing a math costume. That is precisely
what FlashAttention and FlashInfer exist to address — they restructure the
computation to avoid materialising the full score matrix in memory.

### The batching asymmetry

This is the part that changes as you scale up:

- **QKV projection**: batch 32 requests and the weights are read *once*, applied
  to all 32. Cost amortises beautifully.
- **Attention**: every sequence has its own KV cache. 32 requests means 32x the
  cache reads. Nothing amortises.

Which is why attention was only 5-12% in our profiles (short contexts, modest
batch) but becomes the dominant cost at long context or high concurrency. Our
measurements were taken at a friendly operating point — do not generalise them.

---

## 8. Prefill differs in one important way

In decode, token 501 can only see tokens 1-500, because 502 does not exist yet.
The constraint enforces itself.

In prefill, all 1024 tokens are processed simultaneously — so token 12 *could*
see token 900 unless stopped. A **causal mask** explicitly zeroes attention to
future positions.

So prefill computes a full 1024 x 1024 score matrix with roughly half of it
masked out. That is why prefill attention kernels
(`BatchPrefillWithPagedKVCacheKernel`) look different from decode ones in a
profile, and why prefill attention cost scales quadratically with prompt length
while decode scales linearly with context.

---

## Summary in one paragraph

Each layer applies its own learned weights to turn a token's current hidden state
into a query, a key, and a value — computed from that token alone, which is what
makes them cacheable. The query is compared against every previous token's key to
produce relevance scores; those scores are normalised into weights; and the
weights are used to blend the previous tokens' value vectors into a single new
vector. That blend is the information the token pulled in from context. It gets
projected once more and added back into the token's representation, and then the
next layer does the whole thing again against a modified hidden state.

