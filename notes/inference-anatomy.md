# Anatomy of LLM inference

Reference notes for Qwen3-14B-FP8 on RTX 5090. Every number here is either from
the model config or derivable from the kernel shapes in our own profile traces.

---

## 1. The two phases

Inference has two phases with completely different performance characteristics.
Almost every confusing benchmark result traces back to conflating them.

### Prefill

The model reads your prompt. All input tokens are processed **in parallel** —
1024 tokens go through the network at once, as a 1024-row matrix.

- Lots of parallel work → the GPU's arithmetic units are the bottleneck
- **Compute-bound**
- Determines TTFT (time to first token)
- In our profile: 78% of time in four GEMM kernels, no split-K at all

### Decode

The model generates tokens one at a time. Each new token depends on the previous
one, so there is no parallelism across tokens — you process a **single row**
through the entire network, then repeat.

- Must read all 15 GB of weights to produce one token
- Barely any arithmetic per byte read
- **Memory-bandwidth-bound**
- Determines TPOT (time per output token)
- In our profile: 26% of time in split-K reduction, because the GPU is starved
  for parallel work and DeepGEMM splits the K dimension to keep SMs busy

This asymmetry is the single most important fact about inference performance.
Prefill wants big matrices; decode wants fast memory. The same kernel that is
optimal for one is often wrong for the other.

**The decode floor:** weights in bytes ÷ memory bandwidth.
15 GB ÷ 1792 GB/s ≈ **8.4 ms per token**, no matter how good your kernels are.
Our measured TPOT was 15.9 ms.

---

## 2. Qwen3-14B architecture

| Property | Value |
|---|---|
| Hidden size (`d_model`) | 5120 |
| Layers | 40 |
| Attention heads (query) | 40 |
| Key/value heads | 8 (grouped-query attention) |
| Head dimension | 128 |
| MLP intermediate size | 17408 |
| Vocabulary | 151,936 |
| Total parameters | ~14.8 B |

Verify with:

```python
from transformers import AutoConfig
c = AutoConfig.from_pretrained("Qwen/Qwen3-14B-FP8")
print(c)
```

---

## 3. Reading the architecture out of the kernel shapes

This is the satisfying part. Every GEMM in the profile maps to one weight matrix,
and the dimensions tell you the architecture without opening the config.

| Kernel shape seen in trace | What it is |
|---|---|
| `7168 × 5120` | QKV projection (fused) |
| `5120 × 5120` | Output projection |
| `34816 × 5120` | MLP gate + up (fused) |
| `5120 × 17408` | MLP down |
| `151936 × 5120` | LM head |

**Why QKV is 7168.** Q needs 40 heads × 128 dims = 5120. K needs 8 heads × 128 =
1024. V needs another 1024. Total 5120 + 1024 + 1024 = **7168**. The fact that K
and V are 1024 rather than 5120 is grouped-query attention, and you can read it
straight off the kernel name.

**Why MLP gate/up is 34816.** SwiGLU needs two projections of size 17408, computed
in one fused GEMM: 2 × 17408 = **34816**.

---

## 4. What the weights actually are

Per layer (40 of these), there are exactly six weight matrices:

**Attention block**
- `q_proj` — 5120 → 5120
- `k_proj` — 5120 → 1024
- `v_proj` — 5120 → 1024
  (these three are fused into one 5120 → 7168 GEMM at runtime)
- `o_proj` — 5120 → 5120

**MLP block**
- `gate_proj` — 5120 → 17408
- `up_proj` — 5120 → 17408
  (fused into one 5120 → 34816 GEMM)
- `down_proj` — 17408 → 5120

Per-layer parameter count:

```
QKV        5120 × 7168   =  36.7 M
o_proj     5120 × 5120   =  26.2 M
gate+up    5120 × 34816  = 178.3 M
down      17408 × 5120   =  89.1 M
                          ---------
                            330.3 M per layer
× 40 layers               =  13.2 B
```

**Outside the layers**
- `embed_tokens` — 151,936 × 5120 = 778 M (a lookup table, not a matrix multiply)
- `lm_head` — 5120 × 151,936 = 778 M

Total ≈ **14.8 B parameters**.

Note that the MLP holds roughly 80% of the parameters. Attention gets all the
conceptual attention; the MLP gets all the memory traffic.

**Not weights:** the KV cache. It grows with sequence length and batch size, it's
allocated at runtime, and it's what `--gpu-memory-utilization` is really
budgeting for after weights are loaded.

---

## 5. How attention works

The purpose: let each token look at every previous token and pull in what's
relevant. "It" needs to find "the cat" earlier in the sentence.

### The mechanism

For each token, project its 5120-dim vector into three things:

- **Query** — what this token is looking for
- **Key** — what this token offers to others searching
- **Value** — the actual content to pass along if matched

Then, for each token: compare its query against every previous token's key
(a dot product — high score means relevant), softmax those scores into weights
summing to 1, and take the weighted average of the corresponding values.

That's it. The whole mechanism is "score everything, normalize, weighted sum."

### Why multiple heads

One attention operation learns one kind of relationship. Qwen3-14B runs **40 in
parallel per layer**, each with its own 128-dim slice. Empirically, different
heads specialize — some track syntax, some track long-range entity references,
some do positional bookkeeping. The 40 head outputs are concatenated back to
5120 dims and passed through `o_proj`.

The 5120 = 40 × 128 relationship is not a coincidence — head dimension times head
count equals model dimension, by construction.

### Grouped-query attention (GQA)

The expensive part of decode is not attention arithmetic — it's reading the KV
cache from memory. Every previous token's keys and values must be re-read at
every step.

GQA shrinks that. Instead of 40 separate K/V pairs, Qwen3-14B has **8**, each
shared by 5 query heads. Queries stay at full resolution; keys and values are
shared within a group.

The effect on KV cache size is direct — 5× smaller than full multi-head
attention would be. Since decode is bandwidth-bound and KV cache traffic grows
with context length, this is one of the highest-leverage architectural choices
in modern models. It's the reason you can run useful context lengths in 32 GB.

---

## 6. What GEMM is doing

GEMM = **GE**neral **M**atrix **M**ultiply. It is not one step in inference — it
is essentially *all* of inference.

Every one of those weight matrices exists to be multiplied by activations.
Reading the profile: DeepGEMM kernels are 70–80% of GPU time in both phases.
Attention is 5–12%. Everything else — normalization, activation functions,
quantization, sampling — is a rounding error.

**Why matrix multiply specifically:** a neural network layer is "multiply the
input vector by a weight matrix." Batching many tokens turns vector-matrix into
matrix-matrix, which is GEMM. GPUs have dedicated tensor-core hardware for
exactly this operation, which is why they run models and CPUs don't.

### One decode step, concretely

For each of 40 layers:

1. RMSNorm (cheap, elementwise)
2. **GEMM**: input × QKV weights → queries, keys, values
3. Attention: score against the KV cache, weighted sum
4. **GEMM**: attention output × `o_proj`
5. RMSNorm
6. **GEMM**: input × gate/up weights
7. SwiGLU activation (cheap, elementwise)
8. **GEMM**: × `down_proj`

Then once at the end:

9. **GEMM**: final vector × `lm_head` → 151,936 scores
10. Sample one token from those scores

That's **4 GEMMs per layer × 40 layers + 1 = 161 matrix multiplies** to produce a
single token. At 15.9 ms per token, that's about 100 µs of budget each.

### Where FP8 fits

Storing weights in FP8 (1 byte) instead of BF16 (2 bytes) halves the bytes read
per token. Since decode is bandwidth-bound, that nearly doubles decode speed.
Accumulation still happens in higher precision — the kernel signature shows FP8
inputs and BF16 output.

The cost: activations must be quantized to FP8 on the fly, which is the
`per_token_group_quant` and `triton_..._fp8_quant` kernels taking ~8% of time in
our profile. Worth it, but not free.

### Why the LM head is different

`lm_head` is a single 5120 × 151,936 GEMM — 778 M parameters, 1.5 GB at BF16,
about 10% of the model in one operation. It runs once per token, outside the
CUDA graph, in BF16 rather than FP8.

1.5 GB ÷ 1792 GB/s ≈ 0.85 ms theoretical minimum. Measured: 0.94 ms. It's within
10% of the memory-bandwidth floor, which means it's essentially optimal despite
running an Ampere-generation kernel — the bottleneck is physics, not kernel
tuning.

---

## 7. Things worth remembering

- Prefill is compute-bound, decode is bandwidth-bound. They need different things.
- The decode floor is weights ÷ bandwidth. Everything else is overhead on top.
- The MLP holds ~80% of parameters; attention holds the interesting mechanism.
- GQA exists to shrink KV cache traffic, which is the second bandwidth cost.
- Kernel shapes in a profile are readable — they tell you the architecture.
- Almost all GPU time is GEMM. Optimizing anything else is optimizing noise.

