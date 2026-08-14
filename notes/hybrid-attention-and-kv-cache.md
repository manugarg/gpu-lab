# Why KV cache is smaller on a bigger model: linear vs full attention

Companion to `qkv-and-attention.md` and `qwen3.8-27b-options.md`.
Qwen3.8-27B (27B params, 64 layers) needs *less* KV cache per token than
Qwen3-14B (14B params, 40 layers) despite being roughly twice the size.
Looks backwards until you look at what each layer actually remembers.

---

## 1. Two different ways to "remember" the past

**Full attention** (what Qwen3-14B uses in every layer, and what
Qwen3.8-27B uses in 16 of its 64): for every past token, keep its K and V
vectors around, verbatim, forever. At each new token, compare against every
stored K, take a weighted sum of every stored V. This is exact — the model
can reach back and recall any specific past token precisely, no matter how
far back. But storage is O(context length): the KV cache grows without
bound as the sequence grows. This is the mechanism `qkv-and-attention.md`
already covers in depth.

**Linear attention** (Gated DeltaNet, what Qwen3.8-27B uses in the other 48
layers): instead of storing every K,V pair, compress everything the layer
has seen so far into one fixed-size matrix per head — the "state". At each
new token, the state updates recurrently: decay the old state a bit (a
learned gate decides how much to forget), then blend in a correction based
on the new K,V pair — the "delta rule": how far off is the old state's
prediction for this new K,V, and adjust to fix it. vLLM's own
`fused_recurrent_gated_delta_rule` (in
`vllm/third_party/flash_linear_attention/ops/fused_recurrent.py`) takes
exactly `q, k, v, g (decay), beta (delta strength)` and returns one
constant-size `final_state` that seeds the next chunk.

The consequence: storage is O(1) — fixed regardless of how many tokens have
gone by. But it's lossy: old information gets blended and decayed rather
than stored verbatim. The model can't perfectly recall an arbitrary detail
from 200K tokens ago the way full attention can; what survives is a
compressed, decaying summary, not a transcript.

## 2. The hybrid bet

Qwen3.8-27B's config sets `full_attention_interval: 4` — every 4th layer
(indices 3, 7, 11, ..., 63 — 16 of 64) is full attention; the rest (48 of
64) are linear. Deliberate trade: pay the O(context) memory cost in only a
quarter of the layers, get genuine long-range exact recall from those, and
let the other three-quarters do cheap O(1) approximate "gist" tracking.
This is *why* the model can advertise 262K native context (1M extended)
without the KV cache exploding the way it would if all 64 layers were full
attention.

## 3. The actual numbers

**Full-attention KV cache**, FP8, per token — from `config.json`'s
`num_key_value_heads: 4`, `head_dim: 256`, and 16 full-attention layers:

```
2 (K+V) x 16 layers x 4 kv_heads x 256 head_dim x 1 byte = 32 KB/token
```

Compare Qwen3-14B (all 40 layers full attention, 8 kv_heads, head_dim 128):

```
2 x 40 layers x 8 kv_heads x 128 head_dim x 1 byte = 80 KB/token
```

Same formula, but Qwen3.8-27B only pays it in 16 layers instead of 40 — and
those 16 layers also have fewer/narrower KV heads. So despite being the
much bigger model, its *per-token, context-scaling* memory cost is under
half the smaller model's.

**Linear-attention state**, per sequence — fixed, **not** per token — from
vLLM's `MambaStateShapeCalculator.gated_delta_net_state_shape`
(`vllm/model_executor/layers/mamba/mamba_utils.py`). Each linear-attention
layer keeps two pieces of fixed-size state:

- **conv_state**: a short rolling window (`conv_kernel_dim - 1` = 3 steps)
  of the input, for a small causal convolution the layer applies before the
  recurrence. Shape `(head_k_dim*num_k_heads*2 + head_v_dim*num_v_heads, 3)`
  = `(10240, 3)`.
- **temporal_state**: the actual compressed memory — one
  `head_v_dim x head_k_dim` = `128 x 128` matrix *per value head* (48 of
  them). This is the `final_state` from the recurrence above: a per-head
  linear map from key-space to value-space, built by accumulating
  gated/delta-rule updates over the whole sequence so far — a compressed
  associative memory, not a token-indexed cache.

Per layer: `(30720 + 786432) elements x 4 bytes (mamba_ssm_dtype: float32)
≈ 3.27 MB`. x 48 linear-attention layers ≈ **157 MB per sequence**, flat,
whether that sequence is 100 tokens or 250,000 tokens long.

## Summary

Qwen3-14B pays for context length in all 40 layers. Qwen3.8-27B pays for it
in only 16 of 64 — the other 48 pay a flat, context-independent tax instead.
That's the whole reason a 27B hybrid model can have a *smaller* per-token,
context-scaling memory footprint than a 14B dense one: it isn't doing less
work overall, it's doing most of that work in a form that doesn't have to
remember specific tokens, only a compressed running summary of them. The
price is paid elsewhere — in exactness of recall, and in a fixed
per-*sequence* memory tax that scales with how many requests you run
concurrently, not with how long any of them are.
