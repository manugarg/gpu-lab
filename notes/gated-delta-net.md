# Gated DeltaNet: the compute side, and why it caps llama.cpp's power

Companion to `hybrid-attention-and-kv-cache.md` (the memory side) and
`llamacpp-vs-vllm.md` (the measurement). This note covers what Gated
DeltaNet *computes*, why llama.cpp's kernel for it is the source of the
sustained ~160 W prefill draw, and who has to implement it.

---

## 1. What it computes

Gated DeltaNet ("GDN") is the **linear-attention** layer type that 48 of
Qwen3.8-27B's 64 layers use (the other 16 are full attention,
`full_attention_interval: 4`). Instead of storing every past token's K,V
(full attention's O(context) cache), each GDN layer compresses everything
seen so far into **one fixed-size state matrix per head** — `S`, 128×128
here (48 value heads). For each new token it does a rank-1 update:

```
kv    = Sᵀ k                  # the state's current prediction for this key
delta = (v − g·kv) · β        # how wrong that prediction was (the "delta")
S    ← g·S + k·delta          # decay the old state, add the correction
out  = Sᵀ q                  # the layer's output for this token
```

`g` is a learned per-token gate (how much to forget); `β` scales how hard
the layer corrects. The "delta rule" is the difference from plain linear
attention: instead of adding `k⊗v`, it adds `k⊗(v − prediction)`, i.e. it
updates the state by the *residual error*. That makes recall sharper than
naive linear attention, at the same O(1) memory cost. Old info is blended
and decayed, not stored verbatim — lossy, but context-independent.

The math above is **fixed by the model** (its weights + config). It is not
an engine feature. See §3.

---

## 2. Why llama.cpp's kernel for it is the 160 W problem

llama.cpp implements GDN in `ggml/src/ggml-cuda/gated_delta_net.cu`. The
entire prefill is **one sequential loop over every token**
(`gated_delta_net.cu:63`):

```c
for (int t = 0; t < n_tokens; t++) {        // 50K iterations for a 50K prefill
    ...
    s_shard[r] = g * s_shard[r] + k * delta; // :103 — depends on t-1's state
    ...
    warp_reduce_sum(...);                    // a sync barrier every step
}
```

The state `S` is loaded into registers once, updated in registers each
step, and written back once at the end — so the state itself is not the
HBM traffic; the traffic is streaming q,k,v,g,β per token. Two properties
matter:

1. **Sequential in the sequence dimension.** The recurrence means token
   `t` cannot start until token `t−1`'s state update finishes. A 50K
   prefill is 50K *dependent* steps. More context = *longer* at the same
   work rate, not *more* parallel work. This is the "never scaled up":
   power stays flat while tok/s collapses (567 → 165, −71%, from 16K to
   50K).
2. **Low arithmetic intensity, tensor cores idle.** Each step is a handful
   of scalar FMAs per state element plus a warp-reduce sync. It is
   memory- and dependency-bound, so the compute units — including the
   tensor cores — sit mostly idle. An idle-compute, memory-bound kernel
   draws a fraction of a saturated one: ~160 W sustained vs vLLM's 540 W.

And it is **75% of the model** (48/64 layers), so the whole prefill is
power-capped by this one kernel. The `//TODO: Add chunked kernel for even
faster pre-fill` at `gated_delta_net.cu:180` is the acknowledged gap.

### The fix, and why vLLM wins

A **chunked** implementation splits the sequence into blocks. *Within* a
chunk the state update becomes a parallel matrix product (tensor cores,
high power); only the *inter-chunk* state handoff stays sequential. vLLM
runs exactly that — so its power holds at ~540 W and it gets *faster* with
context (8,333 → 6,713 tok/s, −21%) instead of slower. The 160 W is not a
dequant or GEMM story at all; it is that 75% of this model's prefill is a
sequential, memory-bound recurrence that a chunked kernel would
parallelize. That is "Cause 2" in the blog, and the bigger term.

---

## 3. Is GDN engine-independent? Yes — the math is fixed, the kernel is not

GDN is a **model-architecture** component: Qwen3.8-27B's weights and
config define which layers are GDN and what `S`, `g`, `β` mean. It is not
part of any inference engine. Consequence: **every engine that runs this
model has to implement the math itself.** The algorithm is fixed; the
*kernel implementation* is per-engine and can differ a lot:

| | llama.cpp | vLLM |
|---|---|---|
| where | inline in the CUDA backend, `ggml/src/ggml-cuda/gated_delta_net.cu` | delegated to the **flash-linear-attention** component, vendored at `vllm/third_party/flash_linear_attention/` |
| prefill | unchunked sequential kernel (the `:180` TODO) | **chunked** — `qwen_gdn_linear_attn.py` → `self.chunk_gated_delta_rule`; backend is FlashInfer's JIT GDN prefill kernel or the in-tree CuteDSL one; chunked math in `ops/chunk*.py` |
| decode | same kernel, `n_tokens` small | **fused recurrent** — `fused_recurrent_gated_delta_rule_packed_decode` / `fused_sigmoid_gating_delta_rule_update` (`ops/fused_recurrent.py`) |

So the intuition "in vLLM it could be another component" is exactly right:
vLLM's GDN lives in the flash-linear-attention library (plus an optional
FlashInfer JIT kernel), not in vLLM's core. llama.cpp writes its own
kernel inline. Same model, same math, two very different kernels — which
is the whole reason the two engines disagree by 15–40× on prefill for
*this* model while being far closer on an all-full-attention one.

**Do not generalise on the attention axis.** Qwen3.8-27B is a *dense*
model in the usual sense (non-MoE — all 27B params active per token); the
caveat is about **attention type**, not MoE-ness. For a conventional
all-full-attention model (no GDN/linear-attention layers) llama.cpp's
prefill is far more competitive; the gap here is specific to a model that
is 75% linear-attention and to llama.cpp not yet having the chunked
kernel.

---

## Summary

Gated DeltaNet is the linear-attention layer in 48/64 of Qwen3.8-27B's
layers: it keeps a fixed-size 128×128 state per head, updated per token by
a gated, delta-corrected rank-1 rule — O(1) memory, lossy recall. The math
is fixed by the model, so **every engine must implement it**; llama.cpp
does so with an inline, unchunked, sequential kernel
(`gated_delta_net.cu:63`, TODO at `:180`) that is memory- and
dependency-bound with idle tensor cores, which is why its prefill holds at
~160 W and gets *slower* as context grows. vLLM delegates GDN to the
flash-linear-attention component and runs a **chunked** prefill (tensor
cores) plus a fused-recurrent decode, so it holds ~540 W and gets *faster*
with context. The ~160 W sustained draw is this kernel, not the GEMMs.
