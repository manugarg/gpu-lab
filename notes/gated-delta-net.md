# Gated DeltaNet: the compute side — and why it is NOT the prefill bottleneck

Companion to `hybrid-attention-and-kv-cache.md` (the memory side) and
`llamacpp-vs-vllm.md` (the measurement). This note covers what Gated
DeltaNet *computes* and who has to implement it. It also records the nsys
finding that **GDN is only ~2% of GPU time** in a 16K prefill: the real
prefill bottleneck is the full-attention **flash kernel**, not GDN. (This
supersedes the earlier claim in this note — and in the blog's "Cause 2" —
that GDN's unchunked kernel is the dominant term.)

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

## 2. The prefill bottleneck is NOT GDN (nsys, 16K prefill)

An nsys profile of a 16K-token prefill (`llama-prefill-16k.nsys-rep`)
breaks down GPU kernel time as:

| kernel | time | % | instances | grid |
|---|---|---|---|---|
| `flash_attn_ext_f16<256,256,8,8>` | 23.4s | **83%** | 512 | **(2,1,1)** |
| `mul_mat_q` (all GEMMs) | 3.7s | 13% | ~16,300 | — |
| `gated_delta_net_cuda` | 0.58s | **2%** | 1,584 | — |

GDN is **2% of GPU time**, not the dominant term. The dominant term is the
full-attention **flash kernel**, and it is slow for a concrete, fixable
reason:

- It is launched with a **grid of (2,1,1) — 2 thread-blocks — on a 170-SM
  GPU**, so ~99% of the SMs sit idle during attention. 512 = 16
  full-attention layers × 32 query-blocks (16384/512); each launch handles
  one (layer, query-block). The 3.5→90.9 ms per-launch spread is the
  causal-mask gradient (early blocks attend to few KV tokens, late ones to
  all 16K).
- At 16K causal, attention ≈ 53 TFLOP, so the kernel runs at 53/23.4 ≈
  **2.3 TFLOP/s ≈ 0.1% of the 5090's ~2000 TFLOP/s F16 tensor peak**. The
  tensor cores are not slow — they are almost entirely idle. The GEMMs run
  at ~150 TFLOP/s (~7% of peak, a normal GEMM); the ~67× efficiency gap
  matches the 85× grid ratio (170/2).

**The likely fix.** The current source's `launch_fattn`
(`ggml/src/ggml-cuda/fattn-common.cuh:1120`) has a **stream-K** path that
launches with `blocks_num.x = min(max_blocks_per_sm × nsm, …)` = **170**
(all SMs). The profiled binary (built 2026-08-14) launched with 2, so it
**predates that optimization**. Rebuilding from the current checkout should
make attention use all 170 SMs — up to ~85× on that kernel (23.4s → ~0.3s),
dropping the 16K prefill from ~28s to ~5s and making the GEMMs the new
bottleneck. (Verification via rebuild + re-profile is in progress.)

**On the ~160 W figure.** The "idle tensor cores → low power" behaviour
that this note originally attributed to GDN actually characterizes the
flash kernel (2-block grid → 99% idle). GDN's kernel *is* sequential and
memory-bound (below), but at 2% of GPU time it cannot be the sustained
draw. Revisit the 160 W against the flash kernel, not GDN.

### GDN's kernel, for the record

llama.cpp's GDN kernel (`ggml/src/ggml-cuda/gated_delta_net.cu`) *is* a
single **sequential loop over every token** (`:63`):

```c
for (int t = 0; t < n_tokens; t++) {        // 50K iterations for a 50K prefill
    ...
    s_shard[r] = g * s_shard[r] + k * delta; // :103 — depends on t-1's state
    ...
    warp_reduce_sum(...);                    // a sync barrier every step
}
```

So it *does* have the two properties that were (wrongly) blamed for the
prefill gap: sequential in the sequence dimension (token `t` waits on
`t−1`), and low arithmetic intensity with idle tensor cores (memory- and
dependency-bound). vLLM's **chunked** GDN prefill (tensor cores within a
chunk, sequential only at chunk boundaries) is genuinely faster than this —
the `//TODO: Add chunked kernel` at `:180` is a real gap. But at 2% of GPU
time, closing it would not explain the 15–40× prefill gap; that gap is the
flash kernel above.

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
caveat is about **attention type**, not MoE-ness. Note the reversal this
finding implies: the prefill gap is driven by the **full-attention** flash
kernel (§2), not the GDN layers. The 2-block grid is a property of the
profiled binary's `launch_fattn`, not of the model — so it hits every
full-attention layer in any model (an all-full-attention model would run
the slow flash kernel on all 64 layers, not 16). The "llama.cpp is
competitive on all-full-attention prefill" claim was measured on the
pre-stream-K build and should be re-checked against a rebuilt binary.

---

## Summary

Gated DeltaNet is the linear-attention layer in 48/64 of Qwen3.8-27B's
layers: it keeps a fixed-size 128×128 state per head, updated per token by
a gated, delta-corrected rank-1 rule — O(1) memory, lossy recall. The math
is fixed by the model, so **every engine must implement it**; llama.cpp
does so with an inline, unchunked, sequential kernel
(`gated_delta_net.cu:63`, TODO at `:180`), and vLLM delegates GDN to the
flash-linear-attention component with a **chunked** prefill plus a
fused-recurrent decode.

But an nsys 16K-prefill profile shows GDN is only **~2% of GPU time** — it
is *not* the prefill bottleneck. The bottleneck is the full-attention
**flash kernel** (`flash_attn_ext_f16`), which the profiled binary launches
with a **2-block grid** on a 170-SM GPU (~99% idle, ~2.3 TFLOP/s ≈ 0.1% of
peak) and which accounts for **83%** of GPU time. The current source's
stream-K `launch_fattn` fixes the grid (→ 170 blocks); rebuilding should
collapse that 23.4s to ~0.3s. The earlier "GDN caps the ~160 W / prefill"
claim (and the blog's "Cause 2") is superseded by this.
