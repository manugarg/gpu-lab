# Gated DeltaNet: the compute side — and why it is NOT the prefill bottleneck

Companion to `hybrid-attention-and-kv-cache.md` (the memory side) and
`llamacpp-vs-vllm.md` (the measurement). This note covers what Gated
DeltaNet *computes* and who has to implement it. It also records the nsys
finding that **GDN is only ~2% of GPU time** in a 16K prefill: the dominant
term is the full-attention **flash kernel**, not GDN. (This supersedes the
earlier claim in this note — and in the blog's "Cause 2" — that GDN's
unchunked kernel is the dominant term.) Caveat: the flash kernel's 83% share
is inflated by a **build misconfiguration** in the profiled binary (it saw a
1-SM GPU — see §2). Re-profiled on the correctly-linked build at 50K, the
ordering inverts entirely:

| kernel | broken @16K | **fixed @50K** |
|---|---|---|
| `mul_mat_q` (GEMMs) | 13.2% | **58.3%** |
| `flash_attn_ext_f16` | 82.9% | 21.0% |
| `gated_delta_net_cuda` | 2.0% | 10.4% |

So GDN is not the bottleneck on either build, but it is not negligible
either once attention is fixed — 10.4% at 50K, where a chunked kernel
would still be worth having. The prefill bottleneck on a working build is
the **GEMM**, which is where vLLM's remaining ~2.2x advantage lives:
NVFP4 through CUTLASS on native FP4 units against Q5_K through int8 IMMA
with k-quant unpacking per tile.

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
  **2.26 TFLOP/s ≈ 1.08% of the 5090's ~210 TFLOP/s dense FP16 tensor
  peak**. Compare that to the fraction of the GPU actually running: 2
  blocks of 170 SMs is **1.18%**. The two agreeing is the diagnosis —
  per-SM the kernel performs as designed; the work simply isn't spread
  across the hardware. (The GEMMs run at ~150 TFLOP/s, a normal
  quantized-GEMM rate.)

**The cause — a build misconfiguration, not a kernel defect.** The source's
`launch_fattn` (`ggml/src/ggml-cuda/fattn-common.cuh:1120`) has a **stream-K**
path that launches with `blocks_num.x = min(max_blocks_per_sm × nsm, …)`. It
does exactly what the source says: on the profiled binary,
`max_blocks_per_sm × nsm = 2 × 1 = 2`, so the grid is (2,1,1). The `nsm=1` is
garbage — the profiled binary (built 2026-08-14) was compiled with CUDA 13.3
headers but linked against the distro's **CUDA 12.4 `libcudart.so.12`**, and
`cudaDeviceProp`'s layout changed between the two, so `multiProcessorCount`
read back as **1** instead of 170 (see CLAUDE.md "Build hazards"). Linking
against `libcudart.so.13` restores `nsm=170` and the grid to 340 blocks
(2 per SM × 170 SMs). Re-measured on the fixed build: 16K prefill **567 →
3,244 tok/s** (CLAUDE.md).
The stream-K code was present and correct all along — this was a build bug on
this box, not a llama.cpp kernel defect.

**On the ~160 W figure.** The "idle tensor cores → low power" behaviour
that this note originally attributed to GDN actually characterizes the
flash kernel *as profiled* (2-block grid → 99% idle). GDN's kernel *is*
sequential and memory-bound (below), but at 2% of GPU time it cannot be the
sustained draw. Revisit the 160 W against the flash kernel, not GDN — and
note the 2-block grid (hence the idle-SM power signature) is a build
artifact; on the fixed build the flash kernel uses all 170 SMs and the power
picture changes.

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
finding implies: in the profiled (broken) build, the prefill gap is driven
by the **full-attention** flash kernel (§2), not the GDN layers. The 2-block
grid is a **build artifact** (wrong `libcudart` → `nsm=1`), not a property
of the model or the source — it hit every full-attention layer in the
profiled binary (an all-full-attention model would run the 2-block flash
kernel on all 64 layers, not 16). The "llama.cpp is competitive on
all-full-attention prefill" claim was measured on the broken build and must
be re-checked against the fixed build (correct `libcudart`).

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
is *not* the prefill bottleneck. The dominant term in the profiled build is
the full-attention **flash kernel** (`flash_attn_ext_f16`), which that binary
launches with a **2-block grid** on a 170-SM GPU (~99% idle, ~2.26 TFLOP/s
≈ 1.08% of the ~210 TFLOP/s peak) and which accounts for **83%** of GPU
time. The 2-block grid is a **build misconfiguration** (the profiled binary
linked the distro's CUDA 12.4 `libcudart`, so `nsm` read back as 1 — see
CLAUDE.md "Build hazards"), not a kernel defect: linking `libcudart.so.13`
restores `nsm=170` and the grid to 340 blocks (2 per SM × 170 SMs), and the
fixed build runs 16K prefill at **3,244 tok/s** (was 567). The earlier "GDN caps the ~160 W / prefill" claim
(and the blog's "Cause 2") is superseded by this.
