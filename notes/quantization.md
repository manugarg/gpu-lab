# Quantization: algorithms, kernel, and formats

These get conflated constantly because checkpoint names smash them together
(e.g. "W4A16-AWQ-GPTQ"). They're three different layers of the stack.

---

## 1. Algorithms that produce int4 weights: AWQ and GPTQ

Both are *post-training* quantization methods — no fine-tuning, just a
calibration pass over a small dataset — that decide how to round weights down
to 4 bits.

**AWQ (Activation-aware Weight Quantization)**

Not all weight channels matter equally: the ones that get multiplied by
large-magnitude activations matter more, because rounding error there hurts
output quality more. AWQ finds per-channel scaling factors (searched against
calibration data) that shrink those sensitive weight channels before
quantizing, then folds the inverse scale into the activation side so the
computation stays mathematically equivalent. It's a smoothing pass — cheap,
no gradient computation.

**GPTQ**

A different algorithm: quantizes layer by layer, column by column. After
rounding each weight, it uses Hessian (second-order curvature) information
from calibration data to nudge the *remaining* unquantized weights in that
layer to compensate for the error just introduced. More expensive to run
than AWQ (needs the Hessian approximation and the greedy correction pass),
but generally very accurate at 4 bits.

Checkpoints increasingly chain both: AWQ smooths first, then GPTQ does the
actual per-weight rounding with error compensation. This is what
`soyrsoyr/Qwen3.8-27B-W4A16-AWQ-GPTQ`'s `recipe.yaml` does — see
`qwen3.8-27b-options.md`.

## 2. W4A16 — a shape, not an algorithm

**W**eights at **4** bits (int4), **A**ctivations kept at **16** bits
(bf16/fp16). Both AWQ and GPTQ typically target this shape. It's "weight-only"
quantization: only the checkpoint on disk shrinks; activations are computed
and multiplied at full precision.

## 3. Marlin — the kernel, not an algorithm

Marlin is the CUDA kernel that actually executes a W4A16 GEMM at serving
time. It unpacks 4-bit weights and dequantizes them on the fly against
bf16/fp16 activations, tuned so the 4x memory-bandwidth reduction from int4
weights translates into real speedup instead of being eaten by dequant
overhead. It's vLLM's (and the ecosystem's) standard fast path for serving
*any* int4 W4A16 checkpoint — AWQ-produced or GPTQ-produced, doesn't matter,
they converge on the same on-disk int4 format — on NVIDIA GPUs since Ampere.

This matters for what "fallback" means: when a checkpoint's native kernel
path isn't available and vLLM routes it through Marlin instead, that's a
*fallback* only for formats Marlin wasn't designed for (see NVFP4 below).
For genuine int4 AWQ/GPTQ checkpoints, Marlin isn't a fallback at all — it's
the intended, first-class path.

## 4. NVFP4 — a different number format, not an int4 variant

NVFP4 is a 4-bit *floating-point* format (exponent + mantissa bits, plus
fine-grained per-block FP8 scale factors — typically one scale per 16
values), introduced for Blackwell-generation Tensor Cores. Floating point at
4 bits can represent a wider dynamic range than a fixed-point int4 step size,
which is part of why it can match int4-plus-smoothing accuracy at the same
bit width.

The catch: it needs dedicated tensor-core hardware paths to run fast, unlike
int4 W4A16 which Marlin already runs well on any Ampere-or-newer GPU. Whether
that hardware+kernel support actually exists for a given card is a moving
target, not a given — see `qwen3.8-27b-options.md` and CLAUDE.md's
"Known sm_120 gaps" for what's actually true on this RTX 5090, verified
against the checked-out vLLM/FlashInfer versions rather than assumed. This
whole area (SM120 = consumer Blackwell, distinct from SM100 datacenter
Blackwell) is under active development upstream and has shifted under us
before — re-verify rather than trust a stale note, including this one.

## 5. Mixed-precision checkpoints

Not every checkpoint quantizes uniformly. A checkpoint can quantize different
parts of the model to different formats — e.g. keep attention projections
and the LM head at FP8 (more error-sensitive, smaller share of total
parameters) while quantizing MLP projections to NVFP4 (less sensitive,
dominates parameter count for dense transformer-style models). This gets
most of the memory savings where it matters most while hedging quality risk
where it matters most. `unsloth/Qwen3.8-27B-NVFP4` does exactly this — see
`qwen3.8-27b-options.md`.
