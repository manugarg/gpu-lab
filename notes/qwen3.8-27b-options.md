# Qwen3.8-27B: deployment options

Running notes on deploying Qwen3.8-27B on this rig (one RTX 5090, sm_120).
See `quantization.md` for what AWQ/GPTQ/W4A16/Marlin/NVFP4 actually mean,
and `hybrid-attention-and-kv-cache.md` for why this model's KV cache is
smaller per-token than our much smaller Qwen3-14B baseline.

**Leaning towards `unsloth/Qwen3.8-27B-NVFP4`** — smallest verified weights
(23.42 GB) among five real checkpoints checked, the only one that shrinks
its footprint by touching `linear_attn` *and* using true 4-bit for most of
it, lands on kernel paths already confirmed fast on this card (FP8 +
FlashInferCutlass NVFP4), from an established quantizer. Not committed —
still want real `bench/run.sh`/`report.py` numbers once it's downloaded,
and the open questions below (esp. MTP/tool-calling) are unresolved.

---

## Model

Dense 27B (not MoE). Hybrid-attention backbone: linear attention on 48 of 64
layers, full attention on the rest. Built-in MTP draft head. 262K native
context, extensible to 1M. Multimodal (vision blocks + merger). HF
architecture class: `Qwen3_5ForConditionalGeneration`
(MTP: `Qwen3_5MTP`) — both already registered in our `~/tools/vllm` checkout
(v0.26.0), so day-1 vLLM support is not blocked on us patching anything.
(source: recipes.vllm.ai/Qwen/Qwen3.8-27B, checked 2026-08-14)

Official recipe's recommended vLLM flags: `--max-model-len 262144` (up to
1010000 extended), `--kv-cache-dtype fp8`, `--reasoning-parser qwen3`,
`--speculative-config` with MTP. Recipe's hardware guidance:
`--tensor-parallel-size 1` for NVFP4, `--tensor-parallel-size 4` for FP8 —
we only have one GPU, so FP8 as recommended by the recipe doesn't fit
(27B params in FP8 alone is ~27GB, no headroom on one 5090 anyway). NVFP4 is
the realistic single-GPU path *if* it's actually fast here — see below.

## sm_120 NVFP4 kernel finding (supersedes the old CLAUDE.md note)

Verified 2026-08-14 against our `vllm-dev` (v0.26.0) + flashinfer
0.6.16.post3 by reading `vllm/model_executor/kernels/linear/__init__.py`'s
`_POSSIBLE_NVFP4_KERNELS[CUDA]` priority list and running the actual
capability checks on this GPU (not assumed from docs or old notes):

Dense NVFP4 linear layers do **not** fall back to Marlin. The kernel that
actually gets picked is `FlashInferCutlassNvFp4LinearKernel` — a real native
FP4 GEMM path. Full chain in CLAUDE.md's "Known sm_120 gaps". The newest
SM120-native kernel (`FlashInferB12x`) exists in our FlashInfer build but is
manually excluded from auto-selection pending an upstream CUTLASS SM121 bug;
opt in with `--linear-backend flashinfer_b12x` if benchmarking it.

NVFP4 *MoE* on SM120 is separately still messy (open vLLM issue #31085,
various FlashInfer SM120 MoE correctness bugs) — irrelevant here since this
model is dense, but don't generalize the dense finding to MoE models.

## Checkpoint options found on HF (checked 2026-08-14)

### `soyrsoyr/Qwen3.8-27B-W4A16-AWQ-GPTQ` — real, usable

AWQ smoothing pass + GPTQ quantization (see `recipe.yaml` in the repo),
int4, group_size 128, symmetric, `compressed-tensors` format. Quantizes
standard `Linear` targets broadly (attention proj + MLP proj together);
excludes `lm_head`, vision blocks, and `linear_attn` layers (kept
higher-precision). Runs on Marlin as its intended first-class path — not a
fallback, since this is genuine int4, not FP4.

Caveat: uploaded same day as the model, individual uploader (not an
established quantizer), 0 downloads at check time. Smoke-test outputs before
trusting it.

### `unsloth/Qwen3.8-27B-NVFP4` — real, usable, mixed-precision

Not uniform NVFP4. Checked `config.json`'s `quantization_config` directly:

- **FP8**: attention (q/k/v/o proj), `linear_attn` proj, `lm_head`, *and*
  the MLP gate/up/down proj of the last 8 layers (56-63) specifically
- **NVFP4** (group_size 16, FP8 block scales): MLP gate/up/down proj of
  layers 0-55 — the bulk of the parameter count
- KV cache: FP8

Sensible design: FP4 where it saves the most memory (MLP dominates weight
count — matches our own decode-phase component profile showing `mlp (gemm)`
at 57%+ of GPU time), FP8 kept on the more error-sensitive/smaller attention
and output layers. Both precisions land on kernel paths already confirmed
working on this card (FP8 = our existing baseline path; NVFP4 dense = the
FlashInferCutlass path above). Established quantizer (Unsloth), 614 likes on
the companion GGUF repo at check time.

### VRAM footprint: computed 2026-08-14

Weights, from actual on-disk safetensors sizes (HF tree API — this is what
lands on GPU, not a parameter-count estimate): `soyrsoyr` 27.67 GB vs.
`unsloth` 23.42 GB (full four-way ranking including the two checkpoints
below is later in this file).

Counterintuitive: the "int4" checkpoint is *larger*. Its `recipe.yaml`
targets `Linear` broadly but its `ignore` list excludes all 48 `linear_attn`
layers' projections (`in_proj_qkv/z/b/a`, `out_proj`) — those stay at bf16.
Only `self_attn` (16 layers) and MLP (64 layers) get quantized. Unsloth's
checkpoint quantizes the `linear_attn` layers too (to FP8), so despite using
more bits in places, it ends up smaller overall — nothing large is left
unquantized.

KV cache (16 full-attention layers only — see
`hybrid-attention-and-kv-cache.md` for why the other 48 don't need one),
FP8: **32 KB/token** → 0.54 GB at 16K context, 8.6 GB at the native 262K
max, per sequence.

Linear-attention state (Gated DeltaNet, computed from vLLM's actual
`gated_delta_net_state_shape`): **157 MB/sequence, flat** — doesn't grow
with context, scales with concurrency instead. 3.1 GB at 20 concurrent
sequences, 10 GB at 64.

Rough 32GB budget, 16K context + ~20 concurrent sequences, before CUDA
context/activation overhead:
- `soyrsoyr`: 27.67 + 0.54 + 3.14 ≈ **31.4 GB** — razor-thin, may not leave
  room for CUDA context (~1-2GB) or activation workspace.
- `unsloth`: 23.42 + 0.54 + 3.14 ≈ **27.1 GB** — ~5GB headroom.

Not yet done: run both through `bench/run.sh` / `report.py` for real
measured numbers instead of this derivation — vLLM logs its own memory
breakdown at startup, which is the authoritative check.

### `Qwen/Qwen3.8-27B-FP8` — real, official, but too large for us

First-party release from the Qwen team. Fine-grained FP8, block size 128,
`activation_scheme: dynamic`. Only vision blocks are in
`modules_to_not_convert` — unlike the two checkpoints above, this one
*does* quantize the `linear_attn` layers, just at FP8 (8-bit) rather than
NVFP4 (4-bit). Total weights: **30.89 GB** (64 `layers-N.safetensors` +
`outside.safetensors` for embeddings/lm_head + `mtp.safetensors`) — the
largest of the four real checkpoints found, because it's 8-bit almost
everywhere rather than 4-bit where it can afford to be. Leaves only ~1GB
of a 32GB card for KV cache + state + CUDA context — not practical here
despite being the most "official"/trustworthy source.

### `huginnfork/Qwen3.8-27B-NVFP4A16` — real, but larger than Unsloth's

NVFP4 (group_size 16, FP8 scales), `targets: ["Linear"]` broadly — same
structural pattern as `soyrsoyr`'s int4 checkpoint: the `ignore` list
excludes all `linear_attn` sub-layers (336 of 519 ignored tensor names),
`lm_head`, and MTP, leaving them at bf16. Total: **30.99 GB** — larger even
than the official FP8 release, because the untouched `linear_attn` layers
(bf16, 16-bit) dominate the total despite the quantized portion being 4-bit.
Same uploader also has a `Qwen3.8-27B-FP8` variant, not yet checked.

This is the clearest confirmation yet that **whether a checkpoint touches
the `linear_attn` layers matters far more than which bit-width it uses for
the rest** — see the ranked comparison below.

### Ranked by actual weight size (all four verified via HF API, 2026-08-14)

| checkpoint | weights | touches `linear_attn`? |
|---|---|---|
| `unsloth/Qwen3.8-27B-NVFP4` | 23.42 GB | yes (FP8) |
| `soyrsoyr/...-AWQ-GPTQ` | 27.67 GB | no (bf16) |
| `Qwen/Qwen3.8-27B-FP8` (official) | 30.89 GB | yes (FP8, not FP4) |
| `huginnfork/...-NVFP4A16` | 30.99 GB | no (bf16) |

Unsloth's is the only one that's both small enough to leave real headroom
on a 32GB card *and* touches `linear_attn` — it's the only checkpoint doing
both things that actually shrink the footprint.

Checked Unsloth's own collection page (huggingface.co/collections/unsloth/
qwen38) for anything missed: `unsloth/Qwen3.8-27B-FP8` is a mirror of the
official Qwen release, not a distinct option — identical file layout and
total size (30.89 GB, same `layers-N.safetensors`/`outside.safetensors`/
`mtp.safetensors` split). `unsloth/Qwen3.8-2.4T-A95B-GGUF` is a different,
much larger MoE model in the same naming family — not this model.

### `shawnw3i/Qwen3.8-27B-AWQ-MTP` — placeholder, skip

HF repo exists but contains only `.gitattributes` — no weights. Checked
directly via the HF API's file listing, not just the search index (which
still shows it as a hit).

### `unsloth/Qwen3.8-27B-GGUF` — different serving stack, not vLLM

llama.cpp format. Real repo, 28 files, many quant levels (Q3_K_S through
Q8_0, plus Unsloth's "UD" dynamic-quant variants). "Runs on 17GB RAM" claim
≈ `UD-Q4_K_XL.gguf` (17.9GB) or `Q4_K_M` (17.1GB) — closest matches by size.
This targets minimum-spec accessibility, not performance on this rig, which
isn't RAM/VRAM constrained. Not evaluated further: haven't checked whether
llama.cpp has this hybrid-attention + MTP architecture implemented at all —
a GGUF file existing doesn't imply the inference code does, same issue as
vLLM's own architecture registry, just a different project's registry.

## Known vLLM bug: MTP breaks tool calling (open, likely still applies)

vLLM issue #46249: enabling MTP speculative decoding
(`--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`) on
Qwen3.6-27B breaks tool calling on the Responses API — "Failed to advance
FSM for request", a grammar-rejection error. Works fine with MTP disabled.
Filed against v0.23.1rc1.dev207, **still open** at check time, blamed on a
regression in PR #45413, no fix or workaround beyond removing
`--speculative-config` or reverting to v0.23.0.

This was filed against Qwen3.6-27B, not 3.8 — but confirmed via its
`config.json` that Qwen3.6-27B uses the exact same
`Qwen3_5ForConditionalGeneration` architecture class as Qwen3.8-27B, i.e.
the same vLLM code path. Worth testing tool calling with MTP enabled before
relying on both — the recipe recommends MTP for speed and touts "agentic
planning" as a strength, and those two recommendations may conflict right
now.

## Unverified external claims

**SGLang tweet** (x.com/sgl_project, 2026-08-14): "206.1 tok/s decode on a
single RTX 5090, NVFP4 plus DSpark." No batch size, sequence length, or
methodology given — likely single-stream, and DSpark (per
`RadixArk/Qwen3.8-27B-DSpark` on HF, tagged `speculative-decoding`) is
probably a speculative-decoding scheme stacked on NVFP4, which inflates
tok/s via draft-token acceptance rather than raw kernel speed alone. Not
independently verified. The underlying premise (NVFP4 works well on RTX
5090) is *plausible* though — see our own kernel-selection finding above,
which was verified independently on vLLM, not taken from this tweet.

**Unsloth tweet** (x.com/UnslothAI, 2026-08-14): GGUF + NVFP4 uploads — both
checkpoints verified real (see above), but the "run on 17GB RAM" framing is
about accessibility, not about this rig specifically.

## Open questions

- Does vLLM #46249 (MTP breaks tool calling) actually reproduce on our
  v0.26.0 checkout, or has it been fixed since the v0.23.x it was filed
  against? Test before assuming either way.
- Does `--linear-backend flashinfer_b12x` actually work correctly (the
  upstream bug it's blocked on might or might not affect this model's
  shapes) and is it faster than the default `FlashInferCutlass` path?
- Does llama.cpp support this architecture yet, if GGUF is ever worth
  revisiting?
- Real benchmark numbers once the model + a checkpoint are actually pulled
  down and run through `bench/run.sh` / `report.py`.
