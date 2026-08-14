# Qwen3.8-27B: deployment options

Running notes on deploying Qwen3.8-27B on this rig (one RTX 5090, sm_120).
Not a decision yet — a log of what's been checked and what's still open.
See `quantization.md` for what AWQ/GPTQ/W4A16/Marlin/NVFP4 actually mean,
and `hybrid-attention-and-kv-cache.md` for why this model's KV cache is
smaller per-token than our much smaller Qwen3-14B baseline.

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
lands on GPU, not a parameter-count estimate):

| checkpoint | weights |
|---|---|
| `soyrsoyr` AWQ+GPTQ int4 | 27.67 GB |
| `unsloth` NVFP4 mixed | 23.42 GB |

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

- Does `--linear-backend flashinfer_b12x` actually work correctly (the
  upstream bug it's blocked on might or might not affect this model's
  shapes) and is it faster than the default `FlashInferCutlass` path?
- Does llama.cpp support this architecture yet, if GGUF is ever worth
  revisiting?
- Real benchmark numbers once the model + a checkpoint are actually pulled
  down and run through `bench/run.sh` / `report.py`.
