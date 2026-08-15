# llama.cpp vs vLLM on this rig (Qwen3.8-27B)

Measured 2026-08-14 on the RTX 5090. Companion to
`qwen3.8-27b-options.md`. Everything here is from runs on this machine,
not from anyone's published numbers.

---

## Building llama.cpp here

`cmake` was not installed. Installed it as an isolated uv tool rather
than into either vllm venv — CLAUDE.md's packaging rules exist because
re-resolving those venvs has cost hours, and a build tool has no business
in them:

```
uv tool install cmake        # lands in ~/.local/bin
```

Then, with `SSH_AUTH_SOCK` set (see CLAUDE.md — a global
`url.git@github.com:.insteadof https://github.com/` rewrites even https
clones through SSH):

```
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/tools/llama.cpp
cd ~/tools/llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc
cmake --build build --config Release -j $(nproc)
```

**The `-DCMAKE_CUDA_COMPILER` pin is mandatory** — see CLAUDE.md's
"Build hazards". Without it the build dies with a `sm_52` ptxas error
that looks like missing Blackwell support but is actually a mixed
CUDA 12.4/13.3 toolchain.

Do *not* pass `-DCMAKE_CUDA_ARCHITECTURES=120`. llama.cpp's own
`ggml/src/ggml-cuda/CMakeLists.txt` appends `120a-real` for
CUDA >= 12.8 already; overriding it just fights its logic.

Architecture support confirmed present before building (worth checking
first for any new model family): `GGML_OP_GATED_DELTA_NET` in
`ggml/include/ggml.h`, a CUDA kernel at
`ggml/src/ggml-cuda/gated_delta_net.cu`, and `LLM_ARCH_QWEN35` in
`src/llama-arch.cpp`.

## Single-stream throughput, 1024 in / 256 out

| | llama.cpp (UD-Q5_K_XL) | vLLM (NVFP4) |
|---|---|---|
| decode | 69.98 ± 0.32 tok/s | ~68.1 tok/s (TPOT 14.69 ms) |
| prefill | 2211.69 ± 0.93 tok/s (~463 ms) | ~8390 tok/s (TTFT 122 ms) |
| weights on GPU | 18.82 GiB | 21.34 GiB |

**Decode is a tie** (~3% apart). Both beat the ~60 tok/s figures being
reported publicly for this model on a 5090.

**Prefill is not close — vLLM ~3.8x faster.** This is the difference
you'd actually feel on long prompts or a large system prompt, and it
widens with prompt length.

Commands:
```
# llama.cpp
./build/bin/llama-bench -m <gguf> -ngl 99 -p 1024 -n 256

# vLLM (single stream: --max-concurrency 1)
vllm bench serve --model <model> --dataset-name random \
  --random-input-len 1024 --random-output-len 256 \
  --max-concurrency 1 --num-prompts 5
```

Note vLLM's headline "Output token throughput" (66.17 tok/s) folds TTFT
into wall time; the decode row above uses TPOT so it's comparable to
llama-bench's pure `tg256`.

## Maximum context

| | max context | KV dtype |
|---|---|---|
| llama.cpp (UD-Q5_K_XL) | **262,144** (full native) | q8_0 |
| vLLM (NVFP4) | **189,728** | fp8_e4m3 |

llama.cpp reaches the model's full native 262K; vLLM tops out around
190K. vLLM reports its own ceiling precisely rather than guessing:

```
Available KV cache memory: 5.95 GiB
ValueError: To serve at least one request with the model's max seq len
(262144), 8.18 GiB KV cache is needed ... the estimated maximum model
length is 189728.
```

That vLLM figure is with settings tuned *in its favour* —
`--gpu-memory-utilization 0.95 --max-num-seqs 1` — so ~190K is close to
its real ceiling with this checkpoint, not an artifact of a conservative
config.

**This is a quantization difference, not an engine-efficiency
difference.** The gap traces almost entirely to weights: 2.52 GiB more
free x 32 KB/token ~= 82,575 extra tokens predicted vs 72,416 observed
(remainder goes to compute buffers and linear-attention state). Point
vLLM at a smaller checkpoint and most of the gap closes.

**Both need quantized KV to get there.** llama.cpp's default f16 KV is
64 KB/token; at 262K that's a 16 GiB allocation and it OOMs outright:

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 16384.00 MiB on
device 0: cudaMalloc failed: out of memory
```

`-fa on -ctk q8_0 -ctv q8_0` is what makes it fit (28.7 GiB used of
32.1 GiB).

## Gotcha: llama.cpp mislabels the UD quant

`llama-bench` reports `UD-Q5_K_XL` as `qwen35 27B Q4_K - Small`.
Unsloth's "UD" dynamic quants mix tensor types per layer, and llama.cpp
names the file by its dominant type. The size (18.82 GiB) confirms it is
the Q5_K_XL file. Don't read that label as "we downloaded the wrong
quant".

## Reading this

- Decode parity means raw generation speed is *not* a reason to pick
  either one for interactive single-user use.
- vLLM wins prefill decisively, and wins concurrency by construction
  (735 tok/s aggregate at 20 concurrent, vs llama.cpp being oriented at
  single-stream).
- llama.cpp wins max context here (262K vs 190K), driven by the smaller
  quant, plus ~2.5 GiB lower footprint and a simpler operational story
  (one binary, no venv).
- Caveat worth re-checking over time: llama.cpp's Gated DeltaNet is
  still, per its own merge PR, "a basic vector implementation, not the
  chunking implementation" — so its decode number may improve, while
  vLLM's path is the more mature one today.
