# Working log: flash_attn grid=2 mystery (2026-08-19)

Companion to `notes/gated-delta-net.md`. That note records the nsys finding
(flash kernel = 83% of prefill GPU time, launched on a 2-block grid). This log
tracked the follow-up: **why does the binary launch only 2 blocks when the
source's stream-K path should launch all 170 SMs?**

Status: **RESOLVED (2026-08-19).** Not a llama.cpp defect — a build
misconfiguration on this box. See CLAUDE.md "Build hazards" (the
`-DCMAKE_CUDA_COMPILER is NOT sufficient` bullet) and "Current deployment"
(SUPERSEDED table), which already carry the full write-up + re-measured numbers.

## Root cause

`libggml-cuda.so` was compiled with **CUDA 13.3 headers** but linked at runtime
against the **distro's CUDA 12.4 `libcudart.so.12`**. `cudaDeviceProp`'s field
layout changed between CUDA 12 and 13, so llama.cpp read garbage out of it:

| linked runtime        | multiProcessorCount | maxThreadsPerSM |
|-----------------------|---------------------|-----------------|
| `libcudart.so.13` (13.3) | **170**          | 1536            |
| `libcudart.so.12` (12.4) | **1**            | 1               |

So `nsm = 1` (garbage), and `launch_fattn` did exactly what the source says:

```
max_blocks = max_blocks_per_sm × nsm = 2 × 1 = 2   ->  grid = (2,1,1)
```

Our arithmetic was right all along — `ntiles_x=64`, `ntiles_dst=256`, exactly
as predicted. The **only** wrong input was `nsm`.

## How it slipped through

`CMakeCache.txt` — right compiler, wrong runtime:
- `CUDAToolkit_BIN_DIR=/usr/local/cuda-13.3/bin` (the `-DCMAKE_CUDA_COMPILER`
  from the build notes worked)
- `CUDA_cudart_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudart.so` (distro 12.4)

CMake resolves the CUDA *runtime* separately from the *compiler*. Same hazard
already in CLAUDE.md, one layer down — and it fails **silently** (no build
error), so it only shows up as a 1-SM GPU at runtime.

## Fix + verification

- Build with `-DCUDAToolkit_ROOT=/usr/local/cuda-13.3`
  `-DCUDA_cudart_LIBRARY=/usr/local/cuda-13.3/lib64/libcudart.so`.
- Verify: `ldd <built>.so | grep cudart` must say `libcudart.so.13`, not `.12`.
- Current `build/bin/llama-server` (built 2026-08-19 12:08) links
  `libcudart.so.13`; CMakeCache now has the correct `CUDA_cudart_LIBRARY`.
- Re-measured on the fixed build (CLAUDE.md): prefill @16K **567 → 3,244 tok/s**,
  @50K **~165 → 2,754 tok/s**.
- Matched re-measurement (both engines, MTP on, realistic fixture) puts the
  50K gap at **2.2× prefill** (5,962 vs 2,659) and **decode at a tie**
  (89.8 vLLM vs 91.8 llama.cpp) — not the 40×/1.8× the broken build implied.
  The earlier "~1.1× decode" used the random-word fixture, which inflates
  speculative decoding; see `bench/make_fixture.py`.

## Implication for the blog
The "llama.cpp prefill is 15–40× slower than vLLM" thesis was largely an
artifact of this build bug. The note's "Cause 2 = GDN" and the flash-kernel
"2-block grid is a llama.cpp defect" framings are both superseded — the blog
needs rework (owner to handle).
