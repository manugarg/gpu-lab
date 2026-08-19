# Your CUDA build may think your GPU has one SM

*Measured on an RTX 5090 (32 GB, sm_120) running Ubuntu 26.04 with CUDA
13.3 installed alongside the distro's 12.4. Every number here is from
this machine, before and after a one-line build change.*

---

## The check

If you build CUDA software from source on a box with more than one CUDA
toolkit installed, run this against whatever you built:

```
ldd your-binary-or-lib.so | grep cudart
```

If your compiler was CUDA 13 and that line says `libcudart.so.12`, stop
reading and fix your build. Your GPU is probably reporting **one
streaming multiprocessor**, and everything downstream of that number is
sized for a GPU that doesn't exist.

Nothing will tell you. There is no warning, no error, no degraded-mode
message. The software runs, produces correct output, and is slow in a
way that looks like a legitimate performance characteristic.

## What actually breaks

`cudaGetDeviceProperties` fills a `cudaDeviceProp` struct. That struct's
field layout **changed between CUDA 12 and CUDA 13**. A binary compiled
against 13's headers reads fields at 13's offsets. If the runtime that
filled the struct was 12, the offsets don't line up.

Ten lines are enough to see it:

```c
#include <cstdio>
int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("SMs=%d  maxThreadsPerSM=%d\n",
           p.multiProcessorCount,
           p.maxThreadsPerMultiProcessor);
    return 0;
}
```

Same source, same GPU, compiled identically — only the linked runtime
differs:

```
linked libcudart.so.13   ->  SMs=170  maxThreadsPerSM=1536
linked libcudart.so.12   ->  SMs=1    maxThreadsPerSM=1
```

Not a crash. Not a zero. A **plausible small number**, which is the
worst possible failure mode, because every consumer of it does something
reasonable with it.

## Why the compiler flag isn't enough

The usual advice for a multi-toolkit box is to pin the compiler:

```
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc
```

That is necessary and it is not sufficient. CMake resolves the CUDA
*runtime library* separately from the compiler, and it will happily pair
a 13.3 compiler with the distro's 12.4 `libcudart`. The build log looks
clean. `CMakeCache.txt` shows both halves of the mismatch, one line
apart, and it is easy to read the first and stop:

```
# correct -- the compiler
CUDAToolkit_BIN_DIR:PATH=/usr/local/cuda-13.3/bin

# 12.4 -- the runtime
CUDA_cudart_LIBRARY:FILEPATH=/usr/lib/.../libcudart.so
```

The fix is to pin both, then verify with `ldd` rather than trusting the
configure output:

```
cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.3 \
  -DCUDA_cudart_LIBRARY=\
      /usr/local/cuda-13.3/lib64/libcudart.so
```

## What it costs

`multiProcessorCount` is not a display value. It is a *scheduling input*.
Kernel launchers use it to decide how many thread blocks to create.

Here is the concrete path in llama.cpp's flash-attention launcher
(`ggml/src/ggml-cuda/fattn-common.cuh`), which is representative rather
than unusual:

```c
const int max_blocks = max_blocks_per_sm * nsm;
// ...
blocks_num.x = std::min(max_blocks, ntiles_KV * ntiles_dst);
```

With `nsm = 1`, that's `2 * 1 = 2`. **Two thread blocks, 128 threads, on
a GPU with 170 SMs.** The attention kernel occupied 1.2% of the card and
left the rest idle, on every launch, for every layer.

Same build, same flags, same prompts — only the linked runtime changed:

| | broken | fixed | |
|---|---|---|---|
| flash-attn grid | 2 blocks | **340 blocks** | |
| prefill @16K | 567 tok/s | **3,244 tok/s** | 5.7x |
| prefill @50K | ~165 tok/s | **2,754 tok/s** | 16.7x |
| decode @16K | 51.9 tok/s | **63.7 tok/s** | 1.23x |
| decode @50K | 35.2 tok/s | **57.4 tok/s** | 1.63x |
| power draw | ~160 W | **570 W** | |

A 49,737-token prefill went from about five minutes to **18 seconds**.

## How to find this if you don't already know it's there

The reason this is worth a post is that the fault is invisible from
above. Here is the ladder that worked, in increasing order of effort.
Each rung is useful on its own.

**1. Watch power, not utilization.** `nvidia-smi` reported "100% GPU
utilization" the entire time. That metric only means a kernel is
*resident*, not that the silicon is busy. The card drew ~160 W of a
~575 W budget. A GPU that claims to be fully utilized and is drawing 28%
of its power budget is not fully utilized.

```
nvidia-smi -l 1 --format=csv \
  --query-gpu=power.draw,utilization.gpu
```

**2. Profile, and look at the distribution, not the total.** A kernel
summary immediately showed where the time actually was:

```
flash_attn_ext_f16     23.4 s    82.9%
mul_mat_q (all GEMMs)   3.7 s    13.2%
gated_delta_net_cuda    0.6 s     2.0%
```

This alone killed two plausible theories — that the matmul kernels were
the problem, and that a known-unoptimized linear-attention kernel was
the problem. Neither could matter; together they were 15% of the time.

**3. Check achieved FLOP/s against peak.** Attention at 16K over 16
layers is about 53 TFLOP of arithmetic. Over 23.4 s that's 2.26 TFLOP/s
against roughly 210 TFLOP/s of dense FP16 tensor throughput — **1.08% of
peak**.

Then compare that to the fraction of the GPU actually running: 2 blocks
of 170 SMs is **1.18%**.

Those two numbers agreeing is the whole diagnosis. Per-SM, the kernel
was performing exactly as designed. Nothing was slow. The work simply
wasn't being spread across the hardware. A kernel that is *inefficient*
and a kernel that is *starved of parallelism* look identical on a
stopwatch and completely different on this comparison.

**4. Print the launch geometry.** Profilers record grid and block
dimensions. Pull them out and sanity-check against your SM count:

```sql
SELECT gridX, gridY, gridZ, blockX, COUNT(*), SUM(end-start)
FROM CUPTI_ACTIVITY_KIND_KERNEL ...
```

A grid of 2 on a 170-SM GPU is not a tuning problem. It is a bug.

**5. Instrument the launcher.** A `printf` of every input to the grid
calculation ended the investigation in one run: every derived quantity
was correct, and `nsm` was 1.

## The part worth generalizing

Reading source code could not have found this, and two separate careful
attempts to trace the launch arithmetic both concluded the observed grid
was *impossible*. They were right. Given the inputs the code appears to
have, a grid of 2 cannot happen. The reasoning was sound and the
conclusion was wrong, because one input was not what the source implied
it was.

That's the transferable lesson, and it isn't really about CUDA:

- **A number that arrives from outside your program is data, not truth.**
  `nsm` came from a library. It was never validated, never logged, and
  never questioned, by anyone, for weeks.
- **Silent misconfiguration is more expensive than a crash.** A build
  that fails costs an hour. A build that succeeds and quietly reports a
  wrong constant produces confident, reproducible, publishable numbers
  that are all wrong.
- **Cheap invariants beat careful reasoning.** `assert(nsm > 1)` would
  have cost nothing and saved all of it. So would printing the device
  properties once at startup.

## What this means for benchmarks you read

Every performance number that came off this machine before the fix was
real, reproducible, stable across repeated runs — and an artifact of one
library version.

Four separate "findings" dissolved when the link was corrected:

- An engine comparison showing a 40x prefill gap became 2.4x.
- A result that f16 KV cache was 3.7x slower than q8_0 at long context
  became a 2% spread, with f16 marginally *fastest*. There was no
  tradeoff to discover; the KV format that moved the most data through
  the starved kernel simply suffered most.
- A finding that speculative decoding's benefit "evaporates with
  context" — 2.07x at 26 tokens down to 1.06x at 50K — became a flat
  ~2x at every context. Speculation amortizes weight reads and cannot
  amortize attention; when attention is artificially dominant,
  speculation looks useless.
- A public benchmark I was confident I had corrected turned out to be
  approximately right. My correction was the artifact.

None of those were sloppy measurements. They were careful measurements
of a misconfigured machine, which is a different thing and much harder
to notice.

So when you read a benchmark — including your own — the useful question
is not just *what context length was this taken at*, but **was the
machine actually working**. Publish your power draw. Publish your
achieved fraction of peak. Those two numbers would have caught this on
day one, and they cost nothing to collect.

---

*Findings reproduced against llama.cpp at commit `9d57ce4`, CUDA 13.3,
driver 595, on Qwen3.8-27B (UD-Q5_K_XL). The profiling and disassembly
in this post were done with an AI assistant driving the tooling; the
wrong turns described above were a joint effort.*
