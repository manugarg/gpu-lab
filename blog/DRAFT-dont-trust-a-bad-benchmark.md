# Running Qwen3.8-27B on an RTX 5090: an unsuspecting bug, and the journey to a capable local LLM

*All numbers measured on one machine — RTX 5090 (32 GB, sm_120),
Ubuntu 26.04, CUDA 13.3.*

---

This is what runs on my desk now: a 27-billion-parameter hybrid-attention
model, quantized to about 21 GB, serving a coding agent from a single
consumer GPU.

| llama.cpp, Qwen3.8-27B UD-Q5_K_XL | @16K context | @50K context |
|---|---|---|
| prefill | 3,069 tok/s | 2,643 tok/s |
| decode (speculative head on) | 101.3 tok/s | 92.4 tok/s |
| context window | 229,376 tokens | |

It reviews pull requests, and it's quick enough that I stopped thinking
about it. Those decode figures are measured against a prompt the model
has to actually answer — which matters more than it sounds like, for
reasons I'll come back to.

Getting there took three weeks. For most of them the same hardware ran
the same work at roughly a fifth of that speed, and I had a complete set
of measurements explaining why: an engine comparison, a kernel-level
mechanism, a quantization tradeoff, a result about speculative decoding.
Every one of them reproducible. Every one stable across repeated runs.
Every one wrong.

They were all artifacts of my own build, which linked a CUDA runtime one
major version away from its compiler and therefore believed the GPU had
**one** streaming multiprocessor instead of 170. The attention kernel ran
on 1.2% of the card for three weeks.

I never questioned any of it, because the numbers were bad. A
surprisingly *fast* result gets audited. A slow one gets written down and
turned into a comparison table.

What follows is what I should have checked, in the order I should have
checked it. The bug is incidental. The checks are not.

## "Slow" is not a fact about the software

When software underperforms we attribute it to the software. That's
usually reasonable, and in this stack it is frequently wrong, because a
running model sits at the end of a long chain: a build system that
resolved a dozen libraries, a kernel-selection heuristic, a quantization
format, a driver, a runtime reporting what the hardware can do.

Any link can degrade quietly. Almost none of them announce it.

This is the ordinary state of GPU/ML tooling, not an exotic failure. In
three weeks on one machine, all of these were live:

- A quantized-matmul path that asserts on certain weight layouts and
  **silently falls back** to a slower kernel.
- A kernel that exists in the installed library but is **excluded from
  auto-selection** pending an unrelated upstream bug.
- An arch check that reads **the framework's build flags rather than the
  device**, and misreports capability when they disagree.
- A build that pairs the right compiler with the wrong runtime library
  and gets **a wrong constant back about the hardware**.

None of these produce an error. Every one produces a number. Benchmark
on top of them and you get a result that is precise, repeatable, and
about your configuration rather than about the software you think you're
evaluating.

## Attribute the time before you explain it

The first mistake is explaining a number you haven't decomposed.

For LLM inference the minimum split is **prefill** (reading the prompt)
versus **decode** (generating tokens). They're bound by different
things — prefill is compute-bound, decode is memory-bandwidth-bound — so
"tokens per second" without that split says very little. In my case
prefill dominated wall time on one engine and was almost invisible on
the other. Until you know which, every hypothesis is a guess about the
wrong half.

Then split by kernel. A profiler summary took minutes and killed two
theories I'd carried for a week:

```
flash_attn_ext_f16     23.4 s    82.9%
mul_mat_q (GEMMs)       3.7 s    13.2%
gated_delta_net_cuda    0.6 s     2.0%
```

I had published that the gap came from matrix-multiply kernels and from
a known-unoptimized linear-attention kernel. Together they were 15% of
runtime. Neither could produce the effect I'd attributed to them, and I
would have kept believing both indefinitely — both stories were
plausible, and I had never looked at the distribution.

**A performance theory you can't point at in a profile is a story, not a
diagnosis.**

## Utilization lies. Power doesn't.

`nvidia-smi` reported **100% GPU utilization** throughout. That metric
means a kernel is *resident*, not that the hardware is busy. One thread
block on a 170-SM GPU reads as 100% utilized.

Power is much harder to fake. Silicon that switches draws current.

```
nvidia-smi -l 1 --format=csv \
  --query-gpu=power.draw,utilization.gpu
```

On an identical 16K prefill the card averaged **202 W of a ~575 W
budget** while claiming full utilization. After the fix the same work
averaged **532 W**, peaking at 609. That number costs nothing to collect,
and I had been watching it for days without thinking about it.

![Board power during an identical 16,000-token prefill. The broken build
holds about 200 W for 29 seconds; the fixed build spikes to roughly 600 W
and finishes in 5 seconds, using half the total
energy.](power-draw-broken-vs-fixed.png)

The shapes are the whole story. Same prompt, same GPU, same model: a low
flat plateau that runs forever, against a short square block that pins
the card. Those are means of 202 W and 532 W; the fixed run's plateau
sits near 600 W, just above the card's rated 575. And the area under
each curve is energy — **5.85 kJ broken vs 2.87 kJ fixed**. The misconfigured build didn't just take 5.9x longer,
it burned roughly *twice the energy* to do the same work.

If you publish one extra column with a benchmark, publish watts.

## Compare what you got to what exists

The most useful check I found is a ratio, and it needs no profiler.

Work out the arithmetic your workload actually requires, divide by the
time it took, compare to the hardware's peak. Attention over 16K tokens
across 16 layers is ~53 TFLOP. It took 23.4 s. That's 2.26 TFLOP/s
against roughly 210 TFLOP/s of dense FP16 throughput — **1.08% of peak**.

Then ask, separately, what fraction of the machine was *running*. The
attention kernel launched 2 thread blocks on a GPU with 170 SMs:
**1.18%**.

Those two agreeing is the whole diagnosis:

> Achieved ≈ available → the kernel is fine, and starved.
> Achieved ≪ available → the kernel is slow.

A stopwatch cannot tell those apart; both are "23 seconds." They need
completely different fixes. This comparison separates them in about ten
minutes of arithmetic, and it's the check I'd keep if I could keep only
one.

## Read the launch geometry, then stop reading source

Profilers record grid and block dimensions per kernel. Pull them out and
compare against your SM count. A grid of 2 on a 170-SM GPU isn't a
tuning opportunity, it's a defect.

Then a warning about what *didn't* work. Two of us independently traced
the launcher's arithmetic — carefully, from correct source — and both
concluded the observed grid was **impossible**. We were right. Given the
inputs the source implies, that grid cannot occur. The reasoning was
sound and the conclusion was useless, because one input wasn't what the
source implied.

Twenty lines of `printf` in the launcher answered it in one run.

**When careful reading contradicts measurement, the measurement is right
and one of your assumed inputs is wrong.** Print the inputs. This took
me far too long to reach for, because reading feels rigorous and printf
feels crude — and here, reading was the thing generating confident wrong
answers.

## What it turned out to be

The build was compiled against CUDA 13 headers but linked at runtime
against the distro's CUDA 12 `libcudart`. The `cudaDeviceProp` struct
changed layout between those versions, so the binary read device
properties at the wrong offsets and got:

```
multiProcessorCount = 1     (actual: 170)
```

Not a crash, not a zero — a **plausible small number**. Every kernel
launcher that sizes its grid from the SM count then did something
locally reasonable and globally catastrophic.

Pinning the compiler with `-DCMAKE_CUDA_COMPILER`, the usual advice on a
multi-toolkit box, does *not* prevent this: the build system resolves
the runtime library separately. Pin both, then verify what actually got
linked.

```
ldd your-lib.so | grep cudart
```

The specific trap is narrow. The general form is not: **if a constant
describing your hardware arrives from a library, log it once at startup
and assert it's sane.** `assert(nsm > 1)` would have saved all of this.

## What it cost

Same build, same flags, same prompts — only the linked runtime changed:

| speculative decoding off | before | after | |
|---|---|---|---|
| prefill @16K | 567 tok/s | **3,303** | 5.8x |
| prefill @50K | ~165 tok/s | **2,754** | 16.7x |
| decode @50K | 35.2 tok/s | **57.4** | 1.6x |
| power (16K prefill, mean) | 202 W | **532 W** | |

A 49,737-token prefill went from about five minutes to 18 seconds.

Speculative decoding is off in that table so it isolates the build
change; the deployed configuration at the top of this post has it on,
which is why its decode figures are higher.

Four separate published "findings" dissolved with it:

- The 40x engine gap became 2.2x.
- "f16 KV cache is 3.7x slower at long context" became a 2% spread. The
  format pushing the most data through the starved kernel had simply
  suffered most.
- "Speculative decoding's benefit evaporates with context" — 2.07x down
  to 1.06x — became a flat ~2x. Speculation amortizes weight reads and
  can't amortize attention; when attention is artificially dominant,
  speculation looks useless.
- A public benchmark I had confidently corrected turned out to be
  roughly right. My correction was the artifact.

None of those were careless measurements. They were careful measurements
of a misconfigured machine, which is harder to notice and much easier to
publish.

## The short version

Before believing a performance number — especially a bad one, and
especially your own:

1. **Split the time by phase, then by kernel.** Don't explain an
   aggregate.
2. **Watch power, not utilization.** Utilization only says a kernel
   exists.
3. **Compare achieved throughput to peak, and used parallelism to
   available.** If both fractions match, nothing is slow — something is
   starved.
4. **Check launch geometry against your hardware's width.**
5. **Log the constants your code reads about the hardware**, and assert
   they're plausible.

None of that is exotic tooling. It's four shell commands and some
arithmetic. It would have saved me three weeks.

---

*Reproduced against llama.cpp at commit `9d57ce4`, CUDA 13.3, driver
595. The profiling and disassembly here were done with an AI assistant
driving the tooling; the confident wrong turns were a joint effort.*
