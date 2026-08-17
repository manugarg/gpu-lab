# Perplexity: comparing quantizations

Reference for answering "did this quantization make the model worse?"
Companion to `quantization.md` and `llamacpp-vs-vllm.md`.

---

## 1. What it measures

How *surprised* a model is by text it did not generate.

Take a held-out text. At each position, ask what probability the model
assigned to the token that actually came next. Average the
log-probabilities, negate, exponentiate:

```
perplexity = exp( -(1/N) * sum( log P(token_i | tokens_<i) ) )
```

Lower is better. Perplexity of 5 means the model was on average about as
uncertain as picking uniformly among 5 options. A perfect predictor
scores 1.

## 2. Why it's the right tool for quantization

Quantization degrades the model's probability estimates. Feed the *same*
text through two quantizations of the *same* model with the *same*
tokenizer, and perplexity isolates exactly that degradation as a single
number — no task design, no grading, no judgement calls.

That is also its limit: it measures next-token prediction, not whether
the model writes working code. See section 6.

## 3. Running it here

Both engines expose token log-probabilities over the *prompt*, which is
what you need (you're scoring existing text, not generating).

**vLLM** — verified working:

```
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"unsloth/Qwen3.8-27B-NVFP4","prompt":"<text>",
       "max_tokens":1,"temperature":0,"prompt_logprobs":0}'
```

`prompt_logprobs: 0` returns exactly one entry per position — the
**actual** token, keyed by token id:

```json
{"73111": {"logprob": -4.786, "rank": 18, "decoded_token": " fibonacci"}}
```

Note `rank: 18`. That's the real token, not the model's top guess, which
is precisely what's wanted. Position 0 is `null` — nothing precedes it —
so skip it.

```python
vals = [next(iter(v.values()))["logprob"] for v in prompt_logprobs if v]
ppl  = math.exp(-sum(vals)/len(vals))
```

**llama.cpp** — not yet verified. It exposes logprobs on its completions
endpoint, but the exact field shape hasn't been checked on this build.
Do that before trusting a cross-engine number.

## 4. Pitfalls that silently give a wrong answer

- **Taking the top token's logprob instead of the actual token's.** With
  `prompt_logprobs: 0` there is only one entry so it can't happen, but
  with `prompt_logprobs: N > 0` the response also contains the top-N
  alternatives, and picking the max there computes something that looks
  like perplexity and isn't.
- **Skipping the null first position** — include it and the mean is
  wrong (or it crashes).
- **Different tokenizers.** Perplexity is only comparable across models
  sharing a tokenizer. Fine for two quants of one model; meaningless
  between different model families.
- **Different text.** Obvious, but easy to get wrong when generating
  test data with an unseeded RNG.
- **Chat template contamination.** Use the `/v1/completions` endpoint,
  not `/v1/chat/completions`, so no template text is prepended and
  scored.

## 5. Interpreting the difference

| difference | reading |
|---|---|
| < 1% | noise — decide on speed, quality isn't a factor |
| 1-3% | real but small; unlikely to be visible in coding |
| > 5% | genuine degradation, weigh against whatever it bought |

Two well-made quants of the same model usually land within 1-2%, so
"no meaningful difference" is the expected result. The value is in
confirming that rather than assuming it.

## 6. What it does not tell you

Perplexity rewards predicting *the exact continuation of one text*. A
model can be marginally worse at that and equal or better at producing
working code, following instructions, or calling tools correctly.

It's also text-dependent: a quantization that degrades on prose may hold
up on code. **Score on text resembling the workload** — for this rig
that means real source files, not Wikipedia.

If the question is "would I notice?", a behavioural diff answers it more
directly: same prompts at temperature 0 through both engines, compare
outputs. Less rigorous, closer to the thing you care about. The two are
complementary — perplexity says whether the weights got worse, the diff
says whether it shows.

## 7. Result (2026-08-17)

Scored 4,608 tokens of real source code (this repo's Python plus
llama.cpp CUDA), using **llama-perplexity's protocol on both sides**:
512-token chunks, scoring only the second half of each so every scored
token has >=256 tokens of context.

| | perplexity |
|---|---|
| llama.cpp `UD-Q5_K_XL` (18.82 GiB) | **2.2542** +/- 0.0876 |
| vLLM `NVFP4` (21.34 GiB) | **2.2832** |

**+1.29% for NVFP4, against llama.cpp's own +/-3.9% error bar — the two
are statistically indistinguishable.** Take the speed; quality is not a
deciding factor between these two quantizations.

### Getting the protocols to match was the whole job

A first attempt gave vLLM 2.5024 against llama.cpp's 2.2542 — an
apparent 11% gap that was **pure methodology**. `llama-perplexity`
scores only the second half of each chunk; I had scored every token,
including early ones with almost no context, which are much harder to
predict. Same model, same text, same engine — different protocol,
different answer.

To match it: pull token ids from vLLM's `/tokenize`, send them as the
prompt (`/v1/completions` accepts a token-id list, guaranteeing
identical tokenization), and average `prompt_logprobs` over positions
`n_ctx/2 .. n_ctx` only.

### llama.cpp cannot do this over the API

Its `/v1/completions` and native `/completion` both return logprobs for
**generated** tokens only — `echo: true` echoes the completion, not the
prompt. The `logprobs.content` array had exactly one entry. Use the
`llama-perplexity` binary instead; there is no server-side route.

### Practical trap

`prompt_logprobs` materialises logprobs across the full 248,320-token
vocabulary for every prompt position — roughly **4.3 GB for 4,300
tokens**. On a GPU at 94% utilisation it OOMs and **kills the engine**,
surfacing as `EngineDeadError` and a 500. Serve with low
`--gpu-memory-utilization` (0.80 worked; 0.72 was too low to fit the KV
cache at all) and score in bounded chunks.

## 8. Caveats on the above

## 7. Open here

Not yet run for `unsloth/Qwen3.8-27B-NVFP4` (vLLM, 21.34 GiB) vs
`unsloth/Qwen3.8-27B-GGUF UD-Q5_K_XL` (llama.cpp, 18.82 GiB). The two
allocate bits differently — NVFP4 keeps attention, `lm_head` and the
last 8 MLP layers at FP8 while quantizing MLP layers 0-55 to 4-bit;
Q5_K_XL is ~5 bits more uniformly — so which is better is genuinely
open. See `qwen3.8-27b-options.md` for the full per-component
breakdown.
