import gzip, json, re, sys, collections

path = sys.argv[1] if len(sys.argv) > 1 else "trace.pt.trace.json.gz"
ev = json.load(gzip.open(path))["traceEvents"]

pat = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
spans = sorted(
    (e for e in ev if e.get("cat") == "gpu_user_annotation"), key=lambda e: e["ts"]
)

phase_us = collections.Counter()
phase_calls = collections.Counter()
phase_tokens = collections.Counter()
mixed_ctx_tokens = mixed_gen_tokens = 0

for e in spans:
    m = pat.match(e["name"])
    if not m:
        continue
    ctx_reqs, ctx_tok, gen_reqs, gen_tok = map(int, m.groups())
    if ctx_tok and not gen_tok:
        phase = "prefill"
    elif gen_tok and not ctx_tok:
        phase = "decode"
    elif ctx_tok and gen_tok:
        phase = "mixed"
        mixed_ctx_tokens += ctx_tok
        mixed_gen_tokens += gen_tok
    else:
        continue
    phase_us[phase] += e["dur"]
    phase_calls[phase] += 1
    phase_tokens[phase] += ctx_tok + gen_tok

if not spans:
    print("no gpu_user_annotation events found")
    sys.exit(0)

window_us = spans[-1]["ts"] + spans[-1]["dur"] - spans[0]["ts"]
tracked_us = sum(phase_us.values())

print(f"{'phase':<10} {'calls':>7} {'tokens':>9} {'gpu ms':>10} {'% window':>9}")
for phase in ("prefill", "decode", "mixed"):
    if phase_calls[phase]:
        print(
            f"{phase:<10} {phase_calls[phase]:>7} {phase_tokens[phase]:>9} "
            f"{phase_us[phase] / 1000:>10.1f} {100 * phase_us[phase] / window_us:>8.1f}%"
        )
idle_us = window_us - tracked_us
print(
    f"{'idle/gap':<10} {'':>7} {'':>9} {idle_us / 1000:>10.1f} {100 * idle_us / window_us:>8.1f}%"
)

if phase_calls["mixed"]:
    total = mixed_ctx_tokens + mixed_gen_tokens
    print(
        f"\nmixed steps interleave prefill+decode in one forward call and can't be "
        f"split from GPU time alone; by token share within those calls, "
        f"~{100 * mixed_ctx_tokens / total:.0f}% of mixed tokens were prefill, "
        f"~{100 * mixed_gen_tokens / total:.0f}% decode"
    )
