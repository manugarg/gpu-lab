import glob, json, os, sys

# From CLAUDE.md's baseline: Qwen3-14B-FP8, 1024/256, rate 4.
BASELINE = {"output_throughput": 1011, "mean_tpot_ms": 15.9, "mean_ttft_ms": 110}
REGRESSION_PCT = 5


def load_results(label):
    root = os.path.dirname(os.path.abspath(__file__))
    paths = sorted(glob.glob(os.path.join(root, "results", f"{label}*.json")))
    return [(p, json.load(open(p))) for p in paths]


def fmt_delta(value, baseline_value, lower_is_better):
    pct = 100 * (value - baseline_value) / baseline_value
    worse = pct < 0 if not lower_is_better else pct > 0
    flag = "REGRESSION" if worse and abs(pct) > REGRESSION_PCT else "ok"
    return f"{pct:+.1f}% vs baseline [{flag}]"


def run(label):
    results = load_results(label)
    if not results:
        print(f"no bench/results/{label}*.json found")
        return
    for path, r in results:
        print(f"\n{os.path.basename(path)}")
        print(f"  label:              {r.get('label')}")
        print(f"  request_rate:       {r.get('request_rate')}  num_prompts: {r.get('num_prompts')}")
        print(f"  completed/failed:   {r.get('completed')}/{r.get('failed', 0)}")
        print(f"  output_throughput:  {r.get('output_throughput'):.1f} tok/s")
        print(f"  mean_ttft_ms:       {r.get('mean_ttft_ms'):.1f} ms")
        print(f"  mean_tpot_ms:       {r.get('mean_tpot_ms'):.1f} ms")
        if r.get("mean_itl_ms") is not None:
            print(f"  mean_itl_ms:        {r.get('mean_itl_ms'):.1f} ms")
        if "rate4" in r.get("label", ""):
            print(f"  output_throughput:  {fmt_delta(r['output_throughput'], BASELINE['output_throughput'], lower_is_better=False)}")
            print(f"  mean_tpot_ms:       {fmt_delta(r['mean_tpot_ms'], BASELINE['mean_tpot_ms'], lower_is_better=True)}")
            print(f"  mean_ttft_ms:       {fmt_delta(r['mean_ttft_ms'], BASELINE['mean_ttft_ms'], lower_is_better=True)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: summarize.py <label>", file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1])
