import json, re, sys, os, glob, collections
from _lib import load_trace

path = sys.argv[1] if len(sys.argv) > 1 else "."
ev = load_trace(path)

NK_RE = re.compile(r"gemm\w*_impl<\d+u,\s*(\d+)u,\s*(\d+)u")


def model_proj_shapes():
    model = os.environ.get("MODEL", "Qwen/Qwen3-14B-FP8")
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cache_name = "models--" + model.replace("/", "--")
    hits = glob.glob(os.path.join(hf_home, "hub", cache_name, "snapshots", "*", "config.json"))
    if not hits:
        return None
    cfg = json.load(open(hits[0]))
    hidden = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg.get("num_key_value_heads", heads)
    head_dim = cfg.get("head_dim", hidden // heads)
    q, kv = heads * head_dim, kv_heads * head_dim
    return {
        "attention-proj (gemm)": [(q + 2 * kv, hidden), (hidden, q)],  # qkv_proj, o_proj
        "mlp (gemm)": [(2 * inter, hidden), (hidden, inter)],  # gate_up_proj, down_proj
        "lm_head (gemm)": [(cfg["vocab_size"], hidden)],
    }


SHAPES = model_proj_shapes()


def classify(name):
    n = name.lower()
    if "flashinfer" in n and ("attention" in n or "batchprefill" in n or "batchdecode" in n):
        return "attention (core)"
    if any(k in n for k in ("reshape_and_cache", "block_tables", "slot_mapping", "prefill_inputs", "page_indices")):
        return "attention (kv-cache/bookkeeping)"
    if any(k in n for k in ("sampling", "topk", "topp", "temperature_kernel", "sampled_and_rejected", "sampled_and_draft")):
        return "sampling"
    m = NK_RE.search(name)
    if m:
        dims = tuple(sorted(int(x) for x in m.groups()))
        if SHAPES:
            for label, shapes in SHAPES.items():
                if any(tuple(sorted(s)) == dims for s in shapes):
                    return label
        return f"gemm (unrecognized shape {dims})"
    if any(k in n for k in ("cutlass", "gemvx", "gemv")):
        return "gemm (other, unattributed)"
    if "silu" in n:
        return "mlp (activation+quant, fused)" if "quant" in n else "mlp (activation)"
    if "rms_norm" in n or "layernorm" in n:
        return "norm+quant (fused)" if "quant" in n else "norm"
    if "quant" in n:
        return "quant"
    if "nccl" in n or "all_reduce" in n:
        return "communication"
    return "other"


agg_us = collections.Counter()
agg_calls = collections.Counter()
last_gemm_cat = {}
kern = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: (e["pid"], e["tid"], e["ts"]))
for e in kern:
    stream = (e["pid"], e["tid"])
    if "split_k_reduce" in e["name"] and stream in last_gemm_cat:
        # deep_gemm's own reduction step for the GEMM it just ran on this stream;
        # carries no shape info of its own, so inherit the preceding kernel's category.
        cat = last_gemm_cat[stream]
    else:
        cat = classify(e["name"])
        if cat.endswith("(gemm)"):
            last_gemm_cat[stream] = cat
    agg_us[cat] += e["dur"]
    agg_calls[cat] += 1

total_us = sum(agg_us.values())
if SHAPES is None:
    print(f"warning: no cached config.json for MODEL={os.environ.get('MODEL', '(unset)')}; "
          f"attention-proj/mlp GEMMs will show as 'gemm (unrecognized shape ...)'\n")
else:
    print(f"GEMM shapes (N x K) used for classification, from MODEL="
          f"{os.environ.get('MODEL', 'Qwen/Qwen3-14B-FP8')}'s config.json:")
    for label, shapes in SHAPES.items():
        print(f"  {label:<24} " + ", ".join(f"{n}x{k}" for n, k in shapes))
    print()

print(f"{'component':<32} {'calls':>7} {'gpu ms':>10} {'%':>7}")
for cat, us in agg_us.most_common():
    print(f"{cat:<32} {agg_calls[cat]:>7} {us / 1000:>10.1f} {100 * us / total_us:>6.1f}%")
