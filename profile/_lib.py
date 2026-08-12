import glob, gzip, json, os

# vLLM writes one *.pt.trace.json.gz per process when profiling (the GPU
# worker plus, separately, the frontend/AsyncLLM process) - only the worker's
# has kernel/cpu_op events, so pick by content rather than by name.


def resolve_trace(path):
    if os.path.isfile(path):
        return path
    candidates = sorted(glob.glob(os.path.join(path, "*.pt.trace.json.gz")))
    if not candidates:
        raise FileNotFoundError(f"no *.pt.trace.json.gz found in {path}")
    if len(candidates) == 1:
        return candidates[0]
    for c in candidates:
        with gzip.open(c) as f:
            cats = {e.get("cat") for e in json.load(f)["traceEvents"]}
        if "kernel" in cats:
            return c
    return candidates[0]


def load_trace(path):
    return json.load(gzip.open(resolve_trace(path)))["traceEvents"]
