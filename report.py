import glob, os, subprocess, sys

if len(sys.argv) < 2:
    print("usage: report.py <label>", file=sys.stderr)
    sys.exit(1)

label = sys.argv[1]
root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(root, "profile"))
from _lib import resolve_trace


def section(title, cmd):
    print(f"\n{'=' * 20} {title} {'=' * 20}", flush=True)
    subprocess.run(cmd, check=False)


has_bench_results = glob.glob(os.path.join(root, "bench", "results", f"{label}*.json")) or glob.glob(
    os.path.join(root, "profile", "results", label, "*.json")
)
if has_bench_results:
    section("bench", [sys.executable, os.path.join(root, "bench", "summarize.py"), label])
else:
    print(f"(no bench/results/{label}*.json or profile/results/{label}/*.json - skipping bench summary)", flush=True)

trace_dir = os.path.join(root, "profile", "results", label)
try:
    trace_path = resolve_trace(trace_dir)
except FileNotFoundError:
    trace_path = None

if trace_path:
    section("top kernels", [sys.executable, os.path.join(root, "profile", "analyze.py"), trace_path])
    section("phases (prefill/decode)", [sys.executable, os.path.join(root, "profile", "phases.py"), trace_path])
    section("components (attention/mlp/norm/quant)", [sys.executable, os.path.join(root, "profile", "components.py"), trace_path])
else:
    print(f"(no *.pt.trace.json.gz under profile/results/{label}/ - skipping trace analysis)", flush=True)
