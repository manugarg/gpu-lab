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
has_trace = False
try:
    resolve_trace(trace_dir)
    has_trace = True
except FileNotFoundError:
    pass

if has_trace:
    # pass the directory, not a pre-resolved file: components.py reads
    # meta.txt from it (for MODEL provenance), and _lib.load_trace resolves
    # the actual trace file from a directory just as well as from a path.
    section("top kernels", [sys.executable, os.path.join(root, "profile", "analyze.py"), trace_dir])
    section("phases (prefill/decode)", [sys.executable, os.path.join(root, "profile", "phases.py"), trace_dir])
    section("components (attention/mlp/norm/quant)", [sys.executable, os.path.join(root, "profile", "components.py"), trace_dir])
else:
    print(f"(no *.pt.trace.json.gz under profile/results/{label}/ - skipping trace analysis)", flush=True)
