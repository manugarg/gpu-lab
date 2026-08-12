import glob, os, subprocess, sys

if len(sys.argv) < 2:
    print("usage: report.py <label>", file=sys.stderr)
    sys.exit(1)

label = sys.argv[1]
root = os.path.dirname(os.path.abspath(__file__))


def section(title, cmd):
    print(f"\n{'=' * 20} {title} {'=' * 20}", flush=True)
    subprocess.run(cmd, check=False)


if glob.glob(os.path.join(root, "bench", "results", f"{label}*.json")):
    section("bench", [sys.executable, os.path.join(root, "bench", "summarize.py"), label])
else:
    print(f"(no bench/results/{label}*.json - skipping bench summary)", flush=True)

trace_path = os.path.join(root, "profile", "results", label, "trace.pt.trace.json.gz")
if os.path.exists(trace_path):
    section("top kernels", [sys.executable, os.path.join(root, "profile", "analyze.py"), trace_path])
    section("phases (prefill/decode)", [sys.executable, os.path.join(root, "profile", "phases.py"), trace_path])
    section("components (attention/mlp/norm/quant)", [sys.executable, os.path.join(root, "profile", "components.py"), trace_path])
else:
    print(f"(no profile/results/{label}/trace.pt.trace.json.gz - skipping trace analysis)", flush=True)
