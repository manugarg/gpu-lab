import collections, sys
from _lib import load_trace

path = sys.argv[1] if len(sys.argv) > 1 else "."
ev = load_trace(path)
agg = collections.Counter()
for e in ev:
    if e.get("cat") == "kernel":
        agg[e["name"]] += e.get("dur", 0)
total = sum(agg.values())
for name, us in agg.most_common(25):
    print(f"{us/1000:9.1f} ms  {100*us/total:5.1f}%  {name[:80]}")
