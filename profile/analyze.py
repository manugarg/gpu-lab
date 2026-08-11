import gzip, json, collections, sys

path = sys.argv[1] if len(sys.argv) > 1 else "trace.pt.trace.json.gz"
ev = json.load(gzip.open(path))["traceEvents"]
agg = collections.Counter()
for e in ev:
    if e.get("cat") == "kernel":
        agg[e["name"]] += e.get("dur", 0)
total = sum(agg.values())
for name, us in agg.most_common(25):
    print(f"{us/1000:9.1f} ms  {100*us/total:5.1f}%  {name[:80]}")
