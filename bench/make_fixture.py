#!/usr/bin/env python3
"""Build a long-context benchmark fixture that speculative decoding can't cheat.

The old fixtures (dec-ctx50k.json and friends) were a wall of random words.
Asked to continue, the model echoed the prompt back verbatim, so the MTP
draft head hit ~100% acceptance and decode throughput measured nothing:
llama.cpp reported 147 tok/s at 224/225 drafts accepted while emitting a
copy of its own input. Both engines were inflated by it, silently.

A usable fixture needs two properties:

  1. Real content, so attention has something structured to do.
  2. A question whose answer is NOT present in the context, so the model
     must generate novel tokens and the draft head has to actually predict.

This builds one from real source files and sizes it against the server's
own tokenizer, so the token count is exact rather than estimated.

    ./bench/make_fixture.py                      # 50K tokens -> bench/fixtures/
    ./bench/make_fixture.py --tokens 16000 --out bench/fixtures/ctx16k.json

Check it worked by looking at draft acceptance: a good fixture lands well
under 100%. If you see 100%, the model is copying and the number is junk.
"""

import argparse
import json
import os
import pathlib
import sys
import urllib.request

# Real code and prose, in the shape an agent actually sees: this repo's own
# notes plus the CUDA backend we spend all our time reading.
SOURCES = [
    ("notes", "*.md"),
    ("serve", "*.sh"),
    ("profile", "*.py"),
]
EXTRA_DIRS = [
    (os.path.expanduser("~/tools/llama.cpp/ggml/src/ggml-cuda"), "*.cu"),
    (os.path.expanduser("~/tools/llama.cpp/ggml/src/ggml-cuda"), "*.cuh"),
]

# Answerable only by synthesising across the context, and not by quoting it.
QUESTION = """
Based only on the material above, write a short technical assessment in
prose. Do not quote, list, or reproduce any code or text from it.

Explain in your own words: which parts of this system would be hardest to
change safely, and why. Then name one risk a new contributor would likely
miss on their first read. Write in flowing paragraphs, not bullet points.
"""


def count_tokens(text, host, model):
    """Token count from whichever engine is listening.

    The two /tokenize endpoints disagree on both halves of the contract:
    vLLM takes {"model","prompt"} and returns {"count","tokens"};
    llama.cpp takes {"content"} and returns {"tokens"} with no count.
    Try the vLLM shape first, fall back to llama.cpp's.
    """
    for body in ({"model": model, "prompt": text}, {"content": text}):
        req = urllib.request.Request(
            f"http://{host}/tokenize",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
        except urllib.error.HTTPError:
            continue
        # llama.cpp answers 200 to a vLLM-shaped body but tokenizes the
        # absent "content" field, returning an empty list. A zero count for
        # non-empty text means wrong shape, not a short prompt - keep trying.
        n = d.get("count", len(d.get("tokens", [])))
        if n:
            return n
    raise RuntimeError(f"no usable /tokenize on {host}")


def gather(repo):
    out = []
    for sub, pat in SOURCES:
        for f in sorted((repo / sub).glob(pat)):
            out.append(f"===== {f.relative_to(repo)} =====\n{f.read_text(errors='replace')}")
    for d, pat in EXTRA_DIRS:
        p = pathlib.Path(d)
        if p.is_dir():
            for f in sorted(p.glob(pat)):
                out.append(f"===== {f.name} =====\n{f.read_text(errors='replace')}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=50000, help="target prompt tokens")
    ap.add_argument("--out", default="bench/fixtures/ctx50k.json")
    ap.add_argument("--host", default="127.0.0.1:8000", help="a server with /tokenize")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--max-tokens", type=int, default=300)
    args = ap.parse_args()

    repo = pathlib.Path(__file__).resolve().parent.parent
    chunks = gather(repo)
    if not chunks:
        sys.exit("no source files found - check SOURCES/EXTRA_DIRS")

    # Binary search on character count: tokenizing is the slow part, so do it
    # a handful of times rather than growing chunk by chunk.
    body = "\n\n".join(chunks)
    lo, hi = 0, len(body)
    best = ""
    for _ in range(12):
        mid = (lo + hi) // 2
        n = count_tokens(body[:mid] + QUESTION, args.host, args.model)
        if n <= args.tokens:
            best, lo = body[:mid], mid + 1
        else:
            hi = mid - 1
        if lo > hi:
            break

    content = best + QUESTION
    total = count_tokens(content, args.host, args.model)

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    out = repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    print(f"{out}: {total:,} prompt tokens, {len(content):,} chars, "
          f"max_tokens={args.max_tokens}")


if __name__ == "__main__":
    main()
