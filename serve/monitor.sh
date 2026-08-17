#!/usr/bin/env bash
# Live view of llama-server throughput and GPU state. Read-only: HTTP GETs
# and nvidia-smi, nothing that touches the running server.
#
#   ./serve/monitor.sh                      # auto-detect the server log
#   LLAMA_LOG=/path/to/server.log ./serve/monitor.sh
#   LLAMA_PORT=8080 INTERVAL=2 ./serve/monitor.sh
#
# The richest numbers (prefill rate, decode tg, draft acceptance) come from
# llama-server's stdout, so they only appear if that stdout was captured to
# a file. Under systemd, point LLAMA_LOG at a journalctl dump or run:
#   journalctl --user -u llama -f > /tmp/llama.log &
set -uo pipefail

PORT="${LLAMA_PORT:-8080}"
INTERVAL="${INTERVAL:-2}"

# Newest server log, unless told otherwise. Uses find rather than a `**`
# glob: globstar is off by default in bash, so `**` silently matches
# nothing and the log panels quietly disappear.
if [ -z "${LLAMA_LOG:-}" ]; then
  LLAMA_LOG=$(find /tmp/claude-* "${TMPDIR:-/tmp}" -maxdepth 6 -name 'serve*.log' -type f \
                -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2- || true)
fi

hr() { printf '%.0s─' $(seq 1 "${COLUMNS:-72}"); echo; }

while true; do
  # only repaint when attached to a terminal, so piping/logging stays readable
  [ -t 1 ] && clear
  echo "llama-server monitor  ·  :$PORT  ·  $(date '+%H:%M:%S')"
  hr

  if curl -sf --max-time 3 "localhost:$PORT/health" >/dev/null 2>&1; then
    echo "server:  UP"
  else
    echo "server:  DOWN (or still loading)"
  fi

  # --- slots: is_processing is the authoritative "is it busy" signal ---
  curl -s --max-time 3 "localhost:$PORT/slots" 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
except Exception:
    print('slots:   (unavailable)'); sys.exit()
busy=[s for s in d if s.get('is_processing')]
print(f\"slots:   {len(busy)}/{len(d)} processing\", end='')
print(f\"   n_ctx={d[0].get('n_ctx'):,}  speculative={d[0].get('speculative')}\" if d else '')
" 2>/dev/null || echo "slots:   (unavailable)"

  # --- GPU ---
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{
      printf "gpu:     %s%% util  %.1f/%.1f GiB  %sW  %s°C  throttle=%s\n", $1, $2/1024, $3/1024, $4, $5, ($6=="0x0000000000000000"?"none":$6)
    }' || echo "gpu:     (nvidia-smi unavailable)"

  hr

  if [ -n "${LLAMA_LOG:-}" ] && [ -r "$LLAMA_LOG" ]; then
    echo "log:     $LLAMA_LOG"
    python3 - "$LLAMA_LOG" <<'PY'
import re, sys
try:
    txt = open(sys.argv[1], errors='ignore').read()[-400000:]
except Exception as e:
    print(f"         (unreadable: {e})"); raise SystemExit

def last(pat):
    m = re.findall(pat, txt)
    return m[-1] if m else None

pf = last(r'prompt processing, n_tokens = *(\d+), progress = ([\d.]+), t = *([\d.]+) s / *([\d.]+) tokens')
if pf:
    n, prog, t, rate = int(pf[0]), float(pf[1]), float(pf[2]), float(pf[3])
    total = n/prog if prog else 0
    rem = max(total-n, 0)
    eta = rem/rate if rate else 0
    bar = int(prog*30)
    print(f"prefill: [{'#'*bar}{'.'*(30-bar)}] {prog:5.0%}  {n:,}/{total:,.0f} tok")
    print(f"         {rate:.0f} tok/s   elapsed {t:.0f}s   eta ~{eta:.0f}s")

tg = last(r'n_gen = *(\d+), tg = *([\d.]+) t/s, tg_3s = *([\d.]+) t/s')
if tg:
    print(f"decode:  {float(tg[1]):.1f} tok/s overall   {float(tg[2]):.1f} tok/s last-3s   ({int(tg[0]):,} generated)")

da = last(r'draft acceptance = ([\d.]+) \( *(\d+) accepted / *(\d+) generated\), mean len = *([\d.]+)')
if da:
    print(f"mtp:     {float(da[0]):.0%} accepted  ({da[1]}/{da[2]})  mean draft len {da[3]}")
    if float(da[0]) < 0.5:
        print("         ^ low acceptance: MTP may be costing more than it saves")

ev = re.findall(r'prompt eval time = *([\d.]+) ms / *(\d+) tokens.*?\n.*?eval time = *([\d.]+) ms / *(\d+) tokens', txt)
if ev:
    pms, ptok, ems, etok = (float(x) for x in ev[-1])
    print(f"last req: prefill {ptok:,.0f} tok in {pms/1000:.1f}s ({ptok/(pms/1000):.0f} tok/s)"
          f" | decode {etok:,.0f} tok in {ems/1000:.1f}s ({etok/(ems/1000):.0f} tok/s)")
PY
  else
    echo "log:     not found — set LLAMA_LOG=/path/to/server.log"
    echo "         (prefill / decode / MTP panels need llama-server's stdout)"
  fi

  hr
  echo "ctrl-c to exit · refresh ${INTERVAL}s"
  sleep "$INTERVAL"
done
