#!/usr/bin/env bash
deadline=$(( $(date +%s) + ${TIMEOUT:-600} ))
until curl -sf localhost:8000/health >/dev/null; do
  [ "$(date +%s)" -gt "$deadline" ] && { echo "timed out waiting for :8000" >&2; exit 1; }
  [ -n "${1:-}" ] && { kill -0 "$1" 2>/dev/null || { echo "server died" >&2; exit 1; }; }
  sleep 5
done

