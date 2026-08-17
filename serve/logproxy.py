"""Logging reverse proxy for inspecting what a client actually sends.

Sits in front of llama-server and prints each request body, so you can see
exactly which parameters a client (opencode, etc.) emits — rather than
guessing from its docs.

    python3 serve/logproxy.py            # listen 8081 -> forward 8080
    LP_LISTEN=9000 LP_TARGET=8080 python3 serve/logproxy.py

Point the client's baseURL at the proxy, send one message, read the output.
Streaming responses are passed through unbuffered so the client still works
normally while you watch.
"""

import http.server
import json
import os
import sys
import urllib.request

LISTEN = int(os.environ.get("LP_LISTEN", 8081))
TARGET = os.environ.get("LP_TARGET", "127.0.0.1:8080")
INTERESTING = ("chat_template_kwargs", "reasoning", "reasoning_effort", "thinking")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet the default per-request noise
        pass

    def _proxy(self, method):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or None

        print(f"\n=== {method} {self.path} ===", flush=True)
        if body:
            try:
                parsed = json.loads(body)
                # messages are long and rarely what you're inspecting
                trimmed = {k: v for k, v in parsed.items() if k != "messages"}
                msgs = parsed.get("messages")
                if msgs:
                    trimmed["messages"] = f"<{len(msgs)} messages omitted>"
                print(json.dumps(trimmed, indent=2)[:4000], flush=True)
                hits = [k for k in INTERESTING if k in parsed]
                print(f"--> params of interest present: {hits or 'NONE'}", flush=True)
            except Exception:
                print(body[:1000], flush=True)

        req = urllib.request.Request(
            f"http://{TARGET}{self.path}",
            data=body,
            method=method,
            headers={k: v for k, v in self.headers.items() if k.lower() != "host"},
        )
        try:
            with urllib.request.urlopen(req) as up:
                self.send_response(up.status)
                for k, v in up.headers.items():
                    if k.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(k, v)
                self.end_headers()
                while chunk := up.read(8192):
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            payload = e.read()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            print(f"proxy error: {e}", file=sys.stderr, flush=True)
            self.send_response(502)
            self.end_headers()

    def do_POST(self):
        self._proxy("POST")

    def do_GET(self):
        self._proxy("GET")


if __name__ == "__main__":
    print(f"logproxy: :{LISTEN} -> {TARGET}", flush=True)
    http.server.ThreadingHTTPServer(("0.0.0.0", LISTEN), Handler).serve_forever()
