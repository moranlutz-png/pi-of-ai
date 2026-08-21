"""
Static server for the scratch_coder observability page.

A copy of rules_baker/web/serve.py with the COOP/COEP headers removed: those exist
to enable SharedArrayBuffer -> WASM threads for wllama, and this page has no
wllama — it runs a tiny GPT in plain JS. (Said here so the next person doesn't
copy them back.)

Port 8125, not 8123: 8123 is the rules_baker page. Reusing it means one page
shadows the other and you verify the wrong checkout.

    python serve.py [port]        # default 8125
"""
from __future__ import annotations

import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[serve] " + (fmt % args) + "\n")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8125
    print(f"[serve] scratch_coder observability page on http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
