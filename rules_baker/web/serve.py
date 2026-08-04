"""Tiny static server for the Pi-of-AI runtime page.

Adds the two headers that let WebAssembly use threads (multi-thread wllama is
faster). On a static host that can't set headers (e.g. GitHub Pages), drop in
`coi-serviceworker.js` instead — the page still works single-threaded without it.

    py web/serve.py [port]        # default 8123
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
        # Cross-origin isolation -> enables SharedArrayBuffer -> WASM threads.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[serve] " + (fmt % args) + "\n")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
    print(f"[serve] Pi-of-AI runtime page on http://localhost:{port} (COOP/COEP on)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
