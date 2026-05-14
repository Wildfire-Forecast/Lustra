"""Serve fire_map.html with a Referrer-Policy header for OSM tile compatibility."""

from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ReferrerPolicyHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Referrer-Policy", "origin")
        super().end_headers()


def main() -> None:
    root = Path(__file__).resolve().parent
    server = ThreadingHTTPServer(("127.0.0.1", 8001), ReferrerPolicyHandler)
    print(f"Serving {root} at http://127.0.0.1:8001")
    print("Open http://127.0.0.1:8001/fire_map.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
