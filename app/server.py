"""
Simple HTTP Server for AI Study Assistant UI
Path: app/server.py

Zero-dependency local server using Python's built-in http.server.
Run with:
    python app/server.py [port]
"""

import sys
import os
import http.server
import socketserver
from pathlib import Path

DEFAULT_PORT = 3000
APP_DIR = Path(__file__).resolve().parent


class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    def end_headers(self):
        # Enable CORS for local API access
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Request-ID")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()


def run(port: int = DEFAULT_PORT):
    with socketserver.TCPServer(("", port), CustomHandler) as httpd:
        print("=" * 60)
        print(f"AI Study Assistant UI running at: http://localhost:{port}")
        print(f"Serving static components from:   {APP_DIR}")
        print(f"Backend API expected at:          http://127.0.0.1:8000")
        print("=" * 60)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down UI server.")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    run(port)

