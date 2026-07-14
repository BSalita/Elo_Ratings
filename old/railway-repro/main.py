"""Minimal Railway service to reproduce unexplained deployment removals.

Serves a single 'ok' endpoint. No volume, no config-as-code, no Streamlit -
if this deployment is also REMOVED by the hourly sweep, the cause is
account/platform-level, not anything in the Elo_Ratings services.
"""
import http.server
import os

PORT = int(os.environ.get("PORT", 8000))


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, fmt, *args):
        print("[repro]", fmt % args, flush=True)


print(f"[repro] starting on port {PORT}", flush=True)
http.server.HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
