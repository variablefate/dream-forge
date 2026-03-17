"""C.8.1 Local proxy runtime — OpenAI-compatible API with best-of-N.

Loads the 8-bit model with detector probe and exposes an OpenAI-compatible
chat completions endpoint at localhost:8081.

Supports best-of-N via the 'n' parameter:
  n=1: fast pass@1 (same as GGUF/LM Studio)
  n=4: detector-scored selection (reliability-sensitive)

Usage:
    uv run python -m src.runtime.server
    uv run python -m src.runtime.server --port 8081 --n 4

Client:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8081/v1", api_key="local")
    response = client.chat.completions.create(
        model="dream-forge",
        messages=[{"role": "user", "content": "What is Python's GIL?"}],
        extra_body={"n_samples": 4},  # best-of-N
    )
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

from rich.console import Console

console = Console()

DEFAULT_PORT = 8081
DEFAULT_N = 4
DETECTOR_PROBE_PATH = Path("models/detector_probe.pkl")


def create_app(model, tokenizer, bon, default_n: int = 1):
    """Create the HTTP app with the completions endpoint."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class CompletionHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path not in ("/v1/chat/completions", "/chat/completions"):
                self._send_error(404, "Not found")
                return

            # Parse body with size limit
            try:
                content_len = int(self.headers.get("Content-Length", 0))
            except (ValueError, TypeError):
                self._send_error(400, "Invalid Content-Length")
                return

            if content_len > 1_000_000:  # 1MB limit
                self._send_error(413, "Request too large")
                return

            try:
                body = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_error(400, "Invalid JSON body")
                return

            messages = body.get("messages", [])
            try:
                n_samples = max(1, min(int(body.get("n_samples", body.get("n", default_n))), 16))
            except (TypeError, ValueError):
                n_samples = default_n
            temperature = body.get("temperature", 0.7)
            if not isinstance(temperature, (int, float)) or temperature < 0:
                temperature = 0.7
            try:
                max_tokens = max(1, min(int(body.get("max_tokens", 512)), 2048))
            except (TypeError, ValueError):
                max_tokens = 512

            # Handle temperature=0 → greedy (OpenAI convention)
            if temperature == 0:
                temperature = 0.01  # near-greedy; best_of_n always sets do_sample=True

            # Extract the user query (last user message)
            query = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    query = msg.get("content", "")
                    break

            if not query:
                self._send_error(400, "No user message found")
                return

            try:
                t0 = time.monotonic()

                if n_samples <= 1:
                    result = bon.generate(
                        model, tokenizer, query,
                        n=1, temperature=temperature, max_new_tokens=max_tokens)
                else:
                    result = bon.generate(
                        model, tokenizer, query,
                        n=n_samples, temperature=temperature, max_new_tokens=max_tokens)

                elapsed = time.monotonic() - t0
            except Exception as e:
                self._send_error(500, f"Inference error: {str(e)[:200]}")
                return

            # OpenAI-compatible response format
            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "dream-forge",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.text,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 0,  # not tracked
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                # dream-forge extensions
                "dream_forge": {
                    "strategy": result.strategy,
                    "confidence": result.confidence,
                    "n_generated": result.n_generated,
                    "hedged": result.hedged,
                    "elapsed_seconds": elapsed,
                },
            }

            self._send_json(200, response)

        def do_GET(self):
            if self.path in ("/v1/models", "/models"):
                self._send_json(200, {
                    "object": "list",
                    "data": [{
                        "id": "dream-forge",
                        "object": "model",
                        "owned_by": "local",
                    }],
                })
            elif self.path in ("/health", "/"):
                self._send_json(200, {"status": "ok", "model": "dream-forge"})
            else:
                self._send_error(404, "Not found")

        def _send_json(self, code: int, data: dict):
            body = json.dumps(data).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, code: int, message: str):
            """Send OpenAI-compatible error response."""
            self._send_json(code, {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": code,
                }
            })

        def log_message(self, format, *args):
            # Suppress default logging, use Rich instead
            pass

    return HTTPServer, CompletionHandler


def main():
    parser = argparse.ArgumentParser(
        description="dream-forge local inference server (OpenAI-compatible)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Port (default: {DEFAULT_PORT})")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Default best-of-N samples (default: {DEFAULT_N})")
    parser.add_argument("--probe", type=Path, default=DETECTOR_PROBE_PATH,
                        help="Path to detector probe .pkl")
    args = parser.parse_args()

    # Load model
    console.print("[bold]dream-forge server[/bold]")
    console.print(f"  Port: {args.port} | Default N: {args.n}")

    from src.engine.model_loader import load_model
    from src.runtime.best_of_n import BestOfN

    console.print("\n  Loading model...", style="dim")
    model, tokenizer = load_model()

    console.print("  Loading detector probe...", style="dim")
    bon = BestOfN.from_pretrained(args.probe)

    # Create and run server
    HTTPServer, Handler = create_app(model, tokenizer, bon, default_n=args.n)
    server = HTTPServer(("127.0.0.1", args.port), Handler)

    console.print(f"\n[bold green]Server running at http://localhost:{args.port}[/bold green]")
    console.print(f"  POST /v1/chat/completions  (OpenAI-compatible)")
    console.print(f"  GET  /v1/models")
    console.print(f"  GET  /health")
    console.print(f"\n  Client example:")
    console.print(f'    client = OpenAI(base_url="http://localhost:{args.port}/v1", api_key="local")')
    console.print(f"\n  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
        server.shutdown()


if __name__ == "__main__":
    main()
