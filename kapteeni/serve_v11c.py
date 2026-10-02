"""The kapteeni-v1.1c server: the multimodal System One endpoint plus the
local demo website, one process, one origin.

    # trained model (the ship configuration)
    HF_HUB_OFFLINE=1 python3 -m kapteeni.serve_v11c --port 8002

    # mock model (no GPU; interface + demo dry-run)
    python3 -m kapteeni.serve_v11c --mock --port 8002

Routes: POST /v1/systemone (the wire contract, extended with
state.image), GET /v1/models, GET / (the demo website) and
GET /demo/* (its static assets). Same-origin by design so the demo page
needs no CORS. Error contract mirrors kapteeni.serve: 401 (bad key),
422 (validation). KAPTEENI_API_KEY optionally requires a bearer token.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from kapteeni.contract import ContractError, validate_request
from kapteeni.mock import MockV11C

DEFAULT_BASE = "model_cache/qwen3.5-4b"
DEFAULT_OUT = "model_cache/kapteeni_v11c"
DEMO_DIR = Path(__file__).resolve().parent / "demo"
ACCEPTED_MODELS = {"jev-latest", "kapteeni-v1.1c"}

MODEL_CARD = {
    "name": "kapteeni-v1.1c",
    "description": "The multimodal System One model: one Qwen3.5-4B "
    "backbone + LoRA adapter + per-primitive readout heads answering "
    "noul/choice/score questions over states with attached images "
    "(state.image, base64) in English and Chinese. All six pre-registered "
    "gates passed 2026-10-02 (docs/PREREG-KAPTEENI-V11C.md): images "
    "0.966 / Chinese-images 0.941, MNLI 0.88, OCNLI 0.847, the synth2 "
    "rule-skill mastery bars 0.913 (zh) / 0.905 (EN), fitted ECE "
    "0.024/0.018/0.069.",
    "release_date": "2026-10-02",
}


def make_handler(model, api_key: str | None, served_as: str):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, body: dict):
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _file(self, path: Path):
            if not path.is_file():
                self._json(404, {"error": {"message": f"no file {path.name}"}})
                return
            data = path.read_bytes()
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authed(self) -> bool:
            if not api_key:
                return True
            header = self.headers.get("Authorization", "")
            return header == f"Bearer {api_key}"

        def log_message(self, fmt, *args):  # quiet
            pass

        def do_GET(self):
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/v1/models":
                if not self._authed():
                    self._json(401, {"error": {"message": "missing or invalid API key"}})
                    return
                self._json(200, {"models": [
                    {"name": "jev-latest",
                     "description": f"Alias for {served_as}.",
                     "release_date": MODEL_CARD["release_date"]},
                    MODEL_CARD]})
                return
            if path == "/":
                self._file(DEMO_DIR / "index.html")
                return
            if path.startswith("/demo/"):
                rel = path[len("/demo/"):]
                base = DEMO_DIR.resolve()
                target = (base / rel).resolve()
                if not str(target).startswith(str(base)):
                    self._json(403, {"error": {"message": "forbidden"}})
                    return
                self._file(target)
                return
            self._json(404, {"error": {"message": f"no route for {self.path}"}})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                self._json(404, {"error": {"message": f"no route for {self.path}"}})
                return
            if not self._authed():
                self._json(401, {"error": {"message": "missing or invalid API key"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"null")
            except (ValueError, json.JSONDecodeError):
                self._json(422, {"error": {"message": "request body is not valid JSON"}})
                return
            try:
                state, req_model, questions = validate_request(body)
            except ContractError as e:
                self._json(e.status, e.body())
                return
            if req_model not in ACCEPTED_MODELS:
                self._json(422, {"error": {
                    "message": f"unknown model {req_model!r}", "field": "model"}})
                return
            try:
                with lock:  # one GPU; serialize inference
                    answers, usage = model.evaluate(state, questions)
            except ContractError as e:
                self._json(e.status, e.body())
                return
            resp = {"model": served_as, "answers": answers, "usage": usage}
            # optional notice: attached images are standardized to the
            # validated 640x640 serving shape — tell the caller what
            # happened to their pixels when they sent something larger
            try:
                from kapteeni.model_v11c import extract_image
                _, img = extract_image(state)
                if img is not None and (img.width > 640 or img.height > 640):
                    resp["notice"] = (
                        f"state.image ({img.width}x{img.height}) was "
                        f"downscaled to a 640px longest edge before the "
                        f"model saw it: small text, dense tables, and "
                        f"fine detail can become unreadable at that "
                        f"size, and accuracy may differ from what the "
                        f"full-resolution image would give — the model "
                        f"and all six of its gates were trained and "
                        f"validated at 640px. Send images no larger than "
                        f"640px on the long edge when precision matters.")
            except Exception:
                pass  # the notice is advisory; never fail a request on it
            self._json(200, resp)

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--mock", action="store_true",
                    help="serve MockV11C (no GPU; interface + demo dry-run)")
    ap.add_argument("--warmup", action="store_true", default=True,
                    help="pre-run one forward per demo image size at "
                         "startup (amortizes the box's per-shape GDN "
                         "searches; on by default)")
    ap.add_argument("--no-warmup", dest="warmup", action="store_false")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--heads", default="")
    ap.add_argument("--gates", default="",
                    help="final_gates.json (the fitted temperatures)")
    args = ap.parse_args(argv)

    adapter = args.adapter or f"{DEFAULT_OUT}/adapter"
    heads = args.heads or f"{DEFAULT_OUT}/heads.pt"
    gates = args.gates or f"{DEFAULT_OUT}/final_gates.json"
    if args.mock:
        print("serving MockV11C (pass --adapter/--heads for the trained "
              "model, or omit --mock with defaults)", flush=True)
        model = MockV11C()
    else:
        for p in (adapter, heads, gates):
            if not Path(p).exists():
                print(f"error: {p} not found (train v1.1c first, or pass "
                      f"--mock for the interface demo)", flush=True)
                return 2
        temps = json.loads(Path(gates).read_text())["fit_temps"]
        print(f"loading kapteeni-v1.1c (base {args.base}, fitted temps "
              f"{temps}) ...", flush=True)
        from kapteeni.model_v11c import SystemOneV11C
        model = SystemOneV11C(args.base, adapter, heads, temps)
        if args.warmup:
            sizes = [(640, 640)]
            demo_imgs = sorted((DEMO_DIR / "images").glob("*")) \
                if DEMO_DIR.exists() else []
            from PIL import Image
            if demo_imgs:
                sizes += [Image.open(p).size for p in demo_imgs
                          if p.suffix.lower()
                          in (".png", ".jpg", ".jpeg", ".webp")]
            print(f"warmup: pre-running forwards for {len(sizes)} image "
                  f"size(s) (first-touch GDN shape searches amortize "
                  f"into startup) ...", flush=True)
            t0 = time.time()
            model.warmup(sizes)
            print(f"warmup done in {time.time() - t0:.0f}s", flush=True)
    api_key = os.environ.get("KAPTEENI_API_KEY") or None
    httpd = ThreadingHTTPServer((args.host, args.port),
                                make_handler(model, api_key, "kapteeni-v1.1c"))
    print(f"listening on http://{args.host}:{args.port} — API at "
          f"/v1/systemone, demo website at /", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())