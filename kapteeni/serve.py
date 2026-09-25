"""HTTP layer: POST /v1/systemone and GET /v1/models (stdlib only).

    python3 -m kapteeni.serve --port 8000                       # mock model
    python3 -m kapteeni.serve --bundle model_cache/kapteeni_v0.pt  # trained model

Error contract mirrors the reference: 401 (bad/missing key), 422 (validation),
429/529 equivalents left to the gateway. Set KAPTEENI_API_KEY to require a bearer
token; without it the server accepts any/none (local dev).
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from kapteeni.backbone import DEFAULT_MODEL
from kapteeni.contract import ContractError, validate_request
from kapteeni.mock import MockModel
from kapteeni.model import SystemOneModel

from pathlib import Path

MODEL_CARDS = [
    {
        "name": "jev-latest",
        "description": "Alias for the latest kapteeni implementation model.",
        "release_date": "2026-09-26",
    },
]
VARIANT_INFO = {
    "kapteeni-v1": "(legacy name for kapteeni-v1-meticulous)",
    "kapteeni-v1-meticulous": "Conservative confidence: calibrated on "
        "messy, out-of-distribution inputs; the safe default for unknown "
        "traffic.",
    "kapteeni-v1-intuit": "Sharper decisions: strongest on well-formed "
        "numeric, temporal, and multi-step policy traffic.",
}
ACCEPTED_MODELS = {"jev-latest", "kapteeni-v1", "kapteeni-v1-meticulous",
                   "kapteeni-v1-intuit"}


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

        def _authed(self) -> bool:
            if not api_key:
                return True
            header = self.headers.get("Authorization", "")
            return header == f"Bearer {api_key}"

        def log_message(self, fmt, *args):  # quiet
            pass

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                if not self._authed():
                    self._json(401, {"error": {"message": "missing or invalid API key"}})
                    return
                cards = MODEL_CARDS + [{
                    "name": served_as,
                    "description": "From-scratch implementation of the "
                    "TypeSafe System One interface: frozen "
                    "Qwen3-4B-Instruct-2507 backbone + calibrated readout "
                    "heads. " + VARIANT_INFO.get(served_as, ""),
                    "release_date": "2026-09-26",
                }]
                self._json(200, {"models": cards})
            else:
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
                self._json(
                    422,
                    {"error": {"message": f"unknown model {req_model!r}", "field": "model"}},
                )
                return
            with lock:  # one GPU; serialize inference
                answers, usage = model.evaluate(state, questions)
            self._json(200, {"model": served_as, "answers": answers,
                             "usage": usage})

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--bundle", default="", help="trained head bundle; omit for mock")
    ap.add_argument("--readout", default="blend",
                    choices=["blend", "head", "verb"],
                    help="blend = head+verbalizer (Phase 1), head, or verb")
    ap.add_argument("--lora", default="",
                    help="optional LoRA adapter dir (merged at startup)")
    ap.add_argument("--dist", default="",
                    help="packaged distribution dir (merged model + heads + config)")
    ap.add_argument("--hf", default="",
                    help="Hugging Face repo id of a packaged distribution")
    ap.add_argument("--model", default="",
                    help="--bundle path: merged model dir (e.g. a soup); "
                         "default: the base backbone")
    ap.add_argument("--fit", default="",
                    help="blend constants json for --bundle/--lora serving; "
                         "default: the Phase-1 fit file")
    ap.add_argument("--served-as", default="",
                    help="name reported in responses and /v1/models; "
                         "default: the dist's own config, else kapteeni-v1 "
                         "(legacy name for -meticulous)")
    args = ap.parse_args(argv)

    served_as = args.served_as
    if args.hf:
        from huggingface_hub import snapshot_download

        args.dist = snapshot_download(args.hf)
    if args.dist:
        print(f"loading packaged distribution from {args.dist} ...", flush=True)
        cfg = json.loads((Path(args.dist) / "kapteeni-config.json")
                         .read_text()) if Path(args.dist,
                                              "kapteeni-config.json").exists() else {}
        served_as = served_as or cfg.get("served_as", "")
        model = SystemOneModel.from_dist(args.dist, readout=args.readout)
    elif args.bundle:
        print(f"loading trained model from {args.bundle} "
              f"(readout: {args.readout}) ...", flush=True)
        blend = (json.loads(Path(args.fit).read_text())
                 if args.fit else None)
        model = SystemOneModel(args.bundle, readout=args.readout,
                               lora=args.lora or None,
                               model_name=args.model or DEFAULT_MODEL,
                               blend=blend)
    else:
        print("serving MockModel (pass --bundle for the trained model)", flush=True)
        model = MockModel()
    served_as = served_as or "kapteeni-v1"
    if served_as not in (ACCEPTED_MODELS - {"jev-latest"}):
        print(f"error: unknown --served-as {served_as!r}", flush=True)
        return 2
    api_key = os.environ.get("KAPTEENI_API_KEY") or None
    httpd = ThreadingHTTPServer((args.host, args.port),
                                make_handler(model, api_key, served_as))
    print(f"listening on http://{args.host}:{args.port} "
          f"(served as {served_as})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())