"""Ollama client helper (local daemon, cloud-sub teacher).

Lessons baked in from the reference build:
  - trust_env=False: a global HTTPS_PROXY blackholes localhost (verified
    2026-09-22); httpx must bypass the proxy entirely.
  - /api/chat with format:"json" + think:false works with deepseek-v4-pro:cloud
    (~0.6-1.5 s/call); glm-style cloud models ignore think:false — prefer
    deepseek for labels.
  - teachers intermittently wrap JSON in markdown fences despite format:"json";
    always parse loosely.
"""

from __future__ import annotations

import hashlib
import json
import re

import httpx

BASE = "http://localhost:11434"


def content_hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def parse_json_loose(text: str) -> dict | None:
    """Fence-tolerant JSON parse -> dict, else None."""
    if not text:
        return None
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", t, re.S)
    if m:
        t = m.group(1)
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        # last resort: first {...} span
        a, b = t.find("{"), t.rfind("}")
        if 0 <= a < b:
            try:
                v = json.loads(t[a : b + 1])
                return v if isinstance(v, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def ollama_chat(
    model: str,
    prompt: str,
    *,
    temperature: float = 0.0,
    format_json: bool = True,
    think: bool = False,
    base: str = BASE,
    timeout: float = 180.0,
    client: httpx.Client | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": think,
    }
    if format_json:
        payload["format"] = "json"
    if temperature:
        payload["options"] = {"temperature": temperature}
    own = client is None
    c = client or httpx.Client(trust_env=False, timeout=timeout)
    try:
        r = c.post(base + "/api/chat", json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"]
    finally:
        if own:
            c.close()