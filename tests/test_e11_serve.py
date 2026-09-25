"""E11 end-to-end — the server speaks the reference wire format.

Boots the stdlib server (mock model by default; KAPTEENI_TEST_BUNDLE upgrades to
the trained model) and drives it exactly like the docs' curl examples.
"""

import json
import os
import socket
import threading
import urllib.request

import pytest

from kapteeni.serve import make_handler
from http.server import ThreadingHTTPServer


@pytest.fixture(scope="module")
def server():
    model = pytest.importorskip("kapteeni.mock").MockModel()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(model, None, "kapteeni-v1"))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


def _post(base, body: dict, headers: dict | None = None):
    req = urllib.request.Request(
        base + "/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(base, path):
    try:
        with urllib.request.urlopen(base + path) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_docs_example_request_and_response(server):
    status, body = _post(server, {
        "state": "Help! My payouts have been failing for 3 days.",
        "model": "jev-latest",
        "questions": {
            "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}
        },
    })
    assert status == 200
    assert set(body) == {"model", "answers", "usage"}
    assert set(body["answers"]["is_urgent"]) == {"type", "noul"}
    assert 0.0 <= body["answers"]["is_urgent"]["noul"] <= 1.0
    assert set(body["usage"]) == {"input_tokens", "output_tokens"}


def test_docs_three_primitive_example(server):
    status, body = _post(server, {
        "state": "Our API integration started returning 500 errors on every request 20 minutes ago.",
        "model": "jev-latest",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this",
                "criteria": {
                    "billing": "Payment or subscription issues",
                    "technical": "Bugs or integration problems",
                    "sales": "Pricing or account questions",
                },
            },
            "is_urgent": {"type": "noul", "instructions": "The message conveys urgency"},
            "frustration": {
                "type": "score",
                "instructions": "How frustrated the customer appears",
                "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry"],
            },
        },
    })
    assert status == 200
    d = body["answers"]["department"]
    assert d["choice"] in ("billing", "technical", "sales")
    assert sum(d["probabilities"].values()) == 1.0
    f = body["answers"]["frustration"]
    assert 0.0 <= f["score"] <= 2.0
    assert f["legend"] == {"0": "Calm, just stating facts",
                           "1": "Frustrated but civil", "2": "Very angry"}


def test_get_models(server):
    status, body = _get(server, "/v1/models")
    assert status == 200
    cards = body["models"]
    names = {c["name"] for c in cards}
    assert "jev-latest" in names
    card = cards[0]
    assert set(card) == {"name", "description", "release_date"}


def test_422_on_malformed_question(server):
    status, body = _post(server, {
        "state": "s", "model": "jev-latest",
        "questions": {"bad": {"type": "nope", "instructions": "x"}},
    })
    assert status == 422
    assert body["error"]["field"] == "questions.bad.type"


def test_422_on_bad_json(server):
    req = urllib.request.Request(
        server + "/v1/systemone", data=b"{not json",
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        urllib.request.urlopen(req)
        assert False, "should raise"
    except urllib.error.HTTPError as e:
        assert e.code == 422


def test_404_unknown_route(server):
    status, _ = _get(server, "/v1/nothing")
    assert status == 404


def test_401_without_key(server, monkeypatch):
    # server fixture runs without a key; spin one WITH a key
    from kapteeni.mock import MockModel
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(MockModel(), "sekrit", "kapteeni-v1-meticulous"))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        status, body = _post(f"http://127.0.0.1:{port}",
                             {"state": "s", "model": "jev-latest",
                              "questions": {"q": {"type": "noul", "instructions": "?"}}},
                             headers={"Authorization": "Bearer wrong"})
        assert status == 401
        status, _ = _post(f"http://127.0.0.1:{port}",
                          {"state": "s", "model": "jev-latest",
                           "questions": {"q": {"type": "noul", "instructions": "?"}}},
                          headers={"Authorization": "Bearer sekrit"})
        assert status == 200
    finally:
        httpd.shutdown()