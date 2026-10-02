"""Feedback store tests: schema validation, append/review/prune/export
roundtrips, redaction, and the /v1/feedback endpoint's happy/sad
paths through the mock server."""

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from kapteeni.feedback import (DEFAULT_RETENTION_DAYS, FeedbackError,
                               FeedbackStore, hash_state, new_record,
                               redact, validate)


def sample(served_as="kapteeni-v1.1c", **over):
    rec = {
        "kind": "correction",
        "served_as": served_as,
        "question": {"type": "choice",
                     "instructions": "Which plan fits?",
                     "options": ["basic", "pro"]},
        "state": {"plan": "pro", "region": "eu"},
        "feedback": {"answer": 1},
        "prediction": {"answer": 1, "p": 0.51},
        "provenance": {"client": "demo-user-42", "channel": "demo"},
    }
    rec.update(over)
    return rec


class TestSchema:
    def test_new_record_roundtrip(self):
        rec = new_record(served_as="kapteeni-v1.1c", kind="correction",
                         question=sample()["question"],
                         state={"a": 1}, feedback={"answer": 0})
        validate(rec)
        assert rec["id"].startswith("fb-")
        assert rec["state_sha256"] == hash_state({"a": 1})
        assert rec["license"] == "CC BY-SA-4.0"
        assert rec["review"]["status"] == "pending"

    def test_kind_must_be_known(self):
        with pytest.raises(FeedbackError, match="kind"):
            new_record(served_as="m", kind="vibes",
                       question=sample()["question"], state=None,
                       feedback={"answer": 1})

    def test_choice_needs_options(self):
        q = {"type": "choice", "instructions": "pick"}
        with pytest.raises(FeedbackError, match="options"):
            new_record(served_as="m", kind="confirmation", question=q,
                       state=None, feedback={"answer": 1})

    def test_state_hash_required_even_when_redacted(self):
        rec = new_record(served_as="m", kind="outcome",
                         question=sample()["question"], state=None,
                         feedback={"answer": 1})
        validate(rec)
        assert rec["state"] is None and len(rec["state_sha256"]) == 64


class TestStore:
    def test_add_review_export_prune(self, tmp_path):
        st = FeedbackStore(tmp_path / "fb.jsonl")
        r1 = st.add(new_record(served_as="m", kind="correction",
                               question=sample()["question"],
                               state={"s": 1}, feedback={"answer": 1}))
        r2 = st.add(new_record(served_as="m", kind="confirmation",
                               question=sample()["question"] | {"type": "noul"},
                               state={"s": 2}, feedback={"answer": 1}))
        assert st.stats()["total"] == 2
        with pytest.raises(FeedbackError, match="already exists"):
            st.add(dict(r1))  # ids are unique; re-adding is refused
        st.review(r1["id"], "confirmed", reviewer="lucy")
        n = st.export_training(tmp_path / "export.jsonl")
        assert n == 1
        row = json.loads((tmp_path / "export.jsonl").read_text().splitlines()[0])
        assert row["row_id"] == f"feedback-{r1['id']}"
        assert row["label"] == 1
        # prune drops the stale unconfirmed record, keeps the confirmed
        rows = list(st)
        for r in rows:
            if r["id"] == r2["id"]:  # stale-date the UNCONFIRMED one
                r["ts"] = time.time() - (DEFAULT_RETENTION_DAYS + 10) * 86400
        st._rewrite(rows)
        pruned = st.prune(DEFAULT_RETENTION_DAYS)
        assert pruned == 1
        assert [r["id"] for r in st] == [r1["id"]]  # confirmed survives

    def test_review_rejects_unknown_status(self, tmp_path):
        st = FeedbackStore(tmp_path / "fb.jsonl")
        r = st.add(new_record(served_as="m", kind="rejection",
                              question=sample()["question"], state=None,
                              feedback={"answer": 0}))
        with pytest.raises(FeedbackError, match="status"):
            st.review(r["id"], "maybe")
        with pytest.raises(FeedbackError, match="no record"):
            st.review("fb-nope", "confirmed")

    def test_redact_keeps_hash_and_marks_privacy(self):
        rec = new_record(served_as="m", kind="correction",
                         question=sample()["question"],
                         state={"secret": "x", "plan": "pro"},
                         feedback={"answer": 1})
        out = redact(rec, ("secret",))
        assert out["state"]["secret"] == "<redacted>"
        assert out["state"]["plan"] == "pro"
        assert out["privacy"]["redacted"] is True
        assert out["state_sha256"] == rec["state_sha256"]


class TestEndpoint:
    """POST /v1/feedback through the real handler with the mock model."""

    def _serve(self, tmp_path, body):
        from http.server import ThreadingHTTPServer

        from kapteeni.mock import MockV11C
        from kapteeni.serve_v11c import make_handler
        store = FeedbackStore(tmp_path / "fb.jsonl")
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(MockV11C(), None, "kapteeni-v1.1c", store))
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/feedback",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            try:
                resp = urllib.request.urlopen(req, timeout=10)
                return resp.status, json.loads(resp.read()), store
            except urllib.error.HTTPError as e:
                out = json.loads(e.read())
                return e.code, out, store
        finally:
            httpd.shutdown()

    def test_endpoint_accepts_and_stores(self, tmp_path):
        body = sample()
        status, out, store = self._serve(tmp_path, body)
        assert status == 200 and out["review"] == "pending"
        rec = next(iter(store))
        assert rec["served_as"] == "kapteeni-v1.1c"
        assert rec["feedback"]["answer"] == 1

    def test_endpoint_rejects_bad_kind(self, tmp_path):
        status, out, _ = self._serve(tmp_path, sample(kind="vibes"))
        assert status == 422 and "kind" in out["error"]["message"]

    def test_endpoint_404_when_not_enabled(self, tmp_path):
        from http.server import ThreadingHTTPServer

        from kapteeni.mock import MockV11C
        from kapteeni.serve_v11c import make_handler
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(MockV11C(), None, "kapteeni-v1.1c", None))
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/feedback",
                data=json.dumps(sample()).encode(),
                headers={"Content-Type": "application/json"})
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(req, timeout=10)
            assert e.value.code == 404
        finally:
            httpd.shutdown()