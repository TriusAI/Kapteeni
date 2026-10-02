"""User-feedback store for Kapteeni serving deployments.

Collects what a deployment learns about its model in the field —
corrections, confirmations, downstream outcomes — as append-only JSONL
with provenance, privacy, and review metadata, so a future pre-registered
retrain can consume vetted feedback as ordinary training rows (see
docs/FEEDBACK.md for the full policy). Nothing here touches training
today: the store's job is capture, hygiene, and audit-readiness.

Schema (one JSON object per line):

    id               fb-<hex16>          server-assigned, unique
    ts               unix seconds        server-assigned at append time
    kind             correction | confirmation | outcome | rejection
    served_as        e.g. "kapteeni-v1.1c"   which model answered
    served_revision  pinned revision hash | null (dist mode may not know)
    request_id       the served request's id when echoable, else null
    question         {type, instructions, options|levels}   as asked
    state            full state object | null (null after redaction;
                     state_sha256 always identifies the state)
    state_sha256     sha256 of the canonical state JSON
    prediction       {answer, p, distribution} | null  what the model said
    feedback         {answer, comment?}   the user's ground truth
    provenance       {client, channel, session}  who/where (ids, not PII)
    privacy          {redacted, retention_days, opt_out}
    review           {status: pending|confirmed|rejected, reviewer, note}
    license          "CC BY-SA-4.0"   contributor terms (matches weights)

Discipline notes baked into the design:

  - benchmark hygiene: any feedback whose state/question overlaps a
    benchmark item is caught by the contamination audit's --feedback
    flag (scripts/contamination_audit.py) — feedback surfaces are
    audited exactly like generated training rows.
  - anti-poisoning: only review.status == "confirmed" records export,
    and confirmation is a human act. k-confirmation, per-client rate
    caps, and distribution heuristics are the next layer (documented
    in docs/FEEDBACK.md), not something this store silently decides.
  - privacy: states may contain user data. redact() removes values
    (keeping the hash); prune() drops old records by retention policy.

CLI:

    python3 -m kapteeni.feedback --store data_cache/feedback/records.jsonl \
        add --json '{...}'      # validate + append
        review fb-xxx --status confirmed --reviewer lucy
        prune --retention-days 180
        export --out data_cache/feedback/export.jsonl
        stats
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path

KINDS = ("correction", "confirmation", "outcome", "rejection")
REVIEW_STATES = ("pending", "confirmed", "rejected")
DEFAULT_RETENTION_DAYS = 365
CONTRIBUTOR_LICENSE = "CC BY-SA-4.0"


class FeedbackError(Exception):
    pass


def canonical(state) -> str:
    return json.dumps(state, sort_keys=True, ensure_ascii=False)


def hash_state(state) -> str:
    return hashlib.sha256(canonical(state).encode()).hexdigest()


def validate(rec: dict) -> None:
    def need(cond, msg):
        if not cond:
            raise FeedbackError(f"invalid feedback record: {msg}")

    need(isinstance(rec, dict), "not an object")
    need(rec.get("kind") in KINDS, f"kind must be one of {list(KINDS)}")
    need(isinstance(rec.get("served_as"), str) and rec["served_as"],
         "served_as (which model answered) is required")
    q = rec.get("question")
    need(isinstance(q, dict), "question must be an object")
    need(q.get("type") in ("noul", "choice", "score"),
         "question.type must be noul|choice|score")
    need("instructions" in q, "question.instructions is required")
    if q["type"] in ("choice", "score"):
        key = "options" if q["type"] == "choice" else "levels"
        need(isinstance(q.get(key), list) and q[key],
             f"question.{key} required for {q['type']}")
    st = rec.get("state")
    need(st is None or isinstance(st, (dict, str)),
         "state must be an object, a string, or null (redacted)")
    need(isinstance(rec.get("state_sha256"), str) and
         len(rec["state_sha256"]) == 64, "state_sha256 (64 hex) required")
    fb = rec.get("feedback")
    need(isinstance(fb, dict) and "answer" in fb,
         "feedback.answer is required (the user's ground truth)")
    pr = rec.get("provenance", {"client": "anonymous", "channel": "api",
                                "session": "none"})
    need(isinstance(pr, dict), "provenance must be an object")
    pv = rec.get("privacy", {"redacted": False,
                             "retention_days": DEFAULT_RETENTION_DAYS,
                             "opt_out": False})
    need(isinstance(pv, dict) and isinstance(pv.get("retention_days"),
                                            int), "privacy.retention_days "
         "must be an integer")
    rv = rec.get("review", {"status": "pending", "reviewer": None,
                            "note": None})
    need(rv.get("status") in REVIEW_STATES,
         f"review.status must be one of {list(REVIEW_STATES)}")


def new_record(served_as: str, kind: str, question: dict, state,
               feedback: dict, prediction: dict | None = None,
               request_id: str | None = None,
               served_revision: str | None = None,
               provenance: dict | None = None,
               retention_days: int = DEFAULT_RETENTION_DAYS,
               opt_out: bool = False) -> dict:
    rec = {
        "id": f"fb-{uuid.uuid4().hex[:16]}",
        "ts": time.time(),
        "kind": kind,
        "served_as": served_as,
        "served_revision": served_revision,
        "request_id": request_id,
        "question": question,
        "state": state,
        "state_sha256": hash_state(state),
        "prediction": prediction,
        "feedback": feedback,
        "provenance": provenance or {"client": "anonymous", "channel": "api",
                                     "session": "none"},
        "privacy": {"redacted": False, "retention_days": retention_days,
                    "opt_out": opt_out},
        "review": {"status": "pending", "reviewer": None, "note": None},
        "license": CONTRIBUTOR_LICENSE,
    }
    validate(rec)
    return rec


def redact(rec: dict, keys: tuple[str, ...]) -> dict:
    """Replace top-level state values with a placeholder (the hash stays,
    so audit/dedup keys survive). Only in-memory; the store writes what
    you give it."""
    st = rec.get("state")
    if not isinstance(st, dict):
        return rec
    out = dict(rec)
    out["state"] = {k: ("<redacted>" if k in keys else v)
                   for k, v in st.items()}
    out["privacy"] = dict(rec.get("privacy", {}), redacted=True)
    return out


class FeedbackStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        seen = set()
        for rec in self:
            if rec["id"] in seen:
                raise FeedbackError(f"duplicate id {rec['id']} in "
                                    f"{self.path} — the store is corrupt")
            seen.add(rec["id"])

    def __iter__(self):
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def add(self, rec: dict) -> dict:
        validate(rec)
        if not rec.get("id"):
            rec = dict(rec, id=f"fb-{uuid.uuid4().hex[:16]}")
        for existing in self:
            if existing["id"] == rec["id"]:
                raise FeedbackError(f"id {rec['id']} already exists")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def review(self, rec_id: str, status: str, reviewer: str = None,
               note: str = None) -> dict:
        if status not in REVIEW_STATES:
            raise FeedbackError(f"status must be one of {list(REVIEW_STATES)}")
        rows = list(self)
        hit = None
        for r in rows:
            if r["id"] == rec_id:
                r["review"] = {"status": status, "reviewer": reviewer,
                               "note": note}
                r["review"]["reviewed_ts"] = time.time()
                hit = r
        if hit is None:
            raise FeedbackError(f"no record {rec_id} in {self.path}")
        self._rewrite(rows)
        return hit

    def prune(self, retention_days: int) -> int:
        """Drop records past their retention. Confirmed records are
        kept: they are the vetted payload a future retrain decision
        needs; everything else expires on schedule."""
        cutoff = time.time() - retention_days * 86400
        rows = [r for r in self if r.get("ts", 0) >= cutoff
                or r["review"]["status"] == "confirmed"]
        n = self._count() - len(rows)
        self._rewrite(rows)
        return n

    def _count(self) -> int:
        return sum(1 for _ in self)

    def _rewrite(self, rows: list[dict]) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    def export_training(self, out: str | Path) -> int:
        """Confirmed records only, as a minimal training-shaped JSONL.
        NOT wired to any trainer — a future pre-registration decides
        what consuming feedback looks like (and audits it first)."""
        n = 0
        with open(out, "w", encoding="utf-8") as f:
            for r in self:
                if r["review"]["status"] != "confirmed":
                    continue
                f.write(json.dumps({
                    "source": f"feedback:{r['id']}",
                    "row_id": f"feedback-{r['id']}",
                    "state": r["state"],
                    "question": r["question"],
                    "label": r["feedback"]["answer"],
                    "provenance": r["provenance"],
                    "license": r["license"],
                }, ensure_ascii=False) + "\n")
                n += 1
        return n

    def stats(self) -> dict:
        out = {"total": 0, "kinds": {}, "review": {}, "served_as": {}}
        for r in self:
            out["total"] += 1
            out["kinds"][r["kind"]] = out["kinds"].get(r["kind"], 0) + 1
            st = r["review"]["status"]
            out["review"][st] = out["review"].get(st, 0) + 1
            sa = r["served_as"]
            out["served_as"][sa] = out["served_as"].get(sa, 0) + 1
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add")
    p.add_argument("--json", required=True,
                   help="the record body (id/ts may be omitted)")
    for extra in ("served-as",):
        p.add_argument(f"--{extra}")
    p = sub.add_parser("review")
    p.add_argument("id")
    p.add_argument("--status", required=True)
    p.add_argument("--reviewer")
    p.add_argument("--note")
    p = sub.add_parser("prune")
    p.add_argument("--retention-days", type=int,
                    default=DEFAULT_RETENTION_DAYS)
    p = sub.add_parser("export")
    p.add_argument("--out", required=True)
    sub.add_parser("stats")
    args = ap.parse_args(argv)
    store = FeedbackStore(args.store)
    if args.cmd == "add":
        rec = json.loads(args.json)
        if args.served_as and "served_as" not in rec:
            rec["served_as"] = args.served_as
        if "id" not in rec or "ts" not in rec:
            rec = new_record(
                rec.get("served_as", args.served_as), rec["kind"],
                rec["question"], rec.get("state"), rec["feedback"],
                prediction=rec.get("prediction"),
                request_id=rec.get("request_id"),
                served_revision=rec.get("served_revision"),
                provenance=rec.get("provenance"),
                retention_days=rec.get("privacy", {}).get(
                    "retention_days", DEFAULT_RETENTION_DAYS),
                opt_out=rec.get("privacy", {}).get("opt_out", False))
            rec["id"] = json.loads(args.json).get("id", rec["id"])
            rec["ts"] = json.loads(args.json).get("ts", rec["ts"])
        store.add(rec)
        print(f"added {rec['id']}")
        return 0
    if args.cmd == "review":
        r = store.review(args.id, args.status, args.reviewer, args.note)
        print(f"{r['id']} -> {r['review']['status']}")
        return 0
    if args.cmd == "prune":
        print(f"pruned {store.prune(args.retention_days)} records")
        return 0
    if args.cmd == "export":
        print(f"exported {store.export_training(args.out)} confirmed "
              f"records -> {args.out}")
        return 0
    print(json.dumps(store.stats(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())