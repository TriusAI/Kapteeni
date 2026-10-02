# Feedback capture — the on-the-spot learning substrate

Kapteeni serves frozen, calibrated models; nothing in serving changes
weights or constants. What *can* change on the spot is what the
deployment **knows** about its mistakes. `kapteeni/feedback.py` is the
capture layer: an append-only, schema-validated, privacy-aware store
of user ground truths, wired into the server behind an explicit flag.
It deliberately does **not** train anything — what consuming feedback
looks like is a future pre-registration (see the end).

## Enabling capture

```sh
python3 -m kapteeni.serve_v11c --dist ../kapteeni-v1.1c-dist \
    --port 8002 --feedback data_cache/feedback/records.jsonl
# optional: pin what served the answers
KAPTEENI_SERVED_REVISION=9cbc9e32...
```

`POST /v1/feedback` (same auth as `/v1/systemone`):

```json
{
  "kind": "correction",
  "request_id": "<usage id of the served answer, when echoable>",
  "question": {"type": "choice", "instructions": "...", "options": [...]},
  "state": {...},
  "prediction": {"answer": 1, "p": 0.51},
  "feedback": {"answer": 0, "comment": "the pro tier caps at 5 seats"},
  "provenance": {"client": "user-42", "channel": "demo"},
  "privacy": {"retention_days": 365, "opt_out": false}
}
```

The server assigns `id`/`ts`, stamps `served_as` (and
`served_revision` from the env when pinned), hashes the state
(`state_sha256` — survives redaction as a dedup/audit key), and files
the record as `review: pending`. Responses: `200 {id, review: pending}`,
`422` on schema violations, `404` when the operator did not enable
collection, `401` when auth fails.

Kinds: `correction` (the model was wrong, here is the truth),
`confirmation` (the model was right), `outcome` (what happened
downstream), `rejection` (the answer was unusable for reasons other
than correctness).

## Hygiene rules (enforced or documented)

- **Benchmark firewall.** Feedback states are potential future
  training data, so they are audited exactly like generated rows:
  `python3 scripts/contamination_audit.py --feedback
  data_cache/feedback/*.jsonl` shingles feedback surfaces against
  JevBench + OCNLI. The forge runs this automatically for runs that
  list `feedback` paths. A feedback record that quotes a benchmark
  item is a contamination incident, not training material.
- **Review gate.** Only `review: confirmed` records export
  (`feedback review fb-xxx --status confirmed`). Confirmation is a
  human act — the store never auto-promotes. This is the first
  anti-poisoning layer; k-confirmation thresholds, per-client rate
  caps, and distribution heuristics belong to the *consumption*
  pre-registration, not to the capture layer.
- **Privacy.** States may contain user data. `redact(rec, keys)`
  blanks configured keys in place (hash preserved);
  `feedback prune --retention-days N` expires records past retention —
  confirmed records are kept, because they are the vetted payload any
  retrain decision needs. `opt_out: true` marks records the client
  asked to exclude; exports must respect it.
- **Provenance.** `client`/`session` are opaque ids the deployer
  assigns — never assume PII-free, never log more than needed.
- **License.** Records are stamped `CC BY-SA-4.0` (contributor terms
  matching the weights license). A deployment surfacing the endpoint
  should disclose these terms to its users.

## CLI

```sh
python3 -m kapteeni.feedback --store data_cache/feedback/records.jsonl \
    add --json '{...}'                      # validate + append
    review fb-xxx --status confirmed --reviewer <who>
    prune --retention-days 180
    export --out data_cache/feedback/export.jsonl
    stats
```

## What consuming feedback would look like (NOT built)

The honest sequence, when the project chooses to build it:

1. A **pre-registration** (recipe, gates, decision rule — the same
   discipline as every training run) for feedback-augmented training:
   confirmed, audited, deduplicated corrections mixed into a forge
   recipe as an additional data source.
2. Gates re-run in full before any adapter swap — the v1.1b history is
   the standing warning that pressure on new data erodes other
   skills; "learning from users" inherits every rule that governs
   "learning from synthetics".
3. Calibration re-fit + ECE gate, because any weights change
   invalidates fitted temperatures.

True mid-session weight updates (hot-swapping adapters on live
traffic) would invalidate the calibration guarantee on every swap and
are rejected as a serving design; see the discussion in the worklog
(2026-10-03).