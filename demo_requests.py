"""Demo: run the reference docs' example requests against a running kapteeni server.

    python3 -m kapteeni.serve --bundle model_cache/kapteeni_v0.pt --port 8000
    python3 demo_requests.py [--base http://127.0.0.1:8000]

Every request below is taken from docs.typesafe.ai (API reference, primitives,
State page) — running them unchanged against our server is the E12-style
drop-in check.
"""

import argparse
import json
import urllib.request

BASE_DEFAULT = "http://127.0.0.1:8000"


def post(base: str, body: dict) -> dict:
    req = urllib.request.Request(
        base + "/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def show(title: str, body: dict, resp: dict):
    print(f"\n=== {title} ===")
    qs = body["questions"]
    for qid in qs:
        a = resp["answers"][qid]
        if a["type"] == "noul":
            print(f"  {qid}: noul = {a['noul']}")
        elif a["type"] == "choice":
            top3 = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:3]
            dist = "  ".join(f"{k}={p:.2f}" for k, p in top3)
            print(f"  {qid}: choice = {a['choice']!r}  [{dist}]  confidence={a['confidence']}")
        else:
            print(f"  {qid}: score = {a['score']}  confidence={a['confidence']}")
            dist = "  ".join(f"{k}={p:.2f}" for k, p in a["probabilities"].items() if p > 0.01)
            print(f"            {dist}")
    print(f"  usage: {resp['usage']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_DEFAULT)
    args = ap.parse_args()

    # 1) API reference example: urgency
    body = {
        "state": "Help! My payouts have been failing for 3 days.",
        "model": "jev-latest",
        "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}},
    }
    show("docs: urgency (API reference)", body, post(args.base, body))

    # 2) API reference example: department choice
    body = {
        "state": "Help! My payouts have been failing for 3 days.",
        "model": "jev-latest",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Payments, invoicing, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": "Pricing, upgrades, new accounts",
                },
            }
        },
    }
    show("docs: department routing (Choice)", body, post(args.base, body))

    # 3) Primitives page example: three primitives, one request
    body = {
        "state": "Our API integration started returning 500 errors on every request about 20 minutes ago, and we can't process any customer orders until this is fixed.",
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
            "is_urgent": {"type": "noul", "instructions": "The message conveys urgency or time-sensitivity"},
            "frustration": {
                "type": "score",
                "instructions": "How frustrated the customer appears",
                "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"],
            },
        },
    }
    show("docs: three primitives, one request", body, post(args.base, body))

    # 4) State page example: refund workflow (dot-paths into structured state)
    body = {
        "state": {
            "ticket": {"subject": "Duplicate charge",
                       "messages": [{"from": "customer", "text": "I was charged twice for order A-104. Please refund the duplicate."}]},
            "order": {"id": "A-104", "charges": [
                {"amount_usd": 49, "status": "captured"},
                {"amount_usd": 49, "status": "captured"}]},
            "refund_policy": "Duplicate charges are eligible for a refund.",
        },
        "model": "jev-latest",
        "questions": {
            "refund_requested": {"type": "noul", "instructions": "Does `ticket.messages[0].text` request a refund?"},
            "policy_supports": {"type": "noul", "instructions": "Does `refund_policy` cover the request in `ticket.messages[0].text`, given `order.charges`?"},
        },
    }
    show("docs: structured state + dot-paths (State page)", body, post(args.base, body))

    # 5) FEVER-style fact check against the held-out domain
    body = {
        "state": {"claim": "Roman Atwood is a content creator.", "evidence": "Roman Atwood is known for his vlogs and prank videos on YouTube."},
        "model": "jev-latest",
        "questions": {"supported": {"type": "noul", "instructions": "Is `claim` supported by `evidence`?"}},
    }
    show("held-out domain: fact check", body, post(args.base, body))


if __name__ == "__main__":
    main()