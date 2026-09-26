"""Committee serving mode: output-average two Kapteeni variants.

The model soup failed by averaging WEIGHTS across loss basins; averaging
OUTPUT PROBABILITIES is the safe form of ensembling — each model stays in
its own basin, disagreements average out, and no retraining is involved.
Pre-registered in docs/PREREG-COMMITTEE.md; evaluation is deployment-side
only (val slices + measured latency), no benchmark runs.

Serve it:

    python3 -m kapteeni.serve --committee \
        --bundle model_cache/kapteeni_v1.pt \
        --lora model_cache/kapteeni_p2/adapter \
        --bundle2 model_cache/kapteeni_v1_2_1.pt \
        --lora2 model_cache/kapteeni_p2_s3/adapter \
        --fit2 data_cache/phase1/fit_kv_v1_2_1.json \
        --served-as kapteeni-v1-committee --port 8000
"""

from __future__ import annotations

from kapteeni.contract import DECIMALS, _round_repair, confidence, noul_answer


def merge_answers(a: dict, b: dict) -> dict:
    """Merge two served answers to the same question by averaging their
    output probabilities, then re-derive every downstream field with the
    contract's own formulas (argmax, expectation, confidence, rounding)."""
    t = a["type"]
    if t == "noul":
        # absolute readout: average the probabilities directly
        return noul_answer((a["noul"] + b["noul"]) / 2)
    if t == "choice":
        keys = list(a["probabilities"])
        probs = _round_repair(
            [(a["probabilities"][o] + b["probabilities"][o]) / 2 for o in keys])
        top = max(range(len(keys)), key=lambda i: probs[i])
        return {"type": "choice", "choice": keys[top],
                "probabilities": {o: probs[i] for i, o in enumerate(keys)},
                "confidence": confidence(probs)}
    if t == "score":
        keys = sorted(a["probabilities"], key=lambda s: int(s))
        probs = _round_repair(
            [(a["probabilities"][k] + b["probabilities"][k]) / 2 for k in keys])
        score = round(sum(i * p for i, p in enumerate(probs)), DECIMALS)
        return {"type": "score", "score": score, "legend": a["legend"],
                "probabilities": {k: probs[i] for i, k in enumerate(keys)},
                "confidence": confidence(probs)}
    raise ValueError(f"unknown answer type {t!r}")


class CommitteeModel:
    """Serves two variants as one decision model; answers are merged."""

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def evaluate(self, state, questions: dict[str, dict]):
        ans_a, usage_a = self.a.evaluate(state, questions)
        ans_b, usage_b = self.b.evaluate(state, questions)
        answers = {q: merge_answers(ans_a[q], ans_b[q]) for q in ans_a}
        # input tokens are the decision's (state+questions, billed once);
        # output tokens honestly count BOTH models' internal passes.
        usage = {"input_tokens": usage_a["input_tokens"],
                 "output_tokens": usage_a["output_tokens"]
                 + usage_b["output_tokens"]}
        return answers, usage