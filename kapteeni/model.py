"""SystemOneModel: frozen backbone + trained heads behind the contract.

evaluate(state, questions) -> (answers, usage). Passes are built from CONTENT
only — question ids are never serialized, so two questions with identical
content always receive identical answers, and adding/removing a question can
never change another's answer (E1 parity is structural).

Readout modes (--readout / KAPTEENI_READOUT):
  head   trained heads only (the original v0)
  verb   native verbalizer only (yes/no logits at the answer: position)
  blend  geometric blend of both (default): p ∝ p_head^w · exp((1-w)·s_verb/τ)
         — Phase-1 result: raises JevBench Intelligence 30.9 -> 42.5 and
         top-label ECE 0.232 -> 0.065 on out-of-domain rubrics. w and τ are
         config, NOT trained: τ comes from mixed-domain val, w from the
         benchmark's public half (permitted; flagged in docs/JEVBENCH.md).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from kapteeni import contract
from kapteeni.backbone import (
    DEFAULT_MODEL,
    extract_h_last_verb,
    extract_shared_prefix,
    load_backbone,
)
from kapteeni.heads import PassMLP
from kapteeni.serialize import (
    choice_option_pass,
    instructions_text,
    noul_pass,
    score_level_pass,
    state_text,
)

SERVED_AS = "kapteeni-v1"

# blend config: loaded from the fit artifact when present. Constants below
# are the pre-refit fallback (w selected on the JevBench public half —
# permitted, flagged in docs/JEVBENCH.md; tau from mixed-domain val).
BLEND_W_DEFAULT = {"noul": 0.25, "choice": 0.25, "score": 0.25}
VERB_TAU_DEFAULT = {"noul": 2.0, "choice": 2.0, "score": 2.0}
VERB_B_DEFAULT = {"noul": 0.0, "choice": 0.0, "score": 0.0}
FIT_PATH = Path("data_cache/phase1/fit_kv.json")
READOUT = "blend"


def load_blend_params():
    """(w, tau, b) per primitive from the KV-path refit (mixed-domain val,
    gold-fitted; see refit_kv.py), else the pre-refit defaults."""
    if FIT_PATH.exists():
        f = json.loads(FIT_PATH.read_text())
        return ({qt: f[qt]["w"] for qt in ("noul", "choice", "score")},
                {qt: f[qt]["tau"] for qt in ("noul", "choice", "score")},
                {qt: f[qt].get("b", 0.0) for qt in ("noul", "choice", "score")})
    return BLEND_W_DEFAULT, VERB_TAU_DEFAULT, VERB_B_DEFAULT


class SystemOneModel:
    def __init__(self, bundle_path: str | dict, device: str = "cuda",
                 model_name: str = DEFAULT_MODEL, readout: str | None = None,
                 lora: str | None = None, blend: dict | None = None):
        blob = bundle_path if isinstance(bundle_path, dict) else \
            torch.load(bundle_path, weights_only=False)
        self.device = device
        self.model, self.tok = load_backbone(model_name, device)
        if lora:
            from peft import PeftModel

            print(f"merging LoRA adapter from {lora} ...", flush=True)
            pm = PeftModel.from_pretrained(self.model, lora)
            self.model = pm.merge_and_unload()
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        self.backbone_name = model_name
        self.heads: dict[str, PassMLP] = {}
        self.T: dict[str, float] = {}
        for qt in ("noul", "choice", "score"):
            if qt not in blob:
                continue
            head = PassMLP(blob[qt]["in_dim"]).to(device)
            head.load_state_dict(blob[qt]["head"])
            head.eval()
            self.heads[qt] = head
            self.T[qt] = blob[qt]["T"]
        self.meta = blob.get("meta", {})
        # Content-keyed cache: identical pass text -> identical (h, s_verb).
        # Gives bit-exact E1 parity (batched == single) despite bf16 kernel
        # jitter across batch compositions, and mirrors the reference's
        # deterministic serving behavior.
        self._hcache: dict[str, tuple[torch.Tensor, float]] = {}
        self._hcache_cap = 200_000
        self.readout = readout or READOUT
        if blend is not None:
            self.blend_w = {qt: blend[qt]["w"] for qt in ("noul", "choice", "score")}
            self.verb_tau = {qt: blend[qt]["tau"] for qt in ("noul", "choice", "score")}
            self.verb_b = {qt: blend[qt].get("b", 0.0) for qt in ("noul", "choice", "score")}
        else:
            self.blend_w, self.verb_tau, self.verb_b = load_blend_params()

    @classmethod
    def from_dist(cls, dist_dir: str, device: str = "cuda",
                  readout: str | None = None):
        """Load a packaged distribution (the HF / release layout): merged
        model dir + heads.safetensors + kapteeni-config.json."""
        import json as _json

        from safetensors.torch import load_file

        d = Path(dist_dir)
        cfg = _json.loads((d / "kapteeni-config.json").read_text())
        st = load_file(str(d / "heads.safetensors"))
        bundle = {}
        for qt in ("noul", "choice", "score"):
            sd = {k[len(qt) + 1:]: v for k, v in st.items()
                  if k.startswith(qt + ".")}
            bundle[qt] = {"head": sd, "T": cfg["head_temperatures"][qt],
                         "in_dim": cfg["in_dim"]}
        m = cls(bundle, device=device, model_name=str(d), readout=readout,
                blend=cfg["blend"])
        m.dist_dir = str(d)
        return m

    @staticmethod
    def _key(text: str) -> str:
        import hashlib

        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _embed(self, texts: list[str]):
        """(legacy full-forward path) kept for direct use and equivalence tests."""
        keys = [self._key(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._hcache]
        if missing:
            fresh_h, fresh_v = extract_h_last_verb(
                self.model, self.tok, [texts[i] for i in missing],
                device=self.device,
            )
            for j, i in enumerate(missing):
                self._hcache[keys[i]] = (fresh_h[j].cpu(), float(fresh_v[j]))
            if len(self._hcache) > self._hcache_cap:
                for k in list(self._hcache)[: len(self._hcache) // 2]:
                    del self._hcache[k]
        pairs = [self._hcache[k] for k in keys]
        h = torch.stack([p[0] for p in pairs])
        s_verb = [p[1] for p in pairs]
        return h, s_verb

    def _embed_texts(self, state_str: str, texts: list[str],
                     suffixes: list[str]):
        """Serving path: shared-prefix KV. Cache misses for ONE request share
        a single state prefill (they all reference the same state), so the
        whole request costs state_prefill + sum(suffix) tokens instead of
        n_passes x full prefill (~4.5x fewer on long-state decisions)."""
        keys = [self._key(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._hcache]
        if missing:
            fresh_h, fresh_v = extract_shared_prefix(
                self.model, self.tok, state_str,
                [suffixes[i] for i in missing], device=self.device,
            )
            for j, i in enumerate(missing):
                self._hcache[keys[i]] = (fresh_h[j].cpu(), float(fresh_v[j]))
            if len(self._hcache) > self._hcache_cap:
                for k in list(self._hcache)[: len(self._hcache) // 2]:
                    del self._hcache[k]
        pairs = [self._hcache[k] for k in keys]
        h = torch.stack([p[0] for p in pairs])
        s_verb = [p[1] for p in pairs]
        return h, s_verb

    # ---------------------------------------------------------------- passes

    def _build_passes(self, state, questions: dict[str, dict]):
        passes = []  # (qid, kind, key, text)
        for qid, q in questions.items():
            crit = q.get("criteria")
            if q["type"] == "noul":
                passes.append((qid, "noul", None,
                               noul_pass(state, q["instructions"], crit)))
            elif q["type"] == "choice":
                for opt, desc in crit.items():
                    passes.append((qid, "choice_opt", opt,
                                   choice_option_pass(state, q["instructions"], opt, desc)))
            else:  # score
                levels = crit
                for i, lvl in enumerate(levels):
                    passes.append((qid, "score_lvl", i,
                                   score_level_pass(state, q["instructions"], i, len(levels), lvl)))
        return passes

    def _count_tokens(self, texts: list[str]) -> int:
        enc = self.tok(texts, add_special_tokens=False)["input_ids"]
        return sum(len(x) for x in enc)

    # -------------------------------------------------------------- evaluate

    def evaluate(self, state, questions: dict[str, dict]):
        passes = self._build_passes(state, questions)
        state_str = state_text(state)
        texts = [p[3] for p in passes]
        suffixes = [p[3][len(state_str) + 2:] for p in passes]
        h, s_verb = self._embed_texts(state_str, texts, suffixes)

        head_of = {"noul": "noul", "choice_opt": "choice", "score_lvl": "score"}
        verb_pos = {qt: [] for qt in head_of}
        z_flat: list[tuple[int, float]] = []
        for qt in ("noul", "choice_opt", "score_lvl"):
            idx = [i for i, p in enumerate(passes) if p[1] == qt]
            if not idx:
                continue
            hs = h[idx].to(self.device).float()
            with torch.no_grad():
                z = self.heads[head_of[qt]](hs).cpu().tolist()
            for i, val in zip(idx, z):
                z_flat.append((i, val / self.T[head_of[qt]]))
                verb_pos[qt].append((i, s_verb[i]))

        answers: dict = {}
        scores_by_qid: dict[str, dict] = {}
        for i, v in z_flat:
            qid, kind, key, _ = passes[i]
            scores_by_qid.setdefault(qid, {"kind": kind, "scores": {}})["scores"][key] = v

        for qid, info in scores_by_qid.items():
            kind = info["kind"]
            qt = head_of[kind]
            # verbalizer scores aligned with this question's keys
            vmap = {}
            for (i, sv) in verb_pos[kind]:
                pi = passes[i]
                if pi[0] == qid:
                    vmap[pi[2]] = sv

            if self.readout == "head":
                answers[qid] = _answer(kind, info["scores"], questions, qid)
                continue

            if self.readout == "verb":
                s_v = {k: (vmap[k] + self.verb_b[qt]) / self.verb_tau[qt]
                       for k in vmap}
                answers[qid] = _answer(kind, s_v, questions, qid)
                continue

            # blend: geometric mean in score space (Phase 1 formula):
            # s = w * log(p_head) + (1-w) * (s_verb + b)/tau -> softmax/sigmoid
            p_head = _probs(kind, info["scores"])
            w = self.blend_w[qt]
            if kind == "noul":
                sh = math_logit(p_head[None])
                sv = (vmap[None] + self.verb_b["noul"]) / self.verb_tau["noul"]
                s = w * sh + (1 - w) * sv
                answers[qid] = _answer(kind, {None: s}, questions, qid)
            else:
                keys = (sorted(p_head, key=lambda k: int(k))
                        if kind == "score_lvl" else list(p_head))
                s = {}
                for k in keys:
                    sh = math.log(max(p_head[k], 1e-9))
                    sv = (vmap[k] + self.verb_b[qt]) / self.verb_tau[qt]
                    s[k] = w * sh + (1 - w) * sv
                answers[qid] = _answer(kind, s, questions, qid)

        # usage: state once + each question once; output = serialized answers
        in_tok = self._count_tokens([state_text(state)])
        q_texts = []
        for q in questions.values():
            it = instructions_text(q["instructions"])
            crit = json.dumps(q.get("criteria"), ensure_ascii=False) if q.get("criteria") is not None else ""
            q_texts.append(it + " " + crit)
        in_tok += self._count_tokens(q_texts)
        out_tok = self._count_tokens([json.dumps(answers, ensure_ascii=False)])
        usage = {"input_tokens": in_tok, "output_tokens": out_tok}
        return answers, usage


def math_logit(p):
    return math.log(max(p, 1e-9) / max(1 - p, 1e-9))


def _probs(kind, scores) -> dict:
    """scores -> distribution dict (keys preserved)."""
    if kind == "noul":
        v = next(iter(scores.values()))
        return {None: 1 / (1 + math.exp(-v))}
    keys = (sorted(scores, key=lambda k: int(k)) if kind == "score_lvl"
            else list(scores))
    m = max(scores[k] for k in keys)
    e = [math.exp(scores[k] - m) for k in keys]
    t = sum(e)
    return {k: e[j] / t for j, k in enumerate(keys)}


def _answer(kind, scores, questions, qid):
    """scores: PRE-softmax logit scores. The contract functions apply the
    single softmax (double-softmax through here flattened every choice/score
    distribution — the Phase-1b ECE bug; fixed 2026-09-24)."""
    if kind == "noul":
        v = scores[None] if isinstance(scores, dict) else scores
        return contract.noul_answer(1 / (1 + math.exp(-v)))
    if kind == "choice_opt":
        return contract.choice_answer(scores)
    keys = sorted(scores, key=lambda k: int(k))
    return contract.score_answer(
        [scores[k] for k in keys], questions[qid]["criteria"]
    )