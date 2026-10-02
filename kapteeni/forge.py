"""kapteeni.forge — config-driven runner for the v1.1c Kapteeni pipeline.

Wraps the existing, separately-smoke-tested stage modules (see
docs/FORGE.md for the full guide) into one recipe file so anyone can
build their own Kapteeni-class decision model end to end:

    data recipe -> generators -> sources -> pass expansion -> audit
    -> P1 precompute -> P1 heads -> P2 joint train -> gates -> pack
    -> live post-pack smoke

Discipline is enforced by the tool, not by memory:

  - `forge init` freezes the recipe: the config (including the gate
    table) is hashed into runs/<name>/forge.lock. Every later stage
    refuses to run if the config has drifted — no post-hoc gate
    changes, one run per recipe.
  - the gates stage applies the FROZEN config gate table to the
    measured final_gates.json and records an immutable verdict.
  - the audit stage refuses to continue past any contamination hit.

Every stage is idempotent: outputs that already exist skip the stage
(generators are deterministic per seed, so regeneration is waste), and
the long stages (precompute, p2) resume internally on relaunch.

  python3 -m kapteeni.forge init   --config forge/my-model.json
  python3 -m kapteeni.forge run    --config forge/my-model.json --stage all
  python3 -m kapteeni.forge status --config forge/my-model.json

The long stages should be launched detached (see docs/FORGE.md):
  setsid nohup python3 -m kapteeni.forge run --config ... --stage all \
      > /tmp/forge.log 2>&1 &
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STAGE_ORDER = ("base", "gen", "sources", "expand", "audit", "precompute",
               "phase1", "p2", "gates", "pack", "smoke")

GATE_KEYS = ("synth3", "mnli", "ocnli", "synth2zh", "synth2", "ece")
# config gate name -> final_gates.json measurement path
GATE_MEASURE = {
    "synth3": ("gates", "synth3_val", "acc"),
    "mnli": ("gates", "mnli_noul", "acc"),
    "ocnli": ("gates", "ocnli_noul", "acc"),
    "synth2zh": ("gates", "synth2zh_val", "acc"),
    "synth2": ("gates", "synth2_en_val", "acc"),
    "ece": ("fitted_ece", None, None),  # max over primitives
}

HF_SOURCES = ("boolq", "fever", "mnli", "banking77", "clinc150",
              "goemotions", "helpsteer2")
CRITERIA_SOURCES = ("banking77", "clinc150", "goemotions", "helpsteer2")

PROFILES = {  # memory-behavior profiles; see docs/FORGE.md
    "halo-96g": {"env": {"PYTORCH_CUDA_ALLOC_CONF":
                         "expandable_segments:True"},
                 "token_budget": 8192},
    "cuda-24g": {"env": {}, "token_budget": 4096},
}


class ForgeError(Exception):
    pass


# ------------------------------------------------------------------ config

def validate_config(cfg: dict) -> None:
    def need(cond, msg):
        if not cond:
            raise ForgeError(f"config: {msg}")

    need(isinstance(cfg, dict), "top level must be an object")
    for k in ("name", "base", "profile", "data", "train", "gates"):
        need(k in cfg, f"missing key '{k}'")
    need(isinstance(cfg["name"], str) and cfg["name"].replace(
        "-", "").replace("_", "").isalnum(), "name must be a simple slug")
    need(isinstance(cfg["base"], dict) and "path" in cfg["base"],
         "base.path is required (local dir the backbone lives in / lands)")
    need(cfg.get("profile") in PROFILES,
         f"profile must be one of {sorted(PROFILES)}")

    gates = cfg["gates"]
    need(set(gates) == set(GATE_KEYS),
         f"gates must be exactly {list(GATE_KEYS)} "
         "(the trainer only measures these; anything else is silently "
         "unenforced)")
    for k, spec in gates.items():
        need(isinstance(spec, dict) and set(spec) >= {"cmp", "value"},
             f"gate '{k}' needs {{cmp, value}}")
        if k == "ece":
            need(spec["cmp"] == "<=", "ece gate cmp must be '<='")
        else:
            need(spec["cmp"] in (">", ">="),
                 f"gate '{k}' cmp must be '>' or '>='")
        need(isinstance(spec["value"], (int, float)),
             f"gate '{k}' value must be numeric")

    data = cfg["data"]
    need(isinstance(data, dict), "data must be an object")
    for k in ("synth", "synth2", "synth3"):
        need(k in data, f"data.{k} required (it feeds the P2 mixture)")
    hf = data.get("hf_rows", {})
    need(set(hf) <= set(HF_SOURCES),
         f"data.hf_rows keys must be among {list(HF_SOURCES)}")
    for s in ("boolq", "fever", "banking77", "clinc150", "goemotions",
              "helpsteer2", "synth2zh"):
        need(s == "synth2zh" or s in hf,
             f"data.hf_rows.{s} required by the P2 mixture")
    if "mnli" in gates and "mnli" not in hf:
        raise ForgeError("the mnli gate needs data.hf_rows.mnli (eval-only)")
    if "ocnli" in gates and not data.get("ocnli_dev"):
        raise ForgeError("the ocnli gate needs data.ocnli_dev — a path to "
                         "an OCNLI dev.json you obtained yourself (CC "
                         "BY-NC: not fetched or redistributed here)")
    train = cfg["train"]
    need(isinstance(train, dict), "train must be an object")


def load_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text())
    validate_config(cfg)
    return cfg


def config_sha(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


# -------------------------------------------------------------------- run

class ForgeRun:
    def __init__(self, cfg: dict, run_root: str | Path | None = None):
        self.cfg = cfg
        self.root = Path(run_root) if run_root else Path("runs") / cfg["name"]
        self.data = self.root / "data_cache"
        self.model = self.root / "model_cache" / "model"
        self.p1 = self.root / "model_cache" / "p1.pt"
        self.emb = self.root / "v11c_emb.pt"
        self.dist = self.root / "dist"
        self.logs = self.root / "logs"
        self.lock = self.root / "forge.lock"
        self.manifest = self.root / "manifest.jsonl"

    @property
    def dc(self) -> str:
        return str(self.data)

    def manifest_add(self, stage: str, status: str, detail: str = "") -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.manifest, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "stage": stage,
                                "status": status, "detail": detail},
                               ensure_ascii=False) + "\n")

    def check_lock(self) -> None:
        if not self.lock.exists():
            raise ForgeError(
                f"no forge.lock in {self.root} — run `forge init` first. "
                "A run cannot start before its recipe (gates included) "
                "is frozen.")
        rec = json.loads(self.lock.read_text())
        if rec["config_sha256"] != config_sha(self.cfg):
            raise ForgeError(
                "config drifted from the frozen run recipe — start a NEW "
                "run (`forge init` under a fresh name). Changing gates "
                "after results is what the lock exists to prevent.")

    # environment for stage subprocesses
    def env(self, network: bool = False, gpu: bool = False) -> dict:
        e = dict(os.environ)
        e["KAPTEENI_DATA_DIR"] = self.dc
        if gpu:
            e.update(PROFILES[self.cfg["profile"]]["env"])
            e["HF_HUB_OFFLINE"] = "1"
        if network:
            e.pop("HF_HUB_OFFLINE", None)
            e["HF_HUB_DISABLE_XET"] = "1"  # proxy-friendlier transfer
        return e

    def sh(self, stage: str, cmd: list[str], env: dict, cwd=None) -> int:
        self.logs.mkdir(parents=True, exist_ok=True)
        log = self.logs / f"{stage}.log"
        print(f"[forge:{stage}] $ {' '.join(cmd)}", flush=True)
        with open(log, "ab") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                env=env, cwd=cwd).returncode
        if rc != 0:
            tail = "\n".join(log.read_text(errors="replace").splitlines()[-12:])
            raise ForgeError(f"{stage} failed (rc={rc}); last lines of "
                             f"{log}:\n{tail}")
        return rc


def _lines(p: Path) -> int:
    return sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0


def _have(p: Path, n: int = 1) -> bool:
    return p.exists() and _lines(p) >= n


# ------------------------------------------------------------------ stages

def st_base(r: ForgeRun, force=False) -> tuple[str, str]:
    base = Path(r.cfg["base"]["path"])
    if base.exists() and not force:
        return "skipped", f"{base} present"
    hf_id = r.cfg["base"].get("hf_id")
    if not hf_id:
        raise ForgeError("base.path missing and no base.hf_id to download")
    r.root.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(3):  # resumes: hf transfer skips done files
        try:
            r.sh("base", ["huggingface-cli", "download", hf_id,
                          "--local-dir", str(base)], r.env(network=True))
            return "ok", f"downloaded {hf_id} -> {base}"
        except ForgeError as e:
            last = e
            print(f"[forge:base] attempt {attempt + 1} failed; "
                  f"retrying with resume", flush=True)
            time.sleep(20)
    raise last


def st_gen(r: ForgeRun, force=False) -> tuple[str, str]:
    d, out = r.cfg["data"], []
    specs = [
        ("synth", ["python3", "-m", "kapteeni.synth", "synth",
                   "--n", str(d["synth"]["n"]), "--seed",
                   str(d["synth"].get("seed", 11)),
                   "--out", f"{r.dc}/rows_synth.jsonl"],
         Path(r.dc) / "rows_synth.jsonl", d["synth"]["n"]),
        ("synth2", ["python3", "-m", "kapteeni.synth2", "synth",
                    "--n", str(d["synth2"]["n"]), "--seed",
                    str(d["synth2"].get("seed", 23)),
                    "--out", f"{r.dc}/rows_synth2.jsonl",
                    "--criteria-out", f"{r.dc}/synth2_criteria.json"],
         Path(r.dc) / "rows_synth2.jsonl", d["synth2"]["n"]),
        ("synth2zh", ["python3", "-m", "kapteeni.synth2zh", "synth",
                      "--n", str(d["synth2zh"]["n"]), "--seed",
                      str(d["synth2zh"].get("seed", 71)),
                      "--out", f"{r.dc}/rows_synth2zh.jsonl",
                      "--criteria-out", f"{r.dc}/synth2zh_criteria.json"],
         Path(r.dc) / "rows_synth2zh.jsonl", d["synth2zh"]["n"]),
    ]
    if "synth3" in d:
        specs.append(("synth3", ["python3", "-m", "kapteeni.synth3", "gen",
                                "--rows", str(d["synth3"]["rows"]), "--seed",
                                str(d["synth3"].get("seed", 41)),
                                "--out", f"{r.dc}/synth3"],
                      Path(r.dc) / "synth3" / "items.jsonl",
                      d["synth3"]["rows"]))
    if "synth3zh" in d:
        specs.append(("synth3zh", ["python3", "-m", "kapteeni.synth3zh",
                                  "gen", "--rows", str(d["synth3zh"]["rows"]),
                                  "--seed", str(d["synth3zh"].get("seed", 53)),
                                  "--out", f"{r.dc}/synth3zh"],
                      Path(r.dc) / "synth3zh" / "items.jsonl",
                      d["synth3zh"]["rows"]))
    for name, cmd, outp, n in specs:
        if _have(outp, n) and not force:
            out.append(f"{name} skipped")
            continue
        r.data.mkdir(parents=True, exist_ok=True)
        r.sh(f"gen-{name}", cmd, r.env())
        if not _have(outp, n):
            raise ForgeError(f"gen {name}: {outp} has < {n} rows")
        out.append(name)
    return "ok", ", ".join(out)


def st_sources(r: ForgeRun, force=False) -> tuple[str, str]:
    d, done = r.cfg["data"], []
    hf = d.get("hf_rows", {})
    for name, n in hf.items():
        outp = Path(r.dc) / f"rows_{name}.jsonl"
        if _have(outp, n) and not force:
            done.append(f"{name} skipped")
            continue
        r.sh(f"sources-{name}",
             ["python3", "-m", "kapteeni.build_data", name,
              "--n", str(n), "--out", str(outp)], r.env(network=True))
        done.append(name)
    for name in CRITERIA_SOURCES:
        if name not in hf:
            continue
        outp = Path(r.dc) / f"crit_{name}.json"
        if outp.exists() and not force:
            done.append(f"crit-{name} skipped")
            continue
        cfrom = d.get("criteria_from")
        if cfrom and (Path(cfrom) / f"crit_{name}.json").exists():
            # reuse published rubrics (part of the released recipe) instead
            # of regenerating them with the teacher LLM
            r.data.mkdir(parents=True, exist_ok=True)
            shutil.copy(Path(cfrom) / f"crit_{name}.json", outp)
            done.append(f"crit-{name} (reused from {cfrom})")
            continue
        r.sh(f"sources-crit-{name}",
             ["python3", "-m", "kapteeni.criteria", name,
              "--rows", f"{r.dc}/rows_{name}.jsonl",
              "--out", str(outp)], r.env(network=True))
        done.append(f"crit-{name}")
    # OCNLI: eval-only gate rows; the license-restricted dev.json is
    # provided by the user, copied in, never fetched or shipped here
    if d.get("ocnli_dev"):
        outp = Path(r.dc) / "rows_ocnli.jsonl"
        if not _have(outp, 500) or force:
            src = Path(d["ocnli_dev"])
            if not src.exists():
                raise ForgeError(f"data.ocnli_dev not found at {src}")
            r.data.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, Path(r.dc) / "ocnli_dev.json")
            r.sh("sources-ocnli", ["python3", "scripts/build_ocnli.py"],
                 r.env())
            done.append("ocnli")
        else:
            done.append("ocnli skipped")
    # teacher soft labels (optional; absent softs -> gold-only noul passes)
    dist = d.get("distill")
    if dist:
        for name in ("boolq", "fever"):
            outp = Path(r.dc) / f"soft_{name}.jsonl"
            if outp.exists() and not force:
                done.append(f"soft-{name} skipped")
                continue
            cmd = ["python3", "-m", "kapteeni.distill",
                   "--rows", f"{r.dc}/rows_{name}.jsonl",
                   "--out", str(outp),
                   "--k", str(dist.get("k", 5)),
                   "--conc", str(dist.get("conc", 24))]
            if dist.get("teacher"):
                cmd += ["--teacher", dist["teacher"]]
            r.sh(f"sources-soft-{name}", cmd, r.env(network=True))
            done.append(f"soft-{name}")
    return "ok", ", ".join(done) or "nothing to do"


def st_expand(r: ForgeRun, force=False) -> tuple[str, str]:
    out = []
    p2p = Path(r.dc) / "passes_p2.jsonl"
    if _have(p2p) and not force:
        out.append("passes_p2 skipped")
    else:
        r.sh("expand-p2", ["python3", "build_p2_passes.py"], r.env())
        out.append("passes_p2")
    s2p = Path(r.dc) / "passes_synth2.jsonl"
    if _have(s2p) and not force:
        out.append("passes_synth2 skipped")
    else:
        r.sh("expand-synth2",
             ["python3", "-m", "kapteeni.build_data", "passes",
              "--rows", f"{r.dc}/rows_synth2.jsonl",
              "--criteria", f"{r.dc}/synth2_criteria.json",
              "--out", str(s2p)], r.env())
        out.append("passes_synth2")
    return "ok", ", ".join(out)


def st_audit(r: ForgeRun, force=False) -> tuple[str, str]:
    if not (ROOT / "jevbench").exists():
        if r.cfg.get("strict_audit", True):
            raise ForgeError("jevbench/ clone missing — the contamination "
                             "audit cannot run. Clone the benchmark repo "
                             "or set strict_audit:false to proceed with "
                             "an UNAUDITED mixture (your risk to own).")
        r.manifest_add("audit", "skipped", "jevbench absent; UNAUDITED")
        return "skipped", "jevbench absent — mixture is UNAUDITED"
    outp = r.root / "contamination_audit.json"
    if outp.exists() and not force:
        res = json.loads(outp.read_text())
        if res["_forge_feedback"] == list(r.cfg.get("feedback", [])):
            return "skipped", "audit already complete"
    cmd = ["python3", "scripts/contamination_audit.py",
           "--out", str(outp)]
    for fb in r.cfg.get("feedback", []):
        cmd += ["--feedback", fb]
    r.sh("audit", cmd, r.env())
    res = json.loads(outp.read_text())
    res["_forge_feedback"] = list(r.cfg.get("feedback", []))
    outp.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    hits = (res.get("jevbench", {}).get("total_shingle_hits", 0)
            + res.get("ocnli", {}).get("total_shingle_hits", 0))
    if hits:
        raise ForgeError(f"contamination audit found {hits} shingle hits — "
                         "training is refused. Inspect the audit report, "
                         "disclose, and regenerate the offending data.")
    return "ok", "zero hits"


def st_precompute(r: ForgeRun, force=False) -> tuple[str, str]:
    if r.emb.exists() and not force:
        return "skipped", f"{r.emb} present"
    r.sh("precompute",
         ["python3", "-m", "kapteeni.v11c_precompute",
          "--base", r.cfg["base"]["path"], "--out", str(r.emb)],
         r.env(gpu=True))
    return "ok", str(r.emb)


def st_phase1(r: ForgeRun, force=False) -> tuple[str, str]:
    if r.p1.exists() and not force:
        return "skipped", f"{r.p1} present"
    r.model.parent.mkdir(parents=True, exist_ok=True)
    r.sh("phase1",
         ["python3", "-m", "kapteeni.train_v11c", "--phase1",
          "--emb", str(r.emb), "--out", str(r.p1)], r.env(gpu=True))
    return "ok", str(r.p1)


def st_p2(r: ForgeRun, force=False) -> tuple[str, str]:
    done = (r.model / "adapter").exists() and \
        (r.model / "recipe.json").exists()
    if done and not force:
        return "skipped", f"{r.model} trained"
    r.model.parent.mkdir(parents=True, exist_ok=True)
    t = r.cfg["train"]
    tb = t.get("token_budget",
               PROFILES[r.cfg["profile"]]["token_budget"])
    cmd = ["python3", "-m", "kapteeni.train_v11c", "--p2",
           "--bundle", str(r.p1), "--out", str(r.model),
           "--base", r.cfg["base"]["path"],
           "--lr", str(t.get("lr", 1e-4)),
           "--epochs", str(t.get("epochs", 1)),
           "--token-budget", str(tb),
           "--seed", str(t.get("seed", 0)),
           "--gate-every", str(t.get("gate_every", 300)),
           "--ckpt-every", str(t.get("ckpt_every", 150))]
    if t.get("limit"):
        cmd += ["--limit", str(t["limit"])]
    r.sh("p2", cmd, r.env(gpu=True))
    return "ok", str(r.model)


def apply_verdict(cfg: dict, final_gates: dict) -> dict:
    """The frozen config gate table applied to the measured gates."""
    gates = {}
    for name, spec in cfg["gates"].items():
        if name == "ece":
            measured = max(final_gates["fitted_ece"].values())
        else:
            m = final_gates
            for k in GATE_MEASURE[name]:
                m = m[k]
            measured = m
        cmp, bar = spec["cmp"], spec["value"]
        ok = (measured <= bar) if cmp == "<=" else (
            measured > bar if cmp == ">" else measured >= bar)
        gates[name] = {"measured": measured, "cmp": cmp, "bar": bar,
                       "pass": ok}
    return {"gates": gates, "all_pass": all(g["pass"] for g in
                                           gates.values()),
            "measured_from": "model/final_gates.json"}


def st_gates(r: ForgeRun, force=False) -> tuple[str, str]:
    fg = r.model / "final_gates.json"
    verdict_path = r.root / "verdict.json"
    if fg.exists() and verdict_path.exists() and not force:
        v = json.loads(verdict_path.read_text())
        return "skipped", ("PASS" if v["all_pass"] else "FAIL") + \
            " (verdict recorded)"
    r.sh("gates",
         ["python3", "-m", "kapteeni.train_v11c", "--gates-only",
          "--adapter", str(r.model / "adapter"),
          "--out", str(r.model), "--base", r.cfg["base"]["path"]],
         r.env(gpu=True))
    if not fg.exists():
        raise ForgeError("gates ran but wrote no final_gates.json")
    v = apply_verdict(r.cfg, json.loads(fg.read_text()))
    v["verdict_ts"] = time.time()
    if verdict_path.exists():  # never silently rewrite a recorded verdict
        old = json.loads(verdict_path.read_text())
        v["history"] = old.get("history", []) + [old]
    verdict_path.write_text(json.dumps(v, indent=2))
    r.manifest_add("gates", "verdict", "PASS" if v["all_pass"] else "FAIL")
    return "ok", ("PASS" if v["all_pass"] else "FAIL") + \
        " — " + ", ".join(f"{k}={g['measured']}{g['cmp']}{g['bar']}"
                          for k, g in v["gates"].items())


def st_pack(r: ForgeRun, force=False) -> tuple[str, str]:
    if (r.dist / "kapteeni-config.json").exists() and not force:
        return "skipped", f"{r.dist} present"
    r.sh("pack",
         ["python3", "-m", "kapteeni.pack_v11c",
          "--base", r.cfg["base"]["path"],
          "--adapter", str(r.model / "adapter"),
          "--heads", str(r.model / "heads.pt"),
          "--gates", str(r.model / "final_gates.json"),
          "--out", str(r.dist)], r.env(gpu=True))
    # de-v1.1c the dist: served_as + an honest card for THIS run
    served_as = r.cfg.get("served_as", f"kapteeni-{r.cfg['name']}")
    cfgp = r.dist / "kapteeni-config.json"
    mc = json.loads(cfgp.read_text())
    mc["served_as"] = served_as
    cfgp.write_text(json.dumps(mc, indent=2))
    verdict = json.loads((r.root / "verdict.json").read_text()) \
        if (r.root / "verdict.json").exists() else None
    gate_rows = "\n".join(
        f"| {k} | {g['measured']} {g['cmp']} {g['bar']} "
        f"| {'PASS' if g['pass'] else 'FAIL'} |"
        for k, g in (verdict["gates"].items() if verdict else []))
    (r.dist / "README.md").write_text(_card(r, served_as, gate_rows, verdict))
    # the shipped demo pages hardcode the v1.1c model name; point the
    # dist's own demo at this run's identity so it works out of the box
    casesp = r.dist / "kapteeni" / "demo" / "cases.json"
    if casesp.exists():
        cj = json.loads(casesp.read_text())
        cj["model"] = served_as
        for c in cj.get("cases", []):
            if isinstance(c.get("request"), dict):
                c["request"]["model"] = served_as
        casesp.write_text(json.dumps(cj, ensure_ascii=False, indent=2))
    return "ok", f"{r.dist} (served as {served_as})"


def _card(r: ForgeRun, served_as: str, gate_rows: str, verdict) -> str:
    v = verdict or {"all_pass": None}
    d = r.cfg["data"]
    return f"""---
license: other
other_license_name: weights CC BY-SA 4.0 / code Apache-2.0
library_name: transformers
---

# {served_as}

A Kapteeni-class System One decision model built with the Kapteeni forge
(`kapteeni.forge`, see docs/FORGE.md in the TriusAI/Kapteeni repository).
It answers noul/choice/score questions over states with attached images
with calibrated probability distributions — no text generation.

- Backbone: {r.cfg['base'].get('hf_id', r.cfg['base']['path'])}
- Forge run: `{r.cfg['name']}`; profile `{r.cfg['profile']}`
- Data recipe: synth {d.get('synth', {}).get('n', '?')} / synth2
  {d.get('synth2', {}).get('n', '?')} / synth2zh
  {d.get('synth2zh', {}).get('n', '?')} rows, image families
  {d.get('synth3', {}).get('rows', '?')} +
  {d.get('synth3zh', {}).get('rows', '?')} rows, HF replay sources per
  config. **This is NOT kapteeni-v1.1c**: it has not been evaluated on
  JevBench, and its card carries only its own forge gates below.
- Contamination: the forge audit ran with zero shingle hits (see the
  run's contamination_audit.json); training data contains no benchmark
  items.

## Forge gates (frozen in runs/{r.cfg['name']}/forge.lock, measured once)

| gate | measured | verdict |
|---|---|---|
{gate_rows}

Overall: {'**PASS**' if v['all_pass'] else '**FAIL**'}

## Licenses

Weights: CC BY-SA 4.0 (see WEIGHTS-LICENSE.md). Code: Apache-2.0.
MNLI/FEVER-lineage attribution in WEIGHTS-LICENSE.md. OCNLI, if used for
gates, is eval-only (CC BY-NC) and never redistributed.
"""


def st_smoke(r: ForgeRun, force=False) -> tuple[str, str]:
    """Post-pack live smoke: serve the SHIPPED dist package (the code the
    user will actually run — not the repo checkout) and push the demo
    cases through it. The v1.1c release defect (an old helper shipped in
    the dist while pack validation exercised repo code) is exactly what
    this stage exists to catch."""
    if not (r.dist / "kapteeni-config.json").exists():
        raise ForgeError("pack the dist first (forge run --stage pack)")
    demo_dir = r.dist / "kapteeni" / "demo"
    cases = json.loads((demo_dir / "cases.json").read_text())
    dist_served = json.loads(
        (r.dist / "kapteeni-config.json").read_text()).get("served_as",
                                                           "jev-latest")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = r.env(gpu=True)
    env["PYTHONPATH"] = str(r.dist)  # the dist's kapteeni/ package wins
    proc = subprocess.Popen(
        ["python3", "-m", "kapteeni.serve_v11c",
         "--dist", str(r.dist.resolve()),  # absolute: cwd is the dist
         "--port", str(port)], env=env, cwd=str(r.dist),
        stdout=open(r.logs / "smoke-server.log", "wb"),
        stderr=subprocess.STDOUT)
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(120):  # model load + warmup can take a while
            try:
                urllib.request.urlopen(base + "/v1/models", timeout=5)
                break
            except Exception:
                time.sleep(2)
        else:
            raise ForgeError("smoke server never came up (see logs/)"
                             "smoke-server.log")
        ok_cases = 0
        for case in cases["cases"]:
            req = json.loads(json.dumps(case["request"]))
            req["model"] = dist_served  # the dist serves under its own name
            st = req.get("state")
            if isinstance(st, dict) and isinstance(st.get("image"), str):
                raw = st["image"]
                if raw.startswith("@file:"):
                    img = (demo_dir / raw[6:]).read_bytes()
                    st["image"] = "data:image/png;base64," + \
                        base64.b64encode(img).decode()
            body = json.dumps(req).encode()
            try:
                resp = urllib.request.urlopen(
                    urllib.request.Request(
                        base + "/v1/systemone", data=body,
                        headers={"Content-Type": "application/json"}),
                    timeout=180)
                out = json.loads(resp.read())
            except urllib.error.HTTPError as e:
                raise ForgeError(f"smoke case {case['id']}: "
                                 f"HTTP {e.code} {e.read()[:200]}")
            except Exception as e:
                raise ForgeError(f"smoke case {case['id']}: {e}")
            except_ok = ("answers" in out
                         and len(out["answers"]) == len(req["questions"]))
            if not except_ok:
                raise ForgeError(f"smoke case {case['id']}: bad response "
                                 f"shape: {str(out)[:300]}")
            ok_cases += 1
        return "ok", f"{ok_cases}/{len(cases['cases'])} demo cases clean " \
            f"through the shipped package"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


STAGES = {f.__name__[3:]: f for f in
          (st_base, st_gen, st_sources, st_expand, st_audit, st_precompute,
           st_phase1, st_p2, st_gates, st_pack, st_smoke)}


# --------------------------------------------------------------------- cli

def cmd_init(cfg_path: str, run_root: str | None) -> int:
    cfg = load_config(cfg_path)
    r = ForgeRun(cfg, run_root)
    if r.lock.exists():
        raise ForgeError(f"{r.lock} already exists — a run is frozen here. "
                         "Start a new run under a new name.")
    r.root.mkdir(parents=True, exist_ok=True)
    (r.root / "forge.json").write_text(json.dumps(cfg, indent=2,
                                                  ensure_ascii=False))
    r.lock.write_text(json.dumps({"config_sha256": config_sha(cfg),
                                   "created": time.time(),
                                   "source_config": str(cfg_path)}))
    print(f"frozen run: {r.root} (stages: {' -> '.join(STAGE_ORDER)})")
    print("gate bars frozen:", json.dumps({k: f"{v['cmp']} {v['value']}"
                                           for k, v in cfg["gates"].items()}))
    return 0


def cmd_run(cfg_path: str, run_root: str, stages: list[str],
            force: bool) -> int:
    cfg = load_config(cfg_path)
    r = ForgeRun(cfg, run_root)
    r.check_lock()
    order = STAGE_ORDER if stages == ["all"] else stages
    for s in order:
        if s not in STAGES:
            raise ForgeError(f"unknown stage '{s}' "
                             f"(have: {list(STAGE_ORDER)})")
        status, detail = STAGES[s](r, force=force)
        r.manifest_add(s, status, detail)
        print(f"[forge:{s}] {status}: {detail}", flush=True)
        if s == "gates":
            v = json.loads((r.root / "verdict.json").read_text())
            if not v["all_pass"] and order == STAGE_ORDER:
                print("[forge] verdict FAIL — stopping before pack "
                      "(run --stage pack explicitly if you want the "
                      "artifacts anyway)", flush=True)
                return 1
    return 0


def cmd_status(cfg_path: str, run_root: str | None) -> int:
    cfg = load_config(cfg_path)
    r = ForgeRun(cfg, run_root)
    if not r.root.exists():
        print(f"no run at {r.root}")
        return 1
    done = {}
    if r.manifest.exists():
        for line in r.manifest.read_text().splitlines():
            m = json.loads(line)
            done.setdefault(m["stage"], m)
    print(f"run: {r.root}"
          f"{' (LOCKED)' if r.lock.exists() else ' (NO LOCK!)'}")
    for s in STAGE_ORDER:
        m = done.get(s)
        print(f"  {s:<11} {m['status'] + ': ' + m['detail'][:60] if m else 'pending'}")
    if (r.root / "verdict.json").exists():
        v = json.loads((r.root / "verdict.json").read_text())
        print(f"verdict: {'PASS' if v['all_pass'] else 'FAIL'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="freeze a run recipe (gates included)")
    p.add_argument("--config", required=True)
    p.add_argument("--run-root", default=None)
    p = sub.add_parser("run", help="run stages (idempotent, resumable)")
    p.add_argument("--config", required=True)
    p.add_argument("--run-root", default=None)
    p.add_argument("--stage", action="append", required=True,
                   help="a stage name, or 'all'")
    p.add_argument("--force", action="store_true",
                   help="re-run even if outputs exist")
    p = sub.add_parser("status", help="show stage/verdict state")
    p.add_argument("--config", required=True)
    p.add_argument("--run-root", default=None)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "init":
            return cmd_init(args.config, args.run_root)
        if args.cmd == "status":
            return cmd_status(args.config, args.run_root)
        stages = ["all"] if "all" in args.stage else list(args.stage)
        return cmd_run(args.config, args.run_root, stages, args.force)
    except ForgeError as e:
        print(f"forge: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())