# The Kapteeni forge — build your own Kapteeni-class model

The forge (`kapteeni/forge.py`) chains the v1.1c pipeline — the same
stages, files, and gate machinery that produced the published
kapteeni-v1.1c — from one JSON recipe file. It exists so that:

1. **The shipped recipe is reproducible** (`forge/v11c-rebuild.json` is
   the v1.1c recipe, volumes and seeds as shipped), and
2. **Anyone can build their own variant** — different volumes, a
   different backbone, their own decision domains — with the
   project's discipline enforced by the tool instead of by memory.

## What it enforces

- **Pre-registration by lockfile.** `forge init` hashes the whole
  config — gate bars included — into `runs/<name>/forge.lock`. Every
  stage refuses to run if the config has drifted. Want different
  gates? That's a *new run under a new name*, never an edit.
- **One gates reading.** The gates stage applies the frozen bars to
  `final_gates.json` and writes an immutable `verdict.json`
  (re-runs append to a history array; nothing is silently rewritten).
- **Audit before train.** The contamination audit (8-token shingle
  overlap vs JevBench public items + OCNLI gate rows) runs on the
  run's own data; any hit stops the pipeline. Feedback-store surfaces
  are audited the same way (see `docs/FEEDBACK.md`).
- **Post-pack smoke.** The pack is served *from the dist* (the shipped
  code, not the repo checkout) and the demo cases are pushed through
  it — the exact class of defect the v1.1c release hit (a stale
  helper in the shipped package) is what this stage catches.

## Usage

```sh
# 1. write a recipe (see forge/*.json for the two examples)
python3 -m kapteeni.forge init --config forge/my-model.json

# 2. run it (detached: the p2 stage is hours)
setsid nohup python3 -m kapteeni.forge run \
    --config forge/my-model.json --stage all > /tmp/forge.log 2>&1 &

# 3. watch
python3 -m kapteeni.forge status --config forge/my-model.json
tail -f runs/my-model/logs/p2.log
```

Stages (`--stage` takes one or more, or `all`):

```
base -> gen -> sources -> expand -> audit -> precompute
     -> phase1 -> p2 -> gates -> pack -> smoke
```

Every stage is idempotent (existing outputs skip it) and the long ones
(precompute, p2) resume internally after a crash — relaunch `--stage
all` and it picks up.

## The recipe file

See `forge/v11c-rebuild.json` (the shipped recipe) and
`forge/micro-smoke.json` (a tiny end-to-end validation config — it
*will* fail its gates, which are the real bars; it exists to prove the
machinery, not to produce a good model).

| section | what it fixes |
|---|---|
| `name` / `served_as` | run identity; the dist serves under `served_as` |
| `base.path` / `base.hf_id` | local backbone dir; downloaded at the `base` stage if absent |
| `profile` | `halo-96g` (expandable segments, 8192-budget) or `cuda-24g` (4096-budget default) |
| `data.*` | generator volumes + seeds; HF row counts; distill (teacher soft labels); OCNLI source |
| `train.*` | lr, epochs, token budget, seed, step limits |
| `gates.*` | the six frozen bars (exactly these keys — the trainer only measures these) |
| `strict_audit` | fail vs warn when the jevbench clone is absent |

### Things the recipe cannot conjure

- **A teacher LLM** — `data.distill` (noul soft labels for
  boolq/fever) and criteria generation for the four choice sources
  call the teacher endpoint. Omit `distill` (gold-only noul passes)
  and use `data.criteria_from` to reuse the published rubric files,
  and the pipeline needs no teacher at all.
- **OCNLI dev.json** — CC BY-NC: you must obtain it yourself from
  CLUEbenchmark/OCNLI and point `data.ocnli_dev` at it. The ocnli
  gate is mandatory in the gate set, so there is no building without
  it; it is never fetched or redistributed here.
- **Fonts** — synth3/synth3zh render with DejaVu + Noto CJK from the
  system font dirs (see `kapteeni/synth3_gfx.py`).
- **The jevbench clone** — needed by the audit stage. Without it the
  run proceeds only with `strict_audit: false`, and the artifact is
  marked UNAUDITED.
- **Network** — the `sources` stage downloads HF datasets rows and
  (if needed) the backbone; set your proxy env (`HTTPS_PROXY=...`)
  for constrained boxes.

### What "your own model" means here

The forge automates the *mechanics*. The domain knowledge lives in
the generator families (`kapteeni/synth*.py`): the recipe as shipped
reproduces Kapteeni's decision domains (dates/deadlines, policy
chains, image families, bilingual). To teach the model *your* domains,
write a generator that emits rows in the same schema (see any
`synth*.py` + `tests/test_synth2.py` for the contract), add it to the
`gen` stage, and keep the discipline: deterministic seeds, val rows
held out by the `sha256(row_id) % 10` split, and the contamination
audit run against every surface you will ever train on.

## Isolation, and what is shared

Each run keeps its own `runs/<name>/data_cache` (via the
`KAPTEENI_DATA_DIR` env var the stage modules honor), its own
`model_cache`, dist, logs, verdict, and manifest. The repo checkout's
`data_cache/` (the shipped pipeline's cache) is never touched by a
forge run. One recipe per checkout at a time is the practical limit —
stages read whole files, not diffs.

## Publish checklist (after gates PASS)

1. `runs/<name>/dist/` is a self-contained pack (merged model, heads,
   config, card, licenses, the serving package with its own demo).
2. The forge card is honest by construction: it names the run, shows
   the frozen bars and the measured gates, and states the model has
   NOT been evaluated on JevBench. If you benchmark it, record the
   numbers with their caveats — numbers only, no placement claims.
3. Upload: `huggingface-cli upload <org>/<name> runs/<name>/dist .`
   then pin the revision everywhere you cite it (see
   `docs/PUBLISH-HF.md` for the protocol).