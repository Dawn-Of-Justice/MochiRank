# MochiRank — Redrob Track 1: Intelligent Candidate Ranker

Ranks **100,000 candidate profiles** against a "Senior AI Engineer — Founding Team" JD and
produces a top-100 submission CSV (`candidate_id, rank, score, reasoning`).
Built for the Redrob India Runs hackathon, Track 1.

## How it works (30-second version)

Claude is used **offline only** as a teacher: it labels ~2,500 stratified candidates with a
calibrated rubric, and those judgments are distilled into an **XGBoost LambdaMART** ranker.
At ranking time, `rank.py` runs a pure-CPU pipeline — consistency/honeypot gating, hybrid
BM25 + dense retrieval with RRF fusion, a ~48-feature matrix, XGBoost inference, optional
ONNX cross-encoder re-rank, and SHAP-derived reasoning strings — with **zero LLM/network calls**,
in under 5 minutes and 16 GB RAM.

```
OFFLINE (internet OK)                      RANKING STEP (≤5 min, CPU, no network)
embeddings → hypothetical resumes →        rank.py: honeypot filter → hybrid retrieval
sampler → Claude teacher labels →          → features → XGBoost → re-rank → hard gates
XGBoost LambdaMART → artifacts/            → SHAP reasoning → submission.csv
```

## Repository layout

| Path | What's in it |
|------|--------------|
| `CLAUDE.md` | Project context + hard constraints for Claude Code sessions |
| `architecture/` | System design: [spec.md](architecture/spec.md) (pipeline + `rank.py`), [features.md](architecture/features.md) (~48 features), [pipeline.md](architecture/pipeline.md) (stage implementations) |
| `docs/` | [problem.md](docs/problem.md) (framing + eval), [offline-phase.md](docs/offline-phase.md) (teacher labeling + training), [build-plan.md](docs/build-plan.md) (build order + ablations), [submission.md](docs/submission.md) (declarations, risks, references) |
| `dataset/` | Provided hackathon bundle: `candidates.jsonl` (~487 MB, not committed), 50-candidate sample, JD, schema, `validate_submission.py` |
| `requirements.txt` | Python dependencies |
| `src/` *(planned)* | Runtime code imported by `rank.py` — no torch/anthropic |
| `offline/` *(planned)* | Precompute scripts — internet + API calls allowed |
| `artifacts/` *(planned)* | Precomputed embeddings, BM25 index, trained model |
| `rank.py` *(planned)* | The single entry point |

## Getting started (contributors)

1. Read [docs/problem.md](docs/problem.md) first — the traps in the dataset drive every design decision.
2. Then [architecture/spec.md](architecture/spec.md) for the big picture; open the other docs only as needed.
3. Follow [docs/build-plan.md](docs/build-plan.md) for the build order — first deliverable is `src/consistency_checks.py`.

```powershell
pip install -r requirements.txt

# Develop against the 50-candidate sample, never the full 487 MB file:
#   dataset/sample_candidates.json

# Validate any generated submission before upload:
python dataset/validate_submission.py submission.csv
```

### Planned reproduce command

```
python rank.py --candidates ./dataset/candidates.jsonl --out ./submission.csv
```

## Hard constraints (do not break)

- `rank.py`: ≤5 min wall-clock, ≤16 GB RAM, CPU only, **zero network calls** (Docker-reproduced).
- No LLM calls at ranking time — all Claude usage is offline (teacher labeling).
- Eval: `0.50×NDCG@10 + 0.30×NDCG@50 + 0.15×MAP + 0.05×P@10` — top-10 precision is half the score.
- >10% honeypots in the top-100 = disqualification — honeypots are hard-gated out.
- Reasoning strings must be derived from the model's actual decision (SHAP), never free-written.
- Reference date for all recency math: **2026-06-10**.

## Status

Docs/spec phase complete. No code yet — next step per the build plan is `src/consistency_checks.py`.
