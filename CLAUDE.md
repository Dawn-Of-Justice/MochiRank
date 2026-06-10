# MochiRank — Redrob Track 1 Candidate Ranker

Rank 100K candidate profiles against a "Senior AI Engineer — Founding Team" JD and output
a top-100 submission CSV (`candidate_id, rank, score, reasoning`). Hackathon submission.

## Hard constraints (never violate)

- `rank.py` must run in **≤5 min wall-clock, ≤16 GB RAM, CPU only, ZERO network calls**
  (reproduced inside Docker). All heavy work is precomputed into `artifacts/`.
- **No LLM calls at ranking time.** Claude is a teacher/labeler in the offline phase only;
  its judgment is distilled into an XGBoost LambdaMART model.
- Eval metric: `0.50×NDCG@10 + 0.30×NDCG@50 + 0.15×MAP + 0.05×P@10` — optimize top-10 precision.
- Dataset contains traps: keyword stuffers, plain-language great candidates, behavioral twins,
  and ~80 honeypots (>10% honeypots in top-100 = disqualification → hard-gate them out).
- Reasoning strings must be SHAP-derived from the model's actual decision (no hallucination).

## Where things are

| Path | Contents |
|------|----------|
| `dataset/` | Provided bundle: `candidates.jsonl` (100K, ~487MB — never read whole), `candidate_schema.json`, `sample_candidates.json` (50 candidates — use for dev/testing), JD + spec docs (.docx), `validate_submission.py`, metadata template |
| `architecture/spec.md` | Pipeline overview, target repo layout, `rank.py` design |
| `architecture/features.md` | All ~48 features, skill/company lists, behavioral composite |
| `architecture/pipeline.md` | Consistency engine, disqualifiers, retrieval/RRF, reranker, SHAP reasoning |
| `docs/problem.md` | Problem framing, traps, evaluation mechanics |
| `docs/offline-phase.md` | Embeddings, hypothetical/anti-persona resumes, sampler, teacher labeling, training |
| `docs/build-plan.md` | Two-person workstream split (A: runtime `src/`+`rank.py`, B: offline ML), interface contracts, sync points, validation/ablation plan |
| `docs/submission.md` | Metadata declarations, open decisions/risks, research references |

Read only the doc relevant to the task — they are split precisely so you don't load all of them.

## Working conventions

- Code goes in `src/` (runtime, used by `rank.py`) vs `offline/` (precompute, internet OK) —
  keep the split strict; nothing in `src/` may import torch/anthropic at rank time if avoidable.
- Test against `dataset/sample_candidates.json` (50 candidates) before any 100K run.
- Run `dataset/validate_submission.py` on any generated submission CSV.
- Reference date for recency math: **2026-06-10**.
- Commit after each meaningful step (judges check for real git history).
- Follow the build order in `docs/build-plan.md`; start with `src/consistency_checks.py`.
