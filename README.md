# [MochiRank - Intelligent Candidate Ranker](https://youtu.be/aUBgT57dzzA)

Ranks **100,000 candidate profiles** against a job description and produces a ranked top-100 list
(`candidate_id, rank, score, reasoning`). Designed to surface genuinely qualified candidates, not
just keyword-matched ones.

## What it does

Given a large pool of candidate profiles and a job description, MochiRank identifies the best-fit
candidates using a two-phase approach:

- **Offline (precompute):** An LLM acts as a labeling workforce, scoring a stratified sample of
  ~2,500 candidates against a calibrated rubric. Those judgments train an **XGBoost LambdaMART**
  ranker that learns what "great fit" actually means for the role — including signals that don't
  show up as literal keyword matches.

- **At ranking time:** `rank.py` runs a pure-CPU pipeline with no LLM or network calls. It builds
  the dense and BM25 indexes **from the uploaded candidates file itself** (so any dataset the judges
  supply — including unseen `candidate_id`s — ranks correctly), filters out structurally
  inconsistent profiles, retrieves the most relevant candidates via hybrid BM25 + dense search with
  RRF fusion, scores them with the trained model across ~48 features, and generates SHAP-derived
  reasoning strings tied to the model's actual decisions.

```
OFFLINE (internet OK)                      RANKING (≤5 min, CPU, no network)
embeddings → hypothetical resumes →        rank.py: honeypot filter → hybrid retrieval
sampler → LLM teacher labels →             → features → XGBoost → re-rank → hard gates
XGBoost LambdaMART → artifacts/            → SHAP reasoning → ranked CSV
```

## Why not just use embeddings?

Pure cosine similarity over profile embeddings fails several real-world traps:

- **Keyword stuffers** — skills list has every buzzword, but career history doesn't support them
- **Plain-language great candidates** — strong fit, zero industry jargon in their profile
- **Behavioral twins** — near-identical profiles that differ only in engagement signals
- **Inconsistent profiles** — internally contradictory data that looks plausible at a glance

MochiRank addresses all four: consistency checks gate out impossible profiles, the feature matrix
captures behavioral signals and career trajectory (not just text overlap), and the LambdaMART
ranker is trained to distinguish genuine fit from surface-level matches.

## Repository layout

| Path | Contents |
|------|----------|
| `rank.py` | Single entry point — produces submission CSV |
| `src/` | Runtime modules imported by `rank.py` (no torch/anthropic) |
| `offline/` | Precompute scripts — embeddings, hypothetical resumes, teacher labeling, training |
| `artifacts/` | Dataset-independent precomputed outputs — trained model, JD query vectors, hypothetical resumes, and the vendored `potion-base-8M/` embedder. (Dense embeddings + BM25 are built at runtime from the uploaded candidates.) |
| `dataset/` | Input data: `candidates.jsonl` (~487 MB), 50-candidate sample, JD, schema, validator |
| `architecture/` | System design: [spec.md](architecture/spec.md), [features.md](architecture/features.md), [pipeline.md](architecture/pipeline.md) |
| `docs/` | [problem.md](docs/problem.md), [offline-phase.md](docs/offline-phase.md), [build-plan.md](docs/build-plan.md) |
| `requirements.txt` | Python dependencies |

## Pipeline stages

`rank.py` runs these stages in sequence:

1. **Consistency engine** — flags honeypot and impossible profiles across all 100K candidates
2. **Hybrid retrieval** — dense (model2vec) + BM25 indexes built at runtime from the uploaded
   candidates, fused with Reciprocal Rank Fusion → top ~2,000
3. **Feature matrix** — ~48 features per candidate (skill depth, recency, behavioral signals, JD alignment)
4. **XGBoost inference** — LambdaMART model scores the top-2,000
5. **Cross-encoder re-rank** — optional ONNX int8 cross-encoder refines the top 200
6. **Hard gates** — honeypots and disqualifiers are removed from the final top-100
7. **SHAP reasoning** — each output row gets an explanation derived from the model's actual feature weights

## Getting started

```powershell
pip install -r requirements.txt
```

Develop and test against the 50-candidate sample — never load the full 487 MB file during iteration:

```powershell
# Validate any generated submission:
python dataset/validate_submission.py submission.csv
```

### Reproduce command

```
python rank.py --candidates ./dataset/candidates.jsonl --out ./submission.csv
```

**Runtime constraints:** ≤5 min wall-clock, ≤16 GB RAM, CPU only, zero network calls.
(Full 100K reference run: ~54 s on CPU, including ~30 s to build the dense + BM25
indexes from the input file.)

### Reproduce in Docker (network-isolated)

The image installs only `requirements-runtime.txt` (no torch/anthropic) and bakes in
the vendored `potion-base-8M/` embedder, so the run is provably network-free:

```bash
docker build -t mochirank .
docker run --rm --network none \
    -v "$PWD/dataset:/data:ro" -v "$PWD/out:/out" \
    mochirank --candidates /data/candidates.jsonl --out /out/submission.csv
```

`--network none` is the actual proof of the zero-network constraint — the run
succeeds because nothing reaches out: the embedder loads from the baked-in
`artifacts/potion-base-8M/`, and the dense + BM25 indexes are rebuilt in-process
from the mounted candidates file.

## Design decisions

- **LLM as labeler, not ranker.** The LLM's judgments are distilled into XGBoost weights during
  training. At inference time, `rank.py` is pure numpy/scipy/xgboost — fast, reproducible,
  and fully offline.
- **SHAP for reasoning.** Every reasoning string is derived mechanically from the model's feature
  contributions, ensuring specificity and consistency with the actual ranking decision.
- **Strict offline/runtime split.** Nothing in `src/` imports torch or anthropic. The trained
  model and JD-side vectors are precomputed in `artifacts/`; the dense + BM25 indexes are rebuilt
  at runtime from the uploaded candidates (via the torch-free model2vec embedder), which is what
  lets the ranker handle any judge dataset rather than only the one it was precomputed against.
- **Reference date: 2026-06-10** — used for all recency calculations (years of experience, activity
  recency, etc.).
