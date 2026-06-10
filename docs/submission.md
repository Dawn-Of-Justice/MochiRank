# Submission, Risks & References

> Metadata declarations, open decisions, and the research backing the design.
> See [build-plan.md](build-plan.md) for the submission-day checklist.

## Metadata + Declarations

**ai_usage_summary (for submission_metadata.yaml):**
```
Used Claude (Anthropic API, claude-haiku-4-5 model) as a teacher model to generate
calibrated relevance labels (0.0–1.0) for ~2500 stratified candidates using a
Malt-inspired semantic rubric. These labels were used to train an XGBoost LambdaMART
model. Claude was also used for architectural discussion and code review.
No candidate data is fed to any LLM during the ranking step (rank.py makes zero API calls).
Used Claude (claude.ai) for initial architecture design and literature review.
```

**honeypot_check_done:** set to `true` in metadata — the consistency engine explicitly detects them.

## Open Decisions / Risks

| Decision | Options | Recommendation |
|----------|---------|----------------|
| Primary embedding model | bge-small-en-v1.5 vs potion-base-8M | bge-small primary; potion as fallback |
| Teacher model | claude-haiku vs claude-sonnet-4-6 | haiku (cost); sonnet for final 500 |
| Label count | 1500 vs 2500 vs 5000 | 2500 sweet spot; more = better up to ~5K |
| Cross-encoder | MiniLM-L6 ONNX vs FlashRank vs skip | MiniLM-L6 ONNX (best precision/speed) |
| Reasoning | SHAP vs rule templates | SHAP (Stage 4 proof) |
| Sandbox platform | HF Spaces vs Streamlit Cloud | HF Spaces (more reliable, free tier) |

**Main risk:** teacher label noise from Claude. Mitigate with:
- Semantic rubric anchoring (fixed scale in every prompt)
- Anchored batching (always include 1 perfect + 1 terrible candidate per batch)
- Self-consistency check (double-label 100, Pearson r should be ≥ 0.85)
- Monotonic constraints in XGBoost (behavioral signals can only help, not hurt)

## Key Research References

- **PJFNN** (Zhu et al. 2018) — bipartite neural network for person-job fit, established the field
- **ConFit v2** (Yu et al., ACL Findings 2025) — Hypothetical Resume Embedding (+17.5% nDCG),
  Runner-Up Hard-Negative Mining. `arxiv:2502.12361`
- **Malt ranking distillation** (Jouanneau et al. 2026) — LLM-as-teacher for person-job fit,
  semantic rubric anchoring, anchored batching. `arxiv:2601.10321`
- **LinkedIn LTR** (Ha-Thuc et al. SIGIR 2016) — production talent search with learning-to-rank
- **LinkedIn deep LTR** (Ramanath et al. 2018) — embedding features in LTR pipeline
- **RRF** (Cormack et al. 2009) — Reciprocal Rank Fusion, k=60, score-normalization-free fusion
- **Hybrid BM25+dense** — empirically: +8.1pp Recall@5 vs BM25 alone (`arxiv:2604.01733`)
- **Model2Vec** — 500x faster than sentence-transformers on CPU, ~30MB, numpy-only
