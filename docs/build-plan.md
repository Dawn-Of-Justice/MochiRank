# Build Plan & Local Validation

> The day-by-day build order and how to evaluate without a leaderboard.
> Design background: [architecture/spec.md](../architecture/spec.md).

## Build Order (Recommended Sequence)

### Day 1 — Foundation
1. `src/consistency_checks.py` — pure logic, testable immediately against sample_candidates.json
2. `src/utils.py` — company lists, skill lists, parsing helpers
3. Sanity-check consistency engine against all 50 sample candidates

### Day 2 — Offline Precompute
4. `offline/precompute_embeddings.py` — embed 100K, save .npy
5. `offline/generate_hypothetical.py` — generate 5 ideal + 5 anti-persona resumes via Claude
6. Manually review the generated resumes — they need to be realistic, not AI-obvious

### Day 3 — Feature Engineering + Retrieval
7. `src/feature_engineering.py` — implement all 47 features, unit test each group
8. `offline/build_bm25_index.py` — build and pickle BM25 index
9. `src/retrieval.py` — hybrid retrieval + RRF
10. Run end-to-end on 50-candidate sample, check output makes sense

### Day 4 — Teacher Labeling + Training
11. `offline/stratified_sampler.py` — pull 2500 candidates
12. `offline/teacher_label.py` — call Claude API, collect labels, check self-consistency
13. `offline/train_ranker.py` — train XGBoost, validate NDCG@10 locally, run ablations
14. Save `artifacts/ranker_model.json`

### Day 5 — Integration + Reasoning
15. `src/reasoning_generator.py` — SHAP-based reasoning
16. `rank.py` — full pipeline integration
17. Run on 50-sample, verify output format, run `validate_submission.py`

### Day 6 — Submission Infrastructure
18. `sandbox/app.py` — Streamlit app for HF Spaces (accepts ≤100 candidates, returns CSV)
19. Full run on 100K candidates, measure wall-clock time
20. Tune if >4 min (disable cross-encoder, reduce to top-100 rerank)
21. Git history must show real commits — commit after each step above

### Day 7 — Polish + Submit
22. Fill `submission_metadata.yaml`
23. Write `README.md` with setup + reproduce_command
24. Run `validate_submission.py` — must print "Submission is valid."
25. Submit via portal (3 submissions max — save last 2 for corrections)

## Local Validation Approach

No live leaderboard. Build your own eval.

**Step 1:** Hand-label the 50 sample candidates into tiers 0–5 (use Claude to assist,
then review yourself). That's your mini ground truth.

**Step 2:** Run your ranker on the 50. Compute NDCG@10 against your labels.

**Step 3:** Ablations — remove one component at a time, measure NDCG@10 drop:
```
baseline:                  NDCG@10 = X.XX
- no behavioral composite: NDCG@10 = ?  (should drop — behavioral twins test)
- no anti-persona:         NDCG@10 = ?  (should drop — stuffer test)
- no consistency filter:   NDCG@10 = ?  (should drop — honeypot test)
- only BM25 retrieval:     NDCG@10 = ?  (should drop — plain-language Tier 5 test)
- only dense retrieval:    NDCG@10 = ?  (may drop on exact-term queries)
```

Save this table — it's your Stage 4 methodology summary and your Stage 5 interview script.
