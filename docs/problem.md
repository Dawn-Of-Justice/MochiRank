# Problem & Evaluation Mechanics

> Part of the MochiRank docs. See [architecture/spec.md](../architecture/spec.md) for the system design.

## 1. Problem Restatement (Read Between the Lines)

The JD is a 2000-word essay for **Senior AI Engineer — Founding Team** at Redrob AI.
The hidden note at the bottom of the JD (for hackathon participants) is the real brief:

> "The right answer involves reasoning about the **gap between what the JD says and what the JD means**.
> A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' but if their career shows they built
> a recommendation system at a product company, they're a fit. A perfect-on-paper candidate who hasn't
> logged in for 6 months and has a 5% recruiter response rate is not actually available."

Four trap types are explicitly built into the 100K dataset:
- **Keyword stuffers** — skills list has every AI keyword, but career history doesn't support them
- **Plain-language Tier 5s** — great candidate, zero buzzwords in their profile
- **Behavioral twins** — near-identical profiles, differ only in `redrob_signals`
- **~80 honeypots** — internally inconsistent profiles (>10% honeypot rate in top-100 = disqualification)

These traps map directly to architectural decisions. Pure embedding cosine similarity fails all four.

## 2. Evaluation Mechanics (What Actually Scores Points)

```
composite = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```

**50% of the score is the top 10.** Find the 5–10 genuinely great candidates out of 100K.
Volume of mediocre-middle candidates barely moves the needle. Optimize for precision at the top.

**Stage 3 reproduction kills most submissions.** Must reproduce inside Docker: 5 min wall-clock,
16 GB RAM, CPU only, zero network. Plan for this from day one.

**Stage 4 reasoning checks** (if you reach top-N) sample 10 rows and check for:
specificity, JD connection, honest concerns, no hallucination, variation, rank-consistency.
If reasoning is mechanically derived from the model's actual decision, all 6 checks pass automatically.
