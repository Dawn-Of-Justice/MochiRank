"""
B2 — Generate 5 ideal + 5 anti-persona resumes via Claude, then embed them.

Outputs:
  artifacts/hypothetical_resumes.json  — resumes + is_positive flag + jd_text
  artifacts/jd_query_vectors.npy       — float16, shape (10, 384)

Usage:
  python -m offline.generate_hypothetical
  ANTHROPIC_API_KEY must be set.

IMPORTANT: Manually review the generated resumes before proceeding to B3.
They must read as human-written; regenerate any that look AI-obvious.
"""

import json
import re
from pathlib import Path

import numpy as np

ARTIFACTS = Path("artifacts")
DATASET = Path("dataset")

# ------------------------------------------------------------------ #
# JD loading
# ------------------------------------------------------------------ #

def read_jd_text() -> str:
    try:
        import docx
        doc = docx.Document(DATASET / "job_description.docx")
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        print(f"[warn] docx read failed ({e}), using embedded fallback JD")
        return _JD_FALLBACK

_JD_FALLBACK = """
Senior AI Engineer — Founding Team

We are building an AI-powered recruitment intelligence platform.
Requirements:
- 5-9 years of industry experience in ML/AI
- Production retrieval and ranking systems (dense embeddings, BM25, hybrid search)
- Vector databases: FAISS, Pinecone, Weaviate, Elasticsearch
- Learning-to-rank: XGBoost LambdaMART, NDCG, MRR, A/B testing, offline eval
- LLM experience: fine-tuning (LoRA, QLoRA), PEFT, RAG pipelines
- Startup / product company background (not consulting)
- Python expert; engineering rigour, not just notebooks
- Location: Pune / Noida / Hyderabad / Mumbai / Delhi / Bangalore; hybrid or onsite
Explicitly NOT wanted:
- Candidates from pure IT services / consulting background (TCS, Infosys, Wipro, etc.)
- Keyword stuffers with no production deployment history
- Pure academics with no industry shipping experience
"""

# ------------------------------------------------------------------ #
# Prompts
# ------------------------------------------------------------------ #

_IDEAL_PROMPT = """You are generating realistic candidate profiles for the following job description.
Create profiles that are STRONG FITs — real people with consistent career history,
specific accomplishments, and natural language (no buzzword stuffing).

Generate exactly {n} profiles, one per archetype:
1. IR veteran — 8yr, built search/ranking at product company pre-LLM era, now adding modern ML
2. Startup ML shipper — 6yr, 2-3 startups, shipped RAG / rec-sys to real users, scrappy
3. Platform engineer — 7yr, vector DB + hybrid search infra, scale-focused
4. Applied researcher — 5yr MSc/PhD in industry, eval frameworks, A/B testing mindset
5. Product-ML hybrid — 6yr, ex-PM turned engineer, retrieval + ranking + product instincts

For each profile output a JSON object with these exact keys:
  id, archetype, headline (1 line), summary (3-4 sentences, natural prose),
  roles (array of objects: title, company_type, duration_months, description 60-100 words),
  skills (array of objects: name, years_experience)

JOB DESCRIPTION:
{jd_text}

Return a JSON array of {n} profile objects. No extra text.
"""

_ANTI_PROMPT = """You are generating candidate profiles that appear relevant on the surface
but fail the following job description's actual requirements.

Generate exactly {n} profiles, one per anti-archetype:
1. Keyword stuffer — Marketing Manager with every AI keyword in skills list, career is pure marketing
2. Pure researcher — academic lab career (PhD → postdoc), never shipped to production users
3. Consulting lifer — entire career at TCS / Infosys / Wipro / Accenture, no product company
4. Framework enthusiast — only LangChain / OpenAI API wrapper projects, zero pre-LLM ML experience
5. Title chaser — avg tenure < 18 months across 5+ jobs, collecting "Senior" → "Staff" promotions

Each profile should look superficially plausible but fail the actual JD requirements.

For each profile output a JSON object with these exact keys:
  id, archetype, headline (1 line), summary (3-4 sentences),
  roles (array: title, company_type, duration_months, description 60-100 words),
  skills (array: name, years_experience)

JOB DESCRIPTION:
{jd_text}

Return a JSON array of {n} profile objects. No extra text.
"""

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _extract_json_array(text: str) -> list:
    text = re.sub(r"```(?:json)?\n?", "", text).strip()
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON array in response:\n{text[:300]}")
    return json.loads(text[start:end])


def _resume_to_text(r: dict) -> str:
    parts = [r.get("headline", ""), r.get("summary", "")]
    for role in r.get("roles", []):
        if isinstance(role, dict):
            parts.append(
                f"{role.get('title', '')} at {role.get('company_type', '')}: "
                f"{role.get('description', '')}"
            )
    for s in r.get("skills", []):
        if isinstance(s, dict):
            parts.append(s.get("name", ""))
        else:
            parts.append(str(s))
    return " ".join(p for p in parts if p)

# ------------------------------------------------------------------ #
# Generation
# ------------------------------------------------------------------ #

def generate_resumes(jd_text: str, n_ideal: int = 5, n_anti: int = 5) -> list[dict]:
    import anthropic
    client = anthropic.Anthropic()

    print("Generating ideal resumes...")
    ideal_raw = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{"role": "user",
                   "content": _IDEAL_PROMPT.format(n=n_ideal, jd_text=jd_text)}],
    ).content[0].text
    ideals = _extract_json_array(ideal_raw)
    for r in ideals:
        r["is_positive"] = True

    print("Generating anti-persona resumes...")
    anti_raw = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{"role": "user",
                   "content": _ANTI_PROMPT.format(n=n_anti, jd_text=jd_text)}],
    ).content[0].text
    antis = _extract_json_array(anti_raw)
    for r in antis:
        r["is_positive"] = False

    print(f"Generated {len(ideals)} ideal + {len(antis)} anti-persona resumes")
    return ideals + antis


def embed_resumes(resumes: list[dict]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    texts = [_resume_to_text(r) for r in resumes]
    return model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float16)

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)

    jd_text = read_jd_text()
    print(f"JD text: {len(jd_text)} chars")

    resumes = generate_resumes(jd_text)

    resumes_path = ARTIFACTS / "hypothetical_resumes.json"
    with open(resumes_path, "w") as f:
        json.dump({"resumes": resumes, "jd_text": jd_text}, f, indent=2)
    print(f"Saved {resumes_path}")

    print("Embedding resumes...")
    vectors = embed_resumes(resumes)
    vectors_path = ARTIFACTS / "jd_query_vectors.npy"
    np.save(vectors_path, vectors)
    print(f"Saved {vectors_path}  shape={vectors.shape}")

    print("\n=== MANUAL REVIEW REQUIRED ===")
    for r in resumes:
        sign = "IDEAL" if r.get("is_positive") else "ANTI "
        print(f"  [{sign}] {r.get('archetype', '?')}: {r.get('headline', '')[:80]}")
    print("\nCheck: do these read as human-written? Regenerate any that look AI-obvious.")


if __name__ == "__main__":
    main()
