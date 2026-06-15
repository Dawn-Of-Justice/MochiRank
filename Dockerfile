# Reproducible rank-time image for MochiRank.
#
# Verifies the hard constraints: CPU-only, ≤16 GB RAM, ≤5 min, ZERO network.
# Network isolation is proven at RUN time, not build time:
#
#   docker build -t mochirank .
#   docker run --rm --network none \
#       -v "$PWD/dataset:/data:ro" -v "$PWD/out:/out" \
#       mochirank --candidates /data/candidates.jsonl --out /out/submission.csv
#
# --network none guarantees rank.py makes no network calls; it passes because the
# model2vec embedder is vendored into artifacts/potion-base-8M/ and loaded locally.

FROM python:3.10-slim

WORKDIR /app

# Runtime-only deps (no torch/anthropic/streamlit — those are offline-phase only).
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Code + the dataset-independent artifacts needed at rank time. The large
# offline-only artifacts (candidate_embeddings.npy, bm25_index.pkl, candidate_ids.json)
# and the 487 MB dataset are excluded via .dockerignore — rank.py rebuilds the
# dense + BM25 indexes at runtime from whatever candidates file is mounted in.
COPY src/ ./src/
COPY rank.py .
COPY artifacts/ ./artifacts/

ENTRYPOINT ["python", "rank.py"]
