import time
from pathlib import Path
from src.feature_engineering import load_precomputed, compute_features, FEATURE_NAMES
from src.utils import load_candidates_json
from src.consistency_checks import check_consistency

pre = load_precomputed(Path("artifacts"))
print("cid_to_rows:", len(pre.get("cid_to_rows", {})), "unique candidates")
print("emb shape:", pre["candidate_embeddings"].shape)
print("jd_vecs shape:", pre["jd_query_vectors"].shape)

candidates = load_candidates_json("dataset/sample_candidates.json")
t0 = time.time()
errors = 0
for c in candidates:
    is_hp, n_v, _ = check_consistency(c)
    fv = compute_features(c, pre, n_v, is_hp)
    if len(fv) != 48:
        errors += 1
t1 = time.time()
last_fv = fv
print(f"50 candidates: {t1-t0:.3f}s  errors={errors}  dim={len(last_fv)}")
print(f"Extrapolated 100K: {(t1-t0)/50*100000:.1f}s ({(t1-t0)/50*100000/60:.1f} min)")
