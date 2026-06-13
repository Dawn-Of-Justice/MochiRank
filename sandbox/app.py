"""
MochiRank — Streamlit demo app.
Upload candidates as JSON, get a ranked CSV with reasoning back.

Run:  C:/Users/udaya/AppData/Local/Programs/Python/Python311/python.exe -m streamlit run sandbox/app.py
"""

import csv
import io
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

ARTIFACTS = Path(__file__).parent.parent / "artifacts"

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="MochiRank",
    page_icon="🍡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
# Global styles
# ------------------------------------------------------------------ #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Base & font ──────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background-color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        color: #1a1a2e !important;
    }

    /* ── Hide Streamlit chrome ────────────────── */
    #MainMenu                    { visibility: hidden; }
    footer                       { visibility: hidden; }
    [data-testid="stToolbar"]    { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    /* Keep header visible for sidebar toggle, but make it blend in */
    header[data-testid="stHeader"] {
        background: transparent !important;
        box-shadow: none !important;
    }
    /* Always show the collapsed-sidebar expand button */
    [data-testid="collapsedControl"] {
        display: block !important;
        visibility: visible !important;
    }

    /* ── Sidebar ──────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #f0f2f6 !important;
        border-right: 1px solid #e2e4e9 !important;
        padding-top: 1rem !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li {
        color: #4a5568 !important;
        font-size: 0.84rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #dde1e7 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
        margin-bottom: 0.5rem !important;
        overflow: hidden !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
        background: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpanderToggleIcon"] {
        color: #94a3b8 !important;
    }

    /* ── Primary button ───────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #c62a35 0%, #e63946 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.58rem 1.8rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 2px 10px rgba(230,57,70,0.25) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 5px 16px rgba(230,57,70,0.35) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* ── Download button ──────────────────────── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1a6b3c 0%, #22863a 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.58rem 1.8rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        font-family: 'Inter', sans-serif !important;
        box-shadow: 0 2px 10px rgba(34,134,58,0.25) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 5px 16px rgba(34,134,58,0.35) !important;
    }

    /* ── File uploader ────────────────────────── */
    [data-testid="stFileUploader"] {
        background: #fafafa !important;
        border: 2px dashed #d1d5db !important;
        border-radius: 12px !important;
        padding: 0.5rem 1rem !important;
        transition: border-color 0.2s ease !important;
    }
    [data-testid="stFileUploader"]:hover,
    [data-testid="stFileUploader"]:focus-within {
        border-color: #e63946 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] { color: #9ca3af !important; }

    /* ── Alerts ───────────────────────────────── */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
        font-size: 0.88rem !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Progress bar ─────────────────────────── */
    [data-testid="stProgress"] > div > div > div > div {
        background: linear-gradient(90deg, #e63946, #f87171) !important;
        border-radius: 6px !important;
    }
    [data-testid="stProgress"] > div > div > div {
        background: #e5e7eb !important;
        border-radius: 6px !important;
    }

    /* ── Tabs — underline style ───────────────── */
    [data-baseweb="tab-list"] {
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid #e5e7eb !important;
        border-radius: 0 !important;
        padding: 0 !important;
        gap: 0.25rem !important;
        margin-bottom: 1.25rem !important;
    }
    [data-baseweb="tab"] {
        background: transparent !important;
        color: #6b7280 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.65rem 1.25rem !important;
        border-radius: 0 !important;
        border-bottom: 2px solid transparent !important;
        margin-bottom: -2px !important;
        transition: color 0.15s ease !important;
    }
    [data-baseweb="tab"]:hover { color: #1a1a2e !important; }
    [aria-selected="true"][data-baseweb="tab"] {
        background: transparent !important;
        color: #e63946 !important;
        border-bottom: 2px solid #e63946 !important;
        font-weight: 600 !important;
    }
    [data-baseweb="tab-highlight"] { display: none !important; }
    [data-baseweb="tab-border"]    { display: none !important; }

    /* ── Expander (main content) ──────────────── */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
        overflow: hidden !important;
    }
    [data-testid="stExpanderDetails"] {
        background: #ffffff !important;
        padding-top: 0.5rem !important;
    }

    /* ── Dataframe ────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05) !important;
    }

    /* ── Code & monospace ─────────────────────── */
    code {
        font-family: 'JetBrains Mono', monospace !important;
        background: #f3f4f6 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 5px !important;
        padding: 0.15rem 0.45rem !important;
        color: #e63946 !important;
        font-size: 0.82em !important;
    }
    pre code {
        border-radius: 0 !important;
        padding: 0 !important;
        border: none !important;
        color: inherit !important;
    }
    [data-testid="stCodeBlock"] {
        background: #f8f9fa !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    [data-testid="stCodeBlock"] * {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.83rem !important;
    }

    /* ── Markdown text ────────────────────────── */
    .stMarkdown p, .stMarkdown li { color: #374151 !important; line-height: 1.65 !important; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #1a1a2e !important; font-weight: 700 !important; }
    .stMarkdown strong { color: #1a1a2e !important; }
    .stMarkdown a      { color: #e63946 !important; }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #9ca3af !important;
        font-size: 0.78rem !important;
    }
    hr { border-color: #e5e7eb !important; }

    /* ── Selectbox ────────────────────────────── */
    [data-testid="stSelectbox"] > div > div {
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
        color: #1a1a2e !important;
    }

    /* ── Section headings ─────────────────────── */
    .section-heading {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-top: 2.25rem;
        margin-bottom: 0.9rem;
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }

    /* ── Hero banner ──────────────────────────── */
    .hero {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 55%, #e0f2fe 100%);
        border: 1px solid #bfdbfe;
        border-radius: 18px;
        padding: 2.5rem 3rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 4px 20px rgba(59,130,246,0.08);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 250px; height: 250px;
        background: radial-gradient(circle, rgba(230,57,70,0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero h1 {
        color: #1a1a2e;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.35rem;
        letter-spacing: -0.03em;
        font-family: 'Inter', sans-serif;
    }
    .hero p {
        color: #4b5563;
        font-size: 0.95rem;
        margin: 0;
        font-family: 'Inter', sans-serif;
    }
    .hero .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(230,57,70,0.08);
        color: #e63946;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        border: 1px solid rgba(230,57,70,0.2);
        margin-bottom: 0.9rem;
        letter-spacing: 0.1em;
        font-family: 'Inter', sans-serif;
    }
    .hero .badge .dot {
        width: 5px; height: 5px;
        border-radius: 50%;
        background: #22c55e;
    }

    /* ── Metric cards ─────────────────────────── */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 12px 12px 0 0;
    }
    .metric-card.blue::before  { background: linear-gradient(90deg, #2563eb, #60a5fa); }
    .metric-card.red::before   { background: linear-gradient(90deg, #dc2626, #f87171); }
    .metric-card.green::before { background: linear-gradient(90deg, #16a34a, #4ade80); }
    .metric-card .label {
        color: #6b7280;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.4rem;
        font-family: 'Inter', sans-serif;
    }
    .metric-card .value {
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.3rem;
        letter-spacing: -0.04em;
        font-family: 'Inter', sans-serif;
    }
    .metric-card.blue .value  { color: #2563eb; }
    .metric-card.red .value   { color: #dc2626; }
    .metric-card.green .value { color: #16a34a; }
    .metric-card .sub {
        color: #9ca3af;
        font-size: 0.74rem;
        font-family: 'Inter', sans-serif;
    }

    /* ── Warning box ──────────────────────────── */
    .warn-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 3px solid #f59e0b;
        border-radius: 10px;
        padding: 0.85rem 1.25rem;
        color: #92400e;
        font-size: 0.85rem;
        margin-bottom: 1rem;
        font-family: 'Inter', sans-serif;
    }

    /* ── Stage list (sidebar) ─────────────────── */
    .stage {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.3rem;
        font-size: 0.82rem;
        color: #6b7280;
        padding: 0.3rem 0.5rem;
        border-radius: 7px;
        transition: background 0.12s ease, color 0.12s ease;
        font-family: 'Inter', sans-serif;
    }
    .stage:hover { background: #f0f2f6; color: #1a1a2e; }
    .stage .dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #22c55e;
        flex-shrink: 0;
    }
    .stage .code {
        font-weight: 700;
        color: #e63946;
        font-size: 0.74rem;
        background: rgba(230,57,70,0.07);
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        border: 1px solid rgba(230,57,70,0.18);
        letter-spacing: 0.02em;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Empty state ──────────────────────────── */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
    }
    .empty-state .icon { font-size: 2.8rem; margin-bottom: 0.75rem; line-height: 1; }
    .empty-state p { font-size: 0.9rem; color: #9ca3af; margin: 0; font-family: 'Inter', sans-serif; }

    /* ── Results heading ──────────────────────── */
    .results-heading {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0 0 0.5rem;
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.015em;
    }

    /* ── Sidebar brand ────────────────────────── */
    .sidebar-brand {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a2e;
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.015em;
        padding: 0.25rem 0 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Sidebar
# ------------------------------------------------------------------ #
stages = [
    ("A", "Honeypot / consistency filter"),
    ("B", "Hybrid retrieval (BM25 + dense)"),
    ("C", "Feature engineering (48 feats)"),
    ("D", "XGBoost LambdaMART scoring"),
    ("E", "Cross-encoder re-rank"),
    ("F", "Hard JD gates"),
    ("G", "SHAP reasoning"),
]

with st.sidebar:
    st.markdown('<div class="sidebar-brand">🍡 MochiRank</div>', unsafe_allow_html=True)
    st.markdown("---")

    with st.expander("⚙️ Pipeline Stages", expanded=True):
        for code, label in stages:
            st.markdown(
                f'<div class="stage">'
                f'<span class="dot"></span>'
                f'<span class="code">{code}</span>'
                f'{label}'
                f'</div>',
                unsafe_allow_html=True,
            )

    with st.expander("💻 CLI Usage"):
        st.code(
            "python rank.py \\\n  --candidates candidates.jsonl \\\n  --out submission.csv",
            language="bash",
        )
        st.caption("Accepts the same JSON schema as `dataset/sample_candidates.json`.")

# ------------------------------------------------------------------ #
# Hero header
# ------------------------------------------------------------------ #
st.markdown(
    """
    <div class="hero">
        <div class="badge"><span class="dot"></span>REDROB TRACK 1</div>
        <h1>🍡 MochiRank</h1>
        <p>Senior AI Engineer — Founding Team &nbsp;·&nbsp; Candidate Ranker</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Upload section
# ------------------------------------------------------------------ #
st.markdown('<div class="section-heading">📂 Upload Candidates</div>', unsafe_allow_html=True)

col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded = st.file_uploader(
        "Upload candidates JSON",
        type=["json"],
        help="JSON array of candidate objects. Same schema as sample_candidates.json.",
    )

with col_info:
    st.info(
        "**Format:** JSON array  \n"
        "**Schema:** `candidate_schema.json`"
    )

if uploaded is None:
    st.markdown(
        '<div class="empty-state">'
        '<div class="icon">📂</div>'
        '<p>Upload a candidates JSON file above to get started.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ------------------------------------------------------------------ #
# Parse upload
# ------------------------------------------------------------------ #
try:
    candidates = json.load(uploaded)
except Exception as e:
    st.error(f"Could not parse JSON: {e}")
    st.stop()

if not isinstance(candidates, list):
    st.error("Expected a JSON array at the top level.")
    st.stop()

st.success(f"Loaded **{len(candidates)}** candidate{'s' if len(candidates) != 1 else ''}.")

# ------------------------------------------------------------------ #
# Rank button
# ------------------------------------------------------------------ #
st.markdown('<div class="section-heading">🚀 Run Pipeline</div>', unsafe_allow_html=True)

if not st.button("Rank Candidates", type="primary", use_container_width=False):
    st.stop()

# ------------------------------------------------------------------ #
# Pipeline execution
# ------------------------------------------------------------------ #
progress = st.progress(0, text="Initialising…")

try:
    import xgboost as xgb
    from src.consistency_checks import check_consistency
    from src.feature_engineering import FEATURE_NAMES, compute_features, load_precomputed
    from src.reasoning_generator import generate_reasoning

    progress.progress(10, text="Loading artifacts…")
    precomputed = load_precomputed(ARTIFACTS)
    model = xgb.Booster()
    model.load_model(ARTIFACTS / "ranker_model.json")
    cid_to_rows = precomputed.get("cid_to_rows", {})

    # Warn about out-of-sample candidates (semantic features will be zero)
    oos = [c["candidate_id"] for c in candidates if c.get("candidate_id") not in cid_to_rows]
    if oos:
        st.markdown(
            f'<div class="warn-box">⚠️ <b>{len(oos)} candidate{"s" if len(oos) != 1 else ""} not found</b> '
            f'in precomputed artifacts — semantic and BM25 features will be zero for these profiles. '
            f'Scores may be lower than their true ranking.</div>',
            unsafe_allow_html=True,
        )

    progress.progress(25, text="Stage A: consistency & honeypot detection…")
    rows = []
    honeypot_ids = []
    for c in candidates:
        is_hp, n_v, reasons = check_consistency(c)
        feats = compute_features(c, precomputed, n_v, is_hp)
        rows.append({
            "candidate_id": c["candidate_id"],
            "candidate":    c,
            "feats":        feats,
            "is_honeypot":  is_hp,
        })
        if is_hp:
            honeypot_ids.append(c["candidate_id"])

    progress.progress(50, text="Stage D: XGBoost scoring…")
    X = np.array([r["feats"] for r in rows], dtype=np.float32)
    dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
    scores = model.predict(dmat)

    progress.progress(65, text="Stage E: cross-encoder re-rank…")
    try:
        from src.reranker import rerank
        top200_idx = np.argsort(scores)[::-1][:200].tolist()
        top200_rows = [rows[i] for i in top200_idx]
        top200_scores = [float(scores[i]) for i in top200_idx]
        reranked = rerank(
            [(r["candidate_id"], r["candidate"]) for r in top200_rows],
            top200_scores,
        )
        score_map = {cid: s for cid, s in reranked}
        for row, base_score in zip(rows, scores):
            row["score"] = score_map.get(row["candidate_id"], float(base_score))
    except Exception:
        for row, base_score in zip(rows, scores):
            row["score"] = float(base_score)

    progress.progress(75, text="Stage F: hard gates…")
    results = [r for r in rows if not r["is_honeypot"]]
    results.sort(key=lambda x: -x["score"])
    results = results[:100]
    for rank_pos, r in enumerate(results, 1):
        r["rank"] = rank_pos

    progress.progress(88, text="Stage G: SHAP reasoning…")
    X_all = np.array([r["feats"] for r in rows], dtype=np.float32)
    dmat_all = xgb.DMatrix(X_all, feature_names=FEATURE_NAMES)
    shap_matrix = model.predict(dmat_all, pred_contribs=True)

    cid_to_idx = {r["candidate_id"]: i for i, r in enumerate(rows)}
    for r in results:
        idx = cid_to_idx[r["candidate_id"]]
        shap_row = shap_matrix[idx, :-1]
        r["reasoning"] = generate_reasoning(
            r["candidate_id"], r["candidate"], shap_row, FEATURE_NAMES, r["rank"]
        )

    progress.progress(100, text="Done!")
    progress.empty()

except FileNotFoundError as e:
    progress.empty()
    st.error(
        f"Artifact not found: {e}\n\n"
        "Run the offline pipeline first to generate `artifacts/`."
    )
    st.stop()
except Exception:
    progress.empty()
    st.error("Ranking failed.")
    st.code(traceback.format_exc())
    st.stop()

# ------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------ #
n_honeypots = len(honeypot_ids)
n_finalists = len(results)
n_total     = len(candidates)

st.markdown(
    f"""
    <div class="section-heading">📊 Results</div>
    <div class="metric-row">
        <div class="metric-card blue">
            <div class="label">Candidates Processed</div>
            <div class="value">{n_total}</div>
            <div class="sub">from uploaded file</div>
        </div>
        <div class="metric-card red">
            <div class="label">Honeypots Detected</div>
            <div class="value">{n_honeypots}</div>
            <div class="sub">excluded from ranking</div>
        </div>
        <div class="metric-card green">
            <div class="label">Finalists Ranked</div>
            <div class="value">{n_finalists}</div>
            <div class="sub">top candidates</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Download button
# ------------------------------------------------------------------ #
buf = io.StringIO()
writer = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
writer.writeheader()
for r in results:
    writer.writerow({
        "candidate_id": r["candidate_id"],
        "rank":         r["rank"],
        "score":        f"{r['score']:.6f}",
        "reasoning":    r.get("reasoning", ""),
    })

st.download_button(
    label="⬇  Download ranked_candidates.csv",
    data=buf.getvalue(),
    file_name="ranked_candidates.csv",
    mime="text/csv",
    use_container_width=False,
)

# ------------------------------------------------------------------ #
# Results table
# ------------------------------------------------------------------ #
st.markdown('<div class="results-heading">Rankings</div>', unsafe_allow_html=True)

tab_all, tab_top10 = st.tabs([f"All {n_finalists} finalists", "Top 10"])

table_data = [
    {
        "Rank":         f"{'🥇' if r['rank'] == 1 else '🥈' if r['rank'] == 2 else '🥉' if r['rank'] == 3 else r['rank']}",
        "Candidate ID": r["candidate_id"],
        "Score":        round(r["score"], 4),
        "Reasoning":    r.get("reasoning", ""),
    }
    for r in results
]

with tab_all:
    st.dataframe(
        table_data,
        use_container_width=True,
        height=560,
        column_config={
            "Rank":         st.column_config.TextColumn("Rank",         width="small"),
            "Candidate ID": st.column_config.TextColumn("Candidate ID", width="medium"),
            "Score":        st.column_config.NumberColumn("Score",       format="%.4f", width="small"),
            "Reasoning":    st.column_config.TextColumn("Reasoning",    width="large"),
        },
    )

with tab_top10:
    st.dataframe(
        table_data[:10],
        use_container_width=True,
        height=420,
        column_config={
            "Rank":         st.column_config.TextColumn("Rank",         width="small"),
            "Candidate ID": st.column_config.TextColumn("Candidate ID", width="medium"),
            "Score":        st.column_config.NumberColumn("Score",       format="%.4f", width="small"),
            "Reasoning":    st.column_config.TextColumn("Reasoning",    width="large"),
        },
    )

# ------------------------------------------------------------------ #
# Honeypot details (expandable)
# ------------------------------------------------------------------ #
if honeypot_ids:
    with st.expander(f"🚫 Honeypots excluded ({len(honeypot_ids)})"):
        st.caption("These candidates failed consistency checks and were removed from ranking.")
        for cid in honeypot_ids:
            st.markdown(f"- `{cid}`")
