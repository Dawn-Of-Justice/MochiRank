"""
MochiRank — Streamlit demo app.
Upload candidates as JSON, get a ranked CSV with reasoning back.

Run:  C:/Users/udaya/AppData/Local/Programs/Python/Python311/python.exe -m streamlit run sandbox/app.py
"""

import csv
import io
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

ARTIFACTS = Path(__file__).parent.parent / "artifacts"

# ------------------------------------------------------------------ #
# JD hard-gate logic — inlined from rank.py to avoid importing the
# entry-point module and triggering its full import chain at load time.
# ------------------------------------------------------------------ #
_NON_TECH_TITLES = (
    "marketing", "sales", "accountant", "account manager",
    "operations manager", "customer support", "hr ", "human resource",
    "finance", "supply chain", "civil engineer", "mechanical engineer",
    "electrical engineer", "procurement", "recruiter",
)
_CV_ONLY_TERMS = (
    "computer vision", "object detection", "image segmentation",
    "speech recognition", "text to speech", "robotics", "ros ",
)
_NLP_IR_TERMS = (
    "nlp", "retrieval", "ranking", "search", "recommendation",
    "text", "language model", "embedding", "information retrieval",
    "llm", "transformer",
)


def _apply_jd_disqualifiers(candidate: dict, features: dict) -> tuple:
    profile = candidate.get("profile", {})
    career  = candidate.get("career_history", [])
    sig     = candidate.get("redrob_signals", {})
    title   = profile.get("current_title", "").lower()
    if any(t in title for t in _NON_TECH_TITLES):
        return True, f"non-technical title: {profile.get('current_title', '')}"
    if career and features.get("ever_at_it_services_only", 0) > 0.5:
        return True, "entire career at IT services, no product company"
    if profile.get("years_of_experience", 0) < 2.0:
        return True, f"insufficient experience: {profile.get('years_of_experience', 0)} yrs"
    if profile.get("country") != "India" and not sig.get("willing_to_relocate"):
        return True, "outside India, not willing to relocate"
    all_text = " ".join([
        profile.get("summary", ""), profile.get("headline", ""),
        *[j.get("description", "") for j in career],
    ]).lower()
    if any(t in all_text for t in _CV_ONLY_TERMS) and not any(t in all_text for t in _NLP_IR_TERMS):
        return True, "CV/speech/robotics without NLP/IR exposure"
    return False, ""


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
    #MainMenu                    { visibility: hidden !important; }
    footer                       { visibility: hidden !important; }
    [data-testid="stDecoration"] { display: none !important; }
    /* Blend header into white page — never set visibility:hidden on header,
       it would swallow the sidebar toggle button */
    [data-testid="stHeader"] {
        background-color: #ffffff !important;
        box-shadow: none !important;
        border-bottom: 1px solid #f0f2f6 !important;
    }
    /* Hide deploy button only — never touch the whole toolbar with
       opacity/visibility so the sidebar toggle is never affected */
    [data-testid="stToolbar"] { background: transparent !important; }
    [data-testid="stDeployButton"] { display: none !important; }

    /* ── Sidebar expand button (shown by Streamlit only when sidebar is COLLAPSED)
       Do NOT set display:flex here — that would force it visible even when
       the sidebar is open, creating a duplicate arrow. Only style appearance. */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        background: #1a1a2e !important;
        border-radius: 8px !important;
        padding: 4px 10px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.28) !important;
        z-index: 999999 !important;
    }
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapsedControl"] svg *,
    [data-testid="collapsedControl"] svg * {
        fill: #ffffff !important;
        stroke: #ffffff !important;
        color: #ffffff !important;
    }

    /* ── Sidebar — fixed 280 px, no resize ──────── */
    [data-testid="stSidebar"] {
        background: #f0f2f6 !important;
        border-right: 1px solid #e2e4e9 !important;
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
        padding-top: 0 !important;
    }
    [data-testid="stSidebarContent"] {
        overflow-y: auto !important;
        padding: 0.5rem 0.75rem 1.5rem !important;
    }
    /* Hide the drag-to-resize handle */
    [data-testid="stSidebarResizeHandle"],
    [data-testid="stSidebarCollapseHandle"] {
        display: none !important;
        pointer-events: none !important;
    }

    /* ── Sidebar nav rows ─────────────────────── */
    .sb-brand {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1a1a2e;
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.02em;
        padding: 1rem 0.1rem 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .sb-divider {
        height: 1px;
        background: #e2e4e9;
        margin: 0.35rem 0 0.75rem;
    }
    .sb-section-label {
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: #9ca3af;
        padding: 0.1rem 0.1rem 0.4rem;
        font-family: 'Inter', sans-serif;
    }
    .sb-nav-item {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        background: #ffffff;
        border: 1px solid #e2e4e9;
        border-radius: 8px;
        padding: 0.58rem 0.8rem;
        margin-bottom: 0.3rem;
        font-size: 0.82rem;
        color: #374151;
        font-family: 'Inter', sans-serif;
        cursor: default;
        transition: background 0.12s ease, box-shadow 0.12s ease;
    }
    .sb-nav-item:hover {
        background: #f8fafc;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
        color: #1a1a2e;
    }
    .sb-nav-chevron {
        margin-left: auto;
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1;
    }
    .sb-stage-code {
        font-size: 0.68rem;
        font-weight: 700;
        color: #e63946;
        background: rgba(230,57,70,0.07);
        border: 1px solid rgba(230,57,70,0.18);
        border-radius: 4px;
        padding: 0.06rem 0.38rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.02em;
        flex-shrink: 0;
    }
    .sb-stage-dot {
        width: 5px; height: 5px;
        border-radius: 50%;
        background: #22c55e;
        flex-shrink: 0;
    }
    /* Legend section */
    .sb-legend-row {
        display: flex;
        align-items: flex-start;
        gap: 0.45rem;
        font-size: 0.77rem;
        color: #6b7280;
        font-family: 'Inter', sans-serif;
        padding: 0.18rem 0;
        line-height: 1.4;
    }
    .sb-legend-dot {
        width: 5px; height: 5px;
        border-radius: 50%;
        background: #22c55e;
        flex-shrink: 0;
        margin-top: 0.32rem;
    }
    .sb-legend-box {
        background: #ffffff;
        border: 1px solid #e2e4e9;
        border-radius: 8px;
        padding: 0.6rem 0.75rem;
        margin-top: 0.2rem;
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

    </style>
    """,
    unsafe_allow_html=True,
)

# Inject JS to find the sidebar toggle button (whenever Streamlit creates it)
# and paint it black so it's always visible on the white page.
st.markdown(
    """
    <script>
    (function() {
        /* Paint the collapsed-state expand button dark so it's visible on white.
           Runs once on load + watches for DOM changes (sidebar toggle re-renders). */
        function styleToggle() {
            ['[data-testid="stSidebarCollapsedControl"]','[data-testid="collapsedControl"]']
            .forEach(function(sel) {
                var el = document.querySelector(sel);
                if (!el) return;
                el.style.background   = '#1a1a2e';
                el.style.borderRadius = '8px';
                el.style.padding      = '4px 10px';
                el.style.boxShadow    = '0 2px 8px rgba(0,0,0,0.28)';
                el.style.zIndex       = '999999';
                el.querySelectorAll('svg,svg *').forEach(function(n) {
                    n.style.fill   = '#fff';
                    n.style.stroke = '#fff';
                    n.style.color  = '#fff';
                });
            });
        }
        styleToggle();
        var obs = new MutationObserver(styleToggle);
        obs.observe(document.body, { childList: true, subtree: true });
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Sidebar
# ------------------------------------------------------------------ #
_stages = [
    ("A", "🛡️", "Honeypot filter"),
    ("B", "🔍", "Hybrid retrieval"),
    ("C", "⚙️", "Feature engineering"),
    ("D", "🤖", "XGBoost scoring"),
    ("E", "🔁", "Cross-encoder re-rank"),
    ("F", "🚧", "Hard JD gates"),
    ("G", "💡", "SHAP reasoning"),
]

with st.sidebar:
    st.markdown('<div class="sb-brand">🍡 MochiRank</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # Pipeline stages as nav rows
    st.markdown('<div class="sb-section-label">Pipeline Stages</div>', unsafe_allow_html=True)
    for code, icon, label in _stages:
        st.markdown(
            f'<div class="sb-nav-item">'
            f'<span class="sb-stage-dot"></span>'
            f'<span class="sb-stage-code">{code}</span>'
            f'{icon} {label}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sb-divider" style="margin-top:0.75rem"></div>', unsafe_allow_html=True)

    # CLI usage
    st.markdown('<div class="sb-section-label">CLI Usage</div>', unsafe_allow_html=True)
    st.code(
        "python rank.py \\\n  --candidates candidates.jsonl \\\n  --out submission.csv",
        language="bash",
    )

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
        "Upload candidates JSONL or JSON",
        type=["jsonl", "json"],
        help="JSONL (one candidate per line) or a JSON array. No size limit.",
    )

with col_info:
    st.info(
        "**Format:** JSONL or JSON array  \n"
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
with st.spinner(
    f"🍡 Wading through **{uploaded.name}**… "
    "large files take a moment, grab a ☕"
):
    try:
        raw = uploaded.read().decode("utf-8")
        if uploaded.name.endswith(".jsonl"):
            candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            candidates = json.loads(raw)
    except Exception as e:
        st.error(f"Could not parse file: {e}")
        st.stop()

if not isinstance(candidates, list):
    st.error("Expected a JSON array or JSONL file at the top level.")
    st.stop()

st.success(f"Loaded **{len(candidates)}** candidate{'s' if len(candidates) != 1 else ''}.")

# ------------------------------------------------------------------ #
# Rank button
# ------------------------------------------------------------------ #
st.markdown('<div class="section-heading">🚀 Run Pipeline</div>', unsafe_allow_html=True)

if not st.button("Rank Candidates", type="primary", use_container_width=False):
    st.stop()

# ------------------------------------------------------------------ #
# Pipeline execution  (mirrors rank.py stages A–G)
# ------------------------------------------------------------------ #

# Custom HTML progress bar — full style control, no Streamlit theme interference.
_prog_slot = st.empty()

def _prog(pct: int, text: str) -> None:
    _prog_slot.markdown(
        f'<div style="margin:0.5rem 0 1rem">'
        f'<div style="font-size:0.82rem;color:#4b5563;margin-bottom:0.35rem">{text}</div>'
        f'<div style="background:rgba(0,0,0,0.08);border-radius:6px;height:8px;overflow:hidden">'
        f'<div style="background:linear-gradient(90deg,#1d4ed8,#3b82f6);'
        f'width:{pct}%;height:100%;border-radius:6px"></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

_prog(0, "Initialising…")

# Wall-clock timer — surfaced in the results as a speed badge.
_t_start = time.time()

# Convert list → {cid: candidate} dict, same as rank.py
candidates_dict = {c["candidate_id"]: c for c in candidates}

try:
    import xgboost as xgb
    # Clear any stale src.* modules Streamlit may have cached from a previous
    # hot-reload cycle, so every src import below gets a fresh read from disk.
    import sys as _sys
    for _k in list(_sys.modules.keys()):
        if _k == 'src' or _k.startswith('src.'):
            del _sys.modules[_k]
    from src.consistency_checks import check_consistency
    from src.feature_engineering import (
        FEATURE_NAMES,
        compute_features,
        compute_features_dict,
        load_precomputed,
    )
    from src.reasoning_generator import generate_reasoning
    from src.retrieval import bm25_retrieve, dense_retrieve, reciprocal_rank_fusion
    from src.reranker import rerank_top_n
    from src.runtime_index import attach_runtime_index

    # Load JD-side artifacts only (no candidate embeddings/BM25 — built at runtime)
    _prog(5, "Loading artifacts…")
    precomputed = load_precomputed(ARTIFACTS, load_candidate_artifacts=False)
    model = xgb.Booster()
    model.load_model(ARTIFACTS / "ranker_model.json")

    jd_text = ""
    hyp_path = ARTIFACTS / "hypothetical_resumes.json"
    if hyp_path.exists():
        import json as _json
        with open(hyp_path, encoding="utf-8") as _f:
            jd_text = _json.load(_f).get("jd_text", "")

    # Build dense + BM25 indexes from the uploaded candidates at runtime
    _prog(10, "Building runtime indexes — embedding candidates…")
    bm25_data = attach_runtime_index(precomputed, candidates_dict, ARTIFACTS / "potion-base-8M")
    _prog(35, "Runtime indexes built")

    # Stage A: consistency checks
    _prog(38, "Stage A: consistency & honeypot detection…")
    honeypot_ids: set = set()
    violation_counts: dict = {}
    is_honeypot_map: dict = {}
    for cid, c in candidates_dict.items():
        is_hp, n_v, _ = check_consistency(c)
        violation_counts[cid] = n_v
        is_honeypot_map[cid] = is_hp
        if is_hp:
            honeypot_ids.add(cid)

    # Stage B: hybrid retrieval → top 2000
    _prog(48, "Stage B: hybrid retrieval (BM25 + dense)…")
    all_ids = list(candidates_dict.keys())
    bm25_ranking   = bm25_retrieve(bm25_data, all_ids, top_n=5000)
    dense_ranking  = dense_retrieve(precomputed, all_ids, top_n=5000)
    rrf_scores     = reciprocal_rank_fusion([bm25_ranking, dense_ranking])
    top_2000_ids   = sorted(rrf_scores, key=lambda c: -rrf_scores[c])[:2000]

    # Stage C: feature engineering on top-2000
    _prog(58, "Stage C: feature engineering on top 2000…")
    feature_rows = []
    cid_order = []
    for cid in top_2000_ids:
        c = candidates_dict[cid]
        feats = compute_features(c, precomputed, violation_counts.get(cid, 0), is_honeypot_map.get(cid, False))
        feature_rows.append(feats)
        cid_order.append(cid)
    X = np.array(feature_rows, dtype=np.float32)
    cid_to_matrix_idx = {cid: i for i, cid in enumerate(cid_order)}

    # Stage D: XGBoost scoring
    _prog(68, "Stage D: XGBoost scoring…")
    dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
    scores = model.predict(dmat)
    ranked_ids = [cid_order[i] for i in np.argsort(-scores)]

    # Stage E: cross-encoder re-rank on top 200
    _prog(75, "Stage E: cross-encoder re-rank…")
    top_200_candidates = [candidates_dict[cid] for cid in ranked_ids[:200] if cid in candidates_dict]
    reranked = rerank_top_n(top_200_candidates, jd_text, n=200)
    reranked_ids = [cid for cid, _ in reranked]
    reranked_scores_map = {cid: score for cid, score in reranked}
    all_ranked_ids = reranked_ids + [cid for cid in ranked_ids[200:]]

    # Stage F: honeypot filter + JD hard gates → top 100
    _prog(82, "Stage F: hard gates…")
    final_100: list = []
    skipped: set = set()
    for cid in all_ranked_ids:
        if len(final_100) >= 100:
            break
        if cid in honeypot_ids:
            skipped.add(cid)
            continue
        c = candidates_dict[cid]
        feat_dict = compute_features_dict(c, precomputed, violation_counts.get(cid, 0))
        disqualified, _ = _apply_jd_disqualifiers(c, feat_dict)
        if disqualified:
            skipped.add(cid)
            continue
        final_100.append(cid)

    # Stage G: SHAP reasoning
    _prog(90, "Stage G: SHAP reasoning…")
    shap_matrix = model.predict(dmat, pred_contribs=True)[:, :-1]

    results = []
    for rank_pos, cid in enumerate(final_100, start=1):
        c = candidates_dict[cid]
        idx = cid_to_matrix_idx.get(cid, -1)
        score = float(scores[idx]) if idx >= 0 else 0.0
        if idx >= 0:
            reasoning = generate_reasoning(cid, c, shap_matrix[idx], FEATURE_NAMES, rank_pos)
        else:
            yoe = c["profile"].get("years_of_experience", 0)
            title = c["profile"].get("current_title", "professional")
            reasoning = f"{yoe}yr {title}; ranked {rank_pos} by model score."
        results.append({
            "candidate_id": cid,
            "candidate":    c,
            "score":        score,
            "rank":         rank_pos,
            "reasoning":    reasoning,
        })

    honeypot_ids_list = list(honeypot_ids)
    _elapsed_s = time.time() - _t_start   # before the cosmetic sleep below
    _prog(100, "✅ Done! Pipeline complete.")
    time.sleep(1.0)
    _prog_slot.empty()

except FileNotFoundError as e:
    _prog_slot.empty()
    st.error(
        f"Artifact not found: {e}\n\n"
        "Run the offline pipeline first to generate `artifacts/`."
    )
    st.stop()
except Exception:
    _prog_slot.empty()
    st.error("Ranking failed.")
    st.code(traceback.format_exc())
    st.stop()

# ------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------ #
n_honeypots = len(honeypot_ids_list)
n_finalists = len(results)
n_total     = len(candidates_dict)

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
# Speed badge — sells the sub-5-min constraint
# ------------------------------------------------------------------ #
_thru = int(n_total / _elapsed_s) if _elapsed_s > 0 else 0
st.markdown(
    f'<div style="display:inline-flex;align-items:center;gap:0.5rem;'
    f'background:rgba(34,197,94,0.10);border:1px solid rgba(34,197,94,0.30);'
    f'color:#15803d;font-size:0.84rem;font-weight:600;'
    f'padding:0.45rem 1rem;border-radius:20px;margin-bottom:1.5rem;'
    f'font-family:Inter,sans-serif">'
    f'⚡ Ranked <strong>{n_total:,}</strong> candidates in '
    f'<strong>{_elapsed_s:.1f}s</strong>'
    f'<span style="color:#9ca3af;font-weight:400">· ~{_thru:,}/s · CPU only</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Score distribution — separation between strong and borderline
# ------------------------------------------------------------------ #
import altair as alt
import pandas as pd

st.markdown('<div class="results-heading">Score distribution</div>', unsafe_allow_html=True)
_dist_df = pd.DataFrame({
    "Rank":  [r["rank"]  for r in results],
    "Score": [r["score"] for r in results],
})
_chart = (
    alt.Chart(_dist_df)
    .mark_area(
        interpolate="monotone",
        line={"color": "#2563eb", "strokeWidth": 2},
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="#eff6ff", offset=0),
                alt.GradientStop(color="#3b82f6", offset=1),
            ],
            x1=1, x2=1, y1=1, y2=0,
        ),
    )
    .encode(
        x=alt.X("Rank:Q", title="Rank", axis=alt.Axis(grid=False, tickMinStep=1)),
        y=alt.Y("Score:Q", title="Model score", scale=alt.Scale(zero=False)),
        tooltip=["Rank:Q", alt.Tooltip("Score:Q", format=".4f")],
    )
    .properties(height=200)
    .configure_view(strokeOpacity=0)
    .configure_axis(labelColor="#6b7280", titleColor="#6b7280", domainColor="#e5e7eb")
)
st.altair_chart(_chart, use_container_width=True)

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
if honeypot_ids_list:
    with st.expander(f"🚫 Honeypots excluded ({len(honeypot_ids_list)})"):
        st.caption("These candidates failed consistency checks and were removed from ranking.")
        for cid in honeypot_ids_list:
            st.markdown(f"- `{cid}`")
