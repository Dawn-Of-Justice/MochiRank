"""
MochiRank — Streamlit demo app.
Upload candidates as JSON, get a ranked CSV with reasoning back.

Run:  C:/Users/udaya/AppData/Local/Programs/Python/Python311/python.exe -m streamlit run sandbox/app.py
"""

import csv
import html
import io
import json
import sys
import time
import traceback
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
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
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------ #
# Theme state  (persists across reruns so toggle doesn't lose results)
# ------------------------------------------------------------------ #
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None

_dark = st.session_state.theme == "dark"


def _toggle_theme() -> None:
    """Flip the theme. Runs as an on_click callback (during Streamlit's
    widget-processing phase, before the script body re-executes) so the
    switch fires reliably no matter which view — upload, running, or
    results — is currently on screen. Reads/writes session_state directly
    rather than the module-level _dark snapshot, which is stale inside a
    callback."""
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# ------------------------------------------------------------------ #
# CSS — two complete palettes, switched at runtime
# ------------------------------------------------------------------ #
def _get_css(dark: bool) -> str:
    if dark:
        bg      = "#131C30";  card    = "#1C2741"
        hero_bg = "linear-gradient(135deg,#1A2540 0%,#22335A 60%,#1A2540 100%)"
        hero_bd = "rgba(59,130,246,0.18)"
        hero_sh = "0 0 80px rgba(59,130,246,0.06),0 8px 48px rgba(0,0,0,0.45)"
        h1c = "#F1F5FB";  subc = "#9AA6C4"
        txt = "#F1F5FB";  txts = "#B4BFD6";  txtm = "#8794B2";  txstr = "#DCE3F0"
        hdr_bg = "#131C30";  hdr_bd = "rgba(255,255,255,0.06)"
        brd = "rgba(255,255,255,0.09)";  brdh = "rgba(255,255,255,0.16)"
        code_bg = "rgba(255,255,255,0.07)";  code_bd = "rgba(255,255,255,0.11)"
        code_c = "#f87171";  cb_bg = "#16213A"
        tab_i = "#8794B2";  tab_h = "#F1F5FB";  tab_bd = "rgba(255,255,255,0.10)"
        exp_bg = "rgba(255,255,255,0.04)";  exp_bd = "rgba(255,255,255,0.10)";  exp_lbl = "#B4BFD6"
        pip_bg = "rgba(28,39,65,0.85)";  pip_bd = "rgba(255,255,255,0.09)"
        scc = "#f87171";  scbg = "rgba(248,113,113,0.12)";  scbd = "rgba(248,113,113,0.24)"
        slbl = "#8794B2"
        cng = "linear-gradient(90deg,rgba(59,130,246,0.30),rgba(59,130,246,0.08))"
        emp_bg = "rgba(255,255,255,0.025)";  emp_bd = "rgba(255,255,255,0.10)";  emp_h3 = "#DCE3F0";  emp_p = "#8794B2"
        m_bg = "#1C2741"
        mv_b = "#60a5fa";  ms_b = "0 0 24px rgba(59,130,246,0.40)"
        mv_r = "#f87171";  ms_r = "0 0 24px rgba(239,68,68,0.40)"
        mv_g = "#4ade80";  ms_g = "0 0 24px rgba(34,197,94,0.40)"
        m_lbl = "#8794B2";  m_sub = "#8794B2";  m_gop = "0.07"
        sp_bg = "rgba(34,197,94,0.10)";  sp_bd = "rgba(34,197,94,0.22)";  sp_c = "#5be58f";  sp_m = "#8794B2"
        pt = "rgba(255,255,255,0.09)"
        pb = "linear-gradient(90deg,#1d4ed8,#3b82f6,#60a5fa)"
        ptxt = "#B4BFD6";  ppct = "#8794B2"
        up_bg = "rgba(255,255,255,0.04)";  up_bd = "rgba(255,255,255,0.16)"
        up_txt = "#F1F5FB";  up_sm = "#AEB9D2";  fi = "#C2CCE0"
        ub_bg = "rgba(255,255,255,0.05)";  ub_c = "#B4BFD6";  ub_bd = "rgba(255,255,255,0.16)"
        ub_hbg = "rgba(255,255,255,0.10)";  ub_hbd = "rgba(255,255,255,0.26)"
        sc_t = "rgba(255,255,255,0.12)";  sc_h = "rgba(255,255,255,0.20)"
        df_sh = "0 4px 24px rgba(0,0,0,0.28)"
        tog_bg = "rgba(255,255,255,0.08)";  tog_c = "#DCE3F0"
        tog_bd = "rgba(255,255,255,0.16)";  tog_hbg = "rgba(255,255,255,0.14)";  tog_sh = "none"
        hr_c = "rgba(255,255,255,0.10)"
        sel_bg = "rgba(255,255,255,0.05)";  sel_bd = "rgba(255,255,255,0.14)";  sel_c = "#F1F5FB"
        tbl_head = "#1E2B47";  tbl_rbd = "rgba(255,255,255,0.07)";  tbl_hov = "rgba(255,255,255,0.04)";  tbl_id = "#7cb3fb"
        ch_stop0 = "rgba(19,28,48,0)";  ch_ax = "#8794B2";  ch_dom = "rgba(255,255,255,0.10)";  ch_grid = "rgba(255,255,255,0.06)"
    else:
        bg      = "#F8FAFC";  card    = "#FFFFFF"
        hero_bg = "linear-gradient(135deg,#eff6ff 0%,#dbeafe 55%,#e0f2fe 100%)"
        hero_bd = "#bfdbfe"
        hero_sh = "0 4px 20px rgba(59,130,246,0.08)"
        h1c = "#1a1a2e";  subc = "#4b5563"
        txt = "#0F172A";  txts = "#374151";  txtm = "#6B7280";  txstr = "#1a1a2e"
        hdr_bg = "#F8FAFC";  hdr_bd = "#f0f2f6"
        brd = "#e5e7eb";  brdh = "#d1d5db"
        code_bg = "#f3f4f6";  code_bd = "#e5e7eb"
        code_c = "#e63946";  cb_bg = "#f8f9fa"
        tab_i = "#6b7280";  tab_h = "#1a1a2e";  tab_bd = "#e5e7eb"
        exp_bg = "#ffffff";  exp_bd = "#e0e0e0";  exp_lbl = "#374151"
        pip_bg = "#F1F5F9";  pip_bd = "#e2e4e9"
        scc = "#e63946";  scbg = "rgba(230,57,70,0.07)";  scbd = "rgba(230,57,70,0.18)"
        slbl = "#6B7280"
        cng = "linear-gradient(90deg,rgba(59,130,246,0.20),rgba(59,130,246,0.05))"
        emp_bg = "#f8fafc";  emp_bd = "#e2e4e9";  emp_h3 = "#1a1a2e";  emp_p = "#6B7280"
        m_bg = "#FFFFFF"
        mv_b = "#2563eb";  ms_b = "none"
        mv_r = "#dc2626";  ms_r = "none"
        mv_g = "#16a34a";  ms_g = "none"
        m_lbl = "#6b7280";  m_sub = "#9ca3af";  m_gop = "0"
        sp_bg = "rgba(22,163,74,0.08)";  sp_bd = "rgba(22,163,74,0.20)";  sp_c = "#15803d";  sp_m = "#6B7280"
        pt = "rgba(0,0,0,0.07)"
        pb = "linear-gradient(90deg,#1d4ed8,#3b82f6)"
        ptxt = "#374151";  ppct = "#6B7280"
        up_bg = "#fafafa";  up_bd = "#d1d5db"
        up_txt = "#374151";  up_sm = "#6B7280";  fi = "#9ca3af"
        ub_bg = "#ffffff";  ub_c = "#374151";  ub_bd = "#d1d5db"
        ub_hbg = "#f3f4f6";  ub_hbd = "#9ca3af"
        sc_t = "rgba(0,0,0,0.12)";  sc_h = "rgba(0,0,0,0.20)"
        df_sh = "0 2px 12px rgba(0,0,0,0.05)"
        tog_bg = "#ffffff";  tog_c = "#374151"
        tog_bd = "#e2e4e9";  tog_hbg = "#f3f4f6";  tog_sh = "0 1px 4px rgba(0,0,0,0.08)"
        hr_c = "#e5e7eb"
        sel_bg = "#ffffff";  sel_bd = "#d1d5db";  sel_c = "#1a1a2e"
        tbl_head = "#F1F5F9";  tbl_rbd = "#eef1f5";  tbl_hov = "#f8fafc";  tbl_id = "#2563eb"
        ch_stop0 = "rgba(248,250,252,0)";  ch_ax = "#6b7280";  ch_dom = "#e5e7eb";  ch_grid = "#f3f4f6"

    return f"""
    @import url('https://fonts.googleapis.com/css2?family=Exo:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
    .stApp {{ background-color: {bg} !important; color: {txt} !important; }}

    /* Smooth light/dark theme swap — animate colour properties on the
       elements that persist across Streamlit's rerun (React keeps the DOM
       nodes, so the property change transitions instead of snapping). */
    .stApp, [data-testid="stHeader"], [data-testid="stFileUploader"],
    .hero, .hero h1, .hero p, .pipeline-strip, .empty-state,
    .metric-card, .metric-card .value, .speed-badge,
    .rank-table-wrap, .rank-table thead th, .rank-table tbody td,
    [data-testid="stExpander"], [data-baseweb="tab"],
    [data-testid="stFileUploaderDropzone"] button,
    .st-key-theme_toggle .stButton > button {{
        transition: background-color 0.35s ease, color 0.35s ease,
                    border-color 0.35s ease, box-shadow 0.35s ease !important;
    }}

    @media (prefers-reduced-motion: reduce) {{
        .stApp, [data-testid="stHeader"], [data-testid="stFileUploader"],
        .hero, .hero h1, .hero p, .pipeline-strip, .empty-state,
        .metric-card, .metric-card .value, .speed-badge,
        .rank-table-wrap, .rank-table thead th, .rank-table tbody td,
        [data-testid="stExpander"], [data-baseweb="tab"],
        [data-testid="stFileUploaderDropzone"] button,
        .st-key-theme_toggle .stButton > button,
        .mochi-prog-bar {{ transition: none !important; animation: none !important; }}
    }}

    #MainMenu {{ visibility: hidden !important; }}
    footer {{ visibility: hidden !important; }}
    [data-testid="stDecoration"] {{ display: none !important; }}
    [data-testid="stHeader"] {{ background-color: {hdr_bg} !important; box-shadow: none !important; border-bottom: 1px solid {hdr_bd} !important; }}
    [data-testid="stToolbar"] {{ background: transparent !important; }}
    [data-testid="stDeployButton"] {{ display: none !important; }}
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"], [data-testid="stSidebarResizeHandle"] {{
        display: none !important; visibility: hidden !important;
    }}

    .stButton > button {{
        background: linear-gradient(135deg,#b91c1c 0%,#e63946 100%) !important;
        color: #fff !important; border: none !important; border-radius: 10px !important;
        padding: 0.65rem 2rem !important; font-weight: 600 !important; font-size: 0.92rem !important;
        font-family: 'Inter', sans-serif !important; letter-spacing: 0.02em !important;
        box-shadow: 0 4px 24px rgba(230,57,70,0.35) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important; cursor: pointer !important;
    }}
    .stButton > button:hover {{ transform: translateY(-2px) !important; box-shadow: 0 8px 32px rgba(230,57,70,0.52) !important; }}
    .stButton > button:active {{ transform: translateY(0) !important; }}

    .stDownloadButton > button {{
        background: linear-gradient(135deg,#14532d 0%,#16a34a 100%) !important;
        color: #fff !important; border: none !important; border-radius: 10px !important;
        padding: 0.65rem 2rem !important; font-weight: 600 !important; font-size: 0.92rem !important;
        font-family: 'Inter', sans-serif !important; box-shadow: 0 4px 24px rgba(22,163,74,0.30) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important; cursor: pointer !important;
    }}
    .stDownloadButton > button:hover {{ transform: translateY(-2px) !important; box-shadow: 0 8px 32px rgba(22,163,74,0.46) !important; }}

    /* Theme toggle pill — targets the auto-generated st-key-<key> wrapper
       so it reliably overrides the global red button style */
    .st-key-theme_toggle {{ position: relative; z-index: 1000; }}
    .st-key-theme_toggle .stButton > button {{
        background: {tog_bg} !important; color: {tog_c} !important;
        border: 1px solid {tog_bd} !important; box-shadow: {tog_sh} !important;
        border-radius: 20px !important; padding: 0.28rem 0.85rem !important;
        font-size: 0.76rem !important; font-weight: 500 !important; letter-spacing: 0 !important;
        min-height: 0 !important;
    }}
    .st-key-theme_toggle .stButton > button:hover {{
        background: {tog_hbg} !important; transform: none !important;
        box-shadow: {tog_sh} !important; border-color: {tog_bd} !important;
    }}

    [data-testid="stFileUploader"] {{
        background: {up_bg} !important; border: 2px dashed {up_bd} !important;
        border-radius: 14px !important; padding: 0.5rem 1rem !important; transition: border-color 0.25s ease !important;
    }}
    [data-testid="stFileUploader"]:hover,
    [data-testid="stFileUploader"]:focus-within {{ border-color: rgba(230,57,70,0.45) !important; }}
    [data-testid="stFileUploaderDropzone"] {{ background: transparent !important; }}
    /* Force every label/instruction line in the box to a visible colour.
       The main "Drag and drop file here" is a span/div (not a <p>), so target
       broadly; the secondary "Limit … per file" <small> stays a touch dimmer. */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] div,
    [data-testid="stFileUploaderDropzone"] p {{ color: {up_txt} !important; }}
    [data-testid="stFileUploaderDropzoneInstructions"] svg {{ fill: {fi} !important; color: {fi} !important; }}
    [data-testid="stFileUploaderDropzoneInstructions"] small,
    [data-testid="stFileUploaderDropzone"] small {{ color: {up_sm} !important; }}

    /* Subtle "Browse files" button — neutral, not the loud red CTA */
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {{
        background: {ub_bg} !important; color: {ub_c} !important;
        border: 1px solid {ub_bd} !important; box-shadow: none !important;
        border-radius: 8px !important; padding: 0.4rem 1rem !important;
        font-weight: 500 !important; font-size: 0.82rem !important;
        letter-spacing: 0 !important; min-height: 0 !important;
        transition: background-color 0.2s ease, border-color 0.2s ease !important;
    }}
    [data-testid="stFileUploaderDropzone"] button:hover,
    [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]:hover {{
        background: {ub_hbg} !important; border-color: {ub_hbd} !important;
        color: {ub_c} !important; transform: none !important; box-shadow: none !important;
    }}

    [data-testid="stAlert"] {{ border-radius: 10px !important; font-size: 0.88rem !important; font-family: 'Inter', sans-serif !important; }}

    [data-baseweb="tab-list"] {{
        background: transparent !important; border: none !important;
        border-bottom: 1px solid {tab_bd} !important; border-radius: 0 !important;
        padding: 0 !important; gap: 0.2rem !important; margin-bottom: 1.25rem !important;
    }}
    [data-baseweb="tab"] {{
        background: transparent !important; color: {tab_i} !important; font-weight: 500 !important;
        font-size: 0.9rem !important; font-family: 'Inter', sans-serif !important;
        padding: 0.65rem 1.25rem !important; border-radius: 0 !important;
        border-bottom: 2px solid transparent !important; margin-bottom: -1px !important;
        transition: color 0.15s ease !important; cursor: pointer !important;
    }}
    [data-baseweb="tab"]:hover {{ color: {tab_h} !important; }}
    [aria-selected="true"][data-baseweb="tab"] {{
        background: transparent !important; color: #e63946 !important;
        border-bottom: 2px solid #e63946 !important; font-weight: 600 !important;
    }}
    [data-baseweb="tab-highlight"] {{ display: none !important; }}
    [data-baseweb="tab-border"]    {{ display: none !important; }}

    [data-testid="stExpander"] {{
        background: {exp_bg} !important; border: 1px solid {exp_bd} !important;
        border-radius: 12px !important; box-shadow: none !important; overflow: hidden !important;
    }}
    [data-testid="stExpanderDetails"] {{ background: transparent !important; padding-top: 0.5rem !important; }}
    summary {{ color: {exp_lbl} !important; }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {brd} !important; border-radius: 12px !important;
        overflow: hidden !important; box-shadow: {df_sh} !important;
    }}

    code {{
        font-family: 'JetBrains Mono', monospace !important; background: {code_bg} !important;
        border: 1px solid {code_bd} !important; border-radius: 5px !important;
        padding: 0.15rem 0.45rem !important; color: {code_c} !important; font-size: 0.82em !important;
    }}
    pre code {{ border-radius: 0 !important; padding: 0 !important; border: none !important; color: inherit !important; }}
    [data-testid="stCodeBlock"] {{
        background: {cb_bg} !important; border: 1px solid {exp_bd} !important;
        border-radius: 10px !important; overflow: hidden !important;
    }}
    [data-testid="stCodeBlock"] * {{ font-family: 'JetBrains Mono', monospace !important; font-size: 0.83rem !important; }}

    .stMarkdown p, .stMarkdown li {{ color: {txts} !important; line-height: 1.65 !important; }}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ color: {txt} !important; font-weight: 700 !important; }}
    .stMarkdown strong {{ color: {txstr} !important; }}
    .stMarkdown a {{ color: #e63946 !important; }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color: {txtm} !important; font-size: 0.78rem !important; }}
    hr {{ border-color: {hr_c} !important; }}

    [data-testid="stSelectbox"] > div > div {{
        background: {sel_bg} !important; border: 1px solid {sel_bd} !important;
        border-radius: 8px !important; color: {sel_c} !important;
    }}

    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: {sc_t}; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {sc_h}; }}

    /* ── Hero ── */
    .hero {{
        background: {hero_bg}; border: 1px solid {hero_bd}; border-radius: 20px;
        padding: 2rem 3rem; margin-bottom: 0.75rem; box-shadow: {hero_sh};
        text-align: center; position: relative; overflow: hidden;
    }}
    .hero::before {{
        content: ''; position: absolute; top: -100px; right: -100px;
        width: 350px; height: 350px;
        background: radial-gradient(circle, rgba(230,57,70,0.07) 0%, transparent 65%);
        pointer-events: none;
    }}
    .hero::after {{
        content: ''; position: absolute; bottom: -80px; left: -80px;
        width: 280px; height: 280px;
        background: radial-gradient(circle, rgba(59,130,246,0.07) 0%, transparent 65%);
        pointer-events: none;
    }}
    .hero-logo {{ font-size: 3rem; line-height: 1; margin-bottom: 0.6rem; }}
    .hero h1 {{
        color: {h1c}; font-size: 2.5rem; font-weight: 800; margin: 0;
        letter-spacing: -0.04em; font-family: 'Exo', sans-serif;
    }}
    .hero p {{ color: {subc}; font-size: 0.95rem; margin: 0; font-family: 'Inter', sans-serif; }}
    .hero .badge {{
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: rgba(230,57,70,0.10); color: #f87171; font-size: 0.67rem; font-weight: 700;
        padding: 0.28rem 0.9rem; border-radius: 20px; border: 1px solid rgba(230,57,70,0.22);
        margin-bottom: 1.1rem; letter-spacing: 0.14em; font-family: 'Inter', sans-serif; text-transform: uppercase;
    }}
    .hero .badge .dot {{
        width: 5px; height: 5px; border-radius: 50%; background: #22c55e;
        box-shadow: 0 0 8px rgba(34,197,94,0.9); flex-shrink: 0;
    }}

    /* ── Pipeline strip ── */
    .pipeline-strip {{
        display: flex; align-items: center; justify-content: center;
        background: {pip_bg}; border: 1px solid {pip_bd}; border-radius: 14px;
        padding: 1.1rem 1.75rem; margin-bottom: 1.75rem;
        overflow-x: auto; flex-wrap: wrap; row-gap: 0.6rem; gap: 0;
    }}
    .pipeline-step {{ display: flex; flex-direction: column; align-items: center; gap: 0.28rem; flex-shrink: 0; }}
    .pipeline-step-header {{ display: flex; align-items: center; gap: 0.35rem; }}
    .pipeline-step-code {{
        font-size: 0.6rem; font-weight: 700; color: {scc};
        background: {scbg}; border: 1px solid {scbd}; border-radius: 4px;
        padding: 0.05rem 0.38rem; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.04em;
    }}
    .pipeline-step-icon {{ font-size: 0.85rem; line-height: 1; }}
    .pipeline-step-label {{ font-size: 0.64rem; color: {slbl}; font-family: 'Inter', sans-serif; white-space: nowrap; text-align: center; max-width: 80px; }}
    .pipeline-connector {{ width: 22px; height: 1px; background: {cng}; flex-shrink: 0; margin: 0 0.2rem; margin-bottom: 0.85rem; }}

    /* ── Section heading ── */
    .section-heading {{
        font-size: 1.15rem; font-weight: 700; color: {txt}; margin-top: 2rem; margin-bottom: 0.9rem;
        font-family: 'Exo', sans-serif; letter-spacing: -0.02em; display: flex; align-items: center; gap: 0.5rem;
    }}

    /* ── Empty state ── */
    .empty-state {{
        text-align: center; padding: 4.5rem 2rem; background: {emp_bg};
        border: 1px dashed {emp_bd}; border-radius: 16px; margin-top: 1rem;
    }}
    .empty-state .icon {{ font-size: 3.2rem; margin-bottom: 1rem; line-height: 1; }}
    .empty-state h3 {{ font-size: 1.05rem; font-weight: 600; color: {emp_h3}; margin: 0 0 0.4rem; font-family: 'Exo', sans-serif; }}
    .empty-state p {{ font-size: 0.88rem; color: {emp_p}; margin: 0; font-family: 'Inter', sans-serif; }}

    /* ── Metric cards ── */
    .metric-row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem; }}
    .metric-card {{
        background: {m_bg}; border: 1px solid {brd}; border-radius: 14px;
        padding: 1.5rem 1.75rem; position: relative; overflow: hidden;
        transition: border-color 0.2s ease, box-shadow 0.2s ease; cursor: default;
    }}
    .metric-card:hover {{ border-color: {brdh}; box-shadow: 0 10px 40px rgba(0,0,0,0.12); }}
    .metric-card::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; border-radius: 14px 14px 0 0; }}
    .metric-card.blue::before  {{ background: linear-gradient(90deg,#1d4ed8,#60a5fa); }}
    .metric-card.red::before   {{ background: linear-gradient(90deg,#dc2626,#f87171); }}
    .metric-card.green::before {{ background: linear-gradient(90deg,#15803d,#4ade80); }}
    .metric-card .bg-glow {{ position: absolute; top: -30px; right: -30px; width: 120px; height: 120px; border-radius: 50%; opacity: {m_gop}; pointer-events: none; }}
    .metric-card.blue  .bg-glow {{ background: #3B82F6; }}
    .metric-card.red   .bg-glow {{ background: #ef4444; }}
    .metric-card.green .bg-glow {{ background: #22c55e; }}
    .metric-card .label {{ color: {m_lbl}; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.13em; margin-bottom: 0.6rem; font-family: 'Inter', sans-serif; }}
    .metric-card .value {{ font-size: 2.75rem; font-weight: 800; line-height: 1; margin-bottom: 0.4rem; letter-spacing: -0.05em; font-family: 'Exo', sans-serif; }}
    .metric-card.blue  .value {{ color: {mv_b}; text-shadow: {ms_b}; }}
    .metric-card.red   .value {{ color: {mv_r}; text-shadow: {ms_r}; }}
    .metric-card.green .value {{ color: {mv_g}; text-shadow: {ms_g}; }}
    .metric-card .sub {{ color: {m_sub}; font-size: 0.73rem; font-family: 'Inter', sans-serif; }}

    /* ── Speed badge ── */
    .speed-badge {{
        display: inline-flex; align-items: center; gap: 0.45rem;
        background: {sp_bg}; border: 1px solid {sp_bd}; color: {sp_c};
        font-size: 0.83rem; font-weight: 600; padding: 0.48rem 1.1rem;
        border-radius: 24px; margin-bottom: 1.75rem; font-family: 'Inter', sans-serif;
    }}
    .speed-badge .muted {{ color: {sp_m}; font-weight: 400; }}

    /* ── Results heading ── */
    .results-heading {{ font-size: 1rem; font-weight: 700; color: {txt}; margin: 0 0 0.75rem; font-family: 'Exo', sans-serif; letter-spacing: -0.01em; }}

    /* ── Download group label ── */
    .dl-group-label {{
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;
        color: {m_lbl}; margin: 0 0 0.6rem; font-family: 'Inter', sans-serif;
    }}

    /* ── Scrollable rankings table ── */
    .rank-table-wrap {{
        max-height: 600px; overflow-y: auto;
        border: 1px solid {brd}; border-radius: 12px; box-shadow: {df_sh};
    }}
    .rank-table {{
        width: 100%; border-collapse: collapse;
        font-family: 'Inter', sans-serif; font-size: 0.85rem;
    }}
    .rank-table thead th {{
        position: sticky; top: 0; z-index: 2;
        background: {tbl_head}; color: {txts};
        text-align: left; font-weight: 700; font-size: 0.68rem;
        text-transform: uppercase; letter-spacing: 0.08em;
        padding: 0.8rem 1rem; border-bottom: 1px solid {brd};
    }}
    .rank-table tbody td {{
        padding: 0.7rem 1rem; border-bottom: 1px solid {tbl_rbd};
        color: {txts}; vertical-align: top;
    }}
    .rank-table tbody tr:hover td {{ background: {tbl_hov}; }}
    .rank-table tbody tr:last-child td {{ border-bottom: none; }}
    .rank-table .c-rank   {{ width: 64px; font-weight: 700; color: {txt}; }}
    .rank-table .c-id     {{ width: 160px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: {tbl_id}; white-space: nowrap; }}
    .rank-table .c-score  {{ width: 90px; font-family: 'JetBrains Mono', monospace; color: {txt}; white-space: nowrap; }}
    .rank-table .c-reason {{ line-height: 1.55; color: {txts}; min-width: 280px; }}
    """

# Expose chart palette at module level so the altair block can read it
_ch_stop0 = "rgba(19,28,48,0)" if _dark else "rgba(248,250,252,0)"
_ch_ax    = "#8794B2" if _dark else "#6b7280"
_ch_dom   = "rgba(255,255,255,0.10)" if _dark else "#e5e7eb"
_ch_grid  = "rgba(255,255,255,0.06)" if _dark else "#f3f4f6"

# Progress bar inline-style palette
_ptxt = "#B4BFD6" if _dark else "#374151"
_ppct = "#8794B2" if _dark else "#6B7280"
_pt   = "rgba(255,255,255,0.09)" if _dark else "rgba(0,0,0,0.07)"
_pb   = "linear-gradient(90deg,#1d4ed8,#3b82f6,#60a5fa)" if _dark else "linear-gradient(90deg,#1d4ed8,#3b82f6)"

st.markdown(f"<style>{_get_css(_dark)}</style>", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Theme toggle — top-right pill
# ------------------------------------------------------------------ #
_, _tog_col = st.columns([10, 2])
with _tog_col:
    _tog_label = "☀️  Light mode" if _dark else "🌙  Dark mode"
    st.button(_tog_label, key="theme_toggle", on_click=_toggle_theme)

# ------------------------------------------------------------------ #
# Hero header
# ------------------------------------------------------------------ #
st.markdown(
    """
    <div class="hero">
        <h1>MochiRank</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Job description (collapsed by default)
# ------------------------------------------------------------------ #
with st.expander("📋  Job Description — Senior AI Engineer (Founding Team)", expanded=False):
    st.markdown(
        """
        **Job Description:** Senior AI Engineer — Founding Team

        **Company:** Redrob AI (Series A AI-native talent intelligence platform)

        **Location:** Pune/Noida, India (Hybrid — flexible cadence) &nbsp;|&nbsp; Open to relocation candidates from Tier-1 Indian cities

        **Employment Type:** Full-time

        **Experience Required:** 5–9 years
        """
    )

st.markdown("<div style='margin-bottom:1.25rem'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Upload section
# ------------------------------------------------------------------ #
st.markdown(
    '<div class="section-heading">'
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#e63946" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
    '<polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>'
    '</svg> Upload Candidates</div>',
    unsafe_allow_html=True,
)

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

# Clear cached results when file changes or is removed
if uploaded is None:
    st.session_state.pipeline_results = None
    st.markdown(
        '<div class="empty-state">'
        '<div class="icon">📂</div>'
        '<h3>No file uploaded yet</h3>'
        '<p>Upload a candidates JSON or JSONL file above to get started.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()
elif st.session_state.get("_last_file") != uploaded.name:
    st.session_state.pipeline_results = None
    st.session_state["_last_file"] = uploaded.name

# ------------------------------------------------------------------ #
# Parse upload
# ------------------------------------------------------------------ #
with st.spinner(
    f"Wading through **{uploaded.name}**… "
    "large files take a moment, grab a coffee ☕"
):
    try:
        # Re-seek: file_uploader returns the *same* buffer object across reruns,
        # so a prior read left the pointer at EOF. Without this, a rerun (e.g.
        # toggling the theme after ranking) would read an empty string.
        uploaded.seek(0)
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
st.markdown(
    '<div class="section-heading">'
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#e63946" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<polygon points="5 3 19 12 5 21 5 3"/>'
    '</svg> Run Pipeline</div>',
    unsafe_allow_html=True,
)

_run_pipeline = st.button("Rank Candidates", type="primary", use_container_width=False)

if _run_pipeline:
    # ------------------------------------------------------------------ #
    # Pipeline execution  (mirrors rank.py stages A–G)
    # ------------------------------------------------------------------ #

    _prog_slot = st.empty()
    _fact_slot = st.empty()

    # Wall-clock timer — surfaced in the results as a speed badge.
    _t_start = time.time()

    # The bar follows the real pipeline stages: each _prog(pct, …) sets a true
    # checkpoint. To make sure it never *sits still* (even mid-stage while the
    # Python thread is blocked), pass ``creep_to`` = just under the next
    # checkpoint: the bar then trickles forward in the browser at a slow, steady
    # linear rate (~_CREEP_RATE %/sec → the % ticks up every few seconds). The
    # creep duration is sized so the cap is reached only far beyond any real
    # stage time, so the bar is always inching forward and never freezes; the
    # next _prog() snaps it to the true checkpoint. A rotating "fun facts"
    # ticker underneath adds extra reassurance during the longest stage.
    _CREEP_RATE = 0.3   # percent per second — slow trickle, never parked >~3s

    def _prog(pct: int, text: str, creep_to: int = None) -> None:
        if creep_to is not None and creep_to > pct:
            nm = f"{pct}_{creep_to}"
            dur = max(6.0, (creep_to - pct) / _CREEP_RATE)
            style = (
                f"<style>"
                f"@property --mp{nm}{{syntax:'<integer>';initial-value:{pct};inherits:false;}}"
                f"@keyframes mpw{nm}{{from{{width:{pct}%}}to{{width:{creep_to}%}}}}"
                f"@keyframes mpn{nm}{{from{{--mp{nm}:{pct};}}to{{--mp{nm}:{creep_to};}}}}"
                f".mpbar{nm}{{animation:mpw{nm} {dur:.1f}s linear forwards;}}"
                f".mpnum{nm}{{counter-reset:mp var(--mp{nm});"
                f"animation:mpn{nm} {dur:.1f}s linear forwards;}}"
                f".mpnum{nm}::after{{content:counter(mp) '% complete';}}"
                f"</style>"
            )
            bar = (
                f'<div class="mochi-prog-bar mpbar{nm}" '
                f'style="background:{_pb};width:{pct}%;height:100%;border-radius:6px"></div>'
            )
            num = (
                f'<div class="mpnum{nm}" style="font-size:0.68rem;color:{_ppct};'
                f'margin-top:0.22rem;font-family:\'JetBrains Mono\',monospace"></div>'
            )
        else:
            style = ""
            bar = (
                f'<div class="mochi-prog-bar" style="background:{_pb};width:{pct}%;'
                f'height:100%;border-radius:6px;'
                f'transition:width 0.6s cubic-bezier(0.4,0,0.2,1)"></div>'
            )
            num = (
                f'<div style="font-size:0.68rem;color:{_ppct};margin-top:0.22rem;'
                f'font-family:\'JetBrains Mono\',monospace">{pct}% complete</div>'
            )
        _prog_slot.markdown(
            f'{style}'
            f'<div style="margin:0.5rem 0 0.4rem">'
            f'<div style="font-size:0.82rem;color:{_ptxt};margin-bottom:0.4rem;'
            f'font-family:Inter,sans-serif">{text}</div>'
            f'<div style="background:{_pt};border-radius:6px;height:6px;overflow:hidden">{bar}</div>'
            f'{num}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Reassurance ticker — rotates through fun facts about the pipeline so a
    # long-running stage never *looks* stuck. Rendered exactly once; its CSS
    # animation runs in the browser independently of the (possibly blocked)
    # Python thread.
    _FACTS = [
        "Hang tight — good rankings are worth the little wait…",
        "Crunching the numbers, one candidate at a time…",
        "Separating the gold from the glitter…",
        "Warming up the ranking engine…",
        "Almost there — lining up your top picks…",
        "Sifting through the stack so you don't have to…",
        "Reticulating splines… (just kidding, ranking candidates)",
        "Hold on — the best matches are rising to the top…",
    ]

    def _render_facts() -> None:
        n = len(_FACTS)
        dwell = 3.6                     # seconds each fact stays on screen
        total = n * dwell               # full loop duration
        vis = 100.0 / n                 # % of the loop one fact is visible
        fade = vis * 0.16               # fade-in / fade-out portion
        items = "".join(
            f'<div class="mfact" style="animation-delay:{i * dwell:.2f}s">'
            f'<span class="mfact-dot"></span>{html.escape(f)}</div>'
            for i, f in enumerate(_FACTS)
        )
        _fact_slot.markdown(
            f"<style>"
            f".mfact-wrap{{position:relative;height:1.35rem;margin:0 0 1.2rem;overflow:hidden;}}"
            f".mfact{{position:absolute;inset:0;display:flex;align-items:center;gap:0.5rem;"
            f"font-size:0.78rem;color:{_ptxt};font-family:Inter,sans-serif;opacity:0;"
            f"animation:mfactCycle {total:.1f}s infinite;}}"
            f".mfact-dot{{width:6px;height:6px;border-radius:50%;flex-shrink:0;"
            f"background:#3b82f6;box-shadow:0 0 8px rgba(59,130,246,0.7);"
            f"animation:mfactPulse 1.4s ease-in-out infinite;}}"
            f"@keyframes mfactPulse{{0%,100%{{opacity:0.4;}}50%{{opacity:1;}}}}"
            f"@keyframes mfactCycle{{"
            f"0%{{opacity:0;transform:translateY(5px);}}"
            f"{fade:.2f}%{{opacity:1;transform:translateY(0);}}"
            f"{vis - fade:.2f}%{{opacity:1;transform:translateY(0);}}"
            f"{vis:.2f}%{{opacity:0;transform:translateY(-5px);}}"
            f"100%{{opacity:0;transform:translateY(-5px);}}}}"
            f"</style>"
            f'<div class="mfact-wrap">{items}</div>',
            unsafe_allow_html=True,
        )

    _prog(2, "Initialising…", creep_to=5)
    _render_facts()

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
        _prog(6, "Loading artifacts…", creep_to=11)
        precomputed = load_precomputed(ARTIFACTS, load_candidate_artifacts=False)
        model = xgb.Booster()
        model.load_model(ARTIFACTS / "ranker_model.json")

        jd_text = ""
        hyp_path = ARTIFACTS / "hypothetical_resumes.json"
        if hyp_path.exists():
            import json as _json
            with open(hyp_path, encoding="utf-8") as _f:
                jd_text = _json.load(_f).get("jd_text", "")

        # Build dense + BM25 indexes from the uploaded candidates at runtime.
        # This is the single longest stage — let the bar creep toward 33 while
        # the (synchronous, thread-blocking) embedding runs.
        _prog(12, "Building runtime indexes — embedding candidates…", creep_to=34)
        bm25_data = attach_runtime_index(precomputed, candidates_dict, ARTIFACTS / "potion-base-8M")
        _prog(35, "Runtime indexes built")

        # Stage A: consistency checks
        _prog(40, "Stage A: consistency & honeypot detection…", creep_to=49)
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
        _prog(50, "Stage B: hybrid retrieval (BM25 + dense)…", creep_to=59)
        all_ids = list(candidates_dict.keys())
        bm25_ranking   = bm25_retrieve(bm25_data, all_ids, top_n=5000)
        dense_ranking  = dense_retrieve(precomputed, all_ids, top_n=5000)
        rrf_scores     = reciprocal_rank_fusion([bm25_ranking, dense_ranking])
        top_2000_ids   = sorted(rrf_scores, key=lambda c: -rrf_scores[c])[:2000]

        # Stage C: feature engineering on top-2000
        _prog(60, "Stage C: feature engineering on top 2000…", creep_to=69)
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
        _prog(70, "Stage D: XGBoost scoring…", creep_to=77)
        dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
        scores = model.predict(dmat)
        ranked_ids = [cid_order[i] for i in np.argsort(-scores)]

        # Stage E: cross-encoder re-rank on top 200
        _prog(78, "Stage E: cross-encoder re-rank…", creep_to=85)
        top_200_candidates = [candidates_dict[cid] for cid in ranked_ids[:200] if cid in candidates_dict]
        reranked = rerank_top_n(top_200_candidates, jd_text, n=200)
        reranked_ids = [cid for cid, _ in reranked]
        reranked_scores_map = {cid: score for cid, score in reranked}
        all_ranked_ids = reranked_ids + [cid for cid in ranked_ids[200:]]

        # Stage F: honeypot filter + JD hard gates → top 100
        _prog(86, "Stage F: hard gates…", creep_to=91)
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
        _prog(92, "Stage G: SHAP reasoning…", creep_to=99)
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

        _elapsed_s = time.time() - _t_start   # before the cosmetic sleep below
        _prog(100, "Pipeline complete.")
        time.sleep(1.0)
        _prog_slot.empty()
        _fact_slot.empty()

        # Cache results so theme toggle doesn't re-run the pipeline
        st.session_state.pipeline_results = {
            "results":          results,
            "honeypot_ids_list": list(honeypot_ids),
            "elapsed_s":        _elapsed_s,
            "n_total":          len(candidates_dict),
        }

    except FileNotFoundError as e:
        _prog_slot.empty()
        _fact_slot.empty()
        st.error(
            f"Artifact not found: {e}\n\n"
            "Run the offline pipeline first to generate `artifacts/`."
        )
        st.stop()
    except Exception:
        _prog_slot.empty()
        _fact_slot.empty()
        st.error("Ranking failed.")
        st.code(traceback.format_exc())
        st.stop()

# ------------------------------------------------------------------ #
# Results — loaded from session state (survives theme toggle rerun)
# ------------------------------------------------------------------ #
if st.session_state.pipeline_results is None:
    st.stop()

_pr             = st.session_state.pipeline_results
results         = _pr["results"]
honeypot_ids_list = _pr["honeypot_ids_list"]
_elapsed_s      = _pr["elapsed_s"]
n_total         = _pr["n_total"]
n_honeypots     = len(honeypot_ids_list)
n_finalists     = len(results)

# ------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------ #
st.markdown(
    '<div class="section-heading">'
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#e63946" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>'
    '</svg> Results</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="metric-row">
        <div class="metric-card blue">
            <div class="bg-glow"></div>
            <div class="label">Candidates Processed</div>
            <div class="value">{n_total}</div>
            <div class="sub">from uploaded file</div>
        </div>
        <div class="metric-card red">
            <div class="bg-glow"></div>
            <div class="label">Honeypots Detected</div>
            <div class="value">{n_honeypots}</div>
            <div class="sub">excluded from ranking</div>
        </div>
        <div class="metric-card green">
            <div class="bg-glow"></div>
            <div class="label">Finalists Ranked</div>
            <div class="value">{n_finalists}</div>
            <div class="sub">top candidates</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Speed badge
# ------------------------------------------------------------------ #
_thru = int(n_total / _elapsed_s) if _elapsed_s > 0 else 0
st.markdown(
    f'<div class="speed-badge">'
    f'⚡ Ranked <strong>{n_total:,}</strong> candidates in '
    f'<strong>{_elapsed_s:.1f}s</strong>'
    f'<span class="muted">&nbsp;·&nbsp;~{_thru:,}/s&nbsp;·&nbsp;CPU only</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Score distribution chart
# ------------------------------------------------------------------ #
st.markdown('<div class="results-heading">Score distribution</div>', unsafe_allow_html=True)
_dist_df = pd.DataFrame({
    "Rank":  [r["rank"]  for r in results],
    "Score": [r["score"] for r in results],
})
_chart = (
    alt.Chart(_dist_df)
    .mark_area(
        interpolate="monotone",
        line={"color": "#3b82f6", "strokeWidth": 2},
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color=_ch_stop0, offset=0),
                alt.GradientStop(color="#1d4ed8", offset=1),
            ],
            x1=1, x2=1, y1=1, y2=0,
        ),
    )
    .encode(
        x=alt.X("Rank:Q", title="Rank", axis=alt.Axis(grid=False, tickMinStep=1)),
        y=alt.Y("Score:Q", title="Model score", scale=alt.Scale(zero=False)),
        tooltip=["Rank:Q", alt.Tooltip("Score:Q", format=".4f")],
    )
    .properties(height=200, background="transparent")
    .configure_view(strokeOpacity=0, fill="transparent")
    .configure_axis(labelColor=_ch_ax, titleColor=_ch_ax, domainColor=_ch_dom, gridColor=_ch_grid)
)
st.altair_chart(_chart, use_container_width=True)

# ------------------------------------------------------------------ #
# Downloads — three grouped exports
# ------------------------------------------------------------------ #
def _results_to_csv(rows) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    for r in rows:
        writer.writerow({
            "candidate_id": r["candidate_id"],
            "rank":         r["rank"],
            "score":        f"{r['score']:.6f}",
            "reasoning":    r.get("reasoning", ""),
        })
    return buf.getvalue()

def _honeypots_to_csv(ids) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["candidate_id"])
    for cid in ids:
        writer.writerow([cid])
    return buf.getvalue()

_csv_all     = _results_to_csv(results)
_csv_top10   = _results_to_csv(results[:10])
_csv_honeypot = _honeypots_to_csv(honeypot_ids_list)

st.markdown('<div class="dl-group-label">Downloads</div>', unsafe_allow_html=True)
_dl1, _dl2, _dl3 = st.columns(3)
with _dl1:
    st.download_button(
        label=f"⬇  All {n_finalists} Finalists (CSV)",
        data=_csv_all,
        file_name="finalists_all.csv",
        mime="text/csv",
        use_container_width=True,
    )
with _dl2:
    st.download_button(
        label="⬇  Top 10 (CSV)",
        data=_csv_top10,
        file_name="finalists_top10.csv",
        mime="text/csv",
        use_container_width=True,
    )
with _dl3:
    st.download_button(
        label=f"⬇  Honeypot-Excluded ({n_honeypots}) (CSV)",
        data=_csv_honeypot,
        file_name="honeypots_excluded.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=(n_honeypots == 0),
    )

# ------------------------------------------------------------------ #
# Results table
# ------------------------------------------------------------------ #
st.markdown('<div class="results-heading">Rankings</div>', unsafe_allow_html=True)

def _rank_badge(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, str(rank))

def _build_rank_table(rows) -> str:
    """Scrollable HTML table with sticky header and full (wrapped) reasoning."""
    head = (
        "<thead><tr>"
        "<th class='c-rank'>Rank</th>"
        "<th class='c-id'>Candidate ID</th>"
        "<th class='c-score'>Score</th>"
        "<th class='c-reason'>Reasoning</th>"
        "</tr></thead>"
    )
    body = "".join(
        "<tr>"
        f"<td class='c-rank'>{_rank_badge(r['rank'])}</td>"
        f"<td class='c-id'>{html.escape(str(r['candidate_id']))}</td>"
        f"<td class='c-score'>{r['score']:.4f}</td>"
        f"<td class='c-reason'>{html.escape(str(r.get('reasoning', '')))}</td>"
        "</tr>"
        for r in rows
    )
    return f"<div class='rank-table-wrap'><table class='rank-table'>{head}<tbody>{body}</tbody></table></div>"

tab_all, tab_top10 = st.tabs([f"All {n_finalists} finalists", "Top 10"])

with tab_all:
    st.markdown(_build_rank_table(results), unsafe_allow_html=True)

with tab_top10:
    st.markdown(_build_rank_table(results[:10]), unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Honeypot details (expandable)
# ------------------------------------------------------------------ #
if honeypot_ids_list:
    with st.expander(f"🚫 Honeypots excluded ({len(honeypot_ids_list)})"):
        st.caption("These candidates failed consistency checks and were removed from ranking.")
        for cid in honeypot_ids_list:
            st.markdown(f"- `{cid}`")
