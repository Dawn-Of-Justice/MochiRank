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
import threading
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

    /* Popover trigger (score-band candidate-ID button) — subtle neutral chip,
       clearly visible on hover, never the loud red CTA. */
    [data-testid="stPopover"] button {{
        background: {ub_bg} !important; color: {ub_c} !important;
        border: 1px solid {ub_bd} !important; box-shadow: none !important;
        border-radius: 9px !important; padding: 0.45rem 0.95rem !important;
        font-weight: 500 !important; font-size: 0.82rem !important;
        letter-spacing: 0 !important; min-height: 0 !important;
        transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease !important;
    }}
    [data-testid="stPopover"] button:hover,
    [data-testid="stPopover"] button[aria-expanded="true"] {{
        background: {ub_hbg} !important; border-color: {ub_hbd} !important;
        color: {txstr} !important; transform: none !important; box-shadow: none !important;
    }}
    [data-testid="stPopoverBody"] {{
        background: {m_bg} !important; border: 1px solid {brd} !important;
        border-radius: 12px !important; box-shadow: {df_sh} !important;
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
    /* Keep the header on the subtle expander surface in every state — Streamlit
       otherwise flips it to a flat white background when open / hovered. */
    [data-testid="stExpander"] details > summary,
    [data-testid="stExpander"] details[open] > summary,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:focus,
    [data-testid="stExpander"] summary:active {{
        background: transparent !important; box-shadow: none !important;
        border-radius: 12px !important; color: {exp_lbl} !important;
    }}
    [data-testid="stExpander"] summary:hover {{ color: {txt} !important; }}
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {{ color: inherit !important; }}
    [data-testid="stExpander"] summary svg {{ fill: {exp_lbl} !important; color: {exp_lbl} !important; }}

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

    /* ── Candidate-ID chips (score-band drill-down) ── */
    .cid-band-head {{
        font-size: 0.82rem; color: {txts}; margin: 0.2rem 0 0.1rem;
        font-family: 'Inter', sans-serif;
    }}
    .cid-band-head b {{ color: {tbl_id}; font-family: 'JetBrains Mono', monospace; }}
    .cid-wrap {{ display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.5rem 0 0.4rem; }}
    .cid-chip {{
        display: inline-flex; align-items: center; gap: 0.5rem;
        background: {sel_bg}; border: 1px solid {sel_bd}; border-radius: 9px;
        padding: 0.32rem 0.6rem; font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem; color: {tbl_id}; white-space: nowrap;
        transition: border-color 0.15s ease, background 0.15s ease;
    }}
    .cid-chip:hover {{ border-color: {brdh}; }}
    .cid-chip .cid-rank {{ color: {txtm}; font-weight: 700; font-size: 0.68rem; }}
    .cid-chip .cid-score {{ color: {txts}; font-size: 0.68rem; }}

    /* ── HTML score histogram (hover = range + count, click = candidate IDs) ── */
    .hist-ylabel {{ font-size: 0.68rem; color: {ch_ax}; margin: 0 0 0.3rem; font-family: 'Inter', sans-serif; }}
    .hist {{ display: flex; align-items: stretch; gap: 5px; height: 240px; }}
    .hcol {{ flex: 1 1 0; display: flex; flex-direction: column; min-width: 0; }}
    .hbar-d {{ flex: 1; position: relative; }}
    .hbar-d > summary {{
        height: 100%; display: flex; flex-direction: column; justify-content: flex-end;
        cursor: pointer; list-style: none;
    }}
    .hbar-d > summary::-webkit-details-marker {{ display: none; }}
    .hbar {{
        width: 100%; border-radius: 5px 5px 0 0; min-height: 0;
        background: linear-gradient(180deg, #3b82f6 0%, #1d4ed8 100%);
        opacity: 0.9; transition: opacity 0.15s ease, filter 0.15s ease;
    }}
    .hbar-d > summary:hover .hbar {{ opacity: 1; filter: brightness(1.12); }}
    .hbar-d[open] .hbar {{ opacity: 1; outline: 2px solid {tbl_id}; outline-offset: 1px; }}
    .hbar-zero {{ background: {brd}; border-radius: 2px; opacity: 0.6; }}
    .hbar-tip, .hbar-pop {{
        position: absolute; background: {m_bg}; border: 1px solid {brd};
        box-shadow: {df_sh}; z-index: 30;
    }}
    .hbar-tip {{
        color: {txt}; font-size: 0.72rem; white-space: nowrap;
        padding: 0.3rem 0.55rem; border-radius: 7px; opacity: 0;
        pointer-events: none; transition: opacity 0.12s ease; font-family: 'Inter', sans-serif;
    }}
    .hbar-d > summary:hover .hbar-tip {{ opacity: 1; }}
    .hbar-d[open] .hbar-tip {{ display: none; }}
    .hbar-pop {{
        border-radius: 10px; padding: 0.55rem 0.65rem; width: max-content;
        max-width: 340px; max-height: 230px; overflow: auto; z-index: 40;
    }}
    /* Stack chips in a column so the panel hugs the widest chip — no ragged
       empty space on the right that a wrapping flex row would leave. */
    .hbar-pop .cid-wrap {{ flex-direction: column; align-items: flex-start; gap: 0.35rem; }}
    .hbar-tip.l, .hbar-pop.l {{ left: 0; }}
    .hbar-tip.r, .hbar-pop.r {{ right: 0; }}
    .hbar-pop-head {{
        font-size: 0.74rem; color: {tbl_id}; font-family: 'JetBrains Mono', monospace;
        margin-bottom: 0.45rem; white-space: nowrap;
    }}
    .hbar-empty {{ font-size: 0.74rem; color: {txtm}; }}
    .hcol-label {{
        font-size: 0.56rem; color: {ch_ax}; margin-top: 5px; text-align: center;
        white-space: nowrap; font-family: 'JetBrains Mono', monospace; overflow: hidden;
    }}
    .hist-xlabel {{ font-size: 0.68rem; color: {ch_ax}; margin-top: 0.35rem; text-align: right; font-family: 'Inter', sans-serif; }}
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
        help="JSONL (one candidate per line) or a JSON array.",
    )
    st.caption(
        "📎 **File larger than 500MB? Paste a Google Drive or direct URL below** — "
        "the server fetches it directly with no size limit."
    )
    url_input = st.text_input(
        "Load from URL (Google Drive, direct link, etc.)",
        placeholder="https://drive.google.com/file/d/…/view?usp=sharing",
        help="Supports Google Drive share links and any direct download URL.",
    ).strip()

with col_info:
    st.info(
        "**Format:** JSONL or JSON array  \n"
        "**Schema:** `candidate_schema.json`  \n\n"
        "**Tip:** Share your Google Drive file → copy link → paste above."
    )

# Resolve source: URL takes priority over file upload
_source_key = url_input if url_input else (uploaded.name if uploaded else None)

# Clear cached state when source changes or is removed
if _source_key is None:
    st.session_state.pipeline_results = None
    st.session_state.pop("_candidates", None)
    st.session_state.pop("_last_file", None)
    st.markdown(
        '<div class="empty-state">'
        '<div class="icon">📂</div>'
        '<h3>No file uploaded yet</h3>'
        '<p>Upload a candidates JSON or JSONL file above, or paste a URL.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()
elif st.session_state.get("_last_file") != _source_key:
    # New/changed source → drop the cached parse and any old results.
    st.session_state.pipeline_results = None
    st.session_state["_candidates"] = None
    st.session_state["_last_file"] = _source_key

# ------------------------------------------------------------------ #
# Parse source — done ONCE per file/URL and cached in session_state.
# ------------------------------------------------------------------ #
candidates = st.session_state.get("_candidates")
if candidates is None:
    _label = url_input if url_input else uploaded.name
    with st.spinner(f"Loading **{_label}**… large files take a moment, grab a coffee ☕"):
        try:
            import re, tempfile, os, urllib.request

            def _parse_file(path: str, name: str):
                """Parse JSONL or JSON array from a file path — stream line-by-line for JSONL."""
                with open(path, "r", encoding="utf-8") as fh:
                    first = fh.read(1)
                if first == "[":
                    with open(path, "r", encoding="utf-8") as fh:
                        return json.load(fh)
                else:
                    with open(path, "r", encoding="utf-8") as fh:
                        return [json.loads(ln) for ln in fh if ln.strip()]

            if url_input:
                _gdrive = re.search(r'drive\.google\.com', url_input)
                _tmp = tempfile.mktemp(suffix=".tmp")
                try:
                    if _gdrive:
                        import gdown
                        _id_match = (re.search(r'/d/([a-zA-Z0-9_-]+)', url_input) or
                                     re.search(r'[?&]id=([a-zA-Z0-9_-]+)', url_input))
                        _gdrive_url = (
                            f"https://drive.google.com/uc?export=download&id={_id_match.group(1)}"
                            if _id_match else url_input
                        )
                        gdown.download(_gdrive_url, _tmp, quiet=True)
                    else:
                        with urllib.request.urlopen(url_input, timeout=300) as resp:
                            with open(_tmp, "wb") as fh:
                                while chunk := resp.read(8 * 1024 * 1024):
                                    fh.write(chunk)
                    candidates = _parse_file(_tmp, url_input)
                finally:
                    if os.path.exists(_tmp):
                        os.unlink(_tmp)
            else:
                # File upload path: write to temp file to avoid double-buffering
                _tmp = tempfile.mktemp(suffix=".tmp")
                try:
                    uploaded.seek(0)
                    with open(_tmp, "wb") as fh:
                        fh.write(uploaded.read())
                    candidates = _parse_file(_tmp, uploaded.name)
                finally:
                    if os.path.exists(_tmp):
                        os.unlink(_tmp)
        except Exception as e:
            st.error(f"Could not load file: {e}")
            st.stop()

    if not isinstance(candidates, list):
        st.error("Expected a JSON array or JSONL file at the top level.")
        st.stop()

    st.session_state["_candidates"] = candidates

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

# ------------------------------------------------------------------ #
# Pipeline runs in a BACKGROUND THREAD so a theme toggle (which reruns the
# whole script) can never interrupt or reset it. The worker writes progress +
# the final payload into a shared dict held in session_state; the main script
# just polls and renders. Toggling the theme therefore stays instant and the
# run continues untouched.
# ------------------------------------------------------------------ #
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
_CREEP_RATE = 0.3   # %/sec the bar trickles within a stage so it never sits still


def _render_progress(slot, width: float, text: str) -> None:
    w = max(0.0, min(100.0, width))
    pct = int(round(w))
    slot.markdown(
        f'<div style="margin:0.5rem 0 0.4rem">'
        f'<div style="font-size:0.82rem;color:{_ptxt};margin-bottom:0.4rem;'
        f'font-family:Inter,sans-serif">{html.escape(text)}</div>'
        f'<div style="background:{_pt};border-radius:6px;height:6px;overflow:hidden">'
        f'<div class="mochi-prog-bar" style="background:{_pb};width:{w:.1f}%;'
        f'height:100%;border-radius:6px;transition:width 0.5s linear"></div>'
        f'</div>'
        f'<div style="font-size:0.68rem;color:{_ppct};margin-top:0.22rem;'
        f'font-family:\'JetBrains Mono\',monospace">{pct}% complete</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_facts(slot, elapsed: float) -> None:
    # Negative animation-delays keyed to `elapsed` keep the ticker rotating
    # continuously across the ~0.4s poll re-renders (otherwise each rerun would
    # restart the CSS animation and the ticker would freeze on the first fact).
    n = len(_FACTS)
    dwell = 3.6
    total = n * dwell
    vis = 100.0 / n
    fade = vis * 0.16
    items = "".join(
        f'<div class="mfact" style="animation-delay:{i * dwell - elapsed:.2f}s">'
        f'<span class="mfact-dot" style="animation-delay:{-elapsed:.2f}s"></span>'
        f'{html.escape(f)}</div>'
        for i, f in enumerate(_FACTS)
    )
    slot.markdown(
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


def _pipeline_worker(job: dict, candidates_dict: dict) -> None:
    """Run the full ranking pipeline (rank.py stages A–G) off the main thread.
    Writes progress + the final payload into ``job``; touches NO Streamlit APIs,
    so it is safe to run outside the script-run context."""
    def _stage(pct: int, text: str, creep_to: int = None) -> None:
        job["stage_start"] = time.time()
        job["pct"] = pct
        job["creep_to"] = creep_to
        job["text"] = text

    try:
        import xgboost as xgb
        for _k in list(sys.modules.keys()):
            if _k == "src" or _k.startswith("src."):
                del sys.modules[_k]
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

        _stage(6, "Loading artifacts…", 11)
        precomputed = load_precomputed(ARTIFACTS, load_candidate_artifacts=False)
        model = xgb.Booster()
        model.load_model(ARTIFACTS / "ranker_model.json")

        jd_text = ""
        hyp_path = ARTIFACTS / "hypothetical_resumes.json"
        if hyp_path.exists():
            with open(hyp_path, encoding="utf-8") as _f:
                jd_text = json.load(_f).get("jd_text", "")

        _stage(12, "Building runtime indexes — embedding candidates…", 34)
        bm25_data = attach_runtime_index(precomputed, candidates_dict, ARTIFACTS / "potion-base-8M")
        _stage(35, "Runtime indexes built")

        # Stage A: consistency checks
        _stage(40, "Stage A: consistency & honeypot detection…", 49)
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
        _stage(50, "Stage B: hybrid retrieval (BM25 + dense)…", 59)
        all_ids = list(candidates_dict.keys())
        bm25_ranking = bm25_retrieve(bm25_data, all_ids, top_n=5000)
        dense_ranking = dense_retrieve(precomputed, all_ids, top_n=5000)
        rrf_scores = reciprocal_rank_fusion([bm25_ranking, dense_ranking])
        top_2000_ids = sorted(rrf_scores, key=lambda c: -rrf_scores[c])[:2000]

        # Stage C: feature engineering on top-2000
        _stage(60, "Stage C: feature engineering on top 2000…", 69)
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
        _stage(70, "Stage D: XGBoost scoring…", 77)
        dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
        scores = model.predict(dmat)
        ranked_ids = [cid_order[i] for i in np.argsort(-scores)]

        # Stage E: cross-encoder re-rank on top 200
        _stage(78, "Stage E: cross-encoder re-rank…", 85)
        top_200_candidates = [candidates_dict[cid] for cid in ranked_ids[:200] if cid in candidates_dict]
        reranked = rerank_top_n(top_200_candidates, jd_text, n=200)
        reranked_ids = [cid for cid, _ in reranked]
        reranked_scores_map = {cid: score for cid, score in reranked}
        all_ranked_ids = reranked_ids + [cid for cid in ranked_ids[200:]]

        # Stage F: honeypot filter + JD hard gates → top 100
        _stage(86, "Stage F: hard gates…", 91)
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

        # Stage F2: finalist-level weighted honeypot gate (runs on ~100 only).
        # Backfills from the ranked pool.
        def _finalist_hp_score(c: dict) -> int:
            """
            Weighted honeypot score for finalists. >= 2 -> remove and backfill.
            2-pt: signup>last_active (impossible), expert>=12 (implausible inflation).
            1-pt: expert 10-11 (needs second signal to trigger).
            """
            score = 0
            sig2 = c.get("redrob_signals", {})
            skills2 = c.get("skills", [])
            su = sig2.get("signup_date", "")
            la = sig2.get("last_active_date", "")
            if su and la and su > la:
                score += 2
            expert_cnt = sum(1 for s in skills2 if s.get("proficiency") == "expert")
            if expert_cnt >= 12:
                score += 2
            elif expert_cnt >= 10:
                score += 1
            return score

        final_100_set = set(final_100)
        cleaned: list = []
        for cid in final_100:
            if _finalist_hp_score(candidates_dict[cid]) >= 2:
                skipped.add(cid)
                honeypot_ids.add(cid)  # count F2 removals alongside Stage A in UI
            else:
                cleaned.append(cid)

        if len(cleaned) < 100:
            for cid in all_ranked_ids:
                if len(cleaned) >= 100:
                    break
                if cid in final_100_set or cid in skipped or cid in honeypot_ids:
                    continue
                c = candidates_dict[cid]
                feat_dict = compute_features_dict(c, precomputed, violation_counts.get(cid, 0))
                disq, _ = _apply_jd_disqualifiers(c, feat_dict)
                if disq or _finalist_hp_score(c) >= 2:
                    continue
                cleaned.append(cid)

        final_100 = cleaned

        # Build unified scores and re-sort final_100 so rank matches score order.
        # Prefer reranker scores (bounded); fall back to XGBoost raw scores.
        # Normalize everything to [0, 1] via min-max.
        raw_score_map: dict = {}
        for cid in final_100:
            if cid in reranked_scores_map:
                raw_score_map[cid] = reranked_scores_map[cid]
            else:
                _idx = cid_to_matrix_idx.get(cid, -1)
                raw_score_map[cid] = float(scores[_idx]) if _idx >= 0 else 0.0

        _vals = list(raw_score_map.values())
        _s_min, _s_max = min(_vals), max(_vals)
        if _s_max > _s_min:
            normalized_score_map = {
                cid: (raw_score_map[cid] - _s_min) / (_s_max - _s_min)
                for cid in final_100
            }
        else:
            normalized_score_map = {cid: 1.0 for cid in final_100}

        final_100 = sorted(final_100, key=lambda cid: -normalized_score_map[cid])

        # Stage G: SHAP reasoning
        _stage(92, "Stage G: SHAP reasoning…", 99)
        shap_matrix = model.predict(dmat, pred_contribs=True)[:, :-1]

        results = []
        for rank_pos, cid in enumerate(final_100, start=1):
            c = candidates_dict[cid]
            score = normalized_score_map[cid]
            idx = cid_to_matrix_idx.get(cid, -1)
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

        job["payload"] = {
            "results":           results,
            "honeypot_ids_list": list(honeypot_ids),
            "elapsed_s":         time.time() - job["job_start"],
            "n_total":           len(candidates_dict),
        }
        job["pct"], job["creep_to"], job["text"] = 100, None, "Pipeline complete."
        job["status"] = "done"

    except FileNotFoundError as e:
        job["error"] = (
            f"Artifact not found: {e}\n\n"
            "Run the offline pipeline first to generate `artifacts/`."
        )
        job["status"] = "error"
    except Exception:
        job["error"] = traceback.format_exc()
        job["status"] = "error"


_job = st.session_state.get("_job")
_running = _job is not None and _job.get("status") == "running"

_run_pipeline = st.button("Rank Candidates", type="primary", use_container_width=False, disabled=_running)

# Launch the worker on click (guarded so a stray click mid-run can't double-start).
if _run_pipeline and not _running:
    candidates_dict = {c["candidate_id"]: c for c in candidates}
    _now = time.time()
    _job = {
        "status": "running", "pct": 2, "creep_to": 5, "text": "Initialising…",
        "stage_start": _now, "job_start": _now, "payload": None, "error": None,
    }
    st.session_state["_job"] = _job
    st.session_state.pipeline_results = None
    threading.Thread(target=_pipeline_worker, args=(_job, candidates_dict), daemon=True).start()
    _running = True

# While running: render progress from the shared dict, then poll again. The bar
# trickles in Python (time since the current stage started) so it keeps moving
# even during a long blocking stage.
if _running:
    _prog_slot = st.empty()
    _fact_slot = st.empty()
    _now = time.time()
    _base = _job["pct"]
    _cap = _job.get("creep_to")
    if _cap and _cap > _base:
        _w = min(float(_cap), _base + _CREEP_RATE * (_now - _job["stage_start"]))
    else:
        _w = float(_base)
    _render_progress(_prog_slot, _w, _job["text"])
    _render_facts(_fact_slot, _now - _job["job_start"])
    time.sleep(0.4)
    st.rerun()

if _job is not None and _job.get("status") == "error":
    st.session_state["_job"] = None
    st.error("Ranking failed.")
    st.code(_job["error"])
    st.stop()

if _job is not None and _job.get("status") == "done":
    st.session_state.pipeline_results = _job["payload"]
    st.session_state["_job"] = None

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
st.caption("Hover a bar for its score range and candidate count · click a bar to reveal the candidate IDs in it.")

# Pre-bin scores into fixed bands so each bar maps to the candidates in it.
_scores = np.array([r["score"] for r in results], dtype=float)
_n_bins = int(min(15, max(4, len(results) // 7))) if len(results) > 1 else 1
_lo, _hi = float(_scores.min()), float(_scores.max())
if _hi <= _lo:
    _hi = _lo + 1e-6
_edges = np.linspace(_lo, _hi, _n_bins + 1)
# digitize against interior edges so every score lands in [0, _n_bins-1]
_band_of = np.clip(np.digitize(_scores, _edges[1:-1], right=False), 0, _n_bins - 1)
_band_labels = [f"{_edges[i]:.3f} – {_edges[i + 1]:.3f}" for i in range(_n_bins)]

# results grouped by band, each sorted by rank
_bands: dict = {i: [] for i in range(_n_bins)}
for _r, _b in zip(results, _band_of):
    _bands[int(_b)].append(_r)
for _i in _bands:
    _bands[_i].sort(key=lambda x: x["rank"])

_counts = [len(_bands[i]) for i in range(_n_bins)]
_max_count = max(_counts) if _counts else 0
_BAR_AREA = 210  # px of vertical room the tallest bar fills


def _chips_html(rows) -> str:
    if not rows:
        return '<div class="hbar-empty">No candidates in this band.</div>'
    return '<div class="cid-wrap">' + "".join(
        f'<span class="cid-chip"><span class="cid-rank">#{r["rank"]}</span>'
        f'{html.escape(str(r["candidate_id"]))}'
        f'<span class="cid-score">{r["score"]:.3f}</span></span>'
        for r in rows
    ) + "</div>"


# Build the bars. Each bar is a <details>: hovering shows a tooltip (range +
# count), clicking opens a floating panel with the candidate IDs — all pure CSS,
# so there are no Streamlit reruns and clicks always work.
_cols = []
for _i in range(_n_bins):
    _cnt = _counts[_i]
    _h = round(_cnt / _max_count * _BAR_AREA) if _max_count else 0
    if _cnt > 0:
        _h = max(_h, 4)
    _rng = _band_labels[_i]
    _cnt_txt = f"{_cnt} candidate{'s' if _cnt != 1 else ''}"
    _side = "l" if _i < _n_bins / 2 else "r"          # avoid edge clipping
    _low = f"{_edges[_i]:.2f}"
    if _cnt > 0:
        _cols.append(
            '<div class="hcol"><details class="hbar-d" name="histband"><summary>'
            f'<span class="hbar-tip {_side}" style="bottom:{_h + 8}px">'
            f'{html.escape(_rng)} · {_cnt_txt}</span>'
            f'<span class="hbar" style="height:{_h}px"></span>'
            '</summary>'
            f'<div class="hbar-pop {_side}" style="bottom:{_h + 8}px">'
            f'<div class="hbar-pop-head">{html.escape(_rng)} · {_cnt_txt}</div>'
            f'{_chips_html(_bands[_i])}'
            '</div></details>'
            f'<div class="hcol-label">{_low}</div></div>'
        )
    else:
        _cols.append(
            '<div class="hcol"><div class="hbar-d">'
            '<span class="hbar hbar-zero" style="height:2px"></span></div>'
            f'<div class="hcol-label">{_low}</div></div>'
        )

st.markdown(
    '<div class="hist-ylabel">Number of candidates</div>'
    '<div class="hist">' + "".join(_cols) + '</div>'
    '<div class="hist-xlabel">Model score →</div>',
    unsafe_allow_html=True,
)

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
