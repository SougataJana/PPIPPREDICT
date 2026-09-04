"""
PPIP Explorer — Streamlit Enterprise Suite for Protein-Protein Interaction Prediction
Strictly validated against Ahmad & Mizuguchi (2011).
"""

import io
import time
import re
import zipfile

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from inference import load_models, run_prediction

# ---------------------------------------------------------------------------
# Shared figure typography
# ---------------------------------------------------------------------------
try:
    _PLOTLY_SUPPORTS_WEIGHT = tuple(int(x) for x in plotly.__version__.split(".")[:2]) >= (5, 23)
except Exception:
    _PLOTLY_SUPPORTS_WEIGHT = False


def _font(size: int = 14, color: str = "#B9C4D6", bold: bool = True) -> dict:
    """Plot font spec. `weight` only exists on plotly >= 5.23, so it is added
    conditionally and bold is carried by <b> tags in titles/annotations."""
    spec = dict(size=size, color=color, family="Plus Jakarta Sans, sans-serif")
    if bold and _PLOTLY_SUPPORTS_WEIGHT:
        spec["weight"] = "bold"
    return spec


@st.cache_resource
def get_models():
    return load_models("ppip_ensemble_weights.pt")

st.set_page_config(
    page_title="PPIP Explorer | Structural Biology Suite",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS styling
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp {
  background-color: #030712 !important;
  background-image:
      radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0, 242, 254, 0.12), transparent),
      radial-gradient(circle at 95% 20%, rgba(121, 40, 202, 0.1), transparent 40%),
      radial-gradient(circle at 5% 80%, rgba(255, 0, 128, 0.06), transparent 35%) !important;
  color: #f8fafc;
}

#MainMenu, footer, header { visibility: hidden; }

/* Pull the whole page up under the hidden Streamlit header */
.block-container { padding-top: 1.2rem !important; }

h1, h2, h3, h4 {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  letter-spacing: -0.01em;
  color: #f8fafc !important;
}

p, li, span, label, .stMarkdown { color: #94a3b8; }

::selection { background: rgba(0,242,254,0.35); }

.hero-title { font-size: 3.1rem; margin: 0 0 0.6rem 0; line-height: 1.05; font-weight: 800; text-align: center; }
.hero-title span { background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #7928ca 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero-sub, [data-testid="stMarkdownContainer"] p.hero-sub {
  font-size: 1.12rem !important; color: #94a3b8 !important; max-width: 820px !important;
  line-height: 1.55 !important; text-align: center !important;
  margin-left: auto !important; margin-right: auto !important;
  margin-top: 0 !important; margin-bottom: 2rem !important;
}
.hero-title, [data-testid="stMarkdownContainer"] h1.hero-title { text-align: center !important; }

/* Native Streamlit Container Border Styling (Replaces raw HTML ghost cards) */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.93) 0%, rgba(15, 23, 42, 0.72) 100%) !important;
  border: 1px solid rgba(0, 242, 254, 0.22) !important;
  border-top: 3px solid #00f2fe !important;
  border-radius: 20px !important;
  padding: 1.7rem 1.9rem !important;
  margin-bottom: 0 !important;
  box-shadow: 0 18px 45px -12px rgba(0, 0, 0, 0.65), inset 0 1px 0 rgba(255, 255, 255, 0.07) !important;
}

/* Card chrome, matched to the intro box. Several selectors so at least one
   hits whichever DOM this Streamlit build produces. */
.st-key-upload_card,
div[data-testid="stVerticalBlockBorderWrapper"]:has(.card-tag),
div[data-testid="stVerticalBlock"]:has(> div > div > .card-tag) {
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.93) 0%, rgba(15, 23, 42, 0.72) 100%) !important;
  border: 1px solid rgba(0, 242, 254, 0.22) !important;
  border-top: 3px solid #00f2fe !important;
  border-radius: 20px !important;
  padding: 1.7rem 1.9rem !important;
  margin-bottom: 0 !important;
  box-shadow: 0 18px 45px -12px rgba(0, 0, 0, 0.65), inset 0 1px 0 rgba(255, 255, 255, 0.07) !important;
}

/* Tagged cards: methodology + ingestion */
.card-tag { display: none; }
.stMarkdown:has(> div > .card-tag), .stMarkdown:has(.card-tag) { display: none !important; }

div[data-testid="stVerticalBlockBorderWrapper"]:has(.card-tag) {
  border: 1px solid rgba(0, 242, 254, 0.22) !important;
  border-top: 3px solid #00f2fe !important;
  border-radius: 20px !important;
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.93) 0%, rgba(15, 23, 42, 0.72) 100%) !important;
  padding: 1.7rem 1.9rem !important;
  margin-bottom: 0 !important;
  box-shadow: 0 18px 45px -12px rgba(0, 0, 0, 0.65), inset 0 1px 0 rgba(255, 255, 255, 0.07) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.card-tag) h4 {
  color: #00f2fe !important;
  font-size: 1.35rem !important;
  margin-bottom: 0.9rem !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.card-tag) h5 {
  color: #00f2fe !important;
  font-size: 1.25rem !important;
  margin-top: 0.4rem !important;
}

/* Uploader: bold, dark text in the white browse box */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button p,
[data-testid="stFileUploaderDropzone"] button span,
[data-testid="stFileUploaderDropzone"] button div {
  font-weight: 800 !important;
  color: #030712 !important;
}
[data-testid="stFileUploader"] label p {
  font-size: 1.02rem !important;
  color: #E2E8F0 !important;
}

/* Metric Glow Tiles */
[data-testid="stMetric"] {
  background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.005) 100%);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 16px;
  padding: 1.25rem 1rem;
  border-top: 2px solid #00f2fe;
}
[data-testid="stMetricLabel"] { color: #00f2fe !important; font-size: 0.7rem !important; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; color: #f8fafc !important; font-size: 1.8rem !important; }

/* Futuristic Holographic Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: none; background: rgba(15, 23, 42, 0.8); padding: 6px; border-radius: 14px; }
.stTabs [data-baseweb="tab"] {
  background: transparent; border: 1px solid transparent; border-radius: 10px;
  color: #94a3b8; padding: 0.5rem 1rem; font-weight: 600;
}
.stTabs [aria-selected="true"] { 
  background: linear-gradient(135deg, rgba(0, 242, 254, 0.15), rgba(121, 40, 202, 0.15)) !important; 
  border-color: rgba(0, 242, 254, 0.4) !important; 
  color: #00f2fe !important; 
  box-shadow: 0 0 20px rgba(0, 242, 254, 0.2);
}

/* High-Contrast Button Styling */
div.stButton > button:first-child {
  background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
  color: #030712 !important;
  font-weight: 800 !important;
  font-size: 1.05rem !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.8rem 1.5rem !important;
  box-shadow: 0 4px 20px rgba(0, 242, 254, 0.35) !important;
  width: 100% !important;
  transition: all 0.2s ease !important;
}
div.stButton > button:first-child p {
  color: #030712 !important;
  font-weight: 800 !important;
}
div.stButton > button:first-child:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 25px rgba(0, 242, 254, 0.6) !important;
  filter: brightness(1.05);
}

/* Results section headings: bold and bright */
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5, [data-testid="stMarkdownContainer"] h6 {
  font-weight: 800 !important;
}
[data-testid="stMarkdownContainer"] h3 { color: #f8fafc !important; font-size: 1.65rem !important; }
[data-testid="stMarkdownContainer"] h5 {
  color: #f8fafc !important; font-size: 1.22rem !important; letter-spacing: -0.01em !important;
}

/* Tab labels: bold on every element Streamlit/BaseWeb nests the text in */
.stTabs button, .stTabs button *,
button[data-baseweb="tab"], button[data-baseweb="tab"] *,
[data-baseweb="tab"], [data-baseweb="tab"] *,
[data-testid="stTabs"] button, [data-testid="stTabs"] button *,
[role="tab"], [role="tab"] * {
  font-weight: 800 !important;
}
.stTabs [data-baseweb="tab"], .stTabs [data-baseweb="tab"] * { color: #CBD5E1 !important; }
.stTabs [data-baseweb="tab"] p, .stTabs [data-baseweb="tab"] div,
.stTabs [data-baseweb="tab"] span { font-size: 0.98rem !important; }
.stTabs [aria-selected="true"], .stTabs [aria-selected="true"] * { color: #00f2fe !important; }

/* Reset button: fills its (narrow) column, so its right edge is the column's
   right edge — the same edge as the Runtime tile. No nudging needed. */
.st-key-new_pred button { white-space: nowrap !important; }

/* File Exports: bold, bright download buttons */
div.stDownloadButton > button, [data-testid="stDownloadButton"] button {
  background: rgba(15, 23, 42, 0.9) !important;
  border: 1.5px solid rgba(0, 242, 254, 0.55) !important;
  border-radius: 12px !important;
  padding: 0.7rem 1.1rem !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 2px 12px rgba(0,0,0,0.35) !important;
  transition: all 0.18s ease !important;
}
div.stDownloadButton > button p, [data-testid="stDownloadButton"] button p,
div.stDownloadButton > button div, [data-testid="stDownloadButton"] button div,
div.stDownloadButton > button span, [data-testid="stDownloadButton"] button span {
  color: #ffffff !important;
  font-weight: 800 !important;
  font-size: 1.02rem !important;
  letter-spacing: 0.01em !important;
}
div.stDownloadButton > button:hover, [data-testid="stDownloadButton"] button:hover {
  border-color: #00f2fe !important;
  background: rgba(0, 242, 254, 0.12) !important;
  transform: translateY(-1px);
}
div.stDownloadButton > button:hover p, [data-testid="stDownloadButton"] button:hover p {
  color: #00f2fe !important;
}

[data-testid="stFileUploaderDropzone"] {
  background: rgba(255,255,255,0.015);
  border: 1.5px dashed rgba(255,255,255,0.15);
  border-radius: 14px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Strict Engine Helpers & Adapters 
# ---------------------------------------------------------------------------
def _safe_decode(file_obj):
    bytes_data = file_obj.getvalue()
    try:
        return bytes_data.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return bytes_data.decode("latin-1").splitlines()

def _ensure_dense_matrix(results: dict):
    if "top_200" not in results:
        results["top_200"] = sorted(results["all_pairs"], key=lambda x: x[1], reverse=True)[:200]
        
    if "unique_r1" not in results:
        results["unique_r1"] = sorted(list(results["chain1"].keys()), key=lambda x: int(re.search(r"\d+", x).group()))
    if "unique_r2" not in results:
        results["unique_r2"] = sorted(list(results["chain2"].keys()), key=lambda x: int(re.search(r"\d+", x).group()))
        
    if "smoothed_matrix" not in results:
        mat = np.full((len(results["unique_r1"]), len(results["unique_r2"])), np.nan)
        pos1 = {lab: i for i, lab in enumerate(results["unique_r1"])}
        pos2 = {lab: i for i, lab in enumerate(results["unique_r2"])}
        for name, score in results["all_pairs"]:
            r1, r2 = name.split(":")
            if r1 in pos1 and r2 in pos2:
                mat[pos1[r1], pos2[r2]] = score
        results["smoothed_matrix"] = np.nan_to_num(mat, nan=0.0)
        
    if len(results["top_200"]) > 0:
        results["cutoff_score"] = results["top_200"][-1][1] - 1e-9
    else:
        results["cutoff_score"] = 0.0
        
    return results

def _build_svg(top_200_pairs: list, cutoff: float, label: str,
               name1: str = "Chain 1", name2: str = "Chain 2",
               font_size: int = 15, min_gap: float = 17.0) -> str:
    if not top_200_pairs:
        return ""

    pairs = []
    for name, score in top_200_pairs:
        if score <= cutoff:
            continue
        clean = name.replace(':', ' ').split()
        if len(clean) >= 2:
            res1, res2 = clean[0], clean[1]
            try:
                p1 = int(re.search(r"\d+", res1).group())
                p2 = int(re.search(r"\d+", res2).group())
                pairs.append((p1, p2, res1, res2, score))
            except Exception:
                continue

    if not pairs:
        return ""

    numres1 = max([p[0] for p in pairs]) + 5
    numres2 = max([p[1] for p in pairs]) + 5

    startx1 = startx2 = 110
    svglength = 1480
    starty1 = 175
    svgheight = 240
    starty2 = starty1 + svgheight
    endx1 = endx2 = startx1 + svglength
    endy1, endy2 = starty1, starty2

    def _x1(pos):
        return startx1 + (pos * svglength / numres1)

    def _x2(pos):
        return startx2 + (pos * svglength / numres2)

    # Best score per unique residue, so each label is emitted exactly once
    best1: dict[str, tuple[int, float]] = {}
    best2: dict[str, tuple[int, float]] = {}
    for p1, p2, res1, res2, score in pairs:
        if res1 not in best1 or score > best1[res1][1]:
            best1[res1] = (p1, score)
        if res2 not in best2 or score > best2[res2][1]:
            best2[res2] = (p2, score)

    def _pick_labels(best, x_of):
        """Highest-scoring residues win the space; anything closer than
        min_gap to an already-placed label keeps its tick but drops its text."""
        placed_x: list[float] = []
        chosen: list[tuple[str, float]] = []
        for lab, (pos, score) in sorted(best.items(), key=lambda kv: kv[1][1], reverse=True):
            x = x_of(pos)
            if all(abs(x - px) >= min_gap for px in placed_x):
                placed_x.append(x)
                chosen.append((lab, x))
        return chosen

    chosen1 = _pick_labels(best1, _x1)
    chosen2 = _pick_labels(best2, _x2)

    out = io.StringIO()
    out.write(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
              f'viewBox="0 0 {endx1 + 60} {starty2 + 175}" width="100%" height="100%" '
              f'font-family="JetBrains Mono, monospace">\n')

    # Chords first, so labels and ticks sit on top of them
    for p1, p2, res1, res2, score in pairs:
        out.write(f'<line x1="{_x1(p1):.2f}" y1="{starty1}" x2="{_x2(p2):.2f}" y2="{starty2}" '
                  f'stroke="#2563eb" stroke-width="1" stroke-opacity="0.45"></line>\n')

    # Axes
    out.write(f'<line x1="{startx1}" y1="{starty1}" x2="{endx1}" y2="{endy1}" style="stroke:#006600;stroke-width:2;"></line>\n')
    out.write(f'<line x1="{startx2}" y1="{starty2}" x2="{endx2}" y2="{endy2}" style="stroke:#006600;stroke-width:2;"></line>\n')

    # Ticks for every contacting residue, including the ones whose text was dropped
    for lab, (pos, score) in best1.items():
        x = _x1(pos)
        out.write(f'<line x1="{x:.2f}" y1="{starty1 - 5}" x2="{x:.2f}" y2="{starty1}" stroke="#00f2fe" stroke-width="1" stroke-opacity="0.75"></line>\n')
    for lab, (pos, score) in best2.items():
        x = _x2(pos)
        out.write(f'<line x1="{x:.2f}" y1="{starty2}" x2="{x:.2f}" y2="{starty2 + 5}" stroke="#c084fc" stroke-width="1" stroke-opacity="0.75"></line>\n')

    # Labels, rotated clear of the axis
    for lab, x in chosen1:
        out.write(f'<text x="{x:.2f}" y="{starty1 - 9}" transform="rotate(270 {x:.2f} {starty1 - 9})" '
                  f'font-size="{font_size}" font-weight="700" fill="#00f2fe" text-anchor="start">{lab}</text>\n')
    for lab, x in chosen2:
        out.write(f'<text x="{x:.2f}" y="{starty2 + 9}" transform="rotate(90 {x:.2f} {starty2 + 9})" '
                  f'font-size="{font_size}" font-weight="700" fill="#c084fc" text-anchor="start">{lab}</text>\n')

    # Chain captions and label-density note
    out.write(f'<text x="{startx1}" y="28" font-size="17" font-weight="700" fill="#EAF0F7">{label}</text>\n')
    out.write(f'<text x="{endx1}" y="28" font-size="13" font-weight="600" fill="#8B95A7" text-anchor="end">'
              f'labels shown: {len(chosen1)}/{len(best1)} (top) &#183; {len(chosen2)}/{len(best2)} (bottom)</text>\n')
    out.write(f'<text x="{startx1 - 10}" y="{starty1 + 5}" font-size="16" font-weight="700" fill="#00f2fe" text-anchor="end">{name1}</text>\n')
    out.write(f'<text x="{startx2 - 10}" y="{starty2 + 5}" font-size="16" font-weight="700" fill="#c084fc" text-anchor="end">{name2}</text>\n')

    out.write("</svg>\n")
    return out.getvalue()

def _build_circular_plot(top_200_pairs: list, cutoff: float, name1: str = "Chain 1",
                         name2: str = "Chain 2", font_size: int = 14,
                         min_sep_deg: float = 4.5, max_labels: int = 60) -> go.Figure:
    pairs = []
    for name, score in top_200_pairs:
        if score > cutoff:
            try:
                r1, r2 = name.split(":")
                p1 = int(re.search(r"\d+", r1).group())
                p2 = int(re.search(r"\d+", r2).group())
                pairs.append((p1, p2, r1, r2, name, score))
            except Exception:
                continue

    fig = go.Figure()
    if not pairs:
        return fig

    max_p1 = max([p[0] for p in pairs])
    max_p2 = max([p[1] for p in pairs])
    numres1 = max_p1 + 5
    numres2 = max_p2 + 5

    def _angle1(pos):
        return (np.pi - 0.2) - (pos / numres1) * (np.pi - 0.4)

    def _angle2(pos):
        return (np.pi + 0.2) + (pos / numres2) * (np.pi - 0.4)

    theta1 = np.linspace(np.pi - 0.2, 0.2, 100)
    theta2 = np.linspace(np.pi + 0.2, 2*np.pi - 0.2, 100)

    fig.add_trace(go.Scatter(x=np.cos(theta1), y=np.sin(theta1), mode="lines", line=dict(color="#006600", width=4), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=np.cos(theta2), y=np.sin(theta2), mode="lines", line=dict(color="#006600", width=4), hoverinfo="skip"))

    # Chords
    for p1, p2, r1, r2, name, score in pairs:
        a1, a2 = _angle1(p1), _angle2(p2)
        fig.add_trace(go.Scatter(
            x=[np.cos(a1), np.cos(a2)], y=[np.sin(a1), np.sin(a2)], mode="lines",
            line=dict(color="rgba(0, 102, 255, 0.4)", width=1),
            hoverinfo="text", text=f"{name} (Score: {score:.4f})"
        ))

    # Residue nodes: keep the best score seen for each residue on either arc
    best1: dict[str, tuple[int, float]] = {}
    best2: dict[str, tuple[int, float]] = {}
    for p1, p2, r1, r2, name, score in pairs:
        if r1 not in best1 or score > best1[r1][1]:
            best1[r1] = (p1, score)
        if r2 not in best2 or score > best2[r2][1]:
            best2[r2] = (p2, score)

    def _add_arc_nodes(best, angle_fn, colour):
        labels = list(best.keys())
        ang = np.array([angle_fn(best[l][0]) for l in labels])
        fig.add_trace(go.Scatter(
            x=np.cos(ang), y=np.sin(ang), mode="markers",
            marker=dict(size=6, color=colour, line=dict(width=0)),
            hoverinfo="text",
            text=[f"{l} (best score: {best[l][1]:.4f})" for l in labels],
        ))

        # Highest-scoring residues claim the space; anything angularly closer
        # than min_sep keeps its node but drops its text.
        min_sep = np.radians(min_sep_deg)
        placed: list[float] = []
        for lab, (pos, score) in sorted(best.items(), key=lambda kv: kv[1][1], reverse=True):
            if len(placed) >= max_labels:
                break
            a = angle_fn(pos)
            if any(abs(a - pa) < min_sep for pa in placed):
                continue
            placed.append(a)
            deg = np.degrees(a)
            if np.cos(a) >= 0:
                textangle, xanchor = -deg, "left"
            else:
                textangle, xanchor = -deg + 180, "right"
            fig.add_annotation(
                x=1.05 * np.cos(a), y=1.05 * np.sin(a), text=f"<b>{lab}</b>", showarrow=False,
                textangle=textangle, xanchor=xanchor, yanchor="middle",
                font=dict(size=font_size, color=colour, family="JetBrains Mono, monospace"),
            )
        return len(placed), len(best)

    shown1, total1 = _add_arc_nodes(best1, _angle1, "#00f2fe")
    shown2, total2 = _add_arc_nodes(best2, _angle2, "#c084fc")

    fig.add_annotation(x=0, y=1.45, text=f"<b>{name1}</b>", showarrow=False,
                       font=_font(18, "#00f2fe"))
    fig.add_annotation(x=0, y=-1.45, text=f"<b>{name2}</b>", showarrow=False,
                       font=_font(18, "#c084fc"))
    fig.add_annotation(xref="paper", yref="paper", x=1, y=1, xanchor="right", yanchor="top",
                       showarrow=False,
                       text=f"<b>labels shown: {shown1}/{total1} (top) &#183; {shown2}/{total2} (bottom)</b>",
                       font=_font(13, "#8B95A7"))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, font=_font(14),
        xaxis=dict(visible=False, range=[-1.70, 1.70]),
        yaxis=dict(visible=False, range=[-1.70, 1.70], scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=10, b=10), height=700
    )
    return fig


CARD_BOX = (
    "background: linear-gradient(180deg, rgba(15,23,42,0.93) 0%, rgba(15,23,42,0.72) 100%);"
    "border: 1px solid rgba(0,242,254,0.22); border-top: 3px solid #00f2fe;"
    "border-radius: 20px; padding: 1.7rem 1.9rem; margin-bottom: 0;"
    "box-shadow: 0 18px 45px -12px rgba(0,0,0,0.65), inset 0 1px 0 rgba(255,255,255,0.07);"
)
CARD_TEXT = ("color:#B9C4D6; font-size:1.02rem; line-height:1.68; margin-bottom:1rem;"
             " text-align:justify; text-justify:inter-word;")
CARD_HEAD = "color:#00f2fe !important; font-size:1.35rem; font-weight:700; margin:0 0 1rem 0;"


def _hotspot_table_html(pairs: list) -> str:
    """Ranked hotspots as plain HTML: alignment and colours are fully ours,
    so Streamlit's table CSS cannot override them and there is no toolbar."""
    scores = [sc for _, sc in pairs]
    smin, smax = (min(scores), max(scores)) if scores else (0.0, 1.0)
    span = (smax - smin) or 1.0

    rows = []
    for i, (name, score) in enumerate(pairs, 1):
        pct = 6 + ((score - smin) / span) * 94
        rows.append(
            f'<tr><td class="rk">{i}</td><td class="pr">{name}</td>'
            f'<td class="sc"><span class="bar" style="width:{pct:.1f}%"></span>'
            f'<span class="val">{score:.6f}</span></td></tr>'
        )

    return """
<style>
.hs-wrap { max-height: 520px; overflow-y: auto; border: 1px solid rgba(0,242,254,0.20);
           border-radius: 14px; background: rgba(15,23,42,0.75); }
.hs-wrap::-webkit-scrollbar { width: 10px; }
.hs-wrap::-webkit-scrollbar-thumb { background: rgba(0,242,254,0.35); border-radius: 8px; }
table.hs { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; }
table.hs th { position: sticky; top: 0; z-index: 2; background: #0b1220;
              color: #00f2fe !important; font-size: 0.95rem; font-weight: 800;
              padding: 12px 16px;
              border-bottom: 2px solid #01050e; border-right: 1px solid #01050e; }
table.hs td { padding: 9px 16px; font-size: 0.95rem; font-weight: 600;
              color: #EAF0F7 !important;
              border-bottom: 1px solid #01050e; border-right: 1px solid #01050e; }
table.hs th:last-child, table.hs td:last-child { border-right: none; }
table.hs th.rk, table.hs td.rk { text-align: right; width: 70px; color: #7C8798 !important; }
table.hs th.pr, table.hs td.pr { text-align: left; }
table.hs td.pr { color: #f8fafc !important; font-weight: 700; }
table.hs th.sc, table.hs td.sc { text-align: right; width: 46%; }
table.hs td.sc { position: relative; }
table.hs td.sc .bar { position: absolute; right: 0; top: 0; bottom: 0;
                      background: linear-gradient(90deg, rgba(0,242,254,0.04), rgba(0,242,254,0.30)); }
table.hs td.sc .val { position: relative; color: #ffffff !important; font-weight: 700; }
table.hs tr:hover td { background: rgba(0,242,254,0.07); }
</style>
<div class="hs-wrap"><table class="hs">
<thead><tr><th class="rk">Rank</th><th class="pr">Residue Pairs</th><th class="sc">Score</th></tr></thead>
<tbody>""" + "".join(rows) + "</tbody></table></div>"


SECTION_GAP = "2rem"


def _gap(height: str = SECTION_GAP):
    """One explicit spacer, so every section break is the same height
    regardless of what margins Streamlit's own containers carry."""
    st.markdown(f'<div style="height:{height};"></div>', unsafe_allow_html=True)


def _card(key: str | None = None):
    """Bordered container. `key` (Streamlit >= 1.39) puts a st-key-<key> class on
    the wrapper; the marker span is a fallback hook for older builds."""
    try:
        box = st.container(border=True, key=key) if key else st.container(border=True)
    except TypeError:  # container(key=...) predates this Streamlit
        box = st.container(border=True)
    with box:
        st.markdown('<span class="card-tag"></span>', unsafe_allow_html=True)
    return box


def _render_reference():
    with st.container(border=True):
        st.markdown("#### Reference")
        st.markdown('<div style="background: rgba(0, 242, 254, 0.05); border: 1px solid rgba(0, 242, 254, 0.2); padding: 15px; border-radius: 8px;">📖 <a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0029104" target="_blank" style="color: #00f2fe; text-decoration: none;">Ahmad S, Mizuguchi K (2011). Partner-Aware Prediction of Interacting Residues in Protein-Protein Complexes from Sequence Data. PLoS ONE 6(12): e29104.</a></div>', unsafe_allow_html=True)

def _write_legacy_files(results: dict, name1: str, name2: str,
                        elapsed: float | None = None,
                        figures: dict | None = None) -> dict[str, bytes]:
    """Every result the app shows, as downloadable files. Buffers are written
    and released one at a time to keep peak memory low."""
    files = {}
    stem = f"{name1}-{name2}"

    # 1. All scored pairs (legacy final-prediction)
    buf = io.StringIO()
    buf.write("Pair(Seq1:Seq2)\tPrediction-score\n")
    for name, score in results["all_pairs"]:
        buf.write(f"{name}\t{score:.6f}\n")
    files[f"{stem}-final-prediction.txt"] = buf.getvalue().encode()
    buf.close()

    # 2. Top 200 ranked pairs, with rank column
    buf = io.StringIO()
    buf.write("Rank\tResidue_pair(Seq1:Seq2)\tPrediction-score\n")
    for i, (name, score) in enumerate(results["top_200"], 1):
        buf.write(f"{i}\t{name}\t{score:.6f}\n")
    files[f"{stem}-top200.tsv"] = buf.getvalue().encode()
    buf.close()

    # 3. Legacy per-chain profiles
    buf = io.StringIO()
    for res, score in results["chain1"].items():
        buf.write(f"{res} {score:.6f}\n")
    files[f"{stem}-sspred.chain1"] = buf.getvalue().encode()
    buf.close()

    buf = io.StringIO()
    for res, score in results["chain2"].items():
        buf.write(f"{res} {score:.6f}\n")
    files[f"{stem}-sspred.chain2"] = buf.getvalue().encode()
    buf.close()

    # 4. Residue-wise propensities, both chains in one table
    buf = io.StringIO()
    buf.write("Protein\tResidue\tMax_score\n")
    for r in results["unique_r1"]:
        buf.write(f"{name1}\t{r}\t{results['chain1'][r]:.6f}\n")
    for r in results["unique_r2"]:
        buf.write(f"{name2}\t{r}\t{results['chain2'][r]:.6f}\n")
    files[f"{stem}-residue-propensities.tsv"] = buf.getvalue().encode()
    buf.close()

    # 5. Full score matrix (rows = target residues, columns = partner residues)
    buf = io.StringIO()
    buf.write("Residue\t" + "\t".join(results["unique_r2"]) + "\n")
    mat = results["smoothed_matrix"]
    for i, r in enumerate(results["unique_r1"]):
        buf.write(r + "\t" + "\t".join(f"{v:.6f}" for v in mat[i]) + "\n")
    files[f"{stem}-score-matrix.tsv"] = buf.getvalue().encode()
    buf.close()

    # 6. Run summary
    top_pair, top_score = results["top_200"][0]
    buf = io.StringIO()
    buf.write("Metric\tValue\n")
    buf.write(f"Target_protein\t{name1}\n")
    buf.write(f"Partner_protein\t{name2}\n")
    buf.write(f"Sequence_geometry\t{len(results['unique_r1'])} x {len(results['unique_r2'])}\n")
    buf.write(f"Scored_pairs\t{len(results['all_pairs'])}\n")
    buf.write(f"Top200_cutoff_score\t{results['cutoff_score']:.6f}\n")
    buf.write(f"Peak_pair\t{top_pair}\n")
    buf.write(f"Peak_score\t{top_score:.6f}\n")
    if elapsed is not None:
        buf.write(f"Runtime_seconds\t{elapsed:.2f}\n")
    files[f"{stem}-summary.tsv"] = buf.getvalue().encode()
    buf.close()

    # 7. Linear contact map
    files[f"{stem}-contact-map.svg"] = _build_svg(
        results["top_200"], results["cutoff_score"], stem, name1=name1, name2=name2).encode()

    # 8. Interactive figures as self-contained HTML (plotly.js pulled from CDN,
    #    so each file stays small)
    for label, fig_obj in (figures or {}).items():
        try:
            files[f"{stem}-{label}.html"] = fig_obj.to_html(
                include_plotlyjs="cdn", full_html=True).encode()
        except Exception:
            pass

    # 9. Everything above, zipped
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, data in files.items():
            zf.writestr(fname, data)
    files[f"{stem}-all-results.zip"] = zbuf.getvalue()
    zbuf.close()

    return files

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<h1 class="hero-title">PPIP<span>P Explorer</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Artificial Neural Network engine for Protein-Protein Interaction from Partner-aware Prediction.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Landing view: methodology first, then ingestion. Replaced entirely by the
# results view once a prediction has been executed.
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    # Background + pipeline architecture, one card
    st.markdown(
        f'''<div style="{CARD_BOX}">
             <p style="{CARD_TEXT}">Computational prediction of protein–protein interaction (PPI) interfaces remains a significant challenge in systems and structural biology. Conventional machine-learning approaches are often partner-unaware, predicting potential binding sites on individual proteins without considering their specific interaction partners. This web server implements a partner-aware approach developed by Professor Shandar Ahmad and Kenji Mizuguchi. The method simultaneously analyzes sequence-derived Position-Specific Scoring Matrices (PSSMs) from both the target and partner proteins to identify complementary residue-pairing patterns indicative of PPI interfaces. By explicitly incorporating information from both interacting proteins, the approach substantially reduces false-positive predictions and improves the specificity of interface identification, as a binding site is predicted only when the corresponding partner contains a compatible interface region.</p>
             <p style="{CARD_TEXT}">This partner-aware strategy has broad potential applications in disease research and drug development, particularly for investigating disease-associated protein interactions and identifying functionally relevant interfaces that may serve as therapeutic targets. By enabling more precise characterization of PPI interfaces, the server can support the discovery and development of selective PPI modulators and facilitate structure-guided therapeutic design.</p>
             <hr style="border:none; border-top:1px solid rgba(255,255,255,0.12); margin:1.5rem 0;">
             <h4 style="{CARD_HEAD}">Pipeline Architecture</h4>
             <ol style="{CARD_TEXT} padding-left:1.3rem; margin-bottom:0;"><li style="margin-bottom:0.75rem;"><b style="color:#f8fafc;">Stage-1 Composition:</b> Extract the pattern (sparse sequence encoding and PSSM-based evolutionary profile) features from the protein pair.</li><li style="margin-bottom:0.75rem;"><b style="color:#f8fafc;">Neural Network:</b> Consider multiple window sizes (0, 1, 3, 5, 7) across sequences to capture the local neighborhood impact of protein pairs and train 24 distinct Artificial Neural Networks to score candidate interactions.</li><li style="margin-bottom:0.75rem;"><b style="color:#f8fafc;">Stage-2 Composition:</b> The parallel predictions are concatenated column-wise, fusing the 24 independent neural network outputs.</li><li style="margin-bottom:0.75rem;"><b style="color:#f8fafc;">Final Ranking:</b> Pair-wise scores are ranked directly (unsmoothed) to select the top 200 candidate interactions, following Ahmad &amp; Mizuguchi (2011).</li><li style="margin-bottom:0.75rem;"><b style="color:#f8fafc;">Visualization Smoothing (app-only):</b> For the heatmap and 3D views only, a moving-average filter is applied for visual clarity. This step is not part of the original published method and has no effect on the ranked target-partner protein pairs mentioned above.</li></ol>
           </div>''',
        unsafe_allow_html=True)

    _gap()
    st.markdown(
        '<h4 style="color:#00f2fe; font-size:1.35rem; font-weight:700; margin:0 0 0.6rem 0;">'
        'Upload PSSM Profiles</h4>', unsafe_allow_html=True)
    with _card(key="upload_card"):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            file1 = st.file_uploader("**UPLOAD** Protein 1 (Target PSSM)", type=None, key="f1")
        with col2:
            file2 = st.file_uploader("**UPLOAD** Protein 2 (Partner PSSM)", type=None, key="f2")

        st.markdown("<br>", unsafe_allow_html=True)
        run_clicked = st.button("Execute Interaction Prediction Pipeline", type="primary", disabled=not (file1 and file2))

    # -----------------------------------------------------------------------
    # Main Execution Logic
    # -----------------------------------------------------------------------
    if run_clicked and file1 and file2:
        lines1 = _safe_decode(file1)
        lines2 = _safe_decode(file2)

        progress = st.progress(0.0, text="Initializing tensor batches...")

        def _progress_cb(frac, msg):
            progress.progress(frac, text=msg)

        t0 = time.time()
        with st.spinner("Scoring candidate interactions across 24-network ensemble..."):
            try:
                from inference import run_prediction

                results = run_prediction(lines1, lines2, models=get_models())
                results = _ensure_dense_matrix(results)
            except Exception as e:
                progress.empty()
                st.error(f"Prediction failed: {e}")
                st.stop()

        elapsed = time.time() - t0
        progress.empty()

        st.session_state["results"] = results
        st.session_state["name1"] = file1.name
        st.session_state["name2"] = file2.name
        st.session_state["elapsed"] = elapsed

        # Rerun so the landing view is replaced by the results view only
        st.rerun()

    _gap("0.6rem")   # tighter than SECTION_GAP: pulls the reference block up
    _render_reference()
    st.stop()

# ---------------------------------------------------------------------------
# Results view
# ---------------------------------------------------------------------------
results = st.session_state["results"]
name1 = st.session_state["name1"]
name2 = st.session_state["name2"]
elapsed = st.session_state["elapsed"]

# ---------------------------------------------------------------------------
# Results Metrics
# ---------------------------------------------------------------------------
try:
    hcol, rcol = st.columns([4, 1], vertical_alignment="center")
except TypeError:  # vertical_alignment lands in Streamlit 1.36
    hcol, rcol = st.columns([4, 1])
with hcol:
    st.markdown("### Executed Results")
    st.markdown(
        f'<p style="font-weight:700; color:#B9C4D6; font-size:1rem; margin:0.15rem 0 0 0;">'
        f'{name1} vs {name2}</p>', unsafe_allow_html=True)
with rcol:
    if st.button("Go for New Prediction", key="new_pred"):
        for _k in ("results", "name1", "name2", "elapsed", "f1", "f2"):
            st.session_state.pop(_k, None)
        st.rerun()

_gap("1rem")

top_pair, top_score = results["top_200"][0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Sequence Geometry", f"{len(results['unique_r1'])} × {len(results['unique_r2'])}")
m2.metric("Top 200 Cutoff Score", f"{results['cutoff_score']:.4f}")
m3.metric("Peak Interaction", f"{top_score:.3f}", top_pair)
m4.metric("Runtime", f"{elapsed:.2f}s")

_gap("1.6rem")

# ---------------------------------------------------------------------------
# Strict Authentic Tabs
# ---------------------------------------------------------------------------
FIGS: dict = {}

tab_top, tab_chain, tab_heat, tab_3d, tab_dist, tab_diagram, tab_circ, tab_downloads = st.tabs([
    "Interacting Residue Pairs", "Residue wise Propensities", "2D Heatmap", "3D Landscape", 
    "Score Distribution", "Linear Contact Map", "Circular Contact Map", "Export Result Files"
])

with tab_top:
    st.markdown("##### Top 200 Candidate Residue Pairs")
    st.markdown(
        '<p style="color:#94a3b8; font-size:0.97rem; margin:0.35rem 0 1.1rem 0;">'
        'Maximum score may highlight the specific interaction.</p>', unsafe_allow_html=True)
    tcol, _spacer = st.columns([3, 2])
    with tcol:
        st.markdown(_hotspot_table_html(results["top_200"]), unsafe_allow_html=True)

with tab_chain:
    st.markdown("##### Per-Residue Interface Propensity Profiles")
    st.caption("Maximum interaction score achieved by each residue against any partner protein.")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{name1}**")
        c1_df = pd.DataFrame([(r, results["chain1"][r]) for r in results["unique_r1"]], columns=["Residue", "Score"])
        fig1 = go.Figure(go.Scatter(x=c1_df["Residue"], y=c1_df["Score"], mode="lines", line=dict(color="#00f2fe", width=2)))
        fig1.update_layout(
            height=400, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=_font(14, "#B9C4D6"),
            xaxis=dict(title=dict(text="<b>Residue</b>", font=_font(15, "#00f2fe")), tickfont=_font(13)),
            yaxis=dict(title=dict(text="<b>Max score</b>", font=_font(15, "#00f2fe")), tickfont=_font(13)),
        )
        FIGS["propensity-target"] = fig1
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown(f"**{name2}**")
        c2_df = pd.DataFrame([(r, results["chain2"][r]) for r in results["unique_r2"]], columns=["Residue", "Score"])
        fig2 = go.Figure(go.Scatter(x=c2_df["Residue"], y=c2_df["Score"], mode="lines", line=dict(color="#c084fc", width=2)))
        fig2.update_layout(
            height=400, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=_font(14, "#B9C4D6"),
            xaxis=dict(title=dict(text="<b>Residue</b>", font=_font(15, "#c084fc")), tickfont=_font(13)),
            yaxis=dict(title=dict(text="<b>Max score</b>", font=_font(15, "#c084fc")), tickfont=_font(13)),
        )
        FIGS["propensity-partner"] = fig2
        st.plotly_chart(fig2, use_container_width=True)

with tab_heat:
    st.markdown("##### Smoothed Interaction Score Matrix")
    mat = results["smoothed_matrix"]
    fig = go.Figure(data=go.Heatmap(
        z=mat, x=results["unique_r2"], y=results["unique_r1"],
        colorscale=[[0, "#030712"], [0.25, "#1e1b4b"], [0.5, "#0284c7"], [0.75, "#00f2fe"], [1.0, "#f43f5e"]],
        colorbar=dict(title=dict(text="<b>Score</b>", font=_font(15)), tickfont=_font(13)),
    ))
    fig.update_layout(
        height=740, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=_font(14),
        xaxis=dict(title=dict(text=f"<b>{name2}</b>", font=_font(16, "#c084fc")), tickfont=_font(12)),
        yaxis=dict(title=dict(text=f"<b>{name1}</b>", font=_font(16, "#00f2fe")), tickfont=_font(12)),
    )
    FIGS["2d-heatmap"] = fig
    st.plotly_chart(fig, use_container_width=True)

with tab_3d:
    st.markdown("##### 3D Interaction Energy Landscape")
    st.caption("Surface projection of the exact smoothed matrix array.")
    mat = results["smoothed_matrix"]
    fig3d = go.Figure(data=[go.Surface(
        z=mat,
        colorscale="viridis",
        hovertemplate="Score: %{z:.4f}<extra></extra>",
    )])
    fig3d.update_layout(
        height=740, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", font=_font(14),
        scene=dict(
            xaxis=dict(title=dict(text=f"<b>{name2} (index)</b>", font=_font(15, "#c084fc")), tickfont=_font(12)),
            yaxis=dict(title=dict(text=f"<b>{name1} (index)</b>", font=_font(15, "#00f2fe")), tickfont=_font(12)),
            zaxis=dict(title=dict(text="<b>Score</b>", font=_font(15)), tickfont=_font(12)),
        ),
    )
    FIGS["3d-landscape"] = fig3d
    st.plotly_chart(fig3d, use_container_width=True)

with tab_dist:
    st.markdown("##### Score Distribution Histogram")
    st.caption("Descriptive distribution of all calculated pair scores, separating interaction signal from background.")
    scores = [s for _, s in results["all_pairs"]]
    fig_hist = go.Figure(data=[go.Histogram(x=scores, nbinsx=100, marker_color="#00f2fe", opacity=0.8)])
    fig_hist.update_layout(
        height=500, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=_font(14),
        xaxis=dict(title=dict(text="<b>Interaction Score</b>", font=_font(16)), tickfont=_font(13)),
        yaxis=dict(title=dict(text="<b>Frequency Count</b>", font=_font(16)), tickfont=_font(13)),
    )
    FIGS["score-distribution"] = fig_hist
    st.plotly_chart(fig_hist, use_container_width=True)

with tab_diagram:
    st.markdown("##### Bipartite Interaction Wiring Diagram")
    st.caption("Top 200 pairs, drawn on the same geometry as the legacy get-svg.sh. Every contacting residue gets a tick on its axis; text labels are deduplicated and thinned so they never sit on top of each other, with the highest-scoring residues keeping their label.")
    lc1, lc2 = st.columns(2)
    with lc1:
        gap = st.slider("Minimum label spacing (px)", 10, 50, 17, 1,
                        help="Raise this to thin out labels in crowded regions. Ticks stay for every residue.")
    with lc2:
        fsize = st.slider("Label font size", 10, 26, 15, 1)
    svg_str = _build_svg(results["top_200"], results["cutoff_score"], f"{name1}-{name2}",
                         name1=name1, name2=name2, font_size=fsize, min_gap=float(gap))
    st.components.v1.html(svg_str, height=620, scrolling=True)

with tab_circ:
    st.markdown("##### Circular Contact Map")
    st.caption("Polar coordinate projection of the Top 200 interaction pairs. Data and cutoffs map 1:1 with the linear contact diagram. Every contacting residue is a node on its arc; labels radiate outward and are thinned by angular separation so they don't collide, with the highest-scoring residues keeping their text. Hover any node or chord for the full identity and score.")
    cc1, cc2 = st.columns(2)
    with cc1:
        sep = st.slider("Minimum label separation (degrees)", 1.0, 15.0, 4.5, 0.5,
                        help="Raise this to thin out labels on crowded arcs. Nodes stay for every residue.")
    with cc2:
        cfsize = st.slider("Label font size ", 10, 26, 14, 1)
    fig_circ = _build_circular_plot(results["top_200"], results["cutoff_score"],
                                    name1=name1, name2=name2,
                                    font_size=cfsize, min_sep_deg=float(sep))
    FIGS["circular-contact-map"] = fig_circ
    st.plotly_chart(fig_circ, use_container_width=True)

with tab_downloads:
    st.markdown("##### Export Result Files")
    st.markdown(
        '<p style="color:#94a3b8; font-size:0.97rem; margin:0.35rem 0 1.1rem 0;">'
        'Every result shown in this session, as tab-separated tables plus the vector diagram. '
        'The archive contains all of them.</p>', unsafe_allow_html=True)

    files = _write_legacy_files(results, name1, name2, elapsed, figures=FIGS)
    stem = f"{name1}-{name2}"

    st.download_button(f"Download everything ({stem}-all-results.zip)",
                       files[f"{stem}-all-results.zip"],
                       file_name=f"{stem}-all-results.zip", mime="application/zip",
                       use_container_width=True)

    _gap("1rem")

    dc1, dc2, dc3 = st.columns(3)

    with dc1:
        st.download_button("All scored pairs (.txt)", files[f"{stem}-final-prediction.txt"],
                           file_name=f"{stem}-final-prediction.txt", use_container_width=True)
        st.download_button("Top 200 residue pairs (.tsv)", files[f"{stem}-top200.tsv"],
                           file_name=f"{stem}-top200.tsv", use_container_width=True)
        st.download_button("Run summary (.tsv)", files[f"{stem}-summary.tsv"],
                           file_name=f"{stem}-summary.tsv", use_container_width=True)

    with dc2:
        st.download_button("Residue propensities (.tsv)", files[f"{stem}-residue-propensities.tsv"],
                           file_name=f"{stem}-residue-propensities.tsv", use_container_width=True)
        st.download_button("Score matrix (.tsv)", files[f"{stem}-score-matrix.tsv"],
                           file_name=f"{stem}-score-matrix.tsv", use_container_width=True)
        st.download_button("Linear contact map (.svg)", files[f"{stem}-contact-map.svg"],
                           file_name=f"{stem}-contact-map.svg", mime="image/svg+xml",
                           use_container_width=True)

    with dc3:
        st.download_button("Target profile (.chain1)", files[f"{stem}-sspred.chain1"],
                           file_name=f"{stem}-sspred.chain1", use_container_width=True)
        st.download_button("Partner profile (.chain2)", files[f"{stem}-sspred.chain2"],
                           file_name=f"{stem}-sspred.chain2", use_container_width=True)

    _gap("1.2rem")
    st.markdown("##### Figures")
    st.markdown(
        '<p style="color:#94a3b8; font-size:0.97rem; margin:0.35rem 0 1.1rem 0;">'
        'Interactive HTML copies of each plot, openable in any browser with hover and zoom intact. '
        'For a static PNG instead, use the camera icon on any figure above.</p>',
        unsafe_allow_html=True)

    _FIG_LABELS = [
        ("2d-heatmap", "2D heatmap (.html)"),
        ("3d-landscape", "3D landscape (.html)"),
        ("circular-contact-map", "Circular contact map (.html)"),
        ("score-distribution", "Score distribution (.html)"),
        ("propensity-target", "Target propensity plot (.html)"),
        ("propensity-partner", "Partner propensity plot (.html)"),
    ]
    fcols = st.columns(3)
    for i, (slug, label) in enumerate(_FIG_LABELS):
        keyname = f"{stem}-{slug}.html"
        if keyname in files:
            with fcols[i % 3]:
                st.download_button(label, files[keyname], file_name=keyname,
                                   mime="text/html", use_container_width=True)

# ---------------------------------------------------------------------------
# Reference (always rendered at the foot of the page)
# ---------------------------------------------------------------------------
_gap("1.6rem")
_render_reference()
