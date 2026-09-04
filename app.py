"""
PPIP Explorer — Streamlit Enterprise Suite for Protein-Protein Interaction Prediction
Strictly validated against Ahmad & Mizuguchi (2011).
"""

import io
import time
import re

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

h1, h2, h3, h4 {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  letter-spacing: -0.01em;
  color: #f8fafc !important;
}

p, li, span, label, .stMarkdown { color: #94a3b8; }

::selection { background: rgba(0,242,254,0.35); }

.hero-title { font-size: 4.8rem; margin: 0 0 0.6rem 0; line-height: 1.05; font-weight: 800; }
.hero-title span { background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #7928ca 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero-sub { font-size: 1.08rem; color: #94a3b8; max-width: 800px; line-height: 1.55; margin-bottom: 2rem; }

/* Native Streamlit Container Border Styling (Replaces raw HTML ghost cards) */
div[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 18px !important;
  background: rgba(15, 23, 42, 0.65) !important;
  backdrop-filter: blur(20px) saturate(1.6) !important;
  box-shadow: 0 10px 35px -5px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06) !important;
  padding: 1rem !important;
  margin-bottom: 1.5rem !important;
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


def _render_reference():
    with st.container(border=True):
        st.markdown("#### Reference")
        st.markdown('<div style="background: rgba(0, 242, 254, 0.05); border: 1px solid rgba(0, 242, 254, 0.2); padding: 15px; border-radius: 8px;">📖 <a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0029104" target="_blank" style="color: #00f2fe; text-decoration: none;">Ahmad S, Mizuguchi K (2011). Partner-Aware Prediction of Interacting Residues in Protein-Protein Complexes from Sequence Data. PLoS ONE 6(12): e29104.</a></div>', unsafe_allow_html=True)

def _write_legacy_files(results: dict, name1: str, name2: str) -> dict[str, bytes]:
    files = {}

    buf = io.StringIO()
    buf.write("Pair(Seq1:Seq2)\tPrediction-score\n")
    for name, score in results["all_pairs"]:
        buf.write(f"{name}\t{score:.6f}\n")
    files[f"{name1}-{name2}-final-prediction.txt"] = buf.getvalue().encode()

    buf = io.StringIO()
    buf.write("Pair(Seq1:Seq2)\tPrediction-score\n")
    for name, score in results["top_200"]:
        buf.write(f"{name}\t{score:.6f}\n")
    files[f"{name1}-{name2}-top200.txt"] = buf.getvalue().encode()

    buf = io.StringIO()
    for res, score in results["chain1"].items():
        buf.write(f"{res} {score:.6f}\n")
    files[f"{name1}-{name2}-sspred.chain1"] = buf.getvalue().encode()

    buf = io.StringIO()
    for res, score in results["chain2"].items():
        buf.write(f"{res} {score:.6f}\n")
    files[f"{name1}-{name2}-sspred.chain2"] = buf.getvalue().encode()

    svg = _build_svg(results["top_200"], results["cutoff_score"], f"{name1}-{name2}", name1=name1, name2=name2)
    files[f"{name1}-{name2}.svg"] = svg.encode()

    return files

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<h1 class="hero-title">PPIP<span>P Explorer</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Neural engine for Protein-Protein Interaction Prediction. Score every residue pair with a 24-network SNNS ensemble.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Landing view: methodology first, then ingestion. Replaced entirely by the
# results view once a prediction has been executed.
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    with st.container(border=True):
        st.markdown("#### Scientific Background")
        st.markdown("Computational prediction of protein-protein interaction (PPI) interfaces is a fundamental challenge in structural biology. Traditional machine-learning methods are often 'partner-unaware'—they attempt to identify binding sites on a single protein in isolation. This suite is built upon the foundational partner-aware algorithm established by Professor Shandar Ahmad and Kenji Mizuguchi.")
        st.markdown("By evaluating the sequence-derived Position-Specific Scoring Matrices (PSSMs) of both the target and the partner protein simultaneously, the model captures complementary residue pairing. This drastically reduces false-positive predictions, as it explicitly requires the binding partner to possess a compatible interface region.")
        st.markdown("---")
        st.markdown("#### Pipeline Architecture")
        st.markdown("1. **Stage-1 Composition:** Extract the pattern (sparse sequence encoding and PSSM-based evolutionary profile) features from the protein pair.")
        st.markdown("2. **Neural Network:** Consider multiple window sizes (0, 1, 3, 5, 7) across sequences to capture the local neighborhood impact of protein pairs and train 24 distinct Artificial Neural Networks to score candidate interactions.")
        st.markdown("3. **Stage-2 Composition:** The parallel predictions are concatenated column-wise, fusing the 24 independent neural network outputs.")
        st.markdown("4. **Final Ranking:** Pair-wise scores are ranked directly (unsmoothed) to select the top 200 candidate interactions, following Ahmad & Mizuguchi (2011).")
        st.markdown("5. **Visualization Smoothing (app-only):** For the heatmap and 3D views only, a moving-average filter is applied for visual clarity. This step is not part of the original published method and has no effect on the ranked target-partner protein pairs mentioned above.")

    with st.container(border=True):
        st.markdown("#### Upload PSSM Profiles")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            file1 = st.file_uploader("Protein 1 (Target PSSM)", type=None, key="f1")
        with col2:
            file2 = st.file_uploader("Protein 2 (Partner PSSM)", type=None, key="f2")

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
hcol, rcol = st.columns([4, 1])
with hcol:
    st.markdown("### Executed Results")
    st.caption(f"{name1} vs {name2}")
with rcol:
    if st.button("New prediction"):
        for _k in ("results", "name1", "name2", "elapsed", "f1", "f2"):
            st.session_state.pop(_k, None)
        st.rerun()

top_pair, top_score = results["top_200"][0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Sequence Geometry", f"{len(results['unique_r1'])} × {len(results['unique_r2'])}")
m2.metric("Top 200 Cutoff Score", f"{results['cutoff_score']:.4f}")
m3.metric("Peak Interaction", f"{top_score:.3f}", top_pair)
m4.metric("Runtime", f"{elapsed:.2f}s")

# ---------------------------------------------------------------------------
# Strict Authentic Tabs
# ---------------------------------------------------------------------------
tab_top, tab_chain, tab_heat, tab_3d, tab_dist, tab_diagram, tab_circ, tab_downloads = st.tabs([
    "Ranked Hotspots", "Chain Propensities", "2D Heatmap", "3D Landscape", 
    "Score Distribution", "Linear Contact Map", "Circular Contact Map", "File Exports"
])

with tab_top:
    st.markdown("##### Top 200 Candidate Contact Pairs")
    df = pd.DataFrame(results["top_200"], columns=["Pair", "Score"])
    df.index = df.index + 1
    st.dataframe(df.style.background_gradient(subset=["Score"], cmap="GnBu_r"), use_container_width=True, height=500)

with tab_chain:
    st.markdown("##### Per-Residue Interface Propensity Profiles")
    st.caption("Maximum interaction score achieved by each residue against any partner.")
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
    st.download_button("Download this diagram (SVG)", svg_str.encode(),
                       file_name=f"{name1}-{name2}-contact-map.svg", mime="image/svg+xml")

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
    st.plotly_chart(fig_circ, use_container_width=True)

with tab_downloads:
    st.markdown("##### Original Pipeline Outputs")
    st.caption("Direct exports formatted identically to the legacy pipeline bins.")
    
    files = _write_legacy_files(results, name1, name2)
    dc1, dc2 = st.columns(2)
    
    with dc1:
        st.download_button("Download final-prediction.txt", files[f"{name1}-{name2}-final-prediction.txt"], file_name=f"{name1}-{name2}-final-prediction.txt", use_container_width=True)
        st.download_button("Download top200.txt", files[f"{name1}-{name2}-top200.txt"], file_name=f"{name1}-{name2}-top200.txt", use_container_width=True)
        
    with dc2:
        st.download_button("Download .chain1 profile", files[f"{name1}-{name2}-sspred.chain1"], file_name=f"{name1}-{name2}-sspred.chain1", use_container_width=True)
        st.download_button("Download .chain2 profile", files[f"{name1}-{name2}-sspred.chain2"], file_name=f"{name1}-{name2}-sspred.chain2", use_container_width=True)
        st.download_button("Download SVG Diagram", files[f"{name1}-{name2}.svg"], file_name=f"{name1}-{name2}.svg", mime="image/svg+xml", use_container_width=True)

# ---------------------------------------------------------------------------
# Reference (always rendered at the foot of the page)
# ---------------------------------------------------------------------------
_render_reference()
