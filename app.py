"""
PPIP Explorer — Streamlit Enterprise Suite for Protein-Protein Interaction Prediction
Strictly validated against Ahmad & Mizuguchi (2011).
"""

import io
import time
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="PPIP Explorer | Structural Biology Suite",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Advanced Sci-Fi & Biotech Glassmorphic Styling
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

/* Custom HUD Typography */
.eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  color: #00f2fe;
  background: rgba(0,242,254,0.08);
  border: 1px solid rgba(0,242,254,0.25);
  display: inline-block;
  padding: 0.3rem 0.7rem;
  border-radius: 8px;
  margin-bottom: 1.1rem;
}
.eyebrow::before { content: "$ "; color: #6B7688; }

.hero-title { font-size: 3.1rem; margin: 0 0 0.6rem 0; line-height: 1.05; font-weight: 800; }
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

def _build_svg(top_200_pairs: list, cutoff: float, label: str) -> str:
    if not top_200_pairs:
        return ""
        
    pairs = []
    for name, score in top_200_pairs:
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

    max_p1 = max([p[0] for p in pairs])
    max_p2 = max([p[1] for p in pairs])
    
    numres1 = max_p1 + 5
    numres2 = max_p2 + 5

    startx1, startx2 = 10, 10
    starty1 = 100
    svglength, svgheight = 1200, 200
    starty2 = starty1 + svgheight
    endx1 = startx1 + svglength
    endx2 = endx1
    endy1, endy2 = starty1, starty2

    out = io.StringIO()
    out.write(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {endx1 + 40} {starty2 + 40}" width="100%" height="100%">\n')
    out.write(f'<line x1="{startx1}" y1="{starty1}" x2="{endx1}" y2="{endy1}" style="stroke:#006600;"></line>\n')
    out.write(f'<line x1="{startx2}" y1="{starty2}" x2="{endx2}" y2="{endy2}" style="stroke:#006600;"></line>\n')
    out.write(f'<text x="10" y="20" font-size="12" fill="#B9C4D6"> {label} </text>\n')

    for p1, p2, res1, res2, score in pairs:
        if score > cutoff:
            x1 = startx1 + (p1 * svglength / numres1)
            x2 = startx2 + (p2 * svglength / numres2)
            out.write(f'<line x1="{x1}" y1="{starty1}" x2="{x2}" y2="{starty2}" stroke="blue" stroke-width="1"></line>\n')
            out.write(f'<text x="{x1}" y="{starty1}" transform="translate(0 -5) rotate(270 {x1} {starty1})" font-size="12" fill="#EAF0F7">{res1}</text>\n')
            out.write(f'<text x="{x2}" y="{starty2}" transform="translate(0 5) rotate(90 {x2} {starty2})" font-size="12" fill="#EAF0F7">{res2}</text>\n')

    out.write("</svg>\n")
    return out.getvalue()

def _build_circular_plot(top_200_pairs: list, cutoff: float) -> go.Figure:
    pairs = []
    for name, score in top_200_pairs:
        if score > cutoff:
            try:
                r1, r2 = name.split(":")
                p1 = int(re.search(r"\d+", r1).group())
                p2 = int(re.search(r"\d+", r2).group())
                pairs.append((p1, p2, name, score))
            except Exception:
                continue

    fig = go.Figure()
    if not pairs:
        return fig

    max_p1 = max([p[0] for p in pairs])
    max_p2 = max([p[1] for p in pairs])
    numres1 = max_p1 + 5
    numres2 = max_p2 + 5

    theta1 = np.linspace(np.pi - 0.2, 0.2, 100)
    theta2 = np.linspace(np.pi + 0.2, 2*np.pi - 0.2, 100)
    
    fig.add_trace(go.Scatter(x=np.cos(theta1), y=np.sin(theta1), mode="lines", line=dict(color="#006600", width=4), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=np.cos(theta2), y=np.sin(theta2), mode="lines", line=dict(color="#006600", width=4), hoverinfo="skip"))

    for p1, p2, name, score in pairs:
        a1 = (np.pi - 0.2) - (p1 / numres1) * (np.pi - 0.4)
        a2 = (np.pi + 0.2) + (p2 / numres2) * (np.pi - 0.4)
        x1, y1 = np.cos(a1), np.sin(a1)
        x2, y2 = np.cos(a2), np.sin(a2)
        
        fig.add_trace(go.Scatter(
            x=[x1, x2], y=[y1, y2], mode="lines",
            line=dict(color="rgba(0, 102, 255, 0.4)", width=1),
            hoverinfo="text", text=f"{name} (Score: {score:.4f})"
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=10, b=10), height=500
    )
    return fig

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

    svg = _build_svg(results["top_200"], results["cutoff_score"], f"{name1}-{name2}")
    files[f"{name1}-{name2}.svg"] = svg.encode()

    return files

# ---------------------------------------------------------------------------
# Header & Front-Facing Ingestion UI
# ---------------------------------------------------------------------------
st.markdown('<div class="eyebrow"> &lt;pssm_1&gt; &lt;pssm_2&gt;</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">PPI<span>P Explorer</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Neural engine for Protein-Protein Interaction Prediction. Score every residue pair with a 24-network SNNS ensemble.</p>', unsafe_allow_html=True)

# Front-facing file uploader using native Streamlit container with border
with st.container(border=True):
    st.markdown("#### Sequence & Profile Ingestion Pipeline")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        file1 = st.file_uploader("Protein 1 (Target PSSM)", type=None, key="f1")
    with col2:
        file2 = st.file_uploader("Protein 2 (Partner PSSM)", type=None, key="f2")
    
    st.markdown("<br>", unsafe_allow_html=True)
    run_clicked = st.button("Execute Interaction Prediction Pipeline", type="primary", disabled=not (file1 and file2))

# Conditional Methodology Container (Only displays if no results are loaded yet)
if "results" not in st.session_state:
    with st.container(border=True):
        st.markdown("#### Scientific Background & Methodology")
        st.markdown("Computational prediction of protein-protein interaction (PPI) interfaces is a fundamental challenge in structural biology. Traditional machine-learning methods are often 'partner-unaware'—they attempt to identify binding sites on a single protein in isolation. This suite is built upon the foundational partner-aware algorithm established by Professor Shandar Ahmad and Kenji Mizuguchi.")
        st.markdown("By evaluating the sequence-derived Position-Specific Scoring Matrices (PSSMs) of both the target and the partner protein simultaneously, the model captures complementary residue pairing. This drastically reduces false-positive predictions, as it explicitly requires the binding partner to possess a compatible interface region.")
        st.markdown("---")
        st.markdown("##### Pipeline Architecture (Steps)")
        st.markdown("1. **Pattern Extraction:** Slides multiple window sizes (-1 to 3) across sequences, extracting local neighborhood PSSM profiles.")
        st.markdown("2. **Ensemble Neural Network Scoring:** 24 distinct pre-trained Artificial Neural Networks perform symmetric forward passes to score candidate interactions.")
        st.markdown("3. **Stage-2 Composition:** The parallel predictions are concatenated column-wise, fusing the 24 independent network outputs.")
        st.markdown("4. **Matrix Smoothing:** The resulting bipartite scoring matrix is smoothed using a moving-average filter.")
        st.markdown("5. **Final Ranking:** The highest-probability residue pairs are extracted.")
        st.markdown('<div style="background: rgba(0, 242, 254, 0.05); border: 1px solid rgba(0, 242, 254, 0.2); padding: 15px; border-radius: 8px; margin-top: 15px;">📖 <a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0029104" target="_blank" style="color: #00f2fe; text-decoration: none;">Ahmad S, Mizuguchi K (2011). Partner-Aware Prediction of Interacting Residues in Protein-Protein Complexes from Sequence Data. PLoS ONE 6(12): e29104.</a></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main Execution Logic
# ---------------------------------------------------------------------------
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
            results = run_prediction(lines1, lines2)
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

if "results" not in st.session_state:
    st.stop()

results = st.session_state["results"]
name1 = st.session_state["name1"]
name2 = st.session_state["name2"]
elapsed = st.session_state["elapsed"]

# ---------------------------------------------------------------------------
# Results Metrics
# ---------------------------------------------------------------------------
st.markdown("### Execution Results")
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
        fig1.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown(f"**{name2}**")
        c2_df = pd.DataFrame([(r, results["chain2"][r]) for r in results["unique_r2"]], columns=["Residue", "Score"])
        fig2 = go.Figure(go.Scatter(x=c2_df["Residue"], y=c2_df["Score"], mode="lines", line=dict(color="#c084fc", width=2)))
        fig2.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"))
        st.plotly_chart(fig2, use_container_width=True)

with tab_heat:
    st.markdown("##### Smoothed Interaction Score Matrix")
    mat = results["smoothed_matrix"]
    fig = go.Figure(data=go.Heatmap(
        z=mat, x=results["unique_r2"], y=results["unique_r1"],
        colorscale=[[0, "#030712"], [0.25, "#1e1b4b"], [0.5, "#0284c7"], [0.75, "#00f2fe"], [1.0, "#f43f5e"]],
        colorbar=dict(title="Score", tickfont=dict(color="#B9C4D6")),
    ))
    fig.update_layout(height=700, xaxis_title=name2, yaxis_title=name1, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#B9C4D6"))
    st.plotly_chart(fig, use_container_width=True)

with tab_3d:
    st.markdown("##### 3D Interaction Energy Landscape")
    st.caption("Surface projection of the exact smoothed matrix array.")
    fig3d = go.Figure(data=[go.Surface(z=results["smoothed_matrix"], x=results["unique_r2"], y=results["unique_r1"], colorscale="viridis")])
    fig3d.update_layout(height=700, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#B9C4D6"), scene=dict(xaxis_title=name2, yaxis_title=name1, zaxis_title="Score"))
    st.plotly_chart(fig3d, use_container_width=True)

with tab_dist:
    st.markdown("##### Score Distribution Histogram")
    st.caption("Descriptive distribution of all calculated pair scores, separating interaction signal from background.")
    scores = [s for _, s in results["all_pairs"]]
    fig_hist = go.Figure(data=[go.Histogram(x=scores, nbinsx=100, marker_color="#00f2fe", opacity=0.8)])
    fig_hist.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#B9C4D6"), xaxis_title="Interaction Score", yaxis_title="Frequency Count")
    st.plotly_chart(fig_hist, use_container_width=True)

with tab_diagram:
    st.markdown("##### Bipartite Interaction Wiring Diagram")
    st.caption("Strict reconstruction of get-svg.sh. Lines drawn identically for the top 200 pairs.")
    svg_str = _build_svg(results["top_200"], results["cutoff_score"], f"{name1}-{name2}")
    st.components.v1.html(svg_str, height=450, scrolling=True)

with tab_circ:
    st.markdown("##### Circular Contact Map")
    st.caption("Polar coordinate projection of the Top 200 interaction pairs. Data and cutoffs map 1:1 with the linear contact diagram.")
    fig_circ = _build_circular_plot(results["top_200"], results["cutoff_score"])
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
