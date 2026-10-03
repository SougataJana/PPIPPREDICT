
"""
PPIP Explorer — Streamlit Enterprise Suite for Protein-Protein Interaction Prediction
Strictly validated against Ahmad & Mizuguchi (2011).
"""

import base64
import gc
import io
import os
import time
import re
import zipfile

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from inference import load_models, run_prediction, pairs_from_matrix, read_pssm_from_text
import batch as bx

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
