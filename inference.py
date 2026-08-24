"""
inference.py — PyTorch inference over the ppip_ensemble_weights.pt ensemble.

Verified: ppip_ensemble_weights.pt's "-1_0" sub-model matches the real
legacy SNNS net--1-0-1 exactly (weights, biases, and full forward pass
output all agree to float32 precision) — this is a faithful conversion,
not a guess.

TWO FIXES vs. the version this was ported from:

1. LABEL OFF-BY-ONE: read_pssm_from_text() used a 1-based position
   counter (first residue -> "X1"). Verified against real reference
   output (top25.txt / final-prediction.txt cross-checked against the
   real 1avxA.pssm): every residue label matches the raw 0-indexed
   position, not a 1-based one. Fixed here to start at 0.

2. VECTORIZED: the original ran one model.forward() call per residue
   pair per window combo per direction — for two 200-residue proteins
   that's ~1.7M individual forward calls, each paying full Python/tensor
   overhead. This version builds one batched feature matrix per (window
   combo, direction) — 48 batched forward passes total instead of ~1.7M
   — and is numerically identical (same math, just batched).
"""

import torch
import numpy as np
import torch.nn as nn
import re
import sys
import os
from datetime import datetime


class Exact2011Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.hidden = nn.Linear(input_dim, 4)
        self.output = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.output(self.sigmoid(self.hidden(x))))


def get_features(matrix, pos, win):
    feat = list(matrix[pos])
    for m in range(1, win + 1):
        feat.extend(matrix[pos + m])
    for m in range(1, win + 1):
        feat.extend(matrix[pos - m])
    return feat


def read_pssm_from_text(file_lines):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    seq_bin, pssm, residues = [], [], []
    pos = 0  # 0-indexed — matches verified real-output labeling convention
    for l in file_lines:
        parts = l.split()
        if len(parts) > 40 and parts[0].isdigit():
            res = parts[1].upper()
            if res in amino_acids:
                residues.append(f"{res}{pos}")
                pos += 1
                seq_bin.append([1.0 if res == aa else 0.0 for aa in amino_acids])
                pssm.append([float(x) for x in parts[2:22]])
    return np.array(seq_bin), np.array(pssm), residues


def apply_r_smoothing(matrix, halfwin=1, step1=4, step2=1):
    """Mirrors the moving-average smoothing filter from plot.R"""
    size1, size2 = matrix.shape
    smoothed = matrix.copy()
    for i in range(size1):
        for j in range(size2):
            xx, cnt = 0.0, 0
            for k in range(-halfwin, halfwin + 1):
                for l in range(-halfwin, halfwin + 1):
                    ni, nj = i + step1 * k, j + step2 * l
                    if 0 <= ni < size1 and 0 <= nj < size2:
                        xx += matrix[ni, nj]
                        cnt += 1
            if cnt > 0:
                smoothed[i, j] = xx / cnt
    return smoothed


_COMBOS = [(p, b) for p in range(-1, 4) for b in range(-1, 4) if p + b > -2]
assert len(_COMBOS) == 24


def _build_window_block(matrix, rows, win):
    """Vectorized version of get_features() for every row in `rows` at once.
    matrix: (n, dim) array. rows: 1D int array of positions.
    Returns (len(rows), (2*win+1)*dim) in the same self/+/- column order
    get_features() produces (self, +1..+win, -1..-win)."""
    offsets = [0] + list(range(1, win + 1)) + list(range(-1, -win - 1, -1))
    return np.concatenate([matrix[rows + off] for off in offsets], axis=1)


def _score_combo(seq1, pssm1, seq2, pssm2, rows1, rows2, pssmwin, binwin, model):
    """Fully vectorized: builds every (i,j) forward+reverse feature row for
    this window combo in one shot (no Python loop over positions), then one
    batched forward pass each direction. Row order = i outer, j inner —
    matches positions = [(i,j) for i in rows1 for j in rows2].

    Feature order must match get_features()'s original interleaving:
    [binary(chain1)][binary(chain2)][pssm(chain1)][pssm(chain2)] for the
    forward vector — grouped by FEATURE TYPE first, not by chain — with
    chain1/chain2 swapped for the reverse vector."""
    n1u, n2u = len(rows1), len(rows2)

    fwd_parts, rev_parts = [], []
    if binwin > -1:
        bin1 = _build_window_block(seq1, rows1, binwin)
        bin2 = _build_window_block(seq2, rows2, binwin)
        fwd_parts += [np.repeat(bin1, n2u, axis=0), np.tile(bin2, (n1u, 1))]
        rev_parts += [np.tile(bin2, (n1u, 1)), np.repeat(bin1, n2u, axis=0)]
    if pssmwin > -1:
        pssm1_blk = _build_window_block(pssm1, rows1, pssmwin)
        pssm2_blk = _build_window_block(pssm2, rows2, pssmwin)
        fwd_parts += [np.repeat(pssm1_blk, n2u, axis=0), np.tile(pssm2_blk, (n1u, 1))]
        rev_parts += [np.tile(pssm2_blk, (n1u, 1)), np.repeat(pssm1_blk, n2u, axis=0)]

    Xf = np.concatenate(fwd_parts, axis=1)
    Xr = np.concatenate(rev_parts, axis=1)

    with torch.no_grad():
        yf = model(torch.tensor(Xf, dtype=torch.float32)).squeeze(-1).numpy()
        yr = model(torch.tensor(Xr, dtype=torch.float32)).squeeze(-1).numpy()
    return yf, yr


def load_models(weights_path="ppip_ensemble_weights.pt"):
    """Loads the 24-network ensemble from disk. Call this once and reuse
    the result — e.g. wrap it in @st.cache_resource in app.py — since
    rebuilding 24 PyTorch modules from a state dict on every request is
    wasted work that has nothing to do with the actual prediction time."""
    weights = torch.load(weights_path, map_location="cpu")
    models = {}
    for key, w in weights.items():
        model = Exact2011Model(w["input_dim"])
        model.load_state_dict({k: v for k, v in w.items() if k != "input_dim"})
        model.eval()
        models[key] = model
    return models


def run_prediction(lines1, lines2, weights_path="ppip_ensemble_weights.pt", models=None, progress_cb=None):
    start_time = datetime.now().strftime("%a %b %d %H:%M:%S %Y")

    seq1, pssm1, res1 = read_pssm_from_text(lines1)
    seq2, pssm2, res2 = read_pssm_from_text(lines2)

    if models is None:
        models = load_models(weights_path)

    n1, n2 = len(res1), len(res2)
    rows1 = np.arange(5, n1 - 5)
    rows2 = np.arange(5, n2 - 5)
    n1u, n2u = len(rows1), len(rows2)
    n_pairs = n1u * n2u
    pair_names = [f"{res1[i]}:{res2[j]}" for i in rows1 for j in rows2]

    accum_fwd = np.zeros(n_pairs)
    accum_rev = np.zeros(n_pairs)
    counts = np.zeros(n_pairs)

    # Window-fit checks only ever discard positions on very short proteins
    # (margin of 5 already exceeds the max window of 3) — for real inputs
    # every position survives every combo, but we still respect the check
    # generally by only vectorizing combos where it can't matter and
    # falling back exactly otherwise.
    def _fits(win, rows, n):
        if win < 0:
            return True
        return bool(rows.min() - win >= 0 and rows.max() + win < n)

    for combo_idx, (pssmwin, binwin) in enumerate(_COMBOS):
        if progress_cb:
            progress_cb((combo_idx + 1) / len(_COMBOS), f"window pssm={pssmwin} bin={binwin}")

        if not (_fits(pssmwin, rows1, n1) and _fits(pssmwin, rows2, n2)
                and _fits(binwin, rows1, n1) and _fits(binwin, rows2, n2)):
            # short-protein edge case: fall back to the exact per-position
            # loop so boundary skips are handled correctly, at whatever
            # speed that takes (rare path).
            model = models[f"{pssmwin}_{binwin}"]
            for k, (i, j) in enumerate((i, j) for i in rows1 for j in rows2):
                if pssmwin > -1 and (i - pssmwin < 0 or i + pssmwin >= n1 or j - pssmwin < 0 or j + pssmwin >= n2):
                    continue
                if binwin > -1 and (i - binwin < 0 or i + binwin >= n1 or j - binwin < 0 or j + binwin >= n2):
                    continue
                f_fwd, f_rev = [], []
                if binwin > -1:
                    f_fwd.extend(get_features(seq1, i, binwin)); f_fwd.extend(get_features(seq2, j, binwin))
                    f_rev.extend(get_features(seq2, j, binwin)); f_rev.extend(get_features(seq1, i, binwin))
                if pssmwin > -1:
                    f_fwd.extend(get_features(pssm1, i, pssmwin)); f_fwd.extend(get_features(pssm2, j, pssmwin))
                    f_rev.extend(get_features(pssm2, j, pssmwin)); f_rev.extend(get_features(pssm1, i, pssmwin))
                with torch.no_grad():
                    accum_fwd[k] += model(torch.tensor([f_fwd], dtype=torch.float32)).item()
                    accum_rev[k] += model(torch.tensor([f_rev], dtype=torch.float32)).item()
                counts[k] += 1
            continue

        model = models[f"{pssmwin}_{binwin}"]
        yf, yr = _score_combo(seq1, pssm1, seq2, pssm2, rows1, rows2, pssmwin, binwin, model)
        accum_fwd += yf
        accum_rev += yr
        counts += 1

    keep = counts > 0
    if not keep.all():
        dropped = int((~keep).sum())
        print(f"Warning: {dropped} position pairs had no surviving window combo (very short protein?) — dropped.")

    avg_fwd = accum_fwd[keep] / counts[keep]
    avg_rev = accum_rev[keep] / counts[keep]
    final_scores = (avg_fwd + avg_rev) / 2.0
    kept_names = [pair_names[k] for k in np.nonzero(keep)[0]]

    raw_pairs = list(zip(kept_names, final_scores.tolist()))

    sorted_pairs = sorted(raw_pairs, key=lambda x: x[1], reverse=True)
    top_200 = sorted_pairs[:200]

    unique_r1 = sorted(set(p[0].split(":")[0] for p in raw_pairs), key=lambda x: int(re.search(r'\d+', x).group()))
    unique_r2 = sorted(set(p[0].split(":")[1] for p in raw_pairs), key=lambda x: int(re.search(r'\d+', x).group()))

    matrix = np.zeros((len(unique_r1), len(unique_r2)))
    score_lookup = dict(raw_pairs)
    for i, r1 in enumerate(unique_r1):
        for j, r2 in enumerate(unique_r2):
            matrix[i, j] = score_lookup.get(f"{r1}:{r2}", 0.0)

    smoothed_matrix = apply_r_smoothing(matrix)
    avscore = np.mean(smoothed_matrix)
    sdscore = np.std(smoothed_matrix)
    threshold = avscore + (3 * sdscore)

    chain1_scores, chain2_scores = {}, {}
    for pair_name, score in raw_pairs:
        c1, c2 = pair_name.split(":")
        chain1_scores.setdefault(c1, []).append(score)
        chain2_scores.setdefault(c2, []).append(score)

    end_time = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    time_log = f"Start time: {start_time}\nEnd time: {end_time}"

    return {
        "all_pairs": raw_pairs,
        "top_200": top_200,
        "chain1": {k: max(v) for k, v in chain1_scores.items()},
        "chain2": {k: max(v) for k, v in chain2_scores.items()},
        "matrix": matrix,
        "smoothed_matrix": smoothed_matrix,
        "threshold": threshold,
        "unique_r1": unique_r1,
        "unique_r2": unique_r2,
        "time_log": time_log,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inference.py <pssm_file_1> <pssm_file_2>")
        sys.exit(1)

    file1_path, file2_path = sys.argv[1], sys.argv[2]
    name1, name2 = os.path.basename(file1_path), os.path.basename(file2_path)

    with open(file1_path) as f:
        lines1 = f.readlines()
    with open(file2_path) as f:
        lines2 = f.readlines()

    print(f"Processing sequences: {name1} and {name2}...")
    results = run_prediction(lines1, lines2)

    with open(f"{name1}-{name2}-final-prediction.txt", "w") as f:
        f.write("Pair(Seq1:Seq2)\tPrediction-score\n")
        for pair_name, score in results["all_pairs"]:
            f.write(f"{pair_name}: {score:.6f}\n")

    with open(f"{name1}-{name2}-top200.txt", "w") as f:
        f.write("Pair(Seq1:Seq2)\tPrediction-score\n")
        for pair_name, score in results["top_200"]:
            f.write(f"{pair_name}: {score:.6f}\n")

    with open(f"{name1}-{name2}-sspred.chain1", "w") as f:
        for res, score in results["chain1"].items():
            f.write(f"{res} {score:.6f}\n")

    with open(f"{name1}-{name2}-sspred.chain2", "w") as f:
        for res, score in results["chain2"].items():
            f.write(f"{res} {score:.6f}\n")

    print("Done.")
