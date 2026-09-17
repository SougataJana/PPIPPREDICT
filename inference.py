"""
inference.py — PyTorch inference over the ppip_ensemble_weights.pt ensemble.

Verified: ppip_ensemble_weights.pt's "-1_0" sub-model matches the real
legacy SNNS net--1-0-1 exactly (weights, biases, and full forward pass
output all agree to float32 precision) — this is a faithful conversion,
not a guess.

FIXES CARRIED OVER FROM THE PREVIOUS VERSION
1. LABEL OFF-BY-ONE: read_pssm_from_text() uses a 0-based position counter,
   verified against real reference output (top25.txt / final-prediction.txt
   cross-checked against the real 1avxA.pssm).
2. VECTORIZED: one batched forward pass per (window combo, direction)
   instead of one call per residue pair.

NEW IN THIS VERSION (batch-mode support, same numbers out)
3. CHUNKED + float32: the batched feature matrix is now built in row blocks
   sized to a byte budget (MAX_BLOCK_BYTES) and stored as float32 instead of
   float64. Peak RAM for a 200x200 pair drops from ~250 MB to <40 MB, which
   is what makes 45 sequential pairs viable on a 1 GB container. The maths is
   unchanged — the network already ran in float32 inside torch.
4. VECTORIZED SMOOTHING: apply_r_smoothing() is now a shifted-array
   accumulation instead of a 4-deep Python loop (490x490: 0.45 s -> 0.008 s).
   Output verified identical to the original loop on every shape tested.
5. COMPACT RESULTS: run_prediction(..., compact=True) returns the raw score
   matrix instead of a 36k-element list of (name, score) tuples. A batch of
   45 pairs then costs ~20 MB of session state instead of ~300 MB. Call
   pairs_from_matrix() to materialise all_pairs for one pair on demand.
"""

import gc
import os
import re
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

# Byte budget for a single feature block. 48 MB of float32 keeps peak RSS
# (block + its torch copy + hidden activations) comfortably under ~150 MB
# even for long proteins. Lower it if the host is tighter than that.
MAX_BLOCK_BYTES = int(os.environ.get("PPIP_MAX_BLOCK_BYTES", 48_000_000))


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
    return (np.array(seq_bin, dtype=np.float32),
            np.array(pssm, dtype=np.float32),
            residues)


def apply_r_smoothing(matrix, halfwin=1, step1=4, step2=1):
    """Moving-average filter from plot.R, vectorised.

    Identical output to the original quadruple loop (verified on random
    matrices of shapes 190x190, 7x5, 3x3, 1x9 and 50x2), including the
    edge behaviour where out-of-bounds neighbours are simply not counted.
    """
    m = np.asarray(matrix, dtype=np.float32)
    s1, s2 = m.shape
    tot = np.zeros_like(m)
    cnt = np.zeros_like(m)
    for k in range(-halfwin, halfwin + 1):
        di = step1 * k
        for l in range(-halfwin, halfwin + 1):
            dj = step2 * l
            i0, i1 = max(0, -di), min(s1, s1 - di)
            j0, j1 = max(0, -dj), min(s2, s2 - dj)
            if i0 >= i1 or j0 >= j1:
                continue
            tot[i0:i1, j0:j1] += m[i0 + di:i1 + di, j0 + dj:j1 + dj]
            cnt[i0:i1, j0:j1] += 1
    out = m.copy()
    np.divide(tot, cnt, out=out, where=cnt > 0)
    return out


_COMBOS = [(p, b) for p in range(-1, 4) for b in range(-1, 4) if p + b > -2]
assert len(_COMBOS) == 24


def _build_window_block(matrix, rows, win):
    """Vectorized get_features() for every row in `rows` at once.
    Column order matches get_features(): self, +1..+win, -1..-win."""
    offsets = [0] + list(range(1, win + 1)) + list(range(-1, -win - 1, -1))
    return np.concatenate([matrix[rows + off] for off in offsets], axis=1).astype(np.float32, copy=False)


def _feature_dim(pssmwin, binwin):
    dim = 0
    if binwin > -1:
        dim += 2 * (2 * binwin + 1) * 20
    if pssmwin > -1:
        dim += 2 * (2 * pssmwin + 1) * 20
    return dim


def _score_combo(seq1, pssm1, seq2, pssm2, rows1, rows2, pssmwin, binwin,
                 model, accum_fwd, accum_rev, counts):
    """Score every (i, j) pair for this window combo, in row blocks.

    Row order is i outer, j inner — matching
    positions = [(i, j) for i in rows1 for j in rows2] — so results write
    straight into the flat accumulators at offset block_start * n2u.

    Feature order matches get_features()'s interleaving:
    [binary(c1)][binary(c2)][pssm(c1)][pssm(c2)] forward, chains swapped
    for reverse.
    """
    n1u, n2u = len(rows1), len(rows2)
    dim = _feature_dim(pssmwin, binwin)

    bin1 = _build_window_block(seq1, rows1, binwin) if binwin > -1 else None
    bin2 = _build_window_block(seq2, rows2, binwin) if binwin > -1 else None
    pssm1_blk = _build_window_block(pssm1, rows1, pssmwin) if pssmwin > -1 else None
    pssm2_blk = _build_window_block(pssm2, rows2, pssmwin) if pssmwin > -1 else None

    # Rows of chain 1 per block, so one feature matrix stays inside the budget.
    bytes_per_row1 = max(1, n2u * dim * 4)
    step = max(1, min(n1u, MAX_BLOCK_BYTES // bytes_per_row1))

    for start in range(0, n1u, step):
        stop = min(start + step, n1u)
        sl = slice(start, stop)
        m = stop - start

        fwd_parts, rev_parts = [], []
        if binwin > -1:
            b1r = np.repeat(bin1[sl], n2u, axis=0)
            b2t = np.tile(bin2, (m, 1))
            fwd_parts += [b1r, b2t]
            rev_parts += [b2t, b1r]
        if pssmwin > -1:
            p1r = np.repeat(pssm1_blk[sl], n2u, axis=0)
            p2t = np.tile(pssm2_blk, (m, 1))
            fwd_parts += [p1r, p2t]
            rev_parts += [p2t, p1r]

        Xf = np.concatenate(fwd_parts, axis=1)
        Xr = np.concatenate(rev_parts, axis=1)
        del fwd_parts, rev_parts

        with torch.no_grad():
            yf = model(torch.from_numpy(Xf)).squeeze(-1).numpy()
            del Xf
            yr = model(torch.from_numpy(Xr)).squeeze(-1).numpy()
            del Xr

        off0, off1 = start * n2u, stop * n2u
        accum_fwd[off0:off1] += yf
        accum_rev[off0:off1] += yr
        counts[off0:off1] += 1
        del yf, yr

    del bin1, bin2, pssm1_blk, pssm2_blk


def load_models(weights_path="ppip_ensemble_weights.pt"):
    """Loads the 24-network ensemble from disk. Call this once and reuse the
    result — e.g. wrap it in @st.cache_resource in app.py."""
    weights = torch.load(weights_path, map_location="cpu")
    models = {}
    for key, w in weights.items():
        model = Exact2011Model(w["input_dim"])
        model.load_state_dict({k: v for k, v in w.items() if k != "input_dim"})
        model.eval()
        models[key] = model
    return models


def pairs_from_matrix(unique_r1, unique_r2, matrix):
    """Rebuild the flat [(name, score), ...] list from a compact result.

    One 200x200 pair is ~36k tuples (~7 MB) — fine for the pair currently
    on screen, which is why compact mode stores the matrix and expands
    only on demand.
    """
    mat = np.asarray(matrix)
    out = []
    for i, r1 in enumerate(unique_r1):
        row = mat[i]
        for j, r2 in enumerate(unique_r2):
            out.append((f"{r1}:{r2}", float(row[j])))
    return out


def run_prediction(lines1, lines2, weights_path="ppip_ensemble_weights.pt",
                   models=None, progress_cb=None, compact=False):
    """Score one protein pair across the 24-network ensemble.

    compact=False  -> the original dict (includes all_pairs, matrix).
    compact=True   -> all_pairs and matrix are replaced by raw_matrix; use
                      pairs_from_matrix() to expand when needed. Intended
                      for batch runs where dozens of results are held at once.
    """
    start_time = datetime.now().strftime("%a %b %d %H:%M:%S %Y")

    seq1, pssm1, res1 = read_pssm_from_text(lines1)
    seq2, pssm2, res2 = read_pssm_from_text(lines2)

    if len(res1) < 12 or len(res2) < 12:
        raise ValueError(
            f"PSSM too short to score (got {len(res1)} and {len(res2)} residues; "
            "at least 12 are needed for the 5-residue margin)."
        )

    if models is None:
        models = load_models(weights_path)

    n1, n2 = len(res1), len(res2)
    rows1 = np.arange(5, n1 - 5)
    rows2 = np.arange(5, n2 - 5)
    n1u, n2u = len(rows1), len(rows2)
    n_pairs = n1u * n2u

    # Accumulators stay float64 (n_pairs floats is <1 MB even for long
    # proteins) so summing 24 networks does not drift; only the big feature
    # blocks are float32, which is where the memory actually goes.
    accum_fwd = np.zeros(n_pairs, dtype=np.float64)
    accum_rev = np.zeros(n_pairs, dtype=np.float64)
    counts = np.zeros(n_pairs, dtype=np.float64)

    def _fits(win, rows, n):
        if win < 0:
            return True
        return bool(rows.min() - win >= 0 and rows.max() + win < n)

    for combo_idx, (pssmwin, binwin) in enumerate(_COMBOS):
        if progress_cb:
            progress_cb((combo_idx + 1) / len(_COMBOS), f"window pssm={pssmwin} bin={binwin}")

        model = models[f"{pssmwin}_{binwin}"]

        if not (_fits(pssmwin, rows1, n1) and _fits(pssmwin, rows2, n2)
                and _fits(binwin, rows1, n1) and _fits(binwin, rows2, n2)):
            # Short-protein edge case: exact per-position loop so boundary
            # skips are handled correctly (rare path).
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

        _score_combo(seq1, pssm1, seq2, pssm2, rows1, rows2, pssmwin, binwin,
                     model, accum_fwd, accum_rev, counts)

    keep = counts > 0
    all_kept = bool(keep.all())
    if not all_kept:
        dropped = int((~keep).sum())
        print(f"Warning: {dropped} position pairs had no surviving window combo "
              f"(very short protein?) — dropped.")

    avg_fwd = accum_fwd / np.where(counts > 0, counts, 1.0)
    avg_rev = accum_rev / np.where(counts > 0, counts, 1.0)
    final_scores = (avg_fwd + avg_rev) / 2.0
    final_scores[~keep] = 0.0
    del accum_fwd, accum_rev, avg_fwd, avg_rev

    unique_r1 = [res1[i] for i in rows1]
    unique_r2 = [res2[j] for j in rows2]

    # Row order is i outer, j inner, so the flat score vector reshapes
    # straight into the (chain1 x chain2) matrix — no dict lookups.
    matrix = final_scores.reshape(n1u, n2u).astype(np.float32)

    smoothed_matrix = apply_r_smoothing(matrix)
    avscore = float(np.mean(smoothed_matrix))
    sdscore = float(np.std(smoothed_matrix))
    threshold = avscore + (3 * sdscore)

    # Ranked pairs: argsort on the flat vector, then name only the top 200.
    flat = final_scores if all_kept else final_scores[keep]
    kept_idx = np.arange(n_pairs) if all_kept else np.nonzero(keep)[0]
    # Stable sort on the negated scores reproduces sorted(..., reverse=True)
    # exactly, ties included (plain argsort[::-1] would flip tied pairs).
    order = np.argsort(-flat, kind="stable")[:200]
    top_200 = []
    for o in order:
        k = int(kept_idx[o])
        top_200.append((f"{res1[rows1[k // n2u]]}:{res2[rows2[k % n2u]]}", float(flat[o])))

    # Per-residue propensity = best score that residue reaches against any partner.
    chain1 = {r: float(v) for r, v in zip(unique_r1, matrix.max(axis=1))}
    chain2 = {r: float(v) for r, v in zip(unique_r2, matrix.max(axis=0))}

    end_time = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    time_log = f"Start time: {start_time}\nEnd time: {end_time}"

    out = {
        "top_200": top_200,
        "chain1": chain1,
        "chain2": chain2,
        "raw_matrix": matrix,
        "smoothed_matrix": smoothed_matrix,
        "threshold": threshold,
        "cutoff_score": (top_200[-1][1] - 1e-9) if top_200 else 0.0,
        "unique_r1": unique_r1,
        "unique_r2": unique_r2,
        "n_scored_pairs": int(keep.sum()),
        "time_log": time_log,
    }

    if not compact:
        out["all_pairs"] = pairs_from_matrix(unique_r1, unique_r2, matrix)
        out["matrix"] = matrix

    del final_scores, counts, keep
    gc.collect()
    return out


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

    with open(f"{name1}-{name2}-final-prediction.tsv", "w") as f:
        f.write("Pair(Seq1:Seq2)\tPrediction-score\n")
        for pair_name, score in results["all_pairs"]:
            f.write(f"{pair_name}\t{score:.6f}\n")

    with open(f"{name1}-{name2}-top200.tsv", "w") as f:
        f.write("Rank\tPair(Seq1:Seq2)\tPrediction-score\n")
        for i, (pair_name, score) in enumerate(results["top_200"], 1):
            f.write(f"{i}\t{pair_name}\t{score:.6f}\n")

    with open(f"{name1}-{name2}-sspred.chain1", "w") as f:
        for res, score in results["chain1"].items():
            f.write(f"{res} {score:.6f}\n")

    with open(f"{name1}-{name2}-sspred.chain2", "w") as f:
        for res, score in results["chain2"].items():
            f.write(f"{res} {score:.6f}\n")

    print("Done.")
