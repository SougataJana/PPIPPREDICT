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

def get_features_matrix(matrix, win):
    """Precompute sliding window features for all positions in a matrix efficiently."""
    size = matrix.shape[0]
    feat_dim = matrix.shape[1]
    # Total features per position: center + win left + win right
    total_dim = feat_dim * (2 * win + 1)
    features = np.zeros((size, total_dim), dtype=np.float32)
    
    for i in range(win, size - win):
        slice_list = [matrix[i]]
        for m in range(1, win + 1):
            slice_list.append(matrix[i + m])
        for m in range(1, win + 1):
            slice_list.append(matrix[i - m])
        features[i] = np.concatenate(slice_list)
    return features

def read_pssm_from_text(file_lines):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    seq_bin, pssm, residues = [], [], []
    pos = 0
    for l in file_lines:
        parts = l.split()
        if len(parts) > 40 and parts[0].isdigit():
            res = parts[1].upper()
            if res in amino_acids:
                pos += 1
                residues.append(f"{res}{pos}")
                seq_bin.append([1.0 if res == aa else 0.0 for aa in amino_acids])
                pssm.append([float(x) for x in parts[2:22]])
    return np.array(seq_bin, dtype=np.float32), np.array(pssm, dtype=np.float32), residues

def apply_r_smoothing(matrix, halfwin=1, step1=4, step2=1):
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

def run_prediction(lines1, lines2):
    start_time = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    
    seq1, pssm1, res1 = read_pssm_from_text(lines1)
    seq2, pssm2, res2 = read_pssm_from_text(lines2)

    weights = torch.load('ppip_ensemble_weights.pt', map_location='cpu')
    models = {}
    for key, w in weights.items():
        model = Exact2011Model(w['input_dim'])
        model.load_state_dict({k: v for k, v in w.items() if k != 'input_dim'})
        model.eval()
        models[key] = model

    len_res1 = len(res1)
    len_res2 = len(res2)

    valid_configs = []
    for pssmwin in range(-1, 4):
        for binwin in range(-1, 4):
            if pssmwin + binwin > -2:
                valid_configs.append((pssmwin, binwin))

    # Precompute all window features for valid sizes (-1 to 3)
    seq_feats = {w: (get_features_matrix(seq1, w), get_features_matrix(seq2, w)) for w in range(1, 4)}
    pssm_feats = {w: (get_features_matrix(pssm1, w), get_features_matrix(pssm2, w)) for w in range(1, 4)}

    raw_pairs = []

    # Vectorized loop structure across residue pairs with batch evaluation
    for pssmwin, binwin in valid_configs:
        model_key = f"{pssmwin}_{binwin}"
        model = models[model_key]
        
        # Collect all valid index pairs for this configuration
        valid_i, valid_j = [], []
        for i in range(5, len_res1 - 5):
            if pssmwin > -1 and (i - pssmwin < 0 or i + pssmwin >= len_res1):
                continue
            if binwin > -1 and (i - binwin < 0 or i + binwin >= len_res1):
                continue
            for j in range(5, len_res2 - 5):
                if pssmwin > -1 and (j - pssmwin < 0 or j + pssmwin >= len_res2):
                    continue
                if binwin > -1 and (j - binwin < 0 or j + binwin >= len_res2):
                    continue
                valid_i.append(i)
                valid_j.append(j)

        if not valid_i:
            continue

        valid_i = np.array(valid_i)
        valid_j = np.array(valid_j)

        # Build feature vectors for forward and reverse directions
        fwd_list, rev_list = [], []
        if binwin > -1:
            b1_1, b2_1 = seq_feats[binwin]
            fwd_list.append(b1_1[valid_i])
            fwd_list.append(b2_1[valid_j])
            rev_list.append(b2_1[valid_j])
            rev_list.append(b1_1[valid_i])
            
        if pssmwin > -1:
            p1_1, p2_1 = pssm_feats[pssmwin]
            fwd_list.append(p1_1[valid_i])
            fwd_list.append(p2_1[valid_j])
            rev_list.append(p2_1[valid_j])
            rev_list.append(p1_1[valid_i])

        x_fwd = torch.tensor(np.concatenate(fwd_list, axis=1), dtype=torch.float32)
        x_rev = torch.tensor(np.concatenate(rev_list, axis=1), dtype=torch.float32)

        with torch.no_grad():
            scores_fwd = model(x_fwd).squeeze(1).numpy()
            scores_rev = model(x_rev).squeeze(1).numpy()

        # Accumulate results across configurations
        if not raw_pairs:
            for idx in range(len(valid_i)):
                i, j = valid_i[idx], valid_j[idx]
                raw_pairs.append({
                    'name': f"{res1[i]}:{res2[j]}",
                    'fwd_sum': scores_fwd[idx],
                    'rev_sum': scores_rev[idx],
                    'count': 1
                })
            # Convert list to dict for fast O(1) accumulation
            pair_map = {p['name']: p for p in raw_pairs}
        else:
            for idx in range(len(valid_i)):
                i, j = valid_i[idx], valid_j[idx]
                name = f"{res1[i]}:{res2[j]}"
                if name in pair_map:
                    pair_map[name]['fwd_sum'] += scores_fwd[idx]
                    pair_map[name]['rev_sum'] += scores_rev[idx]
                    pair_map[name]['count'] += 1
                else:
                    entry = {'name': name, 'fwd_sum': scores_fwd[idx], 'rev_sum': scores_rev[idx], 'count': 1}
                    pair_map[name] = entry
                    raw_pairs.append(entry)

    # Compute final averaged scores
    final_pairs = []
    for p in raw_pairs:
        avg_fwd = p['fwd_sum'] / p['count']
        avg_rev = p['rev_sum'] / p['count']
        final_score = (avg_fwd + avg_rev) / 2.0
        final_pairs.append((p['name'], float(final_score)))

    sorted_pairs = sorted(final_pairs, key=lambda x: x[1], reverse=True)
    top_200 = sorted_pairs[:200]

    unique_r1 = sorted(list(set([p[0].split(":")[0] for p in final_pairs])), key=lambda x: int(re.search(r'\d+', x).group()))
    unique_r2 = sorted(list(set([p[0].split(":")[1] for p in final_pairs])), key=lambda x: int(re.search(r'\d+', x).group()))
    
    matrix = np.zeros((len(unique_r1), len(unique_r2)))
    score_lookup = dict(final_pairs)
    for i, r1 in enumerate(unique_r1):
        for j, r2 in enumerate(unique_r2):
            matrix[i, j] = score_lookup.get(f"{r1}:{r2}", 0.0)

    smoothed_matrix = apply_r_smoothing(matrix)
    avscore = np.mean(smoothed_matrix)
    sdscore = np.std(smoothed_matrix)
    threshold = avscore + (3 * sdscore)

    chain1_scores, chain2_scores = {}, {}
    for pair_name, score in final_pairs:
        c1, c2 = pair_name.split(":")
        chain1_scores.setdefault(c1, []).append(score)
        chain2_scores.setdefault(c2, []).append(score)

    end_time = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    time_log = f"Start time: {start_time}\nPattern file completed: {start_time}\nStage 1 predictions completed: {start_time}\nEnd time: {end_time}"

    return {
        "all_pairs": final_pairs,
        "top_200": top_200,
        "chain1": {k: max(v) for k, v in chain1_scores.items()},
        "chain2": {k: max(v) for k, v in chain2_scores.items()},
        "matrix": matrix,
        "smoothed_matrix": smoothed_matrix,
        "threshold": threshold,
        "unique_r1": unique_r1,
        "unique_r2": unique_r2,
        "time_log": time_log
    }

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inference.py <pssm_file_1> <pssm_file_2>")
        sys.exit(1)

    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    
    name1 = os.path.basename(file1_path)
    name2 = os.path.basename(file2_path)

    with open(file1_path, 'r') as f:
        lines1 = f.readlines()
    with open(file2_path, 'r') as f:
        lines2 = f.readlines()

    results = run_prediction(lines1, lines2)
    
    final_pred_file = f"{name1}-{name2}-final-prediction.txt"
    with open(final_pred_file, "w") as f:
        f.write("Pair(Seq1:Seq2)\tPrediction-score\n")
        for pair_name, score in results["all_pairs"]:
            f.write(f"{pair_name}: {score:.6f}\n")

    top200_file = f"{name1}-{name2}-top200.txt"
    with open(top200_file, "w") as f:
        f.write("Pair(Seq1:Seq2)\tPrediction-score\n")
        for pair_name, score in results["top_200"]:
            f.write(f"{pair_name}: {score:.6f}\n")

    chain1_file = f"{name1}-{name2}-sspred.chain1"
    with open(chain1_file, "w") as f:
        for res, score in results["chain1"].items():
            f.write(f"{res} {score:.6f}\n")

    chain2_file = f"{name1}-{name2}-sspred.chain2"
    with open(chain2_file, "w") as f:
        for res, score in results["chain2"].items():
            f.write(f"{res} {score:.6f}\n")
