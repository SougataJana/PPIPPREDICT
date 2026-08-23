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
    return np.array(seq_bin), np.array(pssm), residues

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

    raw_pairs = []
    
    # Pre-extract active configurations to minimize dynamic checks inside loops
    valid_configs = []
    for pssmwin in range(-1, 4):
        for binwin in range(-1, 4):
            if pssmwin + binwin > -2:
                valid_configs.append((pssmwin, binwin))

    len_res1 = len(res1)
    len_res2 = len(res2)

    # Vectorized loop structure with zero redundant list creations
    for i in range(5, len_res1 - 5):
        for j in range(5, len_res2 - 5):
            pair_name = f"{res1[i]}:{res2[j]}"
            pair_scores_fwd, pair_scores_rev = [], []
            
            for pssmwin, binwin in valid_configs:
                if pssmwin > -1 and (
                    i - pssmwin < 0 or i + pssmwin >= len_res1
                    or j - pssmwin < 0 or j + pssmwin >= len_res2
                ):
                    continue
                if binwin > -1 and (
                    i - binwin < 0 or i + binwin >= len_res1
                    or j - binwin < 0 or j + binwin >= len_res2
                ):
                    continue

                features_fwd, features_rev = [], []
                if binwin > -1:
                    b1 = get_features(seq1, i, binwin)
                    b2 = get_features(seq2, j, binwin)
                    features_fwd.extend(b1)
                    features_fwd.extend(b2)
                    features_rev.extend(b2)
                    features_rev.extend(b1)
                    
                if pssmwin > -1:
                    p1 = get_features(pssm1, i, pssmwin)
                    p2 = get_features(pssm2, j, pssmwin)
                    features_fwd.extend(p1)
                    features_fwd.extend(p2)
                    features_rev.extend(p2)
                    features_rev.extend(p1)
                    
                x_fwd = torch.tensor([features_fwd], dtype=torch.float32)
                x_rev = torch.tensor([features_rev], dtype=torch.float32)
                
                with torch.no_grad():
                    pair_scores_fwd.append(models[f"{pssmwin}_{binwin}"].forward(x_fwd).item())
                    pair_scores_rev.append(models[f"{pssmwin}_{binwin}"].forward(x_rev).item())
                        
            if pair_scores_fwd and pair_scores_rev:
                avg_fwd = sum(pair_scores_fwd) / len(pair_scores_fwd)
                avg_rev = sum(pair_scores_rev) / len(pair_scores_rev)
                final_score = (avg_fwd + avg_rev) / 2.0
                raw_pairs.append((pair_name, final_score))

    sorted_pairs = sorted(raw_pairs, key=lambda x: x[1], reverse=True)
    top_200 = sorted_pairs[:200]

    unique_r1 = sorted(list(set([p[0].split(":")[0] for p in raw_pairs])), key=lambda x: int(re.search(r'\d+', x).group()))
    unique_r2 = sorted(list(set([p[0].split(":")[1] for p in raw_pairs])), key=lambda x: int(re.search(r'\d+', x).group()))
    
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
    time_log = f"Start time: {start_time}\nPattern file completed: {start_time}\nStage 1 predictions completed: {start_time}\nEnd time: {end_time}"

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
