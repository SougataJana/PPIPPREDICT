# PPIPPREDICT

A modern Python/PyTorch port of **PPIPP**, the partner-aware protein–protein interface predictor of Ahmad & Mizuguchi (2011), with a Streamlit web app.

Given the PSI-BLAST PSSM profiles of two proteins, it scores every residue pair (one residue from each protein) for the likelihood that the two residues are in contact across the interface, using sequence information only (no structure).

> **Reference:** Ahmad S, Mizuguchi K (2011). *Partner-Aware Prediction of Interacting Residues in Protein-Protein Complexes from Sequence Data.* PLoS ONE 6(12): e29104. https://doi.org/10.1371/journal.pone.0029104

## What it does

Most sequence-based predictors score each residue alone ("would this residue bind *any* partner?"). PPIPP scores residue **pairs**, so a residue is predicted to bind only when the specific partner contains a compatible region. In the original paper, pair-trained models outperformed single-residue models on both pair and single-residue prediction (leave-one-out over 124 docking-benchmark complexes).

### Pipeline

1. **Features.** For a residue pair (i, j), take a window of sequence neighbors around each residue, encoded two ways: sparse one-hot amino-acid identity and PSSM evolutionary profile (20 values per position).
2. **Stage 1: 24 networks.** Window half-widths of 0, 1, 2 or 3 neighbors (windows of 1, 3, 5 or 7 residues) or "feature not used" give 5 options per encoding. 5 × 5 − 1 (the empty combination) = **24** small feed-forward networks (one hidden layer of 4 sigmoid units, one sigmoid output). Each pair is scored in both chain orders (i,j) and (j,i) and the two are averaged.
3. **Stage 2.** The 24 network outputs are averaged into the final pair score.
4. **Outputs.** All pair scores, the top 200 ranked pairs, and per-residue propensities (the best score each residue reaches against any partner residue).
5. **Visualization smoothing (app only).** A moving-average filter is applied to the score matrix for the heatmap and 3D views. It is not part of the published method and does not affect the ranked pairs.

## What this repository adds

- The legacy runtime was the Stuttgart Neural Network Simulator (SNNS). Weights were converted into a single PyTorch file, `ppip_ensemble_weights.pt`, holding all 24 sub-models.
- Inference is vectorized: one batched forward pass per (window combination, direction), instead of one call per residue pair.
- The feature matrix is built in blocks sized to a byte budget and stored as float32, which keeps peak memory low (designed to fit a 1 GB container; see [Memory](#memory)).
- A PSSM reader that only accepts real data rows of a PSI-BLAST ASCII PSSM, with 0-based residue numbering.
- A Streamlit app with ranked tables, heatmap, 3D landscape, score histogram, linear and circular contact maps, and downloadable result files. A batch mode scores all pairs from a ZIP of PSSM files.

### Validation status

`inference.py` documents that the `-1_0` sub-model matches the original SNNS network exactly (weights, biases and full forward-pass output agree to float32 precision), and that residue labeling was cross-checked against reference output from the original system. **Agreement for the other 23 sub-models has not been documented here.** Add a full equivalence check before treating the port as bit-for-bit faithful.

## Repository contents

| File | Purpose |
|---|---|
| `inference.py` | Core engine: PSSM parsing, 24-model ensemble scoring, smoothing, command-line interface |
| `app.py` | Streamlit web application |
| `ppip_ensemble_weights.pt` | Converted weights for the 24 networks |
| `requirements.txt` | Python dependencies |
| `sciwhylab_logo.png` | Logo |

> **Note:** `app.py` imports a module named `batch` (batch-mode helpers) and optionally displays `sample_output.html`. Neither is in the repository root at the time of writing. Add them, or the app will fail on startup.

## Installation

```bash
git clone https://github.com/SougataJana/PPIPPREDICT.git
cd PPIPPREDICT
pip install -r requirements.txt
```

Dependencies: `torch`, `numpy`, `pandas`, `plotly`, `streamlit`, `matplotlib`. For PNG export of figures from the app, also install `kaleido`; without it figures are exported as interactive HTML.

## Input format

One **PSI-BLAST ASCII PSSM** file per protein (the `-out_ascii_pssm` output). Lines are read as data rows when they have more than 40 columns and a numeric first column; the amino-acid letter is column 2 and the 20 log-odds scores are columns 3–22.

Requirements and behavior:
- At least 12 residues per protein.
- The first and last 5 residues of each protein are not scored (a margin for the widest window), so scored positions run from residue 5 to n − 6.
- Residues are labeled with 0-based positions, e.g. `A0`, `L41`. Pair names are written `residue1:residue2`, e.g. `K12:D40`.

To generate a PSSM, run PSI-BLAST against a protein database (the original method used 3 iterations against NCBI NR with default parameters).

## Usage

### Command line

```bash
python inference.py protein1.pssm protein2.pssm
```

Writes four files to the current directory, named from the two inputs:

| Output | Content |
|---|---|
| `<p1>-<p2>-final-prediction.tsv` | Score for every residue pair |
| `<p1>-<p2>-top200.tsv` | Top 200 ranked pairs (rank, pair, score) |
| `<p1>-<p2>-sspred.chain1` | Per-residue propensity for protein 1 |
| `<p1>-<p2>-sspred.chain2` | Per-residue propensity for protein 2 |

### Python

```python
from inference import load_models, run_prediction

models = load_models("ppip_ensemble_weights.pt")   # load once, reuse
with open("protein1.pssm") as a, open("protein2.pssm") as b:
    result = run_prediction(a.readlines(), b.readlines(), models=models)

print(result["top_200"][:5])        # [("K12:D40", 0.93), ...]
print(result["chain1"])             # per-residue propensity, protein 1
```

Use `compact=True` to keep the score matrix instead of a full list of pair tuples, which is much lighter on memory when holding many results.

### Web app

```bash
streamlit run app.py
```

Upload two PSSM files (single pair) or a ZIP of PSSM profiles (batch, every pair scored). The app shows ranked pairs, per-residue profiles, heatmap, 3D landscape, score distribution, linear and circular contact maps, and an export tab.

## Memory

`inference.py` builds features in row blocks limited by an environment variable, so large proteins do not need large RAM:

```bash
export PPIP_MAX_BLOCK_BYTES=48000000   # default, about 48 MB of float32 per block
```

Lower it on tighter machines. Accumulators are kept in float64 and are small relative to the feature blocks.

## Limitations

- **Sequence-only.** No structural information is used. As in the original paper, performance is lower for complexes that change shape substantially on binding.
- **Small training data.** The model was trained on 124 complexes from a docking benchmark (DBD 3.0). Treat scores as a ranking signal, not calibrated probabilities.
- **Not state of the art.** The architecture predates protein language models and structure-prediction tools such as AlphaFold-Multimer. It is best used as a fast baseline and a reference implementation of the partner-aware idea.
- **Benchmarking of this port.** [Add: runtime before/after, and agreement with the original binaries on N test pairs.]

## Citation

If you use this software, please cite the original paper:

```
Ahmad S, Mizuguchi K (2011) Partner-Aware Prediction of Interacting Residues in
Protein-Protein Complexes from Sequence Data. PLoS ONE 6(12): e29104.
doi:10.1371/journal.pone.0029104
```

## Acknowledgements

The method was developed by Shandar Ahmad and Kenji Mizuguchi. This port was developed at SciWhy Lab, Jawaharlal Nehru University. [Add: disclosure of your role and your relationship to the original authors.]

## License

[Add a license. None is currently included in the repository. Check that you may redistribute the converted model weights before choosing one.]
