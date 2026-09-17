"""
batch.py — multi-profile PSSM ingestion and all-vs-all batch execution.

The user uploads one ZIP holding up to MAX_PROFILES PSSM files; every
unordered pair is scored with the same engine that scores a single pair,
one pair at a time, so peak memory stays at roughly one pair's cost.

Memory model (why it is done this way):
  * inference.run_prediction(..., compact=True) returns the raw score matrix
    instead of a ~36k-element list of (name, score) tuples. One 200x200 pair
    is then ~0.4 MB of state instead of ~8 MB; 45 pairs fit in ~20 MB.
  * Export files are written per pair and streamed into a ZIP on disk, so the
    master archive never sits in RAM in full.
  * gc.collect() runs between pairs.
"""

from __future__ import annotations

import gc
import io
import itertools
import os
import tempfile
import time
import zipfile

import numpy as np

from inference import read_pssm_from_text, run_prediction

MAX_PROFILES = 10
MAX_MEMBER_BYTES = 8 * 1024 * 1024        # one PSSM file, uncompressed
MAX_TOTAL_BYTES = 40 * 1024 * 1024        # whole archive, uncompressed
MIN_RESIDUES = 12                         # 5-residue margin each side + slack

_SKIP_PREFIXES = ("__MACOSX/", ".")
_SKIP_SUFFIXES = (".ds_store",)


class BatchInputError(ValueError):
    """Raised for anything wrong with the uploaded archive."""


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def decode_lines(raw: bytes) -> list[str]:
    try:
        return raw.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return raw.decode("latin-1").splitlines()


def extract_pssm_zip(zip_bytes: bytes, max_profiles: int = MAX_PROFILES) -> list[dict]:
    """Read a ZIP of PSSM profiles.

    Returns [{name, lines, n_residues}, ...] in archive order. Raises
    BatchInputError with a message meant to be shown to the user.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        raise BatchInputError("That file is not a readable ZIP archive.")

    with zf:
        members = []
        total = 0
        for info in zf.infolist():
            name = info.filename
            base = os.path.basename(name)
            if info.is_dir() or not base:
                continue
            if name.startswith(_SKIP_PREFIXES) or base.startswith("."):
                continue
            if base.lower().endswith(_SKIP_SUFFIXES):
                continue
            if info.file_size > MAX_MEMBER_BYTES:
                raise BatchInputError(
                    f"{base} is {info.file_size / 1e6:.1f} MB uncompressed; "
                    f"the per-file limit is {MAX_MEMBER_BYTES / 1e6:.0f} MB."
                )
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise BatchInputError(
                    f"The archive expands to more than {MAX_TOTAL_BYTES / 1e6:.0f} MB. "
                    "Please upload PSSM profiles only."
                )
            members.append(info)

        if not members:
            raise BatchInputError("The archive contains no files.")
        if len(members) > max_profiles:
            raise BatchInputError(
                f"The archive holds {len(members)} files; the limit is {max_profiles} "
                "PSSM profiles per run."
            )

        profiles, seen, rejected = [], {}, []
        for info in members:
            base = os.path.basename(info.filename)
            lines = decode_lines(zf.read(info))
            _, _, residues = read_pssm_from_text(lines)
            if len(residues) < MIN_RESIDUES:
                rejected.append(base)
                continue
            label = base
            if label in seen:                      # same basename in two folders
                seen[label] += 1
                label = f"{base}#{seen[label]}"
            else:
                seen[label] = 1
            profiles.append({"name": label, "lines": lines, "n_residues": len(residues)})

    if not profiles:
        raise BatchInputError(
            "No valid PSSM profiles found. Files must be PSI-BLAST ASCII PSSMs "
            "(the output of `psiblast -out_ascii_pssm`), each with at least "
            f"{MIN_RESIDUES} scored residues."
        )
    if len(profiles) < 2:
        raise BatchInputError(
            "Only one usable PSSM profile was found — a batch run needs at least two."
            + (f" Unreadable: {', '.join(rejected)}." if rejected else "")
        )

    if rejected:
        profiles[0]["_warning"] = (
            f"Skipped {len(rejected)} file(s) that could not be parsed as a PSSM: "
            + ", ".join(rejected)
        )
    return profiles


# ---------------------------------------------------------------------------
# Pair enumeration
# ---------------------------------------------------------------------------
def enumerate_pairs(names: list[str], include_self: bool = False) -> list[tuple[str, str]]:
    """All unordered combinations.

    The engine averages the forward and reverse direction internally, so
    (A, B) and (B, A) give the same scores — only one of each is run.
    include_self adds the homodimer case (A, A).
    """
    pairs = list(itertools.combinations(names, 2))
    if include_self:
        pairs = [(n, n) for n in names] + pairs
    return pairs


def pair_count(n_profiles: int, include_self: bool = False) -> int:
    return n_profiles * (n_profiles - 1) // 2 + (n_profiles if include_self else 0)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def pair_key(name1: str, name2: str) -> str:
    return f"{name1}||{name2}"


def run_batch(profiles: list[dict], pairs: list[tuple[str, str]], models,
              progress_cb=None, compact: bool = True) -> dict:
    """Score every pair sequentially.

    progress_cb(fraction, message) is called before each pair and after the
    last one, so a Streamlit progress bar can follow along.

    Returns {"order": [key, ...], "entries": {key: entry}, "names": [...],
             "elapsed": seconds}, where each entry is the compact result dict
    plus name1 / name2 / elapsed / n_residues1 / n_residues2.
    """
    by_name = {p["name"]: p for p in profiles}
    entries, order = {}, []
    t_all = time.time()

    for idx, (a, b) in enumerate(pairs):
        if progress_cb:
            progress_cb(idx / max(1, len(pairs)), f"Scoring {a} vs {b} ({idx + 1}/{len(pairs)})")
        t0 = time.time()
        res = run_prediction(by_name[a]["lines"], by_name[b]["lines"],
                             models=models, compact=compact)
        res["name1"], res["name2"] = a, b
        res["elapsed"] = time.time() - t0
        res["n_residues1"] = by_name[a]["n_residues"]
        res["n_residues2"] = by_name[b]["n_residues"]
        key = pair_key(a, b)
        entries[key] = res
        order.append(key)
        gc.collect()

    if progress_cb:
        progress_cb(1.0, f"Completed {len(pairs)} pair(s)")

    return {
        "order": order,
        "entries": entries,
        "names": [p["name"] for p in profiles],
        "elapsed": time.time() - t_all,
    }


# ---------------------------------------------------------------------------
# Batch-level summaries
# ---------------------------------------------------------------------------
def batch_summary_rows(batch: dict) -> list[dict]:
    """One row per pair, ranked by peak score — the batch's headline table.

    Deliberately excludes the smoothed-matrix statistics (mean, SD and the
    mean+3SD line): they describe the visualisation-only smoothed matrix, not
    the ranking, and they are not comparable between pairs of different sizes.
    """
    rows = []
    for key in batch["order"]:
        e = batch["entries"][key]
        top_pair, top_score = e["top_200"][0] if e["top_200"] else ("-", 0.0)
        cutoff = e.get("cutoff_score", 0.0)
        rows.append({
            "Target": e["name1"],
            "Partner": e["name2"],
            "Geometry": f"{len(e['unique_r1'])} x {len(e['unique_r2'])}",
            "Scored_pairs": e.get("n_scored_pairs", len(e["unique_r1"]) * len(e["unique_r2"])),
            "Peak_pair": top_pair,
            "Peak_score": round(float(top_score), 6),
            "Top200_cutoff": round(float(cutoff), 6),
            "Runtime_s": round(float(e.get("elapsed", 0.0)), 2),
            "_key": key,
        })
    rows.sort(key=lambda r: r["Peak_score"], reverse=True)
    return rows


def batch_summary_tsv(batch: dict) -> bytes:
    rows = batch_summary_rows(batch)
    cols = [c for c in rows[0] if not c.startswith("_")] if rows else []
    buf = io.StringIO()
    buf.write("Rank\t" + "\t".join(cols) + "\n")
    for i, r in enumerate(rows, 1):
        buf.write(str(i) + "\t" + "\t".join(str(r[c]) for c in cols) + "\n")
    data = buf.getvalue().encode()
    buf.close()
    return data


def pair_score_matrix(batch: dict, metric: str = "peak") -> tuple[list[str], np.ndarray]:
    """Symmetric protein x protein overview matrix.

    metric "peak"   -> best residue-pair score in that run
           "cutoff" -> the top-200 cutoff score (a crude density measure)
    Cells with no run (the diagonal, when self-pairs were not requested)
    are NaN so the heatmap leaves them blank.
    """
    names = batch["names"]
    idx = {n: i for i, n in enumerate(names)}
    mat = np.full((len(names), len(names)), np.nan, dtype=np.float32)
    for key in batch["order"]:
        e = batch["entries"][key]
        val = (e["top_200"][0][1] if metric == "peak" else e.get("cutoff_score", 0.0)) \
            if e["top_200"] else 0.0
        i, j = idx[e["name1"]], idx[e["name2"]]
        mat[i, j] = val
        mat[j, i] = val
    return names, mat


def top_pairs_across_batch(batch: dict, n: int = 200) -> list[tuple[str, str, str, float]]:
    """Global ranking: the n best residue pairs seen anywhere in the batch."""
    pooled = []
    for key in batch["order"]:
        e = batch["entries"][key]
        for name, score in e["top_200"]:
            pooled.append((e["name1"], e["name2"], name, float(score)))
    pooled.sort(key=lambda r: r[3], reverse=True)
    return pooled[:n]


def global_top_tsv(batch: dict, n: int = 200) -> bytes:
    buf = io.StringIO()
    buf.write("Rank\tTarget\tPartner\tResidue_pair(Seq1:Seq2)\tPrediction-score\n")
    for i, (a, b, pair, score) in enumerate(top_pairs_across_batch(batch, n), 1):
        buf.write(f"{i}\t{a}\t{b}\t{pair}\t{score:.6f}\n")
    data = buf.getvalue().encode()
    buf.close()
    return data


# ---------------------------------------------------------------------------
# Master archive
# ---------------------------------------------------------------------------
def build_master_zip(batch: dict, file_writer, outdir: str | None = None,
                     progress_cb=None) -> str:
    """Write every pair's result files into one ZIP on disk and return its path.

    file_writer(results, name1, name2, elapsed) -> {filename: bytes}; the app
    passes its own exporter. Files are written and dropped one pair at a time,
    so RAM never holds more than a single pair's exports.
    """
    outdir = outdir or tempfile.mkdtemp(prefix="ppip_batch_")
    path = os.path.join(outdir, "ppip-batch-all-results.zip")
    n = len(batch["order"])

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("batch-summary.tsv", batch_summary_tsv(batch))
        zf.writestr("batch-top200-global.tsv", global_top_tsv(batch))
        names, mat = pair_score_matrix(batch)
        buf = io.StringIO()
        buf.write("Protein\t" + "\t".join(names) + "\n")
        for i, nm in enumerate(names):
            buf.write(nm + "\t" + "\t".join(
                "" if np.isnan(v) else f"{v:.6f}" for v in mat[i]) + "\n")
        zf.writestr("batch-pair-peak-score-matrix.tsv", buf.getvalue())
        buf.close()

        for i, key in enumerate(batch["order"]):
            e = batch["entries"][key]
            if progress_cb:
                progress_cb((i + 1) / max(1, n), f"Packing {e['name1']} vs {e['name2']}")
            files = file_writer(e, e["name1"], e["name2"], e.get("elapsed"))
            folder = f"{e['name1']}__vs__{e['name2']}"
            for fname, data in files.items():
                if fname.endswith(".zip"):     # no nested per-pair archives
                    continue
                zf.writestr(f"{folder}/{fname}", data)
            del files
            gc.collect()

    return path
