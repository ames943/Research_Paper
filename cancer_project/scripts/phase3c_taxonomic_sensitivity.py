#!/usr/bin/env python3
"""
Phase 3c: Taxonomic level sensitivity analysis.

Builds CLR matrices at species (S) and phylum (P) level from the same
118 Kraken2 reports used for the genus pipeline, then runs PERMANOVA
(Aitchison distance, 999 perms) for response and batch factors at each level.

Compared against genus-level results from prior phases:
  Genus  : batch R²=0.0768, response R²=0.0068 (ratio 11.3×)

Question: does the batch >> signal pattern hold at species and phylum level?

Outputs: results/ml/taxonomic_sensitivity/
  X_species_clr.tsv
  X_phylum_clr.tsv
  X_species_raw.tsv
  X_phylum_raw.tsv
  permanova_by_level.tsv
  taxonomic_sensitivity_summary.md
"""

import os, math, glob, time
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

REPORT_DIR  = "results/kraken_reports"
LABELS_PATH = "metadata/response_labels_3cohort.tsv"
OUT_DIR     = "results/ml/taxonomic_sensitivity"
os.makedirs(OUT_DIR, exist_ok=True)

PSEUDOCOUNT    = 1e-6
EXCLUDED_RANKS = {"S1", "S2"}  # sub-species — include only S
N_PERMS        = 999
EXCLUDED_GENERA = {"Homo"}

# Rank codes: P=phylum, G=genus, S=species
TARGET_RANKS = {"S": "species", "P": "phylum"}


# ── Parse all Kraken2 reports at multiple ranks ────────────────────────────────

def parse_reports(rank_code):
    """
    Parse all 118 Kraken2 reports at the given rank_code (S or P).
    Returns (samples dict, all_taxa set, excluded set).
    """
    report_files = sorted(glob.glob(f"{REPORT_DIR}/*_report.txt"))
    if not report_files:
        raise SystemExit(f"No Kraken reports found in {REPORT_DIR}")

    samples   = {}
    all_taxa  = set()
    fixed     = {}
    excluded  = set()

    for fp in report_files:
        sample    = os.path.basename(fp).replace("_report.txt", "")
        taxa_data = {}

        with open(fp) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue
                rk = parts[3]
                if rk != rank_code:
                    continue

                pct          = float(parts[0].strip())
                raw_name     = parts[5]
                stripped     = raw_name.strip()
                if not stripped:
                    continue

                # Same name parsing as build_matrix.py
                if stripped.startswith("Candidatus "):
                    name = "Candidatus_" + stripped.split()[1]
                elif " (" in stripped:
                    name = stripped.split(" (")[0]
                else:
                    name = stripped

                # For species: use full canonical name (two words max, collapse extra)
                if rank_code == "S":
                    tokens = name.split()
                    name   = " ".join(tokens[:2]) if len(tokens) > 1 else tokens[0]

                # Exclude host contamination (at phylum level this is unlikely but check)
                if rank_code == "P" and name in EXCLUDED_GENERA:
                    excluded.add(name)
                    continue

                taxa_data[name] = taxa_data.get(name, 0.0) + pct
                all_taxa.add(name)

        samples[sample] = taxa_data

    return samples, all_taxa, excluded


def build_matrix(samples, all_taxa, path_raw, path_clr):
    all_taxa_sorted = sorted(all_taxa)
    labels          = pd.read_csv(LABELS_PATH, sep="\t")["run_accession"].tolist()
    # Keep only samples that have response labels; use label order
    present         = [s for s in labels if s in samples]

    with open(path_raw, "w") as fr, open(path_clr, "w") as fc:
        header = "run_accession\t" + "\t".join(all_taxa_sorted) + "\n"
        fr.write(header); fc.write(header)

        for sample in present:
            vals     = [samples[sample].get(t, 0.0) for t in all_taxa_sorted]
            # raw
            fr.write(sample + "\t" + "\t".join(str(v) for v in vals) + "\n")
            # CLR
            lv       = [math.log(v + PSEUDOCOUNT) for v in vals]
            mean_lv  = sum(lv) / len(lv)
            clr_vals = [v - mean_lv for v in lv]
            fc.write(sample + "\t" + "\t".join(f"{v:.6f}" for v in clr_vals) + "\n")

    return present, all_taxa_sorted


# ── PERMANOVA (identical implementation to batch_diagnostics_3cohort.py) ──────

def _ss_total(d2):
    return float(np.sum(np.triu(d2, k=1))) / d2.shape[0]

def _ss_within(d2, grp):
    sw = 0.0
    for g in np.unique(grp):
        mask = grp == g
        n_g  = int(mask.sum())
        if n_g < 2:
            continue
        sub = d2[np.ix_(mask, mask)]
        sw += float(np.sum(np.triu(sub, k=1))) / n_g
    return sw

def permanova(dist_mat, grouping, n_perms=N_PERMS, seed=42):
    grp = np.asarray(grouping)
    n   = dist_mat.shape[0]
    d2  = dist_mat ** 2
    q   = len(np.unique(grp))

    SS_T = _ss_total(d2)
    SS_W = _ss_within(d2, grp)
    SS_A = SS_T - SS_W
    F    = (SS_A / (q - 1)) / (SS_W / (n - q))
    R2   = SS_A / SS_T

    rng    = np.random.default_rng(seed)
    perm_F = np.empty(n_perms)
    for i in range(n_perms):
        gp        = rng.permutation(grp)
        sw        = _ss_within(d2, gp)
        sa        = SS_T - sw
        perm_F[i] = (sa / (q - 1)) / (sw / (n - q))

    p_val = float((perm_F >= F).sum()) / n_perms
    return dict(R2=round(float(R2), 4), F_stat=round(float(F), 4),
                p_value=round(p_val, 4), n_groups=q)


# ── Main ───────────────────────────────────────────────────────────────────────

labels_df  = pd.read_csv(LABELS_PATH, sep="\t").set_index("run_accession")
response_s = labels_df["response"]

all_results = []

# Prior genus results (from batch_diagnostics and Phase 1 paper context)
prior_genus = [
    dict(level="genus", factor="batch",    n_taxa=2813, R2=0.0768, F_stat=None, p_value=0.001,  n_groups=3, ratio_to_response=None),
    dict(level="genus", factor="response", n_taxa=2813, R2=0.0068, F_stat=None, p_value=0.847,  n_groups=2, ratio_to_response=None),
]
all_results.extend(prior_genus)

for rank_code, rank_name in [("S", "species"), ("P", "phylum")]:
    print(f"\n[{time.strftime('%H:%M:%S')}] Parsing Kraken2 reports at {rank_name} (rank={rank_code}) …", flush=True)
    samples, all_taxa, excluded = parse_reports(rank_code)
    print(f"  Samples parsed: {len(samples)}, {rank_name}: {len(all_taxa)}", flush=True)

    path_raw = f"{OUT_DIR}/X_{rank_name}_raw.tsv"
    path_clr = f"{OUT_DIR}/X_{rank_name}_clr.tsv"
    present, all_taxa_sorted = build_matrix(samples, all_taxa, path_raw, path_clr)
    print(f"  Saved: {path_raw}, {path_clr}  (n={len(present)})", flush=True)

    clr_df = pd.read_csv(path_clr, sep="\t", index_col="run_accession")
    resp   = response_s.reindex(clr_df.index).dropna()
    clr_df = clr_df.loc[resp.index]

    batch  = pd.Series(
        ["cohort1" if a.startswith("SRR5930") else
         "cohort2" if a.startswith("SRR11413") else "cohort3"
         for a in clr_df.index],
        index=clr_df.index,
    )

    print(f"  Computing Aitchison distances (n={len(clr_df)}, {clr_df.shape[1]} {rank_name}) …", flush=True)
    D = squareform(pdist(clr_df.values.astype(float), metric="euclidean"))

    for factor_name, grouping in [("batch", batch.values), ("response", resp.values)]:
        print(f"  PERMANOVA — {factor_name} ({N_PERMS} perms) …", flush=True)
        res = permanova(D, grouping, seed=42 if factor_name == "batch" else 43)
        all_results.append(dict(
            level=rank_name, factor=factor_name,
            n_taxa=clr_df.shape[1], **res,
            ratio_to_response=None,
        ))
        print(f"    R²={res['R2']:.4f}  p={res['p_value']:.4f}  F={res['F_stat']:.4f}", flush=True)

# Compute batch:response ratio per level
result_df = pd.DataFrame(all_results)
for lvl in result_df["level"].unique():
    mask_b = (result_df["level"] == lvl) & (result_df["factor"] == "batch")
    mask_r = (result_df["level"] == lvl) & (result_df["factor"] == "response")
    if mask_b.any() and mask_r.any():
        r2_b = result_df.loc[mask_b, "R2"].iloc[0]
        r2_r = result_df.loc[mask_r, "R2"].iloc[0]
        ratio = round(r2_b / r2_r, 1) if r2_r > 0 else float("inf")
        result_df.loc[mask_b, "ratio_to_response"] = ratio
        result_df.loc[mask_r, "ratio_to_response"] = ratio

result_df.to_csv(f"{OUT_DIR}/permanova_by_level.tsv", sep="\t", index=False)
print(f"\nSaved: {OUT_DIR}/permanova_by_level.tsv", flush=True)

# ── Summary markdown ───────────────────────────────────────────────────────────

lines = ["# Phase 3c: Taxonomic Level Sensitivity\n",
         "PERMANOVA (Aitchison distance, 999 perms) — batch vs. response R² at each taxonomic level\n",
         "Genus-level values from prior batch_diagnostics runs.\n"]

lines.append(f"{'Level':<10} {'Factor':<10} {'n_taxa':>7} {'R²':>8} {'p':>8} {'batch:resp ratio':>16}")
lines.append("-" * 65)

for lvl in ["phylum", "genus", "species"]:
    for fac in ["batch", "response"]:
        row = result_df[(result_df["level"] == lvl) & (result_df["factor"] == fac)]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        sig = " ***" if r["p_value"] < 0.001 else " **" if r["p_value"] < 0.01 else \
              " *"   if r["p_value"] < 0.05  else " ns"
        ratio_str = f"{r['ratio_to_response']:.1f}×" if fac == "batch" and r["ratio_to_response"] is not None else "—"
        lines.append(f"{lvl:<10} {fac:<10} {int(r['n_taxa']) if r['n_taxa'] == r['n_taxa'] else '?':>7} "
                     f"{r['R2']:>8.4f} {r['p_value']:>8.4f}{sig}  {ratio_str:>12}")

lines.append("")
lines.append("## Interpretation")
lines.append("")
lines.append("If batch R² >> response R² at all three levels, the fundamental problem")
lines.append("(batch dominates signal regardless of taxonomic resolution) is not an artifact")
lines.append("of genus-level aggregation — it reflects a genuine platform/protocol confound.")
lines.append("Species level may show LARGER batch effects (more granular = more platform noise)")
lines.append("while phylum level may show SMALLER batch effects (more aggregated = less noise).")
lines.append("Either way, if response R² stays near genus level (0.007), the null finding is robust.")

with open(f"{OUT_DIR}/taxonomic_sensitivity_summary.md", "w") as fh:
    fh.write("\n".join(lines) + "\n")

print(f"\n[{time.strftime('%H:%M:%S')}] Phase 3c complete. Outputs in {OUT_DIR}/", flush=True)
print(result_df[["level","factor","n_taxa","R2","p_value","ratio_to_response"]].to_string(index=False), flush=True)
