#!/usr/bin/env python3
"""
Batch vs. response PERMANOVA for the 3-cohort n=118 dataset.

Produces:
  1. PERMANOVA (Aitchison distance, 999 perms) for 'batch' and 'response'.
  2. PCA scatter (2 panels: by batch, by response).
  3. Updated permanova_comparison.tsv appending the 3-cohort rows.

Outputs (results/ml/batch_diagnostics/):
  permanova_3cohort_results.tsv
  permanova_comparison.tsv         (updated with n=118 rows)
  pca_3cohort_batch_vs_response.png

Usage:
    cd cancer_project/
    python3 scripts/batch_diagnostics_3cohort.py
    python3 scripts/batch_diagnostics_3cohort.py --n-perms 1999 --seed 0
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

ap = argparse.ArgumentParser()
ap.add_argument("--n-perms", type=int, default=999)
ap.add_argument("--seed",    type=int, default=42)
args = ap.parse_args()

CLR_PATH    = "results/ml/n118_3cohort/X_genus_clr.tsv"
LABELS_PATH = "metadata/response_labels_3cohort.tsv"
OLD_CMP     = "results/ml/batch_diagnostics/permanova_comparison.tsv"
OUT_DIR     = "results/ml/batch_diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)

clr    = pd.read_csv(CLR_PATH,    sep="\t", index_col="run_accession")
labels = pd.read_csv(LABELS_PATH, sep="\t").set_index("run_accession")

response = labels.reindex(clr.index)["response"]
cohort   = labels.reindex(clr.index)["cohort"]

batch = pd.Series(
    [
        "cohort1 (SRR5930/Frankel)"  if a.startswith("SRR5930")  else
        "cohort2 (SRR11413)"         if a.startswith("SRR11413") else
        "cohort3 (SRR6000/Matson)"
        for a in clr.index
    ],
    index=clr.index,
)

n_missing = response.isna().sum()
if n_missing:
    print(f"WARNING: {n_missing} samples missing response label — dropping")
    keep     = response.notna()
    clr      = clr.loc[keep]
    response = response.loc[keep]
    batch    = batch.loc[keep]

n = len(clr)
print(f"Samples  : {n}")
print(f"Features : {clr.shape[1]} genera (CLR)")
print(f"Batch    : {batch.value_counts().to_dict()}")
print(f"Response : {response.value_counts().to_dict()}")
print()

# Aitchison distance
print("Computing Aitchison pairwise distances ...")
X      = clr.values.astype(float)
dist1d = pdist(X, metric="euclidean")
D      = squareform(dist1d)
print(f"  Distance matrix: {D.shape}  max={D.max():.2f}  mean={D[D>0].mean():.2f}")
print()


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


def permanova(dist_matrix, grouping, n_perms=999, seed=42):
    grp = np.asarray(grouping)
    n   = dist_matrix.shape[0]
    d2  = dist_matrix ** 2
    q   = len(np.unique(grp))

    SS_T = _ss_total(d2)
    SS_W = _ss_within(d2, grp)
    SS_A = SS_T - SS_W
    F    = (SS_A / (q - 1)) / (SS_W / (n - q))
    R2   = SS_A / SS_T

    rng    = np.random.default_rng(seed)
    perm_F = np.empty(n_perms)
    for i in range(n_perms):
        g_perm   = rng.permutation(grp)
        sw_perm  = _ss_within(d2, g_perm)
        sa_perm  = SS_T - sw_perm
        perm_F[i] = (sa_perm / (q - 1)) / (sw_perm / (n - q))

    p_val = float((perm_F >= F).sum()) / n_perms
    return dict(
        factor=None, n_samples=n, n_groups=q,
        SS_total=round(SS_T, 4), SS_between=round(SS_A, 4), SS_within=round(SS_W, 4),
        R2=round(float(R2), 4), F_stat=round(float(F), 4),
        p_value=round(p_val, 4), n_perms=n_perms,
        perm_F_mean=round(float(perm_F.mean()), 4),
        perm_F_std=round(float(perm_F.std()),  4),
    )


print(f"Running PERMANOVA — batch (n_perms={args.n_perms}) ...")
res_batch = permanova(D, batch.values,    n_perms=args.n_perms, seed=args.seed)
res_batch["factor"] = "batch"

print(f"Running PERMANOVA — response (n_perms={args.n_perms}) ...")
res_resp  = permanova(D, response.values, n_perms=args.n_perms, seed=args.seed + 1)
res_resp["factor"] = "response"

print()
print("=" * 70)
print(f"PERMANOVA results — Aitchison distance, n={n}, 3 cohorts")
print("=" * 70)
print(f"{'Factor':<12} {'R²':>7} {'F':>8} {'p-value':>9}  {'n_groups':>8}  note")
print("-" * 70)
for res in [res_batch, res_resp]:
    sig = " ***" if res["p_value"] < 0.001 else \
          " **"  if res["p_value"] < 0.01  else \
          " *"   if res["p_value"] < 0.05  else \
          " ."   if res["p_value"] < 0.10  else " ns"
    print(f"  {res['factor']:<10} {res['R2']:>7.4f} {res['F_stat']:>8.4f} "
          f"{res['p_value']:>9.4f}{sig}   ({res['n_groups']} groups, "
          f"{args.n_perms} perms)")
print()
print("Interpretation:")
for res in [res_batch, res_resp]:
    pct = round(res["R2"] * 100, 1)
    print(f"  {res['factor']:10s}: {pct:.1f}% of Aitchison variance explained "
          f"(p={res['p_value']:.3f})")
if res_batch["R2"] > 0 and res_resp["R2"] > 0:
    ratio = res_batch["R2"] / res_resp["R2"]
    print(f"  Batch effect is {ratio:.1f}x larger than biological signal")

# Save per-run results
perm_df = pd.DataFrame([res_batch, res_resp])
col_order = ["factor","n_samples","n_groups","SS_total","SS_between","SS_within",
             "R2","F_stat","p_value","n_perms","perm_F_mean","perm_F_std"]
out_tsv = f"{OUT_DIR}/permanova_3cohort_results.tsv"
perm_df[col_order].to_csv(out_tsv, sep="\t", index=False)
print(f"\nSaved: {out_tsv}")

# Update permanova_comparison.tsv
new_rows = [
    {
        "analysis":  "pooled_n118",
        "cohort":    "SRR5930 (Frankel) + SRR11413 + SRR6000 (Matson) — 3 cohorts",
        "n":         n,
        "factor":    "response",
        "R2":        res_resp["R2"],
        "F_stat":    res_resp["F_stat"],
        "p_value":   res_resp["p_value"],
        "n_perms":   args.n_perms,
        "note":      "3-cohort pooled; no batch correction",
    },
    {
        "analysis":  "pooled_n118",
        "cohort":    "SRR5930 (Frankel) + SRR11413 + SRR6000 (Matson) — 3 cohorts",
        "n":         n,
        "factor":    "batch",
        "R2":        res_batch["R2"],
        "F_stat":    res_batch["F_stat"],
        "p_value":   res_batch["p_value"],
        "n_perms":   args.n_perms,
        "note":      f"3-cohort batch effect (3 groups); p={res_batch['p_value']:.3f}",
    },
]

fields = ["analysis","cohort","n","factor","R2","F_stat","p_value","n_perms","note"]
try:
    old_df = pd.read_csv(OLD_CMP, sep="\t")
    # Remove any prior n=118 rows so this is idempotent
    old_df = old_df[old_df["analysis"] != "pooled_n118"]
    new_df = pd.concat([old_df, pd.DataFrame(new_rows)], ignore_index=True)
except FileNotFoundError:
    new_df = pd.DataFrame(new_rows)

new_df[fields].to_csv(OLD_CMP, sep="\t", index=False)
print(f"Updated: {OLD_CMP}")

# PCA
print("\nRunning PCA ...")
pca    = PCA(n_components=2, random_state=42)
scores = pca.fit_transform(X)
var_exp = pca.explained_variance_ratio_ * 100

batch_colors = {
    "cohort1 (SRR5930/Frankel)": "#2196F3",
    "cohort2 (SRR11413)":        "#FF5722",
    "cohort3 (SRR6000/Matson)":  "#9C27B0",
}
resp_colors = {"R": "#4CAF50", "NR": "#F44336"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for b in sorted(batch.unique()):
    mask = batch.values == b
    ax.scatter(scores[mask, 0], scores[mask, 1],
               c=batch_colors.get(b, "gray"),
               label=b, alpha=0.80, s=55, edgecolors="white", linewidths=0.4)
ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontsize=11)
ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)", fontsize=11)
ax.set_title(
    f"PCA by Batch\nPERMANOVA: R²={res_batch['R2']:.3f}, p={res_batch['p_value']:.3f}",
    fontsize=11)
ax.legend(fontsize=8, framealpha=0.8)
ax.grid(True, alpha=0.25)

ax = axes[1]
for r in sorted(response.unique()):
    mask = response.values == r
    label_str = "Responder (R)" if r == "R" else "Non-Responder (NR)"
    ax.scatter(scores[mask, 0], scores[mask, 1],
               c=resp_colors.get(r, "gray"),
               label=label_str, alpha=0.80, s=55, edgecolors="white", linewidths=0.4)
ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontsize=11)
ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)", fontsize=11)
ax.set_title(
    f"PCA by Response\nPERMANOVA: R²={res_resp['R2']:.3f}, p={res_resp['p_value']:.3f}",
    fontsize=11)
ax.legend(fontsize=9, framealpha=0.8)
ax.grid(True, alpha=0.25)

fig.suptitle(
    f"Aitchison PCA (n={n}, 3 cohorts)\n"
    f"Batch explains {res_batch['R2']*100:.1f}% vs Response {res_resp['R2']*100:.1f}% "
    f"of microbiome variance",
    fontsize=12, y=1.02)
fig.tight_layout()

out_pca = f"{OUT_DIR}/pca_3cohort_batch_vs_response.png"
fig.savefig(out_pca, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_pca}")
print("\nDone.")
