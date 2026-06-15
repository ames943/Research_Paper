#!/usr/bin/env python3.13
"""
Batch vs. response signal diagnostics for the n=79 two-cohort CLR matrix.

Produces:
  1. PERMANOVA (Anderson 2001) for 'batch' and 'response' factors using
     Aitchison distance (= Euclidean on CLR-transformed data). Reports
     R², pseudo-F, and empirical p-value (999 permutations each).

  2. PCA scatter: two panels side-by-side, colored by batch and by response.

  3. TSV summary of PERMANOVA results.

Outputs (all in results/ml/batch_diagnostics/):
  permanova_results.tsv
  pca_batch_vs_response.png

Usage:
    python3.13 scripts/batch_diagnostics.py
    python3.13 scripts/batch_diagnostics.py --n-perms 1999 --seed 0
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

# ── CLI ────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--n-perms", type=int, default=999,
                help="Permutations for PERMANOVA p-value (default 999)")
ap.add_argument("--seed",    type=int, default=42)
args = ap.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────────
CLR_PATH    = "results/ml/X_genus_clr.tsv"
LABELS_PATH = "metadata/response_labels.tsv"
OUT_DIR     = "results/ml/batch_diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
clr    = pd.read_csv(CLR_PATH,    sep="\t", index_col="run_accession")
labels = pd.read_csv(LABELS_PATH, sep="\t").set_index("run_accession")

response = labels.reindex(clr.index)["response"]
batch    = pd.Series(
    ["SRR5930 (cohort 1)" if a.startswith("SRR5930") else "SRR11413 (cohort 2)"
     for a in clr.index],
    index=clr.index,
)

n_missing = response.isna().sum()
if n_missing:
    print(f"WARNING: {n_missing} samples missing response label — dropping them")
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


# ── Aitchison distance = Euclidean on CLR ─────────────────────────────────────
print("Computing Aitchison (Euclidean CLR) pairwise distances ...")
X      = clr.values.astype(float)
dist1d = pdist(X, metric="euclidean")
D      = squareform(dist1d)      # (n, n) symmetric distance matrix
print(f"  Distance matrix: {D.shape}  max={D.max():.2f}  mean={D[D>0].mean():.2f}")
print()


# ── PERMANOVA implementation (Anderson 2001) ───────────────────────────────────
def _ss_total(d2: np.ndarray) -> float:
    """SS_T = Σ_{i<j} d²_{ij} / N"""
    n = d2.shape[0]
    return float(np.sum(np.triu(d2, k=1))) / n


def _ss_within(d2: np.ndarray, grp: np.ndarray) -> float:
    """SS_W = Σ_a [ Σ_{i<j both in a} d²_{ij} / n_a ]"""
    sw = 0.0
    for g in np.unique(grp):
        mask = (grp == g)
        n_g  = int(mask.sum())
        if n_g < 2:
            continue
        sub = d2[np.ix_(mask, mask)]
        sw += float(np.sum(np.triu(sub, k=1))) / n_g
    return sw


def permanova(dist_matrix: np.ndarray,
              grouping,
              n_perms: int = 999,
              seed: int = 42) -> dict:
    """
    PERMANOVA (Anderson 2001) for a single grouping factor.

    Returns dict with: F_stat, R2, p_value, SS_total, SS_within, SS_between,
                       n_groups, n_samples, n_perms.
    """
    grp = np.asarray(grouping)
    n   = dist_matrix.shape[0]
    d2  = dist_matrix ** 2
    q   = len(np.unique(grp))

    SS_T = _ss_total(d2)
    SS_W = _ss_within(d2, grp)
    SS_A = SS_T - SS_W
    F    = (SS_A / (q - 1)) / (SS_W / (n - q))
    R2   = SS_A / SS_T

    rng      = np.random.default_rng(seed)
    perm_F   = np.empty(n_perms)
    for i in range(n_perms):
        g_perm   = rng.permutation(grp)
        sw_perm  = _ss_within(d2, g_perm)
        sa_perm  = SS_T - sw_perm
        perm_F[i] = (sa_perm / (q - 1)) / (sw_perm / (n - q))

    p_val = float((perm_F >= F).sum()) / n_perms

    return dict(
        factor       = None,
        n_samples    = n,
        n_groups     = q,
        SS_total     = round(SS_T, 4),
        SS_between   = round(SS_A, 4),
        SS_within    = round(SS_W, 4),
        R2           = round(float(R2), 4),
        F_stat       = round(float(F),  4),
        p_value      = round(p_val, 4),
        n_perms      = n_perms,
        perm_F_mean  = round(float(perm_F.mean()), 4),
        perm_F_std   = round(float(perm_F.std()),  4),
    )


# ── Run PERMANOVA for both factors ─────────────────────────────────────────────
print(f"Running PERMANOVA — factor: batch  (n_perms={args.n_perms}) ...")
res_batch = permanova(D, batch.values,    n_perms=args.n_perms, seed=args.seed)
res_batch["factor"] = "batch"

print(f"Running PERMANOVA — factor: response (n_perms={args.n_perms}) ...")
res_resp  = permanova(D, response.values, n_perms=args.n_perms, seed=args.seed + 1)
res_resp["factor"] = "response"

# ── Print results ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("PERMANOVA results — Aitchison (Euclidean CLR) distance, n=79")
print("=" * 65)
print(f"{'Factor':<12} {'R²':>7} {'F':>8} {'p-value':>9}  {'n_groups':>8}  note")
print("-" * 65)
for res in [res_batch, res_resp]:
    sig = ""
    if   res["p_value"] < 0.001: sig = " ***"
    elif res["p_value"] < 0.01:  sig = " **"
    elif res["p_value"] < 0.05:  sig = " *"
    elif res["p_value"] < 0.10:  sig = " ."
    else:                        sig = " ns"
    print(f"  {res['factor']:<10} {res['R2']:>7.4f} {res['F_stat']:>8.4f} "
          f"{res['p_value']:>9.4f}{sig}   ({res['n_groups']} groups, "
          f"{args.n_perms} perms)")
print()
print("Interpretation:")
for res in [res_batch, res_resp]:
    pct = round(res["R2"] * 100, 1)
    print(f"  {res['factor']:10s}: {pct:.1f}% of Aitchison variance explained  "
          f"(p={res['p_value']:.3f})")


# ── Save PERMANOVA TSV ────────────────────────────────────────────────────────
perm_df = pd.DataFrame([res_batch, res_resp])
col_order = ["factor","n_samples","n_groups","SS_total","SS_between","SS_within",
             "R2","F_stat","p_value","n_perms","perm_F_mean","perm_F_std"]
perm_df[col_order].to_csv(f"{OUT_DIR}/permanova_results.tsv", sep="\t", index=False)
print()
print(f"Saved: {OUT_DIR}/permanova_results.tsv")


# ── PCA ───────────────────────────────────────────────────────────────────────
print("Running PCA ...")
pca   = PCA(n_components=2, random_state=42)
scores = pca.fit_transform(X)
var_exp = pca.explained_variance_ratio_ * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A — colored by batch
batch_labels = batch.values
batch_cats   = sorted(np.unique(batch_labels))
batch_colors = {"SRR5930 (cohort 1)": "#2196F3", "SRR11413 (cohort 2)": "#FF5722"}
ax = axes[0]
for b in batch_cats:
    mask = batch_labels == b
    ax.scatter(scores[mask, 0], scores[mask, 1],
               c=batch_colors.get(b, "gray"),
               label=b, alpha=0.80, s=55, edgecolors="white", linewidths=0.4)
ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontsize=11)
ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)", fontsize=11)
ax.set_title(
    f"PCA by Batch\nPERMANOVA: R²={res_batch['R2']:.3f}, "
    f"p={res_batch['p_value']:.3f}",
    fontsize=11,
)
ax.legend(fontsize=9, framealpha=0.8)
ax.grid(True, alpha=0.25)

# Panel B — colored by response
resp_labels = response.values
resp_cats   = sorted(np.unique(resp_labels))
resp_colors = {"R": "#4CAF50", "NR": "#F44336"}
ax = axes[1]
for r in resp_cats:
    mask = resp_labels == r
    label_str = "Responder (R)" if r == "R" else "Non-Responder (NR)"
    ax.scatter(scores[mask, 0], scores[mask, 1],
               c=resp_colors.get(r, "gray"),
               label=label_str, alpha=0.80, s=55, edgecolors="white", linewidths=0.4)
ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontsize=11)
ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)", fontsize=11)
ax.set_title(
    f"PCA by Response\nPERMANOVA: R²={res_resp['R2']:.3f}, "
    f"p={res_resp['p_value']:.3f}",
    fontsize=11,
)
ax.legend(fontsize=9, framealpha=0.8)
ax.grid(True, alpha=0.25)

fig.suptitle(
    f"Aitchison PCA (n=79, 2 cohorts)\n"
    f"Batch explains {res_batch['R2']*100:.1f}% vs Response {res_resp['R2']*100:.1f}% "
    f"of microbiome variance",
    fontsize=12, y=1.02,
)
fig.tight_layout()

out_pca = f"{OUT_DIR}/pca_batch_vs_response.png"
fig.savefig(out_pca, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_pca}")

print()
print("Done.")
