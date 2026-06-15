"""
Global feature selection pipeline (for exploration/visualization only).

Pipeline:  prevalence filter  ->  variance filter  ->  univariate top-100
WARNING: step 3 uses all 39 labels, so the resulting X_top100_global.tsv
         is NOT suitable for reporting LOOCV metrics.  Use run_loocv_fs.py
         for leak-free metrics.
"""
import os
import pandas as pd
import numpy as np
from scipy import stats

RAW_PATH     = "results/ml/X_genus_raw.tsv"
DATASET_PATH = "results/ml/immunotherapy_dataset.tsv"
OUTPUT_PATH  = "results/ml/X_top100_global.tsv"

PREVALENCE_MIN_SAMPLES  = 4      # ~10 % of 39
VARIANCE_KEEP_FRACTION  = 0.50   # keep top-50 % by variance
TOP_N                   = 100

# ── Load ────────────────────────────────────────────────────────────────────
clr_df = pd.read_csv(DATASET_PATH, sep="\t")
raw_df = pd.read_csv(RAW_PATH,     sep="\t")

feature_cols = [c for c in clr_df.columns if c not in ("run_accession", "response")]
n_start = len(feature_cols)

# Align raw matrix to the same sample order as clr_df
raw_aligned = (
    raw_df.set_index("run_accession")
          .reindex(clr_df["run_accession"])
          [feature_cols]
)

y_binary = (clr_df["response"] == "R").astype(int).values

# ── Step 1: Prevalence filter ────────────────────────────────────────────────
# A genus is "present" in a sample when its raw Kraken percentage was > 0.
# CLR-transformed zeros are structurally different from real observations
# (they carry only the pseudocount), so prevalence should be assessed on
# the raw matrix.
presence_counts = (raw_aligned > 0).sum(axis=0)
prevalent_cols  = presence_counts[presence_counts >= PREVALENCE_MIN_SAMPLES].index.tolist()
print(f"Step 1 – prevalence filter (>= {PREVALENCE_MIN_SAMPLES} / {len(clr_df)} samples): "
      f"{n_start} -> {len(prevalent_cols)} genera")

# ── Step 2: Variance filter ─────────────────────────────────────────────────
X_prev     = clr_df[prevalent_cols]
variances  = X_prev.var(axis=0)
var_cutoff = variances.quantile(1.0 - VARIANCE_KEEP_FRACTION)
high_var_cols = variances[variances >= var_cutoff].index.tolist()
print(f"Step 2 – variance filter (top {int(VARIANCE_KEEP_FRACTION*100)}%): "
      f"{len(prevalent_cols)} -> {len(high_var_cols)} genera")

# ── Step 3: Univariate filter – point-biserial |r| with response ────────────
X_var = clr_df[high_var_cols]
pb_corrs = {}
for col in high_var_cols:
    r, _ = stats.pointbiserialr(y_binary, X_var[col].values)
    pb_corrs[col] = abs(r)

pb_series   = pd.Series(pb_corrs).sort_values(ascending=False)
top_n_use   = min(TOP_N, len(pb_series))
top100_cols = pb_series.head(top_n_use).index.tolist()
print(f"Step 3 – univariate top-{TOP_N}: {len(high_var_cols)} -> {len(top100_cols)} genera")

print(f"\nTop 20 genera by |point-biserial r| (global, includes all 39 labels):")
print(f"  {'Rank':<5} {'|r|':<8} Genus")
for i, (col, val) in enumerate(pb_series.head(20).items(), 1):
    direction = "R+" if pb_corrs[col] == val and \
        stats.pointbiserialr(y_binary, clr_df[col].values)[0] > 0 else "NR+"
    # recompute sign
    sign_r, _ = stats.pointbiserialr(y_binary, clr_df[col].values)
    direction = "R+" if sign_r > 0 else "NR+"
    print(f"  {i:<5} {val:<8.4f} {col}  ({direction})")

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs("results/ml", exist_ok=True)
out_df = clr_df[["run_accession", "response"] + top100_cols]
out_df.to_csv(OUTPUT_PATH, sep="\t", index=False)
print(f"\nSaved {OUTPUT_PATH}  ({len(out_df)} samples x {len(top100_cols)} features + labels)")
print("WARNING: global selection leaks labels — use run_loocv_fs.py for reported metrics.")
