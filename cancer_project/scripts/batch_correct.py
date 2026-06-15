#!/usr/bin/env python3.13
"""
Apply ComBat batch correction to the CLR-transformed genus matrix.

Since X_genus_raw.tsv contains proportions (not integer counts),
ComBat-seq is inappropriate. We apply ComBat (pycombat_norm) — the
continuous-data variant — to the CLR-transformed matrix, which is
the standard approach for compositional microbiome data.

Batch variable  : cohort (SRR5930xxx vs SRR11413xxx)
Biological covar: response label (R/NR), preserved during correction

Inputs:
  results/ml/X_genus_clr.tsv       -- CLR-transformed, 79 x N genera
  metadata/response_labels.tsv      -- run_accession, response

Outputs:
  results/ml/X_genus_combat_clr.tsv -- ComBat-corrected CLR matrix
"""

import os
import numpy as np
import pandas as pd
from inmoose.pycombat import pycombat_norm

CLR_PATH    = "results/ml/X_genus_clr.tsv"
LABELS_PATH = "metadata/response_labels.tsv"
OUT_PATH    = "results/ml/X_genus_combat_clr.tsv"

# ── Load data ──────────────────────────────────────────────────────────────────
clr    = pd.read_csv(CLR_PATH, sep="\t", index_col="run_accession")
labels = pd.read_csv(LABELS_PATH, sep="\t").set_index("run_accession")

print(f"CLR matrix : {clr.shape[0]} samples × {clr.shape[1]} genera")

# ── Assign batch (cohort) labels ───────────────────────────────────────────────
batch = np.array([
    "SRR5930" if acc.startswith("SRR5930") else "SRR11413"
    for acc in clr.index
])
counts = {b: (batch == b).sum() for b in np.unique(batch)}
print(f"Batch sizes: {counts}")

# ── Response labels as biological covariate ────────────────────────────────────
response = labels.reindex(clr.index)["response"]
missing  = response.isna().sum()
if missing:
    raise SystemExit(f"ERROR: {missing} samples have no response label.")

# Encode R=1, NR=0 as a (n_samples, 1) design matrix column
covar_mod = (response == "R").astype(float).values.reshape(1, -1)

# ── Apply ComBat  (pycombat_norm expects features × samples) ──────────────────
data_t = clr.T.values.astype(float)   # shape: (n_features, n_samples)

print("Running ComBat (parametric prior)...")
corrected_t = pycombat_norm(
    counts    = data_t,
    batch     = batch,
    covar_mod = covar_mod,
)

# ── Back to samples × features ────────────────────────────────────────────────
corrected = pd.DataFrame(
    corrected_t.T,
    index   = clr.index,
    columns = clr.columns,
)
corrected.index.name = "run_accession"

os.makedirs("results/ml", exist_ok=True)
corrected.reset_index().to_csv(OUT_PATH, sep="\t", index=False)
print(f"Saved {OUT_PATH}  shape={corrected.shape}")

# ── Sanity check: batch mean shift should be near zero after correction ────────
for b in np.unique(batch):
    mask  = batch == b
    bmean = corrected.values[mask].mean()
    print(f"  Post-correction mean ({b}): {bmean:.4f}  (should be ~equal across batches)")
