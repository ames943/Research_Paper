"""
Join genus CLR feature matrix with immunotherapy response labels.

Inputs:
  results/ml/X_genus_clr.tsv          -- CLR-transformed genus matrix (build_matrix.py)
  metadata/response_labels_mycohort.tsv -- two-column TSV: run_accession, response

Output:
  results/ml/immunotherapy_dataset.tsv  -- feature matrix with response column appended
"""
import os
import pandas as pd

FEATURE_PATH = "results/ml/X_genus_clr.tsv"
LABELS_PATH  = "metadata/response_labels.tsv"
OUTPUT_PATH  = "results/ml/immunotherapy_dataset.tsv"

features = pd.read_csv(FEATURE_PATH, sep="\t")
labels   = pd.read_csv(LABELS_PATH, sep="\t")

merged = features.merge(labels, on="run_accession", how="inner")

n_total  = len(features)
n_merged = len(merged)
n_dropped = n_total - n_merged
if n_dropped:
    dropped = set(features["run_accession"]) - set(merged["run_accession"])
    print(f"WARNING: {n_dropped} sample(s) in feature matrix had no matching label and were dropped: {dropped}")

print(f"Samples after join : {n_merged}")
print(f"Features           : {len(merged.columns) - 2}  (excluding run_accession and response)")
print(f"Class distribution : {merged['response'].value_counts().to_dict()}")

os.makedirs("results/ml", exist_ok=True)
merged.to_csv(OUTPUT_PATH, sep="\t", index=False)
print(f"Saved {OUTPUT_PATH}")
