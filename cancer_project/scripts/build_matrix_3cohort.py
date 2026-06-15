#!/usr/bin/env python3
"""
Build genus-level feature matrix from Kraken2 reports for the 3-cohort n=118 dataset.

Reads all reports in results/kraken_reports/ (SRR5930xxx, SRR11413xxx, SRR6000xxx).
Applies same parsing and CLR transformation as build_matrix.py.
Restricts output to samples present in metadata/response_labels_3cohort.tsv.

Outputs (results/ml/n118_3cohort/):
  X_genus_raw.tsv  — raw percentage feature matrix
  X_genus_clr.tsv  — CLR-transformed feature matrix

Usage:
    cd cancer_project/
    python3 scripts/build_matrix_3cohort.py
"""

import csv
import glob
import math
import os
from pathlib import Path

EXCLUDED_GENERA = {"Homo"}
PSEUDOCOUNT = 1e-6
OUT_DIR = Path("results/ml/n118_3cohort")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the 3-cohort sample set
labels_path = "metadata/response_labels_3cohort.tsv"
known_samples = set()
with open(labels_path) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        known_samples.add(row["run_accession"])

print(f"Known samples from {labels_path}: {len(known_samples)}")

report_files = sorted(glob.glob("results/kraken_reports/*_report.txt"))
if not report_files:
    raise SystemExit("No Kraken reports found in results/kraken_reports/")

samples = {}
all_genera = set()
fixed_names = {}
excluded_hits = set()
skipped = []

for fp in report_files:
    sample = os.path.basename(fp).replace("_report.txt", "")
    if sample not in known_samples:
        skipped.append(sample)
        continue

    genus_data = {}
    with open(fp) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            if parts[3] != "G":
                continue

            pct = float(parts[0].strip())
            stripped = parts[5].strip()
            if not stripped:
                continue

            if stripped.startswith("Candidatus "):
                name = "Candidatus_" + stripped.split()[1]
            elif " (" in stripped:
                name = stripped.split(" (")[0]
            else:
                name = stripped.split()[0]

            old_last = line.strip().split()[-1]
            if old_last != name:
                fixed_names[name] = old_last

            if name in EXCLUDED_GENERA:
                excluded_hits.add(name)
                continue

            genus_data[name] = genus_data.get(name, 0.0) + pct
            all_genera.add(name)

    samples[sample] = genus_data

all_genera = sorted(all_genera)

print(f"\n=== Parser fixes ===")
if fixed_names:
    for c, o in sorted(fixed_names.items()):
        print(f"  FIXED: '{o}' -> '{c}'")
else:
    print("  (none)")

print(f"\n=== Excluded genera ===")
for name in sorted(excluded_hits):
    print(f"  EXCLUDED: {name}")
if not excluded_hits:
    print("  (none found)")

if skipped:
    print(f"\nSkipped {len(skipped)} reports not in 3-cohort label set")

missing = known_samples - set(samples.keys())
if missing:
    print(f"\nWARNING: {len(missing)} labeled samples have no Kraken2 report yet:")
    for s in sorted(missing):
        print(f"  {s}")

print(f"\nSamples with reports : {len(samples)}")
print(f"Genera               : {len(all_genera)}")

# Raw output
raw_out = OUT_DIR / "X_genus_raw.tsv"
with open(raw_out, "w") as out:
    out.write("run_accession\t" + "\t".join(all_genera) + "\n")
    for sample in sorted(samples):
        row = [sample] + [str(samples[sample].get(g, 0.0)) for g in all_genera]
        out.write("\t".join(row) + "\n")
print(f"\nCreated {raw_out}")

# CLR output
clr_out = OUT_DIR / "X_genus_clr.tsv"
with open(clr_out, "w") as out:
    out.write("run_accession\t" + "\t".join(all_genera) + "\n")
    for sample in sorted(samples):
        vals = [samples[sample].get(g, 0.0) for g in all_genera]
        log_vals = [math.log(v + PSEUDOCOUNT) for v in vals]
        mean_log = sum(log_vals) / len(log_vals)
        clr_vals = [v - mean_log for v in log_vals]
        row = [sample] + [f"{v:.6f}" for v in clr_vals]
        out.write("\t".join(row) + "\n")
print(f"Created {clr_out}  (CLR, pseudocount={PSEUDOCOUNT})")

# Cohort breakdown
cohort_map = {}
with open(labels_path) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        cohort_map[row["run_accession"]] = row["cohort"]

from collections import Counter
cohort_counts = Counter(cohort_map[s] for s in samples)
print("\nSamples by cohort:")
for c, n in sorted(cohort_counts.items()):
    print(f"  {c}: {n}")
