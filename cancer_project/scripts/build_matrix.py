import glob
import os
import math

# Genera to exclude before building features.
# "Homo" = residual human host reads (contamination, not microbiome signal).
EXCLUDED_GENERA = {"Homo"}

report_files = sorted(glob.glob("results/kraken_reports/*_report.txt"))

if not report_files:
    raise SystemExit("No Kraken reports found in results/kraken_reports")

samples = {}
all_genera = set()
fixed_names = {}   # maps corrected genus name -> original raw last-token (for the audit log)
excluded_hits = set()

for fp in report_files:
    sample = os.path.basename(fp).replace("_report.txt", "")
    genus_data = {}

    with open(fp) as f:
        for line in f:
            # Kraken2 report columns (tab-delimited):
            #   0: pct  1: reads_clade  2: reads_direct  3: rank  4: taxid  5: name
            # Column 5 has leading spaces for indentation; split() on all whitespace
            # was extracting the last token, which breaks on multi-word names like
            # "Halalkalibacterium (ex Joshi et al. 2022)" -> last token "2022)".
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue

            rank = parts[3]
            if rank != "G":
                continue

            pct = float(parts[0].strip())
            raw_name_field = parts[5]
            stripped = raw_name_field.strip()
            if not stripped:
                continue

            # Kraken2 genus name parsing rules:
            #
            # 1. "Candidatus X"  -> "Candidatus_X"
            #    Provisional genus names always have exactly two words; joining
            #    with underscore keeps them distinct from each other.
            #
            # 2. "Genusnm (ex Author et al. 2022)"  -> "Genusnm"
            #    Some entries carry parenthetical author citations. Strip the
            #    parenthetical so we don't parse "2022)" as the genus name.
            #    The old parser used parts[-1] (whitespace-split), which returned
            #    the last token of the parenthetical instead of the actual genus.
            #
            # 3. Plain single-word genera are returned as-is.
            if stripped.startswith("Candidatus "):
                name = "Candidatus_" + stripped.split()[1]
            elif " (" in stripped:
                name = stripped.split(" (")[0]
            else:
                name = stripped.split()[0]

            # Record changes vs the old parser for the audit log.
            old_last = line.strip().split()[-1]
            if old_last != name:
                fixed_names[name] = old_last

            if name in EXCLUDED_GENERA:
                excluded_hits.add(name)
                continue

            # Accumulate: if the same genus appears multiple times (G + G1 sub-ranks
            # both labeled G in some DB versions), sum abundances.
            genus_data[name] = genus_data.get(name, 0.0) + pct
            all_genera.add(name)

    samples[sample] = genus_data

all_genera = sorted(all_genera)

# ── Audit log ────────────────────────────────────────────────────────────────
print("=== Parser fixes (first-word correction) ===")
if fixed_names:
    for correct, old in sorted(fixed_names.items()):
        print(f"  FIXED: '{old}'  ->  '{correct}'")
else:
    print("  (none)")

print()
print("=== Excluded genera (non-microbial / host contamination) ===")
if excluded_hits:
    for name in sorted(excluded_hits):
        print(f"  EXCLUDED: {name}")
else:
    print("  (none found in these reports)")

print()

# ── Raw output ───────────────────────────────────────────────────────────────
os.makedirs("results/ml", exist_ok=True)

with open("results/ml/X_genus_raw.tsv", "w") as out:
    out.write("run_accession\t" + "\t".join(all_genera) + "\n")
    for sample in sorted(samples):
        row = [sample] + [str(samples[sample].get(g, 0.0)) for g in all_genera]
        out.write("\t".join(row) + "\n")

print(f"Created results/ml/X_genus_raw.tsv")
print(f"  Samples : {len(samples)}")
print(f"  Genera  : {len(all_genera)}")
print()

# ── CLR transformation ───────────────────────────────────────────────────────
# Centered log-ratio: CLR_i = log(x_i + eps) - mean(log(x_j + eps)) for all j.
# Pseudocount eps = 1e-6 replaces structural zeros before log so the
# transformation is defined and preserves the relative ordering of non-zero values.
PSEUDOCOUNT = 1e-6

with open("results/ml/X_genus_clr.tsv", "w") as out:
    out.write("run_accession\t" + "\t".join(all_genera) + "\n")
    for sample in sorted(samples):
        vals = [samples[sample].get(g, 0.0) for g in all_genera]
        log_vals = [math.log(v + PSEUDOCOUNT) for v in vals]
        mean_log = sum(log_vals) / len(log_vals)
        clr_vals = [v - mean_log for v in log_vals]
        row = [sample] + [f"{v:.6f}" for v in clr_vals]
        out.write("\t".join(row) + "\n")

print(f"Created results/ml/X_genus_clr.tsv  (CLR-transformed, pseudocount={PSEUDOCOUNT})")
