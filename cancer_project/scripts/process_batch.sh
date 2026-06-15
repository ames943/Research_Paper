#!/usr/bin/env bash
set -euo pipefail

THREADS=4
DB_DIR="."   # your kraken DB files (hash.k2d, taxo.k2d, opts.k2d) are in the project root

mkdir -p results/trimmed results/fastp results/kraken_reports results/kraken_outputs

for r1 in raw_data/*_1.fastq.gz; do
  base=$(basename "$r1" _1.fastq.gz)
  r2="raw_data/${base}_2.fastq.gz"
  [[ -f "$r2" ]] || { echo "Missing pair for $base, skipping."; continue; }

  report="results/kraken_reports/${base}_report.txt"
  out="results/kraken_outputs/${base}_output.txt"

  if [[ -f "$report" ]]; then
    echo "Already processed $base (found report), skipping."
    continue
  fi

  echo "=== fastp: $base ==="
  fastp \
    -i "$r1" -I "$r2" \
    -o "results/trimmed/${base}_1.trimmed.fastq.gz" \
    -O "results/trimmed/${base}_2.trimmed.fastq.gz" \
    -h "results/fastp/${base}_fastp.html" \
    -j "results/fastp/${base}_fastp.json" \
    --thread "$THREADS"

  echo "=== kraken2: $base ==="
  kraken2 \
    --db "$DB_DIR" \
    --threads "$THREADS" \
    --paired \
    --report "$report" \
    --output "$out" \
    "results/trimmed/${base}_1.trimmed.fastq.gz" \
    "results/trimmed/${base}_2.trimmed.fastq.gz"
done
