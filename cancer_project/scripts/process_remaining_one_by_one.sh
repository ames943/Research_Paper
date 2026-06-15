#!/usr/bin/env bash
set -euo pipefail

# Use all logical CPU threads on this Mac
THREADS="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 10)"
DB_DIR="."

echo "Using THREADS=$THREADS"

# Make sure we're in the correct project folder
for d in raw_data metadata results scripts; do
  if [[ ! -d "$d" ]]; then
    echo "Error: missing folder '$d'."
    echo "Run this script from the project root that contains raw_data, metadata, results, and scripts."
    exit 1
  fi
done

if [[ ! -f metadata/immunotherapy_microbiome_runs.tsv ]]; then
  echo "Error: metadata/immunotherapy_microbiome_runs.tsv not found."
  exit 1
fi

if [[ ! -f remaining_labeled.txt ]]; then
  echo "Error: remaining_labeled.txt not found in project root."
  exit 1
fi

mkdir -p raw_data results/trimmed results/fastp results/kraken_reports results/kraken_outputs metadata/tmp_links

download_and_check() {
  local SRR="$1"
  local URL1="$2"
  local URL2="$3"
  local R1="raw_data/${SRR}_1.fastq.gz"
  local R2="raw_data/${SRR}_2.fastq.gz"

  echo "Downloading FASTQ files for $SRR..."
  wget -c -O "$R1" "$URL1"
  wget -c -O "$R2" "$URL2"

  echo "Testing gzip integrity for $SRR..."
  if gunzip -t "$R1" && gunzip -t "$R2"; then
    return 0
  fi

  echo "Integrity check failed for $SRR. Re-downloading once..."
  rm -f "$R1" "$R2"
  wget -O "$R1" "$URL1"
  wget -O "$R2" "$URL2"

  echo "Re-testing gzip integrity for $SRR..."
  gunzip -t "$R1"
  gunzip -t "$R2"
}

while read -r SRR; do
  [[ -z "$SRR" ]] && continue

  echo "=================================================="
  echo "Processing $SRR"
  echo "=================================================="

  # Skip if already fully processed
  if [[ -f "results/kraken_reports/${SRR}_report.txt" ]]; then
    echo "$SRR already completed — skipping."
    continue
  fi

  # Clean up any stale files from interrupted runs
  rm -f "raw_data/${SRR}_1.fastq.gz" "raw_data/${SRR}_2.fastq.gz"
  rm -f "results/trimmed/${SRR}_1.trimmed.fastq.gz" "results/trimmed/${SRR}_2.trimmed.fastq.gz"
  rm -f "results/fastp/${SRR}_fastp.html" "results/fastp/${SRR}_fastp.json"
  rm -f "results/kraken_outputs/${SRR}_output.txt"

  awk -v srr="$SRR" '$1==srr {print $2}' metadata/immunotherapy_microbiome_runs.tsv \
    | tr ';' '\n' \
    | sed 's|^ftp\.|https://ftp.|' > "metadata/tmp_links/${SRR}.txt"

  if [[ ! -s "metadata/tmp_links/${SRR}.txt" ]]; then
    echo "No download links found for $SRR, skipping."
    continue
  fi

  URL1="$(sed -n '1p' metadata/tmp_links/${SRR}.txt)"
  URL2="$(sed -n '2p' metadata/tmp_links/${SRR}.txt)"

  if [[ -z "$URL1" || -z "$URL2" ]]; then
    echo "Missing one of the paired FASTQ URLs for $SRR, skipping."
    rm -f "metadata/tmp_links/${SRR}.txt"
    continue
  fi

  download_and_check "$SRR" "$URL1" "$URL2"

  R1="raw_data/${SRR}_1.fastq.gz"
  R2="raw_data/${SRR}_2.fastq.gz"

  echo "Running fastp on $SRR with $THREADS threads..."
  fastp \
    -i "$R1" \
    -I "$R2" \
    -o "results/trimmed/${SRR}_1.trimmed.fastq.gz" \
    -O "results/trimmed/${SRR}_2.trimmed.fastq.gz" \
    -h "results/fastp/${SRR}_fastp.html" \
    -j "results/fastp/${SRR}_fastp.json" \
    --thread "$THREADS"

  echo "Running Kraken2 on $SRR with $THREADS threads..."
  kraken2 \
    --db "$DB_DIR" \
    --threads "$THREADS" \
    --paired \
    --report "results/kraken_reports/${SRR}_report.txt" \
    --output "results/kraken_outputs/${SRR}_output.txt" \
    "results/trimmed/${SRR}_1.trimmed.fastq.gz" \
    "results/trimmed/${SRR}_2.trimmed.fastq.gz"

  echo "Cleaning up large FASTQ files for $SRR..."
  rm -f "$R1" "$R2"
  rm -f "results/trimmed/${SRR}_1.trimmed.fastq.gz" "results/trimmed/${SRR}_2.trimmed.fastq.gz"
  rm -f "metadata/tmp_links/${SRR}.txt"

  echo "Finished $SRR"
  echo
done < remaining_labeled.txt

echo "All remaining labeled samples processed."
