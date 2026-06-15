#!/usr/bin/env bash
set -euo pipefail

THREADS="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 10)"
DB_DIR="."
SAMPLES=(SRR6000900 SRR6000892)

mkdir -p raw_data results/trimmed results/fastp results/kraken_reports results/kraken_outputs

for SRR in "${SAMPLES[@]}"; do
    REPORT="results/kraken_reports/${SRR}_report.txt"
    if [[ -f "$REPORT" ]]; then
        echo "$SRR already done, skipping."
        continue
    fi

    echo "=== [$SRR] prefetch ==="
    prefetch --max-size 50G -O raw_data/ "$SRR"

    echo "=== [$SRR] fasterq-dump ==="
    fasterq-dump \
        --split-files \
        --outdir raw_data/ \
        --temp raw_data/ \
        --threads "$THREADS" \
        "raw_data/${SRR}/${SRR}.sra"

    # fasterq-dump writes uncompressed; gzip them
    echo "=== [$SRR] gzip ==="
    gzip -f "raw_data/${SRR}_1.fastq"
    gzip -f "raw_data/${SRR}_2.fastq"

    echo "=== [$SRR] verify ==="
    gunzip -t "raw_data/${SRR}_1.fastq.gz"
    gunzip -t "raw_data/${SRR}_2.fastq.gz"

    echo "=== [$SRR] fastp ==="
    fastp \
        -i  "raw_data/${SRR}_1.fastq.gz" \
        -I  "raw_data/${SRR}_2.fastq.gz" \
        -o  "results/trimmed/${SRR}_1.trimmed.fastq.gz" \
        -O  "results/trimmed/${SRR}_2.trimmed.fastq.gz" \
        -h  "results/fastp/${SRR}_fastp.html" \
        -j  "results/fastp/${SRR}_fastp.json" \
        --thread "$THREADS"

    echo "=== [$SRR] kraken2 ==="
    kraken2 \
        --db "$DB_DIR" \
        --threads "$THREADS" \
        --paired \
        --report "$REPORT" \
        --output "results/kraken_outputs/${SRR}_output.txt" \
        "results/trimmed/${SRR}_1.trimmed.fastq.gz" \
        "results/trimmed/${SRR}_2.trimmed.fastq.gz"

    echo "=== [$SRR] cleanup ==="
    rm -rf "raw_data/${SRR}/" \
           "raw_data/${SRR}_1.fastq.gz" "raw_data/${SRR}_2.fastq.gz" \
           "results/trimmed/${SRR}_1.trimmed.fastq.gz" "results/trimmed/${SRR}_2.trimmed.fastq.gz" \
           "results/kraken_outputs/${SRR}_output.txt"

    echo "=== [$SRR] DONE ==="
done

echo "All done."
