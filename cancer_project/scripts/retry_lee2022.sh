#!/usr/bin/env bash
# Retry pass for Lee 2022 corrupt-gzip failures — uses 1 connection to avoid ENA throttling.
# Resume-safe: skips any ERR whose report already exists.
# Run from git root: nohup bash cancer_project/scripts/retry_lee2022.sh > cancer_project/logs/lee2022_retry.log 2>&1 &

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MANIFEST="metadata/download_manifest_lee2022_retry.tsv"
REPORT_DIR="results/kraken_reports/lee2022"
FASTP_DIR="results/fastp/lee2022"
RAW_DIR="raw_data/lee2022"
TRIM_DIR="results/trimmed/lee2022"
KO_DIR="results/kraken_outputs/lee2022"
THREADS="$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 8)"
ARIA_CONN=1   # 1 connection — avoids ENA throttling that caused corrupt gzip in pass 1

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "$(ts)  $1"; }

log "=== Lee 2022 RETRY START  (ARIA_CONN=${ARIA_CONN}, PROJECT_ROOT=${PROJECT_ROOT}) ==="

if [[ ! -f "hash.k2d" ]]; then
    log "ERROR: hash.k2d not found in ${PROJECT_ROOT}"; exit 1
fi

mkdir -p "$REPORT_DIR" "$FASTP_DIR" "$RAW_DIR" "$TRIM_DIR" "$KO_DIR" logs

TOTAL=$(awk 'NR>1' "$MANIFEST" | wc -l | tr -d ' ')
log "Retry manifest: $TOTAL samples"

safe_rm() { for f in "$@"; do [[ -f "$f" ]] && rm -f "$f"; done; }

SUCCESS=0; FAILED=0; SKIPPED=0; IDX=0

while IFS=$'\t' read -r run ftp1 ftp2 read_count low_flag; do
    [[ "$run" == "run_accession" ]] && continue
    IDX=$((IDX + 1))
    REPORT="${REPORT_DIR}/${run}_report.txt"

    if [[ -f "$REPORT" ]]; then
        log "[$run] ($IDX/$TOTAL) SKIP — already done"
        SKIPPED=$((SKIPPED + 1)); continue
    fi

    log "[$run] ($IDX/$TOTAL) START"

    R1="${RAW_DIR}/${run}_1.fastq.gz"
    R2="${RAW_DIR}/${run}_2.fastq.gz"
    T1="${TRIM_DIR}/${run}_1.trimmed.fastq.gz"
    T2="${TRIM_DIR}/${run}_2.trimmed.fastq.gz"
    KO="${KO_DIR}/${run}_output.txt"

    safe_rm "$R1" "$R2" "$T1" "$T2" "$KO"

    # Download R1
    log "  [$run] download R1"
    if ! aria2c -x"$ARIA_CONN" -s"$ARIA_CONN" --continue=true --max-tries=5 --retry-wait=10 \
            -d "$(dirname "$R1")" -o "$(basename "$R1")" "$ftp1" >/dev/null 2>&1; then
        log "[$run] FAILED: R1 download"; safe_rm "$R1"; FAILED=$((FAILED+1)); continue
    fi

    # Download R2
    log "  [$run] download R2"
    if ! aria2c -x"$ARIA_CONN" -s"$ARIA_CONN" --continue=true --max-tries=5 --retry-wait=10 \
            -d "$(dirname "$R2")" -o "$(basename "$R2")" "$ftp2" >/dev/null 2>&1; then
        log "[$run] FAILED: R2 download"; safe_rm "$R1" "$R2"; FAILED=$((FAILED+1)); continue
    fi

    if ! gzip -t "$R1" 2>/dev/null || ! gzip -t "$R2" 2>/dev/null; then
        log "[$run] FAILED: corrupt gzip after retry"
        safe_rm "$R1" "$R2"; FAILED=$((FAILED+1)); continue
    fi
    log "  [$run] download OK"

    log "  [$run] fastp START"
    if ! fastp -i "$R1" -I "$R2" -o "$T1" -O "$T2" \
            -h "${FASTP_DIR}/${run}_fastp.html" -j "${FASTP_DIR}/${run}_fastp.json" \
            --thread "$THREADS" 2>/dev/null; then
        log "[$run] FAILED: fastp"; safe_rm "$R1" "$R2" "$T1" "$T2"; FAILED=$((FAILED+1)); continue
    fi
    safe_rm "$R1" "$R2"
    log "  [$run] fastp OK"

    log "  [$run] kraken2 START"
    if ! kraken2 --db . --threads "$THREADS" --paired \
            --report "$REPORT" --output "$KO" "$T1" "$T2" 2>/dev/null; then
        log "[$run] FAILED: kraken2"; safe_rm "$T1" "$T2" "$REPORT" "$KO"; FAILED=$((FAILED+1)); continue
    fi
    safe_rm "$T1" "$T2" "$KO"
    log "[$run] DONE — report saved"
    SUCCESS=$((SUCCESS+1))

done < "$MANIFEST"

log "=== Lee 2022 RETRY COMPLETE ==="
log "Total: ${TOTAL} | Success: ${SUCCESS} | Skipped: ${SKIPPED} | Failed: ${FAILED}"
