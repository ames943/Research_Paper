#!/usr/bin/env bash
# Lee et al. 2022 (PRJEB43119) — download → fastp → Kraken2 → cleanup
#
# Must be run from the project root (cancer_project/) where hash.k2d lives,
# OR invoked as: nohup bash cancer_project/scripts/download_lee2022.sh > cancer_project/logs/lee2022_download.log 2>&1 &
# The script auto-cds to its own parent directory.
#
# Resume-safe: skips any ERR whose Kraken2 report already exists.
# Low-depth samples (< 5M reads) are processed normally but flagged to logs/lee2022_lowdepth.log.

set -uo pipefail

# ---- paths ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"   # cancer_project/
cd "$PROJECT_ROOT"

MANIFEST="metadata/download_manifest_lee2022.tsv"
REPORT_DIR="results/kraken_reports/lee2022"
FASTP_DIR="results/fastp/lee2022"
RAW_DIR="raw_data/lee2022"
TRIM_DIR="results/trimmed/lee2022"
KO_DIR="results/kraken_outputs/lee2022"
LOW_DEPTH_LOG="logs/lee2022_lowdepth.log"
THREADS="$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 8)"
ARIA_CONN=16
LOW_DEPTH_THRESHOLD=5000000

# ---- preflight -----------------------------------------------------------
ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "$(ts)  $1"; }

log "=== Lee 2022 pipeline START  (PROJECT_ROOT=${PROJECT_ROOT}) ==="
log "Threads: ${THREADS} | DB: . (${PROJECT_ROOT}) | Reports: ${REPORT_DIR}"

# Verify Kraken2 DB
if [[ ! -f "hash.k2d" ]]; then
    log "ERROR: hash.k2d not found in ${PROJECT_ROOT}"
    log "Restore the k2_standard_08gb database (hash.k2d, taxo.k2d, opts.k2d) to ${PROJECT_ROOT} before running."
    exit 1
fi

# Verify manifest
if [[ ! -f "$MANIFEST" ]]; then
    log "ERROR: manifest not found at ${MANIFEST}"
    exit 1
fi

mkdir -p "$REPORT_DIR" "$FASTP_DIR" "$RAW_DIR" "$TRIM_DIR" "$KO_DIR" logs

TOTAL=$(awk 'NR>1' "$MANIFEST" | wc -l | tr -d ' ')
log "Manifest: $TOTAL samples to process"

# ---- helpers -------------------------------------------------------------
safe_rm() { for f in "$@"; do [[ -f "$f" ]] && rm -f "$f"; done; }

aria_download() {
    local url="$1" dest="$2" label="$3" run="$4"
    log "  [$run] download $label <- $url"
    aria2c -x"$ARIA_CONN" -s"$ARIA_CONN" \
        --continue=true --max-tries=3 --retry-wait=5 \
        -d "$(dirname "$dest")" -o "$(basename "$dest")" \
        "$url" >/dev/null 2>&1
    return $?
}

# ---- main loop -----------------------------------------------------------
SUCCESS=0; FAILED=0; SKIPPED=0; IDX=0

while IFS=$'\t' read -r run ftp1 ftp2 read_count low_flag; do
    [[ "$run" == "run_accession" ]] && continue
    IDX=$((IDX + 1))
    REPORT="${REPORT_DIR}/${run}_report.txt"

    # resume check
    if [[ -f "$REPORT" ]]; then
        log "[$run] ($IDX/$TOTAL) SKIP — report exists"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    log "[$run] ($IDX/$TOTAL) START"

    # low-depth flag
    if [[ "$low_flag" == "LOW_DEPTH" ]]; then
        log "[$run] WARNING: ${read_count} raw reads < ${LOW_DEPTH_THRESHOLD} — flagged for manual review"
        echo "$(ts)  $run  read_count=${read_count}" >> "$LOW_DEPTH_LOG"
    fi

    R1="${RAW_DIR}/${run}_1.fastq.gz"
    R2="${RAW_DIR}/${run}_2.fastq.gz"
    T1="${TRIM_DIR}/${run}_1.trimmed.fastq.gz"
    T2="${TRIM_DIR}/${run}_2.trimmed.fastq.gz"
    KO="${KO_DIR}/${run}_output.txt"

    # clean stale files from prior interrupted run
    safe_rm "$R1" "$R2" "$T1" "$T2" "$KO"

    # --- download ---
    if ! aria_download "$ftp1" "$R1" "R1" "$run"; then
        log "[$run] FAILED: R1 download error"
        safe_rm "$R1"
        FAILED=$((FAILED + 1)); continue
    fi
    if ! aria_download "$ftp2" "$R2" "R2" "$run"; then
        log "[$run] FAILED: R2 download error"
        safe_rm "$R1" "$R2"
        FAILED=$((FAILED + 1)); continue
    fi
    if ! gzip -t "$R1" 2>/dev/null || ! gzip -t "$R2" 2>/dev/null; then
        log "[$run] FAILED: corrupt gzip"
        safe_rm "$R1" "$R2"
        FAILED=$((FAILED + 1)); continue
    fi
    log "[$run] download OK"

    # --- fastp ---
    log "[$run] fastp START"
    if ! fastp \
        -i "$R1" -I "$R2" \
        -o "$T1" -O "$T2" \
        -h "${FASTP_DIR}/${run}_fastp.html" \
        -j "${FASTP_DIR}/${run}_fastp.json" \
        --thread "$THREADS" \
        2>/dev/null; then
        log "[$run] FAILED: fastp error"
        safe_rm "$R1" "$R2" "$T1" "$T2"
        FAILED=$((FAILED + 1)); continue
    fi
    safe_rm "$R1" "$R2"
    log "[$run] fastp OK — raw FASTQs deleted"

    # --- Kraken2 ---
    log "[$run] kraken2 START"
    if ! kraken2 \
        --db . \
        --threads "$THREADS" \
        --paired \
        --report "$REPORT" \
        --output "$KO" \
        "$T1" "$T2" \
        2>/dev/null; then
        log "[$run] FAILED: kraken2 error"
        safe_rm "$T1" "$T2" "$REPORT" "$KO"
        FAILED=$((FAILED + 1)); continue
    fi
    safe_rm "$T1" "$T2" "$KO"
    log "[$run] DONE — trimmed FASTQs deleted, report saved to ${REPORT}"
    SUCCESS=$((SUCCESS + 1))

done < "$MANIFEST"

log "=== Lee 2022 pipeline COMPLETE ==="
log "Total: ${TOTAL} | Success: ${SUCCESS} | Skipped (already done): ${SKIPPED} | Failed: ${FAILED}"
[[ -f "$LOW_DEPTH_LOG" ]] && log "Low-depth samples flagged in: ${LOW_DEPTH_LOG}"
