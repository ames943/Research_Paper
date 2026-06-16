#!/usr/bin/env bash
# =============================================================================
# HUMAnN3 functional pathway pipeline — Cohort 1 (PRJNA397906, n=39)
# =============================================================================
# Tool:    HUMAnN3 (NOT PICRUSt2 — Cohort 1 is WGS shotgun, not 16S amplicon)
# Output:  results/ml/humann3/pathway_abundance_cohort1.tsv (MetaCyc, CPM)
# Pattern: one sample at a time, delete FASTQs after HUMAnN3 input consumed
# Resume:  re-runnable; checkpoints skip already-completed samples
# Log:     logs/humann3_overnight.log (this file's stdout/stderr)
# =============================================================================

set -eo pipefail

# ── configuration ─────────────────────────────────────────────────────────────
PROJECT_DIR="/Users/ameygarg/cancer_project/cancer_project"
CONDA_BASE="/opt/anaconda3"
HUMANN_ENV="humann3"
DB_DIR="$HOME/humann3_dbs"
OUT_DIR="$PROJECT_DIR/results/ml/humann3"
TMP_DIR="$OUT_DIR/tmp"
CHECKPOINT_DIR="$OUT_DIR/checkpoints"
THREADS=10

mkdir -p "$OUT_DIR" "$TMP_DIR" "$CHECKPOINT_DIR"
cd "$PROJECT_DIR"

# ── logging ───────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
warn() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $*"; }
die()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] FATAL: $*"; exit 1; }

log "================================================================"
log "HUMAnN3 Pipeline — Cohort 1 (PRJNA397906)"
log "Threads: $THREADS | DB: $DB_DIR | Out: $OUT_DIR"
log "================================================================"
log "Disk available: $(df -h . | awk 'NR==2{print $4}') free"

# ── conda setup ───────────────────────────────────────────────────────────────
source "$CONDA_BASE/etc/profile.d/conda.sh" \
    || die "Cannot source conda from $CONDA_BASE"
log "Conda $(conda --version 2>&1)"

# Install humann3 env if not present
if ! conda env list 2>/dev/null | grep -q "^${HUMANN_ENV}[[:space:]]"; then
    log "Creating conda env '${HUMANN_ENV}' via bioconda (5-15 min)..."
    conda create -n "$HUMANN_ENV" -c bioconda -c conda-forge humann -y \
        || die "conda create -n $HUMANN_ENV failed"
    log "Conda env created."
else
    log "Conda env '${HUMANN_ENV}' already exists."
fi

conda activate "$HUMANN_ENV" \
    || die "conda activate $HUMANN_ENV failed"

HUMANN_VERSION=$(humann --version 2>&1 | head -1)
MPHLAN_VERSION=$(metaphlan --version 2>&1 | head -1)
log "humann: $HUMANN_VERSION"
log "metaphlan: $MPHLAN_VERSION"

# ── databases ─────────────────────────────────────────────────────────────────
mkdir -p "$DB_DIR"

if [ ! -d "$DB_DIR/chocophlan" ] || [ -z "$(ls -A "$DB_DIR/chocophlan" 2>/dev/null)" ]; then
    log "Downloading ChocoPhlAn full (~15 GB)..."
    humann_databases --download chocophlan full "$DB_DIR" \
        || die "ChocoPhlAn download failed"
    log "ChocoPhlAn download done."
else
    log "ChocoPhlAn already present ($(du -sh "$DB_DIR/chocophlan" | cut -f1))."
fi

if [ ! -d "$DB_DIR/uniref" ] || [ -z "$(ls -A "$DB_DIR/uniref" 2>/dev/null)" ]; then
    log "Downloading UniRef90 diamond (~5.5 GB)..."
    humann_databases --download uniref uniref90_diamond "$DB_DIR" \
        || die "UniRef90 download failed"
    log "UniRef90 download done."
else
    log "UniRef90 already present ($(du -sh "$DB_DIR/uniref" | cut -f1))."
fi

humann_config --update database_folders nucleotide "$DB_DIR/chocophlan"
humann_config --update database_folders protein    "$DB_DIR/uniref"
log "HUMAnN3 database paths configured."
log "Disk after DB setup: $(df -h . | awk 'NR==2{print $4}') free"

# ── sample list (39 labeled samples only) ─────────────────────────────────────
mapfile -t SAMPLES < <(cut -f1 metadata/response_labels_mycohort.tsv)
TOTAL=${#SAMPLES[@]}
log "Samples: $TOTAL"

ALREADY_DONE=$(ls "$CHECKPOINT_DIR"/*.done 2>/dev/null | wc -l | tr -d ' ')
[ "$ALREADY_DONE" -gt 0 ] && log "Resuming: $ALREADY_DONE/$TOTAL already done, skipping those."

# ── timing state ──────────────────────────────────────────────────────────────
T_PIPELINE_START=$(date +%s)
COMPLETED_TIMES=()

# ── per-sample loop ────────────────────────────────────────────────────────────
for i in "${!SAMPLES[@]}"; do
    SRR="${SAMPLES[$i]}"
    IDX=$((i + 1))

    # ── checkpoint ──────────────────────────────────────────────────────────────
    if [ -f "$CHECKPOINT_DIR/${SRR}.done" ]; then
        log "[$IDX/$TOTAL] $SRR — skip (done)"
        continue
    fi

    log "------------------------------------------------------------"
    log "[$IDX/$TOTAL] $SRR — START"
    T0=$(date +%s)

    SAMPLE_TMP="$TMP_DIR/$SRR"
    mkdir -p "$SAMPLE_TMP"

    # ── ENA FTP URLs (pattern: vol1/fastq/SRR593/00{last_digit}/SRR...) ─────────
    LAST="${SRR: -1}"
    FTP_BASE="ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR593/00${LAST}/${SRR}"
    R1_RAW="$SAMPLE_TMP/${SRR}_1.fastq.gz"
    R2_RAW="$SAMPLE_TMP/${SRR}_2.fastq.gz"
    R1_TRIM="$SAMPLE_TMP/${SRR}_1_trimmed.fastq.gz"
    R2_TRIM="$SAMPLE_TMP/${SRR}_2_trimmed.fastq.gz"
    MERGED="$SAMPLE_TMP/${SRR}_merged.fastq.gz"
    HUMANN_OUT="$SAMPLE_TMP/humann_out"

    # ── download ────────────────────────────────────────────────────────────────
    log "  [1/5] Downloading R1+R2..."
    aria2c \
        --dir="$SAMPLE_TMP" \
        --max-connection-per-server=8 \
        --split=8 \
        --min-split-size=10M \
        --max-concurrent-downloads=2 \
        --quiet=true \
        --log-level=warn \
        "${FTP_BASE}/${SRR}_1.fastq.gz" \
        "${FTP_BASE}/${SRR}_2.fastq.gz" \
        || { warn "Download failed for $SRR — skipping"; rm -rf "$SAMPLE_TMP"; continue; }

    R1_SIZE=$(du -sh "$R1_RAW" 2>/dev/null | cut -f1)
    R2_SIZE=$(du -sh "$R2_RAW" 2>/dev/null | cut -f1)
    log "  [1/5] Done. R1=${R1_SIZE} R2=${R2_SIZE}"

    # ── fastp trim ──────────────────────────────────────────────────────────────
    log "  [2/5] fastp trimming..."
    fastp \
        -i "$R1_RAW" -I "$R2_RAW" \
        -o "$R1_TRIM" -O "$R2_TRIM" \
        --thread 8 \
        --detect_adapter_for_pe \
        --qualified_quality_phred 20 \
        --length_required 50 \
        --json "$SAMPLE_TMP/fastp.json" \
        --html /dev/null \
        2>"$SAMPLE_TMP/fastp.log" \
        || { warn "fastp failed for $SRR — skipping"; rm -rf "$SAMPLE_TMP"; continue; }

    rm -f "$R1_RAW" "$R2_RAW"

    # Extract read count from fastp json
    READS_AFTER=$(python3 -c "
import json, sys
try:
    d = json.load(open('$SAMPLE_TMP/fastp.json'))
    print(d['filtering_result']['passed_filter_reads'])
except: print('?')
" 2>/dev/null)
    log "  [2/5] Done. Reads after trim: $READS_AFTER"

    # ── merge R1+R2 for HUMAnN3 (concatenate; standard WGS paired-end approach) ─
    log "  [3/5] Merging R1+R2..."
    cat "$R1_TRIM" "$R2_TRIM" > "$MERGED" \
        || { warn "Merge failed for $SRR — skipping"; rm -rf "$SAMPLE_TMP"; continue; }
    rm -f "$R1_TRIM" "$R2_TRIM"
    log "  [3/5] Done. Merged: $(du -sh "$MERGED" | cut -f1)"

    # ── HUMAnN3 ─────────────────────────────────────────────────────────────────
    log "  [4/5] Running HUMAnN3 (this takes ~25-40 min)..."
    mkdir -p "$HUMANN_OUT"
    humann \
        --input "$MERGED" \
        --output "$HUMANN_OUT" \
        --output-basename "$SRR" \
        --threads "$THREADS" \
        --nucleotide-database "$DB_DIR/chocophlan" \
        --protein-database    "$DB_DIR/uniref" \
        --remove-temp-output \
        2>"$SAMPLE_TMP/humann.log" \
        || { warn "HUMAnN3 failed for $SRR — see $OUT_DIR/errors/${SRR}_humann.log"; \
             mkdir -p "$OUT_DIR/errors"; \
             cp "$SAMPLE_TMP/humann.log" "$OUT_DIR/errors/${SRR}_humann.log"; \
             rm -rf "$SAMPLE_TMP"; continue; }

    rm -f "$MERGED"
    log "  [4/5] HUMAnN3 done."

    # ── save pathway file ────────────────────────────────────────────────────────
    log "  [5/5] Saving pathway abundance..."
    PATHWAY_SRC="$HUMANN_OUT/${SRR}_pathabundance.tsv"
    if [ -f "$PATHWAY_SRC" ]; then
        cp "$PATHWAY_SRC" "$OUT_DIR/${SRR}_pathabundance.tsv"
        PATHWAYS=$(tail -n +2 "$PATHWAY_SRC" | wc -l | tr -d ' ')
        log "  [5/5] Saved. Pathways detected: $PATHWAYS"
    else
        warn "Pathway file missing for $SRR. HUMAnN3 output:"
        ls "$HUMANN_OUT/" 2>/dev/null
        mkdir -p "$OUT_DIR/errors"
        cp "$SAMPLE_TMP/humann.log" "$OUT_DIR/errors/${SRR}_humann.log" 2>/dev/null
    fi

    # ── cleanup ─────────────────────────────────────────────────────────────────
    rm -rf "$SAMPLE_TMP"

    # ── checkpoint + timing ──────────────────────────────────────────────────────
    T1=$(date +%s)
    ELAPSED=$((T1 - T0))
    COMPLETED_TIMES+=("$ELAPSED")
    touch "$CHECKPOINT_DIR/${SRR}.done"

    N_DONE=${#COMPLETED_TIMES[@]}
    SUM_T=0; for t in "${COMPLETED_TIMES[@]}"; do SUM_T=$((SUM_T + t)); done
    MEAN_T=$((SUM_T / N_DONE))
    N_LEFT=$((TOTAL - IDX))
    ETA_S=$((MEAN_T * N_LEFT))
    ETA_H=$((ETA_S / 3600))
    ETA_M=$(((ETA_S % 3600) / 60))
    TOTAL_ELAPSED=$((T1 - T_PIPELINE_START))
    TOTAL_H=$((TOTAL_ELAPSED / 3600))
    TOTAL_M=$(((TOTAL_ELAPSED % 3600) / 60))

    log "  ✓ $SRR done in $((ELAPSED/60))m$((ELAPSED%60))s"
    log "  Progress: $N_DONE processed (+ $ALREADY_DONE previously done) / $TOTAL total"
    log "  ETA for remaining $N_LEFT samples: ~${ETA_H}h${ETA_M}m"
    log "  Wall time so far: ${TOTAL_H}h${TOTAL_M}m"
    log "  Disk free: $(df -h . | awk 'NR==2{print $4}')"
done

# ── merge all pathway tables ──────────────────────────────────────────────────
log "============================================================"
log "All samples processed. Merging pathway tables..."

N_PATHWAY_FILES=$(ls "$OUT_DIR"/*_pathabundance.tsv 2>/dev/null | wc -l | tr -d ' ')
log "Pathway files found: $N_PATHWAY_FILES"

if [ "$N_PATHWAY_FILES" -gt 0 ]; then
    # Join per-sample tables into one matrix
    humann_join_tables \
        --input "$OUT_DIR" \
        --output "$OUT_DIR/pathway_abundance_cohort1_raw.tsv" \
        --file_name pathabundance \
        && log "Joined: $OUT_DIR/pathway_abundance_cohort1_raw.tsv"

    # Normalize RPK → copies per million (CPM)
    humann_renorm_table \
        --input  "$OUT_DIR/pathway_abundance_cohort1_raw.tsv" \
        --output "$OUT_DIR/pathway_abundance_cohort1.tsv" \
        --units  cpm \
        --update-snames \
        && log "Normalized (CPM): $OUT_DIR/pathway_abundance_cohort1.tsv"

    # Quick stats
    N_PATHWAYS=$(tail -n +2 "$OUT_DIR/pathway_abundance_cohort1.tsv" | wc -l | tr -d ' ')
    N_SAMPLES_OUT=$(head -1 "$OUT_DIR/pathway_abundance_cohort1.tsv" | awk -F'\t' '{print NF-1}')
    log "Final matrix: $N_PATHWAYS pathways × $N_SAMPLES_OUT samples"
else
    warn "No pathway files found — merge skipped. Check logs/humann3_overnight.log."
fi

# ── summary log.md ────────────────────────────────────────────────────────────
T_END=$(date +%s)
WALL=$((T_END - T_PIPELINE_START))
WALL_H=$((WALL / 3600))
WALL_M=$(((WALL % 3600) / 60))
N_DONE_TOTAL=$(ls "$CHECKPOINT_DIR"/*.done 2>/dev/null | wc -l | tr -d ' ')

cat > "$OUT_DIR/picrust2_log.md" << MDEOF
# HUMAnN3 Pipeline Log — Cohort 1

**Note:** PICRUSt2 was not used. Cohort 1 (PRJNA397906) is whole-metagenome
shotgun sequencing (WGS), not 16S amplicon data. PICRUSt2 requires 16S sequences
as input and cannot process WGS reads. HUMAnN3 (v3) is the correct tool for
MetaCyc pathway prediction from shotgun metagenomics and was used instead.

## Run summary

| Field | Value |
|---|---|
| Date | $(date '+%Y-%m-%d') |
| Tool | HUMAnN3 $(humann --version 2>&1 \| head -1) |
| Database: ChocoPhlAn | $DB_DIR/chocophlan |
| Database: UniRef90 | $DB_DIR/uniref (diamond) |
| Threads | $THREADS |
| Samples targeted | $TOTAL |
| Samples completed | $N_DONE_TOTAL |
| Total wall time | ${WALL_H}h ${WALL_M}m |
| Output (raw RPK) | results/ml/humann3/pathway_abundance_cohort1_raw.tsv |
| Output (CPM) | results/ml/humann3/pathway_abundance_cohort1.tsv |

## Per-sample times (seconds)
$(for j in "${!SAMPLES[@]}"; do
    SRR="${SAMPLES[$j]}"
    if [ -f "$CHECKPOINT_DIR/${SRR}.done" ] && [ $j -lt ${#COMPLETED_TIMES[@]} ]; then
        echo "- $SRR: ${COMPLETED_TIMES[$j]}s"
    elif [ -f "$CHECKPOINT_DIR/${SRR}.done" ]; then
        echo "- $SRR: done (prior run)"
    else
        echo "- $SRR: NOT COMPLETED"
    fi
done)

## Errors
$(ls "$OUT_DIR/errors/" 2>/dev/null | sed 's/^/- /' || echo "None")
MDEOF

log "Summary written: $OUT_DIR/picrust2_log.md"
log "================================================================"
log "ALL DONE. $N_DONE_TOTAL/$TOTAL samples complete. Wall time: ${WALL_H}h${WALL_M}m"
log "Output: $OUT_DIR/pathway_abundance_cohort1.tsv"
log "================================================================"
