#!/usr/bin/env python3
"""
Process a download manifest through fastp -> Kraken2, one sample at a time,
deleting raw and trimmed FASTQs after each sample to conserve disk space.

Must be run from the project root (the directory containing hash.k2d).

Usage:
    python3 scripts/process_batch.py \
        --manifest metadata/download_manifest_SRR11413xxx.tsv \
        --log     logs/process_SRR11413xxx.log

Resume-safe: skips any run whose Kraken2 report already exists.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Logger:
    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(log_path, "a", buffering=1)  # line-buffered

    def log(self, srr: str, stage: str, msg: str = ""):
        line = f"{ts()}  {srr}  [{stage}]  {msg}"
        print(line)
        self._fh.write(line + "\n")

    def close(self):
        self._fh.close()


def ftp_to_https(url: str) -> str:
    """Convert bare ftp.sra.ebi.ac.uk/... to https://ftp.sra.ebi.ac.uk/..."""
    if url.startswith("ftp://") or url.startswith("https://"):
        return url
    return "https://" + url


def run(cmd: list, log: Logger, srr: str, stage: str, check: bool = True) -> subprocess.CompletedProcess:
    log.log(srr, stage, "START  " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        log.log(srr, stage, f"FAILED (rc={result.returncode})\n{result.stderr[-2000:]}")
        raise RuntimeError(f"{stage} failed for {srr}")
    log.log(srr, stage, f"OK (rc={result.returncode})")
    return result


def download(url: str, dest: Path, log: Logger, srr: str, connections: int = 16) -> bool:
    """Download with aria2c; return True on success."""
    cmd = [
        "aria2c",
        f"-x{connections}", f"-s{connections}",
        "--continue=true",
        "--max-tries=3",
        "--retry-wait=5",
        "-d", str(dest.parent),
        "-o", dest.name,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.log(srr, "download", f"aria2c FAILED for {url}\n{result.stderr[-1000:]}")
        return False
    return True


def verify_gzip(path: Path) -> bool:
    result = subprocess.run(["gzip", "-t", str(path)], capture_output=True)
    return result.returncode == 0


def safe_remove(*paths):
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# per-sample pipeline
# ---------------------------------------------------------------------------

def process_sample(row: dict, dirs: dict, threads: int, log: Logger) -> bool:
    srr   = row["run_accession"]
    url1  = ftp_to_https(row["fastq_ftp_1"])
    url2  = ftp_to_https(row["fastq_ftp_2"])

    r1    = dirs["raw"]  / f"{srr}_1.fastq.gz"
    r2    = dirs["raw"]  / f"{srr}_2.fastq.gz"
    t1    = dirs["trim"] / f"{srr}_1.trimmed.fastq.gz"
    t2    = dirs["trim"] / f"{srr}_2.trimmed.fastq.gz"
    report = dirs["kraken_reports"] / f"{srr}_report.txt"
    output = dirs["kraken_outputs"] / f"{srr}_output.txt"

    # --- resume check ---
    if report.exists():
        log.log(srr, "skip", "Kraken2 report already exists")
        return True

    # --- clean stale files from any prior interrupted run ---
    safe_remove(r1, r2, t1, t2, output)

    # --- download R1 ---
    log.log(srr, "download", f"R1 <- {url1}")
    ok = download(url1, r1, log, srr)
    if not ok:
        log.log(srr, "download", "R1 download failed, skipping sample")
        safe_remove(r1)
        return False

    # --- download R2 ---
    log.log(srr, "download", f"R2 <- {url2}")
    ok = download(url2, r2, log, srr)
    if not ok:
        log.log(srr, "download", "R2 download failed, skipping sample")
        safe_remove(r1, r2)
        return False

    log.log(srr, "download", "DONE")

    # --- gzip verify ---
    for path, label in [(r1, "R1"), (r2, "R2")]:
        if not verify_gzip(path):
            log.log(srr, "verify", f"{label} corrupt, re-downloading")
            path.unlink(missing_ok=True)
            url = url1 if label == "R1" else url2
            ok = download(url, path, log, srr)
            if not ok or not verify_gzip(path):
                log.log(srr, "verify", f"{label} corrupt after retry, skipping sample")
                safe_remove(r1, r2)
                return False
    log.log(srr, "verify", "gzip OK")

    # --- fastp ---
    try:
        run([
            "fastp",
            "-i",  str(r1), "-I", str(r2),
            "-o",  str(t1), "-O", str(t2),
            "-h",  str(dirs["fastp"] / f"{srr}_fastp.html"),
            "-j",  str(dirs["fastp"] / f"{srr}_fastp.json"),
            "--thread", str(threads),
        ], log, srr, "fastp")
    except RuntimeError:
        safe_remove(r1, r2, t1, t2)
        return False

    # raw FASTQs no longer needed
    safe_remove(r1, r2)

    # --- Kraken2 ---
    try:
        run([
            "kraken2",
            "--db", ".",
            "--threads", str(threads),
            "--paired",
            "--report", str(report),
            "--output", str(output),
            str(t1), str(t2),
        ], log, srr, "kraken2")
    except RuntimeError:
        safe_remove(t1, t2, report, output)
        return False

    # trimmed FASTQs and per-read output no longer needed
    safe_remove(t1, t2, output)

    log.log(srr, "done", "sample complete, files cleaned")
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="metadata/download_manifest_SRR11413xxx.tsv")
    parser.add_argument("--log",      default="logs/process_SRR11413xxx.log")
    parser.add_argument("--threads",  type=int, default=None,
                        help="CPU threads for fastp/kraken2 (default: all logical CPUs)")
    args = parser.parse_args()

    threads = args.threads or os.cpu_count() or 4

    dirs = {
        "raw":            Path("raw_data"),
        "trim":           Path("results/trimmed"),
        "fastp":          Path("results/fastp"),
        "kraken_reports": Path("results/kraken_reports"),
        "kraken_outputs": Path("results/kraken_outputs"),
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    log = Logger(Path(args.log))
    log.log("BATCH", "start", f"manifest={args.manifest}  threads={threads}")

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    total   = len(rows)
    success = 0
    skipped = 0
    failed  = []

    for i, row in enumerate(rows, 1):
        srr = row["run_accession"]
        log.log(srr, "progress", f"{i}/{total}")
        ok = process_sample(row, dirs, threads, log)
        if ok:
            # distinguish new completions from pre-existing skips
            if Path(dirs["kraken_reports"] / f"{srr}_report.txt").exists():
                success += 1
            else:
                skipped += 1
        else:
            failed.append(srr)

    log.log("BATCH", "done",
            f"total={total}  success={success}  failed={len(failed)}  "
            f"failed_ids={','.join(failed) if failed else 'none'}")
    log.close()


if __name__ == "__main__":
    main()
