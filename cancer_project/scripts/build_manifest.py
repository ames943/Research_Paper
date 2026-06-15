#!/usr/bin/env python3
"""
Fetch ENA FASTQ download manifest for a given SRR series prefix.

Usage:
    python3 scripts/build_manifest.py --prefix SRR11413 \
        --labels metadata/response_labels.tsv \
        --out metadata/download_manifest_SRR11413xxx.tsv
"""

import argparse
import csv
import json
import sys
import time
import urllib.request

ENA_API = (
    "https://www.ebi.ac.uk/ena/portal/api/filereport"
    "?result=read_run"
    "&fields=run_accession,fastq_ftp,fastq_bytes,read_count"
    "&format=json"
    "&accession={accession}"
)


def fetch_ena(accession: str, retries: int = 3) -> dict:
    url = ENA_API.format(accession=accession)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
                return data[0] if data else {}
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  WARNING: {accession} failed after {retries} attempts: {e}", file=sys.stderr)
                return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="SRR11413")
    parser.add_argument("--labels", default="metadata/response_labels.tsv")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = f"metadata/download_manifest_{args.prefix}xxx.tsv"

    # Load accessions for this prefix
    runs = []
    with open(args.labels) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            acc = row["run_accession"]
            if acc.startswith(args.prefix):
                runs.append((acc, row["response"]))

    if not runs:
        sys.exit(f"No runs found with prefix {args.prefix} in {args.labels}")

    print(f"Fetching ENA metadata for {len(runs)} runs ({args.prefix}xxx)...")

    rows = []
    for i, (acc, label) in enumerate(runs, 1):
        print(f"  [{i:2d}/{len(runs)}] {acc}", end="\r", flush=True)
        meta = fetch_ena(acc)

        ftp_parts = meta.get("fastq_ftp", "").split(";")
        ftp1 = ftp_parts[0] if len(ftp_parts) > 0 else ""
        ftp2 = ftp_parts[1] if len(ftp_parts) > 1 else ""

        byte_parts = meta.get("fastq_bytes", "").split(";")
        total_bytes = sum(int(b) for b in byte_parts if b.isdigit())

        rows.append({
            "run_accession": acc,
            "response_label": label,
            "fastq_ftp_1": ftp1,
            "fastq_ftp_2": ftp2,
            "fastq_bytes": total_bytes,
            "read_count": meta.get("read_count", ""),
        })
        time.sleep(0.3)

    print()  # clear \r line

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    total_bytes = sum(r["fastq_bytes"] for r in rows if isinstance(r["fastq_bytes"], int))
    total_gb = total_bytes / 1e9
    missing = sum(1 for r in rows if not r["fastq_ftp_1"])

    print(f"Manifest written: {args.out}")
    print(f"Runs: {len(rows)}  |  Missing FTP: {missing}  |  Total download size: {total_gb:.1f} GB")


if __name__ == "__main__":
    main()
