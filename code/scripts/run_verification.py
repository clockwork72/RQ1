"""Run the verifier over a CSV of (FP, TP) pairs to surface inconsistencies.

Reads each pair, finds the matching extractions on disk (produced by
run_extraction.py), runs the four cross-policy patterns, then asks the LLM
verifier (VERIFIER_PROMPT in prompts/unified_prompts.py) for a verdict on each
candidate.

Output: a CSV with one row per pattern finding — same columns as the bundled
`data/findings.csv.gz`.

Usage
-----
    python scripts/run_verification.py \\
        --pairs       data/sample_pairs.csv \\
        --extractions data/extractions/ \\
        --out         findings.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.pipeline import run_pairs_pipeline  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True,
                   help='CSV with columns site_etld1, vendor_etld1')
    p.add_argument('--extractions', type=Path, required=True,
                   help='directory of *_extraction.json files (output of run_extraction.py)')
    p.add_argument('--out', type=Path, required=True,
                   help='findings CSV to write')
    args = p.parse_args()

    print(f'verifying findings on pairs from {args.pairs}')
    print(f'  extractions:    {args.extractions}')
    print(f'  output:         {args.out}')

    findings = run_pairs_pipeline(
        pairs_csv=args.pairs,
        extractions_dir=args.extractions,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as f:
        if findings:
            writer = csv.DictWriter(f, fieldnames=list(findings[0].keys()))
            writer.writeheader()
            for row in findings:
                writer.writerow(row)
    print(f'wrote {len(findings):,} findings to {args.out}')


if __name__ == '__main__':
    main()
