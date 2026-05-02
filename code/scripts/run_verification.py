"""Run the four cross-policy patterns + LLM verifier on a list of (FP, TP) pairs.

For each pair listed in the manifest, the pipeline:

  1. Extracts Privacy Practice Statements (PPSes) from both policies.
  2. Builds a knowledge graph for each policy and merges them.
  3. Runs the four cross-policy patterns (Pi1 Modality Contradiction,
     Pi2 Exclusivity Violation, Pi3 Condition Asymmetry, Pi4 Temporal
     Contradiction) on the merged graph.
  4. Asks the LLM verifier (`VERIFIER_PROMPT` in
     `code/prompts/unified_prompts.py`) for a verdict on each candidate
     finding.

Per-pair JSONs are written to `--out` and an aggregated `all_findings.csv`
(same columns as the bundled `data/dataset.tar.gz/findings.csv`) is emitted
alongside.

Manifest CSV columns (same as `data/sample_pairs.csv`):

  pair_id        — unique tag for the pair (e.g. `siteA__vendorB`)
  website_file   — path to first-party policy text (relative to manifest)
  vendor_file    — path to vendor / third-party policy text
  vendor_name    — human-readable vendor name (e.g. "Google LLC")
  service_type   — TP service category (analytics / ads / ...)

Usage
-----
    bash code/llm_serving/serve_vllm.sh                 # local 2x A100 or rented
    source code/llm_serving/env_local.sh                # exports LLM_BASE_URL etc.

    python code/scripts/run_verification.py \\
        --pairs data/sample_pairs.csv \\
        --out   out/findings.csv

The script does not need a separate `--extractions` directory; the pipeline
extracts on the fly. Set `OUTPUT_DIR` (env, see `pipeline/config.py`) to
control where per-pair JSONs land — defaults to `code/pipeline/data/output/`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from pipeline.pipeline import run_batch  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True,
                   help='manifest CSV (see schema in this docstring)')
    p.add_argument('--out', type=Path, default=None,
                   help='per-pair JSONs land in this directory (default: '
                        'code/pipeline/data/output/)')
    args = p.parse_args()

    if not args.pairs.exists():
        print(f'ERROR: manifest {args.pairs} not found.', file=sys.stderr)
        sys.exit(2)

    print(f'verifying findings from {args.pairs}')
    if args.out:
        print(f'  output dir:    {args.out}')

    results = run_batch(manifest_path=args.pairs, output_dir=args.out)
    print(f'\nProcessed {len(results)} pair(s).')
    print(f'Aggregated findings → '
          f'{(args.out or REPO_ROOT / "code/pipeline/data/output") / "all_findings.csv"}')


if __name__ == '__main__':
    main()
