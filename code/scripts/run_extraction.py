"""Extract Privacy Practice Statements (PPSes) from a privacy-policy text file.

Wraps `pipeline.extractor.extract_pps_from_text`, which speaks the OpenAI
chat-completions protocol and is configured via the same `OLLAMA_PRO_BASE_URL`
+ `OLLAMA_PRO_API_KEY` env vars used by the production pipeline. Point those
at a local server (`source llm_serving/env_local.sh`) or at a hosted endpoint
(`https://ollama.com/v1`).

Output: one JSON file per input policy at `<out>/<policy_id>_extraction.json`,
matching the schema shipped at `data/extractions.tar.gz`.

Usage
-----
    # one policy
    python scripts/run_extraction.py --policy data/sample_policy.txt --out out/

    # batch of policies (one .txt per file in a directory)
    python scripts/run_extraction.py --policy-dir data/raw_policies/ --out out/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.extractor import (  # noqa: E402
    extract_pps_from_text,
    compute_clause_gdpr_coverage,
)


def policy_id(path: Path) -> str:
    return path.stem.replace(' ', '_')


def run_one(text_path: Path, out_dir: Path, source: str) -> dict:
    pid = policy_id(text_path)
    out = out_dir / f'{pid}_extraction.json'
    if out.exists():
        return {'policy_id': pid, 'skipped': True}

    text = text_path.read_text(encoding='utf-8', errors='replace')
    t0 = time.time()
    pps = extract_pps_from_text(text, policy_id=pid)
    cov = compute_clause_gdpr_coverage(text, policy_id=pid)
    elapsed = time.time() - t0

    out.write_text(json.dumps({
        'policy_id': pid,
        'policy_source': source,
        'word_count': len(text.split()),
        'n_statements': len(pps),
        'statements': [s.to_dict() if hasattr(s, 'to_dict') else s for s in pps],
        'gdpr_clause_coverage': cov,
        'elapsed_s': elapsed,
    }, indent=2))
    return {'policy_id': pid, 'n_pps': len(pps), 'elapsed_s': elapsed}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, help='single policy text file')
    p.add_argument('--policy-dir', type=Path, help='directory of .txt files')
    p.add_argument('--out', type=Path, required=True, help='output directory')
    p.add_argument('--source', default='first_party',
                   choices=['first_party', 'third_party'],
                   help='whether the policy is the FP or a TP (controls policy_id prefix)')
    args = p.parse_args()

    if not args.policy and not args.policy_dir:
        p.error('pass --policy or --policy-dir')

    args.out.mkdir(parents=True, exist_ok=True)
    paths = [args.policy] if args.policy else sorted(args.policy_dir.glob('*.txt'))

    print(f'extracting {len(paths)} policies into {args.out}/')
    for i, path in enumerate(paths, 1):
        try:
            r = run_one(path, args.out, args.source)
            print(f'  [{i}/{len(paths)}] {path.name:<40s}  {r}')
        except Exception as e:
            print(f'  [{i}/{len(paths)}] {path.name:<40s}  ERROR  {e}')


if __name__ == '__main__':
    main()
