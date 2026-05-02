# When Policies Disagree

Reproducibility package for *When Policies Disagree: A Cross-Policy Audit of
GDPR Transparency on the Web* (CCS '26).

```
.
├── data/
│   └── dataset.tar.gz          # bundled raw data — extracted on first notebook run
├── notebooks/
│   ├── RQ1.ipynb               # availability, length, readability  (Findings.tex §RQ1)
│   ├── RQ2.ipynb               # GDPR completeness                  (Findings.tex §RQ2)
│   ├── RQ3.ipynb               # cross-policy inconsistency         (Findings.tex §RQ3)
│   └── Evaluation.ipynb        # extractor / verifier evaluation    (Evaluation.tex + Appendix.tex)
└── code/
    ├── prompts/                # unified prompt registry (single source of truth)
    ├── pipeline/               # vendored extractor + verifier
    ├── llm_serving/            # 2× A100 vLLM / Ollama setup
    ├── scripts/                # CLIs: run_extraction, run_verification, train_roberta
    ├── requirements.txt
    ├── requirements-server.txt
    └── .env.example
```

## Quickstart — verify the paper numbers without any GPU

```bash
git clone <repo>
cd When-Policies-Disagree

python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt

jupyter nbconvert --to notebook --execute --inplace notebooks/RQ1.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/RQ2.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/RQ3.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/Evaluation.ipynb
```

Each notebook starts by extracting `data/dataset.tar.gz` (a 103 MB tarball
containing the crawl results, per-policy GDPR coverage, and the cross-policy
findings) into `data/raw/` and then computes everything from those raw files.
Each notebook ends with a sanity-check cell that prints every reproduced
number side-by-side with the value reported in the paper and reports
`PASSED` if every check matches.

## Reproducing the inconsistency-detection pipeline end-to-end

You will need a GPU. We ran on **2× NVIDIA A100 80 GB**.

```bash
# 1) Start a local LLM
bash code/llm_serving/serve_vllm.sh        # default: gemma3:27b across both A100s

# 2) Point the pipeline at it
cp code/.env.example code/.env             # the defaults already target localhost
source code/llm_serving/env_local.sh

# 3) Run the verifier on a list of (FP, TP) pairs
python code/scripts/run_verification.py \
    --pairs       data/sample_pairs.csv \
    --extractions data/extractions/ \
    --out         findings_local.csv
```

`code/llm_serving/README.md` documents the hardware and the alternative
hosted-API path.

## Hardware

The full run reported in the paper used **2× NVIDIA A100 80 GB** (NVLink),
256 GB host RAM, CUDA 12.4, vLLM 0.6.x. Models that exceed that envelope
(`qwen3-vl:235b`, `gpt-oss:120b`, `deepseek-v3.1:671b`) were served on
rented GPU instances from **Vast.ai** behind an OpenAI-compatible endpoint —
the same env vars switch between local and rented serving with no code
changes.

## What is in `data/dataset.tar.gz`

| File (after extraction) | What it is |
|---|---|
| `results.jsonl`                       | One record per first party we attempted to crawl. |
| `results.summary.json`                | Crawl-funnel counters (total / home_ok / policy_found / english / qualified). |
| `policy_curation.json`                | Hand-curated FP and TP blacklists. |
| `manifest.csv`                        | One row per unique policy document (sha256, role, word count). |
| `tp_rediscovery_full.jsonl`           | Per third-party domain: lookup outcome (lets us distinguish "no policy" from "unreachable"). |
| `results_shard*.tp_cache.json`        | Third-party policy cache: text, language, word count, keyed by URL. |
| `canonical_qualifying.json`           | The 3,067 first parties / 1,122 third parties used in RQ2 and RQ3. |
| `findings.csv`                        | All 19,692 cross-policy findings (one row per pattern hit). |
| `website_level_breakdown.json`        | Per (FP-cat, TP-type) verdict counts for the heatmap. |
| `gdpr_disclosed_inc.json`             | Per-GDPR-category counts of disclosed-but-inconsistent pairs. |
| `pair_enrichment.json`                | Per pair: GDPR coverage, FKGL, word counts. |
| `gdpr_coverage_per_policy.json`       | Per-policy union of GDPR transparency categories (input to RQ2). |
| `gdpr_roberta_results.json`           | Trained RoBERTa thresholds + per-class F1. |
| `benchmarks/eval_leaderboard_v3.json` | Per-run extractor metrics on the 100-clause holdout (P/R/F1, strict + adjusted). |
| `benchmarks/eval_perturbation_v3.json`| Per-model verifier metrics on the 100 synthetic perturbation cases (3-class + binary view). |
| `benchmarks/eval_verdict_agreement_v7.json` | Per-model verdicts of three verifiers on real findings (input to the agreement table). |
| `benchmarks/gold_claude_holdout_100_v3.jsonl` | 100-clause holdout with 141 gold PPSes used by the extractor leaderboard. |

## Issues / questions

If a sanity-check cell prints something that does not match what the paper
says, that is the most useful kind of bug report — please attach the cell
output when filing an issue.
