# Dataset

`dataset.tar.gz` is a ~19 MB bundle that holds every input the notebooks need to
recompute the paper's numbers. Each notebook extracts it into `raw/` on first
run.

## Contents (after extraction into `data/raw/`)

| File | What it is |
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
| `fp_length_readability_g250.json`     | Pre-computed FP word-count and FKGL bucket stats (groups of 250 by Tranco rank). |
| `tp_length_readability.json`          | Pre-computed TP word-count and FKGL bucket stats (groups of 100 by prevalence rank). |
| `wordcount_fkgl_gdpr_v4.json`         | GDPR-coverage-vs-length and -readability buckets (used by Findings §RQ2). |
| `benchmarks/gold_claude_holdout_100_v3.jsonl` | 100-clause holdout with 141 gold PPSes used by the extractor leaderboard. |
| `benchmarks/eval_leaderboard_v3.json` | Per-run extractor metrics on the 100-clause holdout (P/R/F1, strict + adjusted). |
| `benchmarks/eval_per_field_action_v3.json` | Per-field accuracy and per-action P/R/F1 for the 7 paper-reported configurations. |
| `benchmarks/eval_perturbation_v3.json`| Per-model verifier metrics on the 100 synthetic perturbation cases (3-class + binary view). |
| `benchmarks/eval_verdict_agreement_v7.json` | Per-model verdicts of three verifiers on real findings (input to the agreement table). |

## Provenance

Every file here was produced by the pipeline in `code/` against the Tranco top
16,100 (snapshot date and seed are recorded in `results.summary.json`). The
extractor and verifier model choices behind each artifact are documented in
`Evaluation.tex` of the paper and reproduced from the raw outputs by
`notebooks/Evaluation.ipynb`.
