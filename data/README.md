# Dataset

`dataset.tar.gz` is a ~19 MB bundle that holds every input the notebooks need to
recompute the paper's numbers. Each notebook extracts it into `raw/` on first
run.

## Contents (after extraction into `data/raw/`)

* `results.jsonl` — one record per first party we attempted to crawl.
* `results.summary.json` — crawl-funnel counters (total / home_ok / policy_found / english / qualified).
* `policy_curation.json` — hand-curated FP and TP blacklists.
* `manifest.csv` — one row per unique policy document (sha256, role, word count).
* `tp_rediscovery_full.jsonl` — per third-party domain: lookup outcome (lets us distinguish "no policy" from "unreachable").
* `results_shard*.tp_cache.json` — third-party policy cache: text, language, word count, keyed by URL.
* `canonical_qualifying.json` — the 3,067 first parties / 1,122 third parties used in RQ2 and RQ3.
* `findings.csv` — all 19,692 cross-policy findings (one row per pattern hit).
* `findings_verified.csv` — curated 12,042-row subset of `findings.csv` filtered to `system_verdict == "inconsistent"`. Eight columns: website pair, first-party statement, third-party statement, system verdict, system justification, GDPR categories, website-policy URL, vendor-policy URL.
* `website_level_breakdown.json` — per (FP-cat, TP-type) verdict counts for the heatmap.
* `gdpr_disclosed_inc.json` — per-GDPR-category counts of disclosed-but-inconsistent pairs.
* `pair_enrichment.json` — per pair: GDPR coverage, FKGL, word counts.
* `gdpr_coverage_per_policy.json` — per-policy union of GDPR transparency categories (input to RQ2).
* `gdpr_roberta_results.json` — trained RoBERTa thresholds + per-class F1.
* `fp_length_readability_g250.json` — pre-computed FP word-count and FKGL bucket stats (groups of 250 by Tranco rank).
* `tp_length_readability.json` — pre-computed TP word-count and FKGL bucket stats (groups of 100 by prevalence rank).
* `wordcount_fkgl_gdpr_v4.json` — GDPR-coverage-vs-length and -readability buckets (used by Findings §RQ2).
* `benchmarks/gold_claude_holdout_100_v3.jsonl` — 100-clause holdout with 141 gold PPSes used by the extractor leaderboard.
* `benchmarks/eval_leaderboard_v3.json` — per-run extractor metrics on the 100-clause holdout (P/R/F1, strict + adjusted).
* `benchmarks/eval_per_field_action_v3.json` — per-field accuracy and per-action P/R/F1 for the 7 paper-reported configurations.
* `benchmarks/eval_perturbation_v3.json` — per-model verifier metrics on the 100 synthetic perturbation cases (3-class + binary view).
* `benchmarks/eval_verdict_agreement_v7.json` — per-model verdicts of three verifiers (`deepseek-v3.1:671b`, `gpt-oss:120b`, `gemma4:31b`) on 100 real findings sampled with seed 42. This is the run the paper's appendix table reports (84.8% all-3 unanimous).
* `benchmarks/perturbation_cases.jsonl` — 100 synthetic clause pairs with ground-truth labels. Re-input for `code/scripts/run_evaluation.py perturbation`.
* `benchmarks/holdout_clauses_100.jsonl` — 100 clauses from real privacy policies. Re-input for `code/scripts/run_evaluation.py leaderboard` (paired with `gold_claude_holdout_100_v3.jsonl`).
* `benchmarks/verdict_agreement_findings.jsonl` — 100 inconsistent findings sampled with seed 42 from `findings.csv`. Re-input for `code/scripts/run_evaluation.py agreement`.
