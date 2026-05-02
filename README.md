# When Policies Disagree

Reproducibility package for *When Policies Disagree: A Cross-Policy Audit of
GDPR Transparency on the Web* (CCS '26).

This README is a directory. Each top-level folder has its own README with the
detail; below is a one-line pointer to each.

## Notebooks (`notebooks/`)

Five Jupyter notebooks that reproduce the paper's numbers and figures from
`data/dataset.tar.gz`. Each one ends in a sanity-check cell that prints
`PASSED` when every reproduced value matches the paper.

* [`notebooks/RQ1.ipynb`](notebooks/RQ1.ipynb) — Findings §RQ1: policy
  availability, length, and readability for first parties (FPs) and third
  parties (TPs). Generates the FP / TP availability bar charts, the FKGL +
  word-count panels, and the §4.3 ecosystem-density heatmap.
* [`notebooks/RQ2.ipynb`](notebooks/RQ2.ipynb) — Findings §RQ2: GDPR
  transparency-category coverage per policy. Generates the per-category
  coverage table and the GDPR vs. length / readability figure.
* [`notebooks/RQ3.ipynb`](notebooks/RQ3.ipynb) — Findings §RQ3:
  cross-policy inconsistencies on the random-2 sample. Generates the
  combined verdict-and-GDPR figure used in the paper.
* [`notebooks/Evaluation.ipynb`](notebooks/Evaluation.ipynb) —
  Evaluation.tex + Appendix.tex: extractor leaderboard, verifier
  perturbation, verdict agreement, plus per-field accuracy and
  per-action F1.
* [`notebooks/Classifier.ipynb`](notebooks/Classifier.ipynb) —
  Appendix.tex: GDPR classifier benchmark (RoBERTa vs. BERT-base vs.
  Legal-BERT) and the overall-comparison figure.

## Pipeline code (`code/`)

* [`code/pipeline/README.md`](code/pipeline/README.md) — the extractor +
  verifier engine. Walks through the segment → extract → normalize →
  scope → graph → patterns → verify flow, lists every module, and points
  at the env vars in `config.py`.
* [`code/prompts/unified_prompts.py`](code/prompts/unified_prompts.py) —
  single source of truth for every LLM prompt: `EXTRACTION_PROMPT`,
  `EXTRACTION_PROMPT_FEWSHOT`, `REFLECTION_RECOVERY_PROMPT`,
  `REFLECTION_EXHAUSTION_PROMPT`, `VERIFIER_PROMPT`,
  `VERIFIER_PROMPT_LEGACY`.
* [`code/scripts/`](code/scripts) — three CLIs that wrap the pipeline:
  `run_extraction.py` (extract PPSes from one policy), `run_verification.py`
  (run the four cross-policy patterns + verifier on a list of FP/TP pairs),
  `train_roberta.py` (re-train the GDPR classifier; companion to
  `gdpr_classifier/`).
* [`code/llm_serving/README.md`](code/llm_serving/README.md) — how we
  served the local LLMs (2× A100 vLLM / Ollama). Larger models
  (`qwen3-vl:235b`, `gpt-oss:120b`, `deepseek-v3.1:671b`) ran on rented
  Vast.ai instances behind the same OpenAI-compatible endpoint.
* [`code/figures/`](code/figures) — canonical paper-figure scripts called
  by the notebooks. `WPD_FIGURE_DIR` env var redirects output (the
  notebooks point it at `notebooks/figures/`).

## Scraper (`scraper/`)

[`scraper/README.md`](scraper/README.md) — the crawler that produced
`data/dataset.tar.gz`. Methodology only (Tranco filtering →
homepage fetch → policy discovery → robust-fallback → policy extraction →
third-party observation via Tracker Radar / TrackerDB → TP-policy fetch).
Operator scripts (Slurm, sharding, dashboards) are intentionally omitted.

## GDPR classifier (`gdpr_classifier/`)

[`gdpr_classifier/TRAINING_README.md`](gdpr_classifier/TRAINING_README.md) —
trainer + dataset + per-model evaluation outputs for the 18-category
GDPR transparency classifier. The 476 MB fine-tuned RoBERTa weights are
released separately on the GitHub release page.

## Dataset (`data/`)

[`data/README.md`](data/README.md) — manifest of `dataset.tar.gz`: crawl
results, per-policy GDPR coverage, the 5,372-pair random-2 sample, the
19,692-row `findings.csv`, the 12,042-row `findings_verified.csv`
(curated, inconsistent verdicts only), and the evaluation benchmark
files.
