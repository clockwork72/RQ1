# When Policies Disagree

Reproducibility package for *When Policies Disagree: A Cross-Policy Audit of
GDPR Transparency on the Web* (CCS '26).

```
.
├── data/                       # bundled raw data — extracted on first notebook run
│   ├── dataset.tar.gz
│   └── README.md
├── notebooks/                  # one notebook per research question + evaluation + classifier
│   ├── RQ1.ipynb
│   ├── RQ2.ipynb
│   ├── RQ3.ipynb
│   ├── Evaluation.ipynb
│   └── Classifier.ipynb
├── code/
│   ├── pipeline/               # extractor + verifier engine
│   ├── prompts/
│   ├── llm_serving/
│   ├── scripts/
│   └── figures/
├── gdpr_classifier/            # RoBERTa GDPR-coverage classifier (trainer, dataset, results)
└── scraper/                    # crawler that produced data/dataset.tar.gz
```

## [`notebooks/`](notebooks)

Five Jupyter notebooks that reproduce every paper number and figure from
`data/dataset.tar.gz`. Each one extracts the bundle on first run and ends
with a sanity-check cell that prints `PASSED` when every reproduced value
matches the value reported in the paper.

* `RQ1.ipynb` — policy availability, length, and readability for first
  parties and third parties (Findings §RQ1, plus the §4.3 ecosystem-density
  heatmap).
* `RQ2.ipynb` — GDPR transparency-category coverage per policy
  (Findings §RQ2 table and the GDPR vs. length / readability figure).
* `RQ3.ipynb` — cross-policy inconsistencies on the random-2 sample
  (Findings §RQ3, including the combined verdict-and-GDPR figure).
* `Evaluation.ipynb` — the three appendix evaluation tables and their
  per-field / per-action breakdowns. The leaderboard table (Evaluation §4.1)
  was used to **choose the extractor model**; the verifier perturbation
  table (Evaluation §4.2.1) was used to **choose the verifier model**; the
  verdict-agreement table (Evaluation §4.2.2) shows that the architecture
  is robust — swapping the verifier still produces the same verdict on
  most candidates.
* `Classifier.ipynb` — Appendix.tex GDPR classifier benchmark (RoBERTa vs.
  BERT-base vs. Legal-BERT) and the overall-comparison figure.

## [`code/pipeline/`](code/pipeline)

The extractor + verifier engine. Walks each (FP, TP) pair through clause
segmentation, LLM-based PPS extraction, ontology normalization, scope
classification, knowledge-graph construction, four cross-policy pattern
detectors (Π₁ Modality Contradiction, Π₂ Exclusivity Violation,
Π₃ Condition Asymmetry, Π₄ Temporal Contradiction), and an LLM verifier
that produces the final 3-class verdict
(`inconsistent` / `unspecified` / `non_conflict`). See the folder's own
[`README.md`](code/pipeline/README.md) for the full per-step flow and the
file-level map.

## [`scraper/`](scraper)

The crawler that produced `data/dataset.tar.gz`. Walks each Tranco domain
through homepage fetch, policy discovery, robust fallback (SPA hints +
realistic User-Agents + Wayback), policy-text extraction, and third-party
observation via DuckDuckGo Tracker Radar / TrackerDB. Methodology only —
operator scripts (Slurm, sharding, dashboards) are intentionally omitted.
See the folder's own [`README.md`](scraper/README.md) for the per-stage
description and the file map.

## [`gdpr_classifier/`](gdpr_classifier)

Trainer + dataset + per-model evaluation outputs for the 18-category GDPR
transparency classifier described in `Appendix.tex`. Ships the Rahat et al.
(WPES'22) dataset and the result JSONs the notebook reads; the 476 MB
fine-tuned RoBERTa weights are released separately on the GitHub release
page so reviewers can download once and skip training. See the folder's
own [`TRAINING_README.md`](gdpr_classifier/TRAINING_README.md) for
re-training and re-evaluation instructions.

## [`data/`](data)

Manifest of `dataset.tar.gz`: crawl results, per-policy GDPR coverage, the
5,372-pair random-2 sample, the 19,692-row `findings.csv`, the
12,042-row `findings_verified.csv` (curated, inconsistent verdicts only),
and the evaluation benchmark files. See the folder's own
[`README.md`](data/README.md) for the per-file description.
