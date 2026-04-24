# RQ1 reproducibility package

This repository lets you re-run the RQ1 analysis from our CCS 2026 submission
from scratch, on your own machine. It is self-contained: the crawl output we
used is bundled as a compressed archive, and two Jupyter notebooks recompute
everything the paper reports — both tables, the numbers quoted in the prose,
the three RQ1 figures, the (first-party, third-party) pair set used in the
cross-policy analyses, and the extraction manifest fed to the LLM reader.

We also further curated the dataset to drop privacy policies that turned out
to be corrupted (cookie banners, access-denied pages, generic menus, etc.).
The notebooks re-apply this curation step so the numbers you get are the
same numbers the paper quotes.

## Notebooks

- **`notebook/reproduce_findings.ipynb`** — recomputes every number in
  `Findings.tex` (the two tables + the RQ1 prose) and regenerates the three
  RQ1 figures. Organised as a funnel: each cell takes the population the
  previous cell ended with, adds one more condition, and prints how many
  first parties or third parties survived.
- **`notebook/reproduce_pairs.ipynb`** — rebuilds the random-2-per-FP pair
  set that feeds RQ2–RQ3 from scratch (seed = 42), reports the top-10
  third-party concentration, policy-length distributions, and the
  deduplicated extraction manifest handed to the LLM reader.

## How to run

```bash
git clone https://github.com/clockwork72/RQ1.git
cd RQ1
pip install numpy matplotlib jupyter
jupyter notebook notebook/reproduce_findings.ipynb
```

Run all cells top to bottom (*Cell → Run All*). The first cell in each
notebook unpacks `data/dataset.tar.gz` into `data/raw/`; from then on
everything is local. End-to-end runtime is about a minute per notebook once
the archive is extracted.

When you are done with the findings notebook, open
`notebook/reproduce_pairs.ipynb` and run it the same way — it reuses the
already-extracted data.

## What is in the archive

`data/dataset.tar.gz` (96 MB compressed, ~370 MB extracted) contains the
raw outputs of the crawl pipeline:

- `results.jsonl` — one JSON record per first party we attempted to crawl.
  The fields the notebooks rely on are `status`, `site_etld1`, `rank`,
  `main_category`, `policy_is_english`, `first_party_policy_word_count`,
  `home_ok`, and `third_parties`.
- `results_shard{0..9}.tp_cache.json`,
  `results_shard_aug{0..9}.tp_cache.json`, and
  `results_shard_rediscovered.tp_cache.json` — the third-party policy cache,
  keyed by policy URL. Each entry carries the extracted text, a cached word
  count, and the detected language. The `_aug` and `_rediscovered` shards
  hold the LLM-assisted rediscovery pass.
- `policy_curation.json` — the first-party eTLD+1s whose policies failed the
  curation check on HPC (where the raw first-party texts live). The
  third-party side of the curation is re-applied live in the notebook from
  the cached text.
- `tp_rediscovery_full.jsonl` — one record per third-party domain we tried
  to look up, with the outcome. Lets us distinguish "no policy" from "could
  not reach the domain at all" in the TP prevalence figure.
- `results.summary.json` — pipeline status counter; convenient when you
  want the 5,396 "home rendered ok" number without scanning the full JSONL.

Alongside the archive, `data/manifest.csv` lists one row per unique policy
document to be run through the LLM reader, with `sha256_16`, `word_count`,
and role (first-party vs third-party).
