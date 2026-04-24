# RQ1 reproducibility package

This repository lets you re-run the RQ1 analysis from our CCS 2026 submission
from scratch, on your own machine. It is self-contained: the crawl output we
used is bundled as a compressed archive, and two Jupyter notebooks recompute
everything the paper reports — both tables, the numbers quoted in the prose,
the three RQ1 figures, the (first-party, third-party) pair set used in the
cross-policy analyses, and the extraction manifest fed to the LLM reader.

## Notebooks

- **`notebook/reproduce_findings.ipynb`** — recomputes every number in
  `Findings.tex` (the two tables + the RQ1 prose) and regenerates the three
  RQ1 figures. Organised as a funnel: each cell takes the population the
  previous cell ended with, adds one more condition, and prints how many
  first parties or third parties survived. Every line is labelled with the
  paper's claim next to it, so you can scan for agreement.
- **`notebook/reproduce_pairs.ipynb`** — rebuilds the random-2-per-FP pair
  set that feeds RQ2–RQ4 from scratch (seed = 42), verifies it matches the
  canonical pair list byte-for-byte, and then reports aggregate metrics
  (top-10 third-party concentration, policy-length distributions, and the
  deduplicated extraction manifest handed to the LLM reader).

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

When you are finished with the findings notebook, open
`notebook/reproduce_pairs.ipynb` and run it the same way — it reuses the
already-extracted data.

## What is in the archive

`data/dataset.tar.gz` (96 MB compressed, ~370 MB extracted) contains 25 raw
outputs of the crawl pipeline:

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
- `policy_quality_blacklist.json` — hand-curated first-party eTLD+1s and
  third-party URLs we exclude (the "policy" turned out to be a cookie
  banner, a login wall, or boilerplate scraped by mistake).
- `tp_rediscovery_full.jsonl` — one record per third-party domain we tried
  to look up, with the outcome (`home_unreachable`, `fetch_error`, or a
  discovered URL). Lets us distinguish "no policy" from "could not reach the
  domain at all" in the TP prevalence figure.
- `results.summary.json` — pipeline status counter; convenient when you
  want the 5,396 "home rendered ok" number without scanning the full JSONL.

Alongside the archive, three small JSON / CSV files ship uncompressed so the
`reproduce_pairs.ipynb` notebook can cross-check its reproduction without
re-running the HPC pipeline:

- `data/random2_summary.json` — aggregate counters from the canonical pair
  builder (seed = 42).
- `data/random2_pair_ids.json` — the full 5,372-tuple list of
  `(first-party eTLD+1, third-party eTLD+1)` pairs that the canonical
  builder emitted.
- `data/manifest.csv` and `data/manifest.summary.json` — the extraction
  manifest listing one row per unique policy document to be run through the
  LLM reader, with `sha256_16`, `word_count`, and role.

## What each notebook reproduces

`reproduce_findings.ipynb` — 18 code cells:

| Cell | Reproduces in the paper |
|---|---|
| 1 | Extracts `data/dataset.tar.gz` into `data/raw/` |
| 2 | Imports and helpers |
| 3 | Loads `results.jsonl` |
| 4 | Loads the TP policy cache, blacklists, and rediscovery back-patch |
| 5 | Table 2 top block — 16,100 → 7,489 → 5,396 funnel |
| 6 | Table 2 — 2,755 qualified first parties, 2,735 after same-organisation exclusion |
| 7 | Table 2 — 3,408 / 1,354 / 996 third-party rows |
| 8 | Table 2 — ~30,956 (first party, third party) pairs |
| 9 | Table 1 first-party rows — 7,488 / 4,535 / 3,067; 41.0 %; 67.6 % |
| 10 | Table 1 third-party rows — 4,771 / 1,334 / 1,122; 23.5 %; 84.1 % |
| 11 | Table 1 length rows — FP median 3,947 / mean 5,615; TP median 3,396 / mean 4,982 |
| 12 | Scale paragraph — 21.5 M total words + per-FP third-party density |
| 13 | RQ1 availability paragraph spot-checks |
| 14 | Per-CrUX-category qualifying FP counts (top-6 categories) |
| 15 | Figure — FP availability by CrUX category (vertical bars) |
| 16 | Figure — FP availability by Tranco rank |
| 17 | Figure — TP availability by prevalence rank |

`reproduce_pairs.ipynb` — 17 code cells:

| Cell | Reproduces |
|---|---|
| 1–4 | Decompress + imports + dataset + canonical reference load |
| 5 | Which first parties are eligible (3,067 with inline text) |
| 6 | Distribution of qualifying third parties per first party |
| 7 | Deterministic `random.sample` (seed = 42) — emits 5,372 pairs across 2,751 FPs |
| 8 | Verify the reproduced pair set against `data/random2_pair_ids.json` (bit-identical) |
| 9–11 | Top-10 third-party eTLD+1 / entity concentration + top-10 share |
| 12 | Policy-length distributions on the pair dataset |
| 13 | Figure — top-10 third-party concentration |
| 14 | Extraction-manifest row count (2,751 FPs + 404 TP URLs = 3,155) |
| 15 | Text-level deduplication via the `sha256_16` column in `manifest.csv` |
| 16 | Extraction workload size — 18.6 M words across 3,155 documents |
| 17 | Cross-check reproduced FP / TP sets against `manifest.csv` |

Each cell prints its output next to the canonical value it should match,
in the form `Availability 41.0%  [paper 41.0%]` or
`Reproduced pairs emitted: 5,372  [canonical 5,372]`.
