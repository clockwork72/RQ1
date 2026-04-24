# RQ1 reproducibility package

This repository lets you re-run the RQ1 analysis from our CCS 2026 submission
from scratch, on your own machine, and verify every number in the two tables
and in the RQ1 text of `Findings.tex`.

It is self-contained: the crawl output we used is bundled as a compressed
archive, and the notebook recomputes everything from it — both tables, the
numbers quoted in the prose, and the three RQ1 figures. No HPC access needed.

## How to run it

1. Clone the repo and install the two dependencies the notebook uses.

   ```bash
   git clone https://github.com/clockwork72/RQ1.git
   cd RQ1
   pip install numpy matplotlib jupyter
   ```

2. Open the notebook.

   ```bash
   jupyter notebook notebook/reproduce_findings.ipynb
   ```

3. Run all cells top to bottom (*Cell → Run All* in the Jupyter menu).

   The first cell unpacks `data/dataset.tar.gz` into `data/raw/`. After that,
   each cell loads a piece of the dataset and prints the number it is
   supposed to match, side by side with the value quoted in the paper, so you
   can check them off as you go. The last three cells draw the RQ1 figures.

End-to-end runtime is around a minute on a laptop once the archive is
extracted.

## What is in the archive

`data/dataset.tar.gz` (96 MB compressed, ~370 MB extracted) contains the
25 raw outputs of the crawl pipeline:

- `results.jsonl` — one JSON record per first-party site we attempted
  (7,488 rows after full-crawl; the seven fields we rely on are
  `status`, `site_etld1`, `rank`, `main_category`, `policy_is_english`,
  `first_party_policy_word_count`, and `third_parties`).
- `results_shard{0..9}.tp_cache.json` and
  `results_shard_aug{0..9}.tp_cache.json` and
  `results_shard_rediscovered.tp_cache.json` — the third-party policy cache.
  Keyed by policy URL, each entry gives the extracted text, word count, and
  detected language. The `_aug` and `_rediscovered` shards hold the LLM-assisted
  rediscovery pass.
- `policy_quality_blacklist.json` — hand-curated FP domains and TP URLs
  we exclude (the "policy" turned out to be a cookie banner, a login wall,
  or boilerplate scraped by mistake).
- `tp_rediscovery_full.jsonl` — one record per third-party domain we tried
  to look up, with the outcome (`home_unreachable`, `fetch_error`, or a
  discovered policy URL). Used to tell "we have no policy" apart from "we
  couldn't reach the domain at all" in the TP prevalence figure.
- `results.summary.json` — pipeline status counter; convenient when you
  want the 5,396 "home rendered ok" number without scanning the full JSONL.

## What each notebook cell reproduces

| Cell | Reproduces in the paper |
|---|---|
| 1 | Extracts `data/dataset.tar.gz` into `data/raw/` |
| 2 | Imports and helpers |
| 3 | Loads `results.jsonl` |
| 4 | Loads the TP policy cache, blacklists, and rediscovery back-patch |
| 5 | Table 2 top block — the 16,100 → 7,489 → 5,396 funnel |
| 6 | Table 2 — 2,755 qualified sites, 2,735 after same-organisation exclusion |
| 7 | Table 2 — 3,408 / 1,354 / 996 third-party rows |
| 8 | Table 2 — ~30,956 (first party, third party) pairs |
| 9 | Table 1 first-party rows — 7,488 / 4,535 / 3,067; 41.0 %; 67.6 % |
| 10 | Table 1 third-party rows — 4,771 / 1,334 / 1,122; 23.5 %; 84.1 % |
| 11 | Table 1 length rows — FP median 3,947 / mean 5,615; TP median 3,396 / mean 4,982 |
| 12 | Scale paragraph — 21.5 M total words; per-site vendor density |
| 13 | RQ1 availability paragraph spot-checks |
| 14 | Per-CrUX-category qualifying FP counts (top-6 categories) |
| 15 | Figure: FP availability by CrUX category (vertical bars) |
| 16 | Figure: FP availability by Tranco rank |
| 17 | Figure: TP availability by prevalence rank |

Each cell prints something like `Availability 41.0%  (paper 41.0%)` so the
comparison is in the output itself.

## A note on two small discrepancies

Two of the cross-policy numbers — the pair count in cell 8 (~31,000 vs. the
paper's 30,956) and the same-organisation drop in cell 6 — come out of a
simplified entity-matching heuristic rather than the canonical pair builder
we used when writing the paper. The difference is under 0.2 % and does not
affect the availability or length tables.
