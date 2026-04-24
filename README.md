# ResearchDataset — RQ1 reproducibility package

This repository contains the dataset and a Jupyter notebook that reproduce every
numeric claim in RQ1 of our CCS 2026 submission (`Findings.tex`), including both
tables (`tab:pipeline-attrition`, `tab:rq1`) and the three RQ1 figures.

## Contents

```
.
├── notebook/
│   └── reproduce_findings.ipynb   # one cell per claim; regenerates all figures
├── data/
│   └── dataset.tar.gz             # bundled HPC outputs (extracted by cell 1)
└── README.md
```

When `reproduce_findings.ipynb` runs cell 1, it extracts the tarball into
`data/raw/`. The extracted contents are:

| File | Size | Purpose |
|---|---|---|
| `results.jsonl` | ~60 MB | one JSON row per crawled first-party site |
| `results_shard{0..9}.tp_cache.json` | ~30 MB each | cached third-party policies (text + word count + language) |
| `results_shard_rediscovered.tp_cache.json` | ~13 MB | homepage-rediscovery back-patch |
| `results_shard_aug{0..9}.tp_cache.json` | ~500 KB each | LLM-assisted augmentation pass |
| `policy_quality_blacklist.json` | ~200 KB | hand-curated FP and TP URL blacklists |
| `tp_rediscovery_full.jsonl` | ~14 MB | per-domain rediscovery outcomes (reachability signal) |
| `results.summary.json` | ~4 KB | pipeline status counter (7,489 / 5,396 / 4,535 / …) |

## Quick start

```bash
git clone https://github.com/clockwork72/ResearchDataset.git
cd ResearchDataset
pip install numpy matplotlib jupyter
jupyter notebook notebook/reproduce_findings.ipynb
```

Run all cells top-to-bottom. End-to-end runtime: ~60 seconds after the tarball
extracts (~350 MB of JSON).

## What each cell reproduces

| Cell | Reproduces |
|---|---|
| 1 | Dataset extraction (`data/dataset.tar.gz` → `data/raw/`) |
| 2 | Imports + shared helpers |
| 3 | Load `results.jsonl` (7,488 site rows) |
| 4 | Load TP policy cache + blacklists + rediscovery back-patch |
| 5 | **Table 2** top block — FP funnel (16,100 → 7,489 → 5,396 → 4,535 / 861 / 1,067 / 981 / 44) |
| 6 | **Table 2** — Qualified sites (2,755) and same-organisation pair exclusion (2,735) |
| 7 | **Table 2** — Third parties on qualified sites (3,408 / 1,354 / 996) |
| 8 | **Table 2** — Cross-policy (FP, TP) pairs (~30,956) |
| 9 | **Table 1** first-party rows — 7,488 / 4,535 / 3,067; 41.0 %; 67.6 % |
| 10 | **Table 1** third-party rows — 4,771 / 1,334 / 1,122; 23.5 %; 84.1 % |
| 11 | **Table 1** length rows — FP median 3,947 / mean 5,615; TP median 3,396 / mean 4,982; IQRs |
| 12 | Scale paragraph — 21.5 M total words (17.2 M FP + 4.3 M TP), per-site vendor density |
| 13 | RQ1 availability paragraph spot-checks (unreachable TPs, URL-known share, retention) |
| 14 | Per-CrUX-category qualifying FP counts (top-6: Business, Tech, Entertainment, News, Education, E-commerce) |
| 15 | **Figure** — FP availability by CrUX category (v2, vertical bars) |
| 16 | **Figure** — FP availability by Tranco rank |
| 17 | **Figure** — TP availability by prevalence rank |

Each cell prints its output next to the paper's claim, e.g.
`Availability 41.0%  (paper 41.0%)`.

## Known minor drift from paper numbers

- Cross-policy pair count in cell 8 uses a simplified same-organisation filter
  and produces ~31,000 pairs vs. the paper's canonical 30,956. The canonical
  filter uses a richer entity-group mapping; the delta is < 0.2 %.
- Cell 6's heuristic same-org drop is similarly ~10–15 sites off from the
  canonical 2,755 → 2,735 delta.

Neither drift affects availability or length tables (cells 5, 7, 9, 10, 11).

## License

The code in this notebook is released under the MIT license.
The dataset is derived from public web-crawl outputs; redistribution of the raw
policy text in `results_shard*.tp_cache.json` is subject to the original
publishers' terms.
