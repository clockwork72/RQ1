# Scraper — methodology reference

The crawler used to produce `data/dataset.tar.gz`. This directory ships the
core methodology modules so reviewers can read end-to-end how a Tranco rank
becomes a row in `results.jsonl`. Operator scripts (Slurm wrappers, sharding,
catalog ingest, dashboards) are intentionally omitted.

## Pipeline

For each input domain `etld1` from the Tranco list:

1. **Tranco filtering** — `tranco_list.py`. Drop suffixes that aren't real
   user-facing sites (e.g., `cdn-cgi.com`, `googleusercontent.com`,
   ad-network domains).
2. **Homepage fetch** — `crawler.py` calls `crawl4ai` (headless Chromium) to
   render the homepage. Status fields (`home_ok`, `status_code`,
   `home_fetch_mode`) record success/failure.
3. **Policy discovery** — `policy_finder.py` walks the rendered DOM looking
   for `<a>` text + `href` patterns matching common policy phrasings
   (multilingual). Returns the most-likely policy URL.
4. **Robust fallback** — `robust_scraping.py` (`--robust-scrape`) layers a
   second pass when the homepage doesn't render cleanly: SPA hints, rotated
   realistic User-Agents, manual policy-URL overrides for known-hard
   top-Tranco sites, and a Wayback Machine fallback when all direct fetches
   fail.
5. **Policy fetch & extraction** — `text_extract.py` strips boilerplate from
   the policy page (trafilatura → readability-lxml → BeautifulSoup
   fallback chain), keeps the language tag, and records word count.
6. **Third-party observation** — `third_party.py` examines all
   sub-resource requests fired during the crawl and groups them by eTLD+1.
   Each TP eTLD+1 is then matched against:
   - **Tracker Radar** (`tracker_radar.py`) — DuckDuckGo's static index of
     trackers with a curated `categories` field. Source of the heatmap's
     TP-service buckets (Ads, Analytics, CDN, Social, Embed, Tag Mgmt,
     Consent, ID & Pay, High Risk).
   - **TrackerDB** (`trackerdb.py`) — DuckDuckGo's secondary index used as
     a fallback for entries Tracker Radar misses.
7. **TP policy fetch** — same extraction pipeline, but starting from the
   TP organisation page (where Tracker Radar tells us to look).
8. **Output** — one JSON line per input domain in `results.jsonl`, plus a
   sharded TP cache keyed by URL (`results_shard*.tp_cache.json`).

## Layout

```
scraper/
├── README.md                    # this file
├── pyproject.toml
├── requirements.txt
└── privacy_research_dataset/
    ├── crawler.py               # orchestration: per-site state machine
    ├── crawl4ai_client.py       # headless-Chromium driver wrapping crawl4ai
    ├── policy_finder.py         # rendered-DOM policy-link extraction
    ├── text_extract.py          # multi-strategy boilerplate stripper
    ├── third_party.py           # eTLD+1 grouping over sub-resource hosts
    ├── tracker_radar.py         # DuckDuckGo Tracker Radar lookup
    ├── trackerdb.py             # DuckDuckGo TrackerDB lookup (fallback)
    ├── tranco_list.py           # input-list filtering
    ├── robust_scraping.py       # SPA hints + UA pool + Wayback fallback
    ├── openwpm_engine.py        # OpenWPM cache reader (alt-engine path)
    ├── data/
    │   ├── site_policy_overrides.csv    # manual FP policy URL overrides
    │   └── tp_policy_overrides.csv      # manual TP policy URL overrides
    └── utils/
        ├── etld.py              # tldextract wrapper
        ├── logging.py           # warn() helper
        ├── io.py                # JSON/JSONL helpers
        └── asyncio.py           # bounded gather
```

## What's omitted (and why)

These pieces ship with the production scraper but aren't methodology
relevant — they're operator plumbing for our specific HPC environment:

- Slurm orchestration / sharding scripts
- Postgres-backed catalog (`catalog_*.py`)
- HPC control plane (`hpc_*.py`)
- Annotation tooling (`annotate_*.py`, `annotator.py`)
- The CLI wrappers (`cli.py`, `summary.py`)
