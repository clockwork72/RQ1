# Pipeline

This folder is the engine that turns two privacy policies (a website's and a
third-party vendor's) into a list of cross-policy inconsistencies.

It is library code. The end-user entry points are the scripts in
`code/scripts/`:

* `run_extraction.py` — extract Privacy Practice Statements (PPSes) from one policy.
* `run_verification.py` — run the four cross-policy patterns + the LLM verifier on a list of (FP, TP) pairs.

Both scripts call into the modules described below.

## What runs in what order

For one (website, vendor) pair, `pipeline.run_pair()` walks these steps:

1. **Segment.** Cut each policy into clauses (`extractor.segment_clauses`).
2. **Extract PPSes.** Send each clause to the LLM and parse its JSON output
   into typed `PPS` records (`extractor.extract_pps_from_policy`). One PPS
   = one atomic statement of the form
   *(actor, action, data_object, purpose, recipient, modality, condition,
   temporality, scope)*.
3. **Normalize.** Map free-text data-types and purposes to canonical
   buckets via the small ontology in `normalizer.py` so cross-policy
   comparison is meaningful (e.g. *"learner data"* and *"student records"*
   end up at the same key).
4. **Classify scope.** Tag each PPS with the product/feature/audience it
   applies to (`scope_classifier.py`). Used so a donation-page clause
   doesn't get compared with the vendor's general data-processing clause.
5. **Build graphs.** Turn each policy's PPSes into a knowledge graph
   (data-type nodes, purpose nodes, statement edges); merge the two graphs
   so aligned data types line up across policies (`graph_neighborhoods.py`,
   helpers in `pipeline.py`).
6. **Run patterns.** Walk the merged graph with four detectors
   (`patterns.run_all_patterns`):

   | ID | Name | What it catches |
   |----|------|-----------------|
   | Π₁ | Modality Contradiction | One side commits *not* to do X; the other side does X. |
   | Π₂ | Exclusivity Violation  | First party says *only for purpose A*; vendor uses the same data for purpose B. |
   | Π₃ | Condition Asymmetry    | One side gates a practice on consent / opt-in; the other does not. |
   | Π₄ | Temporal Contradiction | Stated retention / deletion windows disagree. |

   Each hit is a *candidate* `Inconsistency` with a `pattern_verdict` of
   `inconsistent` or `underspecified`.

7. **Verify with LLM.** Each candidate goes to a bounded LLM call
   (`verifier.verify_candidate`) that re-reads the two clauses in
   isolation and labels the candidate one of:

   * `inconsistent` — the third party's practice contradicts or exceeds
     the first party's commitment.
   * `unspecified` — evidence is ambiguous; we can't tell.
   * `non_conflict` — pattern misfired; the two clauses are compatible.

   Only `inconsistent` rows count toward the paper's findings.

8. **Cache.** Per-pair results are cached on disk
   (`pair_cache.py`) keyed on the SHA-256 of the input policy texts plus
   the pinned versions of every step, so re-running a pair with the same
   inputs is free.

`pipeline.run_batch()` is the same flow over a list of pairs;
`pipeline.run_gdpr_*` variants compute the GDPR-completeness side of the
analysis used in RQ2.

## File map

| File | What's in it |
|------|--------------|
| `schema.py`            | The data classes everything else passes around: `Clause`, `PPS`, `Inconsistency`, plus the `Modality` / `ConditionType` / `TemporalityType` / `Verdict` / `Severity` enums and the 18-entry `GDPR_CATEGORIES` list. |
| `config.py`            | Env-driven settings: model names, API base URLs, pinned prompt versions, paths under `data/`. Reads `code/.env` if present. |
| `extractor.py`         | Clause segmentation (`segment_clauses`) and the LLM-based PPS extractor (`extract_pps_from_policy`). Also holds GDPR-coverage helpers used by RQ2. |
| `normalizer.py`        | Small data-type and purpose ontology with fuzzy + canonical lookup. Logs unknown terms to `data/output/normalization_misses.log` so the ontology can be audited. |
| `scope_classifier.py`  | Per-PPS scope tag (LLM with regex fallback). Default scope is `global`. |
| `graph_neighborhoods.py` | Helpers for grouping PPSes by data-type cluster in the merged graph (used by patterns Π₁–Π₄). |
| `patterns.py`          | The four pattern detectors. `run_all_patterns` is the dispatch entry point. |
| `verifier.py`          | Wraps the LLM verifier (`verify_candidate`, `verify_candidates`). Handles caching, retry, and JSON parsing of the verdict. |
| `pair_cache.py`        | Disk cache for per-pair results, keyed on input-text hash + pinned versions. |
| `pipeline.py`          | Orchestration: `run_pair`, `run_batch`, `run_single_policy_gdpr`, `run_gdpr_pair_comparison`, `run_gdpr_batch`, `run_demo`. |

## Configuration

Everything you'd plausibly want to flip lives in environment variables read
by `config.py`. The most common ones:

* `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` — OpenAI-compatible endpoint
  for the extractor / verifier (a local 2× A100 server or a rented Vast.ai
  4× A100 instance, both speaking the same /v1 protocol).
* `EXTRACTION_BACKEND`, `EXTRACTION_PROMPT_VERSION`,
  `VERIFIER_PROMPT_VERSION` — which prompt strings (in
  `code/prompts/unified_prompts.py`) to use.
* `EXTRACTION_TEMPERATURE`, `EXTRACTION_REFLECTION_ENABLED`,
  `EXTRACTION_REFLECTION_ROUNDS` — extractor tuning.
* `PI8_ALLOW_SAME_GROUP_BYPASS` — pattern-side toggle for cross-pattern Π₁.

Defaults are sensible for the local-server setup documented in
`code/llm_serving/README.md`.

## What this folder does *not* do

* It does not crawl websites or fetch policies — that's `scraper/`.
* It does not start the LLM server — that's `code/llm_serving/`.
* It does not train the GDPR classifier — that's `gdpr_classifier/`.
* It does not produce the figures or tables — that's `notebooks/`.
