"""Clause segmentation and LLM-based PPS extraction."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

try:
    import anthropic
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    anthropic = None
try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    OpenAI = None

from .config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_EXTRACTION_MODEL,
    API_CALL_DELAY,
    BUILD_PPS_VERSION,
    CACHE_DIR,
    EXTRACTION_BACKEND,
    EXTRACTION_MAX_TOKENS,
    EXTRACTION_PROMPT_VERSION,
    EXTRACTION_REFLECTION_ENABLED,
    EXTRACTION_REFLECTION_ROUNDS,
    EXTRACTION_TEMPERATURE,
    INITIAL_BACKOFF,
    LLAMACPP_BASE_URL,
    LLAMACPP_MODEL_NAME,
    MAX_CLAUSE_LENGTH,
    MAX_RETRIES,
    MIN_CLAUSE_LENGTH,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    OPENAI_API_KEY,
    OPENAI_EXTRACTION_MODEL,
)
import numpy as np

from .normalizer import DATA_SYNONYMS, normalize_data_type, normalize_purpose
from .schema import Clause, ConditionType, GDPR_CATEGORIES, Modality, PPS, TemporalityType, VALID_ACTIONS

# Reproducibility guard: every downstream reader of the cache + results JSON
# assumes deterministic extraction. If anyone turns the temperature above 0
# via env var, self-identify the run as non-deterministic at module load so
# it shows up in run logs rather than silently corrupting comparisons.
if EXTRACTION_TEMPERATURE > 0:
    import warnings as _w
    _w.warn(
        f"EXTRACTION_TEMPERATURE={EXTRACTION_TEMPERATURE} > 0 — extraction is "
        f"non-deterministic. Cached results and reproducibility metadata are "
        f"tagged, but side-by-side comparisons against other runs will drift.",
        RuntimeWarning, stacklevel=2,
    )

# ---------------------------------------------------------------------------
# GDPR RoBERTa classifier (singleton)
# ---------------------------------------------------------------------------

_GDPR_LABEL2IDX = {label: i for i, label in enumerate(sorted(GDPR_CATEGORIES))}
_GDPR_IDX2LABEL = {i: label for label, i in _GDPR_LABEL2IDX.items()}
_GDPR_NUM_LABELS = len(GDPR_CATEGORIES)
_GDPR_MODEL_DIR = Path(__file__).resolve().parent / "data" / "models" / "gdpr_roberta"
_GDPR_RESULTS_FILE = Path(__file__).resolve().parent / "data" / "benchmarks" / "gdpr_roberta_results.json"


class _GDPRClassifier:
    """Singleton RoBERTa-based GDPR category classifier.

    Loads lazily on first call.  Runs on CPU — 125M params, ~5ms per text.
    Falls back to rule-based inference if model files are missing.

    Thread-safety: ``_load()`` is guarded by ``_LOAD_LOCK`` because the
    pipeline dispatches extraction via a ThreadPoolExecutor and all
    workers call ``_gdpr_classifier.classify()`` concurrently.
    Without the lock, transformers 5.x's lazy-import ``__getattr__``
    races on ``from transformers import AutoTokenizer``: threads that
    arrive mid-initialisation see a partially-populated module and
    raise ``ImportError('cannot import name AutoTokenizer')``. We hit
    this reproducibly in the smoke_v3 run (4/8 workers failing on
    pair 1). The lock serialises the first import; subsequent calls
    short-circuit via ``_available`` before acquiring it.
    """

    # Class-level lock so the singleton's race guard survives re-import.
    import threading as _threading
    _LOAD_LOCK = _threading.Lock()

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._thresholds = None
        self._available = None  # None = not checked yet

    def _load(self) -> bool:
        # Fast path — no lock needed once initialised.
        if self._available is not None:
            return self._available
        with self._LOAD_LOCK:
            # Double-check after acquiring: another thread may have won
            # the race while we were waiting.
            if self._available is not None:
                return self._available
            return self._load_locked()

    def _load_locked(self) -> bool:
        # Pre-warm transformers' lazy loader in this single-threaded
        # context before any submodule import races can happen. A first
        # attempt may still race against a concurrent un-guarded import
        # from elsewhere in the process — retry once after a short
        # back-off if that happens.
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification  # noqa: F401
                import torch  # noqa: F401
                break
            except ImportError as e:
                last_err = e
                import time as _time
                _time.sleep(0.25)
        else:
            raise RuntimeError(
                f"[GDPR] transformers lazy-import still racing after retry: {last_err!r}"
            ) from last_err

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            if not (_GDPR_MODEL_DIR / "model.safetensors").exists():
                raise RuntimeError(
                    f"[GDPR] RoBERTa model not found at {_GDPR_MODEL_DIR}/model.safetensors. "
                    "GDPR classification requires RoBERTa — copy the model weights to this path."
                )

            self._tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            self._model = AutoModelForSequenceClassification.from_pretrained(str(_GDPR_MODEL_DIR))
            self._model.eval()

            # Load per-class thresholds
            self._thresholds = np.full(_GDPR_NUM_LABELS, 0.5)
            if _GDPR_RESULTS_FILE.exists():
                import json as _json
                results = _json.loads(_GDPR_RESULTS_FILE.read_text())
                key = list(results.keys())[0]
                for label, t in results[key].get("per_class_thresholds", {}).items():
                    if label in _GDPR_LABEL2IDX:
                        self._thresholds[_GDPR_LABEL2IDX[label]] = t

            print(f"  [GDPR] RoBERTa classifier loaded ({_GDPR_NUM_LABELS} categories)")
            self._available = True
            return True
        except Exception as e:
            raise RuntimeError(
                f"[GDPR] Failed to load RoBERTa classifier: {e}. "
                "GDPR classification with RoBERTa is mandatory. "
                "Ensure 'transformers' and 'torch' are installed and model weights exist."
            ) from e

    @property
    def available(self) -> bool:
        return self._load()

    def classify(self, text: str) -> list[str]:
        """Classify a single text, returning the set of GDPR categories whose
        sigmoid probability exceeds their class-specific threshold.

        Returns a possibly-empty list of 0 to ~6 categories. Empty results
        mean the classifier is not confident in any category for the input,
        which is a legitimate multi-label signal — the statement contributes
        nothing to coverage rather than being forced into a low-confidence
        bucket.

        Reverted 2026-04-19 from the 2026-04-18 softmax+argmax to sigmoid +
        per-class thresholds, matching the training regime of Rahat et al.
        (ACM WPES'22): the model was trained with BCEWithLogitsLoss +
        problem_type="multi_label_classification" and its threshold-tuned
        per-class cutoffs are carried in gdpr_roberta_results.json. The
        argmax deployment was a distribution shift vs the training regime
        and also mis-modelled the 3.6% of training segments that carry 2+
        true labels (see gdpr_classifier/TRAINING_README.md)."""
        if not self._load():
            return []
        import torch

        encoding = self._tokenizer(
            text[:1000],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**encoding).logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        return [
            _GDPR_IDX2LABEL[i]
            for i in range(_GDPR_NUM_LABELS)
            if probs[i] >= self._thresholds[i]
        ]

    def classify_batch(self, texts: list[str]) -> list[list[str]]:
        """Classify multiple texts at once (more efficient). Same sigmoid +
        per-class-threshold scheme as :meth:`classify`; each returned list
        may contain 0 to ~6 categories."""
        if not self._load():
            return [[] for _ in texts]
        import torch

        encoding = self._tokenizer(
            [t[:1000] for t in texts],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**encoding).logits
            probs = torch.sigmoid(logits).cpu().numpy()

        results: list[list[str]] = []
        for row in probs:
            results.append([
                _GDPR_IDX2LABEL[i]
                for i in range(_GDPR_NUM_LABELS)
                if row[i] >= self._thresholds[i]
            ])
        return results


# Global singleton
_gdpr_classifier = _GDPRClassifier()


# ─────────────────────────────────────────────────────────────────────────
# GDPR completeness analysis
# ─────────────────────────────────────────────────────────────────────────

# GDPR Article 13/14 groups the 18 disclosure requirements into three
# conceptual clusters. Used below so the completeness report can surface
# *which families* a policy is weakest in, not just raw counts.
_GDPR_CATEGORY_FAMILIES = {
    "Data Handling": {
        "Processing Purpose", "Data Categories", "Data Recipients",
        "Source of Data", "Storage Period",
    },
    "Data Subject Rights": {
        "Right to Access", "Right to Erase", "Right to Object",
        "Right to Portability", "Right to Restrict",
        "Withdraw Consent", "Lodge Complaint",
    },
    "Legal / Organisational": {
        "Provision Requirement", "Adequacy Decision", "Profiling",
        "DPO Contact", "Controller Contact", "Safeguards Copy",
    },
}


def compute_gdpr_completeness(statements: "list[PPS]") -> dict:
    """Summarise GDPR Article 13/14 disclosure coverage over a set of PPS.

    Each PPS carries a multi-label ``gdpr_categories`` list with 0 to ~6
    entries produced by the sigmoid + per-class-threshold classifier (see
    ``_GDPRClassifier.classify``). Matches the training regime of Rahat et
    al. (ACM WPES'22), where 96.4% of training segments carry exactly one
    label and 3.6% carry 2–6. Statements whose classifier confidence falls
    below every class threshold legitimately contribute no coverage (empty
    list) rather than being forced into a low-confidence bucket.

    Returned fields:

      - ``covered``         : sorted list of categories hit by ≥1 statement
      - ``missing``         : sorted list of the 18 canonical categories
                               never emitted for this policy
      - ``per_category_count``: histogram of (category, number of statements
                                  tagged with it) — under multi-label a
                                  single statement may contribute to several
                                  buckets
      - ``per_family_coverage``: coverage ratio inside each of the three
                                  Article 13/14 disclosure families
      - ``coverage_pct``    : |covered| / 18 as a percentage
      - ``is_complete``     : True iff every one of the 18 categories is hit
      - ``n_statements``    : how many statements contributed
      - ``n_statements_with_category``: how many statements had at least one
                                         category assigned (the remainder
                                         were classifier-rejected)

    Call once per policy (website + vendor separately) during pipeline
    finalisation. Pure aggregation — no LLM calls.
    """
    total = len(statements)
    per_cat: dict[str, int] = {c: 0 for c in GDPR_CATEGORIES}
    labelled = 0
    for s in statements:
        cats = list(getattr(s, "gdpr_categories", []) or [])
        if cats:
            labelled += 1
        for c in cats:
            if c in per_cat:
                per_cat[c] += 1
    covered = sorted(c for c, n in per_cat.items() if n > 0)
    missing = sorted(c for c, n in per_cat.items() if n == 0)

    per_family: dict[str, dict] = {}
    for family, members in _GDPR_CATEGORY_FAMILIES.items():
        covered_in_family = sorted(m for m in members if per_cat.get(m, 0) > 0)
        missing_in_family = sorted(m for m in members if per_cat.get(m, 0) == 0)
        per_family[family] = {
            "size": len(members),
            "covered": covered_in_family,
            "missing": missing_in_family,
            "coverage_pct": round(100.0 * len(covered_in_family) / len(members), 1),
        }

    return {
        "n_statements": total,
        "n_statements_with_category": labelled,
        "covered": covered,
        "missing": missing,
        "per_category_count": per_cat,
        "per_family_coverage": per_family,
        "coverage_pct": round(100.0 * len(covered) / len(GDPR_CATEGORIES), 1),
        "is_complete": len(missing) == 0,
    }


def compute_clause_gdpr_coverage(policy_text: str, policy_id: str = "policy") -> dict:
    """Disclosure-coverage upper bound over a policy's segmented clauses.

    Companion to :func:`compute_gdpr_completeness`, which is bounded by what
    the LLM extractor produced: any clause that yielded zero PPS (rights
    language, contact info, legal-basis disclosures, glossaries, etc.) is
    silently dropped from the PPS-based coverage, even when RoBERTa would
    have classified it. This function classifies every segmented clause
    directly and reports the coverage that reflects the policy's disclosures
    — not the extraction pipeline's output.

    The gap between this dict and the PPS-based one is the **extraction
    gap**: categories present in the policy text but missing from the PPS.
    Pair with :func:`compute_extraction_gap` for the delta view.

    Returns the same shape as ``compute_gdpr_completeness`` minus
    ``n_statements`` / ``n_statements_with_category`` and plus ``n_clauses``.
    If RoBERTa is unavailable, returns an empty shell with
    ``available=False`` so pipeline code can distinguish "no disclosures"
    from "classifier not loaded".
    """
    shell = {
        "available": False,
        "n_clauses": 0,
        "n_clauses_with_category": 0,
        "covered": [],
        "missing": sorted(GDPR_CATEGORIES),
        "per_category_count": {c: 0 for c in GDPR_CATEGORIES},
        "per_family_coverage": {
            family: {
                "size": len(members),
                "covered": [],
                "missing": sorted(members),
                "coverage_pct": 0.0,
            }
            for family, members in _GDPR_CATEGORY_FAMILIES.items()
        },
        "coverage_pct": 0.0,
        "is_complete": False,
    }
    if not _gdpr_classifier.available:
        return shell

    clauses = segment_clauses(policy_text, policy_id)
    if not clauses:
        shell["available"] = True
        return shell

    clause_texts = [c.text for c in clauses if c.text and c.text.strip()]
    if not clause_texts:
        shell["available"] = True
        return shell

    unique_texts = list(dict.fromkeys(clause_texts))
    labels_by_text = dict(zip(unique_texts, _gdpr_classifier.classify_batch(unique_texts)))

    per_cat: dict[str, int] = {c: 0 for c in GDPR_CATEGORIES}
    labelled = 0
    for txt in clause_texts:
        cats = list(labels_by_text.get(txt, []) or [])
        if cats:
            labelled += 1
        for c in cats:
            if c in per_cat:
                per_cat[c] += 1

    covered = sorted(c for c, n in per_cat.items() if n > 0)
    missing = sorted(c for c, n in per_cat.items() if n == 0)

    per_family: dict[str, dict] = {}
    for family, members in _GDPR_CATEGORY_FAMILIES.items():
        covered_in_family = sorted(m for m in members if per_cat.get(m, 0) > 0)
        missing_in_family = sorted(m for m in members if per_cat.get(m, 0) == 0)
        per_family[family] = {
            "size": len(members),
            "covered": covered_in_family,
            "missing": missing_in_family,
            "coverage_pct": round(100.0 * len(covered_in_family) / len(members), 1),
        }

    return {
        "available": True,
        "n_clauses": len(clause_texts),
        "n_clauses_with_category": labelled,
        "covered": covered,
        "missing": missing,
        "per_category_count": per_cat,
        "per_family_coverage": per_family,
        "coverage_pct": round(100.0 * len(covered) / len(GDPR_CATEGORIES), 1),
        "is_complete": len(missing) == 0,
    }


def compute_extraction_gap(pps_coverage: dict, clause_coverage: dict) -> dict:
    """Delta between clause-level (disclosure) and PPS-level (extraction) coverage.

    Answers: "which GDPR categories does the policy actually disclose that
    our PPS pipeline failed to surface?" The returned categories are the
    quantitative signal for prioritising extractor improvements — a large
    DSR extraction gap, for instance, means the schema/prompts are missing
    rights-language even though the policy contains it.

    Inputs are the dicts produced by :func:`compute_gdpr_completeness` and
    :func:`compute_clause_gdpr_coverage` respectively. A missing
    ``clause_coverage`` (e.g. RoBERTa unavailable) returns zeros so callers
    don't need to special-case it.
    """
    pps_covered    = set(pps_coverage.get("covered", []))
    clause_covered = set(clause_coverage.get("covered", []))

    # Categories in the policy that never reached a PPS — the extraction gap.
    gap_cats = sorted(clause_covered - pps_covered)
    # Categories on PPS but not in clause coverage — usually empty;
    # non-empty would point at RoBERTa inconsistency between the two passes.
    leakage = sorted(pps_covered - clause_covered)

    return {
        "available":            bool(clause_coverage.get("available")),
        "pps_coverage_pct":     float(pps_coverage.get("coverage_pct", 0.0) or 0.0),
        "clause_coverage_pct":  float(clause_coverage.get("coverage_pct", 0.0) or 0.0),
        "delta_pct":            round(
            float(clause_coverage.get("coverage_pct", 0.0) or 0.0)
            - float(pps_coverage.get("coverage_pct", 0.0) or 0.0),
            1,
        ),
        "extraction_gap_categories": gap_cats,
        "pps_only_categories":       leakage,
    }


def compare_gdpr_completeness(website: dict, vendor: dict) -> dict:
    """Diff two per-policy completeness summaries into cross-policy signal.

    The per-side `compute_gdpr_completeness` output is a summary. This diff
    is the *comparative* artifact: it surfaces the asymmetries between a
    website's disclosures and its vendor's — exactly the pattern
    cross-policy analysis exists to detect. A website that promises Right
    to Erase but whose vendor never discloses any DSR category creates a
    disclosure gap: the user's right is promised but the downstream
    processor doesn't recognise it.

    Inputs
    ------
    website, vendor : dicts produced by :func:`compute_gdpr_completeness`.
        Must contain ``covered``, ``missing``, ``per_family_coverage``,
        and ``coverage_pct`` keys (raises KeyError on malformed input).

    Output
    ------
    A dict with:

    - ``only_website_covers``  : categories the website covers and the vendor does not
    - ``only_vendor_covers``   : categories the vendor covers and the website does not
    - ``both_cover``           : categories both sides cover
    - ``both_miss``            : categories neither side covers (joint blind spot)
    - ``coverage_pct_delta``   : website coverage_pct − vendor coverage_pct
    - ``per_family_diff``      : per-family website-vs-vendor coverage + asymmetry label
    - ``asymmetry_flags``      : boolean flags flagging the four patterns a
      cross-policy reviewer cares about most:

        * ``dsr_gap``                    : website covers ≥75 % of DSR, vendor <50 %.
          Website promises rights the vendor's disclosures don't back up.
        * ``vendor_broader_data_handling`` : vendor covers ≥20 pp more of the
          Data Handling family than the website — vendor discloses more
          processing context than the website told users to expect.
        * ``website_strong_legal_vendor_weak`` : website ≥75 % of Legal /
          Organisational, vendor <50 % — governance asymmetry.
        * ``joint_blind_spot_legal``     : both sides cover <50 % of Legal /
          Organisational — neither side discloses adequate governance.

    Pure aggregation, no LLM calls; safe to call from any stage.
    """
    # Strict schema check: every required key must be present. Silent
    # defaults used to turn a corrupt upstream payload into a misleading
    # but structurally valid diff; fail loud instead so a bad input is
    # surfaced at the call site.
    required_keys = ("covered", "missing", "per_family_coverage", "coverage_pct")
    for side_name, side in (("website", website), ("vendor", vendor)):
        missing_keys = [k for k in required_keys if k not in side]
        if missing_keys:
            raise KeyError(
                f"compare_gdpr_completeness: {side_name} side missing required "
                f"keys {missing_keys}; got {sorted(side.keys())}"
            )
    w_covered = set(website["covered"])
    v_covered = set(vendor["covered"])

    only_w = sorted(w_covered - v_covered)
    only_v = sorted(v_covered - w_covered)
    both_cov = sorted(w_covered & v_covered)
    all_cats = set(GDPR_CATEGORIES)
    both_miss = sorted(all_cats - w_covered - v_covered)

    per_family_diff: dict[str, dict] = {}
    w_family = website["per_family_coverage"]
    v_family = vendor["per_family_coverage"]
    for family in _GDPR_CATEGORY_FAMILIES:
        w_pct = float(w_family.get(family, {}).get("coverage_pct", 0.0))
        v_pct = float(v_family.get(family, {}).get("coverage_pct", 0.0))
        delta = round(w_pct - v_pct, 1)
        if delta >= 20.0:
            asymmetry = "website_stronger"
        elif delta <= -20.0:
            asymmetry = "vendor_stronger"
        else:
            asymmetry = "balanced"
        per_family_diff[family] = {
            "website_coverage_pct": w_pct,
            "vendor_coverage_pct": v_pct,
            "delta_pct": delta,
            "asymmetry": asymmetry,
        }

    # Asymmetry flags — the four patterns a cross-policy reviewer cares about.
    dsr = per_family_diff.get("Data Subject Rights", {})
    dh = per_family_diff.get("Data Handling", {})
    legal = per_family_diff.get("Legal / Organisational", {})

    flags = {
        "dsr_gap": bool(
            dsr.get("website_coverage_pct", 0.0) >= 75.0
            and dsr.get("vendor_coverage_pct", 0.0) < 50.0
        ),
        "vendor_broader_data_handling": bool(
            dh.get("vendor_coverage_pct", 0.0) - dh.get("website_coverage_pct", 0.0) >= 20.0
        ),
        "website_strong_legal_vendor_weak": bool(
            legal.get("website_coverage_pct", 0.0) >= 75.0
            and legal.get("vendor_coverage_pct", 0.0) < 50.0
        ),
        "joint_blind_spot_legal": bool(
            legal.get("website_coverage_pct", 0.0) < 50.0
            and legal.get("vendor_coverage_pct", 0.0) < 50.0
        ),
    }

    return {
        "only_website_covers": only_w,
        "only_vendor_covers": only_v,
        "both_cover": both_cov,
        "both_miss": both_miss,
        "coverage_pct_delta": round(
            float(website["coverage_pct"]) - float(vendor["coverage_pct"]),
            1,
        ),
        "per_family_diff": per_family_diff,
        "asymmetry_flags": flags,
        "n_asymmetry_flags": sum(1 for v in flags.values() if v),
    }


# ---------------------------------------------------------------------------
# Pattern → GDPR Article 13/14 attribution
# ---------------------------------------------------------------------------
#
# The compliance pipeline emits two independent outputs (cross-policy
# inconsistencies and GDPR completeness). The attribution layer links them:
# every Π_k finding is mapped to the Article 13/14 categories the pattern
# structurally breaches, and the finding is flagged when the primary
# category is ALSO absent from one side's disclosure (a coverage gap that
# compounds the clause-level inconsistency).
#
# The static primary/secondary sets were chosen from the legal reading of
# each pattern:
#   Π₁ Modality         – a first-party prohibition on share/sell/transfer
#                         contradicted by a vendor practice most directly
#                         breaks Data Recipients and Processing Purpose;
#                         Profiling and Data Categories are secondary.
#   Π₂ Exclusivity      – "only for X purpose" vs a vendor using the same
#                         data for purpose Y is a Processing Purpose breach.
#   Π₃ Condition        – consent gate on the first-party side vs by-default
#                         third-party collection is primarily about Withdraw
#                         Consent; Provision Requirement kicks in when one
#                         side makes data provision conditional.
#   Π₄ Temporal         – divergent retention windows are a Storage Period
#                         breach.

PATTERN_GDPR_ATTRIBUTION: dict[str, dict[str, list[str] | str]] = {
    "Π₁": {
        "primary":   ["Data Recipients", "Processing Purpose"],
        "secondary": ["Profiling", "Data Categories"],
        "rationale": (
            "A first-party prohibition or limit that is broken by a third-party "
            "practice most directly breaches the Data Recipients disclosure "
            "(the first party commits not to onward-share, the vendor does) "
            "and Processing Purpose (the permissible uses no longer match)."
        ),
    },
    "Π₂": {
        "primary":   ["Processing Purpose"],
        "secondary": [],
        "rationale": (
            "Exclusivity clauses bind a data type to a single purpose; a "
            "conflicting use for a different purpose directly breaches "
            "Processing Purpose."
        ),
    },
    "Π₃": {
        "primary":   ["Withdraw Consent"],
        "secondary": ["Provision Requirement"],
        "rationale": (
            "A consent-gated first-party clause paired with a by-default or "
            "opt-out third-party counterpart breaks the consent obligation "
            "(Withdraw Consent); Provision Requirement applies when one side "
            "makes provision conditional and the other mandatory."
        ),
    },
    "Π₄": {
        "primary":   ["Storage Period"],
        "secondary": [],
        "rationale": (
            "Conflicting retention windows between the first and third party "
            "directly breach the Storage Period disclosure requirement."
        ),
    },
}


def attribute_finding_to_gdpr(
    pattern_id: str,
    finding_gdpr_categories: list[str] | None = None,
    website_completeness: dict | None = None,
    vendor_completeness: dict | None = None,
) -> dict:
    """Map one inconsistency finding onto Article 13/14 categories.

    Returns a dict with:

    - ``primary_categories``   – categories the pattern always breaches
      (from :data:`PATTERN_GDPR_ATTRIBUTION`). Static and legally-motivated.
    - ``secondary_categories`` – categories the pattern may breach
      depending on the clauses' action/purpose. Static.
    - ``empirical_categories`` – the categories the RoBERTa classifier
      assigned to the two clauses behind this finding (union of
      ``statement_1.gdpr_categories`` and ``statement_2.gdpr_categories``
      — the caller passes them in ``finding_gdpr_categories``).
    - ``coverage_gap``         – True when any primary category is ALSO
      missing from at least one side's disclosure. Distinguishes an
      inconsistency where both sides disclose the category (a genuine
      cross-policy conflict) from one where a side doesn't mention the
      category at all (a conflict plus a completeness gap).
    - ``coverage_gap_categories`` – the primary categories that triggered
      ``coverage_gap``.
    - ``rationale``            – short legal gloss explaining why the
      pattern breaches its primary categories.

    Pure aggregation; no LLM calls.
    """
    mapping = PATTERN_GDPR_ATTRIBUTION.get(
        pattern_id, {"primary": [], "secondary": [], "rationale": ""}
    )
    primary = list(mapping.get("primary", []))
    secondary = list(mapping.get("secondary", []))
    empirical = list(finding_gdpr_categories or [])

    gap_categories: list[str] = []
    for side_cov in (website_completeness, vendor_completeness):
        if not side_cov:
            continue
        missing_set = set(side_cov.get("missing") or [])
        for category in primary:
            if category in missing_set and category not in gap_categories:
                gap_categories.append(category)

    return {
        "pattern_id": pattern_id,
        "primary_categories": primary,
        "secondary_categories": secondary,
        "empirical_categories": empirical,
        "coverage_gap": bool(gap_categories),
        "coverage_gap_categories": gap_categories,
        "rationale": str(mapping.get("rationale", "")),
    }


RIGHTS_ACTIONS = {"access_right", "deletion_right", "optout_right", "portability_right"}
PROCESSING_ACTIONS = {"collect", "use", "share", "sell", "retain", "delete", "transfer", "process"}
GENERIC_DATA_OBJECTS = {
    "information",
    "data",
    "your information",
    "your data",
    "their information",
    "their data",
    "the information",
    "the data",
    "this information",
    "this data",
    "that information",
    "that data",
    "information we collect",
    "data we collect",
    "collected data",
    "collected information",
    "all information",
    "all data",
    "any information",
    "any data",
    "additional information",
    "other information",
    "other data",
    "unspecified data",
    "unspecified personal data",
    "unspecified information",
}
WEAK_GROUNDING_TOKENS = {
    "data",
    "information",
    "personal",
    "system",
    "address",
    "details",
    "detail",
    "type",
}
NEGATIVE_ACTION_PATTERNS = {
    "collect": (r"\b(?:do|does|did|will|shall|can)\s+not\s+collect\b", r"\bnever\s+collect\b"),
    "use": (r"\b(?:do|does|did|will|shall|can)\s+not\s+use\b", r"\bnever\s+use\b"),
    "share": (
        r"\b(?:do|does|did|will|shall|can)\s+not\s+share\b",
        r"\bnever\s+share\b",
        r"\b(?:do|does|did|will|shall|can)\s+not\s+disclos(?:e|ing)\b",
    ),
    "sell": (r"\b(?:do|does|did|will|shall|can)\s+not\s+sell\b", r"\bnever\s+sell\b"),
    "retain": (
        r"\b(?:do|does|did|will|shall|can)\s+not\s+retain\b",
        r"\bnever\s+retain\b",
        r"\b(?:do|does|did|will|shall|can)\s+not\s+keep\b",
        r"\b(?:do|does|did|will|shall|can)\s+not\s+store\b",
    ),
    "transfer": (r"\b(?:do|does|did|will|shall|can)\s+not\s+transfer\b", r"\bnever\s+transfer\b"),
    "process": (r"\b(?:do|does|did|will|shall|can)\s+not\s+process\b", r"\bnever\s+process\b"),
}
OPTIONAL_INPUT_PATTERNS = (
    r"\byou (?:are|re) not required to provide\b",
    r"\bnot required to provide\b",
    r"\bif you choose to provide\b",
    r"\bif you wish to provide\b",
    r"\boptional(?:ly)? provide\b",
)
POSITIVE_PROCESSING_PATTERN = re.compile(
    r"\b(?:we|our|service providers|third parties|vendors?)\s+"
    r"(?:may\s+|can\s+|will\s+|do\s+|does\s+|might\s+|could\s+)?"
    r"(?:collect|use|share|sell|retain|transfer|process|store|keep|disclose)\b"
)


from prompts.unified_prompts import (  # vendored prompts — see prompts/unified_prompts.py
    EXTRACTION_PROMPT,
    REFLECTION_EXHAUSTION_PROMPT,
    REFLECTION_RECOVERY_PROMPT,
)


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    parts = [part.strip() for part in parts if part.strip()]
    # Multi-modality split: sentences that carry both a positive practice and
    # a negated practice should be split on the contrastive conjunction so each
    # half carries a single modality. Matters when downstream uses a per-clause
    # modality classifier (BERT Squad); a no-op for the LLM path, which can
    # already emit multiple PPS per clause.
    return [s for p in parts for s in _split_contrastive(p)]


# Positive practice verbs that, when paired with a negation in the same
# sentence, signal two separate practices joined by a contrastive conjunction.
_PRACTICE_VERB_RE = re.compile(
    r"\b(collect|use|share|sell|retain|process|store|disclose|transfer|"
    r"access|delete|provide|keep|gather)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(do not|does not|did not|will not|won['\u2019]t|never|no\s+longer|"
    r"cannot|can['\u2019]t|shall not|shan['\u2019]t)\b",
    re.IGNORECASE,
)
# Contrastive split points (order matters — longer phrases first)
_CONTRASTIVE_SPLIT_RE = re.compile(
    r"\s*[,;]?\s+(?:but|however|yet|though|although|while|whereas)\s+",
    re.IGNORECASE,
)


def _split_contrastive(sentence: str) -> list[str]:
    """Split a sentence on contrastive conjunctions IF the sentence is likely
    to carry multiple modalities (contains a practice verb AND a negation).

    Returns the sentence unchanged when the multi-modality signal is absent.
    Never splits below MIN_CLAUSE_LENGTH per half."""
    if not _PRACTICE_VERB_RE.search(sentence) or not _NEGATION_RE.search(sentence):
        return [sentence]
    parts = _CONTRASTIVE_SPLIT_RE.split(sentence)
    if len(parts) < 2:
        return [sentence]
    # Only keep splits where BOTH halves have enough content and BOTH look like
    # practice clauses (contain a practice verb). Otherwise return the original.
    cleaned = [p.strip().rstrip(",.;") for p in parts if p.strip()]
    if len(cleaned) < 2:
        return [sentence]
    if any(len(p) < MIN_CLAUSE_LENGTH for p in cleaned):
        return [sentence]
    if not all(_PRACTICE_VERB_RE.search(p) for p in cleaned):
        return [sentence]
    return cleaned


def _looks_like_header(line: str) -> bool:
    if not line or len(line) > 90:
        return False
    lower_line = line.lower()
    if "privacy policy" in lower_line or "terms and privacy" in lower_line:
        return True
    if re.match(r"^(#{1,6}\s+|\d+(?:\.\d+)*\.?\s+)", line):
        return True
    if line.endswith(":") and len(line) < 60:
        words = line.rstrip(":").split()
        if len(words) <= 6 and lower_line.split()[:1] not in (["we"], ["you"], ["your"], ["our"], ["this"], ["these"]):
            return True
    if line.isupper() and len(line) > 3:
        return True
    if re.match(r"^[A-Z][A-Za-z0-9&,'()/ -]{1,70}$", line) and line.count(".") <= 1:
        words = line.split()
        return len(words) <= 8
    return False


def _clean_header(line: str) -> str:
    line = re.sub(r"^#{1,6}\s*", "", line)
    line = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", line)
    return line.rstrip(":").strip()


def _bullet_content(line: str) -> str | None:
    match = re.match(r"^(?:[-*•►▪]\s+|\d+[.)]\s+|[a-z][.)]\s+)(.+)$", line)
    if not match:
        return None
    return match.group(1).strip()


def _looks_like_metadata(line: str) -> bool:
    lowered = line.lower()
    return bool(
        re.match(r"^(last updated|effective date|updated on|date):", lowered)
        or re.match(r"^[A-Z][a-z]+ \d{1,2}, \d{4}$", line)
    )


def _chunk_text(text: str) -> list[str]:
    text = _normalize_whitespace(text)
    if len(text) < MIN_CLAUSE_LENGTH:
        return []
    if len(text) <= MAX_CLAUSE_LENGTH:
        return [text]

    sentences = _split_sentences(text)
    if not sentences:
        return [text[:MAX_CLAUSE_LENGTH].strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        candidate_len = current_len + len(sentence) + (1 if current else 0)
        if current and candidate_len > MAX_CLAUSE_LENGTH:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len = candidate_len
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if len(chunk) >= MIN_CLAUSE_LENGTH]


def segment_clauses(text: str, policy_id: str = "policy") -> list[Clause]:
    """Split policy text into meaningful clause-sized segments."""

    clauses: list[Clause] = []
    current_section = ""
    paragraph_lines: list[str] = []
    list_context = ""
    clause_index = 0

    def add_clause(clause_text: str) -> None:
        nonlocal clause_index
        for chunk in _chunk_text(clause_text):
            clauses.append(
                Clause(
                    clause_id=f"{policy_id}_c{clause_index}",
                    text=chunk,
                    section_header=current_section,
                    position_index=clause_index,
                )
            )
            clause_index += 1

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        paragraph_text = _normalize_whitespace(" ".join(paragraph_lines))
        paragraph_lines = []
        if paragraph_text:
            add_clause(paragraph_text)

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            flush_paragraph()
            list_context = ""
            continue

        if _looks_like_metadata(line):
            flush_paragraph()
            continue

        if _looks_like_header(line):
            flush_paragraph()
            current_section = _clean_header(line)
            list_context = ""
            continue

        bullet = _bullet_content(line)
        if bullet is not None:
            if paragraph_lines:
                paragraph_text = _normalize_whitespace(" ".join(paragraph_lines))
                if paragraph_text.endswith(":"):
                    list_context = paragraph_text.rstrip(":")
                    paragraph_lines = []
                else:
                    flush_paragraph()
            clause_text = f"{list_context}: {bullet}" if list_context else bullet
            add_clause(clause_text)
            continue

        if line.endswith(":") and len(line) < MAX_CLAUSE_LENGTH:
            flush_paragraph()
            list_context = line.rstrip(":")
            continue

        paragraph_lines.append(line)

    flush_paragraph()
    print(f"  Segmented into {len(clauses)} clauses")
    return clauses


def _response_text(response) -> str:
    blocks = []
    for block in response.content:
        text = getattr(block, "text", "")
        if text:
            blocks.append(text)
    return "\n".join(blocks).strip()


def _active_model_name() -> str:
    if EXTRACTION_BACKEND == "anthropic":
        return ANTHROPIC_EXTRACTION_MODEL
    if EXTRACTION_BACKEND == "openai":
        return OPENAI_EXTRACTION_MODEL
    if EXTRACTION_BACKEND == "llamacpp":
        return f"llamacpp:{LLAMACPP_MODEL_NAME}"
    if EXTRACTION_BACKEND == "openai_compat":
        return f"openai_compat:{LLM_MODEL}"
    if EXTRACTION_BACKEND == "squad":
        raise RuntimeError(
            "EXTRACTION_BACKEND='squad' uses an internal DeBERTa specialist "
            "ensemble that is not shipped with this artifact. Use 'anthropic', "
            "'openai', 'llamacpp', or 'openai_compat'."
        )
    raise RuntimeError(f"Unsupported EXTRACTION_BACKEND '{EXTRACTION_BACKEND}'")


def _parse_json_response(text: str) -> list[dict]:
    """Parse JSON from an LLM response using several fallbacks."""

    content = text.strip()
    if not content:
        return []

    for candidate in (content,):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass

    code_match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE)
    if code_match:
        try:
            parsed = json.loads(code_match.group(1))
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass

    array_match = re.search(r"\[.*\]", content, flags=re.DOTALL)
    if array_match:
        try:
            parsed = json.loads(array_match.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    print(f"    WARNING: Failed to parse JSON response: {content[:200]}...")
    return []


def _source_mentions_data_object(raw_data_object: str, source_text: str) -> bool:
    source = _normalize_whitespace(source_text.lower())
    candidate = _normalize_whitespace(raw_data_object.lower())
    if not candidate:
        return False

    canonical = normalize_data_type(candidate)
    phrases = {candidate}
    if canonical:
        phrases.add(canonical)
    for alias, canonical_value in DATA_SYNONYMS.items():
        if canonical_value == canonical:
            phrases.add(alias)
    for phrase in sorted((phrase for phrase in phrases if phrase), key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", source):
            return True

    grounding_source = canonical or candidate
    grounding_tokens = [
        word
        for word in re.findall(r"[a-z0-9]+", grounding_source)
        if len(word) >= 3 and word not in WEAK_GROUNDING_TOKENS
    ]
    if not grounding_tokens:
        # All tokens are weak (e.g. "personal data", "behavioral data").
        # These are abstract parent types the LLM was instructed to use as
        # canonical labels — accept them when the clause contains at least one
        # weak token, confirming it is about data at all.
        weak_tokens = re.findall(r"[a-z0-9]+", grounding_source)
        return any(tok in source for tok in weak_tokens)
    return all(token in source for token in grounding_tokens)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _blankish(value: str) -> bool:
    return value.strip().lower() in {"", "none", "null", "n/a", "na", "unspecified"}


def _clean_optional_text(value: object) -> str:
    cleaned = _normalize_whitespace(str(value))
    return "" if _blankish(cleaned) else cleaned


def _merge_gdpr_categories(*category_lists: list[str]) -> list[str]:
    merged: list[str] = []
    for category in GDPR_CATEGORIES:
        if any(category in categories for categories in category_lists) and category not in merged:
            merged.append(category)
    return merged


# Trailing corporate suffix preceded by a comma: "Meta Platforms, Inc." →
# "Meta Platforms". Anchored to end-of-string so we don't disturb earlier
# commas (multi-recipient lists are kept whole; downstream handles them).
_CORPORATE_SUFFIX_RE = re.compile(
    r",\s*(?:"
    r"Inc|LLC|Ltd|Limited|Corp|Corporation|Co|Company"
    r"|GmbH|PLC|SA|S\.A|SL|S\.L|AG|BV|B\.V|NV|N\.V|SE|LLP|LP"
    r"|Holdings|Group"
    r")\.?\s*$",
    re.IGNORECASE,
)


def _normalize_recipient(value: str) -> str:
    """Canonicalize an extracted recipient string.

    Strips leading possessive pronouns ("our X" → "X"), collapses "Google
    Analytics" variants, and removes trailing corporate suffixes
    ("Meta Platforms, Inc." → "Meta Platforms") while preserving the entity
    name.

    Prior to 2026-04-19 this function did ``cleaned.split(",")[-1].strip()``,
    which kept only the suffix — "Meta Platforms, Inc." became "Inc.", and
    multi-recipient strings like "A, B, C" collapsed to "C". See audit finding
    F13/F28. Downstream: graph.py recipient nodes, Π₈'s recipient gate, and
    evaluation.py's extraction-benchmark signature all relied on this
    function.
    """
    cleaned = value.strip()
    cleaned = re.sub(r"^(our|their|its)\s+", "", cleaned, flags=re.IGNORECASE)
    if "google analytics" in cleaned.lower():
        return "Google Analytics"
    cleaned = _CORPORATE_SUFFIX_RE.sub("", cleaned).strip()
    return cleaned


def _normalize_temporality_value(temporality: TemporalityType, value: str) -> str:
    if not value:
        return ""
    lowered = value.lower()
    if temporality == TemporalityType.INDEFINITE:
        return ""
    duration_match = re.search(r"\b\d+(?:\.\d+)?\s*(?:hour|day|week|month|year)s?\b", lowered)
    if duration_match:
        return duration_match.group(0)
    if lowered in {"never", "at any time"}:
        return ""
    return value


def _is_childrens_clause(source_text: str) -> bool:
    """Detect age-scoped children's/COPPA/minors clauses.

    These are conditional prohibitions scoped to minors, not blanket prohibitions
    on collecting personal data from all users. When extracted as blanket PROHIBITION,
    they cause massive false positives in cross-policy matching (Π₈).
    """
    _CHILDREN_PATTERNS = (
        re.compile(r"children\s+(?:under|below|younger than)\s+(?:the age of\s+)?\d+", re.IGNORECASE),
        re.compile(r"(?:under|below)\s+(?:the age of\s+)?\d+\s+years?\s+(?:of age|old)", re.IGNORECASE),
        re.compile(r"(?:minors?|child|children)\s+(?:should not|must not|are not|do not)", re.IGNORECASE),
        re.compile(r"(?:do not|does not|we do not)\s+(?:knowingly\s+)?(?:collect|solicit|target)\b.*(?:children|minors|under\s+\d+)", re.IGNORECASE),
        re.compile(r"COPPA|Children'?s Online Privacy", re.IGNORECASE),
        re.compile(r"(?:persons?|individuals?|users?)\s+under\s+(?:the age of\s+)?\d+", re.IGNORECASE),
        re.compile(r"not\s+(?:intended|directed|designed)\s+(?:for|to)\s+(?:children|minors|individuals?\s+under)", re.IGNORECASE),
    )
    return any(p.search(source_text) for p in _CHILDREN_PATTERNS)


def _is_rights_clause(source_text: str) -> bool:
    """Detect data subject rights clauses (GDPR Art. 15-22, CCPA consumer rights).

    These describe user rights, not company prohibitions — even when they contain
    negation-like language ("right to object", "right to restrict").
    """
    _RIGHTS_PATTERNS = (
        re.compile(r"(?:you have |your )(?:the )?right to", re.IGNORECASE),
        re.compile(r"right to (?:access|delete|erase|object|port|restrict|rectif|correct|withdraw)", re.IGNORECASE),
        re.compile(r"(?:data subject|consumer|user) rights?\b", re.IGNORECASE),
        re.compile(r"you (?:can|may) (?:request|exercise|submit)", re.IGNORECASE),
    )
    return any(p.search(source_text) for p in _RIGHTS_PATTERNS)


def _infer_modality(source_text: str, is_negative: bool) -> Modality:
    """Weak fallback modality inference — trusts the LLM and only overrides
    in clear-cut structural cases. Called only when LLM returned UNSPECIFIED
    or OBLIGATION at the extraction callsite.

    The v6 audit showed that aggressive post-hoc negation detection flipped
    ~40% of modalities wrong (scoped negations, deletion processes, opt-out
    support, rights descriptions). This version keeps the structural safety
    nets (rights, children's, disclosure) and only infers PROHIBITION when
    the negation is clearly company-subject and action-adjacent.
    """
    source = source_text.lower()

    # Safety net 1: legal-basis statements are commitments, not prohibitions
    if re.search(r"do not require (?:your )?consent", source):
        return Modality.COMMITMENT

    # Safety net 2: rights clauses ("right to object/restrict/limit") are user permissions
    if re.search(r"right to (?:object|restrict|limit|erase|delete|access|port|rectif|correct|withdraw)", source):
        return Modality.PERMISSION

    if _is_rights_clause(source_text):
        if re.search(r"\b(may|can|allowed to)\b", source):
            return Modality.PERMISSION
        return Modality.COMMITMENT

    # Safety net 3: children's/COPPA clauses are age-scoped compliance, not blanket prohibitions
    if _is_childrens_clause(source_text):
        return Modality.COMMITMENT

    # Safety net 4: legal basis enumerations
    if re.search(r"(?:legal basis|lawful basis|legal ground)", source) and \
       re.search(r"(?:legitimate interest|consent|contract performance|vital interest)", source):
        return Modality.COMMITMENT

    # Safety net 5: disclosure/definition clauses
    if _is_disclosure_or_definition(source_text):
        return Modality.COMMITMENT

    # Only trust LLM's is_negative flag for PROHIBITION — do not re-infer from
    # loose negation keywords. The LLM has the context to know whether a "does
    # not" is a company prohibition, a scoped exception, or a description of a
    # deletion/opt-out process.
    if is_negative:
        return Modality.PROHIBITION

    if re.search(r"\b(must|shall|required to|is required to|are required to)\b", source):
        return Modality.OBLIGATION
    if re.search(r"\b(may|can|allowed to)\b", source):
        return Modality.PERMISSION
    if re.search(r"\b(might|could)\b", source):
        return Modality.POSSIBILITY
    if re.search(r"\b(generally|typically|usually)\b", source):
        return Modality.HEDGED
    return Modality.COMMITMENT


_DISCLOSURE_PATTERNS = [
    re.compile(r"the following categories", re.IGNORECASE),
    re.compile(r"categories (?:listed|described) in", re.IGNORECASE),
    re.compile(r"as defined (?:by|in|under)", re.IGNORECASE),
    re.compile(r"personal information (?:does not include|means|includes)", re.IGNORECASE),
    re.compile(r"this (?:privacy (?:policy|statement|notice)|section) describes", re.IGNORECASE),
    re.compile(r"we (?:collect|process|use|may collect) the following", re.IGNORECASE),
    re.compile(r"california customer records", re.IGNORECASE),
    re.compile(r"protected classification characteristics", re.IGNORECASE),
    re.compile(r"categories of (?:personal )?(?:information|data) (?:we|that)", re.IGNORECASE),
    re.compile(r"(?:publicly|publicly) available information", re.IGNORECASE),
    # Fix 12: Section headers, CCPA table entries, introductory paragraphs
    re.compile(r"^(?:how (?:do|does) we|what (?:do|does) we|why do we)", re.IGNORECASE),
    re.compile(r"^this (?:chart|table|section) (?:below |above )?(?:summarizes|describes|lists)", re.IGNORECASE),
    re.compile(r"^\*{0,2}[\"']?(?:sale|sharing|collection|use|disclosure)[\"']?\*{0,2}\s*$", re.IGNORECASE),
]


def _is_disclosure_or_definition(source_text: str) -> bool:
    """Detect CCPA category lists, legal definitions, and scope descriptions.

    These clauses often contain negation ("does not include") but are definitional,
    not prohibitive — they should not receive PROHIBITION modality.
    """
    return any(p.search(source_text) for p in _DISCLOSURE_PATTERNS)


# Round 9: Non-practice clause detectors. Round 8's audit showed that modality
# misextraction is the dominant false-positive source: the LLM (and the weak
# _infer_modality fallback) assigns PROHIBITION to clauses that are actually
# CCPA category tables, deletion/retention processes, user rights, scoped
# negations, legal-basis statements, glossary definitions, or section headers.
# These detectors let us downgrade PROHIBITION on any clause — whether the
# modality came from the LLM or from the fallback.

_DELETION_PROCESS_RE = re.compile(
    r"\b(?:we|you can)\s+(?:will\s+)?(?:delete|erase|destroy|remove|anonymize|anonymise|de-?identify)\b"
    r"|\bdeleted\s+(?:within|after|once|when)\b"
    r"|\bremoved\s+(?:within|after|once|when)\b"
    r"|\bwill\s+(?:be\s+)?(?:deleted|removed|destroyed|anonymized|anonymised)\b"
    r"|\bwhen\s+(?:no\s+longer\s+needed|you\s+(?:delete|close))\b",
    re.IGNORECASE,
)

_RETENTION_PROCESS_RE = re.compile(
    r"\b(?:we|retain|retained|keep|store|hold)\b[^.]*\b"
    r"(?:for (?:as\s+)?(?:long\s+as|the\s+(?:duration|time|period))|"
    r"up\s+to\s+\d+\s*(?:hour|day|week|month|year)|"
    r"no\s+longer\s+than|"
    r"only\s+as\s+long\s+as|"
    r"(?:\d+\s*(?:hour|day|week|month|year)s?)|"
    r"until\s+(?:you|no\s+longer|the))",
    re.IGNORECASE,
)

_LEGAL_BASIS_RE = re.compile(
    r"\b(?:lawful\s+basis|legal\s+basis|legal\s+ground|legitimate\s+interest|"
    r"legal\s+obligation|vital\s+interest|contract(?:ual)?\s+(?:necessity|performance)|"
    r"performance\s+of\s+(?:a\s+|the\s+)?contract|"
    r"do(?:es)?\s+not\s+require\s+(?:your\s+)?consent|"
    r"where\s+(?:we\s+)?(?:have|rely\s+on)\s+(?:a\s+)?legitimate\s+interest)\b",
    re.IGNORECASE,
)

_SCOPED_NEGATION_RE = re.compile(
    # "reports/logs/insights/analytics do not include/contain ..."
    r"\b(?:reports?|logs?|insights?|analytics|metrics|statistics|aggregated\s+data)"
    r"\s+(?:do|does|shall|will)\s+not\s+(?:include|contain|identify|reveal)\b"
    # "does not apply where/when/to"
    r"|\bdo(?:es)?\s+not\s+apply\s+(?:to|where|when|in)\b"
    # "non-personally identifiable / does not identify individual"
    r"|\bdoes\s+not\s+identify\s+(?:any\s+)?(?:individual|specific|natural\s+person)"
    r"|\bnon[-\s]?personally[-\s]identif"
    # "do not [action] in this context / in this case"
    r"|\b(?:do|does|will|shall)\s+not\s+\w+\s+(?:any\s+)?(?:personal\s+)?(?:data|information)"
    r"\s+in\s+this\s+(?:context|case|regard)\b",
    re.IGNORECASE,
)

_GLOSSARY_DEF_RE = re.compile(
    r"\b(?:\w+\s+){0,3}(?:means|shall\s+mean|refers\s+to|is\s+defined\s+as)\b"
    r"|\"[^\"]{2,30}\"\s+means\b"
    r"|^[A-Z][a-zA-Z\s]{2,40}:\s",
    re.IGNORECASE,
)

_SECTION_HEADER_RE = re.compile(
    r"^\s*\*{0,2}[\"']?(?:how\s+(?:do|does)\s+we|what\s+(?:do|does)\s+we|why\s+do\s+we|"
    r"when\s+do\s+we|who\s+(?:we|do\s+we))[^.?!]*[?:]?\*{0,2}\s*$"
    r"|^\s*(?:sale|sharing|collection|use|disclosure|retention|security|cookies?|"
    r"your\s+rights?|categories)(?:\s+of\s+[\w\s]+)?\s*:?\s*$",
    re.IGNORECASE,
)


def _clause_is_non_practice(source_text: str) -> tuple[bool, str]:
    """Return (True, reason) if the clause is non-practice text that should
    never carry PROHIBITION modality. Reasons: rights, legal_basis, deletion,
    retention, disclosure, scoped_negation, glossary, section_header, children.
    """
    if _is_rights_clause(source_text):
        return True, "rights"
    if _is_disclosure_or_definition(source_text):
        return True, "disclosure"
    if _is_childrens_clause(source_text):
        return True, "children"
    if _LEGAL_BASIS_RE.search(source_text):
        return True, "legal_basis"
    if _DELETION_PROCESS_RE.search(source_text):
        return True, "deletion_process"
    if _RETENTION_PROCESS_RE.search(source_text):
        return True, "retention_process"
    if _SCOPED_NEGATION_RE.search(source_text):
        return True, "scoped_negation"
    if _GLOSSARY_DEF_RE.search(source_text):
        return True, "glossary"
    # Section header detection only on short lines (<120 chars) to avoid matching
    # the first sentence of a paragraph.
    if len(source_text.strip()) < 120 and _SECTION_HEADER_RE.search(source_text.strip()):
        return True, "section_header"
    return False, ""


_SCOPING_LANGUAGE_RE = re.compile(
    # Scoping phrases that indicate the negation is conditional/limited,
    # NOT a blanket prohibition. Note: "without your consent" is intentionally
    # EXCLUDED — that's a strong prohibition with a consent exception, not
    # a scoped negation. The vendor violates it by processing without consent.
    r"\b(?:"
    r"unless\s+(?:you|the\s+user|they)\s+(?:specifically|explicitly|expressly)?\s*(?:sign|log|opt|agree|consent|choose|decide)"
    r"|except\s+(?:as|when|where|in)\s+(?:described|outlined|stated|specified|required|permitted|noted)"
    r"|in\s+(?:this|that)\s+context"
    r"|in\s+(?:this|that)\s+case"
    r"|in\s+(?:this|that)\s+regard"
    r"|(?:only|solely)\s+(?:if|when|where)\s+(?:you|the\s+user|they|we|it)"
    r"|to\s+the\s+extent\s+(?:that|necessary|required|permitted)"
    r"|does\s+not\s+apply\s+(?:to|where|when|in)"
    r")\b",
    re.IGNORECASE,
)


def _has_explicit_negative_practice(source_text: str, action: str, purpose: str) -> bool:
    """Detect explicit negative practice patterns like 'we do not share'.

    Returns False if the negation is scoped by exceptions like 'without
    your consent', 'unless you opt in', 'except as described', etc.
    Scoped negations are not blanket prohibitions — they coexist with
    permitted uses and should not be forced to PROHIBITION.
    """
    source = source_text.lower()

    # Check if the text contains scoping language that limits the negation
    has_scoping = bool(_SCOPING_LANGUAGE_RE.search(source))

    for pattern in NEGATIVE_ACTION_PATTERNS.get(action, ()):
        if re.search(pattern, source):
            if has_scoping:
                return False  # scoped negation, not a blanket prohibition
            return True
    if action in {"use", "process"} and purpose in {"advertising", "targeted advertising", "marketing"}:
        if re.search(
            r"\b(?:do|does|did|will|shall|can)\s+not\s+(?:use|process)\b.*"
            r"\b(?:advertising|ads|personalized ads|targeted advertising|marketing)\b",
            source,
        ):
            if has_scoping:
                return False
            return True
    return False


_NEGATION_WORDS_RE = re.compile(
    r"\b(?:do\s+not|does\s+not|will\s+not|shall\s+not|cannot|can\s+not|never|"
    r"not\s+(?:sell|share|transfer|disclose|rent|collect|use|process|retain|store))\b",
    re.IGNORECASE,
)


def _validate_prohibition(
    source_text: str,
    modality: "Modality",
    is_negative: bool,
    action: str,
) -> tuple["Modality", bool]:
    """Round 11: Validate PROHIBITION assignments.

    Returns (modality, is_negative) after validation. Only keeps
    PROHIBITION when the text has clear negation language AND no
    escape/scoping clause. Downgrades to COMMITMENT otherwise.

    This catches:
    - Modality misextraction (PROHIBITION without negation words)
    - Scoped negations ("without consent", "unless you opt in")
    - Feature disclaimers ("In AI Chat, we do not share account data")
    - Delegation ("we don't process cards, we use Stripe")
    - Third-party descriptions ("your telecom operator does not...")
    - Security descriptions ("we protect/encrypt your data")
    """
    from schema import Modality as Mod

    text = source_text.lower()

    # Step 1: No negation language at all → definitely not PROHIBITION
    if not _NEGATION_WORDS_RE.search(text):
        return Mod.COMMITMENT, False

    # Step 2: Scoping language makes it a COMMITMENT, not a blanket prohibition
    if _SCOPING_LANGUAGE_RE.search(text):
        return Mod.COMMITMENT, False

    # Step 3: Feature/product-specific disclaimers
    if re.search(
        r"(?:to\s+(?:access|use)\s+\w+|when\s+(?:you\s+)?(?:use|access)\s+\w+)"
        r".*(?:(?:do|does|will)\s+not\s+(?:share|send|transmit|transfer))",
        text, re.DOTALL,
    ):
        return Mod.COMMITMENT, False

    # Step 4: Delegation to third party
    if re.search(
        r"(?:does\s+not\s+(?:process|store|handle|receive).*"
        r"(?:instead|but)\s+(?:uses?|relies?))",
        text, re.DOTALL,
    ):
        return Mod.COMMITMENT, False
    if any(phrase in text for phrase in (
        "rely on trusted third",
        "instead uses a third",
        "instead we use",
        "handled by our payment",
    )):
        return Mod.COMMITMENT, False

    # Step 5: Technical/system facts
    if any(phrase in text for phrase in (
        "is not retained",
        "not stored in",
        "generated dynamically",
        "automatically deleted",
        "automatically removed",
    )):
        return Mod.COMMITMENT, False

    # Step 6: Glossary/definitional
    if any(phrase in text for phrase in (
        "means information that",
        "is defined as",
        "refers to information",
        "personal information means",
        "personal data means",
    )):
        return Mod.COMMITMENT, False

    # Step 7: Third-party disclaimer
    if any(phrase in text for phrase in (
        "this privacy policy does not cover",
        "this policy does not apply",
        "not controlled by us",
        "governed by their privacy",
    )):
        return Mod.COMMITMENT, False

    # Step 8: Third-party actor (not the company)
    if any(phrase in text for phrase in (
        "your telecom operator",
        "your internet service provider",
        "the telecom operator",
        "the mobile operator",
    )):
        return Mod.COMMITMENT, False

    # Step 9: CCPA headers / opt-out labels (not policy statements)
    if any(phrase in text for phrase in (
        "do not sell or share my personal",
        "do not sell my personal",
        "opt out of the sale",
        "opt-out of the sale",
        "opt out of our sharing",
        "opt-out of our sharing",
        "shine the light",
    )):
        return Mod.COMMITMENT, False

    # Step 10: Weak preferences (not firm prohibitions)
    if any(phrase in text for phrase in (
        "we would prefer",
        "we prefer you",
        "we encourage you not to",
        "we recommend you do not",
        "we suggest you",
    )):
        return Mod.COMMITMENT, False

    # Step 11: Education/student/children scoped
    if any(phrase in text for phrase in (
        "students using",
        "student data",
        "student information",
        "children under",
        "minors under",
        "knowingly collect",
        "knowingly sell",
        "knowingly share",
    )):
        return Mod.COMMITMENT, False

    # Passed all checks — keep as PROHIBITION
    return modality, is_negative


def _is_optional_nonpractice_clause(source_text: str) -> bool:
    source = source_text.lower()
    return any(re.search(pattern, source) for pattern in OPTIONAL_INPUT_PATTERNS) and not POSITIVE_PROCESSING_PATTERN.search(
        source
    )


def _infer_condition(source_text: str) -> ConditionType:
    source = source_text.lower()
    if re.search(r"\b(consent|cookie consent|explicit consent|with your consent)\b", source):
        return ConditionType.UPON_CONSENT
    if re.search(r"\b(if you opt in|if opted in|opt-in)\b", source):
        return ConditionType.IF_OPTED_IN
    if re.search(r"\b(unless you opt out|unless opted out|opt-out|opt out)\b", source):
        return ConditionType.UNLESS_OPTED_OUT
    if re.search(r"\b(required by law|required by applicable law|when required|legal obligation)\b", source):
        return ConditionType.WHEN_REQUIRED
    if re.search(r"\b(by default|automatically|no additional user consent)\b", source):
        return ConditionType.BY_DEFAULT
    return ConditionType.UNSPECIFIED


def _default_actor(policy_source: str) -> str:
    if policy_source.startswith("third_party:"):
        vendor_name = policy_source.split(":", 1)[1].strip()
        return f"ThirdParty:{vendor_name}" if vendor_name else "ThirdParty"
    return "FirstParty"


def _normalize_actor(raw_actor: str, policy_source: str, action: str) -> str:
    lowered = raw_actor.lower()
    if lowered in {"we", "us", "our", "ours", "company", "the company", "website operator"}:
        return _default_actor(policy_source)
    if lowered in {"you", "user", "users", "individual user", "individual users"} and action in RIGHTS_ACTIONS:
        return "DataSubject"
    if policy_source.startswith("third_party:") and lowered in {"google", "google analytics"}:
        return policy_source.split(":", 1)[1].strip() or raw_actor
    return raw_actor


def _is_too_generic_data_object(raw_data_object: str, normalized_data_object: str) -> bool:
    raw_norm = _normalize_whitespace(raw_data_object.lower())
    canonical = _normalize_whitespace(normalized_data_object.lower())
    if raw_norm in GENERIC_DATA_OBJECTS or canonical in GENERIC_DATA_OBJECTS:
        return True
    if raw_norm.startswith(("this ", "that ", "such ", "any ", "all ")) and canonical in {
        "information",
        "data",
        "personal data",
    }:
        return True
    if canonical in {"information", "data"}:
        return True
    return False


def _infer_gdpr_categories(
    action: str,
    purpose: str,
    recipient: str,
    condition: ConditionType,
    temporality: TemporalityType,
    data_object: str,
    source_text: str,
) -> list[str]:
    source = source_text.lower()
    inferred: list[str] = []

    # Data Handling
    if action in {"collect", "use", "process"} and data_object:
        inferred.append("Data Categories")
    if action in {"use", "process", "collect"} and purpose:
        inferred.append("Processing Purpose")
    if action in {"share", "sell", "transfer"} or recipient:
        inferred.append("Data Recipients")
    if re.search(r"\b(obtained from|collected from|received from|provided by|sourced from|gather.*from)\b", source):
        inferred.append("Source of Data")
    if action in {"retain", "delete"} or temporality != TemporalityType.UNSPECIFIED:
        inferred.append("Storage Period")

    # Data Subject Rights
    if action == "access_right":
        inferred.append("Right to Access")
    elif action == "deletion_right":
        inferred.append("Right to Erase")
    elif action == "portability_right":
        inferred.append("Right to Portability")
    elif action == "optout_right":
        inferred.append("Right to Object")
    if re.search(r"\b(restrict processing|restriction of processing|limit.*processing)\b", source):
        inferred.append("Right to Restrict")
    if action == "optout_right" or "withdraw consent" in source or "unsubscribe" in source:
        inferred.append("Withdraw Consent")
    if re.search(r"\b(lodge a complaint|file a complaint|supervisory authority|data protection authority)\b", source):
        inferred.append("Lodge Complaint")

    # Legal & Organizational
    if re.search(r"\b(legal obligation|required by law|required by applicable law|vital interests|legitimate interests|contractual necessity|legal basis|obligation to provide)\b", source):
        inferred.append("Provision Requirement")
    if re.search(r"\b(adequacy decision|adequate level of protection|standard contractual clauses|binding corporate rules)\b", source):
        inferred.append("Adequacy Decision")
    if re.search(r"\b(automated decision-making|automated processing|profiling|machine learning|create user segments|predict user interests)\b", source):
        inferred.append("Profiling")
    if re.search(r"\b(data protection officer|dpo|privacy officer|contact our dpo)\b", source):
        inferred.append("DPO Contact")
    if re.search(r"\b(data controller|controller.*contact|contact.*controller|identity of the controller)\b", source):
        inferred.append("Controller Contact")
    if re.search(r"\b(encryption|security measures|secure servers|protect.*data|industry-standard security|data security|safeguards|technical.*measures|organizational.*measures)\b", source):
        inferred.append("Safeguards Copy")

    return _merge_gdpr_categories(inferred)


def _build_pps(raw: dict, clause: Clause, policy_source: str, idx: int) -> PPS | None:
    """Build a validated PPS object from extracted JSON."""

    action = str(raw.get("action", "")).strip().lower()
    if action not in VALID_ACTIONS:
        print(f"    WARNING: Invalid action '{action}', skipping")
        return None
    if action in PROCESSING_ACTIONS and _is_optional_nonpractice_clause(clause.text):
        print("    WARNING: Optional-input clause without an affirmative processing practice, skipping")
        return None

    data_object_raw = _normalize_whitespace(str(raw.get("data_object", "")))
    if not data_object_raw:
        print("    WARNING: Empty data_object, skipping")
        return None

    data_object = normalize_data_type(data_object_raw)
    if _is_too_generic_data_object(data_object_raw, data_object):
        print(f"    WARNING: data_object '{data_object_raw}' is too generic, skipping")
        return None

    if not _source_mentions_data_object(data_object_raw, clause.text):
        print(f"    WARNING: data_object '{data_object_raw}' not grounded in source clause, skipping")
        return None

    try:
        modality = Modality[str(raw.get("modality", "UNSPECIFIED")).strip().upper()]
    except KeyError:
        modality = Modality.UNSPECIFIED

    try:
        condition = ConditionType(str(raw.get("condition", "unspecified")).strip().lower())
    except ValueError:
        condition = ConditionType.UNSPECIFIED

    try:
        temporality = TemporalityType(str(raw.get("temporality", "unspecified")).strip().lower())
    except ValueError:
        temporality = TemporalityType.UNSPECIFIED

    is_negative = _coerce_bool(raw.get("is_negative", False))
    inferred_modality = _infer_modality(clause.text, is_negative)
    if modality in {Modality.UNSPECIFIED, Modality.OBLIGATION} or (
        modality == Modality.POSSIBILITY and inferred_modality == Modality.PERMISSION
    ):
        modality = inferred_modality

    # Round 9: Universal PROHIBITION downgrade. Regardless of whether the LLM
    # or the fallback assigned PROHIBITION, strip it if the clause is actually
    # non-practice text. Round 8 audit showed ~35% of FPs come from PROHIBITION
    # being assigned to rights clauses, deletion/retention processes, CCPA
    # category tables, legal-basis statements, scoped negations, glossary
    # definitions, and section headers.
    if modality == Modality.PROHIBITION:
        is_non_practice, reason = _clause_is_non_practice(clause.text)
        if is_non_practice:
            if reason == "rights":
                modality = Modality.PERMISSION
            else:
                modality = Modality.COMMITMENT
            is_negative = False

    # Round 11: PROHIBITION validation. Only keep PROHIBITION when the text
    # contains clear negation language AND no escape/exception clause.
    # Catches: modality_misextraction (31% of FPs), scoped negations (13%),
    # feature-specific disclaimers, delegation statements, etc.
    if modality == Modality.PROHIBITION or is_negative:
        modality, is_negative = _validate_prohibition(
            clause.text, modality, is_negative, action,
        )

    inferred_condition = _infer_condition(clause.text)
    if condition == ConditionType.UNSPECIFIED or (
        condition == ConditionType.BY_DEFAULT and inferred_condition == ConditionType.UNSPECIFIED
    ):
        condition = inferred_condition

    purpose = normalize_purpose(_clean_optional_text(raw.get("purpose", "")))
    recipient = _normalize_recipient(_clean_optional_text(raw.get("recipient", "")))
    actor = _clean_optional_text(raw.get("actor", "")) or _default_actor(policy_source)
    actor = _normalize_actor(actor, policy_source, action)
    temporality_value = _normalize_temporality_value(
        temporality,
        _clean_optional_text(raw.get("temporality_value", "")),
    )

    if recipient.lower() in {"we", "us", "our", "ours", actor.lower()}:
        recipient = ""
    if purpose == "unspecified":
        purpose = ""
    if _has_explicit_negative_practice(clause.text, action, purpose):
        # Round 9: Only force PROHIBITION if the clause is genuine practice
        # text. Non-practice clauses (rights, deletion processes, etc.) can
        # contain "do not" negation keywords without being company prohibitions.
        is_non_practice, _reason = _clause_is_non_practice(clause.text)
        if not is_non_practice:
            is_negative = True
            modality = Modality.PROHIBITION
    if action in RIGHTS_ACTIONS:
        purpose = ""
        recipient = ""
        temporality_value = ""
        condition = ConditionType.UNSPECIFIED
        temporality = TemporalityType.UNSPECIFIED
    if modality == Modality.PROHIBITION and temporality in {
        TemporalityType.INDEFINITE,
        TemporalityType.UNSPECIFIED,
    }:
        temporality_value = ""
        temporality = TemporalityType.UNSPECIFIED
    # GDPR categories: RoBERTa classifier (mandatory)
    gdpr_categories = _gdpr_classifier.classify(clause.text)

    scope = _normalize_scope(_clean_optional_text(raw.get("scope", "")))

    return PPS(
        id=f"{clause.clause_id}_s{idx}",
        actor=actor,
        action=action,
        modality=modality,
        data_object=data_object,
        purpose=purpose,
        recipient=recipient,
        condition=condition,
        temporality=temporality,
        temporality_value=temporality_value,
        is_negative=is_negative,
        gdpr_categories=gdpr_categories,
        source_text=clause.text,
        source_section=clause.section_header,
        policy_source=policy_source,
        scope=scope,
    )


def _normalize_scope(raw_scope: str) -> str:
    """Normalize the LLM-provided scope to a short lowercase descriptor.

    Defaults to "global" when missing, empty, or 'unspecified'. Scope is the
    9th PPS field added in R10 (2026-04-12); it gates cross-clause pattern
    firing so two clauses with incompatible audiences are not flagged as
    contradictions.
    """
    value = (raw_scope or "").strip().lower()
    if not value or value in {"none", "null", "unspecified", "n/a", "na", "any", "all"}:
        return "global"
    # Collapse whitespace and cap length
    value = re.sub(r"\s+", " ", value)
    if len(value) > 60:
        value = value[:60].rsplit(" ", 1)[0]
    return value or "global"


def _get_cache_path(text: str) -> Path:
    # BUILD_PPS_VERSION gates the _build_pps post-processing cascade (modality
    # validation, recipient normalization, non-practice filter, GDPR reclassify).
    # Including it in the key invalidates stale PPS when that logic evolves —
    # audit F27.
    cache_identity = (
        f"{EXTRACTION_BACKEND}:{_active_model_name()}"
        f":{EXTRACTION_PROMPT_VERSION}:{BUILD_PPS_VERSION}:{text}"
    )
    digest = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{digest}.json"


# Segment-level GDPR cache: per-policy RoBERTa classification over every
# segmented clause, plus the aggregate clause-coverage dict. Bump this
# version when (a) the RoBERTa model is swapped or retrained, (b) per-
# class thresholds change, or (c) segment_clauses() output changes. The
# cache is read-only in normal pipeline code — only extract_only_batch
# and any downstream tool that wants segment-level GDPR output should
# touch it.
GDPR_SEGMENT_CACHE_VERSION = "2026-04-22-a"


def _get_gdpr_segment_cache_path(text: str) -> Path:
    cache_identity = f"gdpr_seg:{GDPR_SEGMENT_CACHE_VERSION}:{text}"
    digest = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"gdpr_seg_{digest}.json"


def load_gdpr_segment_cache(text: str) -> dict | None:
    """Return the cached payload {segments, coverage} or None on miss.
    Tolerates partial / corrupt entries (returns None)."""
    p = _get_gdpr_segment_cache_path(text)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as fh:
            d = json.load(fh)
        if isinstance(d, dict) and "segments" in d and "coverage" in d:
            return d
        return None
    except Exception:
        return None


def save_gdpr_segment_cache(text: str, segments: list[dict], coverage: dict) -> None:
    """Atomic first-writer-wins save. No-op if a peer already wrote it."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _get_gdpr_segment_cache_path(text)
    if p.exists():
        return
    tmp = p.parent / f"{p.name}.tmp.{os.getpid()}"
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump({
                "version":  GDPR_SEGMENT_CACHE_VERSION,
                "segments": segments,
                "coverage": coverage,
            }, fh)
        os.replace(tmp, p)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _get_clause_cache_path(section: str, clause_text: str) -> Path:
    # Clause cache stores raw LLM items (pre-_build_pps). Including
    # BUILD_PPS_VERSION here is defensive: a future refactor that moved part of
    # _build_pps into the raw-items pipeline would otherwise read stale cache.
    cache_identity = (
        f"clause:{EXTRACTION_BACKEND}:{_active_model_name()}"
        f":{EXTRACTION_PROMPT_VERSION}:{BUILD_PPS_VERSION}:{section}:{clause_text}"
    )
    digest = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"clause_{digest}.json"


def _load_cache(text: str) -> list[dict] | None:
    cache_path = _get_cache_path(text)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if isinstance(cached, dict) and isinstance(cached.get("statements"), list):
            return cached["statements"]
        if isinstance(cached, list):
            return cached
    except (OSError, json.JSONDecodeError):
        print(f"  WARNING: Cache read failed for {cache_path}, regenerating")
    return None


def _atomic_write_json(path: Path, payload) -> None:
    """Write JSON to ``path`` atomically on lustre / any POSIX filesystem.

    Strategy: write to a sibling temp file (same directory so ``os.rename``
    is a true rename, not a cross-device copy) and then rename — ``rename``
    is atomic on POSIX, so no reader ever sees a half-written file even
    when multiple HPC shards are writing the same cache key concurrently.
    The first writer wins if an entry already exists (race-resilient)."""
    if path.exists():
        # Another shard beat us to it; the content is deterministic under
        # our cache key, so skip the write to keep lustre I/O down.
        return
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)          # atomic on POSIX
    except Exception:
        # Clean up a leftover temp file; the caller's next read just
        # regenerates from source, no corruption exposed.
        try: tmp.unlink()
        except FileNotFoundError: pass
        raise


def _save_cache(text: str, results: list[dict]) -> None:
    cache_path = _get_cache_path(text)
    _atomic_write_json(cache_path, {
        "backend": EXTRACTION_BACKEND,
        "model": _active_model_name(),
        "prompt_version": EXTRACTION_PROMPT_VERSION,
        "statements": results,
    })


def _load_clause_cache(section: str, clause_text: str) -> list[dict] | None:
    cache_path = _get_clause_cache_path(section, clause_text)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if isinstance(cached, list):
            return cached
    except (OSError, json.JSONDecodeError):
        print(f"    WARNING: Clause cache read failed for {cache_path}, regenerating")
    return None


def _save_clause_cache(section: str, clause_text: str, raw_items: list[dict]) -> None:
    cache_path = _get_clause_cache_path(section, clause_text)
    _atomic_write_json(cache_path, raw_items)


def _create_extraction_client():
    if EXTRACTION_BACKEND == "anthropic":
        if anthropic is None:
            raise RuntimeError(
                "The 'anthropic' package is not installed. Run 'pip install -r requirements.txt' first."
            )
        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Extraction requires a valid Anthropic API key."
            )
        return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    if EXTRACTION_BACKEND == "openai":
        if OpenAI is None:
            raise RuntimeError(
                "The 'openai' package is not installed. Run 'pip install -r requirements.txt' first."
            )
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Extraction requires a valid OpenAI API key."
            )
        return OpenAI(api_key=OPENAI_API_KEY)

    if EXTRACTION_BACKEND == "llamacpp":
        if OpenAI is None:
            raise RuntimeError(
                "The 'openai' package is not installed (needed for llamacpp backend). "
                "Run 'pip install -r requirements.txt' first."
            )
        return OpenAI(base_url=LLAMACPP_BASE_URL, api_key="not-needed")

    if EXTRACTION_BACKEND == "openai_compat":
        if OpenAI is None:
            raise RuntimeError(
                "The 'openai' package is not installed (needed for the openai_compat backend). "
                "Run 'pip install -r requirements.txt' first."
            )
        if not LLM_API_KEY:
            raise RuntimeError(
                "LLM_API_KEY is not set. The openai_compat backend (local 2× A100 server "
                "or rented Vast.ai 4× A100 endpoint) requires a key."
            )
        return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    if EXTRACTION_BACKEND == "squad":
        # The squad needs no HTTP client — it loads DeBERTa heads locally.
        # Return a sentinel string so downstream _extract_one sees a truthy
        # "client" and routes to the squad adapter branch.
        return "squad"

    raise RuntimeError(
        f"Unsupported EXTRACTION_BACKEND '{EXTRACTION_BACKEND}'. "
        f"Use 'anthropic', 'openai', 'llamacpp', or 'openai_compat'."
    )


def _format_pps_for_reflection(raw_items: list[dict]) -> str:
    """Format previously extracted PPS as JSON for the reflection prompt."""
    simplified = []
    for item in raw_items:
        entry = {}
        for key in ["actor", "action", "modality", "data_object", "purpose",
                     "recipient", "condition", "temporality", "scope"]:
            if key in item and item[key]:
                entry[key] = item[key]
        if "is_negative" in item and item["is_negative"]:
            entry["is_negative"] = True
        simplified.append(entry)
    return json.dumps(simplified, indent=2, ensure_ascii=False)


def _call_extraction_model_multiturn(client, messages: list[dict]) -> str:
    """Call the extraction model with a multi-turn conversation (for reflection)."""
    if EXTRACTION_BACKEND == "anthropic":
        response = client.messages.create(
            model=ANTHROPIC_EXTRACTION_MODEL,
            max_tokens=EXTRACTION_MAX_TOKENS,
            temperature=EXTRACTION_TEMPERATURE,
            messages=messages,
        )
        return _response_text(response)

    if EXTRACTION_BACKEND == "openai":
        # OpenAI responses API doesn't support multi-turn the same way;
        # fall back to chat completions style
        from openai import OpenAI as _OAI
        oai_client = _OAI(api_key=OPENAI_API_KEY)
        response = oai_client.chat.completions.create(
            model=OPENAI_EXTRACTION_MODEL,
            messages=messages,
            max_tokens=EXTRACTION_MAX_TOKENS,
            temperature=EXTRACTION_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()

    if EXTRACTION_BACKEND == "llamacpp":
        response = client.chat.completions.create(
            model=LLAMACPP_MODEL_NAME,
            messages=messages,
            max_tokens=EXTRACTION_MAX_TOKENS,
            temperature=EXTRACTION_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()

    if EXTRACTION_BACKEND == "openai_compat":
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=EXTRACTION_MAX_TOKENS,
            temperature=EXTRACTION_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()

    raise RuntimeError(f"Unsupported EXTRACTION_BACKEND '{EXTRACTION_BACKEND}'")


def _parse_exhaustion_response(text: str) -> bool:
    """Parse the exhaustion-check response to a bool.

    Returns True when the LLM says extraction is complete (exhausted=true),
    False when at least one practice may be missing.

    Parser order:
      1. Direct json.loads of the full response.
      2. Regex-extract the first {...} object containing "exhausted".
      3. Unparseable → conservative default: True (stop rather than loop on
         garbage). The outer loop caps at EXTRACTION_REFLECTION_ROUNDS, so
         this default cannot cause pathological under-extraction — at worst
         it stops one round early on a malformed response.
    """
    content = (text or "").strip()
    if not content:
        return True

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "exhausted" in parsed:
            return bool(parsed["exhausted"])
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[^{}]*"exhausted"[^{}]*\}', content)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and "exhausted" in parsed:
                return bool(parsed["exhausted"])
        except json.JSONDecodeError:
            pass

    return True


def _check_if_exhausted(client, prompt: str, raw_items: list[dict]) -> bool:
    """Ask the LLM if more statements remain. Returns True if exhausted.

    On transient server error, returns False (not exhausted) so the outer
    reflection loop keeps trying, capped by EXTRACTION_REFLECTION_ROUNDS.
    Silent under-extraction on a server hiccup is the worse failure mode
    than one extra call per clause.
    """
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": _format_pps_for_reflection(raw_items)},
        {"role": "user", "content": REFLECTION_EXHAUSTION_PROMPT},
    ]
    try:
        response_text = _call_extraction_model_multiturn(client, messages)
    except Exception:
        # Transient server error → assume NOT exhausted; the outer loop's
        # round cap bounds the worst case to one extra call per clause.
        return False

    return _parse_exhaustion_response(response_text)


def _extract_with_reflection(client, prompt: str) -> list[dict]:
    """Iterative extraction with reflection rounds (inspired by PoliGrapher-LM).

    Round 1: Standard extraction.
    Round 2+: Feed back previous extractions and ask for missed statements.
    After each round, check if the LLM thinks extraction is exhausted.
    """
    all_items: list[dict] = []
    seen_keys: set[str] = set()

    for round_num in range(EXTRACTION_REFLECTION_ROUNDS):
        if round_num == 0:
            # First round: standard single-turn extraction
            response_text = _call_extraction_model(client, prompt)
            new_items = _parse_json_response(response_text)
        else:
            # Reflection round: feed back previous results, ask for more
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _format_pps_for_reflection(all_items)},
                {"role": "user", "content": REFLECTION_RECOVERY_PROMPT},
            ]
            try:
                response_text = _call_extraction_model_multiturn(client, messages)
                new_items = _parse_json_response(response_text)
            except Exception:
                new_items = []

        # Deduplicate: skip items we've already seen
        unique_new = []
        for item in new_items:
            if not isinstance(item, dict):
                continue
            # Round 12 fix: dedup key now includes condition, temporality,
            # temporality_value, scope, is_negative. The previous 5-field key
            # silently collapsed real variants — e.g. "collect email by_default"
            # and "collect email upon_consent" are distinct practices that Π₄
            # depends on, but deduped to the same key. Reflection could ask
            # the model for these variants, but dedup killed them before they
            # reached the graph. Now all 10 schema-material fields vote.
            item_key = json.dumps(
                {k: item.get(k, "") for k in [
                    "action", "data_object", "modality", "purpose", "recipient",
                    "condition", "temporality", "temporality_value",
                    "is_negative", "scope",
                ]},
                sort_keys=True,
            )
            if item_key not in seen_keys:
                seen_keys.add(item_key)
                unique_new.append(item)

        if not unique_new:
            if round_num > 0:
                print(f"    Reflection round {round_num + 1}: no new statements")
            break

        all_items.extend(unique_new)
        print(f"    {'Extraction' if round_num == 0 else f'Reflection round {round_num + 1}'}: {len(unique_new)} statement(s)")

        # Check exhaustion (skip on last allowed round)
        if round_num < EXTRACTION_REFLECTION_ROUNDS - 1:
            if _check_if_exhausted(client, prompt, all_items):
                print(f"    Exhaustion check: complete ({len(all_items)} total)")
                break

    return all_items


def _call_extraction_model(client, prompt: str) -> str:
    if EXTRACTION_BACKEND == "anthropic":
        response = client.messages.create(
            model=ANTHROPIC_EXTRACTION_MODEL,
            max_tokens=EXTRACTION_MAX_TOKENS,
            temperature=EXTRACTION_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        return _response_text(response)

    if EXTRACTION_BACKEND == "openai":
        response = client.responses.create(
            model=OPENAI_EXTRACTION_MODEL,
            input=prompt,
            max_output_tokens=EXTRACTION_MAX_TOKENS,
        )
        output_text = getattr(response, "output_text", "")
        return str(output_text).strip()

    if EXTRACTION_BACKEND == "llamacpp":
        response = client.chat.completions.create(
            model=LLAMACPP_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=EXTRACTION_MAX_TOKENS,
            temperature=EXTRACTION_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()

    if EXTRACTION_BACKEND == "openai_compat":
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=EXTRACTION_MAX_TOKENS,
            temperature=EXTRACTION_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()

    raise RuntimeError(f"Unsupported EXTRACTION_BACKEND '{EXTRACTION_BACKEND}'")


def _reclassify_gdpr(statements: list[PPS]) -> list[PPS]:
    """Re-classify GDPR categories for all statements using RoBERTa.

    Runs batch classification on unique source texts, then updates each PPS.
    Falls back to keeping existing categories if RoBERTa is unavailable.
    """
    if not _gdpr_classifier.available:
        return statements
    if not statements:
        return statements

    # Batch classify unique source texts
    unique_texts = list({s.source_text for s in statements if s.source_text})
    if not unique_texts:
        return statements

    categories = _gdpr_classifier.classify_batch(unique_texts)
    text_to_cats = dict(zip(unique_texts, categories))

    # Update each statement
    for stmt in statements:
        if stmt.source_text in text_to_cats:
            stmt.gdpr_categories = text_to_cats[stmt.source_text]

    return statements


def extract_pps_from_policy(
    policy_text: str,
    policy_source: str = "first_party",
    policy_id: str = "policy",
    skip_gdpr_reclassify: bool = False,
) -> list[PPS]:
    """Segment a policy and extract PPS statements for each clause.

    ``skip_gdpr_reclassify``: when True, the final ``_reclassify_gdpr``
    batch (RoBERTa over every PPS source_text) is skipped. Use this
    when the caller is already classifying the segmented clauses for
    an unrelated purpose (e.g. segment-level GDPR coverage) — PPS
    source_text == clause.text in every observed case, so inheriting
    labels from the segment classification is equivalent and halves
    the RoBERTa work on fresh extractions. Cached PPS still carry
    their pre-existing ``gdpr_categories`` from ``data/cache/``."""

    print(f"\n{'=' * 60}")
    print(f"Extracting from: {policy_source}")
    print(f"{'=' * 60}")

    cached = _load_cache(policy_text)
    if cached is not None:
        print(f"  Using cached results ({len(cached)} statements)")
        statements = [PPS.from_dict(item) for item in cached]
        # Statement IDs in the cache embed whatever policy_id was used the
        # first time this text was extracted. If the same policy text is
        # shared across pairs (e.g., a vendor policy used by multiple sites),
        # those foreign IDs would leak into this pair's results and cause
        # cross-pair ID contamination downstream. Re-namespace the IDs to the
        # current policy_id, and also reset policy_source so the cached PPS
        # carry this pair's first_party/third_party labelling.
        for pps in statements:
            old_id = pps.id or ""
            marker = "_c"
            cut = old_id.rfind(marker)
            if cut >= 0:
                pps.id = f"{policy_id}{old_id[cut:]}"
            pps.policy_source = policy_source
        if skip_gdpr_reclassify:
            return statements
        return _reclassify_gdpr(statements)

    print("  Segmenting clauses...")
    clauses = segment_clauses(policy_text, policy_id)
    if not clauses:
        print("  WARNING: No clauses found after segmentation")
        return []

    # Round 9 task #71: filter non-practice clauses BEFORE the LLM extraction
    # pass. The same detector already runs in _build_pps to downgrade
    # PROHIBITION, but those clauses still generate PPS (often COMMITMENT) that
    # feed Π₂/Π₄ as reinforcing-clause false positives. Dropping them upstream
    # prevents both modality misextractions and downstream pattern noise.
    # Only skip clauses with reasons that are confidently non-practice:
    # rights, legal_basis, deletion/retention process text, glossary, and
    # section headers. We deliberately keep 'disclosure' clauses because they
    # often carry genuine practice statements alongside the disclosure framing.
    _SKIP_REASONS = {
        "rights",
        "legal_basis",
        "deletion_process",
        "retention_process",
        "glossary",
        "section_header",
    }
    skipped_by_reason: dict[str, int] = {}
    kept_clauses = []
    for c in clauses:
        is_np, reason = _clause_is_non_practice(c.text)
        if is_np and reason in _SKIP_REASONS:
            skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1
            continue
        kept_clauses.append(c)
    if skipped_by_reason:
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(skipped_by_reason.items()))
        print(f"  Skipped {sum(skipped_by_reason.values())} non-practice clauses ({breakdown})")
    clauses = kept_clauses
    if not clauses:
        print("  WARNING: All clauses filtered as non-practice")
        return []

    print(f"  Extraction backend: {EXTRACTION_BACKEND} ({_active_model_name()})")
    client = _create_extraction_client()

    # Parallelize per-clause extraction with a thread pool. Each clause is
    # independent: cache lookup, optional LLM call, then _build_pps. The
    # llama.cpp server queues concurrent requests to its n_slots, so overrunning
    # the server is safe. Cache hits are pure disk I/O and benefit from
    # threading on lustre (per-file latency dominates). We preserve clause order
    # by writing into a pre-allocated result list indexed by clause position.
    from concurrent.futures import ThreadPoolExecutor

    def _extract_one(idx_clause: tuple[int, "Clause"]) -> tuple[int, list[PPS]]:
        index, clause = idx_clause
        section = clause.section_header or "(no section)"
        prompt = EXTRACTION_PROMPT.format(section=section, clause=clause.text)

        raw_items = _load_clause_cache(section, clause.text)
        cache_hit = raw_items is not None
        if not cache_hit:
            raw_items = []
            for attempt in range(MAX_RETRIES):
                try:
                    if EXTRACTION_BACKEND == "squad":
                        raise RuntimeError(
                            "EXTRACTION_BACKEND='squad' uses an internal "
                            "DeBERTa specialist ensemble that is not shipped "
                            "with this artifact."
                        )
                    elif EXTRACTION_REFLECTION_ENABLED:
                        raw_items = _extract_with_reflection(client, prompt)
                    else:
                        response_text = _call_extraction_model(client, prompt)
                        raw_items = _parse_json_response(response_text)
                    _save_clause_cache(section, clause.text, raw_items)
                    break
                except Exception as exc:
                    delay = INITIAL_BACKOFF * (2**attempt)
                    print(f"    [clause {index}] API error: {exc}. Retrying in {delay:.1f}s")
                    time.sleep(delay)
            # Only sleep when we actually made an LLM call. Squad is local.
            if EXTRACTION_BACKEND != "squad":
                time.sleep(API_CALL_DELAY)

        out: list[PPS] = []
        for item_index, raw in enumerate(raw_items):
            if not isinstance(raw, dict):
                continue
            pps = _build_pps(raw, clause, policy_source, item_index)
            if pps is None:
                continue
            out.append(pps)
        tag = "cache" if cache_hit else "live"
        print(f"  [clause {index}/{len(clauses)}] {tag}: {len(out)} PPS")
        return index, out

    max_workers = int(os.environ.get("EXTRACTION_WORKERS", "8"))
    clause_statements: list[list[PPS]] = [[] for _ in clauses]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for idx, out in pool.map(_extract_one, list(enumerate(clauses, start=1))):
            clause_statements[idx - 1] = out

    statements: list[PPS] = [pps for block in clause_statements for pps in block]

    print(f"\n  Total statements extracted: {len(statements)}")
    _save_cache(policy_text, [statement.to_dict() for statement in statements])

    # Re-classify GDPR categories with RoBERTa. Callers classifying the
    # same segments for a separate purpose (see extract_only_batch) can
    # skip this — PPS source_text == clause.text, so their labels can
    # be inherited from the segment classification instead.
    if skip_gdpr_reclassify:
        return statements
    statements = _reclassify_gdpr(statements)
    return statements
