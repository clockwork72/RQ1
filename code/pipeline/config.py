"""Configuration for PoliReasoner."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    load_dotenv = None

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "output"
BENCHMARK_DIR = DATA_DIR / "benchmarks"

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

# Ensure directories exist
for d in [INPUT_DIR, CACHE_DIR, OUTPUT_DIR, BENCHMARK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# API backends
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_EXTRACTION_BACKEND = "anthropic" if ANTHROPIC_API_KEY else "openai"
EXTRACTION_BACKEND = os.environ.get("EXTRACTION_BACKEND", DEFAULT_EXTRACTION_BACKEND).strip().lower()

# Model config
ANTHROPIC_EXTRACTION_MODEL = os.environ.get("ANTHROPIC_EXTRACTION_MODEL", "claude-sonnet-4-20250514")
OPENAI_EXTRACTION_MODEL = os.environ.get("OPENAI_EXTRACTION_MODEL", "gpt-5.2-2025-12-11")

# llama.cpp / local backend (OpenAI-compatible API served by llama-server)
LLAMACPP_BASE_URL = os.environ.get("LLAMACPP_BASE_URL", "http://localhost:8930/v1")
LLAMACPP_MODEL_NAME = os.environ.get("LLAMACPP_MODEL_NAME", "qwen25-72b")

# OpenAI-compatible HTTP backend.
# Either a local server on a 2× A100 box (see llm_serving/serve_vllm.sh) or a
# rented 4× A100 instance on Vast.ai exposing the same /v1 endpoint.
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:27b")

# Verifier config — defaults to Qwen2.5-14B-Instruct on its own llama-server.
# Qwen2.5-14B won the in-house verification benchmark (87% acc, balanced F1
# across hard_contradiction / soft_tension / non_conflict); see
# data/benchmarks/eval_verification_qwen14b_v2.json. Non-reasoning per project
# constraint. Runs on a separate llama-server (different port) from extraction.
VERIFIER_BACKEND = os.environ.get("VERIFIER_BACKEND", "llamacpp").strip().lower() or "llamacpp"
VERIFIER_BASE_URL = os.environ.get("VERIFIER_BASE_URL", "http://localhost:8931/v1")
VERIFIER_MODEL_NAME = os.environ.get("VERIFIER_MODEL_NAME", "qwen25-14b")

EXTRACTION_TEMPERATURE = 0.0
EXTRACTION_MAX_TOKENS = 4096

# ─────────────────────────────────────────────────────────────────────────
# Pattern feature flags (ablation switches)
# ─────────────────────────────────────────────────────────────────────────

# P4: when True, Π₈'s pair check allows "same-group" action fall-through —
# pairs like (share, transfer) or (collect, use) proceed to evaluation even
# when strict subsumption does not hold. Rationale: extractors occasionally
# pick "transfer" for a clause a policy author would call "share". Set to
# "0" (False) to require strict _action_subsumes in both the general pair
# gate and the prohibition arm of Π₈ — used by the ablation scripts to
# measure the precision/recall trade-off this rule introduces.
PI8_ALLOW_SAME_GROUP_BYPASS = os.environ.get(
    "PI8_ALLOW_SAME_GROUP_BYPASS", "1"
) not in ("0", "false", "False", "")
# Bumped 2026-04-18: rule numbering 1-12 sequential + 3 few-shot examples
# (scoped negation, user rights, consent-gated variant). Forces cache miss.
EXTRACTION_PROMPT_VERSION = "2026-04-18-a"
# Bumped 2026-04-18: stratified per-pattern genuine-rate prior replaces the
# uniform ~12% figure; decision rules reordered so scope + rule-10 (public vs
# private) sit before the verdict block; three few-shot examples appended
# covering hard_contradiction / non_conflict-modality_misextraction /
# soft_tension. Forces cache miss on the Stage-1 verifier.
# Bumped 2026-04-24: prompt rewritten for cross-policy-only framing (all four
# patterns are now cross-party; the former intra/cross split is gone). Π₁–Π₄
# framed as first-party / third-party audit, pattern-specific guidance
# simplified, `different_contexts` FP category folded into `scope_mismatch`
# (rule 2 renamed to PRODUCT/AUDIENCE MISMATCH).
# Bumped 2026-04-24 (b): empirical-rate lines removed from the skepticism
# block — only MODERATE default / HIGH when ambiguous remains.
# Bumped 2026-04-24 (c): verdict vocabulary collapsed to three labels —
# the LLM now emits one of {inconsistent, unspecified, non_conflict}
# directly. The previous hard_contradiction / soft_tension split is
# gone; Stage-2 cluster verification is gone entirely (see
# verifier.py history for the deleted CLUSTER_VERIFY_PROMPT /
# verify_cluster / verify_clusters_only code paths). Forces a cache
# miss on all Stage-1 verdicts.
# Bumped 2026-04-24 (d): DATA TYPE HIERARCHY block added to the prompt
# and rule 7 "different_data_types" narrowed so the LLM accepts parent→
# child subsumption pairs as candidates for inconsistency rather than
# defaulting to non_conflict. Unlocks Π₁ / Π₃ cross findings where FP
# names a parent type and TP a child type (or vice versa). Forces a
# cache miss on every per-call verdict so gemma3:27b re-adjudicates
# under the corrected guidance.
VERIFIER_PROMPT_VERSION = "2026-04-24-d"
# Added 2026-04-18: scope classifier prompt now has a default-global preamble,
# canonical vocabulary (3 example buckets), and three few-shot examples.
SCOPE_PROMPT_VERSION = "2026-04-18-a"
# Retired 2026-04-24: Stage-2 cluster verification was removed from the
# pipeline. The verifier now returns the final 3-label verdict directly
# in a single Stage-1 pass. This constant is kept only so historical
# result JSONs that stored the Stage-2 prompt version in repro_meta can
# still be loaded; no new write uses it.
CLUSTER_VERIFY_PROMPT_VERSION = "retired-2026-04-24"
# Added 2026-04-19: version string for extractor._build_pps post-processing logic
# (modality cascade, PROHIBITION validation, recipient normalization, non-practice
# pre-filter, GDPR reclassification). Folded into both extraction cache keys so
# changes to this logic invalidate stale caches. Bump whenever extractor.py's
# _build_pps / _infer_modality / _validate_prohibition / _normalize_recipient /
# _SKIP_REASONS / _reclassify_gdpr behaviour changes.
BUILD_PPS_VERSION = "2026-04-19-a"

# Per-pattern version strings. Bump the individual pattern's entry
# whenever its cross-policy arm's trigger, gate, or dedup logic
# changes (e.g. tightening Π₃'s strictness-gap floor, adding Π₄'s
# "as long as / as necessary" filter). The runner stamps this whole
# dict into every <pair_id>_results.json, so a bump on Π₃ alone
# invalidates exactly the pairs whose recorded pi3 version differs,
# while pi1/pi2/pi4 findings on unchanged pairs stay valid. Verifier
# per-call cache (data/cache/verify_*.json) is orthogonal — its key
# is (VERIFIER_PROMPT_VERSION, model, pattern_id, s1_id, s2_id) —
# so pattern re-runs re-use verdicts for candidates we've already
# adjudicated and only pay the LLM cost on genuinely new candidates.
PATTERN_VERSIONS = {
    "pi1": "2026-04-24-a",
    "pi2": "2026-04-24-a",
    # pi3: -c restores subsumption-aware data gate (-b dropped it), keeps
    # PI4_STRICTNESS_GAP_MIN=4, and populates subsume_path on findings so
    # the verifier's DATA TYPE RELATIONSHIP block renders — paired with
    # VERIFIER_PROMPT_VERSION -d which teaches the verifier about the
    # ontology.
    "pi3": "2026-04-24-c",
    # pi4: -c narrows the open-ended-temporality filter to only drop
    # INDEFINITE + open-ended SPECIFIC pairs (-b dropped mixed
    # numeric-vs-open-ended cases that can be real conflicts).
    "pi4": "2026-04-24-c",
}

# Rate limiting
API_CALL_DELAY = 0.3          # seconds between API calls
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0         # seconds, doubles on each retry

# Extraction
MAX_CLAUSE_LENGTH = 2000      # characters per clause

# Iterative reflection (PoliGrapher-LM style) — always enabled, integral to pipeline quality
EXTRACTION_REFLECTION_ROUNDS = int(os.environ.get("EXTRACTION_REFLECTION_ROUNDS", "3"))
EXTRACTION_REFLECTION_ENABLED = True  # Mandatory — do not disable
MIN_CLAUSE_LENGTH = 20        # skip very short clauses
