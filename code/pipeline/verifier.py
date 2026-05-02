"""LLM-based verification of pattern-detected inconsistency candidates.

The pattern-matching step (Π₁, Π₂, Π₃, Π₄) has high recall but imperfect precision.
This module runs a bounded LLM call over each candidate — just the two source
clauses and the pattern's finding — to classify it as:

  inconsistent        — third-party practice contradicts or exceeds the
                        first-party commitment (covers both direct
                        logical contradictions and softer purpose /
                        condition / retention drift)
  unspecified         — evidence is ambiguous; scope or modality signal
                        is unreliable and context is not decisive
  non_conflict        — compatible in context; pattern misfired

Results are cached by (prompt_version, pattern_id, statement_1_id, statement_2_id)
so re-runs are free after the first pass.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

try:
    import anthropic
except ModuleNotFoundError:
    anthropic = None
try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

from .config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_EXTRACTION_MODEL,
    API_CALL_DELAY,
    CACHE_DIR,
    EXTRACTION_BACKEND,
    EXTRACTION_MAX_TOKENS,
    INITIAL_BACKOFF,
    LLAMACPP_BASE_URL,
    LLAMACPP_MODEL_NAME,
    MAX_RETRIES,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    OPENAI_API_KEY,
    OPENAI_EXTRACTION_MODEL,
    VERIFIER_BACKEND,
    VERIFIER_BASE_URL,
    VERIFIER_MODEL_NAME,
    VERIFIER_PROMPT_VERSION,
)

# Resolve which backend the verifier uses
_VERIFIER_BACKEND = VERIFIER_BACKEND or EXTRACTION_BACKEND

# Valid values for verifier output fields. The verifier emits one of three
# labels directly — there is no Stage-2 combination or severity split.
_VALID_VERDICTS = {"inconsistent", "unspecified", "non_conflict"}
_VALID_FALSE_ALARM_CATEGORIES = {
    "none",
    "modality_misextraction",
    "different_contexts",
    "scoped_negation",
    "same_text",
    "section_header",
    "non_pii_mismatch",
    "different_data_types",
    "rights_vs_practice",
    "borderline_drafting",
    "scope_mismatch",
}

# JSON schema for constrained decoding. llama.cpp and OpenAI-compatible servers
# that support `response_format={"type": "json_object"}` will return well-formed
# JSON; we still validate every field ourselves before accepting a response.
_VERIFIER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": sorted(_VALID_VERDICTS)},
        "explanation": {"type": "string", "minLength": 20},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "false_alarm_category": {
            "type": "string",
            "enum": sorted(_VALID_FALSE_ALARM_CATEGORIES),
        },
    },
    "required": ["verdict", "explanation", "confidence", "false_alarm_category"],
}


PATTERN_DESCRIPTIONS: dict[str, str] = {
    "Π₁": (
        "Modality Contradiction — the first party prohibits or limits a practice "
        "but the third party commits to / performs the practice on the same (or "
        "ontology-subsumed) data type. Includes blanket negations violated by "
        "vendor action, purpose-scoped limits exceeded by broader vendor use, "
        "and consent requirements bypassed by by-default vendor processing."
    ),
    "Π₂": (
        "Exclusivity Violation — the first party binds a data type to one "
        "purpose using 'only / solely / exclusively', and the third party uses "
        "the same data for a different-family purpose."
    ),
    "Π₃": (
        "Condition Asymmetry — the same action on the same (or subsumed) data "
        "type is gated by materially different consent conditions between the "
        "two policies (e.g. first party requires explicit consent, third party "
        "processes by default)."
    ),
    "Π₄": (
        "Temporal Contradiction — the two policies give different retention "
        "periods for the same data type, or one specifies a bounded duration "
        "while the other retains indefinitely."
    ),
}

# For patterns that compare two distinct clauses (Π₁, Π₂, Π₃, Π₄).
#
# Decision criteria mirror the strict 5-agent deep-audit protocol in
# data/output/round8_audit_instructions.md. Per-pattern empirical genuine
# rates (now embedded in the prompt body as pattern-stratified priors) were
# measured on the strict v4 deep-audit corpus and replace the former uniform
# ~11.7% default; see commit 618bd18 (2026-04-18).
from prompts.unified_prompts import VERIFIER_PROMPT  # vendored prompt — see prompts/unified_prompts.py

# Round 9: after pruning Π₃ and Π₆, there are no single-clause gap patterns.
# All remaining patterns compare two distinct clauses. GAP_VERIFIER_PROMPT and
# GAP_PATTERNS have been removed along with their code path in verify_candidate.
COMPARISON_PATTERNS = {"Π₁", "Π₂", "Π₃", "Π₄"}


# ---------------------------------------------------------------------------
# Verdict layers
# ---------------------------------------------------------------------------
#
# The pipeline has two verdict layers only:
#   pattern-level : inconsistent | underspecified | non_conflict
#                   (emitted by the structural pattern detectors; drives
#                    severity ranking during curation — Severity.{CRITICAL,
#                    HIGH, MEDIUM, LOW} is the orthogonal intensity field).
#   LLM verifier  : inconsistent | unspecified | non_conflict
#                   (single-stage; the final public finding label.
#                    Previously a two-stage Stage-1/Stage-2 verifier;
#                    collapsed into this three-label single-pass vocabulary.)
#
# The cluster-level Stage-2 verifier and its combination rules
# (strongest / emergent_inconsistency / needs_investigation /
# confirmed_non_conflict / no_cluster*) have been removed. `llm_verdict`
# on an Inconsistency is the authoritative post-verification verdict
# for downstream CSVs, annotation exports, and the paper's aggregate
# statistics.

PUBLIC_VERDICTS = ("inconsistent", "unspecified", "non_conflict")


def _extract_context(full_text: str, clause_text: str, window: int = 600) -> str:
    """Return the surrounding paragraph(s) from full_text around where clause_text appears.

    Falls back to the clause itself if full_text is empty or the clause cannot
    be located. Uses the first 120 characters of clause_text as the match key
    to tolerate minor whitespace/encoding differences.
    """
    clause_text = (clause_text or "").strip()
    if not full_text or not clause_text:
        return clause_text
    needle = clause_text[:120]
    idx = full_text.find(needle)
    if idx < 0:
        # Try a shorter needle with normalized whitespace
        import re as _re
        norm_full = _re.sub(r"\s+", " ", full_text)
        norm_clause = _re.sub(r"\s+", " ", clause_text)[:120]
        idx = norm_full.find(norm_clause)
        if idx < 0:
            return clause_text
        full_text = norm_full
    start = max(0, idx - window)
    end = min(len(full_text), idx + len(clause_text) + window)
    # Expand to paragraph boundaries if possible (double newline or period)
    snippet = full_text[start:end]
    prefix = "... " if start > 0 else ""
    suffix = " ..." if end < len(full_text) else ""
    return f"{prefix}{snippet}{suffix}"


def _resolve_policy_text(
    policy_texts: dict | None, pair_id: str, policy_source: str
) -> str:
    """Look up the full policy text for a given pair and policy source.

    policy_texts has shape {pair_id: {"first_party": "...", "third_party": "..."}}
    or {pair_id: {"website": "...", "vendor": "..."}}. Either key scheme works.
    """
    if not policy_texts or not pair_id:
        return ""
    pair_entry = policy_texts.get(pair_id)
    if not pair_entry:
        return ""
    if "third_party" in policy_source or "vendor" in policy_source:
        return pair_entry.get("third_party") or pair_entry.get("vendor") or ""
    return pair_entry.get("first_party") or pair_entry.get("website") or ""


def _active_verifier_model() -> str:
    """Return a stable identifier for the currently-selected verifier model.

    Used to bind cache entries to (backend, model) so switching models
    invalidates stale verdicts instead of serving them silently — without
    this, swapping ``VERIFIER_MODEL_NAME`` would return the previous
    model's verdicts from cache.
    """
    if _VERIFIER_BACKEND == "anthropic":
        # Verifier shares ANTHROPIC_EXTRACTION_MODEL for the anthropic backend
        # (no dedicated ANTHROPIC_VERIFIER_MODEL knob today).
        return f"anthropic:{ANTHROPIC_EXTRACTION_MODEL}"
    if _VERIFIER_BACKEND == "openai":
        return f"openai:{OPENAI_EXTRACTION_MODEL}"
    if _VERIFIER_BACKEND == "llamacpp":
        return f"llamacpp:{VERIFIER_MODEL_NAME}"
    if _VERIFIER_BACKEND == "openai_compat":
        return f"openai_compat:{LLM_MODEL}"
    return _VERIFIER_BACKEND


def _cache_path(pattern_id: str, s1_id: str, s2_id: str) -> Path:
    key = (
        f"verify:{VERIFIER_PROMPT_VERSION}:{_active_verifier_model()}"
        f":{pattern_id}:{s1_id}:{s2_id}"
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"verify_{digest}.json"


def _load_cache(pattern_id: str, s1_id: str, s2_id: str) -> dict | None:
    path = _cache_path(pattern_id, s1_id, s2_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _save_cache(pattern_id: str, s1_id: str, s2_id: str, result: dict) -> None:
    path = _cache_path(pattern_id, s1_id, s2_id)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)


def _create_client():
    # Per-request HTTP timeout. Default OpenAI-SDK timeout is 600s — when a
    # remote endpoint's upstream socket wedges (seen on the 2026-04-24 random2
    # run: one worker stuck 10 min on a silent TCP read that never came
    # back, heartbeat frozen, 78→79 pair progress stalled), that 600s is
    # the full cost of one hang. Cap at 60s so a hung call raises quickly
    # and the verifier's own retry logic (MAX_RETRIES=3 with exponential
    # backoff) recovers without stalling the pipeline. Applies to every
    # OpenAI-compatible backend (openai, llamacpp, openai_compat).
    _REQUEST_TIMEOUT_S = 60
    if _VERIFIER_BACKEND == "anthropic":
        if anthropic is None:
            raise RuntimeError("anthropic package not installed")
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY, timeout=_REQUEST_TIMEOUT_S,
        )
    if _VERIFIER_BACKEND == "openai":
        if OpenAI is None:
            raise RuntimeError("openai package not installed")
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        return OpenAI(api_key=OPENAI_API_KEY, timeout=_REQUEST_TIMEOUT_S)
    if _VERIFIER_BACKEND == "llamacpp":
        if OpenAI is None:
            raise RuntimeError("openai package not installed (needed for llamacpp backend)")
        # The llamacpp verifier backend reaches a separate llama-server
        # (different port / model) so the extraction server stays unblocked.
        # The paper's chosen verifier is gemma3:27b; reviewers can serve it
        # over the openai_compat backend instead (the default).
        return OpenAI(
            base_url=VERIFIER_BASE_URL, api_key="not-needed",
            timeout=_REQUEST_TIMEOUT_S,
        )
    if _VERIFIER_BACKEND == "openai_compat":
        if OpenAI is None:
            raise RuntimeError("openai package not installed (needed for the openai_compat backend)")
        if not LLM_API_KEY:
            raise RuntimeError("LLM_API_KEY not set")
        return OpenAI(
            base_url=LLM_BASE_URL, api_key=LLM_API_KEY,
            timeout=_REQUEST_TIMEOUT_S,
        )
    raise RuntimeError(f"Unsupported VERIFIER_BACKEND '{_VERIFIER_BACKEND}'")


def _call_model(client, prompt: str) -> str:
    if _VERIFIER_BACKEND == "anthropic":
        response = client.messages.create(
            model=ANTHROPIC_EXTRACTION_MODEL,
            max_tokens=512,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        blocks = [getattr(b, "text", "") for b in response.content]
        return "\n".join(b for b in blocks if b).strip()
    if _VERIFIER_BACKEND == "openai":
        response = client.responses.create(
            model=OPENAI_EXTRACTION_MODEL,
            input=prompt,
            max_output_tokens=512,
        )
        return str(getattr(response, "output_text", "")).strip()
    if _VERIFIER_BACKEND == "llamacpp":
        # JSON mode — llama.cpp's OpenAI-compatible server honours this for
        # recent builds; falls through to free-form if the server rejects it.
        try:
            response = client.chat.completions.create(
                model=VERIFIER_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = client.chat.completions.create(
                model=VERIFIER_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0,
            )
        return response.choices[0].message.content.strip()
    if _VERIFIER_BACKEND == "openai_compat":
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    raise RuntimeError(f"Unsupported VERIFIER_BACKEND '{_VERIFIER_BACKEND}'")


def _parse_verdict_response(text: str) -> dict | None:
    """Parse the LLM's JSON verdict response with fallbacks."""
    content = text.strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "verdict" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    code_match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    if code_match:
        try:
            parsed = json.loads(code_match.group(1))
            if isinstance(parsed, dict) and "verdict" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    obj_match = re.search(r"\{.*\}", content, re.DOTALL)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group(0))
            if isinstance(parsed, dict) and "verdict" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def _validate_verdict_response(parsed: dict) -> tuple[bool, str]:
    """Strict validator for the verifier's JSON response.

    Returns (is_valid, reason). Rejecting invalid responses — instead of
    silently backfilling — is the fix for the Round 9 "rubber stamp" failure:
    previously a missing `explanation` field was quietly replaced by the
    pattern's own auto-generated text, which made every output look LLM-verified
    when it was really a fallback. Now we require every structured field to be
    present, correctly typed, and semantically coherent before we accept it.
    """
    if not isinstance(parsed, dict):
        return False, "not a dict"
    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in _VALID_VERDICTS:
        return False, f"verdict '{verdict}' not in {_VALID_VERDICTS}"
    explanation = str(parsed.get("explanation", "")).strip()
    if len(explanation) < 20:
        return False, f"explanation too short ({len(explanation)} chars, need ≥20)"
    confidence = str(parsed.get("confidence", "")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        return False, f"confidence '{confidence}' invalid"
    category = str(parsed.get("false_alarm_category", "")).strip().lower()
    if category not in _VALID_FALSE_ALARM_CATEGORIES:
        return False, f"false_alarm_category '{category}' invalid"
    # Semantic coherence. Two cases:
    #
    # (1) non_conflict + category=none: the model reached the correct
    #     verdict but didn't fill in the taxonomy slot. Previously we
    #     rejected and retried 3× — on gemma3:27b this burns ~15% of
    #     the per-pair verifier budget because the model consistently
    #     returns "none". Auto-fix to "borderline_drafting" (the
    #     catch-all bucket) and accept.
    #
    # (2) genuine verdict (inconsistent / unspecified) with a specific
    #     category: still a contradiction with itself — reject so the
    #     model retries and either sticks with the category (→ we
    #     override to non_conflict) or drops it (→ accepted).
    if verdict == "non_conflict" and category == "none":
        parsed["false_alarm_category"] = "borderline_drafting"
        return True, ""
    if verdict in {"inconsistent", "unspecified"} and category != "none":
        return False, f"genuine verdict '{verdict}' must have false_alarm_category=none"
    return True, ""


def _policy_label(policy_source: str, pair_id: str) -> str:
    if "third_party" in policy_source or "vendor" in policy_source:
        vendor = pair_id.split("__")[-1].replace("_", ".") if "__" in pair_id else "vendor"
        return f"vendor policy ({vendor})"
    return "website policy"


def _field(statement: dict, key: str, default: str = "unspecified") -> str:
    val = str(statement.get(key, "")).strip()
    return val if val and val.lower() not in {"", "none", "null", "unspecified"} else default


def _build_cluster_narrative(nh_ctx: dict) -> str:
    """Build a natural language narrative from neighborhood context.

    Instead of raw metadata (which LLMs ignore), produce a readable paragraph
    that integrates cluster evidence directly into the verification reasoning.
    """
    data_type = nh_ctx.get("data_type", "this data type")
    w_prohib = nh_ctx.get("website_prohibitions", 0)
    w_commit = nh_ctx.get("website_commitments", 0)
    v_share = nh_ctx.get("vendor_sharing_actions", 0)
    v_proc = nh_ctx.get("vendor_processing_actions", 0)
    bridge = nh_ctx.get("bridge_confirmed", False)
    w_support = nh_ctx.get("website_supporting", [])
    v_support = nh_ctx.get("vendor_supporting", [])

    parts: list[str] = ["=== BROADER POLICY CONTEXT ==="]
    parts.append(
        f"These two clauses are part of a larger pattern about '{data_type}' "
        f"across both policies."
    )

    # Website pattern narrative
    if w_prohib >= 2:
        parts.append(
            f"The website policy contains {w_prohib} SEPARATE prohibitions "
            f"regarding {data_type}. This is not an isolated statement — "
            f"the website REPEATEDLY restricts sharing/selling of this data type."
        )
    elif w_prohib == 1 and w_commit:
        parts.append(
            f"The website has 1 prohibition and {w_commit} commitments "
            f"about {data_type}. The prohibition coexists with permitted uses."
        )

    # Vendor pattern narrative
    if v_share >= 1:
        parts.append(
            f"The vendor policy has {v_share} clause(s) about sharing/selling "
            f"{data_type}, indicating active distribution — not just collection."
        )

    # Data type ontology relationship
    subsume_path = nh_ctx.get("subsume_path")
    w_data = nh_ctx.get("website_data_type", "")
    v_data = nh_ctx.get("vendor_data_type", "")
    if subsume_path and w_data != v_data:
        parts.append(
            f"DATA TYPE RELATIONSHIP: The two clauses reference different data "
            f"types, but they are ontologically related: {subsume_path}. "
            f"In privacy law, a prohibition on a PARENT type (e.g., "
            f"'{w_data}') applies to ALL child types (e.g., '{v_data}'). "
            f"Therefore, if the website prohibits sharing '{w_data}', "
            f"the vendor sharing '{v_data}' (a subtype) IS a violation — "
            f"do NOT dismiss this as 'different data types'."
        )

    # Action subsumption relationship
    w_action = nh_ctx.get("website_action", "")
    v_action = nh_ctx.get("vendor_action", "")
    if w_action and v_action and w_action != v_action:
        from patterns import _action_subsumes
        if _action_subsumes(w_action, v_action):
            parts.append(
                f"ACTION RELATIONSHIP: The website prohibits '{w_action}' and the "
                f"vendor '{v_action}s' data. In privacy law, '{w_action}' is a BROADER "
                f"action that SUBSUMES '{v_action}' — {v_action} is a specific form "
                f"of {w_action}. Therefore, a prohibition on '{w_action}' IS violated "
                f"by '{v_action}'. Do NOT dismiss this as 'different actions'."
            )
        elif _action_subsumes(v_action, w_action):
            parts.append(
                f"ACTION RELATIONSHIP: The website prohibits '{w_action}' (specific) "
                f"and the vendor '{v_action}s' (broader). '{v_action}' subsumes "
                f"'{w_action}', so vendor {v_action}ing does NOT necessarily mean "
                f"they are {w_action}ing — consider whether the vendor's broader "
                f"practice actually includes {w_action}."
            )

    # Bridge confirmation
    if bridge:
        parts.append(
            f"IMPORTANT: A confirmed data flow exists — the website explicitly "
            f"sends {data_type} to this vendor. The vendor demonstrably receives "
            f"this data, making policy contradictions directly relevant."
        )

    # Supporting clauses (actual text, not metadata)
    if w_support:
        parts.append(f"Other relevant website clauses about {data_type}:")
        for s in w_support[:3]:
            # Extract just the source text portion
            text_part = s.split(": ", 1)[-1] if ": " in s else s
            parts.append(f'  > "{text_part}"')

    if v_support:
        parts.append(f"Other relevant vendor clauses about {data_type}:")
        for s in v_support[:3]:
            text_part = s.split(": ", 1)[-1] if ": " in s else s
            parts.append(f'  > "{text_part}"')

    return "\n".join(parts) + "\n"


def verify_candidate(item: dict, client, policy_texts: dict | None = None) -> dict:
    """Call the LLM to verify one candidate. Returns a dict with verdict/explanation/confidence.

    Raises no exceptions — on LLM failure returns a fallback result whose
    verdict mirrors the pattern-level decision (mapped into the new
    inconsistent / unspecified / non_conflict vocabulary).
    After the Round 9 prune, all remaining patterns are two-clause comparisons.

    policy_texts: optional {pair_id: {"first_party": str, "third_party": str}} mapping
    used to extract surrounding context for each clause. When omitted the verifier
    falls back to the extracted source_text only (old behaviour).
    """
    pattern_id = item.get("pattern_id", "")
    s1 = item.get("statement_1", {})
    s2 = item.get("statement_2", {})
    s1_id = s1.get("id", item.get("inconsistency_id", ""))
    s2_id = s2.get("id", s1_id)

    cached = _load_cache(pattern_id, s1_id, s2_id)
    if cached is not None:
        return cached

    pair_id = item.get("pair_id", "")
    pattern_name = item.get("pattern_name", pattern_id)
    pattern_description = PATTERN_DESCRIPTIONS.get(pattern_id, "")
    s1_source = s1.get("policy_source", "")
    s2_source = s2.get("policy_source", "")

    s1_full = _resolve_policy_text(policy_texts, pair_id, s1_source)
    s2_full = _resolve_policy_text(policy_texts, pair_id, s2_source)
    s1_text = s1.get("source_text", "")
    s2_text = s2.get("source_text", "")
    context_1 = _extract_context(s1_full, s1_text) if s1_full else "(full policy text not available — use extracted clause only)"
    context_2 = _extract_context(s2_full, s2_text) if s2_full else "(full policy text not available — use extracted clause only)"
    nh_ctx = item.get("neighborhood_context", {})
    nh_context_str = _build_cluster_narrative(nh_ctx) if nh_ctx else ""

    # Inject supporting clauses directly into the context sections
    # so the verifier can reason about the full cluster evidence
    if nh_ctx:
        w_support = nh_ctx.get("website_supporting", [])
        v_support = nh_ctx.get("vendor_supporting", [])
        if w_support:
            extra_w = "\n\nOther relevant website clauses about this data type:\n"
            for s in w_support[:3]:
                text_part = s.split(": ", 1)[-1] if ": " in s else s
                extra_w += f'  > "{text_part}"\n'
            context_1 = context_1.rstrip() + extra_w
        if v_support:
            extra_v = "\n\nOther relevant vendor clauses about this data type:\n"
            for s in v_support[:3]:
                text_part = s.split(": ", 1)[-1] if ": " in s else s
                extra_v += f'  > "{text_part}"\n'
            context_2 = context_2.rstrip() + extra_v

    prompt = VERIFIER_PROMPT.format(
        pattern_id=pattern_id,
        pattern_name=pattern_name,
        pattern_description=pattern_description,
        policy_1_label=_policy_label(s1_source, pair_id),
        source_text_1=s1_text[:800],
        context_1=context_1[:1800],
        actor_1=_field(s1, "actor"),
        action_1=_field(s1, "action"),
        modality_1=_field(s1, "modality"),
        data_1=_field(s1, "data_object"),
        purpose_1=_field(s1, "purpose"),
        condition_1=_field(s1, "condition"),
        policy_2_label=_policy_label(s2_source, pair_id),
        source_text_2=s2_text[:800],
        context_2=context_2[:1800],
        actor_2=_field(s2, "actor"),
        action_2=_field(s2, "action"),
        modality_2=_field(s2, "modality"),
        data_2=_field(s2, "data_object"),
        purpose_2=_field(s2, "purpose"),
        condition_2=_field(s2, "condition"),
        pattern_explanation=item.get("explanation", "")[:400],
        neighborhood_context=nh_context_str,
    )

    # Retry loop combines API errors and structural-validation failures into
    # a single MAX_RETRIES budget. A model that returns parseable JSON but
    # omits `explanation` or emits non_conflict with category=none is treated
    # the same as an API error — re-prompt, don't accept it.
    last_reason = "no response"
    parsed: dict | None = None
    for attempt in range(MAX_RETRIES):
        raw_text = ""
        try:
            raw_text = _call_model(client, prompt)
        except Exception as exc:
            last_reason = f"API error: {exc}"
            delay = INITIAL_BACKOFF * (2 ** attempt)
            print(f"    Verifier API error (attempt {attempt+1}/{MAX_RETRIES}): {exc}. "
                  f"Retrying in {delay:.1f}s")
            time.sleep(delay)
            continue
        if not raw_text:
            last_reason = "empty response"
            continue
        candidate = _parse_verdict_response(raw_text)
        if candidate is None:
            last_reason = f"unparseable JSON: {raw_text[:120]}"
            print(f"    WARNING: Verifier response unparseable for {pattern_id} {s1_id} "
                  f"(attempt {attempt+1}/{MAX_RETRIES}): {raw_text[:120]}")
            time.sleep(INITIAL_BACKOFF)
            continue
        ok, reason = _validate_verdict_response(candidate)
        if not ok:
            last_reason = f"validation failed: {reason}"
            print(f"    WARNING: Verifier response invalid for {pattern_id} {s1_id} "
                  f"(attempt {attempt+1}/{MAX_RETRIES}): {reason}")
            time.sleep(INITIAL_BACKOFF)
            continue
        parsed = candidate
        break

    if parsed is None:
        # Explicit fallback. Unlike the old code, we do NOT copy the pattern's
        # own explanation into `llm_explanation` — that masked the failure. We
        # mark the row so downstream filters and audits can find rows that
        # never actually saw a valid LLM judgment. The fallback verdict is
        # always `unspecified` — we saw no valid LLM signal, so calling the
        # finding either inconsistent or non_conflict would overclaim.
        fallback = {
            "verdict": "unspecified",
            "explanation": f"[VERIFIER FALLBACK — {last_reason}]",
            "confidence": "low",
            "false_alarm_category": "none",
            "llm_verified": False,
        }
        # Intentionally NOT cached — a future run should retry.
        return fallback

    result = {
        "verdict": str(parsed["verdict"]).strip().lower(),
        "explanation": str(parsed["explanation"]).strip(),
        "confidence": str(parsed["confidence"]).strip().lower(),
        "false_alarm_category": str(parsed["false_alarm_category"]).strip().lower(),
        "llm_verified": True,
    }
    _save_cache(pattern_id, s1_id, s2_id, result)
    time.sleep(API_CALL_DELAY)
    return result




def verify_candidates(
    selected: list[dict],
    max_workers: int = 1,
    policy_texts: dict | None = None,
) -> list[dict]:
    """Verify a list of candidates with the LLM and annotate each with llm_* fields.

    Candidates classified as non_conflict are marked but NOT removed here —
    the caller decides whether to filter them out and backfill.

    Args:
        max_workers: Number of concurrent threads for API calls. Use 1 for
            sequential (default), 4-8 for concurrent remote endpoints (e.g.
            a rented Vast.ai 4× A100 instance behind /v1).
        policy_texts: Optional {pair_id: {"first_party": str, "third_party": str}}
            mapping used to extract surrounding paragraphs for each clause.
            Strongly recommended — matches the deep-audit protocol where the
            auditor reads the full policy, not just the extracted clause.
    """
    print(f"\n[Verifier] Verifying {len(selected)} candidates with LLM "
          f"({_VERIFIER_BACKEND}, {max_workers} workers)...")
    client = _create_client()

    if max_workers <= 1:
        # Sequential mode (original behavior)
        results: list[dict] = []
        for index, item in enumerate(selected, start=1):
            pattern_id = item.get("pattern_id", "?")
            s1 = item.get("statement_1", {})
            s1_id = s1.get("id", item.get("inconsistency_id", ""))
            cache_label = "(cached)" if _load_cache(pattern_id, s1_id, s1_id) else ""
            print(f"  [{index}/{len(selected)}] {pattern_id} {item.get('pair_id','')} {cache_label}")
            verification = verify_candidate(item, client, policy_texts=policy_texts)
            annotated = dict(item)
            annotated["llm_verdict"] = verification["verdict"]
            annotated["llm_explanation"] = verification["explanation"]
            annotated["llm_confidence"] = verification["confidence"]
            annotated["llm_false_alarm_category"] = verification.get("false_alarm_category", "none")
            # Safe default: if the verification dict somehow lacks this key,
            # treat as NOT verified. Prior True-default masked malformed
            # fallbacks as genuine findings in downstream counts.
            annotated["llm_verified"] = bool(verification.get("llm_verified", False))
            results.append(annotated)
    else:
        # Concurrent mode — thread pool for I/O-bound API calls
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        _lock = threading.Lock()
        _done = [0]

        def _verify_one(item: dict) -> dict:
            verification = verify_candidate(item, client, policy_texts=policy_texts)
            annotated = dict(item)
            annotated["llm_verdict"] = verification["verdict"]
            annotated["llm_explanation"] = verification["explanation"]
            annotated["llm_confidence"] = verification["confidence"]
            annotated["llm_false_alarm_category"] = verification.get("false_alarm_category", "none")
            # Safe default: if the verification dict somehow lacks this key,
            # treat as NOT verified. Prior True-default masked malformed
            # fallbacks as genuine findings in downstream counts.
            annotated["llm_verified"] = bool(verification.get("llm_verified", False))
            with _lock:
                _done[0] += 1
                if _done[0] % 10 == 0 or _done[0] == len(selected):
                    print(f"  [{_done[0]}/{len(selected)}] verified...", flush=True)
            return annotated

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_verify_one, item): i
                       for i, item in enumerate(selected)}
            # Collect in submission order
            ordered = [None] * len(selected)
            for future in as_completed(futures):
                idx = futures[future]
                ordered[idx] = future.result()
            results = ordered

    non_conflict_count = sum(1 for r in results if r["llm_verdict"] == "non_conflict")
    keeping = len(results) - non_conflict_count
    print(f"[Verifier] Stage 1 done. non_conflict: {non_conflict_count}, keeping: {keeping}")

    return results
