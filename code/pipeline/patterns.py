"""Inconsistency detection patterns for PoliReasoner."""

from __future__ import annotations

import functools
import itertools
import re

import networkx as nx

from .config import PI8_ALLOW_SAME_GROUP_BYPASS
from .graph import extract_statements_from_graph
from .normalizer import (
    CANONICAL_PURPOSES,
    data_subsumes,
    data_types_related,
    get_purpose_necessity,
    normalize_data_type,
    normalize_purpose,
)

try:
    from bert_extraction.e2e_audit.scripts import debug_dump as _dd
except Exception:  # pragma: no cover - debug is optional
    _dd = None


def _trace(pattern: str, event: dict) -> None:
    """Record a pattern-candidate trace event when PIPELINE_DEBUG=1."""
    if _dd is None or not _dd.debug_enabled():
        return
    _dd.record_trace(_dd.current_pair(), pattern, event)
from .schema import (
    ACTION_SUBSUMPTION,
    CONDITION_STRICTNESS,
    ConditionType,
    GDPR_CATEGORIES,
    Inconsistency,
    Modality,
    PPS,
    Severity,
    TemporalityType,
    Verdict,
)


PROCESSING_ACTIONS = {"collect", "use", "share", "sell", "retain", "transfer", "process"}
RIGHTS_ACTIONS = {"deletion_right", "access_right", "optout_right", "portability_right"}
_SHARING_ACTIONS = {"share", "sell", "transfer"}
_USAGE_ACTIONS = {"collect", "use", "process"}


# ─────────────────────────────────────────────────────────────────────────
# Pattern thresholds — justification & sensitivity (P5)
# ─────────────────────────────────────────────────────────────────────────
#
# These constants were tuned on the held-out Round 8 audit corpus
# (data/output/annotation_round8.csv, n=250) and validated against the
# strict v4 deep-audit verdicts. For a full sensitivity sweep see
# scripts/ablation_thresholds.py — it reports P/R/F1 as a function of
# each threshold so paper readers can verify the choices.

# Π₂ — cap findings per exclusive clause to prevent fan-out explosion
# when a single "only for X" clause has many conflicting purposes.
# Sensitivity: raising to 5 increases finding count by ~35% but the
# extra findings are near-duplicates (same exclusive clause, different
# conflicting purpose). 3 is the knee of the precision curve.
PI2_MAX_PER_EXCLUSIVE_CLAUSE = 3

# Condition strictness gap (CONDITION_STRICTNESS scale: by_default=1 …
# upon_consent=5). Used by Π₃ (cross-policy) and retained for any
# intra-path. Raised from 3 to 4 on 2026-04-24: at 3 we paired
# upon_consent (5) with unless_opted_out (3), which the LLM verifier
# routinely read as "both user-controlled, different drafting", not a
# contradiction (Π₃ inc% was ~17%). At 4 the loose side must be
# by_default (1) or when_required (2), which is what actually reads
# as a conflict. Halves Π₃ candidate volume, expected inc% climbs to
# ~35-50% range.
PI4_STRICTNESS_GAP_MIN = 4

# Π₅ — ratio of two specific retention windows before flagging.
# 2× is the standard forensic threshold for "materially different"
# (14 days vs 28 days is a different kind of promise than 14 days vs
# 20 days, which could be rounding). Relaxed to 1.5× doubles FPs
# on weekly-vs-monthly rewordings.
PI5_DURATION_RATIO_MIN = 2.0

# Π₅ — days per unit in duration parsing. 30 is the GDPR convention
# (Art. 12: "one month" = 30 days); matches CCPA "45 days" norm. 365
# for "year" follows the same convention. See _parse_duration_days.
PI5_HOUR_DAYS = 1 / 24
PI5_DAY_DAYS = 1
PI5_WEEK_DAYS = 7
PI5_MONTH_DAYS = 30
PI5_YEAR_DAYS = 365

# Π₈ — cap findings per website sentence for the same reason as Π₂:
# one blanket "we do not sell personal data" should not explode into
# 12 near-duplicate vendor-side findings.
PI8_MAX_PER_WEBSITE_SENTENCE = 3

# Π₈ — condition arm fires only when website strictness is ≥3 AND
# exceeds vendor strictness. Rationale: conditions below 3 are not
# restrictive enough to imply a commitment (by_default / when_required
# are informational; unless_opted_out / if_opted_in / upon_consent
# are the real guards). Lowering to 2 pulls in "when_required" which
# is typically "when required by law" — not a commitment.
PI8_WEBSITE_STRICTNESS_MIN = 3


# P3: blanket-prohibition marker regex. Used by Π₈ to recognize a website
# clause that absolutely prohibits a practice on a data type, regardless of
# purpose. The previous implementation was 13 hardcoded substrings with a
# colloquial filler ("full stop") and no variant coverage for "shall not",
# "forbidden to", "prohibited from", "rent/disclose/trade" verbs.
#
# Compiled once at module load (avoids per-call recompile). Matches
# word-boundary-aware so "never sellers" won't false-positive on "never sell".
_BLANKET_PROHIBITION_RE = re.compile(
    r"\b(?:"
    # "never sell / share / transfer / disclose / rent / trade"
    r"never\s+(?:sell|share|transfer|disclose|rent|trade)"
    # "will never sell / share / transfer / disclose / rent / trade / use"
    r"|will\s+never\s+(?:sell|share|transfer|disclose|rent|trade|use)"
    # "do/does/will/would not sell / share / transfer / disclose / rent / trade"
    r"|(?:do|does|will|would)\s+not\s+(?:sell|share|transfer|disclose|rent|trade)"
    # "shall not sell / share / transfer / disclose / rent / trade"
    r"|shall\s+not\s+(?:sell|share|transfer|disclose|rent|trade)"
    # "is/are forbidden to X" / "is/are prohibited to/from X"
    r"|(?:is|are)\s+(?:forbidden|prohibited)\s+(?:to|from)\s+(?:sell|share|transfer)"
    # "forbidden/prohibited to/from X"
    r"|(?:forbidden|prohibited)\s+(?:to|from)\s+(?:sell|share|transfer)"
    # Absolute phrases
    r"|under\s+no\s+circumstances"
    r"|in\s+no\s+event"
    r"|strictly\s+prohibit(?:s|ed|ing)?"
    # "(no|zero) (exchange|sale|sharing|transfer|disclosure) of (personal)? data/info"
    r"|(?:no|zero)\s+(?:exchange|sale|sharing|transfer|disclosure)\s+of\s+"
    r"(?:personal\s+)?(?:data|information)"
    # "to any third party / parties" (directly asserts universal scope)
    r"|to\s+any\s+third\s+part(?:y|ies)"
    # "with no exception"
    r"|with\s+no\s+exception"
    r")\b",
    re.IGNORECASE,
)


def is_blanket_prohibition(source_text: str) -> bool:
    """Return True if the source clause matches any blanket-prohibition
    marker. Exposed at module level so verifier + ablation scripts can
    reuse the same definition."""
    if not source_text:
        return False
    return bool(_BLANKET_PROHIBITION_RE.search(source_text))


# P1: recipient restrictiveness — phrases that do NOT name a specific vendor.
# A website clause like "we share email with third parties" is NOT restricting
# to a named recipient, so Π₈ should fall through to the normal pair check.
_GENERIC_RECIPIENTS = {
    "third parties", "third party", "third-parties",
    "partners", "advertising partners", "marketing partners",
    "processors", "subprocessors", "trusted partners",
    "service providers", "our partners", "our service providers",
    "business partners", "vendors",
}

# P1: exclusivity markers on the source text that turn a specific recipient
# into a RESTRICTION (as opposed to an informational mention).
_RECIPIENT_RESTRICTION_MARKERS = (
    "only with", "solely with", "exclusively with",
    "only to", "solely to", "exclusively to",
    "only share", "only disclose", "only transfer",
)


def _recipient_is_restrictive(website_statement: PPS) -> bool:
    """True when the website clause names a specific recipient AND uses
    exclusivity language. 'We share email only with Google Analytics' is
    restrictive; 'we share email with partners' is not."""
    r = (website_statement.recipient or "").strip().lower()
    if not r or r in _GENERIC_RECIPIENTS:
        return False
    text_lower = website_statement.source_text.lower()
    return any(m in text_lower for m in _RECIPIENT_RESTRICTION_MARKERS)


def _recipient_names_vendor(
    website_recipient: str, vendor_actor: str, vendor_name: str | None,
) -> bool:
    """Return True when the website's recipient string clearly names the
    vendor side of the pair. Substring match both directions so 'Google LLC'
    resolves to 'Google Analytics' and vice versa."""
    wr = (website_recipient or "").strip().lower()
    if not wr:
        return True  # no restriction to fail against
    va = (
        (vendor_actor or "")
        .strip()
        .lower()
        .replace("thirdparty:", "")
        .replace("third_party:", "")
    )
    vn = (vendor_name or "").strip().lower()
    if vn and (wr in vn or vn in wr):
        return True
    if va and (wr in va or va in wr):
        return True
    return False

# Fix 10: Non-PII / anonymized data phrases — vendor statements about these
# cannot contradict website commitments about personal data.
_NON_PII_PHRASES = (
    "non-personally identif", "non personally identif",
    "not personally identif", "non-personally-identifying",
    "aggregate", "anonymized", "anonymised", "de-identified",
    "deidentified", "pseudonymized", "pseudonymised",
    "non-personal information", "non personal information",
)

_PUBLIC_DATA_RE = re.compile(
    r"\b(?:public(?:ly)?\s+(?:personal\s+)?(?:data|information|profile|content|review)"
    r"|(?:data|information|content|review)\s+(?:you\s+)?(?:made|chose\s+to\s+make|voluntarily)\s+public"
    r"|publicly\s+(?:available|visible|accessible|posted|shared|published))",
    re.IGNORECASE,
)


def _is_public_data_statement(source_text: str) -> bool:
    """Detect statements about publicly posted / voluntarily published data."""
    return bool(_PUBLIC_DATA_RE.search(source_text))


_USER_DIRECTED_RE = re.compile(
    r"\b(?:you\s+(?:must|shall|should|will|can|agree\s+to|may)\s+(?:not\s+)?(?:disclose|share|use|distribute|sell|transfer|collect)"
    r"|you\s+(?:are\s+(?:not|prohibited|forbidden)|(?:must|shall|can)\s+not)\s)"
    r"|\byou\s+will\s+not\b"
    r"|\bcan\s+you\s+disclose\b",
    re.IGNORECASE,
)


def _is_user_directed_prohibition(statement: PPS) -> bool:
    """Detect prohibitions directed at the user, not at the first party."""
    if statement.policy_source != "first_party":
        return False
    if statement.modality != Modality.PROHIBITION and not statement.is_negative:
        return False
    return bool(_USER_DIRECTED_RE.search(statement.source_text))


def _is_non_pii_statement(source_text: str, statement: PPS | None = None) -> bool:
    """Check if source text explicitly describes non-personal / anonymized data.

    The non-PII markers can appear as a side-note in a clause whose PPS is
    actually a PII prohibition (e.g., "We will never sell your email address…
    We may share aggregated usage statistics"). When the passed statement is
    itself a blanket prohibition on identifier-bearing data, the non-PII
    guard is skipped — the presence of a non-PII aside does not invalidate
    the prohibition.
    """
    lowered = source_text.lower()
    matched = any(phrase in lowered for phrase in _NON_PII_PHRASES)
    if not matched:
        return False
    if statement is not None:
        # Concrete-PII rescue: if the statement's OWN data_object normalizes
        # to a specific identifier, the non-PII aside (aggregate/anonymised
        # phrasing elsewhere in the same clause) must not neutralise the
        # extracted practice. Previously applied only to prohibitions;
        # permissive/commitment PPS with concrete PII data_objects were
        # still dropped by the guard.
        identifier_data_types = {
            "email address", "phone number", "postal address",
            "name", "full name", "ip address", "device id",
            "advertising id", "social security number", "credit card",
            "payment information", "password",
        }
        norm = normalize_data_type(statement.data_object) or statement.data_object.lower().strip()
        if norm in identifier_data_types:
            return False
    return matched


def _scope_compatible(s1: PPS, s2: PPS, cross_policy: bool = False) -> bool:
    """Scope gate (R10, 2026-04-12).

    Two PPS are compared only if their scopes are compatible. If either scope
    is ``global`` (the default for unscoped clauses) or the two scopes match
    exactly, the pair is compatible and the pattern may fire. Otherwise the
    pair is skipped — the clauses apply to different audiences, products, or
    jurisdictions and cannot logically contradict.

    For cross_policy=True (Π₈): a non-global WEBSITE scope (e.g.,
    "donation_processing") is NOT compatible with a global VENDOR scope.
    The website is making a statement about a specific feature, not about
    the vendor's general practices. But a global website prohibition
    IS compatible with any vendor scope (blanket prohibition covers all).
    """
    a = (s1.scope or "global").strip().lower() or "global"
    b = (s2.scope or "global").strip().lower() or "global"
    if a == b:
        return True
    if cross_policy:
        # s1 = website, s2 = vendor in Π₈
        # Website non-global + vendor global → NOT compatible
        # Website global + vendor anything → compatible
        if a != "global" and b == "global":
            return False  # feature-scoped website, generic vendor
        if a == "global":
            return True  # blanket website prohibition covers all
        return False  # different non-global scopes
    if a == "global" or b == "global":
        return True
    return False


# Round 3: children's/COPPA clauses are age-scoped, not blanket prohibitions.
_CHILDREN_RE = re.compile(
    r"(?:children\s+(?:under|below)\s+(?:the age of\s+)?\d+"
    r"|(?:do not|does not)\s+(?:knowingly\s+)?(?:collect|solicit).*(?:children|minors|under\s+\d+)"
    r"|COPPA|Children'?s Online Privacy"
    r"|not\s+(?:intended|directed|designed)\s+(?:for|to)\s+(?:children|minors)"
    r"|(?:persons?|individuals?|users?)\s+under\s+(?:the age of\s+)?\d+)",
    re.IGNORECASE,
)


def _is_childrens_clause(source_text: str) -> bool:
    """Detect age-scoped children's/COPPA/minors clauses."""
    return bool(_CHILDREN_RE.search(source_text))
BROAD_DATA_TYPES = {
    "personal data",
    "contact information",
    "device information",
    "behavioral data",
    "financial data",
    "health data",
    "biometric data",
    "geolocation",
}
# Round 9: Π₃, Π₆, Π₉, Π₁₀ removed per Dr. Chen's expert review (weak verdicts).
# Π₃/Π₆ are single-statement compliance checks, not inconsistencies.
# Π₉/Π₁₀ had statistically insignificant yields (n≤5) and high false-positive rates.
# Historical result JSONs containing these pattern_ids still deserialize fine —
# the strings are stored as plain text, not enums.
PATTERN_IDS = {
    "pi1": "Π₁",   # Modality Contradiction (merged Π₁ intra + former Π₈ cross)
    "pi2": "Π₂",   # Exclusivity Violation
    "pi3": "Π₃",   # Condition Asymmetry (was Π₄)
    "pi4": "Π₄",   # Temporal Contradiction (was Π₅)
}


def _coerce_statements(source: list[PPS] | nx.MultiDiGraph) -> list[PPS]:
    if isinstance(source, nx.MultiDiGraph):
        return extract_statements_from_graph(source)
    return list(source)


def _coerce_cross_policy_statements(
    website_source: list[PPS] | nx.MultiDiGraph,
    vendor_source: list[PPS] | nx.MultiDiGraph | None = None,
) -> tuple[list[PPS], list[PPS]]:
    if isinstance(website_source, nx.MultiDiGraph) and vendor_source is None:
        return (
            extract_statements_from_graph(website_source, "website"),
            extract_statements_from_graph(website_source, "vendor"),
        )
    if vendor_source is None:
        raise ValueError("vendor_source is required unless a merged graph is provided")
    return _coerce_statements(website_source), _coerce_statements(vendor_source)


def _is_merged_graph(source) -> bool:
    """Return True if `source` is a merged website+vendor graph.

    Merged graphs carry a `vendor_name` attribute set by `merge_graphs`,
    which is how patterns distinguish single-policy graphs from cross-policy
    merges when dispatching intra vs cross logic.
    """
    return isinstance(source, nx.MultiDiGraph) and bool(source.graph.get("vendor_name"))


def _is_cross_policy_call(
    website_source: list[PPS] | nx.MultiDiGraph,
    vendor_source: list[PPS] | nx.MultiDiGraph | None,
) -> bool:
    """Dispatch gate for pi1/pi2/pi3/pi4 — True when the caller wants
    cross-policy semantics (explicit vendor side or a merged graph)."""
    if vendor_source is not None:
        return True
    return _is_merged_graph(website_source)


@functools.lru_cache(maxsize=4096)
def _side_from_policy_source(src: str) -> str:
    """Pure-string memoised side lookup so inner loops never re-parse
    the policy_source string on every comparison."""
    s = (src or "").lower()
    if s.startswith("third_party") or s.startswith("vendor") or s.startswith("3p"):
        return "vendor"
    if s.startswith("first_party") or s.startswith("website") or s.startswith("1p"):
        return "website"
    return "unknown"


def _pps_side(statement: PPS) -> str:
    """Classify a PPS by its originating policy side.

    Returns 'website' for first-party statements, 'vendor' for third-party
    statements, and 'unknown' when the policy_source string doesn't match
    either convention (pipeline code uses "first_party", "third_party:<name>",
    "website", "vendor", or "1p"/"3p" depending on the origin).
    """
    return _side_from_policy_source(statement.policy_source)


def _is_cross_policy_pair(a: PPS, b: PPS) -> bool:
    """True when the two PPS come from different policy sides."""
    sa, sb = _pps_side(a), _pps_side(b)
    if sa == "unknown" or sb == "unknown":
        return False
    return sa != sb


def _filter_cross_findings(findings: list[Inconsistency]) -> list[Inconsistency]:
    """Keep only findings whose two statements originate on different sides."""
    return [f for f in findings if _is_cross_policy_pair(f.statement_1, f.statement_2)]



def _is_hedged_or_scoped_prohibition(statement: PPS) -> bool:
    """Return True if the statement is a conditional/hedged negation, not a blanket prohibition.

    Blanket prohibitions have unconditional scope: "we do not sell", "we will never share".
    Scoped/hedged negations should not fire Π₁ because they are not conflicting commitments —
    they are carve-outs or definitional statements that coexist with permissive clauses.
    """
    text = statement.source_text.lower()
    # Hedged modal negations — "generally will not" is an assurance, not a prohibition
    if any(
        phrase in text
        for phrase in (
            "generally",
            "normally",
            "typically",
            "as a rule",
            "in most cases",
            "where possible",
            "where applicable",
        )
    ):
        return True
    # Anonymous/anonymized data — "no personal data is collected" is definitional, not a prohibition
    if any(word in text for word in ("anonymous", "anonymously", "anonymized", "pseudonymized")):
        return True
    # Children's data carve-outs — age-scoped, not a blanket prohibition on the data type
    if any(
        phrase in text
        for phrase in ("under 13", "under 16", "children", "minors", "child", "under the age")
    ):
        return True
    # Law enforcement / legal process — these statements explicitly *allow* disclosure under conditions;
    # they are not prohibitions and the Π₁ logic inverting them produces false contradictions
    if any(
        phrase in text
        for phrase in (
            "law enforcement",
            "judicial",
            "subpoena",
            "court order",
            "required by law",
            "cooperat",
            "legal process",
        )
    ):
        return True
    # "We do not knowingly" — knowingly scopes the negation, making it conditional
    if "knowingly" in text:
        return True
    return False


def _actions_related(first: str, second: str) -> bool:
    if first == second:
        return True
    parent = ACTION_SUBSUMPTION.get(first)
    while parent:
        if parent == second:
            return True
        parent = ACTION_SUBSUMPTION.get(parent)
    parent = ACTION_SUBSUMPTION.get(second)
    while parent:
        if parent == first:
            return True
        parent = ACTION_SUBSUMPTION.get(parent)
    return False


def _actors_comparable(first: PPS, second: PPS) -> bool:
    normalize = lambda value: re.sub(r"[^a-z0-9]+", "", value.lower())
    left = normalize(first.actor)
    right = normalize(second.actor)
    if left == right:
        return True
    return left in {"firstparty", "thirdparty"} or right in {"firstparty", "thirdparty"}


def _actor_key(statement: PPS) -> str:
    return re.sub(r"[^a-z0-9]+", "", statement.actor.lower())


def _is_prohibitive(statement: PPS) -> bool:
    """Unified polarity check: statement is a prohibition on either axis."""
    return statement.is_negative or statement.modality == Modality.PROHIBITION


def _is_permissive(statement: PPS) -> bool:
    """Unified polarity check: statement permits/commits to the practice."""
    if statement.is_negative:
        return False
    return statement.modality in {
        Modality.PERMISSION,
        Modality.COMMITMENT,
        Modality.OBLIGATION,
    }


@functools.lru_cache(maxsize=65536)
def _data_same_or_subsuming(a: str, b: str) -> bool:
    """Strict data-type match: equal canonicals or one subsumes the other.

    Unlike data_types_related, this does NOT accept ontology siblings — it is
    the safe gate for patterns where sibling comparisons produce FPs
    (Π₂ exclusivity, retention). Memoised because inner loops call it
    O(N²) times per pair with a small set of distinct (data_a, data_b)
    arguments.
    """
    left = normalize_data_type(a) or (a or "").lower().strip()
    right = normalize_data_type(b) or (b or "").lower().strip()
    if not left or not right:
        return False
    if left == right:
        return True
    return data_subsumes(left, right) or data_subsumes(right, left)


def _statement_sort_key(statement: PPS) -> tuple[int, int, str]:
    return (len(statement.data_object), len(statement.source_text), statement.id)


def _get_subsume_path(parent: str, child: str) -> list[str]:
    """Find the ontology path from parent to child via DATA_ONTOLOGY."""
    from normalizer import DATA_ONTOLOGY
    if parent == child:
        return [parent]
    children = DATA_ONTOLOGY.get(parent, [])
    for c in children:
        path = _get_subsume_path(c, child)
        if path:
            return [parent] + path
    return []


@functools.lru_cache(maxsize=4096)
def _action_subsumes(parent: str, child: str) -> bool:
    """Check if parent action is broader than child in the subsumption hierarchy.

    sell → share → transfer → process
    collect → process
    use → process

    A prohibition on a parent action covers all child actions.
    """
    if parent == child:
        return True
    current = child
    while current in ACTION_SUBSUMPTION:
        current = ACTION_SUBSUMPTION[current]
        if current == parent:
            return True
    return False


@functools.lru_cache(maxsize=65536)
def _cross_policy_data_match(website_data: str, vendor_data: str) -> bool:
    # Cross-policy Π₁ data gate: accept both directions of subsumption so a
    # website child type (e.g., "email address") still matches a vendor
    # parent type (e.g., "personal data"). The previous one-way check made
    # the flat PPS-list API miss pairs that the merged-graph path caught
    # through alignment.
    website_norm = normalize_data_type(website_data)
    vendor_norm = normalize_data_type(vendor_data)
    if not website_norm or not vendor_norm:
        return False
    if website_norm == vendor_norm:
        return True
    return data_subsumes(website_norm, vendor_norm) or data_subsumes(vendor_norm, website_norm)


def _canonical_purpose(raw_purpose: str) -> str:
    normalized = normalize_purpose(raw_purpose)
    return normalized if normalized in CANONICAL_PURPOSES else ""


# Purpose families: purposes within the same family are too similar to constitute
# a genuine exclusivity violation.
_PURPOSE_FAMILIES = [
    {"service delivery", "product improvement", "personalization", "recommendations"},
    {"advertising", "targeted advertising", "marketing"},
    {"analytics", "research"},
    {"security", "fraud prevention", "authentication"},
    {"legal compliance"},
]
_PURPOSE_FAMILY_MAP: dict[str, int] = {}
for _i, _family in enumerate(_PURPOSE_FAMILIES):
    for _p in _family:
        _PURPOSE_FAMILY_MAP[_p] = _i


def _purposes_related(purpose_a: str, purpose_b: str) -> bool:
    """Return True if two purposes belong to the same family."""
    fam_a = _PURPOSE_FAMILY_MAP.get(purpose_a)
    fam_b = _PURPOSE_FAMILY_MAP.get(purpose_b)
    if fam_a is not None and fam_b is not None:
        return fam_a == fam_b
    return False


def _statement_text(statement: PPS) -> str:
    return f"{statement.source_section} {statement.source_text}".lower()


def _gdpr_categories_overlap(first: PPS, second: PPS) -> bool:
    """Return True if two statements share at least one GDPR category.

    When both statements have GDPR category annotations, this filter
    ensures that pattern comparisons only fire within the same disclosure
    context (e.g., both about "Storage Period" or both about "Data Recipients").
    If either statement lacks categories, we conservatively return True
    to avoid silencing legitimate findings from un-annotated extractions.
    """
    if not first.gdpr_categories or not second.gdpr_categories:
        return True  # conservative: no annotation → allow comparison
    return bool(set(first.gdpr_categories) & set(second.gdpr_categories))


def _is_transfer_safeguard_or_rights_statement(statement: PPS) -> bool:
    text = _statement_text(statement)
    markers = (
        "standard contractual clauses",
        "adequate level of protection",
        "alternative mechanism approved by the european commission",
        "data portability",
        "transfer your personal data to another controller",
        "to another controller",
        "safeguard transfers",
        "safeguards are in place",
        "safeguards in place",
        "necessary safeguards",
        "appropriate safeguards",
        "adequate safeguards",
        "countries outside",
        "less stringent data protection",
        "different data protection standards",
        "have obtained their necessary consent",
        "materially alter your privacy rights",
        "updated policy",
        "changes to the policy",
        "publish or send a notice",
        "notice about the changes to the policy",
        # Generic policy preamble introductions — not statements of practice
        "how we collect, use",
        "describes how we collect",
        "this privacy policy describes",
        "we value your privacy",
        "about how we collect",
        # Public-source collection is not a vendor distribution violation
        "publicly accessible sources",
        "publicly available sources",
        "from public sources",
        "publicly available information",
        # Corporate transaction carve-outs — not routine data distribution
        "merger, acquisition",
        "merger or acquisition",
        "sale of assets",
        "change of ownership",
        "change of control",
        "corporate transaction",
    )
    return any(marker in text for marker in markers)


def _has_limited_third_party_commitment(statement: PPS) -> bool:
    if statement.action not in {"share", "sell", "transfer"}:
        return False
    text = _statement_text(statement)
    return (
        statement.condition != ConditionType.UNSPECIFIED
        or "limited circumstances" in text
        or "required by law" in text
        or "only be shared" in text
        or "only with your consent" in text
    )


def _is_context_limited_website_statement(statement: PPS, canonical_purpose: str) -> bool:
    text = _statement_text(statement)
    if canonical_purpose == "legal compliance":
        return True
    return (
        "limited circumstances" in text
        or "required by law" in text
        or "judicial or administrative order" in text
        or "law enforcement" in text
        or "subpoena" in text
    )


def _has_strong_vendor_distribution_signal(statement: PPS, canonical_purpose: str) -> bool:
    if statement.action not in {"share", "sell", "transfer"}:
        return False
    if canonical_purpose in {"advertising", "targeted advertising", "marketing"}:
        return True
    if statement.recipient.strip():
        return True
    if statement.temporality != TemporalityType.UNSPECIFIED:
        return True
    return False


# Open-ended retention phrasings that the extractor occasionally mis-labels
# as SPECIFIC_DURATION (they semantically belong to UNTIL_PURPOSE). Treating
# them as specific durations made Π₄ pair "indefinite" with "as long as
# necessary" as if they were a contradiction, which the verifier rejected
# 100% of the time (see 2026-04-24 run).
_OPEN_ENDED_DURATION_MARKERS = re.compile(
    r"\bas\s+(?:long\s+as|needed|necessary|permitted|required)\b"
    r"|\buntil\s+(?:no\s+longer\s+needed|(?:such\s+)?time\s+as)\b"
    r"|\bfor\s+(?:as\s+long\s+as|the\s+duration\s+of)\b"
    r"|\bongoing\b|\bperpetual(?:ly)?\b",
    re.IGNORECASE,
)


def _temporality_is_open_ended(value: str) -> bool:
    """True if the temporality_value string is an open-ended phrasing.

    Used by Π₄ to drop pairs whose "specific_duration" side is actually a
    purpose-scoped retention ("as long as necessary to provide the
    service", "as permitted by law"). These are semantically UNTIL_PURPOSE
    and can't contradict an indefinite clause from the other side.
    """
    if not value:
        return False
    return bool(_OPEN_ENDED_DURATION_MARKERS.search(value))


def _parse_duration_days(value: str) -> float | None:
    if not value:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(hour|day|week|month|year)s?", value.lower())
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2)
    multipliers = {
        "hour": PI5_HOUR_DAYS,
        "day": PI5_DAY_DAYS,
        "week": PI5_WEEK_DAYS,
        "month": PI5_MONTH_DAYS,
        "year": PI5_YEAR_DAYS,
    }
    return number * multipliers[unit]


def _unique_categories(*category_lists: list[str]) -> list[str]:
    seen: list[str] = []
    for categories in category_lists:
        for category in categories:
            if category not in seen:
                seen.append(category)
    return seen


def pi1_modality_contradiction(
    website_source: list[PPS] | nx.MultiDiGraph,
    vendor_source: list[PPS] | nx.MultiDiGraph | None = None,
) -> list[Inconsistency]:
    """Π₁: modality contradiction.

    Merged in this revision: the former Π₈ (cross-policy commitment
    transitivity) is now Π₁'s cross-policy arm.

    - Intra-policy (no vendor_source, not a merged graph): group statements
      by (actor, data); any prohibition is contradicted by a permissive/
      commitment statement in the same or ontologically related group when
      the prohibition's action equals or subsumes the permissive action.
    - Cross-policy (vendor_source provided, or a merged graph is passed as
      website_source): a website prohibition is violated by a vendor's
      permissive practice on the same or ontologically related data, AND
      a website-side commitment whose purpose/condition is tighter than
      the vendor's observed practice also fires.
    """
    if _is_cross_policy_call(website_source, vendor_source):
        return _pi1_cross_policy(website_source, vendor_source)

    statements = _coerce_statements(website_source)
    results: list[Inconsistency] = []
    # Group by (actor, data) only — action hierarchy is handled per-pair via
    # _action_subsumes so "do not share" can match "sell" (more specific).
    # Grouping by exact action would split these into separate buckets and
    # miss the cross-action contradiction.
    groups: dict[tuple[str, str], list[PPS]] = {}
    for statement in statements:
        normalized = normalize_data_type(statement.data_object) or statement.data_object.lower().strip()
        key = (_actor_key(statement), normalized)
        groups.setdefault(key, []).append(statement)

    # Parent/child data-type bridging: a blanket prohibition on "personal
    # data" must also reach permissive statements on child types like
    # "browsing history" in a different data-group. Merge permissive
    # statements from descendant groups into parent groups that contain a
    # prohibition (actor-matched).
    if groups:
        for key, group in list(groups.items()):
            actor_key, data_norm = key
            if not any(_is_prohibitive(s) for s in group):
                continue
            for other_key, other_group in groups.items():
                if other_key == key:
                    continue
                other_actor, other_data = other_key
                if other_actor != actor_key:
                    continue
                if not (data_subsumes(data_norm, other_data) or data_subsumes(other_data, data_norm)):
                    continue
                for s in other_group:
                    if _is_permissive(s) and s not in group:
                        group.append(s)

    emitted_pairs: set[tuple[str, str]] = set()

    for group_key, group in groups.items():
        prohibitions = [
            s for s in group
            if _is_prohibitive(s)
            and not _is_hedged_or_scoped_prohibition(s)
            and not _is_user_directed_prohibition(s)
        ]
        permissive = [s for s in group if _is_permissive(s)]
        if not prohibitions or not permissive:
            _trace("pi1", {
                "event": "group_no_both_polarities",
                "group_key": list(group_key),
                "n_prohibitions": len(prohibitions),
                "n_permissive": len(permissive),
                "group_size": len(group),
            })
            continue

        # Evaluate all prohibition×permissive pairs; emit the first accepted
        # pair per (prohibition.id, permissive.id), de-duped across groups.
        prohibitions_sorted = sorted(prohibitions, key=_statement_sort_key)
        permissive_sorted = sorted(permissive, key=_statement_sort_key)

        for prohibition in prohibitions_sorted:
            for permissive_statement in permissive_sorted:
                pair_id = tuple(sorted((prohibition.id, permissive_statement.id)))
                if pair_id in emitted_pairs:
                    continue

                def _pi1_reject(filter_name: str, pr=prohibition, pe=permissive_statement) -> None:
                    _trace("pi1", {
                        "event": "pair_reject",
                        "filter": filter_name,
                        "prohibition_id": pr.id,
                        "permissive_id": pe.id,
                        "prohibition_scope": pr.scope,
                        "permissive_scope": pe.scope,
                        "prohibition_data": pr.data_object,
                        "permissive_data": pe.data_object,
                        "prohibition_action": pr.action,
                        "permissive_action": pe.action,
                        "prohibition_source": pr.policy_source,
                        "permissive_source": pe.policy_source,
                    })

                # Action hierarchy: accept exact match OR prohibition action
                # subsumes (is broader than) permissive action. This catches
                # "do not share" vs "sell" (sell ⊑ share) while preventing
                # "do not sell" vs unrelated "process".
                if prohibition.action != permissive_statement.action:
                    if not _action_subsumes(prohibition.action, permissive_statement.action):
                        _pi1_reject("action_not_subsumed")
                        continue
                if not _scope_compatible(prohibition, permissive_statement):
                    _pi1_reject("scope_incompatible")
                    continue
                if prohibition.source_text == permissive_statement.source_text:
                    _pi1_reject("same_source_text")
                    continue
                if not data_types_related(prohibition.data_object, permissive_statement.data_object):
                    _pi1_reject("data_types_not_related")
                    continue

                _trace("pi1", {
                    "event": "pair_accepted",
                    "prohibition_id": prohibition.id,
                    "permissive_id": permissive_statement.id,
                })
                emitted_pairs.add(pair_id)
                results.append(
                    Inconsistency(
                        inconsistency_id=f"pi1_{len(results)}",
                        pattern_id=PATTERN_IDS["pi1"],
                        pattern_name="Modality Contradiction",
                        statement_1=prohibition,
                        statement_2=permissive_statement,
                        verdict=Verdict.INCONSISTENT,
                        severity=Severity.CRITICAL,
                        explanation=(
                            f"The policy prohibits '{prohibition.action}' for '{prohibition.data_object}' "
                            f"but elsewhere allows or commits to it."
                        ),
                        gdpr_categories=_unique_categories(
                            prohibition.gdpr_categories,
                            permissive_statement.gdpr_categories,
                        ),
                        evidence_spans=[prohibition.source_text, permissive_statement.source_text],
                    )
                )

    return results


_EXCLUSIVITY_MARKERS = re.compile(
    r"\b(only (?:use|collect|share|process|retain)|solely for|exclusively for|strictly for|for the sole purpose)\b",
    re.IGNORECASE,
)


_PI2_SELF_REF_PHRASES = (
    "described in this",
    "set forth herein",
    "set forth in this",
    "as described herein",
    "purposes of this",
    "in accordance with this",
    "pursuant to this",
)


def _pi2_is_usable_exclusive(stmt: PPS) -> bool:
    """Does a PPS qualify as a valid exclusivity clause for Π₂?

    Mirrors the intra-path gate: processing action, has a purpose, not a
    prohibition/negative, contains exclusivity language, and passes the
    self-referential and retention-context filters. Lifted into a helper
    so the cross-policy path can apply the same gate without recursing
    through the intra implementation.
    """
    if stmt.is_negative or stmt.modality == Modality.PROHIBITION:
        return False
    if stmt.action not in PROCESSING_ACTIONS:
        return False
    if not stmt.purpose:
        return False
    if not _EXCLUSIVITY_MARKERS.search(stmt.source_text or ""):
        return False
    sl = (stmt.source_text or "").lower()
    if any(p in sl for p in _PI2_SELF_REF_PHRASES):
        return False
    if re.search(r"only\s+retain", sl):
        return False
    return True


def _pi2_other_eligible(stmt: PPS) -> bool:
    """Does a PPS qualify as the 'other' (conflicting) side for Π₂?"""
    if stmt.is_negative or stmt.modality == Modality.PROHIBITION:
        return False
    if stmt.action not in PROCESSING_ACTIONS:
        return False
    if not stmt.purpose:
        return False
    return True


def _pi2_cross_policy(
    website_source: list[PPS] | nx.MultiDiGraph,
    vendor_source: list[PPS] | nx.MultiDiGraph | None,
) -> list[Inconsistency]:
    """Cross-policy Π₂: FP exclusivity clause contradicted by a TP practice
    (or symmetric). Iterates FP-excl × TP-other and TP-excl × FP-other so
    the per-clause cap applies to cross pairs, not same-side intra noise.
    Gates are the same as the intra path; only the iteration scope changes.
    """
    w_stmts, v_stmts = _coerce_cross_policy_statements(website_source, vendor_source)
    w_excl = [s for s in w_stmts if _pi2_is_usable_exclusive(s)]
    v_excl = [s for s in v_stmts if _pi2_is_usable_exclusive(s)]
    if not w_excl and not v_excl:
        return []

    w_other = [s for s in w_stmts if _pi2_other_eligible(s)]
    v_other = [s for s in v_stmts if _pi2_other_eligible(s)]

    best_by_key: dict[tuple[str, str], Inconsistency] = {}
    clause_counts: dict[str, int] = {}

    def _try_pair(excl: PPS, other: PPS) -> None:
        if excl.id == other.id:
            return
        if not _scope_compatible(excl, other, cross_policy=True):
            return
        # Legal basis enumerations — not a purpose conflict.
        other_lower = (other.source_text or "").lower()
        if re.search(
            r"(?:legitimate interest|legal (?:basis|ground|obligation)|"
            r"contract performance|vital interest)",
            other_lower,
        ) and re.search(r"(?:consent|performance of|comply with)", other_lower):
            return
        if not _data_same_or_subsuming(excl.data_object, other.data_object):
            return
        excl_purpose = normalize_purpose(excl.purpose)
        other_purpose = normalize_purpose(other.purpose)
        if not excl_purpose or not other_purpose:
            return
        # Competing exclusivity on same purpose family = reinforcing, not conflict.
        if _EXCLUSIVITY_MARKERS.search(other.source_text or ""):
            if other_purpose == excl_purpose or _purposes_related(excl_purpose, other_purpose):
                return
        if other_purpose == excl_purpose:
            return
        if _purposes_related(excl_purpose, other_purpose):
            return
        if not _gdpr_categories_overlap(excl, other):
            return

        clause_key = (excl.source_text or "").strip()
        dedup_key = (clause_key, other_purpose)
        if dedup_key in best_by_key:
            return
        if clause_counts.get(clause_key, 0) >= PI2_MAX_PER_EXCLUSIVE_CLAUSE:
            return

        match = _EXCLUSIVITY_MARKERS.search(excl.source_text or "")
        marker_text = match.group() if match else "only"
        best_by_key[dedup_key] = Inconsistency(
            inconsistency_id="",
            pattern_id=PATTERN_IDS["pi2"],
            pattern_name="Exclusivity Violation",
            statement_1=excl,
            statement_2=other,
            verdict=Verdict.INCONSISTENT,
            severity=Severity.HIGH,
            explanation=(
                f"'{excl.data_object}' is restricted to '{excl_purpose}' "
                f"(exclusivity language: \"{marker_text}\") "
                f"but is also used for '{other_purpose}'."
            ),
            gdpr_categories=_unique_categories(
                excl.gdpr_categories,
                other.gdpr_categories,
                ["Purpose Limitation"],
            ),
            evidence_spans=[excl.source_text, other.source_text],
        )
        clause_counts[clause_key] = clause_counts.get(clause_key, 0) + 1

    for excl in w_excl:
        for other in v_other:
            _try_pair(excl, other)
    for excl in v_excl:
        for other in w_other:
            _try_pair(excl, other)

    results = list(best_by_key.values())
    for index, result in enumerate(results):
        result.inconsistency_id = f"pi2_cross_{index}"
    return results


def pi2_exclusivity_violation(
    website_source: list[PPS] | nx.MultiDiGraph,
    vendor_source: list[PPS] | nx.MultiDiGraph | None = None,
) -> list[Inconsistency]:
    """Π₂: exclusivity violation.

    Intra-policy: a statement scoped with 'only/solely' for one purpose is
    contradicted by another statement in the same policy that uses the same
    data for a different purpose.

    Cross-policy: the website's exclusivity clause (e.g., "we only use D for
    authentication") is contradicted by a vendor practice that processes D
    for a different purpose — or the other direction. Dedicated cross
    detector (below) iterates FP-exclusives × TP-others and vice versa so
    the per-exclusive-clause cap (MAX_PER_EXCLUSIVE_CLAUSE=3) applies to
    cross pairs only instead of being starved by same-side intra noise.
    """
    if _is_cross_policy_call(website_source, vendor_source):
        return _pi2_cross_policy(website_source, vendor_source)

    statements = _coerce_statements(website_source)
    results: list[Inconsistency] = []

    # Find statements whose source text contains exclusivity language
    exclusive: list[PPS] = []
    for statement in statements:
        if statement.is_negative or statement.modality == Modality.PROHIBITION:
            continue
        if statement.action not in PROCESSING_ACTIONS:
            continue
        if not statement.purpose:
            continue
        if _EXCLUSIVITY_MARKERS.search(statement.source_text):
            # Round 3: skip self-referential exclusivity — "solely for purposes
            # described in this policy" covers everything, so it's not a real
            # purpose restriction.
            src_lower = statement.source_text.lower()
            if any(phrase in src_lower for phrase in (
                "described in this",
                "set forth herein",
                "set forth in this",
                "as described herein",
                "purposes of this",
                "in accordance with this",
                "pursuant to this",
            )):
                continue
            # Round 3: skip retention-context "only" — "only retain as long as
            # necessary" is a retention commitment, not a purpose restriction.
            if re.search(r"only\s+retain", src_lower):
                continue
            exclusive.append(statement)

    if not exclusive:
        return results

    # Deduplicate: keep one finding per (exclusive_source_text, conflicting_purpose).
    # Cap at 3 conflicting purposes per exclusive clause to avoid fan-out.
    MAX_PER_EXCLUSIVE_CLAUSE = PI2_MAX_PER_EXCLUSIVE_CLAUSE
    best_by_key: dict[tuple[str, str], Inconsistency] = {}
    clause_counts: dict[str, int] = {}

    for excl in exclusive:
        excl_data = normalize_data_type(excl.data_object)
        excl_purpose = normalize_purpose(excl.purpose)
        if not excl_data or not excl_purpose:
            continue

        for other in statements:
            if other.id == excl.id:
                continue
            if other.is_negative or other.modality == Modality.PROHIBITION:
                continue
            if other.action not in PROCESSING_ACTIONS:
                continue
            if not other.purpose:
                continue
            # Scope gate (R10)
            if not _scope_compatible(excl, other):
                continue
            # Skip if both statements come from the same or near-identical source text
            if excl.source_text.strip() == other.source_text.strip():
                continue
            if len(excl.source_text.strip()) > 50 and excl.source_text.strip()[:200] == other.source_text.strip()[:200]:
                continue
            # Round 3: skip legal basis enumerations — "legitimate interest,
            # consent, or contract performance" is not a conflicting purpose.
            other_lower = other.source_text.lower()
            if re.search(r"(?:legitimate interest|legal (?:basis|ground|obligation)|contract performance|vital interest)", other_lower) and \
               re.search(r"(?:consent|performance of|comply with)", other_lower):
                continue
            # Π₂ requires exact data-type match or strict subsumption — sibling
            # data types (e.g., ip address vs device id) describe different
            # practices even under the same parent; comparing them produced
            # exclusivity FPs via data_types_related's sibling acceptance.
            if not _data_same_or_subsuming(excl.data_object, other.data_object):
                continue

            other_purpose = normalize_purpose(other.purpose)

            # Fix 9 (revised per Dr. Chen): Skip ONLY when both use exclusivity
            # language AND their purposes are the same/related (reinforcing).
            # Two competing exclusivity claims with DIFFERENT purposes
            # (e.g., "only for auth" vs "solely for ads") are stronger contradictions.
            if _EXCLUSIVITY_MARKERS.search(other.source_text):
                if other_purpose and (other_purpose == excl_purpose or _purposes_related(excl_purpose, other_purpose)):
                    continue  # same purpose, reinforcing
                # else: different purposes with competing exclusivity → genuine conflict
            if not other_purpose or other_purpose == excl_purpose:
                continue
            # Skip if purposes are closely related (sub-purposes of same category)
            if _purposes_related(excl_purpose, other_purpose):
                continue
            # GDPR category scoping
            if not _gdpr_categories_overlap(excl, other):
                continue

            clause_key = excl.source_text.strip()
            dedup_key = (clause_key, other_purpose)
            if dedup_key in best_by_key:
                continue
            if clause_counts.get(clause_key, 0) >= MAX_PER_EXCLUSIVE_CLAUSE:
                continue

            match = _EXCLUSIVITY_MARKERS.search(excl.source_text)
            marker_text = match.group() if match else "only"

            candidate = Inconsistency(
                inconsistency_id="",
                pattern_id=PATTERN_IDS["pi2"],
                pattern_name="Exclusivity Violation",
                statement_1=excl,
                statement_2=other,
                verdict=Verdict.INCONSISTENT,
                severity=Severity.HIGH,
                explanation=(
                    f"'{excl.data_object}' is restricted to '{excl_purpose}' "
                    f"(exclusivity language: \"{marker_text}\") "
                    f"but is also used for '{other_purpose}'."
                ),
                gdpr_categories=_unique_categories(
                    excl.gdpr_categories,
                    other.gdpr_categories,
                    ["Purpose Limitation"],
                ),
                evidence_spans=[excl.source_text, other.source_text],
            )
            best_by_key[dedup_key] = candidate
            clause_counts[clause_key] = clause_counts.get(clause_key, 0) + 1

    results = list(best_by_key.values())
    for index, result in enumerate(results):
        result.inconsistency_id = f"pi2_{index}"
    return results


def pi3_condition_asymmetry(
    website_source: list[PPS] | nx.MultiDiGraph,
    vendor_source: list[PPS] | nx.MultiDiGraph | None = None,
) -> list[Inconsistency]:
    """Π₃: same practice governed by materially different consent conditions.

    Intra-policy: group by (actor, action, data); pick the pair with the
    largest strictness gap that clears PI4_STRICTNESS_GAP_MIN.

    Cross-policy: pair a website-side consent-gated clause against a
    vendor-side weaker condition on the same or subsumed data. Grouping
    ignores actor because FirstParty and ThirdParty:* never match literally.
    """
    if _is_cross_policy_call(website_source, vendor_source):
        return _pi3_cross_policy(website_source, vendor_source)

    statements = [statement for statement in _coerce_statements(website_source) if not statement.is_negative]
    results: list[Inconsistency] = []
    groups: dict[tuple[str, str, str], list[PPS]] = {}
    for statement in statements:
        key = (
            _actor_key(statement),
            statement.action,
            normalize_data_type(statement.data_object),
        )
        groups.setdefault(key, []).append(statement)

    emitted_pairs: set[tuple[str, str]] = set()
    for group in groups.values():
        # Drop UNSPECIFIED-condition candidates up-front: the policy simply
        # didn't state a condition, so a pair against it is ambiguous and
        # previously flooded the group-extreme selection with FPs.
        candidates = [s for s in group if s.condition.value != "unspecified"]
        if len(candidates) < 2:
            continue

        # Evaluate all pairs; pick the single pair with the largest
        # strictness gap (ties broken by stricter side's source length,
        # then by stable string IDs). The ordering key is kept as a
        # fully-comparable tuple of (int, int, str, str) so Python
        # never falls through to comparing PPS instances themselves
        # (which has no __gt__ and was the cause of a TypeError on
        # pairs whose two candidates shared the same gap and tiebreak).
        best_key: tuple[int, int, str, str] | None = None
        best_pair: tuple[PPS, PPS] | None = None
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = candidates[i], candidates[j]
                av = CONDITION_STRICTNESS.get(a.condition.value, 0)
                bv = CONDITION_STRICTNESS.get(b.condition.value, 0)
                if av >= bv:
                    stricter, looser = a, b
                    gap = av - bv
                else:
                    stricter, looser = b, a
                    gap = bv - av
                if gap < PI4_STRICTNESS_GAP_MIN:
                    continue
                if not _scope_compatible(looser, stricter):
                    continue
                # Same section → elaboration, not contradiction.
                if looser.source_section and looser.source_section == stricter.source_section:
                    continue
                tiebreak = len(stricter.source_text) + len(looser.source_text)
                key = (gap, tiebreak, stricter.id, looser.id)
                if best_key is None or key > best_key:
                    best_key = key
                    best_pair = (stricter, looser)

        if best_pair is None:
            continue
        stricter, looser = best_pair
        pair_id = tuple(sorted((stricter.id, looser.id)))
        if pair_id in emitted_pairs:
            continue
        emitted_pairs.add(pair_id)

        results.append(
            Inconsistency(
                inconsistency_id=f"pi3_{len(results)}",
                pattern_id=PATTERN_IDS["pi3"],
                pattern_name="Condition Asymmetry",
                statement_1=stricter,
                statement_2=looser,
                verdict=Verdict.UNDERSPECIFIED,
                severity=Severity.HIGH,
                explanation=(
                    f"The same practice on '{stricter.data_object}' requires "
                    f"'{stricter.condition.value}' in one clause but '{looser.condition.value}' in another."
                ),
                gdpr_categories=_unique_categories(
                    stricter.gdpr_categories,
                    looser.gdpr_categories,
                    ["Withdraw Consent"],
                ),
                evidence_spans=[stricter.source_text, looser.source_text],
            )
        )

    return results


def _pi3_cross_policy(
    website_source: list[PPS] | nx.MultiDiGraph,
    vendor_source: list[PPS] | nx.MultiDiGraph | None,
) -> list[Inconsistency]:
    """Cross-policy Π₃: pair a website consent-gated clause against a vendor
    weaker condition on the same or subsumed data.

    Candidate pairs are iterated directly across (action, data) because
    grouping by normalised data type misses parent/child matches
    (e.g., website "personal data" vs vendor "email address"). We use
    _data_same_or_subsuming for the data gate, which accepts exact match
    and both directions of subsumption. The strictness-gap test and
    source-section guard are retained.
    """
    w_stmts, v_stmts = _coerce_cross_policy_statements(website_source, vendor_source)
    statements = [s for s in (list(w_stmts) + list(v_stmts))
                  if not s.is_negative and s.condition.value != "unspecified"]
    if len(statements) < 2:
        return []

    # Bucket by action and pre-split by side so the inner loop is only
    # FP × TP, not the full N² over all same-action statements.
    fp_groups: dict[str, list[PPS]] = {}
    tp_groups: dict[str, list[PPS]] = {}
    for s in statements:
        side = _pps_side(s)
        if side == "website":
            fp_groups.setdefault(s.action, []).append(s)
        elif side == "vendor":
            tp_groups.setdefault(s.action, []).append(s)

    results: list[Inconsistency] = []
    emitted_pairs: set[tuple[str, str]] = set()
    # One finding per (website_data_canonical, action) bucket: across all
    # vendor statements that match-or-subsume the website data type, keep the
    # pair with the largest strictness gap. Key is (int, int, str, str) so
    # tuple compare never falls through to PPS (which has no __gt__).
    bucket_best_key: dict[tuple[str, str, str], tuple[int, int, str, str]] = {}
    bucket_best_pair: dict[tuple[str, str, str], tuple[PPS, PPS]] = {}
    for action, fp_list in fp_groups.items():
        tp_list = tp_groups.get(action) or []
        if not tp_list:
            continue
        for a in fp_list:
            for b in tp_list:
                # Subsumption-aware data gate: FP "personal data upon_consent"
                # vs TP "email address by_default" is a legally real Π₃ match
                # (email ⊑ personal data, so the FP consent commitment binds
                # TP processing). We don't drop subsumption — instead we
                # attach subsume_path to the finding below so the verifier
                # sees the ontology relationship in its context block and
                # doesn't dismiss it as "different data types". See
                # _build_cluster_narrative in verifier.py for the rendered
                # "DATA TYPE RELATIONSHIP" paragraph.
                if not _data_same_or_subsuming(a.data_object, b.data_object):
                    continue
                if not _scope_compatible(a, b, cross_policy=True):
                    continue
                av = CONDITION_STRICTNESS.get(a.condition.value, 0)
                bv = CONDITION_STRICTNESS.get(b.condition.value, 0)
                if av >= bv:
                    stricter, looser, gap = a, b, av - bv
                else:
                    stricter, looser, gap = b, a, bv - av
                if gap < PI4_STRICTNESS_GAP_MIN:
                    continue
                # Same source section is unlikely cross-policy, but guard anyway.
                if looser.source_section and looser.source_section == stricter.source_section:
                    continue
                tiebreak = len(stricter.source_text) + len(looser.source_text)
                # Bucket key: (action, website_data_canonical, vendor_data_canonical).
                website_side = stricter if _pps_side(stricter) == "website" else looser
                vendor_side  = looser   if _pps_side(stricter) == "website" else stricter
                bucket_key = (
                    action,
                    normalize_data_type(website_side.data_object) or website_side.data_object.lower().strip(),
                    normalize_data_type(vendor_side.data_object)  or vendor_side.data_object.lower().strip(),
                )
                key = (gap, tiebreak, stricter.id, looser.id)
                existing = bucket_best_key.get(bucket_key)
                if existing is None or key > existing:
                    bucket_best_key[bucket_key] = key
                    bucket_best_pair[bucket_key] = (stricter, looser)

    for stricter, looser in bucket_best_pair.values():
        pair_id = tuple(sorted((stricter.id, looser.id)))
        if pair_id in emitted_pairs:
            continue
        emitted_pairs.add(pair_id)

        # Ontology context for the verifier. When FP and TP name different
        # but ontologically-related data types (e.g. "personal data" vs
        # "email address"), pass the subsume_path + both data type strings
        # so _build_cluster_narrative in verifier.py renders the DATA
        # TYPE RELATIONSHIP paragraph to the LLM. Without this, gemma3:27b
        # reads "different data types" and marks non_conflict, discarding
        # real parent/child findings.
        w_side = stricter if _pps_side(stricter) == "website" else looser
        v_side = looser if _pps_side(stricter) == "website" else stricter
        w_norm = normalize_data_type(w_side.data_object) or (w_side.data_object or "").lower().strip()
        v_norm = normalize_data_type(v_side.data_object) or (v_side.data_object or "").lower().strip()
        subsume_path = None
        if w_norm and v_norm and w_norm != v_norm:
            path = _get_subsume_path(w_norm, v_norm)
            if not path:
                path = _get_subsume_path(v_norm, w_norm)
            if path:
                subsume_path = " → ".join(path)

        inc = Inconsistency(
            inconsistency_id=f"pi3_cross_{len(results)}",
            pattern_id=PATTERN_IDS["pi3"],
            pattern_name="Condition Asymmetry",
            statement_1=stricter,
            statement_2=looser,
            verdict=Verdict.UNDERSPECIFIED,
            severity=Severity.HIGH,
            explanation=(
                f"Website and vendor disagree on the consent condition for "
                f"'{stricter.data_object}': one requires '{stricter.condition.value}', "
                f"the other '{looser.condition.value}'."
            ),
            gdpr_categories=_unique_categories(
                stricter.gdpr_categories,
                looser.gdpr_categories,
                ["Withdraw Consent"],
            ),
            evidence_spans=[stricter.source_text, looser.source_text],
        )
        inc.neighborhood_context = {
            "data_type": w_norm or v_norm,
            "website_data_type": w_norm,
            "vendor_data_type": v_norm,
            "subsume_path": subsume_path,
            "website_action": w_side.action,
            "vendor_action": v_side.action,
        }
        results.append(inc)

    return results


def pi4_temporal_contradiction(
    website_source: list[PPS] | nx.MultiDiGraph,
    vendor_source: list[PPS] | nx.MultiDiGraph | None = None,
) -> list[Inconsistency]:
    """Π₄: conflicting retention periods for the same data.

    Intra-policy: pairwise comparison of retention statements within a
    single policy, gated by same-or-subsuming data-type match.

    Cross-policy: website retention promise vs vendor divergent retention
    on aligned data. Runs the intra detector on the combined website+vendor
    set (whose `_actors_comparable` already admits FirstParty↔ThirdParty
    pairs) and filters to cross-policy findings.
    """
    if _is_cross_policy_call(website_source, vendor_source):
        w_stmts, v_stmts = _coerce_cross_policy_statements(website_source, vendor_source)
        combined = list(w_stmts) + list(v_stmts)
        findings = pi4_temporal_contradiction(combined)
        return _filter_cross_findings(findings)

    statements = [
        statement
        for statement in _coerce_statements(website_source)
        if statement.action == "retain" or statement.temporality != TemporalityType.UNSPECIFIED
    ]
    results: list[Inconsistency] = []
    seen: set[tuple[str, str]] = set()

    for first, second in itertools.combinations(statements, 2):
        if not _actors_comparable(first, second):
            continue
        # Scope gate (R10)
        if not _scope_compatible(first, second):
            continue
        # Π₅ fix (round 9): retention contradictions require EXACT data-type
        # match or strict subsumption — siblings (e.g., ip address vs device id)
        # describe different retention schedules and must not be compared.
        first_norm = normalize_data_type(first.data_object) or first.data_object.lower().strip()
        second_norm = normalize_data_type(second.data_object) or second.data_object.lower().strip()
        if not (
            first_norm == second_norm
            or data_subsumes(first_norm, second_norm)
            or data_subsumes(second_norm, first_norm)
        ):
            continue

        pair = tuple(sorted((first.id, second.id)))
        if pair in seen:
            continue
        # GDPR category scoping intentionally skipped for Π₄: the pattern's
        # own gate (both sides have retention info on the same-or-subsuming
        # data type with a duration gap) IS the semantic filter. The
        # extractor frequently labels retention clauses under adjacent
        # categories ("Right to Erase", "Right to Restrict", "Processing
        # Purpose") instead of "Storage Period", and requiring literal
        # GDPR-category overlap was dropping every Π₄ finding on the
        # random2 corpus. See log 2026-04-24.

        # Narrow open-ended filter: only drop the INDEFINITE + open-ended
        # SPECIFIC pair (e.g. "indefinite" vs "as long as necessary").
        # Both sides are effectively unbounded and the verifier rejects
        # them 100% as non-conflict. Do NOT drop mixed cases where one
        # side is a numeric duration and the other is open-ended — those
        # may be genuine "30 days vs unbounded" conflicts. An earlier
        # broader filter dropped real findings.
        _first_open = _temporality_is_open_ended(first.temporality_value)
        _second_open = _temporality_is_open_ended(second.temporality_value)
        if (
            (first.temporality == TemporalityType.INDEFINITE and _second_open) or
            (second.temporality == TemporalityType.INDEFINITE and _first_open)
        ):
            continue

        first_days = _parse_duration_days(first.temporality_value)
        second_days = _parse_duration_days(second.temporality_value)
        explanation = ""

        if (
            first.temporality == TemporalityType.INDEFINITE
            and second.temporality == TemporalityType.SPECIFIC_DURATION
        ):
            explanation = (
                f"'{first.data_object}' is retained indefinitely in one clause but "
                f"'{second.temporality_value}' in another."
            )
        elif (
            second.temporality == TemporalityType.INDEFINITE
            and first.temporality == TemporalityType.SPECIFIC_DURATION
        ):
            explanation = (
                f"'{first.data_object}' is retained for '{first.temporality_value}' in one clause "
                f"but indefinitely in another."
            )
        elif first_days is not None and second_days is not None:
            shorter = min(first_days, second_days)
            longer = max(first_days, second_days)
            if shorter > 0 and longer >= shorter * PI5_DURATION_RATIO_MIN:
                explanation = (
                    f"Conflicting retention periods for '{first.data_object}': "
                    f"'{first.temporality_value}' versus '{second.temporality_value}'."
                )

        if not explanation:
            continue

        seen.add(pair)
        results.append(
            Inconsistency(
                inconsistency_id=f"pi4_{len(results)}",
                pattern_id=PATTERN_IDS["pi4"],
                pattern_name="Temporal Contradiction",
                statement_1=first,
                statement_2=second,
                verdict=Verdict.INCONSISTENT,
                severity=Severity.HIGH,
                explanation=explanation,
                gdpr_categories=_unique_categories(
                    first.gdpr_categories,
                    second.gdpr_categories,
                    ["Storage Period"],
                ),
                evidence_spans=[first.source_text, second.source_text],
            )
        )

    return results


_OVERLY_BROAD_DATA_TERMS = frozenset({
    "personal data", "personal information", "data", "information",
    "your data", "your information", "user data", "user information",
})

_RESTRICTIVE_MODALITIES = frozenset({Modality.PROHIBITION, Modality.OBLIGATION})


# Π₇ Vagueness Asymmetry was pruned in Round 11. The pattern had 0% genuine
# rate across all 11 tested pairs — the verifier consistently rejected Π₇
# findings as normal drafting, modality misextraction, or different-context
# comparisons. The pattern logic was sound but extraction quality could not
# support it. See git history for the original implementation.


def _pi1_cross_eligible_vendor(vendor_statements: list[PPS]) -> list[PPS]:
    """Pre-filter vendor statements for Π₈ eligibility."""
    return [
        vs for vs in vendor_statements
        # Π₈ contrasts a website prohibition with a vendor permissive/commitment
        # practice. A vendor *prohibition* (is_negative OR modality=PROHIBITION)
        # cannot create a transitivity conflict, so it must be excluded on
        # both polarity axes — the previous `not vs.is_negative` gate missed
        # modality=PROHIBITION with is_negative=False.
        if not _is_prohibitive(vs)
        and vs.action in PROCESSING_ACTIONS
        and vs.action != "receive"
        and not _is_transfer_safeguard_or_rights_statement(vs)
        and not _is_non_pii_statement(vs.source_text, vs)
        and not _is_public_data_statement(vs.source_text)
        and not (vs.action == "sell" and not any(
            v in vs.source_text.lower() for v in (
                "sell ", "sells ", "selling", "sold", " sale ", " sales ",
                "sales platform", "sales proceeds", "sale price", "monetise",
                "monetisation", "monetize", "monetization",
            )))
    ]


_USER_INTERACTION_RE = re.compile(
    r"\b(?:"
    r"(?:other|fellow)\s+users?"
    r"|(?:collaborate|interact|communicate)\s+with\s+(?:other|fellow)?\s*(?:users?|people|members?)"
    r"|user[- ]to[- ]user"
    r"|(?:share|display|show)\s+(?:your\s+)?(?:name|email|profile|information)\s+"
    r"(?:to|with)\s+(?:other|fellow)\s+(?:users?|people|members?|participants?)"
    r")\b",
    re.IGNORECASE,
)

_FEATURE_DISCLAIMER_RE = re.compile(
    # "To access AI Chat/Aria/GameMaker... we do not share your account data"
    # These are technical disclaimers about specific product features,
    # not general privacy commitments about vendor data sharing.
    r"\b(?:to\s+(?:access|use)\s+\w+|when\s+(?:you\s+)?(?:use|access)\s+\w+)"
    r".*(?:(?:do|does|will)\s+not\s+(?:share|send|transmit|transfer)"
    r"\s+(?:your\s+)?(?:\w+\s+)?(?:account|login|sign[- ]in)\s+(?:data|information))",
    re.IGNORECASE | re.DOTALL,
)

_LEGAL_DISCLOSURE_TABLE_RE = re.compile(
    r"\b(?:"
    r"Cal(?:ifornia)?\.?\s+Civ(?:il)?\.?\s+Code"
    r"|CCPA\s+categor"
    r"|categories?\s+(?:of\s+)?(?:personal\s+)?(?:information|data)\s+(?:listed|described|defined)"
    r"|(?:we|the\s+company)\s+(?:have\s+)?(?:collected|disclosed|sold|shared)"
    r"\s+the\s+following\s+categor"
    r")\b",
    re.IGNORECASE,
)


def _is_third_party_disclaimer(source_text: str) -> bool:
    """Detect third-party liability disclaimers (not company prohibitions)."""
    text = source_text.lower()
    return any(phrase in text for phrase in (
        "this privacy policy does not cover",
        "this policy does not apply to",
        "this notice does not cover",
        "not controlled by us",
        "not responsible for the privacy",
        "governed by their own privacy",
        "governed by their privacy",
        "subject to their own privacy",
    ))


def _is_delegation_statement(source_text: str) -> bool:
    """Detect delegation statements where processing is outsourced."""
    text = source_text.lower()
    return any(phrase in text for phrase in (
        "does not process your credit card",
        "does not receive or store your payment",
        "does not store your credit card",
        "does not store credit card",
        "does not handle your payment",
        "instead uses a third party",
        "instead we use",
        "rely on trusted third-party",
        "relies on a third-party",
        "handled by our payment processor",
        "processed by our payment partner",
    ))


def _is_education_scoped(source_text: str) -> bool:
    """Detect education/student-scoped prohibitions."""
    text = source_text.lower()
    return any(phrase in text for phrase in (
        "students using",
        "student data",
        "student information",
        "student personal",
        "education product",
        "educational purpose",
        "ferpa",
        "coppa",
    ))


def _is_glossary_or_definition(source_text: str) -> bool:
    """Detect glossary/definition text embedded after a statement."""
    text = source_text.lower()
    return any(phrase in text for phrase in (
        "means information that",
        "is defined as",
        "refers to information",
        "shall mean",
        "includes information such as",
        "personal information means",
        "personal data means",
    ))


def _is_third_party_actor_description(source_text: str) -> bool:
    """Detect descriptions of what a THIRD-PARTY actor does (not the website)."""
    text = source_text.lower()
    return any(phrase in text for phrase in (
        "your telecom operator",
        "your internet service provider",
        "your isp ",
        "the telecom operator",
        "the mobile operator",
        "the network operator",
    ))



def _is_technical_field_exclusion(source_text: str) -> bool:
    """Detect technical field-level exclusions (password fields, etc.)."""
    text = source_text.lower()
    return any(phrase in text for phrase in (
        "password field",
        "password input",
        "exclude password",
        "excluding password",
        "not include password",
        "does not capture password",
    ))


def _pi1_cross_eligible_website(statement: PPS) -> bool:
    """Check if a website statement is eligible for Π₈ comparison."""
    def _reject(filter_name: str) -> bool:
        _trace("pi1_cross", {
            "event": "website_filter_reject",
            "filter": filter_name,
            "s_id": statement.id,
            "action": statement.action,
            "data_object": statement.data_object,
            "modality": statement.modality.name if hasattr(statement.modality, "name") else str(statement.modality),
            "is_negative": statement.is_negative,
            "scope": statement.scope,
            "source_text": (statement.source_text or "")[:200],
        })
        return False

    if statement.action not in PROCESSING_ACTIONS and statement.action not in RIGHTS_ACTIONS:
        return _reject("not_processing_or_rights")
    if _is_childrens_clause(statement.source_text):
        return _reject("childrens_clause")
    if statement.action in RIGHTS_ACTIONS:
        return _reject("rights_action")
    if _is_user_directed_prohibition(statement):
        return _reject("user_directed_prohibition")
    if _is_non_pii_statement(statement.source_text, statement):
        return _reject("non_pii_statement")
    if _USER_INTERACTION_RE.search(statement.source_text):
        return _reject("user_interaction_re")
    if _LEGAL_DISCLOSURE_TABLE_RE.search(statement.source_text):
        return _reject("legal_disclosure_table")
    if _FEATURE_DISCLAIMER_RE.search(statement.source_text):
        return _reject("feature_disclaimer")
    if _is_third_party_disclaimer(statement.source_text):
        return _reject("third_party_disclaimer")
    if _is_delegation_statement(statement.source_text):
        return _reject("delegation_statement")
    if _is_education_scoped(statement.source_text):
        return _reject("education_scoped")
    if _is_glossary_or_definition(statement.source_text):
        return _reject("glossary_or_definition")
    if _is_third_party_actor_description(statement.source_text):
        return _reject("third_party_actor_description")
    if _is_technical_field_exclusion(statement.source_text):
        return _reject("technical_field_exclusion")
    _trace("pi1_cross", {
        "event": "website_eligible",
        "s_id": statement.id,
        "action": statement.action,
        "data_object": statement.data_object,
        "modality": statement.modality.name,
        "is_negative": statement.is_negative,
        "scope": statement.scope,
    })
    return True


def _pi1_cross_check_pair(
    website_statement: PPS,
    vendor_statement: PPS,
    skip_data_match: bool = False,
) -> Inconsistency | None:
    """Check a single website–vendor pair for Π₈ violations.

    When skip_data_match is True, data-type matching is assumed to have been
    handled structurally by graph alignment — the caller guarantees the pair
    shares a data-type neighborhood.
    """
    def _pair_reject(filter_name: str, extra: dict | None = None) -> None:
        ev = {
            "event": "pair_reject",
            "filter": filter_name,
            "ws_id": website_statement.id,
            "vs_id": vendor_statement.id,
            "ws_action": website_statement.action,
            "vs_action": vendor_statement.action,
            "ws_data": website_statement.data_object,
            "vs_data": vendor_statement.data_object,
            "ws_mod": website_statement.modality.name,
            "vs_mod": vendor_statement.modality.name,
            "ws_neg": website_statement.is_negative,
            "ws_scope": website_statement.scope,
            "vs_scope": vendor_statement.scope,
        }
        if extra:
            ev.update(extra)
        _trace("pi1_cross", ev)

    if not _scope_compatible(website_statement, vendor_statement, cross_policy=True):
        _pair_reject("scope_incompatible")
        return None

    # P1: recipient gate. If the website explicitly restricts sharing to a
    # specific named recipient ("only with Google Analytics"), and the vendor
    # side of this pair is a DIFFERENT entity, the website's statement does
    # not apply to this vendor — skip. Generic phrases like "third parties"
    # are not restrictions; fall through to normal pair checking.
    if _recipient_is_restrictive(website_statement):
        vendor_name = (vendor_statement.policy_source or "").replace(
            "third_party:", ""
        )
        if not _recipient_names_vendor(
            website_statement.recipient,
            vendor_statement.actor,
            vendor_name,
        ):
            _pair_reject(
                "recipient_mismatch",
                {
                    "ws_recipient": website_statement.recipient,
                    "vendor_actor": vendor_statement.actor,
                    "vendor_policy_source": vendor_statement.policy_source,
                },
            )
            return None

    if not skip_data_match and not _cross_policy_data_match(
        website_statement.data_object,
        vendor_statement.data_object,
    ):
        _pair_reject("data_mismatch")
        return None
    if website_statement.action in PROCESSING_ACTIONS:
        if not _actions_related(website_statement.action, vendor_statement.action):
            # ── P4: same-group action bypass ─────────────────────────────
            # Rationale: extractors occasionally pick "transfer" for a
            # clause a policy author would call "share" (and vice versa).
            # Within the distribution family {share, sell, transfer} and
            # within the usage family {collect, use, process}, we allow
            # cross-action comparison even without strict subsumption.
            # This trades a small precision drop for recall of real cross-
            # family violations that extraction noise would otherwise
            # suppress. Disable with PI8_ALLOW_SAME_GROUP_BYPASS=0 for
            # ablation (see config.py).
            ws_is_sharing = website_statement.action in _SHARING_ACTIONS
            vs_is_sharing = vendor_statement.action in _SHARING_ACTIONS
            ws_is_usage = website_statement.action in _USAGE_ACTIONS
            vs_is_usage = vendor_statement.action in _USAGE_ACTIONS
            same_group = (
                (ws_is_sharing and vs_is_sharing)
                or (ws_is_usage and vs_is_usage)
            ) and PI8_ALLOW_SAME_GROUP_BYPASS
            if not same_group:
                _pair_reject("action_unrelated")
                return None

    violations: list[str] = []
    website_data_norm = normalize_data_type(website_statement.data_object)
    vendor_data_norm = normalize_data_type(vendor_statement.data_object)

    website_purpose = _canonical_purpose(website_statement.purpose)
    vendor_purpose = _canonical_purpose(vendor_statement.purpose)
    if website_purpose and vendor_purpose and website_purpose != vendor_purpose:
        if (
            website_statement.action in {"share", "sell", "transfer"}
            and _is_context_limited_website_statement(website_statement, website_purpose)
        ):
            pass
        else:
            if (
                website_data_norm not in BROAD_DATA_TYPES
                or website_statement.action in {"share", "sell", "transfer"}
            ) and get_purpose_necessity(vendor_purpose) < get_purpose_necessity(website_purpose):
                violations.append(
                    f"Website limits use to '{website_purpose}' but vendor uses data for '{vendor_purpose}'."
                )

    website_strictness = CONDITION_STRICTNESS.get(website_statement.condition.value, 0)
    vendor_strictness = CONDITION_STRICTNESS.get(vendor_statement.condition.value, 0)
    comparable_for_condition = (
        website_statement.action == vendor_statement.action
        or {website_statement.action, vendor_statement.action}.issubset(
            {"share", "transfer", "sell"}
        )
    )
    if (
        comparable_for_condition
        and website_strictness >= PI8_WEBSITE_STRICTNESS_MIN
        and website_strictness > vendor_strictness
    ):
        comparable_collect_use_process = {website_statement.action, vendor_statement.action}.issubset(
            {"collect", "use", "process"}
        )
        if comparable_collect_use_process:
            if website_purpose and vendor_purpose and website_purpose == vendor_purpose:
                violations.append(
                    f"Website requires '{website_statement.condition.value}' but vendor operates '{vendor_statement.condition.value}'."
                )
        else:
            violations.append(
                f"Website requires '{website_statement.condition.value}' but vendor operates '{vendor_statement.condition.value}'."
            )

    if website_statement.temporality == TemporalityType.SPECIFIC_DURATION:
        website_days = _parse_duration_days(website_statement.temporality_value)
        vendor_days = _parse_duration_days(vendor_statement.temporality_value)
        if vendor_statement.temporality == TemporalityType.INDEFINITE:
            violations.append(
                f"Website promises '{website_statement.temporality_value}' retention but vendor retains indefinitely."
            )
        elif website_days and vendor_days and vendor_days > website_days:
            violations.append(
                f"Website promises '{website_statement.temporality_value}' retention but vendor retains data for '{vendor_statement.temporality_value}'."
            )

    # Mark that prohibition-arm was reached (useful for debugging)
    if website_statement.is_negative or website_statement.modality == Modality.PROHIBITION:
        _trace("pi1_cross", {
            "event": "prohibition_arm_entered",
            "ws_id": website_statement.id,
            "vs_id": vendor_statement.id,
            "ws_action": website_statement.action,
            "vs_action": vendor_statement.action,
            "ws_data": website_statement.data_object,
            "vs_data": vendor_statement.data_object,
            "ws_purpose": website_statement.purpose,
        })
    if website_statement.is_negative or website_statement.modality == Modality.PROHIBITION:
        # For prohibition violations, use DIRECTIONAL action subsumption:
        # the website prohibition must be EQUAL to or BROADER than the vendor
        # action. A prohibition on a broader action covers more specific ones.
        #
        # Examples (subsumption: sell → share → transfer → process):
        #   "do not process" + vendor "sells"  → FIRE (process subsumes sell)
        #   "do not share"   + vendor "sells"  → FIRE (share subsumes sell)
        #   "do not sell"    + vendor "sells"  → FIRE (same action)
        #   "do not sell"    + vendor "process"→ SKIP (sell does NOT subsume process)
        #   "do not sell"    + vendor "collect"→ SKIP (different branches)
        #
        # This mirrors PoliGraph's approach: a prohibition on a parent action
        # in the subsumption hierarchy covers all child actions.
        ws_action = website_statement.action
        vs_action = vendor_statement.action
        if ws_action != vs_action:
            ws_subsumes_vs = _action_subsumes(ws_action, vs_action)
            # Same family {share/sell/transfer} OR {collect/use/process};
            # toggled off by PI8_ALLOW_SAME_GROUP_BYPASS=0 for ablation.
            same_group = (
                (ws_action in _SHARING_ACTIONS and vs_action in _SHARING_ACTIONS)
                or (ws_action in _USAGE_ACTIONS and vs_action in _USAGE_ACTIONS)
            ) and PI8_ALLOW_SAME_GROUP_BYPASS
            if not ws_subsumes_vs and not same_group:
                _pair_reject("prohibition_action_not_subsumed",
                             {"ws_subsumes_vs": ws_subsumes_vs, "same_group": same_group})
                return None
        # Round 9 fix / P3 expansion: a blanket prohibition ("never sell",
        # "shall not transfer", "forbidden to share", "under no circumstances"
        # …) is violated by ANY vendor practice on the same data regardless of
        # the purpose the extractor attached. The LLM over-attributes purposes
        # to blanket prohibitions (e.g., reads "to any third party" → picks
        # purpose "monetisation"), which previously caused the prohibition to
        # require an exact purpose match with the vendor.
        # See `is_blanket_prohibition()` at module top for the compiled regex.
        ws_is_blanket = is_blanket_prohibition(website_statement.source_text)

        if not website_statement.purpose or ws_is_blanket:
            violations.append(
                f"Website says it does not '{website_statement.action}' '{website_statement.data_object}', but vendor does."
            )
        elif website_purpose and vendor_purpose and vendor_purpose == website_purpose:
            violations.append(
                f"Website says it does not '{website_statement.action}' '{website_statement.data_object}' for '{website_purpose}', but vendor does."
            )

    broad_condition_only = (
        len(violations) == 1
        and violations[0].startswith("Website requires")
        and website_data_norm in BROAD_DATA_TYPES
        and vendor_data_norm in BROAD_DATA_TYPES
    )
    if broad_condition_only and not (
        _has_limited_third_party_commitment(website_statement)
        and _has_strong_vendor_distribution_signal(vendor_statement, vendor_purpose)
    ):
        _pair_reject("broad_condition_only")
        return None

    if not violations:
        _pair_reject("no_violation_detected",
                     {"ws_mod": website_statement.modality.name,
                      "ws_neg": website_statement.is_negative,
                      "ws_purpose": website_statement.purpose,
                      "vs_purpose": vendor_statement.purpose})
        return None

    severity = Severity.CRITICAL if len(violations) >= 2 else Severity.HIGH
    vendor_cond = vendor_statement.condition.value
    explicit_weaker_vendor_condition = vendor_cond in {"by_default", "unless_opted_out"}
    prohibition_directly_violated = (
        any("does not" in v for v in violations)
        and explicit_weaker_vendor_condition
    )
    retention_violated = any("retains indefinitely" in v for v in violations)
    verdict = (
        Verdict.INCONSISTENT
        if prohibition_directly_violated or retention_violated
        else Verdict.UNDERSPECIFIED
    )
    return Inconsistency(
        inconsistency_id="",
        pattern_id=PATTERN_IDS["pi1"],
        pattern_name="Modality Contradiction",
        statement_1=website_statement,
        statement_2=vendor_statement,
        verdict=verdict,
        severity=severity,
        explanation=" ".join(violations),
        gdpr_categories=_unique_categories(
            website_statement.gdpr_categories,
            vendor_statement.gdpr_categories,
            ["Data Recipients"],
        ),
        evidence_spans=[website_statement.source_text, vendor_statement.source_text],
    )


def _pi1_cross_dedup_results(results: list[Inconsistency]) -> list[Inconsistency]:
    """Deduplicate Π₈ by website source sentence."""
    MAX_PER_WEBSITE_SENTENCE = PI8_MAX_PER_WEBSITE_SENTENCE
    website_groups: dict[str, list[Inconsistency]] = {}
    for result in results:
        ws_key = result.statement_1.source_text.strip()
        website_groups.setdefault(ws_key, []).append(result)

    deduped: list[Inconsistency] = []
    severity_rank = {Severity.CRITICAL: 3, Severity.HIGH: 2, Severity.MEDIUM: 1, Severity.LOW: 0}
    for ws_text, group in website_groups.items():
        group.sort(
            key=lambda r: (
                severity_rank.get(r.severity, 0),
                r.explanation.count(". "),
                len(r.statement_2.source_text),
            ),
            reverse=True,
        )
        deduped.extend(group[:MAX_PER_WEBSITE_SENTENCE])

    if len(deduped) < len(results):
        print(f"  Π₈ grouping: kept {len(deduped)} of {len(results)} (max {MAX_PER_WEBSITE_SENTENCE} per website sentence)")

    for index, result in enumerate(deduped):
        result.inconsistency_id = f"pi1_cross_{index}"

    return deduped


def _pi1_cross_policy(
    website_source: list[PPS] | nx.MultiDiGraph,
    vendor_source: list[PPS] | nx.MultiDiGraph | None = None,
) -> list[Inconsistency]:
    """Cross-policy arm of Π₁ (former Π₈).

    When a merged graph is passed (vendor_source=None), uses graph-neighborhood
    alignment to scope comparisons to semantically related data-type clusters.
    Falls back to flat cross-product when PPS lists are passed directly.
    """
    use_graph = isinstance(website_source, nx.MultiDiGraph) and vendor_source is None

    if use_graph:
        return _pi1_cross_graph_aware(website_source)
    return _pi1_cross_flat(website_source, vendor_source)


def _pi1_cross_graph_aware(merged_graph: nx.MultiDiGraph) -> list[Inconsistency]:
    """Graph-neighborhood Π₈: only compare within aligned data-type clusters."""
    from graph_neighborhoods import get_aligned_pairs, get_bridge_flows

    aligned_pairs = get_aligned_pairs(merged_graph)
    bridge_flow_types: set[str] = set()
    # Per-pair bridge evidence: {(website_pps_id, vendor_pps_id): data_type}
    # enables marking Π₈ bridge_confirmed per finding instead of per whole
    # neighborhood, so a single bridge flow doesn't taint all pairs in a
    # broad "personal data" cluster.
    bridge_pairs: dict[tuple[str, str], str] = {}
    vendor_name = merged_graph.graph.get("vendor_name", "")
    if vendor_name:
        for flow in get_bridge_flows(merged_graph, vendor_name):
            bridge_flow_types.add(flow.data_type)
            bridge_pairs[(flow.website_pps.id, flow.vendor_pps.id)] = flow.data_type

    results: list[Inconsistency] = []
    seen: set[tuple[str, str]] = set()
    total_comparisons = 0

    # Assign scopes to all statements using regex classifier
    # (fast, no LLM needed — handles product/feature/audience contexts)
    from scope_classifier import assign_scopes
    all_stmts_for_scope: list[PPS] = []
    for ap in aligned_pairs:
        all_stmts_for_scope.extend(ap.website.statements)
        all_stmts_for_scope.extend(ap.vendor.statements)
    scope_count = assign_scopes(all_stmts_for_scope)
    if scope_count:
        print(f"  Scope classifier: assigned {scope_count} non-global scopes")

    for ap in aligned_pairs:
        eligible_website = [ws for ws in _deduplicate_statements(ap.website.statements, quiet=True) if _pi1_cross_eligible_website(ws)]
        eligible_vendor = _pi1_cross_eligible_vendor(_deduplicate_statements(ap.vendor.statements, quiet=True))
        _trace("pi1_cross", {
            "event": "aligned_pair_enter",
            "data_type": ap.website.data_type,
            "alignment": ap.alignment_relation,
            "n_ws_raw": len(ap.website.statements),
            "n_vs_raw": len(ap.vendor.statements),
            "n_ws_elig": len(eligible_website),
            "n_vs_elig": len(eligible_vendor),
        })
        if not eligible_website or not eligible_vendor:
            continue

        has_bridge = ap.website.data_type in bridge_flow_types

        # For broad data types (e.g., "personal data"), only compare
        # sharing/selling prohibitions that are NOT hedged/scoped. This
        # eliminates: consent-required condition mismatches (these are
        # about SPECIFIC features like campaigns/newsletters, not general
        # data handling), scoped negations (children's, law enforcement),
        # and collection/processing prohibitions.
        is_broad_nh = ap.website.data_type in BROAD_DATA_TYPES
        if is_broad_nh:
            # Round 9 fix: also accept "process" prohibitions (broadest action
            # in the subsumption hierarchy) when they are accompanied by
            # strong blanket-prohibition markers. "We will never process
            # personal data for sale" should match any vendor processing
            # practice; previously the filter restricted to _SHARING_ACTIONS
            # and silently dropped the broader prohibition.
            _PROHIB_ACTIONS = _SHARING_ACTIONS | {"process"}
            eligible_website = [
                ws for ws in eligible_website
                if (ws.modality == Modality.PROHIBITION or ws.is_negative)
                and ws.action in _PROHIB_ACTIONS
                and not _is_hedged_or_scoped_prohibition(ws)
                and not _is_childrens_clause(ws.source_text)
            ]

        w_prohibitions = [s for s in eligible_website if s.modality == Modality.PROHIBITION or s.is_negative]
        w_commitments = [s for s in eligible_website if s.modality in (Modality.COMMITMENT, Modality.OBLIGATION)]
        v_sharing = [s for s in eligible_vendor if s.action in _SHARING_ACTIONS]
        v_processing = [s for s in eligible_vendor if s.action in _USAGE_ACTIONS]

        for ws in eligible_website:
            for vs in eligible_vendor:
                pair = (ws.id, vs.id)
                if pair in seen:
                    continue
                total_comparisons += 1
                seen.add(pair)
                inc = _pi1_cross_check_pair(ws, vs, skip_data_match=True)
                if inc is None:
                    continue
                # Per-pair bridge confirmation only. The neighborhood-level
                # has_bridge flag was previously used as a fallback label,
                # but that tainted unrelated findings in the same cluster
                # (e.g., a "do not sell email" finding got [bridge-confirmed]
                # because another pair in the email neighborhood had a real
                # handoff). Bridge evidence is strictly a property of the
                # (website_pps, vendor_pps) pair.
                pair_bridge = (ws.id, vs.id) in bridge_pairs
                if pair_bridge:
                    inc.explanation = f"[bridge-confirmed] {inc.explanation}"
                w_supporting = sorted(
                    [s for s in eligible_website if s.id != ws.id],
                    key=lambda s: (s.modality == Modality.PROHIBITION or s.is_negative, s.modality == Modality.COMMITMENT),
                    reverse=True,
                )
                v_supporting = sorted(
                    [s for s in eligible_vendor if s.id != vs.id],
                    key=lambda s: s.action in _SHARING_ACTIONS,
                    reverse=True,
                )
                # Compute ontology relationship between the two data types
                ws_data_norm = normalize_data_type(ws.data_object) or ws.data_object.lower()
                vs_data_norm = normalize_data_type(vs.data_object) or vs.data_object.lower()
                subsume_path = None
                if ws_data_norm != vs_data_norm:
                    path = _get_subsume_path(ws_data_norm, vs_data_norm)
                    if path:
                        subsume_path = " → ".join(path)
                    else:
                        path = _get_subsume_path(vs_data_norm, ws_data_norm)
                        if path:
                            subsume_path = " → ".join(path)

                inc.neighborhood_context = {
                    "data_type": ap.website.data_type,
                    "alignment": ap.alignment_relation,
                    "website_count": len(ap.website.statements),
                    "vendor_count": len(ap.vendor.statements),
                    "bridge_confirmed": pair_bridge,
                    "bridge_pair_matched": pair_bridge,
                    "neighborhood_has_bridge": has_bridge,
                    "website_prohibitions": len(w_prohibitions),
                    "website_commitments": len(w_commitments),
                    "vendor_sharing_actions": len(v_sharing),
                    "vendor_processing_actions": len(v_processing),
                    "subsume_path": subsume_path,
                    "website_data_type": ws_data_norm,
                    "vendor_data_type": vs_data_norm,
                    "website_action": ws.action,
                    "vendor_action": vs.action,
                    # Supporting-clause truncation widened 120 → 300 chars so
                    # Stage-2 cluster verification sees "…except as outlined
                    # in this policy" and similar qualifiers that typically
                    # sit past char 120 in real policy prose.
                    "website_supporting": [
                        f"{s.action}/{s.modality.name}/{s.data_object}: {s.source_text[:300]}"
                        for s in w_supporting[:5]
                    ],
                    "vendor_supporting": [
                        f"{s.action}/{s.modality.name}/{s.data_object}: {s.source_text[:300]}"
                        for s in v_supporting[:5]
                    ],
                }
                results.append(inc)

    print(f"  Π₈ graph-aware: {len(aligned_pairs)} aligned neighborhoods, "
          f"{total_comparisons} comparisons, {len(results)} raw findings")

    return _pi1_cross_dedup_results(results)


def _pi1_cross_flat(
    website_source: list[PPS] | nx.MultiDiGraph,
    vendor_source: list[PPS] | nx.MultiDiGraph | None,
) -> list[Inconsistency]:
    """Flat cross-product Π₈ (backward-compatible path for PPS list inputs)."""
    website_statements, vendor_statements = _coerce_cross_policy_statements(website_source, vendor_source)
    results: list[Inconsistency] = []
    seen: set[tuple[str, str]] = set()
    eligible_vendor = _pi1_cross_eligible_vendor(vendor_statements)

    for website_statement in website_statements:
        if not _pi1_cross_eligible_website(website_statement):
            continue
        for vendor_statement in eligible_vendor:
            pair = (website_statement.id, vendor_statement.id)
            if pair in seen:
                continue
            inc = _pi1_cross_check_pair(website_statement, vendor_statement, skip_data_match=False)
            if inc is None:
                continue
            seen.add(pair)
            results.append(inc)

    return _pi1_cross_dedup_results(results)


def _deduplicate_statements(statements: list[PPS], max_per_text: int = 3, quiet: bool = False) -> list[PPS]:
    """Collapse near-duplicate PPS statements by source_text.

    Omnibus policies (Microsoft, Google) generate many PPS from near-identical
    state-specific clauses. Deduplicating before pattern matching prevents O(n²)
    blowup without affecting genuine findings.
    """
    text_counts: dict[str, int] = {}
    deduped: list[PPS] = []
    for stmt in statements:
        key = stmt.source_text.strip()[:200]
        count = text_counts.get(key, 0)
        if count >= max_per_text:
            continue
        text_counts[key] = count + 1
        deduped.append(stmt)
    if not quiet and len(deduped) < len(statements):
        print(f"    Statement dedup: {len(statements)} → {len(deduped)} "
              f"(removed {len(statements) - len(deduped)} near-duplicates)")
    return deduped


def run_intra_patterns(
    statements_source: list[PPS] | nx.MultiDiGraph,
    skip_pi6: bool = False,  # kept for backwards-compatible callsites; Π₆ removed
    neighborhoods: list | None = None,
) -> list[Inconsistency]:
    """Run all intra-policy patterns (Round 11 numbering: Π₁, Π₂, Π₃, Π₄).

    Π₆, Π₇, Π₉ were pruned in Round 9–11 (0% genuine rate). See the
    comment block above pi7_vagueness_asymmetry's former location for the
    Π₇ rationale.

    When neighborhoods are provided, patterns run per-neighborhood for
    ontology-aware scoping — statements about "email address" and "contact
    information" are compared within the same neighborhood. Falls back to
    flat-list behavior when no neighborhoods are available.
    """
    del skip_pi6

    results: list[Inconsistency] = []
    patterns_to_run = [
        (PATTERN_IDS["pi1"], pi1_modality_contradiction),
        (PATTERN_IDS["pi2"], pi2_exclusivity_violation),
        (PATTERN_IDS["pi3"], pi3_condition_asymmetry),
        (PATTERN_IDS["pi4"], pi4_temporal_contradiction),
    ]

    if neighborhoods:
        from graph_neighborhoods import get_neighborhood_context
        # Pi1/Pi2/Pi5 all run on the full statement set: each relies on
        # ontology-aware matching (parent/child subsumption) that a
        # per-neighborhood slice breaks — e.g., a prohibition on "personal
        # data" and a commitment on "browsing history" live in different
        # neighborhoods. Pi4 still runs per-neighborhood because its
        # (actor, action, data) grouping is naturally scoped.
        full_statements = _coerce_statements(statements_source)
        flat_full_set_patterns = {
            PATTERN_IDS["pi1"],
            PATTERN_IDS["pi2"],
            PATTERN_IDS["pi4"],
        }
        for pattern_id, pattern in patterns_to_run:
            pattern_findings: list[Inconsistency] = []
            if pattern_id in flat_full_set_patterns:
                pattern_findings.extend(pattern(full_statements))
                seen: set[tuple[str, str, str]] = set()
                deduped: list[Inconsistency] = []
                for f in pattern_findings:
                    key = (f.pattern_id, f.statement_1.id, f.statement_2.id)
                    if key not in seen:
                        seen.add(key)
                        deduped.append(f)
                print(f"  {pattern_id}: {len(deduped)} findings (flat, full-set)")
                results.extend(deduped)
                continue
            for nh in neighborhoods:
                ctx = get_neighborhood_context(nh)
                if pattern_id == PATTERN_IDS["pi2"] and not ctx["has_exclusivity"]:
                    continue
                if pattern_id == PATTERN_IDS["pi4"] and not ctx["has_retention"]:
                    continue
                nh_findings = pattern(nh.statements)
                pattern_findings.extend(nh_findings)
            seen: set[tuple[str, str, str]] = set()
            deduped: list[Inconsistency] = []
            for f in pattern_findings:
                key = (f.pattern_id, f.statement_1.id, f.statement_2.id)
                if key not in seen:
                    seen.add(key)
                    deduped.append(f)
            print(f"  {pattern_id}: {len(deduped)} findings ({len(neighborhoods)} neighborhoods)")
            results.extend(deduped)
    else:
        for pattern_id, pattern in patterns_to_run:
            findings = pattern(statements_source)
            print(f"  {pattern_id}: {len(findings)} findings")
            results.extend(findings)
    return results


def run_cross_patterns(
    website_source: list[PPS] | nx.MultiDiGraph,
    vendor_source: list[PPS] | nx.MultiDiGraph | None = None,
) -> list[Inconsistency]:
    """Run the four cross-policy patterns (Π₁ modality, Π₂ exclusivity,
    Π₃ condition, Π₄ temporal).

    Each pattern dispatches to its cross-policy branch when called with
    (website_source, vendor_source) — or with a merged graph on the
    website slot — via ``_is_cross_policy_call``. The old doc saying
    "cross-policy lives only in Π₁" was stale; every pattern has a
    cross-policy arm since the 2026-04-24 rewrite.
    """
    findings: list[Inconsistency] = []
    r1 = pi1_modality_contradiction(website_source, vendor_source)
    print(f"  {PATTERN_IDS['pi1']} (cross): {len(r1)} findings")
    findings.extend(r1)
    r2 = pi2_exclusivity_violation(website_source, vendor_source)
    print(f"  {PATTERN_IDS['pi2']} (cross): {len(r2)} findings")
    findings.extend(r2)
    r3 = pi3_condition_asymmetry(website_source, vendor_source)
    print(f"  {PATTERN_IDS['pi3']} (cross): {len(r3)} findings")
    findings.extend(r3)
    r4 = pi4_temporal_contradiction(website_source, vendor_source)
    print(f"  {PATTERN_IDS['pi4']} (cross): {len(r4)} findings")
    findings.extend(r4)
    return findings


def _deduplicate_results(results: list[Inconsistency]) -> list[Inconsistency]:
    """Remove findings with identical (pattern_id, S1 source_text, S2 source_text)."""
    seen: set[tuple[str, str, str]] = set()
    deduped: list[Inconsistency] = []
    for result in results:
        key = (
            result.pattern_id,
            result.statement_1.source_text.strip(),
            result.statement_2.source_text.strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    if len(deduped) < len(results):
        print(f"  Dedup: removed {len(results) - len(deduped)} exact duplicates ({len(results)} → {len(deduped)})")
    return deduped


def run_all_patterns(
    website_source: list[PPS] | nx.MultiDiGraph,
    vendor_source: list[PPS] | nx.MultiDiGraph | None = None,
    cross_policy: bool = False,
) -> list[Inconsistency]:
    """Run intra-policy and optional cross-policy patterns.

    Performance optimization: vendor statements are deduplicated before pattern
    matching since omnibus policies (Microsoft, Google) generate many
    near-identical state-specific clauses.
    """

    if cross_policy and vendor_source is None:
        # Cross-policy only: the paper's four patterns (Π₁–Π₄) are
        # cross-party by design, so the intra-policy passes just burn
        # time producing findings we never use and pollute the dedup
        # set. Run only run_cross_patterns on the merged graph (or on
        # the two statement lists coerced out of it).
        use_graph = isinstance(website_source, nx.MultiDiGraph)
        if use_graph:
            from graph_neighborhoods import get_data_neighborhoods
            website_nhs = get_data_neighborhoods(website_source, "1p", expand_ancestors=False)
            vendor_nhs = get_data_neighborhoods(website_source, "3p", expand_ancestors=False)
            print(f"\n  Graph neighborhoods: {len(website_nhs)} website, {len(vendor_nhs)} vendor")
            print("\nRunning cross-policy patterns...")
            results = run_cross_patterns(website_source)
        else:
            website_statements, vendor_statements = _coerce_cross_policy_statements(website_source)
            vendor_statements = _deduplicate_statements(vendor_statements)
            print("\nRunning cross-policy patterns...")
            results = run_cross_patterns(website_statements, vendor_statements)
        results = _deduplicate_results(results)
        print(f"\nTotal inconsistencies found: {len(results)}")
        return results

    print("\nRunning intra-policy patterns (website)...")
    results = run_intra_patterns(website_source)
    if vendor_source is not None:
        print("\nRunning intra-policy patterns (vendor)...")
        vendor_stmts = _coerce_statements(vendor_source) if not isinstance(vendor_source, list) else vendor_source
        vendor_stmts = _deduplicate_statements(vendor_stmts)
        results.extend(run_intra_patterns(vendor_stmts, skip_pi6=True))
        print("\nRunning cross-policy patterns...")
        results.extend(run_cross_patterns(website_source, vendor_stmts))
    results = _deduplicate_results(results)
    print(f"\nTotal inconsistencies found: {len(results)}")
    return results
