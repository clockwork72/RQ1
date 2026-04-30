"""Post-extraction scope classifier for PPS statements.

Assigns a product/feature scope to each PPS based on its source text.
Uses an LLM (via Ollama Pro or llama.cpp) for accuracy, with a regex
fallback for speed when no LLM is available.

The scope is used in pattern matching to prevent cross-context
comparisons: a donation-context clause should not be compared against
a vendor's general data processing.
"""

from __future__ import annotations

import json
import os
import re
from .schema import PPS


SCOPE_PROMPT = """Identify the SCOPE of this privacy policy clause — what specific product, feature, audience, or jurisdiction it applies to.

**DEFAULT TO "global" WHEN NO SPECIFIC PRODUCT, FEATURE, OR AUDIENCE IS NAMED.** Cross-policy analysis treats incompatible scopes as non-contradictions, so over-scoping a clause silently suppresses legitimate findings downstream. Err on the side of "global" when the qualifier is vague, implicit, or merely contextual.

Return ONLY a JSON object: {{"scope": "..."}}

Canonical scope vocabulary (pick ONE, lowercase, underscores between words, no hyphens, no spaces):

  global                       — applies to all users / all data (DEFAULT)

  Product-specific examples:
    vpn_pro, ai_chat, swiftkey, outlook, xbox, teams, edge,
    online_shop, gamemaker, hulu, disney_plus, aria

  Feature-specific examples:
    donation_processing, community_sharing, newsletter, account_creation,
    payment_processing, promotional_campaign, ai_features, content_moderation

  Audience-specific examples:
    children_under_13, california_residents, eu_residents, uk_residents,
    enterprise_dpa, job_applicants, account_holders, students, healthcare_providers

Rules:
- Pick exactly ONE scope value.
- 1-3 tokens, lowercase, underscores between words (e.g. `children_under_13`, not `children under 13` or `children-under-13`).
- If the clause has NO explicit product/feature/audience qualifier, return "global".
- If the clause qualifier is vague ("some users", "certain contexts", "where applicable"), return "global".
- If multiple audiences apply, pick the most specific (e.g. a clause about "enterprise customers in the EU" → `enterprise_dpa`).
- If the clause names a product or feature NOT listed above, use a short snake_case descriptor of your own (e.g. "ai_chat_beta", "loyalty_program"). Do not invent generic labels — be specific.

Examples:

Clause: "We collect your email address to send order confirmations."
Output: {{"scope": "global"}}

Clause: "For users of our SwiftKey keyboard, we collect typing patterns to improve auto-correct."
Output: {{"scope": "swiftkey"}}

Clause: "We do not knowingly collect personal information from children under 13 without verified parental consent."
Output: {{"scope": "children_under_13"}}

Clause: {clause}"""


# Regex fallback for when no LLM is available
_SCOPE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("donation_processing", re.compile(
        r"(?:make\s+a\s+donation|donate|donation|contribution)", re.I)),
    ("community_sharing", re.compile(
        r"(?:figma\s+community|posting\s+to\s+\w+\s+community|link\s+sharing"
        r"|to\s+the\s+general\s+public)", re.I)),
    ("account_creation", re.compile(
        r"(?:sign(?:ing)?\s+up\s+for\s+(?:a|an|your)\s+\w+\s*account"
        r"|creat(?:e|ing)\s+(?:a|an|your)\s+\w+\s*account"
        r"|you\s+(?:may|can)\s+access\s+(?:some\s+of\s+)?(?:our|these)\s+services?\s+by)", re.I)),
    ("newsletter", re.compile(
        r"(?:sign\s+up\s+for\s+(?:our\s+)?newsletter|subscribe\s+to\s+(?:our\s+)?newsletter"
        r"|newsletter\s+subscription)", re.I)),
    ("payment_processing", re.compile(
        r"(?:make\s+a\s+purchase|payment\s+(?:method|information|processing)"
        r"|credit\s+card\s+(?:information|number|details)"
        r"|(?:stripe|paypal|payment\s+processor))", re.I)),
    ("promotional_campaign", re.compile(
        r"(?:promotional\s+campaign|contest|sweepstake|survey|giveaway)", re.I)),
    ("children_under_13", re.compile(
        r"(?:children\s+under\s+(?:the\s+age\s+of\s+)?1[36]|coppa|minors"
        r"|children'?s\s+online\s+privacy)", re.I)),
    ("california_residents", re.compile(
        r"(?:california\s+residents?|ccpa|cpra|cal\.\s*civ)", re.I)),
    ("eu_residents", re.compile(
        r"(?:eu\s+residents?|eea\s+residents?|gdpr|european\s+(?:union|economic\s+area))", re.I)),
    ("enterprise_dpa", re.compile(
        r"(?:enterprise|business\s+customers?|data\s+processing\s+(?:agreement|addendum)"
        r"|B2B|DPA\b)", re.I)),
]

# Product names that indicate product-specific scope
_PRODUCT_NAMES = re.compile(
    r"\b(?:VPN\s+Pro|AI\s+Chat|Aria|GameMaker|GX\.gear|SwiftKey|Outlook"
    r"|Xbox|Teams|Edge|Canva\s+Education|Hulu|Disney\+)\b",
    re.IGNORECASE,
)


# Round 9 fix: clauses containing strong blanket-prohibition markers
# ("never", "full stop", "under no circumstances", "we will never") should
# stay scope=global even when they happen to mention ancillary feature
# keywords. Otherwise the regex over-assigns feature scopes (e.g.,
# "we do not sell personal data. Any sharing is governed by data processing
# agreements" → enterprise_dpa), masking cross-policy Π₈ violations.
_BLANKET_PROHIBITION_RE = re.compile(
    r"\b(?:"
    r"will\s+never|never\s+sell|never\s+share|do\s+not\s+sell|do\s+not\s+share"
    r"|under\s+no\s+circumstances|in\s+no\s+event|full\s+stop"
    r"|strictly\s+prohibited|prohibited\s+under\s+this\s+policy"
    r"|is\s+never\s+(?:sold|shared|transferred|disclosed)"
    r")\b",
    re.IGNORECASE,
)


def classify_scope_regex(source_text: str) -> str:
    """Fast regex-based scope classification (no LLM needed)."""
    # Blanket-prohibition guard: don't attach a non-global scope to a clause
    # whose primary content is an unconditional prohibition. The mention of
    # feature keywords in the surrounding sentences should not shrink the
    # scope of the prohibition itself.
    if _BLANKET_PROHIBITION_RE.search(source_text):
        return "global"

    # Check product names first
    product_match = _PRODUCT_NAMES.search(source_text)
    if product_match:
        name = product_match.group().lower().replace(" ", "_").replace(".", "")
        return name

    # Check feature/audience patterns
    for scope_name, pattern in _SCOPE_PATTERNS:
        if pattern.search(source_text):
            return scope_name

    return "global"


def classify_scope_llm(
    source_text: str,
    client,
    model: str = "gemma4:31b",
) -> str:
    """LLM-based scope classification (more accurate, slower)."""
    prompt = SCOPE_PROMPT.format(clause=source_text[:400])
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            result = json.loads(m.group())
            return result.get("scope", "global")
        return json.loads(raw).get("scope", "global")
    except Exception:
        # Fallback to regex
        return classify_scope_regex(source_text)


def assign_scopes(
    statements: list[PPS],
    use_llm: bool = False,
    client=None,
    model: str = "gemma4:31b",
) -> int:
    """Assign scopes to PPS statements in place.

    Returns the number of statements that got non-global scopes.
    """
    changed = 0
    for pps in statements:
        if not pps.source_text:
            continue
        current = (pps.scope or "global").strip().lower()
        if current != "global":
            continue  # already has a specific scope

        if use_llm and client:
            new_scope = classify_scope_llm(pps.source_text, client, model)
        else:
            new_scope = classify_scope_regex(pps.source_text)

        if new_scope != "global":
            pps.scope = new_scope
            changed += 1

    return changed
