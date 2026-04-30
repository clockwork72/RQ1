"""Unified prompt registry for When-Policies-Disagree.

All prompts used by the extractor, verifier, and evaluation harnesses live
here. Importing from this module guarantees that benchmarks, the production
pipeline, and the open-science notebooks share the exact same prompt strings.

Exports
-------
EXTRACTION_PROMPT
    Current extractor prompt (default for the pipeline).
EXTRACTION_PROMPT_FEWSHOT
    Extractor prompt with worked examples; used in the qwen3-vl-instruct
    few-shot configuration of the extractor leaderboard
    (Evaluation §4.1).
VERIFIER_PROMPT
    Current 3-verdict verifier prompt
    (``inconsistent`` / ``unspecified`` / ``non_conflict``). Used by the
    production verifier and the perturbation evaluation
    (Evaluation §4.2.1).
VERIFIER_PROMPT_LEGACY
    Older 3-verdict prompt
    (``hard_contradiction`` / ``soft_tension`` / ``non_conflict``) with a
    ``non_conflict`` default. Used to replicate the multi-model
    verdict-agreement experiment (Evaluation §4.2.2).

These strings are the single source of truth. Any script in this repository
that runs an LLM call must import the relevant constant from here.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a privacy policy analyst. Extract every Privacy Practice Statement from the clause below.

Return only a JSON array. Each item must have exactly these keys:
- actor
- action
- modality
- data_object
- purpose
- recipient
- condition
- temporality
- temporality_value
- is_negative
- scope

Rules:

1. Extract every distinct practice. If one clause names multiple data types, emit multiple JSON objects. If the clause describes no data practice at all (definitions, section headers, glossary text, CCPA category tables), return [].

2. `action` must be one of: collect, use, share, sell, retain, delete, transfer, process, access_right, deletion_right, optout_right, portability_right.

3. `condition` must be one of: by_default, upon_consent, if_opted_in, unless_opted_out, when_required, jurisdictional, unspecified.

4. `temporality` must be one of: specific_duration, until_purpose, until_account_deletion, indefinite, unspecified. When `temporality = specific_duration`, fill `temporality_value` with a short phrase like "30 days", "26 months", "2 years".

5. `modality` is the most important field. It must be one of: PROHIBITION, OBLIGATION, COMMITMENT, PERMISSION, POSSIBILITY, HEDGED, UNSPECIFIED.

   - PROHIBITION: the company explicitly refuses to perform the action, and the refusal is (a) BLANKET — applies broadly, not limited to one feature/product/context, (b) UNCONDITIONAL — no "without consent", "except as described", "unless you opt in", and (c) ABOUT THE COMPANY'S OWN PRACTICE — not about what a third party does. Pair with `is_negative = true`.

   - COMMITMENT: the company DOES or WILL do something, including ALL of the following "negative-sounding" cases where the refusal is scoped or technical:
     * scoped negations: "We do not share X without your consent"
     * data-minimization promises: "We only collect what is necessary"
     * technical facts: "The token is not retained on our servers"
     * feature disclaimers: "In AI Chat, we do not share account data"
     * delegation: "We do not process credit cards — we use Stripe"
     * third-party descriptions: "Your telecom operator does not share ..."
     In all of these cases `is_negative = false`.

   - PERMISSION: something is allowed but not guaranteed ("may share"). ALSO use PERMISSION for USER RIGHTS — "You have the right to delete your data" → modality=PERMISSION, action=deletion_right.

   - OBLIGATION: a mandatory requirement imposed on the company. "We are required by law to retain records."

   - HEDGED: qualified with "generally", "typically", "usually", "in most cases".

   - UNSPECIFIED: cannot determine from the text.

   Common MISTAKES to AVOID:
   - "You have the right to object/delete/access" → PERMISSION, not PROHIBITION.
   - "We do not require your consent for X" → COMMITMENT (legal basis), not PROHIBITION.
   - "hold in confidence and not disclose" → COMMITMENT (confidentiality), not PROHIBITION.
   - "We process data only for the purposes described" → COMMITMENT with exclusivity, not PROHIBITION.
   - "We do not share ... without your explicit consent" → COMMITMENT (the escape clause makes it non-blanket).
   - "We do not sell personal information of students using [Product]" → COMMITMENT (product-scoped, not blanket).
   - "This privacy policy does not cover third-party apps" → DISCLAIMER; return [] or use COMMITMENT.
   - "Personal information means..." → DEFINITION; return [].
   - Section headers, table labels, glossary entries → return [].

6. `actor` normalization:
   - "FirstParty" for the company operating the policy ("we", "our company", the named company name)
   - "DataSubject" for the user ("you", "users", "customers", "consumers")
   - A proper name for named third parties ("Google", "Trustpilot") — use the shortest unambiguous form
   - "ThirdParty" for unnamed / generic third parties ("third-party service providers", "advertising partners")

7. `data_object` — use canonical types where applicable. Preferred vocabulary:
   personal data, contact information, email address, phone number, name, postal address,
   device information, ip address, device id, advertising id, browser type, operating system,
   device type, screen resolution, language settings, timezone,
   behavioral data, browsing history, search queries, purchase history, clickstream data,
   app usage, interaction data,
   financial data, payment information, transaction history, bank account,
   health data, medical records, medical conditions, biometric data,
   geolocation, precise geolocation, coarse geolocation, country,
   account data, username, password, user preferences, account settings,
   demographic data, age, date of birth, gender, race or ethnicity,
   user-generated content, review, photo, video, message, feedback,
   tracking technology, cookie id, pixel, session id, browser fingerprint,
   government id, ssn, driver license, passport number,
   professional data, employment information, education information,
   communication data, call records, sms data, email content.
   If the clause names a subtype not in this list, use a short (2-4 word) normalized label. Do NOT use long verbatim phrases as `data_object` values.

8. `scope` is a short (2-6 words) descriptor of WHO or WHAT the practice applies to. It gates cross-clause pattern matching — two clauses with incompatible scopes are NOT contradictions, so OVER-SCOPING silently suppresses legitimate findings downstream. **When no specific product, feature, or audience is named, return "global".**
   - "global" — applies to all users / all data broadly (DEFAULT)
   - "children under 13" — COPPA / children's clauses
   - "EU residents" / "California residents" — jurisdictional carve-outs
   - "enterprise DPA" — B2B data-processing addendum terms, not consumer policy
   - "SwiftKey users" / "Outlook users" — product-specific clauses inside omnibus policies
   - "advertising partners" — clauses about a specific recipient class
   - "job applicants" — HR / recruiting clauses
   - "account holders" — registered-user-only clauses
   Keep scope short, lowercase. If multiple audiences apply, pick the most specific.

9. Output valid JSON only. No markdown, no prose, no code fences.

Examples:

Clause: "We do not sell your personal information to any third party."
Output: [{{"actor": "FirstParty", "action": "sell", "modality": "PROHIBITION", "data_object": "personal data", "purpose": "", "recipient": "third parties", "condition": "unspecified", "temporality": "unspecified", "temporality_value": "", "is_negative": true, "scope": "global"}}]

Clause: "You have the right to request deletion of your account data at any time by contacting support."
Output: [{{"actor": "DataSubject", "action": "deletion_right", "modality": "PERMISSION", "data_object": "account data", "purpose": "", "recipient": "", "condition": "unspecified", "temporality": "unspecified", "temporality_value": "", "is_negative": false, "scope": "global"}}]

Clause: "We do not share your browsing history with advertisers without your consent; where you have consented, we share it with Google Analytics for analytics for up to 26 months."
Output: [{{"actor": "FirstParty", "action": "share", "modality": "COMMITMENT", "data_object": "browsing history", "purpose": "analytics", "recipient": "Google Analytics", "condition": "upon_consent", "temporality": "specific_duration", "temporality_value": "26 months", "is_negative": false, "scope": "global"}}]

Section: {section}
Clause: {clause}
"""


EXTRACTION_PROMPT_FEWSHOT = """You are a privacy policy analyst. Extract every Privacy Practice Statement from the clause below.

Return only a JSON array. Each item must have exactly these keys:
- actor
- action
- modality
- data_object
- purpose
- recipient
- condition
- temporality
- temporality_value
- is_negative
- scope

Rules:

1. Extract every distinct practice. **Many clauses are list items, sentence fragments, or section headers leading into a list. When a fragment plainly describes a practice (e.g., "Title, first and last name, address, email address" under a contact-form heading), still emit a PPS — extract from the fragment treating the section header as the parent context.** If the clause describes no data practice (a glossary definition, contact info, terms-of-use boilerplate), return [].

2. `action` must be one of: collect, use, share, sell, retain, delete, transfer, process, access_right, deletion_right, optout_right, portability_right.

3. `condition` must be one of: by_default, upon_consent, if_opted_in, unless_opted_out, when_required, jurisdictional, unspecified.

4. `temporality` must be one of: specific_duration, until_purpose, until_account_deletion, indefinite, unspecified.

5. `modality` must be one of: PROHIBITION, OBLIGATION, COMMITMENT, PERMISSION, POSSIBILITY, HEDGED, UNSPECIFIED.
   - PROHIBITION = blanket, unconditional refusal about the company's own practice (is_negative=true).
   - COMMITMENT = the company does or will do something; ALSO scoped negations like "We do not share X without your consent" (is_negative=false), data-minimization promises, delegations.
   - PERMISSION = "may", or USER RIGHTS — opt-out, deletion, access, portability (action=*_right).
   - OBLIGATION = mandatory legal/regulatory requirement on the company.
   - HEDGED = "generally", "typically".

6. `actor` normalization:
   - "FirstParty" for the company (we, us, named operator).
   - "DataSubject" for the user (you, customers).
   - A short proper name for named third parties.
   - "ThirdParty" for unnamed third parties.

7. `data_object` — short canonical label (1-4 words). Prefer general categories: personal data, contact information, email address, ip address, device id, browser type, behavioral data, browsing history, geolocation, account data, payment information, health data, biometric data, tracking technology, user-generated content, etc. **For lists of specific data types, emit one PPS per type, OR a single PPS with the most general matching category if the list is long.**

8. `scope` — short descriptor; default "global".

9. Output valid JSON only. No markdown, no code fences.

## Examples

### Example 1: Section-header / list-stem clause that introduces collection
Clause: "**. We may collect information from you automatically when you use our Services. This Personal Information may include the following:**"
Output: [{{"actor": "FirstParty", "action": "collect", "modality": "COMMITMENT", "data_object": "personal data", "purpose": "automatic collection", "recipient": "", "condition": "unspecified", "temporality": "unspecified", "temporality_value": "", "is_negative": false, "scope": "global"}}]

### Example 2: Sentence fragment with parent contact-form context
Section: "Data Protection Information"
Clause: "As a user of **contact forms**: Title, first and last name, address, email address and telephone number and other information regarding the contact."
Output: [{{"actor": "FirstParty", "action": "collect", "modality": "COMMITMENT", "data_object": "contact information", "purpose": "contact form submissions", "recipient": "", "condition": "unspecified", "temporality": "unspecified", "temporality_value": "", "is_negative": false, "scope": "global"}}]

### Example 3: Opt-out right with verbose phrasing
Clause: "You may be able to opt out of receiving personalized advertisements from other companies who are members of the Network Advertising Initiative or who subscribe to the Digital Advertising Alliance's Self-Regulatory Principles for Online Behavioral Advertising."
Output: [{{"actor": "DataSubject", "action": "optout_right", "modality": "PERMISSION", "data_object": "behavioral data", "purpose": "opt out of personalized ads via NAI/DAA", "recipient": "", "condition": "unspecified", "temporality": "unspecified", "temporality_value": "", "is_negative": false, "scope": "global"}}]

### Example 4: Implicit practice (purpose listed without verb)
Clause: "To provide access to online or offline events, content and blogs, forums and similar online and interactive experiences"
Output: [{{"actor": "FirstParty", "action": "use", "modality": "COMMITMENT", "data_object": "personal data", "purpose": "event and content provision", "recipient": "", "condition": "unspecified", "temporality": "unspecified", "temporality_value": "", "is_negative": false, "scope": "global"}}]

### Example 5: Genuine non-practice (return empty)
Clause: "**c. International Transfers of Personal Data:**"
Output: []

### Example 6: Scoped negation (COMMITMENT, not PROHIBITION)
Clause: "Quantcast will not share Advertise client data with, or use Advertise client data for, any other Quantcast client."
Output: [{{"actor": "FirstParty", "action": "share", "modality": "COMMITMENT", "data_object": "personal data", "purpose": "scoped negation - not shared with other clients", "recipient": "other Quantcast clients", "condition": "unspecified", "temporality": "unspecified", "temporality_value": "", "is_negative": false, "scope": "Advertise client data"}}]

Section: {section}
Clause: {clause}
"""


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

VERIFIER_PROMPT = """You are a strict privacy-policy auditor. A pattern-matching system has flagged two clauses \
— one from a first-party policy, one from its declared third-party policy — as a potential \
cross-policy inconsistency. Your job is to decide whether a reasonable reviewer, reading \
both clauses in their surrounding context, would conclude the third party's stated practice \
contradicts or exceeds what the first party committed to its users.

The two clauses come from DIFFERENT companies BY DESIGN. That alone is not a disqualifier. \
The question is: on the same (or ontology-aligned) data type, does the third party go \
beyond, operate under weaker conditions than, or directly contradict what the first party \
promised?

Return exactly one of three verdicts:

  • **inconsistent**  — the third-party practice contradicts or exceeds the first-party \
    commitment on the same data type, in a way a reasonable reviewer would flag. This \
    covers both clear logical contradictions and softer tensions where the third party \
    goes meaningfully beyond what the first-party clause led users to expect.
  • **unspecified**   — the evidence is ambiguous. Key fields are missing, the clauses \
    talk past each other on scope/audience/jurisdiction, or the extracted modality is \
    unreliable and the surrounding context is not decisive either way.
  • **non_conflict**  — the candidate is a false alarm. One of the decision rules below \
    applies, or the clauses are simply compatible.

=== Skepticism ===

The pattern stage is tuned for recall, so most candidates need a hard second look. Apply \
MODERATE skepticism by default; raise to HIGH when the surrounding context is ambiguous or \
when the extraction looks wrong (see decision rules below). Prefer **unspecified** over \
**inconsistent** when you are on the fence.

You are currently evaluating **{pattern_id} — {pattern_name}**.
{pattern_description}

=== Inputs ===

--- Clause 1: first-party ({policy_1_label}) ---
{source_text_1}

Surrounding context:
{context_1}

Fields: actor={actor_1} | action={action_1} | modality={modality_1} | data={data_1} | \
purpose={purpose_1} | condition={condition_1}

--- Clause 2: third-party ({policy_2_label}) ---
{source_text_2}

Surrounding context:
{context_2}

Fields: actor={actor_2} | action={action_2} | modality={modality_2} | data={data_2} | \
purpose={purpose_2} | condition={condition_2}

Pattern finding: {pattern_explanation}

{neighborhood_context}
=== Decision rules ===

USE THE SURROUNDING CONTEXT — not just the extracted clause or the extracted fields — to \
decide what the author meant. Extraction errors are common; if the context contradicts the \
extracted modality or action, trust the context.

SCOPE AWARENESS. Every PPS carries a `scope` field (e.g. "global", "children_under_13", \
"enterprise_dpa", "eu_residents", "swiftkey_users"). If the two clauses carry incompatible \
scopes and neither is "global", default to **non_conflict** with \
category=`scope_mismatch`.

PUBLIC vs PRIVATE DATA. If one side refers to voluntarily published content (public reviews, \
public profiles, posts) and the other to private data (contact info, billing, account data), \
these are categorically different data types — default to **non_conflict** with \
category=`different_data_types`.

DATA TYPE HIERARCHY. Privacy policies use both broad parent types ("personal data", \
"contact information", "device data") and narrow child types ("email address", "phone \
number", "IP address"). A first-party commitment on a PARENT type legally binds third-party \
processing of any CHILD subtype: if the first party promises consent-only for "personal \
data" and the vendor collects "email addresses" by default, that IS a contradiction, NOT a \
different-data-types mismatch. The context block below may include a "DATA TYPE \
RELATIONSHIP" paragraph spelling out the specific parent→child path for this pair; when \
present, treat it as authoritative and do NOT reject with category=`different_data_types`.
ONLY reject as `different_data_types` when the two data types are ontologically unrelated \
(e.g. "image" vs "contact information", "browser type" vs "health data"), not when they sit \
on the same parent-child branch.

Label the candidate **non_conflict** if ANY of the following applies (dominant false-positive \
patterns from our deep audits):

1. MODALITY MISEXTRACTION. The extracted modality (usually PROHIBITION) is wrong because the \
   source text is actually:
   - a user-right clause ("you have the right to delete / object / port / restrict / access");
   - a legal-basis statement ("we do not require your consent", "legitimate interest");
   - a deletion or retention process ("we delete data within 30 days", "retained for 12 months");
   - a CCPA category table entry or section header;
   - a glossary definition ("personal data means …").

2. PRODUCT / AUDIENCE MISMATCH. The first-party clause is scoped to a product or audience \
   the third-party service does not actually serve (for example, a first-party omnibus \
   policy covering SwiftKey / Outlook / Teams matched against a consumer-web ad-tech vendor; \
   a B2B DPA clause matched against a consumer analytics SDK). Emit \
   category=`scope_mismatch`. Being from different companies alone is NEVER grounds for this \
   rule.

3. SCOPED NEGATION TREATED AS BLANKET PROHIBITION. "We do not share X with advertisers \
   without consent" is not a blanket prohibition on all sharing. A scoped prohibition with \
   an explicit escape clause ("except as outlined in this policy", "without consent") paired \
   with a third-party practice that fits within the exception is non_conflict; if the \
   third-party practice goes beyond the stated exception, the verdict remains inconsistent.

4. SAME-TEXT OR REINFORCING CLAUSES. The two clauses come from the same or near-identical \
   text, or both use exclusivity language ("only X", "solely X") that agrees rather than \
   contradicts.

5. SECTION HEADERS OR SENTENCE FRAGMENTS. One or both clauses are headings, CCPA category \
   listings, introductory questions ("How do we use personal information?"), or fragments \
   extracted as substantive commitments.

6. NON-PII vs PII MISMATCH. One side explicitly describes "non-personally identifiable", \
   "aggregated", "anonymized", or "de-identified" data; the other is about personal data.

7. DIFFERENT DATA TYPES. The pattern fired on ONTOLOGICALLY UNRELATED data types that \
   genuinely describe different practices (e.g. "image" vs "contact information", "browser \
   type" vs "health data"). Do NOT apply this rule to parent/child pairs ("personal data" \
   subsumes "email address"); those are covered by the DATA TYPE HIERARCHY guidance above \
   and are NOT non_conflict.

8. RIGHTS CLAUSE vs PRACTICE CLAUSE. A data-subject-rights statement ("you can request \
   deletion") is not contradicted by a processing clause ("we process for analytics").

9. BORDERLINE DRAFTING VARIATION. Both clauses are standard privacy-policy boilerplate, \
   phrased differently but saying compatible things.

=== Pattern-specific notes ===

- **Π₁ Modality** — requires a real first-party prohibition or hard limit and a real \
  third-party practice that violates it on the same (or ontology-subsumed) data. Key \
  genuine patterns: (a) first-party blanket negation + third-party openly performs the \
  action; (b) first-party purpose-scoped limit + third-party uses the data for a broader \
  purpose; (c) first-party consent requirement + third-party processes by default. \
  Non-conflict when the first-party negation has an explicit carve-out that matches the \
  third-party behaviour, or when the third-party policy covers services unrelated to the \
  first-party site.

- **Π₂ Exclusivity** — requires a first-party "only / solely / exclusively" clause that \
  genuinely limits the purpose, and a third-party using the same data for a different \
  purpose family. Reinforcing "only for X" + "only for X" is non_conflict. Meta-scoped \
  "only as described in this policy" is non_conflict.

- **Π₃ Condition** — requires the same action on the same (or subsumed) data type at \
  genuinely different consent strictness between the two sides. Two clauses from different \
  products with coincidentally different conditions are non_conflict.

- **Π₄ Temporal** — requires conflicting retention windows on the SAME data type. \
  Compatible per-category retentions ("contact data kept 30 days" + "log data kept 1 year") \
  are non_conflict.

=== Cluster context ===

When a Neighborhood Context section is present, use it. Multiple first-party clauses saying \
the same thing (e.g. 4 separate prohibitions on sharing) strengthens the case for a genuine \
finding; a single isolated clause is weaker evidence. Bridge-confirmed findings have a \
proven data flow from the first party to this third party — treat these as stronger \
evidence of a real cross-policy conflict. The surrounding cluster lines tell you whether \
the finding represents a consistent policy stance or an isolated drafting oddity.

=== Examples ===

Example A — inconsistent (direct contradiction):
  First-party clause: "We never sell your personal data to any third party, full stop."
  Third-party clause: "We sell advertising identifiers to our advertising partners by default."
  → {{"verdict": "inconsistent", "explanation": "First party makes a blanket 'never sell' commitment on personal data with no carve-out; third party openly sells advertising identifiers (a subtype of personal data) to advertising partners by default. The two are logically incompatible about the same data practice.", "confidence": "high", "false_alarm_category": "none"}}

Example B — non_conflict (modality_misextraction):
  Clause 1 extracted as PROHIBITION: "You have the right to delete your account data at any time."
  Clause 2 extracted as COMMITMENT:  "We retain your account data for the duration of your account."
  → {{"verdict": "non_conflict", "explanation": "Clause 1 is a user-rights statement ('you have the right to delete'), not a company prohibition — the extracted PROHIBITION modality is an extraction error. A user right does not contradict a retention practice; the user may exercise the right to terminate retention.", "confidence": "high", "false_alarm_category": "modality_misextraction"}}

Example C — inconsistent (purpose drift):
  First-party clause: "We share browsing history with analytics providers only to improve our services."
  Third-party clause: "We use browsing history for cross-site behavioural advertising and audience profiling."
  → {{"verdict": "inconsistent", "explanation": "First party explicitly limits shared browsing history to service improvement; third party uses the same data for behavioural advertising and profiling, which goes beyond what a user reading the first-party policy would reasonably expect. Real assurance gap on the same data type.", "confidence": "medium", "false_alarm_category": "none"}}

Example D — unspecified (ambiguous scope):
  First-party clause: "We limit collection of device identifiers to what is strictly necessary for our security monitoring product."
  Third-party clause: "We collect device identifiers across our SDK deployments."
  → {{"verdict": "unspecified", "explanation": "First party scopes the restriction to its security monitoring product; third party's clause does not specify which of its SDK deployments serve the first party or whether device-identifier collection falls inside the first party's scoped limit. The evidence is not decisive either way.", "confidence": "medium", "false_alarm_category": "none"}}

=== Output format ===

Return ONLY valid JSON (no markdown, no prose, no code fences). All fields are required:

{{"verdict": "inconsistent|unspecified|non_conflict", \
"explanation": "≥20 chars, 1-3 sentences quoting or paraphrasing specific language from the surrounding context", \
"confidence": "high|medium|low", \
"false_alarm_category": "none|modality_misextraction|scope_mismatch|scoped_negation|same_text|section_header|non_pii_mismatch|different_data_types|rights_vs_practice|borderline_drafting"}}

Strict rules (your response is rejected and re-prompted if violated):
- All fields required; none may be empty.
- `explanation` must be ≥20 characters and must quote or paraphrase specific language from \
  the surrounding context, not just from the extracted clause.
- `false_alarm_category` must be "none" when verdict is inconsistent or unspecified; must \
  be a specific category (not "none") when verdict is non_conflict.
- When in doubt between inconsistent and unspecified, choose unspecified.
- When in doubt between unspecified and non_conflict, prefer non_conflict only if one of \
  the decision rules clearly applies.
"""


VERIFIER_PROMPT_LEGACY = """You are a strict privacy-policy auditor. A pattern-matching system has flagged two clauses as a potential inconsistency. Your job is to decide whether a reasonable privacy-policy reviewer, reading both clauses in full context, would conclude the policies are making incompatible promises about the same data type. For intra-policy patterns (Π₁/Π₂/Π₄/Π₅/Π₇), both clauses come from the same policy and should describe the same actor and context. For cross-policy patterns (Π₈), the clauses come from a WEBSITE policy and its VENDOR's policy.

Default to **non_conflict** when in doubt. Only label genuine when the contradiction is clear from the actual source text.

Pattern: {pattern_id} — {pattern_name}

=== Clause 1 (first-party) ===
{source_text_1}
Fields: action={action_1} | modality={modality_1} | data={data_1}

=== Clause 2 (third-party) ===
{source_text_2}
Fields: action={action_2} | modality={modality_2} | data={data_2}

Pattern finding: {pattern_explanation}

Verdicts:
- hard_contradiction: logically incompatible.
- soft_tension: individually valid but together leave users with weaker protection.
- non_conflict: false alarm.

Return ONLY JSON: {{"verdict": "hard_contradiction|soft_tension|non_conflict", "explanation": "≥20 chars", "confidence": "high|medium|low"}}

When in doubt → non_conflict. Prefer soft_tension over hard_contradiction unless unambiguous."""


__all__ = [
    "EXTRACTION_PROMPT",
    "EXTRACTION_PROMPT_FEWSHOT",
    "VERIFIER_PROMPT",
    "VERIFIER_PROMPT_LEGACY",
]
