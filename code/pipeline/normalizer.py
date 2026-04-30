"""Data type and purpose normalization with lightweight ontology helpers."""

from __future__ import annotations

import functools
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from threading import Lock


# Tracked-passthrough logging: when normalize_data_type / normalize_purpose
# hit the final `return normalized` path (term not in dictionary, not a
# canonical, no fuzzy match), we either drop them silently (status quo, which
# breaks cross-policy comparability because "student records" and
# "learner data" become distinct keys) or log them so a reviewer can audit
# which terms the ontology is missing. Default ON; disable with
# NORMALIZER_LOG_MISSES=0 env var.
_LOG_MISSES = os.environ.get("NORMALIZER_LOG_MISSES", "1") not in ("0", "false", "")
_MISS_LOG_PATH = Path(__file__).resolve().parent / "data" / "output" / "normalization_misses.log"
_MISS_SEEN: set[tuple[str, str]] = set()
_MISS_LOCK = Lock()


def _record_passthrough(kind: str, value: str) -> None:
    """One-line-per-miss log (kind, value). Dedupes within a process; each
    run appends to the shared file. Safe to call from multi-threaded code."""
    if not _LOG_MISSES or not value:
        return
    key = (kind, value)
    with _MISS_LOCK:
        if key in _MISS_SEEN:
            return
        _MISS_SEEN.add(key)
        try:
            _MISS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _MISS_LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(f"{kind}\t{value}\n")
        except OSError:
            # Log directory may be read-only in some test harnesses; fail open
            # rather than crash the pipeline.
            pass


# Ontology aligned with PoliGraph (Cui et al., USENIX Security 2023) hierarchy
# and extended with W3C DPV (Data Privacy Vocabulary) categories.
# ~65 canonical types across 5 levels. Covers >95% of data types found in real policies.
DATA_ONTOLOGY = {
    # Level 0 → Level 1
    "personal data": [
        "contact information",
        "device information",
        "behavioral data",
        "financial data",
        "health data",
        "biometric data",
        "geolocation",
        "account data",
        "demographic data",
        "user-generated content",
        "tracking technology",
        "government id",
        "professional data",
        "communication data",
    ],
    # Level 1 → Level 2: Contact
    "contact information": ["email address", "phone number", "name", "postal address"],
    # Level 1 → Level 2: Device (aligned with PoliGraph "device information")
    "device information": [
        "ip address",
        "device id",
        "advertising id",
        "browser type",
        "operating system",
        "device type",
        "screen resolution",
        "language settings",
        "timezone",
    ],
    # Level 1 → Level 2: Behavioral (aligned with PoliGraph "internet activity")
    "behavioral data": [
        "browsing history",
        "search queries",
        "purchase history",
        "clickstream data",
        "app usage",
        "interaction data",
    ],
    # Level 1 → Level 2: Financial
    "financial data": ["payment information", "transaction history", "bank account"],
    # Level 1 → Level 2: Geolocation (aligned with PoliGraph)
    "geolocation": ["precise geolocation", "coarse geolocation", "country"],
    # Level 1 → Level 2: Account (from DPV "Authenticating")
    "account data": ["username", "password", "user preferences", "account settings"],
    # Level 1 → Level 2: Demographic (aligned with PoliGraph "protected classification")
    "demographic data": ["age", "date of birth", "gender", "race or ethnicity"],
    # Level 1 → Level 2: User-generated content (from DPV "Content")
    "user-generated content": ["review", "photo", "video", "message", "feedback"],
    # Level 1 → Level 2: Tracking (aligned with PoliGraph "device identifier" / DPV "Tracking")
    "tracking technology": ["cookie id", "pixel", "session id", "browser fingerprint"],
    # Level 1 → Level 2: Government ID (from PoliGraph "government identifier")
    "government id": ["ssn", "driver license", "passport number", "tax id"],
    # Level 1 → Level 2: Professional (from DPV "Professional")
    "professional data": ["employment information", "education information"],
    # Level 1 → Level 2: Health (extended with DPV subtypes)
    "health data": ["medical records", "medical conditions"],
    # Level 1 → Level 2: Communication (from DPV "Communication")
    "communication data": ["call records", "sms data", "email content"],
}


# Sensitivity levels aligned with GDPR special categories (Art. 9) and DPV risk levels.
DATA_SENSITIVITY = {
    # Abstract parent categories — inherit highest child sensitivity
    "personal data": "HIGH",
    "contact information": "HIGH",
    "device information": "MEDIUM",
    "behavioral data": "MEDIUM",
    "geolocation": "HIGH",
    # CRITICAL — GDPR Art. 9 special categories
    "health data": "CRITICAL",
    "biometric data": "CRITICAL",
    "race or ethnicity": "CRITICAL",
    "medical records": "CRITICAL",
    "medical conditions": "CRITICAL",
    # HIGH — directly identifying or financially sensitive
    "precise geolocation": "HIGH",
    "financial data": "HIGH",
    "email address": "HIGH",
    "phone number": "HIGH",
    "postal address": "HIGH",
    "payment information": "HIGH",
    "bank account": "HIGH",
    "password": "HIGH",
    "government id": "HIGH",
    "ssn": "HIGH",
    "driver license": "HIGH",
    "passport number": "HIGH",
    "tax id": "HIGH",
    "call records": "HIGH",
    "email content": "HIGH",
    # MEDIUM — pseudonymous or contextually identifying
    "name": "MEDIUM",
    "browsing history": "MEDIUM",
    "search queries": "MEDIUM",
    "ip address": "MEDIUM",
    "device id": "MEDIUM",
    "cookie id": "MEDIUM",
    "advertising id": "MEDIUM",
    "purchase history": "MEDIUM",
    "transaction history": "MEDIUM",
    "account data": "MEDIUM",
    "username": "MEDIUM",
    "demographic data": "MEDIUM",
    "age": "MEDIUM",
    "date of birth": "MEDIUM",
    "photo": "MEDIUM",
    "video": "MEDIUM",
    "message": "MEDIUM",
    "tracking technology": "MEDIUM",
    "pixel": "MEDIUM",
    "session id": "MEDIUM",
    "browser fingerprint": "MEDIUM",
    "clickstream data": "MEDIUM",
    "interaction data": "MEDIUM",
    "sms data": "MEDIUM",
    "country": "MEDIUM",
    "communication data": "MEDIUM",
    "professional data": "MEDIUM",
    "employment information": "MEDIUM",
    "education information": "MEDIUM",
    # LOW — non-identifying or aggregatable
    "browser type": "LOW",
    "operating system": "LOW",
    "coarse geolocation": "LOW",
    "device type": "LOW",
    "user preferences": "LOW",
    "account settings": "LOW",
    "gender": "LOW",
    "user-generated content": "LOW",
    "review": "LOW",
    "feedback": "LOW",
    "app usage": "LOW",
    "screen resolution": "LOW",
    "language settings": "LOW",
    "timezone": "LOW",
}


PURPOSE_NECESSITY = {
    "service delivery": 5,
    "authentication": 5,
    "security": 4,
    "fraud prevention": 4,
    "legal compliance": 4,
    "product improvement": 3,
    "analytics": 3,
    "research": 3,
    "personalization": 2,
    "recommendations": 2,
    "advertising": 1,
    "marketing": 1,
    "targeted advertising": 1,
}


DATA_SYNONYMS = {
    # --- Email ---
    "your email": "email address",
    "email": "email address",
    "e-mail": "email address",
    "e-mail address": "email address",
    "email account": "email address",
    "emails": "email address",
    # --- Name ---
    "your name": "name",
    "full name": "name",
    "first name": "name",
    "last name": "name",
    "company name": "name",
    # --- Username ---
    "username": "username",
    "usernames": "username",
    "your username": "username",
    "code repository username": "username",
    "login name": "username",
    "login name history": "username",
    # --- Postal address ---
    "your postal address": "postal address",
    "billing address": "postal address",
    "shipping address": "postal address",
    "street address": "postal address",
    "home address": "postal address",
    "physical address": "postal address",
    "mailing address": "postal address",
    "company address": "postal address",
    # --- Phone ---
    "phone": "phone number",
    "telephone number": "phone number",
    "mobile number": "phone number",
    # --- Contact information ---
    "contact details": "contact information",
    "contact info": "contact information",
    "other contact details": "contact information",
    "contact and identity information": "contact information",
    "contact information for follow-up": "contact information",
    # --- Payment / financial ---
    "payment details": "payment information",
    "billing information": "payment information",
    "credit card": "payment information",
    "credit card information": "payment information",
    "other payment method": "payment information",
    "financial information": "financial data",
    "bank details": "bank account",
    "bank account information": "bank account",
    "bank account details": "bank account",
    "bank account number": "bank account",
    "transaction data": "transaction history",
    # --- Personal data ---
    "personal information": "personal data",
    "personal info": "personal data",
    "personally identifiable information": "personal data",
    "pii": "personal data",
    "personal data about you": "personal data",
    "personal data you provide": "personal data",
    "personal data we hold about you": "personal data",
    "personal data held by us": "personal data",
    "personal data directly provided by users": "personal data",
    "private personal data": "personal data",
    "your information": "personal data",
    "your data": "personal data",
    "information about you": "personal data",
    # NOTE (2026-04-19): aggregated/anonymized/anonymous/de-identified data are
    # intentionally NOT mapped to "personal data". Under GDPR Art. 4(1) + Recital
    # 26 they are explicitly NOT personal data, so the prior mapping inverted
    # their semantics and produced Π₁ false positives (e.g., "we only process
    # anonymized data" vs "we collect personal data" flagged as a contradiction).
    # They now fall through to passthrough and enter the graph as non-canonical
    # nodes (is_canonical=False → excluded from ALIGNED_TO per graph.py G1).
    # --- Device information ---
    "device information": "device information",
    "device identifiers": "device id",
    "device identifier": "device id",
    "unique device identifier": "device id",
    "device unique identifier": "device id",
    "udid": "device id",
    "device characteristics": "device information",
    # --- Cookie / tracking technology ---
    "cookie ids": "cookie id",
    "cookies": "cookie id",
    "tracking cookies": "cookie id",
    "cookie data": "cookie id",
    "cookies and similar technologies": "tracking technology",
    "similar technologies": "tracking technology",
    "pixels": "pixel",
    "beacons": "pixel",
    "tags": "pixel",
    "scripts": "tracking technology",
    "session cookies": "cookie id",
    "authentication cookies": "cookie id",
    "analytics tracking": "browsing history",
    "google analytics tracking": "browsing history",
    # --- Advertising ID ---
    "advertising ids": "advertising id",
    "ad id": "advertising id",
    "mobile advertising id": "advertising id",
    "idfa": "advertising id",
    "gaid": "advertising id",
    # --- IP address ---
    "ip": "ip address",
    "ip addresses": "ip address",
    "internet protocol address": "ip address",
    # --- Browser ---
    "browser information": "browser type",
    "browser details": "browser type",
    "user agent": "browser type",
    "browser": "browser type",
    "browser version": "browser type",
    "browser language": "browser type",
    "browser data": "browser type",
    "browser settings type of browser": "browser type",
    "browser type and settings": "browser type",
    "browser characteristics": "browser type",
    # --- OS ---
    "os": "operating system",
    "operating system version": "operating system",
    # --- Geolocation ---
    "location data": "geolocation",
    "location information": "geolocation",
    "geo-location": "geolocation",
    "country location": "geolocation",
    "specific location": "precise geolocation",
    "precise location": "precise geolocation",
    "exact location": "precise geolocation",
    "approximate location": "coarse geolocation",
    "general location": "coarse geolocation",
    "geolocation information": "geolocation",
    # --- Browsing / behavioral ---
    "browsing data": "browsing history",
    "web browsing history": "browsing history",
    "browsing activity": "browsing history",
    "online activity": "browsing history",
    "page views": "browsing history",
    # --- App / usage ---
    "usage data": "app usage",
    "usage information": "app usage",
    # --- Search ---
    "search history": "search queries",
    "search data": "search queries",
    # --- Clickstream ---
    "click data": "clickstream data",
    "click history": "clickstream data",
    "user interaction data": "clickstream data",
    # --- Purchase ---
    "purchase data": "purchase history",
    "buying history": "purchase history",
    # --- Health ---
    "health information": "health data",
    "medical data": "health data",
    "medical information": "health data",
    "health": "health data",
    "medical records": "health data",
    "medical conditions": "health data",
    # --- Biometric ---
    "biometric information": "biometric data",
    "fingerprint data": "biometric data",
    "facial recognition data": "biometric data",
    # --- Account data ---
    "account data": "account data",
    "your account data": "account data",
    "user account data": "account data",
    "your user account information": "account data",
    "account information": "account data",
    "account": "account data",
    "accounts": "account data",
    # --- User preferences ---
    "preferences": "user preferences",
    "settings": "user preferences",
    "user settings": "user preferences",
    # --- Demographic ---
    "date of birth": "date of birth",
    "age": "age",
    "gender": "gender",
    # --- User-generated content ---
    "review": "review",
    "reviews": "review",
    "your review": "review",
    "your reviews": "review",
    "the reviews you write": "review",
    "photo": "photo",
    "photos": "photo",
    "message": "message",
    "messages": "message",
    "feedback": "feedback",
    # --- Session / pixel ---
    "session id": "session id",
    "session id chrome extension": "session id",
    # --- Screen / display ---
    "screen resolution": "screen resolution",
    "display resolution": "screen resolution",
    # --- Language / timezone ---
    "language settings": "language settings",
    "language": "language settings",
    "language preference": "language settings",
    "browser settings browser language": "language settings",
    "browser settings time zone": "timezone",
    "timezone": "timezone",
    "time zone": "timezone",
    # --- Country (coarse geo) ---
    "country": "country",
    "country location": "country",
    "country of residence": "country",
    # --- Interaction data ---
    "interaction data": "interaction data",
    "interactions": "interaction data",
    "interactions with our services": "interaction data",
    "interaction with our platform": "interaction data",
    "interaction with our marketing emails": "interaction data",
    # --- Government ID (PoliGraph: "government identifier") ---
    "social security number": "ssn",
    "social security": "ssn",
    "ssn": "ssn",
    "driver s license": "driver license",
    "driver s license number": "driver license",
    "drivers license": "driver license",
    "driving license": "driver license",
    "passport": "passport number",
    "passport number": "passport number",
    "tax identification number": "tax id",
    "tax id": "tax id",
    "national id": "government id",
    "government issued id": "government id",
    "government-issued identification": "government id",
    # --- Professional / Employment (DPV: "Professional") ---
    "employment information": "employment information",
    "employment history": "employment information",
    "employment data": "employment information",
    "professional or employment-related information": "employment information",
    "job title": "employment information",
    "employer": "employment information",
    "education information": "education information",
    "education history": "education information",
    "education data": "education information",
    # --- Communication (DPV: "Communication") ---
    "call records": "call records",
    "call recordings": "call records",
    "call recordings inbound and outbound telephone or video calls": "call records",
    "sms data": "sms data",
    "sms routing information": "sms data",
    "text messages": "sms data",
    "email content": "email content",
    "email content you receive": "email content",
    "email content you write": "email content",
    "email communications": "email content",
    # --- Race / Ethnicity (PoliGraph: "protected classification") ---
    "race or ethnicity": "race or ethnicity",
    "race": "race or ethnicity",
    "ethnicity": "race or ethnicity",
    "ethnic origin": "race or ethnicity",
    "information revealing ethnic origin": "race or ethnicity",
    "racial origin": "race or ethnicity",
    # --- Video ---
    "video": "video",
    "videos": "video",
    "videos submitted with your reviews": "video",
    # --- Browser fingerprint ---
    "browser fingerprint": "browser fingerprint",
    "digital fingerprint": "browser fingerprint",
    "device fingerprint": "browser fingerprint",
    # --- Medical subtypes ---
    "medical records": "medical records",
    "medical conditions": "medical conditions",
    "health conditions": "medical conditions",
    "patient-related information": "health data",
    # --- Additional high-frequency passthrough catches ---
    # Browsing / visit data
    "page visits": "browsing history",
    "site visits": "browsing history",
    "visit information": "browsing history",
    "visitor data": "browsing history",
    "visitor data and information": "browsing history",
    "aggregate statistics about our visitors": "browsing history",
    "number of visits": "browsing history",
    "duration of page and site visits": "browsing history",
    "navigation paths": "browsing history",
    "referring/exit pages": "browsing history",
    "date/time stamp": "browsing history",
    "time and date of your visit": "browsing history",
    "log file data": "browsing history",
    "server logs": "browsing history",
    "aggregate information from server logs": "browsing history",
    "temporary server logs": "browsing history",
    "information server logs": "browsing history",
    # Cookies (additional variants)
    "cookies on your device": "cookie id",
    "stored cookies": "cookie id",
    "functional cookies": "cookie id",
    "cookies browser cookies": "cookie id",
    "advertising cookies": "cookie id",
    "analytics cookies": "cookie id",
    "aggregate site analytics cookies": "cookie id",
    # Device / browser (additional)
    "internet service provider isp": "device information",
    "isp": "device information",
    "unique identifiers": "device id",
    "device identification number udid": "device id",
    # Account / user data
    "user data": "personal data",
    "contact data": "contact information",
    "your content": "user-generated content",
    "user contributions": "user-generated content",
    "review content": "review",
    "replies to reviews by businesses": "review",
    "replies to reviews": "review",
    "survey information": "feedback",
    "responses to surveys": "feedback",
    "all data from surveys": "feedback",
    # Content / communication
    "your communication": "communication data",
    "sms messages": "sms data",
    "text message": "sms data",
    # Consent (not really a data type, but appears frequently)
    "consent": "user preferences",
    # Engagement / interaction
    "engagement data": "interaction data",
    "profiling information": "behavioral data",
    "user behavior": "behavioral data",
    # Donations / payment variants
    "donations": "payment information",
    "online transactions": "transaction history",
    "details of transactions you carry out through our sites": "transaction history",
    "information about your transactions": "transaction history",
}


PURPOSE_SYNONYMS = {
    "to provide services": "service delivery",
    "provide our services": "service delivery",
    "deliver our services": "service delivery",
    "to provide the service": "service delivery",
    "functionality": "service delivery",
    "core functionality": "service delivery",
    "to process transactions": "service delivery",
    "to process your orders": "service delivery",
    "order confirmations": "service delivery",
    "to send order confirmations": "service delivery",
    "create an account": "service delivery",
    "account creation": "service delivery",
    "account management": "service delivery",
    "add to your account": "service delivery",
    "save to your account": "service delivery",
    "provide care and treatment": "service delivery",
    "care and treatment": "service delivery",
    "to improve your shopping experience": "personalization",
    "ads": "advertising",
    "advertisement": "advertising",
    "ad serving": "advertising",
    "to advertise": "advertising",
    "advertising purposes": "advertising",
    "ad personalization": "targeted advertising",
    "personalized ads": "targeted advertising",
    "interest-based advertising": "targeted advertising",
    "behavioral advertising": "targeted advertising",
    "remarketing": "targeted advertising",
    "retargeting": "targeted advertising",
    "to market": "marketing",
    "promotional purposes": "marketing",
    "email marketing": "marketing",
    "data analytics": "analytics",
    "statistical analysis": "analytics",
    "analysis": "analytics",
    "to analyze": "analytics",
    "usage analytics": "analytics",
    "web analytics": "analytics",
    "analytics purposes": "analytics",
    "reporting": "analytics",
    "reporting for the website operator": "analytics",
    "to understand website traffic patterns": "analytics",
    "website traffic patterns": "analytics",
    "trend analysis": "research",
    "research purposes": "research",
    "trend analysis and research purposes": "research",
    "to improve our services": "product improvement",
    "improve our products": "product improvement",
    "service improvement": "product improvement",
    "product development": "product improvement",
    "enhancing our services": "product improvement",
    "to personalize": "personalization",
    "tailored experience": "personalization",
    "customization": "personalization",
    "content personalization": "personalization",
    "to secure": "security",
    "to protect": "security",
    "security purposes": "security",
    "prevent fraud": "fraud prevention",
    "fraud detection": "fraud prevention",
    "anti-fraud": "fraud prevention",
    "to comply with law": "legal compliance",
    "legal obligation": "legal compliance",
    "regulatory compliance": "legal compliance",
    "comply with legal requirements": "legal compliance",
    "scientific research": "research",
    "internal research": "research",
    "research and development": "research",
    "to authenticate": "authentication",
    "identity verification": "authentication",
    "user verification": "authentication",
    "account recovery": "authentication",
    "password recovery": "authentication",
    "recover your account": "authentication",
    "access your account": "authentication",
    "help you access your account": "authentication",
    "payment processing": "service delivery",
    "process payment": "service delivery",
    "process payments": "service delivery",
    "make a reservation": "service delivery",
    "booking service": "service delivery",
    "complete the transaction": "service delivery",
    "transaction fulfillment": "service delivery",
    "notify you of changes": "legal compliance",
    "notify users of changes": "legal compliance",
    "materially alter your privacy rights": "legal compliance",
    "material changes": "legal compliance",
    "changes to this policy": "legal compliance",
    "to recommend": "recommendations",
    "content recommendations": "recommendations",
    "suggested content": "recommendations",
}


def _normalize_text(value: str) -> str:
    value = value.strip().lower().replace("_", " ")
    value = re.sub(r"[^\w\s/-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _all_canonical_data_types() -> set[str]:
    values: set[str] = set()
    for parent, children in DATA_ONTOLOGY.items():
        values.add(parent)
        values.update(children)
    return values


CANONICAL_DATA_TYPES = _all_canonical_data_types()
CANONICAL_PURPOSES = set(PURPOSE_NECESSITY)


def _fuzzy_best_match(raw: str, candidates: set[str], threshold: float = 0.78) -> str | None:
    best_match = None
    best_score = 0.0
    for candidate in candidates:
        score = SequenceMatcher(None, raw, candidate).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate
    if best_score >= threshold:
        return best_match
    return None


def _normalize_via_dictionary(raw: str, synonyms: dict[str, str], canonicals: set[str]) -> str:
    normalized = _normalize_text(raw)
    if not normalized:
        return ""

    if normalized in synonyms:
        return synonyms[normalized]

    if normalized in canonicals:
        return normalized

    for phrase, canonical in synonyms.items():
        if normalized == phrase or normalized in phrase or phrase in normalized:
            return canonical

    fuzzy = _fuzzy_best_match(normalized, canonicals)
    if fuzzy:
        return fuzzy

    _record_passthrough("purpose", normalized)
    return normalized


@functools.lru_cache(maxsize=16384)
def normalize_data_type(raw: str) -> str:
    """Normalize a data type string to a canonical form.

    Memoised: pattern inner loops call this O(N²) times per pair on a
    small set of distinct raw strings. The cache drops per-pair time
    on large pairs by ~3-5× without changing output.
    """
    normalized = _normalize_text(raw)
    if not normalized:
        return ""

    if normalized in DATA_SYNONYMS:
        return DATA_SYNONYMS[normalized]

    if normalized in CANONICAL_DATA_TYPES:
        return normalized

    for phrase, canonical in sorted(DATA_SYNONYMS.items(), key=lambda item: len(item[0]), reverse=True):
        if len(phrase.split()) < 2:
            continue
        if re.search(rf"\b{re.escape(phrase)}\b", normalized):
            return canonical

    fuzzy = _fuzzy_best_match(normalized, CANONICAL_DATA_TYPES)
    if fuzzy:
        return fuzzy

    _record_passthrough("data_type", normalized)
    return normalized


def normalize_data_type_with_flag(raw: str) -> tuple[str, bool]:
    """Like normalize_data_type, but returns (value, is_canonical).

    is_canonical=False when the string was passthrough (not a known synonym,
    not a canonical type, no fuzzy match above threshold). Callers that need
    to skip uncanonical nodes in structural reasoning — most importantly
    ALIGNED_TO edge generation in graph.merge_graphs — should use this
    variant so that extractor hallucinations don't contaminate cross-policy
    alignment.
    """
    result = normalize_data_type(raw)
    return result, result in CANONICAL_DATA_TYPES


@functools.lru_cache(maxsize=16384)
def normalize_purpose(raw: str) -> str:
    """Normalize a purpose string to a canonical form. Memoised for
    the same reason as normalize_data_type."""

    return _normalize_via_dictionary(raw, PURPOSE_SYNONYMS, CANONICAL_PURPOSES)


def get_data_sensitivity(data_type: str) -> str:
    """Get sensitivity level for a data type."""

    normalized = normalize_data_type(data_type)
    return DATA_SENSITIVITY.get(normalized, "MEDIUM")


def get_purpose_necessity(purpose: str) -> int:
    """Get necessity score for a purpose."""

    normalized = normalize_purpose(purpose)
    # Default to 3 (medium necessity) for unrecognized purpose strings.
    # Unrecognized purposes are almost always verbatim policy fragments, not canonical low-necessity
    # uses. A default of 2 would fire Π₃ on any unrecognized purpose string paired with HIGH
    # sensitivity data, producing false positives. Legitimate low-necessity purposes (advertising,
    # marketing, personalization) are all in the vocabulary and still resolve to 1-2.
    return PURPOSE_NECESSITY.get(normalized, 3)


@functools.lru_cache(maxsize=16384)
def _parent_data_types_tuple(data_type: str) -> tuple[str, ...]:
    """Memoised ontology-walk helper. Returns a tuple so the cache
    guarantees immutability; public get_parent_data_types wraps into a
    fresh list per call so callers can safely mutate."""
    normalized = normalize_data_type(data_type)
    parents: list[str] = []
    for parent, children in DATA_ONTOLOGY.items():
        if normalized in children:
            parents.append(parent)
            parents.extend(_parent_data_types_tuple(parent))
    return tuple(parents)


def get_parent_data_types(data_type: str) -> list[str]:
    """Return all parent types in the ontology hierarchy."""
    return list(_parent_data_types_tuple(data_type))


def get_child_data_types(data_type: str) -> list[str]:
    """Return all child types in the ontology hierarchy."""

    normalized = normalize_data_type(data_type)
    children: list[str] = []
    if normalized in DATA_ONTOLOGY:
        for child in DATA_ONTOLOGY[normalized]:
            children.append(child)
            children.extend(get_child_data_types(child))
    return children


def data_types_related(dt1: str, dt2: str) -> bool:
    """Check if two data types are identical, subsuming, or siblings."""

    first = normalize_data_type(dt1)
    second = normalize_data_type(dt2)

    if not first or not second:
        return False
    if first == second:
        return True
    if first in get_parent_data_types(second):
        return True
    if second in get_parent_data_types(first):
        return True
    for _, children in DATA_ONTOLOGY.items():
        if first in children and second in children:
            return True
    return False


@functools.lru_cache(maxsize=65536)
def data_subsumes(general: str, specific: str) -> bool:
    """Check if the general data type subsumes the specific one.
    Memoised — ontology parent walks are expensive and results are
    deterministic within a run."""

    general_norm = normalize_data_type(general)
    specific_norm = normalize_data_type(specific)
    if not general_norm or not specific_norm:
        return False
    if general_norm == specific_norm:
        return True
    return general_norm in get_parent_data_types(specific_norm)


# ---------------------------------------------------------------------------
# Actor normalization
# ---------------------------------------------------------------------------

# Exact matches (after lowercasing) → canonical actor
ACTOR_SYNONYMS: dict[str, str] = {
    "you": "DataSubject",
    "user": "DataSubject",
    "data subject": "DataSubject",
    "the user": "DataSubject",
    "users": "DataSubject",
    "customer": "DataSubject",
    "patient": "DataSubject",
    "visitor": "DataSubject",
    "the visitor": "DataSubject",
}

# Substrings that, if present in the actor string, map to DataSubject
_DATA_SUBJECT_MARKERS = ("datasubject", "data subject")

# Substrings that indicate the actor is the first-party company
_FIRST_PARTY_MARKERS = (
    "firstparty",
    "first party",
    "first-party",
    "the website",
    "the platform",
    "the site",
    "our site",
    "our servers",
    "our system",
    "this website",
    "this service",
    "this site",
)

# Substrings that indicate a generic third party (not a named entity)
_THIRD_PARTY_MARKERS = (
    "third party",
    "third-party",
    "third parties",
    "thirdparty",
)


def normalize_actor(
    raw: str,
    first_party_names: list[str] | None = None,
) -> str:
    """Normalize an actor string to a canonical form.

    Parameters
    ----------
    raw : str
        The actor label extracted by the LLM.
    first_party_names : list[str] | None
        Company names that should be treated as first party (e.g., ["Sedo", "Sedo.com LLC"]).
        These are matched case-insensitively as substrings.
    """
    if not raw or not raw.strip():
        return "FirstParty"

    lowered = raw.strip().lower()

    # Exact synonym match
    if lowered in ACTOR_SYNONYMS:
        return ACTOR_SYNONYMS[lowered]

    # DataSubject markers
    if any(marker in lowered for marker in _DATA_SUBJECT_MARKERS):
        return "DataSubject"

    # First-party markers (generic)
    if any(marker in lowered for marker in _FIRST_PARTY_MARKERS):
        return "FirstParty"

    # First-party by company name
    if first_party_names:
        for fp_name in first_party_names:
            if fp_name.lower() in lowered or lowered in fp_name.lower():
                return "FirstParty"

    # Named third-party with prefix like "ThirdParty:Google"
    if lowered.startswith("thirdparty:"):
        name = raw.split(":", 1)[1].strip()
        return name if name else "ThirdParty"

    # Generic third-party descriptions → "ThirdParty"
    if any(marker in lowered for marker in _THIRD_PARTY_MARKERS):
        return "ThirdParty"

    # Cookies / tracking as actors → "ThirdParty"
    if lowered in ("cookies", "these cookies", "tracking cookies"):
        return "ThirdParty"

    # Preserve named entities as-is (e.g., "Google", "Trustpilot")
    return raw.strip()
