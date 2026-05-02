from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from .utils.etld import etld1
from .utils.logging import warn

# High-recall multilingual keyword list (EU-heavy).
PRIVACY_KEYWORDS = [
    # English
    "privacy", "privacy policy", "privacy notice", "data protection", "personal data",
    # French
    "confidentialité", "politique de confidentialité", "protection des données",
    # German
    "datenschutz", "datenschutzerklärung", "datenschutzrichtlinie",
    # Spanish
    "privacidad", "política de privacidad", "protección de datos",
    # Italian
    "informativa sulla privacy", "protezione dei dati",
    # Portuguese
    "privacidade", "política de privacidade", "proteção de dados",
    # Dutch
    "privacybeleid", "privacyverklaring",
    # Swedish / Danish / Norwegian
    "integritet", "sekretess", "personuppgifter", "personvern", "personvernpolicy",
    "persondatapolitik",
    # Polish
    "polityka prywatności", "ochrona danych",
    # Czech / Slovak
    "ochrana osobních údajů", "zásady ochrany osobných údajov",
    # Romanian
    "politica de confidențialitate", "protecția datelor",
]

COOKIE_KEYWORDS = [
    "cookie", "cookies", "cookie policy", "cookie notice",
    "gestion des cookies", "politique de cookies",
    "cookies-richtlinie", "política de cookies", "politica sui cookie",
]

LEGAL_HUB_KEYWORDS = [
    "legal", "imprint", "impressum", "terms", "conditions", "about", "company",
    "mentions légales", "aviso legal", "note legali", "rechtliches"
]

# Common policy hosters where the "first-party" policy might be hosted on a different eTLD+1.
POLICY_HOSTERS_ETLD1 = {
    "iubenda.com", "termly.io", "termsfeed.com", "privacypolicies.com",
    "cookiebot.com", "onetrust.com", "trustarc.com",
}

COMMON_PRIVACY_PATHS = [
    "/privacy", "/privacy-policy", "/privacy_policy", "/privacy-notice", "/privacy_notice",
    "/legal/privacy", "/legal/privacy-policy", "/policies/privacy",
    "/datenschutz", "/datenschutzerklaerung", "/datenschutzerklärung",
    "/politique-de-confidentialite", "/politique-de-confidentialité",
    "/politica-privacidad", "/politica-de-privacidad", "/politica-privacidade",
    "/polityka-prywatnosci", "/polityka-prywatności",
    "/privacy-statement", "/gdpr", "/rgpd",
]

def _norm_space(s: str) -> str:
    return " ".join((s or "").split()).strip()

def _is_http_url(u: str) -> bool:
    try:
        p = urlparse(u)
        return p.scheme in ("http", "https")
    except Exception:
        return False

def _clean_href(href: str) -> str | None:
    if not href:
        return None
    href = href.strip()
    if href.startswith(("mailto:", "tel:", "javascript:", "#")):
        return None
    return href

@dataclass
class LinkCandidate:
    url: str
    anchor_text: str
    score: float
    source: str  # footer/body/hub/fallback
    candidate_etld1: str | None
    is_same_site: bool

def score_link(anchor_text: str, url: str, *, is_same_site: bool) -> float:
    t = _norm_space(anchor_text).lower()
    u = url.lower()
    score = 0.0

    # strong URL signals
    if any(k in u for k in ("privacy", "datenschutz", "confidential", "privacidad", "privacidade")):
        score += 5.0
    if "cookie" in u or "cookies" in u:
        score -= 1.0  # prefer privacy over cookie when ambiguous

    # anchor text signals
    for kw in PRIVACY_KEYWORDS:
        if kw in t:
            score += 4.0
            break
    for kw in COOKIE_KEYWORDS:
        if kw in t:
            score -= 0.5
            break

    # penalize obvious non-policies
    if any(bad in u for bad in ("login", "signin", "account", "cart", "checkout")):
        score -= 2.0

    # Slight preference for same-site hosting
    score += 0.8 if is_same_site else -0.8
    return score

def _has_privacy_keyword(text: str) -> bool:
    t = _norm_space(text).lower()
    return any(kw in t for kw in PRIVACY_KEYWORDS)

def _url_privacy_signal(url: str) -> bool:
    u = (url or "").lower()
    return any(k in u for k in (
        "privacy", "privacy-policy", "privacy_policy", "privacy-notice",
        "data-protection", "data_protection", "gdpr", "rgpd",
        "datenschutz", "confidential", "privacidad", "privacidade",
    ))

def _allow_external_candidate(site_etld1: str, cand_url: str, anchor_text: str, source: str) -> bool:
    cand_et = etld1(cand_url)
    if not cand_et:
        return False
    if cand_et == site_etld1:
        return True
    if cand_et in POLICY_HOSTERS_ETLD1:
        return True
    # Heuristic: if the site domain label appears in the URL (common with hosted policy pages)
    site_label = site_etld1.split(".")[0].lower() if site_etld1 else ""
    if site_label and site_label in cand_url.lower():
        return True
    # Allow external links when there's a strong privacy signal (usually a site-owned policy).
    anchor_has_privacy = _has_privacy_keyword(anchor_text)
    url_has_privacy = _url_privacy_signal(cand_url)
    if source in ("footer", "hub", "fallback"):
        return anchor_has_privacy or url_has_privacy
    # For body links, require stronger evidence.
    return anchor_has_privacy and url_has_privacy

def extract_link_candidates(html: str, base_url: str, site_etld1: str) -> list[LinkCandidate]:
    soup = BeautifulSoup(html, "lxml")

    # Collect footer links first (higher precision)
    footer_links: list[tuple[str, str]] = []
    for footer in soup.find_all("footer"):
        for a in footer.find_all("a", href=True):
            footer_links.append((_norm_space(a.get_text(" ")), a["href"]))

    body_links: list[tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        if a.find_parent("footer") is not None:
            continue
        body_links.append((_norm_space(a.get_text(" ")), a["href"]))

    def build(links: Iterable[tuple[str, str]], source: str) -> list[LinkCandidate]:
        out: list[LinkCandidate] = []
        for text, href in links:
            href = _clean_href(href)
            if not href:
                continue
            abs_url = urljoin(base_url, href)
            if not _is_http_url(abs_url):
                continue
            if not _allow_external_candidate(site_etld1, abs_url, text, source):
                continue

            cand_et = etld1(abs_url)
            same = (cand_et == site_etld1)
            out.append(LinkCandidate(
                url=abs_url,
                anchor_text=text,
                score=score_link(text, abs_url, is_same_site=same),
                source=source,
                candidate_etld1=cand_et,
                is_same_site=same,
            ))
        return out

    cands = build(footer_links, "footer") + build(body_links, "body")

    # De-duplicate by URL (keep max score)
    best: dict[str, LinkCandidate] = {}
    for c in cands:
        cur = best.get(c.url)
        if cur is None or c.score > cur.score:
            best[c.url] = c

    return sorted(best.values(), key=lambda x: x.score, reverse=True)

def extract_legal_hub_urls(candidates: list[LinkCandidate], limit: int = 3) -> list[str]:
    hubs: list[str] = []
    for c in candidates:
        t = c.anchor_text.lower()
        u = c.url.lower()
        if any(k in t for k in LEGAL_HUB_KEYWORDS) or any(k in u for k in ("legal", "imprint", "impressum", "terms", "about")):
            hubs.append(c.url)
        if len(hubs) >= limit:
            break
    return hubs

def fallback_privacy_urls(base_url: str, site_etld1: str) -> list[LinkCandidate]:
    out: list[LinkCandidate] = []
    for path in COMMON_PRIVACY_PATHS:
        url = urljoin(base_url, path)
        cand_et = etld1(url)
        same = (cand_et == site_etld1)
        out.append(LinkCandidate(url=url, anchor_text=path, score=2.0 + (0.8 if same else -0.8), source="fallback",
                                 candidate_etld1=cand_et, is_same_site=same))
    return out

def policy_likeliness_score(text: str) -> float:
    # Lightweight heuristic: longer + contains policy-ish terms across languages.
    if not text:
        return -10.0
    t = text.lower()
    words = t.split()
    score = 0.0
    score += min(len(words) / 400.0, 6.0)  # cap length benefit

    # Terms suggesting privacy policy
    terms = [
        "privacy", "personal data", "data protection", "gdpr",
        "datenschutz", "personenbez", "confidentialit", "données",
        "privacidad", "datos personales", "protezione dei dati",
        "polityka prywat", "ochrona danych", "rgpd",
    ]
    hit = 0
    for term in terms:
        if term in t:
            hit += 1
    score += min(hit, 6) * 1.2

    # Penalty if it looks like cookie-only
    if "cookie" in t and "privacy" not in t and "datenschutz" not in t and "confidential" not in t:
        score -= 1.5
    return score
