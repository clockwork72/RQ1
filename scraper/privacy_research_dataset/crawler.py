from __future__ import annotations

import asyncio
import json
import re
import shutil
from datetime import datetime, timezone
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable
from urllib.parse import urlparse, urlunparse

if TYPE_CHECKING:
    import openai as _openai_t

import aiohttp
from bs4 import BeautifulSoup

from .crawl4ai_client import Crawl4AIClient, Crawl4AIResult
from .policy_finder import (
    extract_link_candidates,
    extract_legal_hub_urls,
    fallback_privacy_urls,
    policy_likeliness_score,
    LinkCandidate,
)
from .robust_scraping import site_override_urls, wayback_candidates


async def _wayback_cdx_discover(domain: str, timeout_s: float = 8.0) -> list[str]:
    """Query Wayback Machine's CDX API for archived privacy-related URLs on this domain.

    Returns deduplicated candidate URLs sorted by relevance (privacy > cookie > legal).
    Free, no auth, rate-limited at ~15 req/s by archive.org.
    """
    if not domain:
        return []
    queries = [
        f"url={domain}/*privacy*&output=json&limit=10&fl=original&filter=statuscode:200",
        f"url={domain}/*policy*&output=json&limit=5&fl=original&filter=statuscode:200",
        f"url={domain}/*datenschutz*&output=json&limit=3&fl=original&filter=statuscode:200",
    ]
    seen: set[str] = set()
    results: list[str] = []
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for q in queries:
                try:
                    url = f"https://web.archive.org/cdx/search/cdx?{q}"
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            continue
                        import json as _json
                        data = _json.loads(await resp.text())
                        for row in data:
                            if not isinstance(row, list) or len(row) < 1:
                                continue
                            candidate = str(row[0]).strip()
                            if candidate in ("original", "") or candidate in seen:
                                continue
                            # Basic relevance: must look like a privacy/policy/legal page
                            low = candidate.lower()
                            if any(k in low for k in ("privacy", "datenschutz", "confidential",
                                                       "policy", "legal", "rgpd", "gdpr")):
                                seen.add(candidate)
                                results.append(candidate)
                except Exception:
                    continue
    except Exception:
        pass
    # Sort: exact "privacy" in path first, then "policy", then others
    def rank(u: str) -> int:
        l = u.lower()
        if "privacy" in l: return 0
        if "policy" in l: return 1
        return 2
    results.sort(key=rank)
    return results[:8]
from .third_party import third_parties_from_network_logs
from .tracker_radar import TrackerRadarIndex, TrackerRadarEntry
from .trackerdb import TrackerDbIndex, TrackerDbEntry
from .openwpm_engine import run_openwpm_for_third_parties
from .utils.asyncio import wait_for_with_cancel_grace
from .utils.etld import etld1
from .utils.logging import log, warn

_HTML_MARKER = re.compile(r"(?is)<\s*!doctype\s+html|<\s*html\b|<\s*head\b|<\s*body\b")

# ---------------- Cookie helpers (robust-run additions) ----------------

def _parse_set_cookie_header(val: str) -> dict[str, Any] | None:
    # Very lightweight parser; returns {name,value,domain,path,secure,httponly,samesite,expires}.
    if not isinstance(val, str) or "=" not in val:
        return None
    segs = [s.strip() for s in val.split(";") if s.strip()]
    if not segs:
        return None
    name, _, value = segs[0].partition("=")
    attrs: dict[str, Any] = {
        "name": name.strip(),
        "value": value.strip(),
        "domain": None,
        "path": None,
        "expires": None,
        "max_age": None,
        "secure": False,
        "http_only": False,
        "same_site": None,
    }
    for seg in segs[1:]:
        if "=" in seg:
            k, _, v = seg.partition("=")
            k = k.strip().lower(); v = v.strip()
            if k == "domain": attrs["domain"] = v.lstrip(".")
            elif k == "path": attrs["path"] = v
            elif k == "expires": attrs["expires"] = v
            elif k == "max-age": attrs["max_age"] = v
            elif k == "samesite": attrs["same_site"] = v
        else:
            k = seg.strip().lower()
            if k == "secure": attrs["secure"] = True
            elif k == "httponly": attrs["http_only"] = True
    return attrs if attrs["name"] else None


def _extract_third_party_urls_from_html(html: str | None, base_url: str) -> list[dict[str, Any]]:
    """Fallback TP extraction when no live network capture is available.

    Parses static HTML for resource references (scripts, iframes, images, links,
    media) and synthesizes pseudo network_request entries. Won't catch JS-injected
    XHRs but covers static analytics/ad tags, tracking pixels, social embeds, etc.
    """
    if not html:
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return []
    # Tag/attribute pairs to scan. Multi-attr tags handled by checking each.
    pairs = [
        ("script", "src"),
        ("iframe", "src"),
        ("img", "src"),
        ("img", "data-src"),
        ("source", "src"),
        ("source", "srcset"),
        ("video", "src"),
        ("video", "poster"),
        ("audio", "src"),
        ("link", "href"),
        ("a", "href"),  # broad; mostly same-site so dedup will trim
        ("embed", "src"),
        ("object", "data"),
        ("track", "src"),
    ]
    for tag, attr in pairs:
        for el in soup.find_all(tag):
            val = el.get(attr)
            if not val:
                continue
            # srcset has comma-separated URLs with descriptors; take each first token
            if attr == "srcset":
                candidates = [piece.strip().split(" ", 1)[0] for piece in str(val).split(",")]
            else:
                candidates = [str(val).strip()]
            for raw in candidates:
                if not raw or raw.startswith(("javascript:", "mailto:", "tel:", "#", "data:", "blob:")):
                    continue
                try:
                    abs_url = urljoin(base_url, raw)
                except Exception:
                    continue
                p = urlparse(abs_url)
                if p.scheme not in ("http", "https"):
                    continue
                if abs_url in seen:
                    continue
                seen.add(abs_url)
                out.append({"event_type": "request", "url": abs_url})
    return out


def _normalize_pw_cookie(c: dict[str, Any]) -> dict[str, Any]:
    """Playwright cookies -> our schema (same field names where possible)."""
    return {
        "name": c.get("name"),
        "value": c.get("value"),
        "domain": (c.get("domain") or "").lstrip(".") or None,
        "path": c.get("path"),
        "expires": c.get("expires"),
        "secure": bool(c.get("secure")),
        "http_only": bool(c.get("httpOnly")),
        "same_site": c.get("sameSite"),
    }


def _extract_cookies_from_network(home: "Crawl4AIResult", site_etld1: str) -> dict[str, Any]:
    """Primary: Playwright context.cookies() bucket by cookie domain's eTLD+1.
    Fallback: parse Set-Cookie from main response_headers."""
    first_party: list[dict[str, Any]] = []
    third_party: dict[str, list[dict[str, Any]]] = {}
    unique_tp_hosts: set[str] = set()

    # PRIMARY path: Playwright cookies (covers HttpOnly + third-party jar entries).
    pw = getattr(home, "playwright_cookies", None) or []
    for raw in pw:
        c = _normalize_pw_cookie(raw) if isinstance(raw, dict) else None
        if not c or not c.get("name"):
            continue
        dom = c.get("domain") or ""
        cand_et = etld1("https://" + dom) if dom else ""
        if not cand_et or cand_et == site_etld1:
            first_party.append(c)
        else:
            third_party.setdefault(cand_et, []).append(c)
            unique_tp_hosts.add(cand_et)
    if first_party or third_party:
        return {
            "first_party": first_party,
            "third_party": third_party,
            "cookie_counts": {
                "first_party": len(first_party),
                "third_party_total": sum(len(v) for v in third_party.values()),
                "third_party_unique_hosts": len(unique_tp_hosts),
            },
        }

    # FALLBACK: parse Set-Cookie from response_headers
    first_party, third_party, unique_tp_hosts = [], {}, set()

    # 1) Main-page response headers (most reliable first-party bucket)
    main_hdrs = (home.response_headers or {}) if home else {}
    # headers dict may have lowercase or capitalized keys, and Set-Cookie can appear once (comma-joined) or as list
    for k, v in (main_hdrs.items() if isinstance(main_hdrs, dict) else []):
        if str(k).lower() != "set-cookie":
            continue
        vals = v if isinstance(v, list) else [v]
        for vv in vals:
            for part in str(vv).split(", "):
                c = _parse_set_cookie_header(part)
                if c:
                    first_party.append(c)

    # 2) Per-response events if present (bucket by eTLD+1 of the URL)
    for ev in (home.network_responses or []) if home else []:
        url = ev.get("url") or ""
        headers = ev.get("headers") or ev.get("response_headers") or {}
        if not isinstance(headers, dict):
            continue
        set_cookies = []
        for k, v in headers.items():
            if str(k).lower() == "set-cookie":
                vals = v if isinstance(v, list) else [v]
                for vv in vals:
                    for part in str(vv).split(", "):
                        c = _parse_set_cookie_header(part)
                        if c:
                            set_cookies.append(c)
        if not set_cookies:
            continue
        cand_et = etld1(url) or ""
        if not cand_et or cand_et == site_etld1:
            first_party.extend(set_cookies)
        else:
            third_party.setdefault(cand_et, []).extend(set_cookies)
            unique_tp_hosts.add(cand_et)

    return {
        "first_party": first_party,
        "third_party": third_party,
        "cookie_counts": {
            "first_party": len(first_party),
            "third_party_total": sum(len(v) for v in third_party.values()),
            "third_party_unique_hosts": len(unique_tp_hosts),
        },
    }


def _count_words(text: str | None) -> int:
    if not text:
        return 0
    return len((text or "").split())


# Lightweight English detector reused from catalog_ingest.
_EN_WORDS = set("the and of to a in that is for on with as by be this it from or at an are we you your our their".split())
def _is_english_text(text: str, min_ratio: float = 0.07) -> bool:
    if not text:
        return False
    words = [w.lower().strip(".,;:!?()\"'") for w in text.split()[:300] if w.strip()]
    if len(words) < 20:
        return False
    hits = sum(1 for w in words if w in _EN_WORDS)
    return hits / max(1, len(words)) >= min_ratio

_NON_BROWSABLE_PATTERNS = [
    re.compile(pat, re.I)
    for pat in (
        r"access denied",
        r"forbidden",
        r"request blocked",
        r"service unavailable",
        r"temporarily unavailable",
        r"bad gateway",
        r"error\s*404",
        r"404\s*not\s*found",
        r"\bnot found\b",
        r"no such bucket",
        r"nosuchbucket",
        r"nosuchkey",
        r"invalid url",
        r"permission denied",
        r"not authorized",
        r"domain.*for sale",
        r"under construction",
        r"default web site page",
        r"iis windows server",
    )
]
_POLICY_SCAN_FULL_PAGE_DOMAINS = ("onetrust.com", "cookielaw.org", "cookiepro.com")
_POLICY_DISCOVERY_BATCH_SIZE = 2
_POLICY_DISCOVERY_MAX_CANDIDATES = 6
_POLICY_DISCOVERY_MAX_HUB_PAGES = 1
_POLICY_DISCOVERY_MAX_CONSECUTIVE_TIMEOUTS = 2
_TP_POLICY_FETCH_BATCH_SIZE = 4
_LOW_VALUE_POLICY_PATH_TOKENS = ("sitemap", "robots.txt", "/feed", "/rss")

_WALL_VENDOR_DOMAINS = frozenset({
    "cloudflare.com", "datadome.com", "akamai.com", "imperva.com",
    "incapsula.com", "perimeterx.com", "humansecurity.com", "kasada.com",
    "sucuri.net", "f5.com", "distilnetworks.com",
})
_WIDGET_VENDOR_DOMAINS = frozenset({
    "google.com", "facebook.com", "meta.com", "apple.com", "microsoft.com",
    "twitter.com", "x.com", "amazon.com", "linkedin.com",
})
_CRAWL_CANCEL_GRACE_SEC = 1.0

# ---------------------------------------------------------------------------
# LLM-based semantic policy cleaner
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """\
You are a privacy policy text extractor for academic research.

You will receive raw text extracted from a privacy policy webpage. This text \
may contain navigation menus, cookie consent panels, site headers/footers, \
feedback widgets, and other non-policy content mixed with the actual privacy policy.

Your task: return ONLY the full privacy policy content, preserving it verbatim.

KEEP:
- All privacy policy sections and their full text
- Section headings and document structure
- Effective / last-updated dates
- Contact details referenced within the policy
- Legal definitions and terms that are part of the policy

REMOVE:
- Site navigation menus and header/footer links
- Cookie consent or preference-center panels (OneTrust, Cookiebot, TrustArc, etc.)
- Feedback widgets ("Was this helpful?", "Rate this page", star ratings)
- Page breadcrumbs, language/region selectors, search bars
- Copyright notices and unrelated boilerplate at the page footer
- Any content that is clearly NOT part of the privacy policy text

RULES:
- Do NOT paraphrase, summarize, or alter the policy wording in any way
- Preserve markdown headings (##, ###) and list structure (-, *, numbered lists)
- Return ONLY the cleaned policy text — no preamble, explanation, or commentary
- If the entire input is already clean policy text, return it unchanged
"""


def _chunk_policy_text(text: str, max_chars: int = 60_000) -> list[str]:
    """Split Markdown policy text at heading boundaries with breadcrumb context.

    Each chunk starts with the heading hierarchy of its position so the LLM
    has section context even when processing a slice of the full document.
    Prevents silent output truncation when text exceeds gpt-4o-mini's 16K
    output-token window.
    """
    if len(text) <= max_chars:
        return [text]

    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    breadcrumb: list[tuple[int, str]] = []  # (heading_level, heading_line)

    for line in lines:
        m = re.match(r"^(#{1,3}) ", line)
        if m:
            level = len(m.group(1))
            breadcrumb = [(lvl, h) for lvl, h in breadcrumb if lvl < level]
            breadcrumb.append((level, line.rstrip()))
        if current_len + len(line) > max_chars and current:
            chunks.append("".join(current))
            ctx = [h + "\n" for _, h in breadcrumb]
            current = ctx
            current_len = sum(len(h) for h in ctx)
        current.append(line)
        current_len += len(line)

    if current:
        chunks.append("".join(current))
    return chunks


async def _llm_clean_policy_text(
    text: str,
    *,
    client: "_openai_t.AsyncOpenAI",
    model: str = "gpt-4o-mini",
    max_input_chars: int = 60_000,
) -> str | None:
    """Semantically clean raw extracted policy text using an LLM.

    Long texts are split into heading-bounded chunks (each ≤ max_input_chars)
    so the LLM output never silently truncates. Uses the provided AsyncOpenAI
    client. Returns None on failure so callers can fall back to raw text.
    """
    if not text or not text.strip():
        return None

    chunks = _chunk_policy_text(text, max_chars=max_input_chars)
    results: list[str] = []
    for chunk in chunks:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": chunk},
                ],
                temperature=0,
                max_tokens=16384,
            )
            result = response.choices[0].message.content
            if result and result.strip():
                results.append(result.strip())
        except Exception as e:
            warn(f"LLM policy cleaning failed ({model}): {e}")

    if not results:
        return None
    return "\n\n".join(results)


def _url_host(url: str | None) -> str:
    if not url:
        return ""
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _should_scan_full_page_policy(url: str | None) -> bool:
    host = _url_host(url)
    if not host:
        return False
    return any(host == d or host.endswith(f".{d}") for d in _POLICY_SCAN_FULL_PAGE_DOMAINS)


def _is_low_value_policy_candidate(candidate_url: str, site_url: str) -> bool:
    try:
        candidate = urlparse(candidate_url)
        site = urlparse(site_url)
    except Exception:
        return False
    path = (candidate.path or "/").lower()
    if candidate.hostname and site.hostname and candidate.hostname.lower() == site.hostname.lower() and path in {"", "/"}:
        return True
    if path.endswith(".xml"):
        return True
    if any(token in path for token in _LOW_VALUE_POLICY_PATH_TOKENS):
        return True
    cand_etld = etld1(candidate_url) or ""
    site_etld = etld1(site_url) or ""
    if cand_etld and site_etld and cand_etld != site_etld:
        if cand_etld in _WALL_VENDOR_DOMAINS:
            return True
        if cand_etld in _WIDGET_VENDOR_DOMAINS:
            return True
    return False


def _homepage_looks_like_policy(url: str, text: str, *, score: float) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    path = (urlparse(url).path or "/").lower()
    has_policy_path = any(token in path for token in ("privacy", "policy", "notice", "datenschutz"))
    has_policy_phrase = "privacy policy" in text_lower or "privacy notice" in text_lower
    return len(text) >= 1000 and score >= 8.0 and (has_policy_path or has_policy_phrase)

def _safe_dirname(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)[:200]

def _write_text(p: Path, text: str | None) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text or "", encoding="utf-8")

def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _html_to_text(html: str | None) -> str | None:
    if not html:
        return None
    try:
        soup = BeautifulSoup(html, "lxml")
        return "\n".join([ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()])
    except Exception:
        return None


def _combine_errors(*msgs: str | None) -> str | None:
    parts = [m for m in msgs if m and str(m).strip()]
    if not parts:
        return None
    return " | ".join(parts)


def _timeout_result(url: str, *, timeout_s: float, phase: str) -> Crawl4AIResult:
    return Crawl4AIResult(
        url=url,
        success=False,
        status_code=None,
        raw_html=None,
        cleaned_html=None,
        text=None,
        network_requests=None,
        error_message=f"{phase}_timed_out_after_{timeout_s:.1f}s",
        text_extraction_method=None,
    )


async def _await_crawl_result(
    awaitable: Awaitable[Crawl4AIResult],
    *,
    url: str,
    timeout_s: float,
    phase: str,
) -> Crawl4AIResult:
    try:
        return await wait_for_with_cancel_grace(
            awaitable,
            timeout_s=timeout_s,
            cancel_grace_s=_CRAWL_CANCEL_GRACE_SEC,
        )
    except asyncio.TimeoutError:
        warn(f"{phase} timed out after {timeout_s:.1f}s for {url}")
        return _timeout_result(url, timeout_s=timeout_s, phase=phase)


async def _fetch_home_with_retry(
    client: Crawl4AIClient,
    site_url: str,
    *,
    capture_network: bool,
    fetch_timeout_sec: float,
    max_attempts: int = 2,
    retry_delay_s: float = 0.8,
) -> tuple[Crawl4AIResult | None, str, int, list[str]]:
    errors: list[str] = []
    total_ms = 0
    home_fetch_mode = "crawl4ai"
    attempt_timeout_sec = min(
        fetch_timeout_sec,
        max(8.0, float(client.page_timeout_ms) / 1000.0 * 1.25),
    )
    fallback_timeout_ms = max(5_000, min(int(attempt_timeout_sec * 1000), int(client.page_timeout_ms)))
    for attempt in range(1, max_attempts + 1):
        # Only in aggressive mode: enable enhanced rendering (stealth+magic+scan) on last attempt.
        use_enhanced = bool(getattr(client, "aggressive", False)) and attempt == max_attempts
        enhanced_timeout_sec = max(attempt_timeout_sec, min(fetch_timeout_sec, 90.0)) if use_enhanced else attempt_timeout_sec
        t_home = time.perf_counter()
        home = await _await_crawl_result(
            client.fetch(
                site_url,
                capture_network=capture_network,
                remove_overlays=True,
                magic=False,
                scan_full_page=False,
                enhanced=use_enhanced,
            ),
            url=site_url,
            timeout_s=enhanced_timeout_sec,
            phase="home_fetch",
        )
        total_ms += int((time.perf_counter() - t_home) * 1000)

        if home.success and not home.cleaned_html and home.raw_html:
            home.cleaned_html = home.raw_html
        if home.success and not home.text and home.cleaned_html:
            home.text = _html_to_text(home.cleaned_html)

        if home.success and home.cleaned_html:
            return home, home_fetch_mode, total_ms, errors

        t_home_fb = time.perf_counter()
        fallback = await _simple_http_fetch(
            site_url,
            user_agent=client.user_agent,
            timeout_ms=fallback_timeout_ms,
            allow_http_fallback=True,
        )
        total_ms += int((time.perf_counter() - t_home_fb) * 1000)
        if fallback.success and fallback.cleaned_html:
            return fallback, "simple_http", total_ms, errors

        errors.append(_combine_errors(home.error_message, fallback.error_message) or "home_fetch_failed")

        if attempt < max_attempts:
            await asyncio.sleep(retry_delay_s * attempt)

    return None, home_fetch_mode, total_ms, errors

def _detect_wall_vendor(home: Crawl4AIResult) -> str | None:
    """Identify anti-bot walls / challenge pages by signature (vendor-specific markers)."""
    html = (home.raw_html or home.cleaned_html or "") if home else ""
    text = (home.text or "") if home else ""
    headers = (home.response_headers or {}) if home else {}
    body = (html + "\n" + text).lower()
    hdr_keys = {str(k).lower(): str(v or "").lower() for k, v in (headers.items() if isinstance(headers, dict) else [])}

    # Cloudflare
    if ("cf-ray" in hdr_keys or "cf-chl-" in body or "__cf_chl_" in body
        or "challenges.cloudflare.com" in body or "cf_clearance" in body
        or "just a moment..." in body or "checking your browser before accessing" in body
        or "cloudflare ray id" in body):
        return "cloudflare_challenge"
    # DataDome
    if ("datadome" in body or "captcha-delivery.com" in body or "geo.captcha-delivery.com" in body
        or "datadome-" in hdr_keys.get("server", "")):
        return "datadome_challenge"
    # Akamai Bot Manager
    if ("akamaighost" in hdr_keys.get("server", "") or "akamai reference" in body
        or "_abck" in body or "ak_bmsc" in body or "bm_sv" in body
        or ("access denied" in body and "reference #" in body)):
        return "akamai_challenge"
    # Imperva / Incapsula
    if ("incapsula" in body or "incap_ses" in body or "x-iinfo" in hdr_keys
        or "_incap_" in body or "visid_incap" in body):
        return "imperva_challenge"
    # PerimeterX / HUMAN
    if ("px-captcha" in body or "px-captcha.com" in body or "perimeterx" in body
        or "_px3" in body or "human-security" in body):
        return "perimeterx_challenge"
    # Sucuri
    if "sucuri website firewall" in body or "sucuri/cloudproxy" in hdr_keys.get("server", ""):
        return "sucuri_challenge"
    # Kasada
    if "kasada" in body or "ips.js" in body and "kpsdk" in body:
        return "kasada_challenge"
    # Shape Security / F5
    if "jstracker" in body or "_shape_" in body or "shape-security" in body:
        return "shape_challenge"
    # Distil
    if "distil_r" in body or "distil-" in body:
        return "distil_challenge"
    # Generic captcha
    if ("recaptcha" in body and ("verify you are human" in body or "i'm not a robot" in body)) \
       or ("hcaptcha" in body and "human" in body):
        return "captcha_wall"
    # Generic consent/GDPR wall that hides all content
    if ("accept all cookies" in body or "manage cookies" in body) and len(text) < 500:
        return "consent_wall"
    # Generic "forbidden" / "access denied" without vendor markers
    if "access denied" in body or "403 forbidden" in body or "you don't have permission" in body:
        return "access_denied"
    return None


def _classify_non_browsable(home: Crawl4AIResult) -> tuple[bool, str | None]:
    # Vendor-specific wall / challenge detection first (most informative).
    vendor = _detect_wall_vendor(home)
    if vendor:
        return True, vendor
    # Treat explicit HTTP errors as non-browsable when we did get a page.
    if home.status_code and home.status_code >= 400:
        return True, f"http_status_{home.status_code}"

    text = (home.text or _html_to_text(home.cleaned_html) or "").strip()
    text_len = len(text)

    # Error page markers.
    low_text = text.lower()
    for pat in _NON_BROWSABLE_PATTERNS:
        if pat.search(low_text):
            return True, "error_page_text"

    # Link-sparse + short text: often infra/service or placeholder.
    if home.cleaned_html:
        try:
            soup = BeautifulSoup(home.cleaned_html, "lxml")
            anchor_count = len(soup.find_all("a", href=True))
        except Exception:
            anchor_count = 0
    else:
        anchor_count = 0

    if text_len < 200 and anchor_count == 0:
        return True, "no_links_short_text"
    if text_len < 80 and anchor_count <= 1:
        return True, "very_sparse_page"

    return False, None

async def _simple_http_fetch(
    url: str,
    *,
    user_agent: str | None,
    timeout_ms: int,
    max_bytes: int = 2_000_000,
    allow_http_fallback: bool = True,
) -> Crawl4AIResult:
    headers = {"Accept-Language": "en-US,en;q=0.9"}
    if user_agent:
        headers["User-Agent"] = user_agent
    parsed = urlparse(url)
    urls_to_try = [url]
    if allow_http_fallback and parsed.scheme == "https":
        urls_to_try.append(urlunparse(parsed._replace(scheme="http")))

    timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)
    # Raise header limits — some big sites (streaming, ad-heavy) send
    # Set-Cookie / CSP / feature-policy values well over aiohttp's default 8 KB.
    async with aiohttp.ClientSession(
        headers=headers,
        max_line_size=32768,
        max_field_size=65536,
    ) as session:
        last_error: str | None = None
        for u in urls_to_try:
            try:
                async with session.get(u, timeout=timeout, allow_redirects=True) as resp:
                    if resp.status >= 400:
                        last_error = f"http_status_{resp.status}"
                        continue
                    ctype = (resp.headers.get("content-type") or "").lower()
                    raw = await resp.content.read(max_bytes)
                    if not raw:
                        last_error = "empty_body"
                        continue
                    text = raw.decode("utf-8", errors="ignore")
                    if ("text/html" not in ctype) and ("application/xhtml" not in ctype):
                        if not _HTML_MARKER.search(text):
                            last_error = f"non_html_content_type:{ctype}"
                            continue
                    if not _HTML_MARKER.search(text):
                        last_error = "html_marker_missing"
                        continue

                    cleaned = text
                    extracted_text = _html_to_text(cleaned)
                    return Crawl4AIResult(
                        url=str(resp.url),
                        success=True,
                        status_code=resp.status,
                        raw_html=text,
                        cleaned_html=cleaned,
                        text=extracted_text,
                        network_requests=[],
                        error_message=None,
                    )
            except Exception as e:
                last_error = str(e)
                continue

    return Crawl4AIResult(
        url=url,
        success=False,
        status_code=None,
        raw_html=None,
        cleaned_html=None,
        text=None,
        network_requests=None,
        error_message=last_error or "simple_http_fetch_failed",
    )

async def _fetch_best_policy(
    client: Crawl4AIClient,
    site_url: str,
    home_cleaned_html: str,
    *,
    max_candidates: int = _POLICY_DISCOVERY_MAX_CANDIDATES,
    max_hub_pages: int = _POLICY_DISCOVERY_MAX_HUB_PAGES,
    fetch_timeout_sec: float,
    policy_fetcher: Callable[[str], Awaitable[Crawl4AIResult]] | None = None,
) -> dict[str, Any]:
    site_et = etld1(site_url) or ""

    candidates = [
        candidate
        for candidate in extract_link_candidates(home_cleaned_html, site_url, site_et)
        if not _is_low_value_policy_candidate(candidate.url, site_url)
    ]
    tried: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None
    best_fallback: dict[str, Any] | None = None
    best_key: tuple[float, int] | None = None

    async def try_candidate(c: LinkCandidate) -> dict[str, Any]:
        fetch_awaitable = (
            policy_fetcher(c.url)
            if policy_fetcher is not None
            else client.fetch(
                c.url,
                capture_network=False,
                remove_overlays=True,
                magic=False,
                scan_full_page=_should_scan_full_page_policy(c.url),
            )
        )
        res = await _await_crawl_result(
            fetch_awaitable,
            url=c.url,
            timeout_s=fetch_timeout_sec,
            phase="policy_fetch",
        )
        # SIMPLE-HTTP FALLBACK (Fix 7): trigger whenever Playwright returned no
        # usable text — covers crashes AND silent SPA/headless-detection failures
        # where Playwright reports success=True with an empty or stripped DOM.
        # Privacy pages are overwhelmingly static HTML; aiohttp usually wins here.
        if not res.success or not (res.text or "").strip():
            fallback_timeout_ms = max(5000, int(fetch_timeout_sec * 1000))
            try:
                simple = await _simple_http_fetch(
                    c.url,
                    user_agent=getattr(client, "user_agent", None),
                    timeout_ms=fallback_timeout_ms,
                    allow_http_fallback=False,
                )
                if simple.success and (simple.text or "").strip():
                    res = simple
            except Exception:
                pass
        rec = dict(
            url=c.url,
            anchor_text=c.anchor_text,
            score=c.score,
            source=c.source,
            candidate_etld1=c.candidate_etld1,
            is_same_site=c.is_same_site,
            fetch_success=res.success,
            status_code=res.status_code,
            error_message=res.error_message,
            text_extraction_method=res.text_extraction_method,
        )
        text = (res.text or "").strip()
        rec["text_len"] = len(text)
        rec["likeliness_score"] = policy_likeliness_score(text)
        return rec | {"text": text, "cleaned_html": res.cleaned_html, "raw_html": res.raw_html}

    def is_policy_candidate(rec: dict[str, Any]) -> bool:
        if not rec.get("fetch_success"):
            return False
        score = float(rec.get("likeliness_score") or -10.0)
        text_len = int(rec.get("text_len") or 0)
        if score >= 5.0 and text_len >= 300:
            return True
        if score >= 4.0 and text_len >= 500:
            return True
        return score >= 3.0 and text_len >= 800

    def consider_best(rec: dict[str, Any]) -> None:
        nonlocal best_fallback, best_key
        if not rec.get("fetch_success"):
            return
        score = float(rec.get("likeliness_score") or -10.0)
        text_len = int(rec.get("text_len") or 0)
        if score < 3.0 or text_len < 200:
            return
        key = (score, text_len)
        if best_key is None or key > best_key:
            best_key = key
            best_fallback = rec

    consecutive_timeouts = 0
    max_consecutive_timeouts = _POLICY_DISCOVERY_MAX_CONSECUTIVE_TIMEOUTS

    def _check_timeout(rec: dict[str, Any]) -> bool:
        """Return True if we should bail out due to too many consecutive timeouts."""
        nonlocal consecutive_timeouts
        err = rec.get("error_message") or ""
        if "timed_out" in err:
            consecutive_timeouts += 1
            if consecutive_timeouts >= max_consecutive_timeouts:
                warn(f"[{site_et}] Bailing out of policy discovery after {consecutive_timeouts} consecutive timeouts")
                return True
        else:
            consecutive_timeouts = 0
        return False

    async def evaluate_candidates(batch_candidates: list[LinkCandidate]) -> bool:
        nonlocal chosen
        for idx in range(0, len(batch_candidates), _POLICY_DISCOVERY_BATCH_SIZE):
            batch = batch_candidates[idx: idx + _POLICY_DISCOVERY_BATCH_SIZE]
            recs = await asyncio.gather(*(try_candidate(candidate) for candidate in batch))
            for rec in recs:
                tried.append({k: rec[k] for k in rec.keys() if k not in ("text", "cleaned_html", "raw_html")})
                consider_best(rec)
                if is_policy_candidate(rec):
                    chosen = rec
                    return True
                if _check_timeout(rec):
                    return True
        return False

    # 1) Try top candidates directly
    await evaluate_candidates(candidates[:max_candidates])

    # 2) Fallback common paths
    if chosen is None and consecutive_timeouts < max_consecutive_timeouts:
        await evaluate_candidates(list(fallback_privacy_urls(site_url, site_et)))

    # 3) Legal hub expansion (depth 1): fetch 1-2 legal/terms pages and rescan for privacy links
    if chosen is None and candidates and consecutive_timeouts < max_consecutive_timeouts:
        hub_urls = extract_legal_hub_urls(candidates, limit=max_hub_pages)
        for hub in hub_urls:
            hub_res = await _await_crawl_result(
                client.fetch(
                    hub,
                    capture_network=False,
                    remove_overlays=True,
                    magic=False,
                    scan_full_page=_should_scan_full_page_policy(hub),
                ),
                url=hub,
                timeout_s=fetch_timeout_sec,
                phase="policy_hub_fetch",
            )
            if not hub_res.success or not hub_res.cleaned_html:
                if _check_timeout({"error_message": hub_res.error_message}):
                    break
                continue
            consecutive_timeouts = 0  # hub fetch succeeded, reset
            hub_cands = extract_link_candidates(hub_res.cleaned_html, hub_res.url, site_et)
            hub_batch = [
                LinkCandidate(
                    url=c.url,
                    anchor_text=c.anchor_text,
                    score=c.score + 0.2,
                    source="hub",
                    candidate_etld1=c.candidate_etld1,
                    is_same_site=c.is_same_site,
                )
                for c in hub_cands[:max_candidates]
            ]
            stop = await evaluate_candidates(hub_batch)
            if stop:
                break
            if chosen is not None:
                break

    # 4) Best-effort fallback: pick the strongest policy-like page even if shorter.
    if chosen is None and best_fallback is not None:
        chosen = best_fallback

    return {
        "site_etld1": site_et,
        "candidates_top": [
            {
                "url": c.url,
                "anchor_text": c.anchor_text,
                "score": c.score,
                "source": c.source,
                "candidate_etld1": c.candidate_etld1,
                "is_same_site": c.is_same_site,
            }
            for c in candidates[:25]
        ],
        "tried": tried,
        "chosen": (None if chosen is None else {k: chosen[k] for k in chosen.keys() if k in (
            "url","anchor_text","score","source","candidate_etld1","is_same_site","status_code","likeliness_score","text_len","text_extraction_method"
        )}) ,
        "_chosen_full": chosen,  # internal (includes text/html)
    }

def _normalize_url(url: str | None) -> str:
    """Normalize a URL for use as a registry key (lowercase host, strip default ports, drop fragment)."""
    u = (url or "").strip()
    if not u:
        return ""
    try:
        p = urlparse(u)
        scheme = (p.scheme or "https").lower()
        host = (p.hostname or "").lower()
        if not host:
            return u
        port = p.port
        default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
        netloc = host if (port is None or default_port) else f"{host}:{port}"
        path = p.path or "/"
        return urlunparse((scheme, netloc, path, "", p.query, ""))
    except Exception:
        return u


async def _copy_policy_artifact(
    norm_url: str,
    dst_dir: Path,
    registry: dict[str, Path],
    lock: asyncio.Lock,
) -> bool:
    """Copy policy.txt + policy.extraction.json from registry src to dst_dir.

    Returns True if a valid artifact was found and copied; False otherwise.
    """
    async with lock:
        src_dir = registry.get(norm_url)
    if src_dir is None or src_dir == dst_dir:
        return False
    src_policy = src_dir / "policy.txt"
    if not src_policy.exists() or src_policy.stat().st_size == 0:
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_policy, dst_dir / "policy.txt")
    src_extraction = src_dir / "policy.extraction.json"
    if src_extraction.exists():
        shutil.copy2(src_extraction, dst_dir / "policy.extraction.json")
    return True


async def _register_policy_artifact(
    norm_url: str,
    art_dir: Path,
    registry: dict[str, Path],
    lock: asyncio.Lock,
) -> None:
    """Register art_dir as the canonical artifact location for norm_url (first writer wins)."""
    if not norm_url:
        return
    async with lock:
        registry.setdefault(norm_url, art_dir)


async def process_site(
    client: Crawl4AIClient,
    domain_or_url: str,
    *,
    rank: int | None,
    artifacts_dir: str | Path,
    tracker_radar: TrackerRadarIndex | None = None,
    trackerdb: TrackerDbIndex | None = None,
    fetch_third_party_policies: bool = True,
    third_party_policy_max: int = 30,
    third_party_engine: str = "crawl4ai",  # crawl4ai|openwpm
    run_id: str | None = None,
    stage_callback: Callable[[str], None] | None = None,
    exclude_same_entity: bool = False,
    first_party_policy_url_override: str | None = None,
    first_party_policy_fetcher: Callable[[str], Awaitable[Crawl4AIResult]] | None = None,
    third_party_policy_fetcher: Callable[[str], Awaitable[Crawl4AIResult]] | None = None,
    openai_client: "_openai_t.AsyncOpenAI | None" = None,
    llm_model: str = "gpt-4o-mini",
    policy_artifact_registry: dict[str, Path] | None = None,
    policy_artifact_lock: asyncio.Lock | None = None,
    fetch_timeout_sec: float | None = None,
) -> dict[str, Any]:
    """
    Process a single website:
    - Fetch homepage
    - Find and fetch best privacy policy
    - Extract third-party domains from network logs (Crawl4AI) or OpenWPM (optional)
    - Map third parties via Tracker Radar / Ghostery TrackerDB (+ optionally fetch their policy texts)
    """
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    t_total = time.perf_counter()
    fetch_timeout_sec = max(0.001, float(fetch_timeout_sec or max(60.0, client.page_timeout_ms / 1000 * 4.0)))

    site_url = domain_or_url.strip()
    if not site_url:
        return {"input": domain_or_url, "error": "empty_input"}
    if "://" not in site_url:
        site_url = "https://" + site_url

    site_art_dir = Path(artifacts_dir) / _safe_dirname(etld1(site_url) or domain_or_url)
    site_art_dir.mkdir(parents=True, exist_ok=True)

    # 1) Homepage fetch
    if stage_callback:
        stage_callback("home_fetch")
    capture_net = (third_party_engine == "crawl4ai")
    home, home_fetch_mode, home_fetch_ms, home_errors = await _fetch_home_with_retry(
        client,
        site_url,
        capture_network=capture_net,
        fetch_timeout_sec=fetch_timeout_sec,
    )

    if not home:
        return {
            "rank": rank,
            "input": domain_or_url,
            "site_url": site_url,
            "final_url": site_url,
            "site_etld1": etld1(site_url),
            "status": "home_fetch_failed",
            "status_code": None,
            "error_message": _combine_errors(*home_errors),
            "home_fetch_mode": home_fetch_mode,
            "error_code": "home_fetch_failed",
            "home_fetch_ms": home_fetch_ms,
            "home_fetch_attempts": len(home_errors),
            "total_ms": int((time.perf_counter() - t_total) * 1000),
            "run_id": run_id,
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        }

    # Fix 5 — home-level simple_http fallback when Playwright returned 200 but the DOM
    # is too thin to find any links (SPA hydration timeout / headless detection serving
    # a stripped response). aiohttp often gets the real, fully-rendered HTML.
    if home and home.success and home.cleaned_html:
        is_nb_pre, reason_pre = _classify_non_browsable(home)
        if is_nb_pre and reason_pre in ("no_links_short_text", "very_sparse_page"):
            try:
                fallback_timeout_ms = max(5000, int(fetch_timeout_sec * 1000))
                simple = await _simple_http_fetch(
                    site_url,
                    user_agent=getattr(client, "user_agent", None),
                    timeout_ms=fallback_timeout_ms,
                    allow_http_fallback=False,
                )
                if simple.success and simple.cleaned_html:
                    is_nb_post, _ = _classify_non_browsable(simple)
                    if not is_nb_post:
                        log(f"[home_fallback] swapped Playwright→simple_http for {etld1(home.url) or etld1(site_url)} (reason={reason_pre})")
                        # Preserve Playwright's network capture so third-party extraction still works
                        simple.network_requests = home.network_requests
                        simple.playwright_cookies = home.playwright_cookies
                        home = simple
                        home_fetch_mode = "simple_http_fallback"
            except Exception:
                pass

    # Cookies (best-effort): parse Set-Cookie from main response + captured responses.
    site_et_for_cookies = etld1(home.url) or etld1(site_url) or ""
    cookies_bundle = _extract_cookies_from_network(home, site_et_for_cookies)
    try:
        _write_json(site_art_dir / "cookies.json", {
            "first_party": cookies_bundle["first_party"],
            "third_party": cookies_bundle["third_party"],
        })
    except Exception:
        pass

    # 2) Privacy policy discovery + fetch
    if stage_callback:
        stage_callback("policy_discovery")
    t_policy = time.perf_counter()
    chosen_full: dict[str, Any] | None = None
    manual_policy_url_override = (first_party_policy_url_override or "").strip() or None

    if manual_policy_url_override:
        if first_party_policy_fetcher is not None:
            override_res = await _await_crawl_result(
                first_party_policy_fetcher(manual_policy_url_override),
                url=manual_policy_url_override,
                timeout_s=fetch_timeout_sec,
                phase="manual_policy_fetch",
            )
        else:
            override_res = await _await_crawl_result(
                client.fetch(
                    manual_policy_url_override,
                    capture_network=False,
                    remove_overlays=True,
                    magic=False,
                    scan_full_page=_should_scan_full_page_policy(manual_policy_url_override),
                ),
                url=manual_policy_url_override,
                timeout_s=fetch_timeout_sec,
                phase="manual_policy_fetch",
            )
        override_text = (override_res.text or "").strip()
        if override_res.success and override_text:
            chosen_full = {
                "url": override_res.url or manual_policy_url_override,
                "status_code": override_res.status_code,
                "likeliness_score": policy_likeliness_score(override_text),
                "text_len": len(override_text),
                "text": override_text,
                "cleaned_html": override_res.cleaned_html,
                "raw_html": override_res.raw_html,
                "text_extraction_method": override_res.text_extraction_method or "fallback",
            }
        else:
            warn(
                f"[{etld1(home.url)}] Manual policy URL override fetch failed "
                f"({manual_policy_url_override}); falling back to automatic discovery."
            )

    # Robust mode: try curated per-site policy URL overrides first.
    # Hard cap per-URL at 30 s so anti-bot challenge pages don't hang the whole budget.
    if chosen_full is None and getattr(client, "robust", False):
        per_url_timeout = min(float(fetch_timeout_sec), 30.0)
        for override_url in site_override_urls(etld1(home.url)):
            ov_res = await _await_crawl_result(
                (first_party_policy_fetcher(override_url) if first_party_policy_fetcher
                 else client.fetch(override_url, capture_network=False, remove_overlays=True,
                                    magic=False, scan_full_page=_should_scan_full_page_policy(override_url))),
                url=override_url,
                timeout_s=per_url_timeout,
                phase="site_override_policy_fetch",
            )
            # simple_http fallback on override too (same logic as Fix 7 in try_candidate)
            if not ov_res.success or not (ov_res.text or "").strip():
                try:
                    fallback_ms = max(5000, int(per_url_timeout * 1000))
                    simple = await _simple_http_fetch(
                        override_url,
                        user_agent=getattr(client, "user_agent", None),
                        timeout_ms=fallback_ms,
                        allow_http_fallback=False,
                    )
                    if simple.success and (simple.text or "").strip():
                        ov_res = simple
                except Exception:
                    pass
            ov_text = (ov_res.text or "").strip()
            if ov_res.success and ov_text and policy_likeliness_score(ov_text) >= 3.0:
                chosen_full = {
                    "url": ov_res.url or override_url,
                    "status_code": ov_res.status_code,
                    "likeliness_score": policy_likeliness_score(ov_text),
                    "text_len": len(ov_text),
                    "text": ov_text,
                    "cleaned_html": ov_res.cleaned_html,
                    "raw_html": ov_res.raw_html,
                    "text_extraction_method": ov_res.text_extraction_method or "fallback",
                    "policy_source": "override",
                }
                log(f"[{etld1(home.url)}] Policy fetched via site override: {override_url}")
                break

    if chosen_full is None:
        policy_info = await _fetch_best_policy(
            client,
            home.url,
            home.cleaned_html,
            fetch_timeout_sec=fetch_timeout_sec,
            policy_fetcher=first_party_policy_fetcher,
        )
        chosen_full = policy_info.get("_chosen_full")
    policy_fetch_ms = int((time.perf_counter() - t_policy) * 1000)

    if chosen_full is None and home.text:
        # If the homepage itself looks like a privacy policy, accept it.
        home_text = (home.text or "").strip()
        home_score = policy_likeliness_score(home_text)
        if _homepage_looks_like_policy(home.url, home_text, score=home_score):
            chosen_full = {
                "url": home.url,
                "status_code": home.status_code,
                "likeliness_score": home_score,
                "text_len": len(home_text),
                "text": home_text,
                "cleaned_html": home.cleaned_html,
                "raw_html": home.raw_html,
                "text_extraction_method": home.text_extraction_method or "fallback",
            }
    first_party_policy = None
    if chosen_full:
        fp_url = chosen_full.get("url") or ""
        norm_fp_url = _normalize_url(fp_url)
        use_registry = (
            bool(norm_fp_url)
            and policy_artifact_registry is not None
            and policy_artifact_lock is not None
        )

        reused_fp = False
        if use_registry:
            reused_fp = await _copy_policy_artifact(
                norm_fp_url, site_art_dir, policy_artifact_registry, policy_artifact_lock  # type: ignore[arg-type]
            )
            if reused_fp:
                log(f"[{etld1(site_url)}] Reused first-party policy artifact for {fp_url}")

        if reused_fp:
            # Read back what was copied so we can populate the metadata record.
            policy_txt_path = site_art_dir / "policy.txt"
            cleaned_text = policy_txt_path.read_text(encoding="utf-8") if policy_txt_path.exists() else ""
            raw_text = chosen_full.get("text") or ""
            try:
                ext_data = json.loads((site_art_dir / "policy.extraction.json").read_text(encoding="utf-8"))
            except Exception:
                ext_data = {}
            final_method = ext_data.get("method") or "reused"
            first_party_policy = {
                "url": fp_url,
                "status_code": chosen_full.get("status_code"),
                "likeliness_score": chosen_full.get("likeliness_score"),
                "text_len": len(cleaned_text),
                "text_len_raw": len(raw_text),
                "extraction_method": final_method,
                "word_count": _count_words(cleaned_text),
                "is_english": _is_english_text(cleaned_text),
            }
        else:
            raw_text = chosen_full.get("text") or ""
            llm_cleaned: str | None = None
            if openai_client is not None and raw_text:
                llm_cleaned = await _llm_clean_policy_text(
                    raw_text, client=openai_client, model=llm_model
                )
            cleaned_text = llm_cleaned if llm_cleaned else raw_text
            base_method = chosen_full.get("text_extraction_method") or "fallback"
            final_method = "llm_cleaned" if llm_cleaned else base_method
            first_party_policy = {
                "url": fp_url,
                "status_code": chosen_full.get("status_code"),
                "likeliness_score": chosen_full.get("likeliness_score"),
                "text_len": len(cleaned_text),
                "text_len_raw": len(raw_text),
                "extraction_method": final_method,
                "word_count": _count_words(cleaned_text),
                "is_english": _is_english_text(cleaned_text),
            }
            _write_text(site_art_dir / "policy.txt", cleaned_text)
            _write_json(
                site_art_dir / "policy.extraction.json",
                {
                    "method": final_method,
                    "base_extraction": base_method,
                    "llm_model": llm_model if llm_cleaned else None,
                    "source_url": fp_url,
                },
            )
            if use_registry:
                await _register_policy_artifact(
                    norm_fp_url, site_art_dir, policy_artifact_registry, policy_artifact_lock  # type: ignore[arg-type]
                )

    # Robust mode: Wayback Machine fallback when direct discovery failed entirely.
    if first_party_policy is None and getattr(client, "robust", False):
        wb_urls: list[str] = []
        # Try override candidates first (most likely to have a valid Wayback snapshot).
        for ov in site_override_urls(etld1(home.url)):
            wb_urls.extend(wayback_candidates(ov))
        # Also try the homepage + /privacy as a guess.
        home_origin = f"{urlparse(home.url).scheme}://{urlparse(home.url).netloc}"
        wb_urls.extend(wayback_candidates(home_origin + "/privacy"))
        wb_urls.extend(wayback_candidates(home_origin + "/privacy-policy"))
        # Dedup preserving order.
        seen: set[str] = set()
        wb_urls = [u for u in wb_urls if not (u in seen or seen.add(u))]
        wb_per_url_timeout = min(float(fetch_timeout_sec), 30.0)
        for wb in wb_urls[:6]:
            try:
                wb_res = await _await_crawl_result(
                    client.fetch(wb, capture_network=False, remove_overlays=True, magic=False, scan_full_page=False),
                    url=wb, timeout_s=wb_per_url_timeout, phase="wayback_policy_fetch",
                )
            except Exception:
                continue
            wb_text = (wb_res.text or "").strip()
            if wb_res.success and wb_text and policy_likeliness_score(wb_text) >= 3.0:
                final_method = "wayback"
                _write_text(site_art_dir / "policy.txt", wb_text)
                _write_json(site_art_dir / "policy.extraction.json",
                            {"method": final_method, "source_url": wb_res.url or wb, "wayback": True})
                first_party_policy = {
                    "url": wb_res.url or wb,
                    "status_code": wb_res.status_code,
                    "likeliness_score": policy_likeliness_score(wb_text),
                    "text_len": len(wb_text),
                    "text_len_raw": len(wb_text),
                    "extraction_method": final_method,
                    "word_count": _count_words(wb_text),
                    "is_english": _is_english_text(wb_text),
                }
                policy_source = "wayback_blind"
                log(f"[{etld1(home.url)}] Policy recovered via Wayback: {wb}")
                break

    # CDX-based Wayback discovery: when override + normal discovery both failed,
    # query Wayback's CDX index for archived privacy URLs on this domain.
    policy_source = "direct"
    if first_party_policy is not None and chosen_full and chosen_full.get("policy_source"):
        policy_source = chosen_full["policy_source"]
    elif first_party_policy is not None:
        policy_source = "direct"

    if first_party_policy is None and getattr(client, "robust", False):
        site_domain = etld1(home.url) or etld1(site_url) or ""
        try:
            cdx_urls = await _wayback_cdx_discover(site_domain)
        except Exception:
            cdx_urls = []
        if cdx_urls:
            log(f"[cdx_discover] {site_domain}: found {len(cdx_urls)} candidates")
        for cdx_url in cdx_urls[:4]:
            # Try live fetch first (via simple_http — fast, no Playwright needed)
            try:
                cdx_res = await _simple_http_fetch(
                    cdx_url,
                    user_agent=getattr(client, "user_agent", None),
                    timeout_ms=max(5000, int(fetch_timeout_sec * 1000)),
                    allow_http_fallback=False,
                )
            except Exception:
                cdx_res = None
            if cdx_res and cdx_res.success and (cdx_res.text or "").strip():
                cdx_text = (cdx_res.text or "").strip()
                if policy_likeliness_score(cdx_text) >= 3.0:
                    _write_text(site_art_dir / "policy.txt", cdx_text)
                    _write_json(site_art_dir / "policy.extraction.json",
                                {"method": "cdx_discovery", "source_url": cdx_res.url or cdx_url})
                    first_party_policy = {
                        "url": cdx_res.url or cdx_url,
                        "status_code": cdx_res.status_code,
                        "likeliness_score": policy_likeliness_score(cdx_text),
                        "text_len": len(cdx_text),
                        "text_len_raw": len(cdx_text),
                        "extraction_method": "cdx_discovery",
                        "word_count": _count_words(cdx_text),
                        "is_english": _is_english_text(cdx_text),
                    }
                    policy_source = "cdx_discovery"
                    log(f"[cdx_discover] {site_domain}: recovered policy via {cdx_url}")
                    break

    wall_vendor = _detect_wall_vendor(home)
    if first_party_policy is None:
        status = "policy_not_found"
        non_browsable_reason: str | None = None
        is_nb, reason = _classify_non_browsable(home)
        if is_nb:
            status = "non_browsable"
            non_browsable_reason = reason
            warn(f"[{etld1(home.url)}] Classified as non-browsable ({reason}).")
        else:
            warn(f"[{etld1(home.url)}] Privacy policy not found.")
        return {
            "rank": rank,
            "input": domain_or_url,
            "site_url": site_url,
            "final_url": home.url,
            "site_etld1": etld1(home.url),
            "status": status,
            "home_status_code": home.status_code,
            "home_fetch_mode": home_fetch_mode,
            "home_fetch_attempts": max(1, len(home_errors) + 1),
            "home_ok": status != "non_browsable",
            "policy_found": False,
            "policy_is_english": False,
            "policy_source": None,
            "first_party_policy": None,
            "non_browsable_reason": non_browsable_reason,
            "wall_vendor": wall_vendor,
            "third_parties": [],
            "third_party_policy_fetches": [],
            "third_party_policy_fetched_count": 0,
            "third_party_with_english_policy_count": 0,
            "cookies": {
                "first_party": cookies_bundle["first_party"],
                "third_party": cookies_bundle["third_party"],
            },
            "cookie_counts": cookies_bundle["cookie_counts"],
            "error_code": status,
            "home_fetch_ms": home_fetch_ms,
            "policy_fetch_ms": policy_fetch_ms,
            "third_party_extract_ms": 0,
            "third_party_policy_fetch_ms": 0,
            "first_party_policy_url_override": manual_policy_url_override,
            "total_ms": int((time.perf_counter() - t_total) * 1000),
            "run_id": run_id,
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        }

    # 3) Third-party extraction
    if stage_callback:
        stage_callback("third_party_extract")
    t_tp = time.perf_counter()
    tp_extraction_source = "browser_network"
    if third_party_engine == "openwpm":
        openwpm_dir = site_art_dir / "openwpm"
        try:
            urls = run_openwpm_for_third_parties(home.url, out_dir=openwpm_dir, headless=True)
            network_like = [{"url": u} for u in urls]
            obs = third_parties_from_network_logs(home.url, network_like)
            tp_extraction_source = "openwpm"
        except Exception as e:
            warn(f"[{etld1(home.url)}] OpenWPM failed; falling back to Crawl4AI network logs: {e}")
            obs = third_parties_from_network_logs(home.url, home.network_requests)
    else:
        obs = third_parties_from_network_logs(home.url, home.network_requests)

    # Fix 6 — HTML-parse fallback when no live network capture (simple_http home,
    # Playwright capture race, or capture disabled). Synthesize requests from
    # static HTML resource references.
    if not obs.third_party_etld1s:
        html_for_fallback = home.raw_html or home.cleaned_html
        if html_for_fallback:
            html_requests = _extract_third_party_urls_from_html(html_for_fallback, home.url)
            if html_requests:
                obs2 = third_parties_from_network_logs(home.url, html_requests)
                if obs2.third_party_etld1s:
                    obs = obs2
                    tp_extraction_source = "html_static_fallback"
                    log(f"[tp_fallback] used HTML-parse for {etld1(home.url)} → {len(obs.third_party_etld1s)} TPs")
    third_party_extract_ms = int((time.perf_counter() - t_tp) * 1000)

    third_party_etlds = obs.third_party_etld1s

    # Load TP policy override CSV (curated URLs for trackers missing from Radar/Ghostery)
    _tp_overrides: dict[str, str] | None = None
    def _tp_policy_override(tp_etld: str) -> str | None:
        nonlocal _tp_overrides
        if _tp_overrides is None:
            _tp_overrides = {}
            import csv as _csv
            tp_csv = Path(__file__).resolve().parent / "data" / "tp_policy_overrides.csv"
            if tp_csv.exists():
                try:
                    for row in _csv.DictReader(tp_csv.open("r", encoding="utf-8")):
                        k = (row.get("tp_etld1") or "").strip().lower()
                        v = (row.get("policy_url") or "").strip()
                        if k and v:
                            _tp_overrides[k] = v
                except Exception:
                    pass
        return _tp_overrides.get((tp_etld or "").lower())

    def _merge_entries(radar_entry: TrackerRadarEntry | None, db_entry: TrackerDbEntry | None, tp_etld: str = "") -> dict[str, Any]:
        # Mixed mode: prefer Tracker Radar if present; otherwise fall back to TrackerDB.
        # Then fall back to tp_policy_overrides.csv for policy_url.
        if radar_entry:
            pol = radar_entry.policy_url or _tp_policy_override(tp_etld)
            return {
                "entity": radar_entry.entity,
                "categories": list(radar_entry.categories or []),
                "prevalence": radar_entry.prevalence,
                "policy_url": pol,
                "tracker_radar_source_domain_file": radar_entry.source_domain_file,
                "trackerdb_source_pattern_file": None,
                "trackerdb_source_org_file": None,
            }
        if db_entry:
            pol = db_entry.policy_url or _tp_policy_override(tp_etld)
            return {
                "entity": db_entry.entity,
                "categories": list(db_entry.categories or []),
                "prevalence": db_entry.prevalence,
                "policy_url": pol,
                "tracker_radar_source_domain_file": None,
                "trackerdb_source_pattern_file": db_entry.source_pattern_file,
                "trackerdb_source_org_file": db_entry.source_org_file,
            }
        return {
            "entity": None,
            "categories": [],
            "prevalence": None,
            "policy_url": _tp_policy_override(tp_etld),
            "tracker_radar_source_domain_file": None,
            "trackerdb_source_pattern_file": None,
            "trackerdb_source_org_file": None,
        }

    site_entity: str | None = None
    site_etld = etld1(home.url) or ""
    if tracker_radar:
        site_entry = tracker_radar.lookup(site_etld)
        if site_entry and site_entry.entity:
            site_entity = site_entry.entity
    if not site_entity and trackerdb:
        site_entry_db = trackerdb.lookup(site_etld)
        if site_entry_db and site_entry_db.entity:
            site_entity = site_entry_db.entity

    third_party_records: list[dict[str, Any]] = []
    for tp in third_party_etlds:
        radar_entry = tracker_radar.lookup(tp) if tracker_radar else None
        db_entry = trackerdb.lookup(tp) if trackerdb else None
        merged = _merge_entries(radar_entry, db_entry, tp_etld=tp)
        tp_entity = merged.get("entity")
        if exclude_same_entity and site_entity and tp_entity and tp_entity == site_entity:
            continue
        third_party_records.append({
            "third_party_etld1": tp,
            "entity": merged.get("entity"),
            "categories": merged.get("categories") or [],
            "prevalence": merged.get("prevalence"),
            "policy_url": merged.get("policy_url"),
            "tracker_radar_source_domain_file": merged.get("tracker_radar_source_domain_file"),
            "trackerdb_source_pattern_file": merged.get("trackerdb_source_pattern_file"),
            "trackerdb_source_org_file": merged.get("trackerdb_source_org_file"),
        })

    # 4) Optional: fetch third-party policy texts (best-effort)
    if stage_callback:
        stage_callback("third_party_policy_fetch")
    t_tp_policy = time.perf_counter()
    third_party_policy_fetches: list[dict[str, Any]] = []
    if fetch_third_party_policies and (tracker_radar or trackerdb):
        def sort_key(r: dict[str, Any]):
            p = r.get("prevalence")
            return (-(p if isinstance(p, (int, float)) else -1.0), r["third_party_etld1"])

        policy_fetch_plans: list[dict[str, Any]] = []
        policy_groups: dict[str, dict[str, Any]] = {}
        for rec in third_party_records:
            purl = rec.get("policy_url")
            if not isinstance(purl, str) or not purl.strip():
                continue
            normalized_policy_url = _normalize_url(purl)
            if not normalized_policy_url:
                continue
            group = policy_groups.get(normalized_policy_url)
            if group is None:
                group = {
                    "normalized_policy_url": normalized_policy_url,
                    "leader": rec,
                    "members": [rec],
                }
                policy_groups[normalized_policy_url] = group
                policy_fetch_plans.append(group)
                continue
            group["members"].append(rec)
            if sort_key(rec) < sort_key(group["leader"]):
                group["leader"] = rec

        tp_consecutive_timeouts = 0
        tp_max_consecutive_timeouts = 3
        async def fetch_third_party_policy(plan: dict[str, Any]) -> dict[str, Any] | None:
            rec = plan["leader"]
            purl = rec.get("policy_url")
            if not purl:
                return None
            norm_tp_url = _normalize_url(purl)
            use_tp_registry = (
                bool(norm_tp_url)
                and policy_artifact_registry is not None
                and policy_artifact_lock is not None
            )
            tp_dir = site_art_dir / "third_party" / _safe_dirname(rec["third_party_etld1"])

            reused_tp = False
            if use_tp_registry:
                reused_tp = await _copy_policy_artifact(
                    norm_tp_url, tp_dir, policy_artifact_registry, policy_artifact_lock  # type: ignore[arg-type]
                )
                if reused_tp:
                    log(f"[{etld1(site_url)}] Reused third-party policy artifact for {purl}")

            if reused_tp:
                tp_policy_path = tp_dir / "policy.txt"
                tp_text = tp_policy_path.read_text(encoding="utf-8") if tp_policy_path.exists() else ""
                try:
                    tp_ext_data = json.loads((tp_dir / "policy.extraction.json").read_text(encoding="utf-8"))
                except Exception:
                    tp_ext_data = {}
                tp_method = tp_ext_data.get("method") or "reused"
                return {
                    "third_party_etld1": rec["third_party_etld1"],
                    "policy_url": purl,
                    "fetch_success": True,
                    "status_code": None,
                    "text_len": len(tp_text),
                    "text_len_raw": len(tp_text),
                    "extraction_method": tp_method,
                    "word_count": _count_words(tp_text),
                    "is_english": _is_english_text(tp_text),
                    "error_message": None,
                    "_members": [member["third_party_etld1"] for member in plan["members"]],
                    "_timed_out": False,
                }

            tp_dir.mkdir(parents=True, exist_ok=True)
            if third_party_policy_fetcher is not None:
                res = await _await_crawl_result(
                    third_party_policy_fetcher(purl),
                    url=purl,
                    timeout_s=fetch_timeout_sec,
                    phase="third_party_policy_fetch",
                )
            else:
                res = await _await_crawl_result(
                    client.fetch(
                        purl,
                        capture_network=False,
                        remove_overlays=True,
                        magic=False,
                        scan_full_page=_should_scan_full_page_policy(purl),
                    ),
                    url=purl,
                    timeout_s=fetch_timeout_sec,
                    phase="third_party_policy_fetch",
                )
            tp_text_raw = (res.text or "").strip()
            tp_llm_cleaned: str | None = None
            if openai_client is not None and tp_text_raw:
                tp_llm_cleaned = await _llm_clean_policy_text(
                    tp_text_raw, client=openai_client, model=llm_model
                )
            tp_text = tp_llm_cleaned if tp_llm_cleaned else tp_text_raw
            tp_base_method = res.text_extraction_method or "fallback"
            tp_method = "llm_cleaned" if tp_llm_cleaned else tp_base_method
            if tp_text:
                _write_text(tp_dir / "policy.txt", tp_text)
                _write_json(
                    tp_dir / "policy.extraction.json",
                    {
                        "method": tp_method,
                        "base_extraction": tp_base_method,
                        "llm_model": llm_model if tp_llm_cleaned else None,
                        "source_url": purl,
                    },
                )
            if use_tp_registry and tp_text:
                await _register_policy_artifact(
                    norm_tp_url, tp_dir, policy_artifact_registry, policy_artifact_lock  # type: ignore[arg-type]
                )
            return {
                "third_party_etld1": rec["third_party_etld1"],
                "policy_url": purl,
                "fetch_success": res.success,
                "status_code": res.status_code,
                "text_len": len(tp_text),
                "text_len_raw": len(tp_text_raw),
                "extraction_method": tp_method,
                "word_count": _count_words(tp_text),
                "is_english": _is_english_text(tp_text),
                "error_message": res.error_message,
                "_members": [member["third_party_etld1"] for member in plan["members"]],
                "_timed_out": "timed_out" in (res.error_message or ""),
            }

        ordered_tp_plans = sorted(policy_fetch_plans, key=lambda plan: sort_key(plan["leader"]))[:third_party_policy_max]
        stop_tp_fetch = False
        for idx in range(0, len(ordered_tp_plans), _TP_POLICY_FETCH_BATCH_SIZE):
            batch = ordered_tp_plans[idx: idx + _TP_POLICY_FETCH_BATCH_SIZE]
            batch_results = await asyncio.gather(*(fetch_third_party_policy(plan) for plan in batch))
            for item in batch_results:
                if item is None:
                    continue
                member_etlds = item.pop("_members", [item.get("third_party_etld1")])
                timed_out = bool(item.pop("_timed_out", False))
                for member_etld in member_etlds:
                    third_party_policy_fetches.append({
                        **item,
                        "third_party_etld1": member_etld,
                    })
                if timed_out:
                    tp_consecutive_timeouts += 1
                else:
                    tp_consecutive_timeouts = 0
                if tp_consecutive_timeouts >= tp_max_consecutive_timeouts:
                    warn(f"[{etld1(site_url)}] Bailing out of third-party policy fetch after {tp_consecutive_timeouts} consecutive timeouts")
                    stop_tp_fetch = True
                    break
            if stop_tp_fetch:
                break
    third_party_policy_fetch_ms = int((time.perf_counter() - t_tp_policy) * 1000)

    fetch_method_by_tp = {
        str(item.get("third_party_etld1")): item.get("extraction_method")
        for item in third_party_policy_fetches
        if item.get("third_party_etld1")
    }
    if fetch_method_by_tp:
        for tp in third_party_records:
            et = str(tp.get("third_party_etld1") or "")
            tp["policy_extraction_method"] = fetch_method_by_tp.get(et)

    # 5) Final record
    tp_fetched_ok = sum(1 for p in third_party_policy_fetches if p.get("fetch_success"))
    tp_english_ok = sum(1 for p in third_party_policy_fetches if p.get("fetch_success") and p.get("is_english"))
    return {
        "rank": rank,
        "input": domain_or_url,
        "site_url": site_url,
        "final_url": home.url,
        "site_etld1": etld1(home.url),
        "status": "ok",
        "home_status_code": home.status_code,
        "home_fetch_mode": home_fetch_mode,
        "home_fetch_attempts": max(1, len(home_errors) + 1),
        "home_ok": True,
        "policy_found": True,
        "policy_is_english": bool((first_party_policy or {}).get("is_english")),
        "policy_source": policy_source,
        "first_party_policy": first_party_policy,
        "non_browsable_reason": None,
        "wall_vendor": _detect_wall_vendor(home),
        "third_parties": third_party_records,
        "tp_extraction_source": tp_extraction_source,
        "third_party_policy_fetches": third_party_policy_fetches,
        "third_party_policy_fetched_count": tp_fetched_ok,
        "third_party_with_english_policy_count": tp_english_ok,
        "cookies": {
            "first_party": cookies_bundle["first_party"],
            "third_party": cookies_bundle["third_party"],
        },
        "cookie_counts": cookies_bundle["cookie_counts"],
        "error_code": None,
        "home_fetch_ms": home_fetch_ms,
        "policy_fetch_ms": policy_fetch_ms,
        "third_party_extract_ms": third_party_extract_ms,
        "third_party_policy_fetch_ms": third_party_policy_fetch_ms,
        "first_party_policy_url_override": manual_policy_url_override,
        "total_ms": int((time.perf_counter() - t_total) * 1000),
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
