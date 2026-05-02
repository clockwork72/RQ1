from __future__ import annotations
from urllib.parse import urlparse

# We prefer `tldextract` for correct eTLD+1 parsing, but provide a no-deps fallback
# so the project can still run in constrained environments.
try:
    import tldextract  # type: ignore
    _EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=None)
except Exception:  # pragma: no cover
    _EXTRACTOR = None

def hostname(url: str) -> str | None:
    try:
        h = urlparse(url).hostname
        return h.lower() if h else None
    except Exception:
        return None

def etld1(host_or_url: str) -> str | None:
    h = host_or_url
    if '://' in host_or_url:
        h = hostname(host_or_url) or ""
    if not h:
        return None
    h = h.lower()

    if _EXTRACTOR is not None:
        ext = _EXTRACTOR(h)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
        return h

    # Fallback: naive last-two-label heuristic (OK for many, not all, ccTLDs)
    parts = [p for p in h.split(".") if p]
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return h
