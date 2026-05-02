from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
from urllib.parse import urlparse
from typing import Any, Optional

from .text_extract import extract_main_text_with_method
from .utils.logging import warn
from .robust_scraping import STEALTH_JS as _ROBUST_STEALTH_JS, EXPAND_AND_SCROLL_JS as _ROBUST_EXPAND_JS, pick_ua as _robust_pick_ua

# Run in the browser after page load to expose hidden/collapsed content before extraction.
_EXPAND_COLLAPSED_JS = (
    "document.querySelectorAll('button[aria-expanded=\"false\"]')"
    ".forEach(function(el){try{el.click();}catch(e){}});"
    "document.querySelectorAll('[aria-hidden=\"true\"]')"
    ".forEach(function(el){el.removeAttribute('aria-hidden');});"
)

@dataclass
class Crawl4AIResult:
    url: str
    success: bool
    status_code: int | None
    raw_html: str | None
    cleaned_html: str | None
    text: str | None
    network_requests: list[dict[str, Any]] | None
    error_message: str | None
    text_extraction_method: str | None = None
    response_headers: dict[str, Any] | None = None
    network_responses: list[dict[str, Any]] | None = None
    playwright_cookies: list[dict[str, Any]] | None = None

def _extract_network(result: Any) -> list[dict[str, Any]] | None:
    # Crawl4AI docs mention `result.network_requests` (v0.7.x).
    # Some older docs/examples mention `captured_requests`. Support both.
    nr = getattr(result, "network_requests", None)
    if nr is None:
        nr = getattr(result, "captured_requests", None)
    return nr

def _extract_text(result: Any) -> str | None:
    # Prefer Crawl4AI markdown if available, otherwise fall back to cleaned_html text.
    md = getattr(result, "markdown", None)
    if isinstance(md, str):
        return md
    if md is not None:
        for attr in ("raw_markdown", "fit_markdown", "markdown"):
            v = getattr(md, attr, None)
            if isinstance(v, str) and v.strip():
                return v
    # Some versions expose markdown_v2; treat as deprecated fallback
    md2 = getattr(result, "markdown_v2", None)
    if isinstance(md2, str) and md2.strip():
        return md2
    return None


def _filter_kwargs(cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Filter kwargs to only those accepted by a class' __init__.

    IMPORTANT: If the target __init__ accepts **kwargs, we must NOT filter by
    signature names. Otherwise we accidentally drop valid parameters like
    `verbose`, `log_console`, `capture_network_requests`, etc., which causes
    Crawl4AI to fall back to its own defaults (often verbose=True).
    """
    # Always drop Nones
    cleaned = {k: v for k, v in kwargs.items() if v is not None}

    try:
        sig = inspect.signature(cls.__init__)

        # If __init__ has **kwargs, do not filter (best compatibility).
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return cleaned

        allowed = set(sig.parameters.keys())
        allowed.discard("self")
        return {k: v for k, v in cleaned.items() if k in allowed}

    except Exception:
        # If introspection fails, fall back to best-effort (keep cleaned)
        return cleaned


def _proxy_to_proxy_config(proxy: str) -> dict[str, str]:
    """Convert a proxy URL to Crawl4AI ProxyConfig dict.

    Crawl4AI uses Playwright-style proxy fields: server, username, password.
    """
    p = urlparse(proxy)
    cfg: dict[str, str] = {"server": proxy}
    if p.username:
        cfg["username"] = p.username
    if p.password:
        cfg["password"] = p.password
    return cfg

class Crawl4AIClient:
    """
    Thin wrapper around Crawl4AI AsyncWebCrawler.

    We keep the interface stable and handle minor API differences across versions.
    """

    def __init__(
        self,
        browser_type: str = "chromium",
        headless: bool = True,
        verbose: bool = False,
        user_agent: str | None = None,
        proxy: str | None = None,
        locale: str | None = None,
        timezone_id: str | None = None,
        page_timeout_ms: int = 15000,
        fetch_semaphore: asyncio.Semaphore | None = None,
        robust: bool = False,
        aggressive: bool = False,
        user_agents: list[str] | None = None,
        block_heavy_assets: bool = False,
    ) -> None:
        self.browser_type = browser_type
        self.headless = headless
        self.verbose = verbose
        self.user_agent = user_agent
        self.proxy = proxy
        self.locale = locale
        self.timezone_id = timezone_id
        self.page_timeout_ms = page_timeout_ms
        self.fetch_semaphore = fetch_semaphore
        self._crawler = None
        # robust  -> enable per-site policy-URL overrides + Wayback fallback (light-touch)
        # aggressive -> stealth JS + UA rotation + enhanced rendering (off by default:
        #               empirically causes regressions on anti-bot datacenter IPs)
        self.robust = bool(robust)
        self.aggressive = bool(aggressive)
        self.user_agents = list(user_agents or [])
        self._ua_counter = 0
        self.block_heavy_assets = bool(block_heavy_assets)

    async def __aenter__(self) -> "Crawl4AIClient":
        try:
            from crawl4ai import AsyncWebCrawler, BrowserConfig
        except Exception as e:
            raise RuntimeError(
                "Crawl4AI is not installed or failed to import. Install with `pip install crawl4ai`."
            ) from e

        # BrowserConfig evolves rapidly; keep this robust by filtering kwargs.
        bc_kwargs: dict[str, Any] = dict(
            browser_type=self.browser_type,
            headless=self.headless,
            verbose=self.verbose,
            user_agent=self.user_agent,
        )
        # Disable image/video/font loading at the Chromium level to cut bandwidth ~70%.
        # Policies are text, we never need rendered media. Blink setting plus --disable-features
        # catches most sources. Applied via `extra_args` on BrowserConfig when supported.
        if self.block_heavy_assets:
            bc_kwargs["extra_args"] = [
                "--blink-settings=imagesEnabled=false",
                "--disable-features=AutoplayIgnoresWebAudio",
                "--disable-gpu",
                "--disable-remote-fonts",
                "--disable-images",
                "--disable-dev-shm-usage",
                "--disable-software-rasterizer",
            ]
        # Crawl4AI docs (v0.7.x) use `proxy_config` rather than `proxy`.
        if self.proxy:
            proxy_cfg = _proxy_to_proxy_config(self.proxy)
            bc_kwargs["proxy_config"] = proxy_cfg
            # Some older versions may accept `proxy`.
            bc_kwargs["proxy"] = self.proxy

        bc_kwargs = _filter_kwargs(BrowserConfig, bc_kwargs)
        browser_cfg = BrowserConfig(**bc_kwargs)
        self._crawler = AsyncWebCrawler(config=browser_cfg)
        await self._crawler.start()

        # Register hook to grab cookies from Playwright context right before HTML retrieval.
        # Cookies are keyed by both the page's full URL and by hostname (robust to
        # www/non-www + trailing-slash mismatches between hook and caller URLs).
        self._cookies_by_url: dict[str, list[dict[str, Any]]] = {}
        self._cookies_by_host: dict[str, list[dict[str, Any]]] = {}
        try:
            strat = getattr(self._crawler, "crawler_strategy", None)
            if strat is not None and hasattr(strat, "set_hook"):
                async def _cookie_hook(page, context=None, config=None, **_kw):
                    try:
                        ctx = context or getattr(page, "context", None)
                        if ctx is None:
                            return
                        all_cookies = list(await ctx.cookies() or [])
                        purl = getattr(page, "url", "") or ""
                        if purl:
                            self._cookies_by_url[purl] = all_cookies
                        host = urlparse(purl).netloc if purl else ""
                        if host:
                            # Strip www. to normalize between "www.x.com" and "x.com".
                            host_noww = host[4:] if host.startswith("www.") else host
                            self._cookies_by_host[host_noww] = all_cookies
                    except Exception:
                        pass
                strat.set_hook("before_retrieve_html", _cookie_hook)

                # Route-based blocking is handled by Crawl4AI's text_mode
                # at context creation (blocks image/font/media file extensions).
                # JS stays enabled because --disable-javascript is NOT in extra_args.
        except Exception:
            pass
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._crawler:
            await self._crawler.close()
        self._crawler = None

    async def fetch(
        self,
        url: str,
        *,
        capture_network: bool = False,
        remove_overlays: bool = True,
        magic: bool = False,
        scan_full_page: bool = False,
        wait_for: str | None = None,
        wait_for_timeout_ms: int | None = None,
        enhanced: bool = False,
    ) -> Crawl4AIResult:
        if not self._crawler:
            raise RuntimeError("Crawl4AIClient must be used as an async context manager.")

        from crawl4ai import CrawlerRunConfig, CacheMode

        effective_magic = bool(magic) or (enhanced and self.aggressive)
        effective_scan = bool(scan_full_page) or (enhanced and self.aggressive)
        if enhanced and self.aggressive:
            js_code = _ROBUST_STEALTH_JS + "\n" + _ROBUST_EXPAND_JS + "\n" + _EXPAND_COLLAPSED_JS
        elif self.aggressive:
            js_code = _ROBUST_STEALTH_JS + "\n" + _EXPAND_COLLAPSED_JS
        else:
            js_code = _EXPAND_COLLAPSED_JS

        per_request_ua: str | None = None
        if self.aggressive and self.user_agents:
            per_request_ua = _robust_pick_ua(self._ua_counter)
            self._ua_counter += 1

        # CrawlerRunConfig evolves rapidly; keep this robust by filtering kwargs.
        cfg_kwargs: dict[str, Any] = {
    "cache_mode": CacheMode.BYPASS,

    # Control Crawl4AI verbosity (docs show default can be verbose=True)
    "verbose": bool(self.verbose),

    # Ensure we DON'T collect/print JS console chatter unless explicitly asked
    "log_console": False,
    "capture_console_messages": False,

    # Overlay removal parameter name differs across versions.
    "remove_overlay_elements": remove_overlays,
    "remove_overlays": remove_overlays,

    "magic": effective_magic,
    "scan_full_page": effective_scan,
    "js_code": js_code,

    # Locale/timezone belong to *CrawlerRunConfig* in recent versions.
    "locale": self.locale,
    "timezone_id": self.timezone_id,
    "extra_http_headers": {"Accept-Language": "en-US,en;q=0.9"},
}
        if per_request_ua:
            cfg_kwargs["user_agent"] = per_request_ua


        # Network capture flags (names per docs v0.7.x)
        if capture_network:
            cfg_kwargs["capture_network_requests"] = True
        # Strip images/fonts from payload extraction too (belt-and-suspenders with Chromium flag).
        if self.block_heavy_assets:
            cfg_kwargs["exclude_all_images"] = True
            cfg_kwargs["exclude_external_images"] = True

        # Waiting controls
        effective_wait_for = wait_for
        if enhanced and self.aggressive and not effective_wait_for:
            # Wait for at least one anchor or footer element before scraping.
            # Soft-fail: Crawl4AI will just time out to page_timeout if absent.
            effective_wait_for = "css:a[href], footer, [class*='footer'], [id*='footer']"
        if effective_wait_for:
            cfg_kwargs["wait_for"] = effective_wait_for
        if wait_for_timeout_ms is not None:
            cfg_kwargs["wait_for_timeout"] = wait_for_timeout_ms
        # Extend page timeout significantly in enhanced mode (SPAs need to settle).
        effective_page_timeout = self.page_timeout_ms
        if enhanced and self.aggressive:
            effective_page_timeout = max(self.page_timeout_ms, 45000)
        cfg_kwargs["page_timeout"] = effective_page_timeout

        run_cfg = CrawlerRunConfig(**_filter_kwargs(CrawlerRunConfig, cfg_kwargs))

        try:
            if self.fetch_semaphore is None:
                res = await self._crawler.arun(url=url, config=run_cfg)
            else:
                async with self.fetch_semaphore:
                    res = await self._crawler.arun(url=url, config=run_cfg)
        except Exception as e:
            return Crawl4AIResult(
                url=url,
                success=False,
                status_code=None,
                raw_html=None,
                cleaned_html=None,
                text=None,
                text_extraction_method=None,
                network_requests=None,
                error_message=str(e),
            )

        success = bool(getattr(res, "success", False))
        status_code = getattr(res, "status_code", None)
        raw_html = getattr(res, "html", None)
        cleaned_html = getattr(res, "cleaned_html", None)
        error_message = getattr(res, "error_message", None)
        network_requests = None
        network_responses = None
        if capture_network:
            nr = _extract_network(res) or []
            keep_req = {"request", "request_failed"}
            network_requests = [
                ev for ev in nr
                if isinstance(ev, dict) and ev.get("event_type") in keep_req and isinstance(ev.get("url"), str)
            ]
            # Also capture responses so we can extract Set-Cookie headers.
            network_responses = [
                ev for ev in nr
                if isinstance(ev, dict) and ev.get("event_type") == "response" and isinstance(ev.get("url"), str)
            ]

        # Text extraction (job 2): Trafilatura-first from cleaned/raw HTML.
        text, extraction_method = extract_main_text_with_method(cleaned_html or raw_html, source_url=url)
        if not text or not text.strip():
            # Fallback to Crawl4AI markdown fields if extraction yields nothing.
            text = _extract_text(res)
            if text and text.strip():
                extraction_method = "fallback"
        if not text or not text.strip():
            warn(f"Text extraction returned empty output for {url}")
            text = None
            extraction_method = None

        response_headers = getattr(res, "response_headers", None)
        if not isinstance(response_headers, dict):
            response_headers = None

        final_url = getattr(res, "url", url) or url
        # Pull cookies gathered by the hook. Try URL, then redirected URL, then hostname
        # (robust to www/non-www differences between hook's page.url and caller URLs).
        playwright_cookies: list[dict[str, Any]] | None = None
        for key in (final_url, getattr(res, "redirected_url", None), url):
            if key and key in self._cookies_by_url:
                playwright_cookies = self._cookies_by_url.pop(key)
                break
        if playwright_cookies is None:
            for key in (final_url, url):
                if not key: continue
                host = urlparse(key).netloc
                host_noww = host[4:] if host.startswith("www.") else host
                if host_noww in self._cookies_by_host:
                    playwright_cookies = self._cookies_by_host.pop(host_noww)
                    break
        if playwright_cookies is None:
            playwright_cookies = []

        return Crawl4AIResult(
            url=final_url,
            success=success,
            status_code=status_code,
            raw_html=raw_html,
            cleaned_html=cleaned_html,
            text=text,
            text_extraction_method=extraction_method,
            network_requests=network_requests,
            error_message=error_message,
            response_headers=response_headers,
            network_responses=network_responses,
            playwright_cookies=playwright_cookies,
        )
