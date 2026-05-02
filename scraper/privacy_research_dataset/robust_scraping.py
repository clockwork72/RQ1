"""Opt-in helpers that lift policy-fetch success rate for hard sites.

Enabled by `--robust-scrape`. Packs together:
  - SPA render hints (scan_full_page + explicit waits + stealth patches)
  - Rotated realistic User-Agent pool
  - Manual policy-URL overrides for known-hard top-Tranco sites
  - Wayback Machine fallback when all direct fetches fail
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from urllib.parse import quote

USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
]


def pick_ua(i: int) -> str:
    return USER_AGENTS[i % len(USER_AGENTS)]


# Injected before page interaction. Patches common automation fingerprints that
# anti-bot systems (Cloudflare, DataDome, Akamai's basic tier) look for.
STEALTH_JS: str = """
try {
  Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
  Object.defineProperty(navigator, 'languages', { get: () => ['en-GB','en'] });
  Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
  Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
  Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 });
  const orig = WebGLRenderingContext.prototype.getParameter;
  WebGLRenderingContext.prototype.getParameter = function(p){
    if (p === 37445) return 'Intel Inc.';
    if (p === 37446) return 'Intel Iris OpenGL Engine';
    return orig.apply(this, [p]);
  };
  window.chrome = window.chrome || { runtime: {} };
} catch(e){}
"""

# Mild JS that nudges lazy-loaded footers/legal links into the DOM and dismisses
# common consent modals so links are visible to the policy-discovery step.
EXPAND_AND_SCROLL_JS: str = """
try {
  ['button[aria-expanded="false"]','[data-testid*="accept"]','[id*="accept"]','[class*="accept"]']
    .forEach(sel => document.querySelectorAll(sel).forEach(el => { try{ el.click(); }catch(e){} }));
  window.scrollTo(0, document.body.scrollHeight);
  window.scrollTo(0, 0);
} catch(e){}
"""


# Curated list of hard-to-discover policy URLs for high-traffic sites. Keyed by
# the site's eTLD+1. First URL is preferred; rest are fallbacks.
# Sources: each site's footer or privacy hub, verified manually.
SITE_POLICY_OVERRIDES: dict[str, list[str]] = {
    "facebook.com":   ["https://www.facebook.com/privacy/policy/"],
    "instagram.com":  ["https://privacycenter.instagram.com/policy/"],
    "whatsapp.com":   ["https://www.whatsapp.com/legal/privacy-policy"],
    "wikipedia.org":  ["https://foundation.wikimedia.org/wiki/Policy:Privacy_policy"],
    "wikimedia.org":  ["https://foundation.wikimedia.org/wiki/Policy:Privacy_policy"],
    "x.com":          ["https://x.com/en/privacy", "https://twitter.com/en/privacy"],
    "twitter.com":    ["https://twitter.com/en/privacy"],
    "amazon.com":     ["https://www.amazon.com/gp/help/customer/display.html?nodeId=468496"],
    "amazon.co.uk":   ["https://www.amazon.co.uk/gp/help/customer/display.html?nodeId=201909010"],
    "amazon.de":      ["https://www.amazon.de/gp/help/customer/display.html?nodeId=201909010"],
    "amazon.fr":      ["https://www.amazon.fr/gp/help/customer/display.html?nodeId=201909010"],
    "apple.com":      ["https://www.apple.com/legal/privacy/en-ww/"],
    "icloud.com":     ["https://www.apple.com/legal/privacy/en-ww/"],
    "microsoft.com":  ["https://privacy.microsoft.com/en-us/privacystatement"],
    "bing.com":       ["https://privacy.microsoft.com/en-us/privacystatement"],
    "office.com":     ["https://privacy.microsoft.com/en-us/privacystatement"],
    "live.com":       ["https://privacy.microsoft.com/en-us/privacystatement"],
    "linkedin.com":   ["https://www.linkedin.com/legal/privacy-policy"],
    "github.com":     ["https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement"],
    "tiktok.com":     ["https://www.tiktok.com/legal/page/row/privacy-policy/en"],
    "pinterest.com":  ["https://policy.pinterest.com/en/privacy-policy"],
    "reddit.com":     ["https://www.reddit.com/policies/privacy-policy"],
    "netflix.com":    ["https://help.netflix.com/legal/privacy"],
    "spotify.com":    ["https://www.spotify.com/us/legal/privacy-policy/"],
    "snapchat.com":   ["https://values.snap.com/privacy/privacy-policy"],
    "twitch.tv":      ["https://www.twitch.tv/p/legal/privacy-notice/"],
    "ebay.com":       ["https://www.ebay.com/help/policies/member-behaviour-policies/user-privacy-notice-privacy-policy?id=4260"],
    "paypal.com":     ["https://www.paypal.com/us/legalhub/privacy-full"],
    "google.com":     ["https://policies.google.com/privacy?hl=en"],
    "youtube.com":    ["https://policies.google.com/privacy?hl=en"],
    "gmail.com":      ["https://policies.google.com/privacy?hl=en"],
    "blogspot.com":   ["https://policies.google.com/privacy?hl=en"],
    "adobe.com":      ["https://www.adobe.com/privacy/policy.html"],
    "zoom.us":        ["https://explore.zoom.us/en/privacy/"],
    "dropbox.com":    ["https://www.dropbox.com/privacy"],
    "slack.com":      ["https://slack.com/trust/privacy/privacy-policy"],
    "salesforce.com": ["https://www.salesforce.com/company/privacy/full_privacy/"],
    "oracle.com":     ["https://www.oracle.com/legal/privacy/privacy-policy.html"],
    "ibm.com":        ["https://www.ibm.com/privacy"],
    "intel.com":      ["https://www.intel.com/content/www/us/en/privacy/intel-privacy-notice.html"],
    "nvidia.com":     ["https://www.nvidia.com/en-us/about-nvidia/privacy-policy/"],
    "samsung.com":    ["https://www.samsung.com/us/account/privacy-policy/"],
    "hp.com":         ["https://www.hp.com/us-en/privacy/privacy.html"],
    "dell.com":       ["https://www.dell.com/learn/us/en/uscorp1/policies-privacy"],
    "cisco.com":      ["https://www.cisco.com/c/en/us/about/legal/privacy-full.html"],
    "cloudflare.com": ["https://www.cloudflare.com/privacypolicy/"],
    "digicert.com":   ["https://www.digicert.com/legal-repository/legal-privacy"],
    "verisign.com":   ["https://www.verisign.com/en_US/company-information/privacy/index.xhtml"],
    "wordpress.org":  ["https://wordpress.org/about/privacy/"],
    "wordpress.com":  ["https://automattic.com/privacy/"],
    "mozilla.org":    ["https://www.mozilla.org/en-US/privacy/websites/"],
    "yahoo.com":      ["https://legal.yahoo.com/us/en/yahoo/privacy/index.html"],
    "duckduckgo.com": ["https://duckduckgo.com/privacy"],
    "openai.com":     ["https://openai.com/policies/privacy-policy"],
    "bbc.com":        ["https://www.bbc.com/privacy/"],
    "bbc.co.uk":      ["https://www.bbc.co.uk/privacy/"],
    "cnn.com":        ["https://edition.cnn.com/privacy"],
    "nytimes.com":    ["https://help.nytimes.com/hc/en-us/articles/10940941449序-Privacy-Policy"],
    "reuters.com":    ["https://www.reuters.com/info-pages/privacy-policy/"],
    "theguardian.com":["https://www.theguardian.com/help/privacy-policy"],
    "spiegel.de":     ["https://www.spiegel.de/datenschutz/"],
    "lemonde.fr":     ["https://www.lemonde.fr/confidentialite/"],
    "aliexpress.com": ["https://rule.alibaba.com/rule/detail/2034.htm"],
    "alibaba.com":    ["https://rule.alibaba.com/rule/detail/2034.htm"],
    "baidu.com":      ["https://privacy.baidu.com/detail?id=284"],
    "yandex.com":     ["https://yandex.com/legal/confidential/"],
    "yandex.ru":      ["https://yandex.ru/legal/confidential/"],
    "vk.com":         ["https://vk.com/privacy"],
    "canva.com":      ["https://www.canva.com/policies/privacy-policy/"],
    "figma.com":      ["https://www.figma.com/privacy/"],
    "medium.com":     ["https://policy.medium.com/medium-privacy-policy-f03bf92035c9"],
    "quora.com":      ["https://www.quora.com/about/privacy"],
    "stackoverflow.com": ["https://stackoverflow.com/legal/privacy-policy"],
    "discord.com":    ["https://discord.com/privacy"],
    "telegram.org":   ["https://telegram.org/privacy"],
    "booking.com":    ["https://www.booking.com/content/privacy.html"],
    "airbnb.com":     ["https://www.airbnb.com/help/article/3175"],
    "uber.com":       ["https://www.uber.com/legal/en/document/?name=privacy-notice&country=united-states&lang=en"],
    "shopify.com":    ["https://www.shopify.com/legal/privacy"],
    "wix.com":        ["https://www.wix.com/about/privacy"],
    "squarespace.com":["https://www.squarespace.com/privacy"],
    "vimeo.com":      ["https://vimeo.com/privacy"],
    "etsy.com":       ["https://www.etsy.com/legal/privacy/"],
    "soundcloud.com": ["https://soundcloud.com/pages/privacy"],
    "tumblr.com":     ["https://www.tumblr.com/privacy"],
    "flickr.com":     ["https://www.flickr.com/help/privacy"],
    "imgur.com":      ["https://imgur.com/privacy"],
    "nike.com":       ["https://agreementservice.svs.nike.com/rest/agreement?uxId=com.nike.commerce.nikedotcom.web&country=US&language=en&agreementType=privacyPolicy"],
    "zoom.com":       ["https://explore.zoom.us/en/privacy/"],
    "webex.com":      ["https://www.cisco.com/c/en/us/about/legal/privacy-full.html"],
    # Added from failure investigation — top-1000 sites blocked by anti-bot or geo-localized
    "msn.com":        ["https://privacy.microsoft.com/en-us/privacystatement"],
    "nginx.org":      ["https://nginx.com/privacy-policy/"],
    "t.me":           ["https://telegram.org/privacy"],
    "telegram.me":    ["https://telegram.org/privacy"],
    "nist.gov":       ["https://www.nist.gov/privacy-policy"],
    "nih.gov":        ["https://www.nih.gov/web-policies-notices"],
    "archive.org":    ["https://archive.org/about/terms.php"],
    "godaddy.com":    ["https://www.godaddy.com/legal/agreements/privacy-policy"],
    "weather.com":    ["https://weather.com/privacy-settings"],
    "reuters.com":    ["https://www.reuters.com/info-pages/privacy-policy/"],
    "php.net":        ["https://www.php.net/privacy.php"],
    "prudential.com": ["https://www.prudential.com/privacy-center/online-privacy-policy"],
    "wsj.com":        ["https://www.dowjones.com/privacy-notice/"],
    "stanford.edu":   ["https://www.stanford.edu/site/privacy/"],
    "espn.com":       ["https://privacy.thewaltdisneycompany.com/en/current-privacy-policy/"],
    "telegraph.co.uk":["https://www.telegraph.co.uk/about-us/privacy-policy/"],
    "temu.com":       ["https://www.temu.com/privacy-and-cookie-policy.html"],
    "independent.co.uk": ["https://www.independent.co.uk/service/privacy-policy-a6184181.html"],
    "steamcommunity.com": ["https://store.steampowered.com/privacy_agreement/"],
    "ui.com":         ["https://www.ui.com/legal/privacypolicy/"],
    "ip-api.com":     ["https://ip-api.com/docs/legal"],
    "naver.com":      ["https://policy.naver.com/policy/privacy_en.html"],
    "appsflyer.com":  ["https://www.appsflyer.com/legal/privacy-policy/"],
    "weibo.com":      ["https://weibo.com/signup/v5/privacy"],
    "oxylabs.io":     ["https://oxylabs.io/legal/privacy"],
    "adobe.com":      ["https://www.adobe.com/privacy/policy.html"],
}


_CSV_OVERRIDES: dict[str, list[str]] | None = None


def _load_csv_overrides() -> dict[str, list[str]]:
    """Load overrides from data/site_policy_overrides.csv if it exists."""
    global _CSV_OVERRIDES
    if _CSV_OVERRIDES is not None:
        return _CSV_OVERRIDES
    import csv
    from pathlib import Path
    csv_path = Path(__file__).resolve().parent / "data" / "site_policy_overrides.csv"
    out: dict[str, list[str]] = {}
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                site = (row.get("site_etld1") or "").strip().lower()
                url = (row.get("policy_url") or "").strip()
                if site and url:
                    out.setdefault(site, []).append(url)
    # Merge with embedded dict (CSV takes precedence, embedded fills gaps)
    for site, urls in SITE_POLICY_OVERRIDES.items():
        if site.lower() not in out:
            out[site.lower()] = list(urls)
    _CSV_OVERRIDES = out
    return out


def site_override_urls(site_etld1: str | None) -> list[str]:
    if not site_etld1:
        return []
    overrides = _load_csv_overrides()
    return list(overrides.get(site_etld1.lower(), []))


# Wayback Machine fallback: return candidate archived URLs for a given live
# policy URL. `web.archive.org/web/YYYY/<url>` redirects to the closest snapshot.
# We generate a few recency-ordered candidates.
def wayback_candidates(policy_url: str, years: Iterable[int] = (2025, 2024, 2023)) -> list[str]:
    if not policy_url:
        return []
    encoded = policy_url  # Wayback accepts raw URLs in the path
    out: list[str] = []
    for y in years:
        out.append(f"https://web.archive.org/web/{y}/{encoded}")
    # Also a generic "latest" form that Wayback resolves via 302
    out.append(f"https://web.archive.org/web/2025*/{encoded}")
    return out


@dataclass
class RobustConfig:
    enabled: bool = False
    rotate_ua: bool = True
    stealth: bool = True
    retry_home_enhanced: bool = True
    site_overrides: bool = True
    wayback_fallback: bool = True
