from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .utils.etld import etld1, hostname

@dataclass
class ThirdPartyObservation:
    site_etld1: str
    third_party_etld1s: list[str]
    raw_hosts: list[str]

def third_parties_from_network_logs(site_url: str, network_requests: list[dict[str, Any]] | None) -> ThirdPartyObservation:
    site_et1 = etld1(site_url) or ""
    hosts: set[str] = set()
    etlds: set[str] = set()

    if not network_requests:
        return ThirdPartyObservation(site_etld1=site_et1, third_party_etld1s=[], raw_hosts=[])

    for ev in network_requests:
        url = ev.get("url")
        if not url or not isinstance(url, str):
            continue
        h = hostname(url)
        if not h:
            continue
        hosts.add(h)
        e = etld1(h)
        if not e:
            continue
        if e != site_et1:
            etlds.add(e)

    return ThirdPartyObservation(
        site_etld1=site_et1,
        third_party_etld1s=sorted(etlds),
        raw_hosts=sorted(hosts),
    )
