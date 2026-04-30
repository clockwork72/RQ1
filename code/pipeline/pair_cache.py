"""Pair-level analysis cache.

Keyed on the SHA-256 of the input policy texts plus the pinned versions of
every downstream step, so a cache hit is a byte-identical guarantee that
re-running :func:`pipeline.run_pair` would produce the same extraction,
graph, pattern, and Stage-1 verifier output. Stored on lustre at
``data/cache/pairs/<key>.json`` and written first-writer-wins via atomic
``os.replace`` — safe for concurrent HPC shards.

Two use sites:

* :func:`pipeline.run_pair` checks this cache at entry; a hit short-circuits
  all LLM + graph + pattern work and returns the cached analysis with the
  current pair_id/timestamp spliced in.
* ``scripts/carryover_by_policy_hash.py`` seeds the cache from an archived
  run whose policy texts still byte-match the current manifest.

Stage-2 cluster verdicts are *not* part of the cached payload because they
are computed corpus-wide after all per-pair work. ``run_batch`` re-runs
Stage-2 on every returned result (cached or fresh), so reused pairs get
corpus-consistent cluster fields grafted on downstream.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .config import (
    BUILD_PPS_VERSION,
    CACHE_DIR,
    EXTRACTION_PROMPT_VERSION,
    VERIFIER_PROMPT_VERSION,
)

PAIR_CACHE_DIR = CACHE_DIR / "pairs"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def pair_cache_key(
    website_text: str,
    vendor_text: str,
    vendor_name: str,
    service_type: str,
    patterns_version: str,
) -> str:
    """Deterministic SHA-256 over every input that affects run_pair output.

    Policy texts are hashed once and their digest is folded into the outer
    hash — this keeps the key stable regardless of large-text ordering and
    avoids rehashing multi-MB policies repeatedly on lookup.
    """
    body = "|".join([
        BUILD_PPS_VERSION,
        EXTRACTION_PROMPT_VERSION,
        VERIFIER_PROMPT_VERSION,
        patterns_version,
        vendor_name or "",
        service_type or "",
        _sha(website_text),
        _sha(vendor_text),
    ])
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def pair_cache_path(key: str) -> Path:
    return PAIR_CACHE_DIR / f"{key}.json"


def load_pair_cache(key: str) -> dict | None:
    p = pair_cache_path(key)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        # A half-written / corrupt cache file should not abort the pair;
        # treat as a miss and let run_pair regenerate it.
        return None


def save_pair_cache(key: str, payload: dict) -> bool:
    """Atomic, first-writer-wins write. Returns True if this call wrote the
    file, False if a peer won the race (already exists) or write failed."""
    PAIR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = pair_cache_path(key)
    if p.exists():
        return False
    tmp = p.parent / f"{p.name}.tmp.{os.getpid()}"
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, p)
        return True
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        return False
