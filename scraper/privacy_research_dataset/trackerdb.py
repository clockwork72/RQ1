from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class TrackerDbEntry:
    etld1: str
    entity: str | None
    categories: list[str]
    prevalence: float | None
    policy_url: str | None
    source_pattern_file: str | None
    source_org_file: str | None

class TrackerDbIndex:
    """
    Loads a prebuilt JSON index produced by scripts/build_trackerdb_index.py

    Index format:
      {
        "<etld1>": {
          "entity": "...",
          "categories": [...],
          "prevalence": null,
          "policy_url": "...",
          "source_pattern_file": "db/patterns/....eno",
          "source_org_file": "db/organizations/....eno"
        },
        ...
      }
    """
    def __init__(self, index_path: str | Path) -> None:
        p = Path(index_path)
        data = json.loads(p.read_text(encoding="utf-8"))
        self._data: dict[str, dict[str, Any]] = data

    def lookup(self, etld1: str) -> TrackerDbEntry | None:
        rec = self._data.get(etld1)
        if not rec:
            return None
        return TrackerDbEntry(
            etld1=etld1,
            entity=rec.get("entity"),
            categories=list(rec.get("categories") or []),
            prevalence=rec.get("prevalence"),
            policy_url=rec.get("policy_url"),
            source_pattern_file=rec.get("source_pattern_file"),
            source_org_file=rec.get("source_org_file"),
        )
