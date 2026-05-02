from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Iterable

def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False, indent=2)
    tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(p)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
