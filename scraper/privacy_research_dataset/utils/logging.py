from __future__ import annotations
import sys
from datetime import datetime, timezone

def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    sys.stdout.write(f"[{ts}] {msg}\n")
    sys.stdout.flush()

def warn(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    sys.stderr.write(f"[{ts}] WARN: {msg}\n")
    sys.stderr.flush()
