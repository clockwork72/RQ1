from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .utils.logging import warn

def run_openwpm_for_third_parties(
    site_url: str,
    *,
    out_dir: str | Path,
    headless: bool = True,
    sleep: int = 15,
    timeout: int = 60,
) -> list[str]:
    """
    Run a minimal OpenWPM visit to collect HTTP request URLs.

    This is OPTIONAL and only used when you explicitly select the OpenWPM engine.
    OpenWPM is heavyweight; for many Step-1 datasets, Crawl4AI's network capture is sufficient.

    OpenWPM docs describe enabling `http_instrument` and using the `http_requests` table.
    """
    try:
        from openwpm.config import BrowserParams, ManagerParams, validate_crawl_configs
        from openwpm.task_manager import TaskManager
        from openwpm.command_sequence import CommandSequence
    except Exception as e:
        raise RuntimeError(
            "OpenWPM is not installed/available. Install OpenWPM separately (often via its Docker/conda setup)."
        ) from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manager_params = ManagerParams(num_browsers=1)
    # Try to set output directory where possible
    for attr in ("data_directory", "output_directory", "log_directory"):
        if hasattr(manager_params, attr):
            try:
                setattr(manager_params, attr, out_dir)
            except Exception:
                pass

    browser_params = [BrowserParams() for _ in range(manager_params.num_browsers)]
    bp = browser_params[0]

    # Instrumentation (http_requests / http_responses tables)
    if hasattr(bp, "http_instrument"):
        bp.http_instrument = True
    else:
        # Older configs used dict-style keys; keep as last resort
        try:
            bp["http_instrument"] = True  # type: ignore[index]
        except Exception:
            pass

    # Headless
    if hasattr(bp, "display_mode"):
        bp.display_mode = "headless" if headless else "native"

    # Validate configs (OpenWPM will raise if inconsistent)
    validate_crawl_configs(manager_params, browser_params)

    # Run one site visit
    manager = TaskManager(manager_params, browser_params)
    try:
        cs = CommandSequence(site_url, reset=True)
        cs.get(sleep=sleep, timeout=timeout)
        manager.execute_command_sequence(cs)
    finally:
        manager.close()

    # Locate sqlite DB
    sqlite_files = sorted(out_dir.rglob("*.sqlite"))
    if not sqlite_files:
        sqlite_files = sorted(out_dir.rglob("*.db"))
    if not sqlite_files:
        warn(f"OpenWPM finished but no sqlite DB found under {out_dir}")
        return []

    db_path = sqlite_files[0]
    urls: set[str] = set()
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        # Standard join used in OpenWPM docs
        for (url,) in cur.execute("SELECT DISTINCT h.url FROM http_requests as h;"):
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                urls.add(url)
        con.close()
    except Exception as e:
        warn(f"Failed to query OpenWPM DB {db_path}: {e}")
        return []

    return sorted(urls)
