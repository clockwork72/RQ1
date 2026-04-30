#!/usr/bin/env python3
"""Horizontal first-party availability per content category, abbreviated labels.

Produces fp_availability_by_category_h.{pdf,png}. Matches the visual style
of the other RQ1 figures (style.py palette, 300 DPI, ACM column-friendly
size). Denominator is the set of sites that *successfully rendered their
homepage* (home_ok=True), which lines up with the Tranco-rank panel.

Usage:
  python3 scripts_aug/fp_availability_by_category_h.py \\
      --results outputs/unified_mo_full2/results.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import COLORS, FIGURE_DIR, setup_style  # noqa: E402

MIN_WORDS = 500

# Short labels for cramped horizontal axes. Categories not in the map
# render under their original CrUX name.
ABBREV = {
    "Nonprofit & Religion": "Nonprofit",
    "Social & Communication": "Social",
    "Technology": "Tech",
    "Web Infrastructure": "Web Infra",
    "Business & Finance": "Business",
    "Entertainment": "Entert.",
    "E-commerce": "E-comm.",
    "Education": "Edu.",
    "News & Media": "News",
    "Government": "Gov't",
    "Security Risks": "Security",
}


def _load_blacklists(bl_path: pathlib.Path) -> tuple[set[str], set[str]]:
    if not bl_path.exists():
        return set(), set()
    bl = json.load(open(bl_path))
    return set(bl.get("fp_blacklist_etld1", [])), set(bl.get("tp_blacklist_urls", []))


def render(results_path: pathlib.Path, out_dir: pathlib.Path) -> list[pathlib.Path]:
    fp_blacklist, _ = _load_blacklists(results_path.parent / "policy_quality_blacklist.json")

    per_cat_total: collections.Counter[str] = collections.Counter()
    per_cat_qual: collections.Counter[str] = collections.Counter()

    with open(results_path) as fh:
        for ln in fh:
            r = json.loads(ln)
            # Universe: sites whose homepage rendered (successfully visited).
            if r.get("home_ok") is not True:
                continue
            cat = r.get("main_category") or "Uncategorized"
            per_cat_total[cat] += 1
            et = (r.get("site_etld1") or "").lower()
            qualifies = (
                r.get("status") == "ok"
                and r.get("policy_is_english")
                and (r.get("first_party_policy_word_count") or 0) >= MIN_WORDS
                and et not in fp_blacklist
            )
            if qualifies:
                per_cat_qual[cat] += 1

    # Drop categories with too few visited sites to be statistically meaningful.
    cats = [c for c in per_cat_total if per_cat_total[c] >= 5]
    avail_pct = [100 * per_cat_qual[c] / per_cat_total[c] for c in cats]

    # Sort high-to-low so the eye reads left to right by prevalence.
    order = np.argsort(avail_pct)[::-1]
    cats_s = [cats[i] for i in order]
    avail_s = [avail_pct[i] for i in order]
    labels = [ABBREV.get(c, c) for c in cats_s]

    # ACM column width ~3.35in. Vertical bar layout matching the original
    # v2 figure but with abbreviated x-tick labels so the panel slots into
    # the combined first-party-availability figure without crowding.
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    x = np.arange(len(cats_s))
    ax.bar(x, avail_s,
           color=COLORS["primary"],
           edgecolor="#0b3d6b",
           linewidth=0.5,
           width=0.72)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("Share with qualifying policy", fontsize=7.5)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(axis="y", labelsize=7, pad=2)
    ax.tick_params(axis="x", pad=1)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[pathlib.Path] = []
    for ext in ("pdf", "png"):
        p = out_dir / f"fp_availability_by_category_h.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.05)
        paths.append(p)
    plt.close(fig)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=pathlib.Path)
    ap.add_argument("--output", type=pathlib.Path, default=None)
    args = ap.parse_args()

    setup_style()
    out = args.output or FIGURE_DIR
    print("[h] FP availability by content category (horizontal, abbreviated)...")
    for p in render(args.results, out):
        print(f"  saved: {p}")


if __name__ == "__main__":
    main()
