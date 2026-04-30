#!/usr/bin/env python3
"""Combined first-party availability figure (full-width, two panels).

Renders the Tranco-rank panel and the content-category panel into a single
matplotlib figure with two side-by-side subplots. Each panel preserves the
proportions of its single-panel sibling (about 3.35in wide, 2.5in tall axes
area), so the combined PDF inserts at the paper's full \\textwidth without
crushing either panel. The two axes share identical heights via
constrained_layout so they line up cleanly.

Output:
  scripts_aug/output/fp_availability_combined.{pdf,png}

Usage:
  python3 scripts_aug/fp_availability_combined.py \\
      --results outputs/unified_mo_full2/results.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import COLORS, FIGURE_DIR, setup_style  # noqa: E402

MIN_WORDS = 500
WORD_RE = re.compile(r"\S+")


def _load_blacklists(bl_path: pathlib.Path) -> tuple[set[str], set[str]]:
    if not bl_path.exists():
        return set(), set()
    bl = json.load(open(bl_path))
    return set(bl.get("fp_blacklist_etld1", [])), set(bl.get("tp_blacklist_urls", []))


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def _scan_results(results_path: pathlib.Path):
    fp_blacklist, _ = _load_blacklists(results_path.parent / "policy_quality_blacklist.json")

    per_cat_total: collections.Counter[str] = collections.Counter()
    per_cat_qual: collections.Counter[str] = collections.Counter()
    rank_records: list[tuple[int, int]] = []

    with open(results_path) as fh:
        for ln in fh:
            r = json.loads(ln)
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

            rank = r.get("rank")
            if isinstance(rank, int):
                rank_records.append((rank, 1 if qualifies else 0))

    return per_cat_total, per_cat_qual, rank_records


def _category_data(per_cat_total: collections.Counter, per_cat_qual: collections.Counter):
    cats = sorted(per_cat_total.keys(), key=lambda c: per_cat_total[c], reverse=True)
    cats = [c for c in cats if per_cat_total[c] >= 5]
    avail_pct = [100 * per_cat_qual[c] / per_cat_total[c] for c in cats]
    order = np.argsort(avail_pct)[::-1]
    return [cats[i] for i in order], [avail_pct[i] for i in order]


def _rank_data(rank_records: list[tuple[int, int]]):
    rank_records.sort(key=lambda x: x[0])
    ranks = np.array([r for r, _ in rank_records])
    qual = np.array([q for _, q in rank_records])

    buckets = [
        (1, 100),
        (101, 500),
        (501, 1000),
        (1001, 2500),
        (2501, 5000),
        (5001, 10000),
        (10001, int(ranks.max()) if len(ranks) else 16100),
    ]
    bucket_labels: list[str] = []
    avail_pct: list[float] = []
    for lo, hi in buckets:
        mask = (ranks >= lo) & (ranks <= hi)
        tot = int(mask.sum())
        if tot == 0:
            continue
        cov = int(qual[mask].sum())
        bucket_labels.append(f"{lo:,}–{hi:,}")
        avail_pct.append(100 * cov / tot)
    return bucket_labels, avail_pct


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
def render_combined(results_path: pathlib.Path, out_dir: pathlib.Path) -> list[pathlib.Path]:
    per_cat_total, per_cat_qual, rank_records = _scan_results(results_path)
    cats, cat_avail = _category_data(per_cat_total, per_cat_qual)
    rank_labels, rank_avail = _rank_data(rank_records)

    # Each single-panel sibling was rendered at figsize ~ (3.35, 2.5).
    # Panel (b) has 16 rotated category labels and needs more horizontal
    # room than panel (a) to avoid collisions, so the width_ratios favour
    # (b). figsize is the full ACM \textwidth (~7.3 in) and a height that
    # leaves room for rotated labels and the panel-letter titles.
    # constrained_layout keeps the two axes at the same height despite
    # different x/y label heights.
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.6, 3.7),
        gridspec_kw={"width_ratios": [0.78, 1.22], "wspace": 0.30},
        constrained_layout=True,
    )

    # ----- Panel (a): Tranco rank — horizontal stacked bars -----
    y = np.arange(len(rank_labels))[::-1]
    covered = np.array(rank_avail)
    uncovered = 100 - covered
    bar_h = 0.7
    ax_a.barh(y, covered, height=bar_h,
              color=COLORS["primary"], edgecolor="white", linewidth=0.5)
    ax_a.barh(y, uncovered, left=covered, height=bar_h,
              color="#d9d9d9", edgecolor="white", linewidth=0.5)
    for yi, pct in zip(y, rank_avail):
        if pct >= 18:
            ax_a.text(pct / 2, yi, f"{pct:.0f}%",
                      ha="center", va="center", color="white",
                      fontsize=7.5, fontweight="bold")
        else:
            ax_a.text(pct + 1.5, yi, f"{pct:.0f}%",
                      ha="left", va="center", color="#333333",
                      fontsize=7.5, fontweight="bold")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(rank_labels, fontsize=8)
    ax_a.set_xlim(0, 102)
    ax_a.set_xticks([0, 25, 50, 75, 100])
    ax_a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%" if v <= 100 else ""))
    ax_a.set_ylabel("Tranco rank", fontsize=9)
    ax_a.tick_params(axis="x", labelsize=8, pad=2)
    ax_a.tick_params(axis="y", pad=2)
    ax_a.grid(True, axis="x", alpha=0.25, linestyle=":", linewidth=0.5)
    ax_a.set_axisbelow(True)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.set_title("(a) Availability by Tranco rank",
                   fontsize=9, loc="left", pad=4)

    # ----- Panel (b): Content category — vertical bars -----
    x = np.arange(len(cats))
    ax_b.bar(x, cat_avail,
             color=COLORS["primary"], edgecolor="#0b3d6b",
             linewidth=0.5, width=0.72)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(cats, fontsize=7, rotation=45, ha="right",
                         rotation_mode="anchor")
    ax_b.set_ylabel("Share with qualifying policy", fontsize=9)
    ax_b.set_ylim(0, 100)
    ax_b.set_yticks([0, 25, 50, 75, 100])
    ax_b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_b.tick_params(axis="y", labelsize=8, pad=2)
    ax_b.tick_params(axis="x", pad=1)
    ax_b.grid(True, axis="y", alpha=0.25, linestyle=":", linewidth=0.5)
    ax_b.set_axisbelow(True)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.set_title("(b) Availability by content category",
                   fontsize=9, loc="left", pad=4)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[pathlib.Path] = []
    for ext in ("pdf", "png"):
        p = out_dir / f"fp_availability_combined.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.05)
        paths.append(p)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=pathlib.Path)
    ap.add_argument("--output", type=pathlib.Path, default=None)
    args = ap.parse_args()

    setup_style()
    out = args.output or FIGURE_DIR
    print("[combined] FP availability (Tranco rank + content category)...")
    for p in render_combined(args.results, out):
        print(f"  saved: {p}")


if __name__ == "__main__":
    main()
