#!/usr/bin/env python3
"""Two availability figures from the post-augmentation dataset:

  fig1: first-party policy availability per CrUX content category
        (share of sites in each category with an EN ≥500w policy).

  fig2: third-party policy availability as a function of prevalence rank
        (rolling coverage rate over ranked TP eTLD+1s, showing head-vs-tail
        availability asymmetry).

Usage:
  python3 scripts/figures/availability_figures.py \
      --results outputs/unified_mo_full2/results.jsonl
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np

from style import COLORS, FIGURE_DIR, setup_style

MIN_WORDS = 500
WORD_RE = re.compile(r"\S+")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _wc(entry) -> int | None:
    if not isinstance(entry, dict):
        return None
    if entry.get("language", "en") != "en":
        return None
    wc = entry.get("word_count")
    if isinstance(wc, int):
        return wc
    return len(WORD_RE.findall(entry.get("text") or ""))


def _load_tp_cache(results_dir: pathlib.Path) -> dict:
    cache: dict = {}
    for f in glob.glob(str(results_dir / "results_shard*.tp_cache.json")):
        try:
            cache.update(json.load(open(f)))
        except (OSError, json.JSONDecodeError):
            continue
    mono = results_dir / "results.tp_cache.json"
    if mono.exists():
        try:
            cache.update(json.load(open(mono)))
        except (OSError, json.JSONDecodeError):
            pass
    return cache


def _load_blacklists(bl_path: pathlib.Path) -> tuple[set[str], set[str]]:
    if not bl_path.exists():
        return set(), set()
    bl = json.load(open(bl_path))
    return set(bl.get("fp_blacklist_etld1", [])), set(bl.get("tp_blacklist_urls", []))


# ---------------------------------------------------------------------------
# fig1: FP availability per category
# ---------------------------------------------------------------------------
def fig1_fp_by_category(results_path: pathlib.Path, out_dir: pathlib.Path) -> list[pathlib.Path]:
    fp_blacklist, _ = _load_blacklists(results_path.parent / "policy_quality_blacklist.json")

    per_cat_total = collections.Counter()  # home_ok=True sites per category
    per_cat_qual = collections.Counter()
    per_cat_no_cat = 0

    with open(results_path) as fh:
        for ln in fh:
            r = json.loads(ln)
            if r.get("home_ok") is not True:
                continue
            cat = r.get("main_category") or "Uncategorized"
            per_cat_total[cat] += 1
            if r.get("status") != "ok":
                continue
            if not r.get("policy_is_english"):
                continue
            w = r.get("first_party_policy_word_count") or 0
            if w < MIN_WORDS:
                continue
            et = (r.get("site_etld1") or "").lower()
            if et in fp_blacklist:
                continue
            per_cat_qual[cat] += 1

    # Drop "Uncategorized" from the visible list if small, keep the rest
    cats = sorted(per_cat_total.keys(),
                  key=lambda c: per_cat_total[c], reverse=True)
    # Filter to CrUX categories (hide obviously empty or uncategorized)
    cats = [c for c in cats if per_cat_total[c] >= 5]

    avail_pct = [100 * per_cat_qual[c] / per_cat_total[c] for c in cats]
    totals = [per_cat_total[c] for c in cats]
    quals = [per_cat_qual[c] for c in cats]

    # Sort by availability for cleaner horizontal bar chart
    order = np.argsort(avail_pct)
    cats_s = [cats[i] for i in order]
    avail_s = [avail_pct[i] for i in order]
    totals_s = [totals[i] for i in order]
    quals_s = [quals[i] for i in order]

    # ACM single-column width ~3.33"; compact height for 16 rows.
    fig, ax = plt.subplots(figsize=(3.35, 3.6))
    y = np.arange(len(cats_s))
    bars = ax.barh(y, avail_s,
                   color=COLORS["primary"],
                   edgecolor="#0b3d6b",
                   linewidth=0.5,
                   height=0.68)

    for bar, pct in zip(bars, avail_s):
        ax.text(bar.get_width() + 1.0,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.0f}\\%" if plt.rcParams.get("text.usetex") else f"{pct:.0f}%",
                va="center", ha="left", fontsize=6.5)

    ax.set_yticks(y)
    ax.set_yticklabels(cats_s, fontsize=7)
    ax.set_xlabel("Share with qualifying policy", fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.tick_params(axis="x", labelsize=7, pad=2)
    ax.tick_params(axis="y", pad=2)
    ax.grid(True, axis="x", alpha=0.25, linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.3)
    paths: list[pathlib.Path] = []
    for ext in ("pdf", "png"):
        p = out_dir / f"fp_availability_by_category.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.05)
        paths.append(p)
    plt.close(fig)

    # -------- v2: vertical bars (x=category, y=availability) ---------------
    # Sort highest-to-lowest so the eye scans left-to-right by prevalence.
    order_v = np.argsort(avail_pct)[::-1]
    cats_v = [cats[i] for i in order_v]
    avail_v = [avail_pct[i] for i in order_v]

    fig2v, axv = plt.subplots(figsize=(3.35, 2.6))
    x = np.arange(len(cats_v))
    axv.bar(x, avail_v,
            color=COLORS["primary"],
            edgecolor="#0b3d6b",
            linewidth=0.5,
            width=0.72)

    axv.set_xticks(x)
    axv.set_xticklabels(cats_v, fontsize=6.5, rotation=45, ha="right")
    axv.set_ylabel("Share with qualifying policy", fontsize=7.5)
    axv.set_ylim(0, 100)
    axv.set_yticks([0, 25, 50, 75, 100])
    axv.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    axv.tick_params(axis="y", labelsize=7, pad=2)
    axv.tick_params(axis="x", pad=1)
    axv.grid(True, axis="y", alpha=0.25, linestyle=":", linewidth=0.5)
    axv.set_axisbelow(True)
    axv.spines["top"].set_visible(False)
    axv.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        p = out_dir / f"fp_availability_by_category_v2.{ext}"
        fig2v.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.05)
        paths.append(p)
    plt.close(fig2v)

    return paths


# ---------------------------------------------------------------------------
# fig2: TP availability by prevalence rank
# ---------------------------------------------------------------------------
def fig2_tp_by_prevalence(results_path: pathlib.Path, out_dir: pathlib.Path) -> list[pathlib.Path]:
    results_dir = results_path.parent

    # Use the canonical qualifying-TP list so the figure matches the paper's
    # headline (1,122 / 4,771 = 23.5%). Falls back to the per-observation
    # policy_url heuristic only if the canonical file is missing.
    canon_path = results_dir / "canonical_qualifying.json"
    if canon_path.exists():
        canon = json.load(open(canon_path))
        tp_qualifying = {d.lower() for d in canon.get("tp_qualifying_etld1", [])}
    else:
        tp_qualifying = None

    tp_obs = collections.Counter()
    tp_covered: set[str] = set()
    with open(results_path) as fh:
        for ln in fh:
            r = json.loads(ln)
            if r.get("home_ok") is not True:
                continue
            for tp in r.get("third_parties") or []:
                if not isinstance(tp, dict):
                    continue
                et = (tp.get("third_party_etld1") or "").lower()
                if not et:
                    continue
                tp_obs[et] += 1
                if tp_qualifying is not None and et in tp_qualifying:
                    tp_covered.add(et)

    print(f"  [fig2] TPs observed: {len(tp_obs)}, qualifying: {len(tp_covered)} "
          f"({100*len(tp_covered)/max(1,len(tp_obs)):.1f}%)")

    ranked = sorted(tp_obs.items(), key=lambda x: -x[1])
    n = len(ranked)
    covered_flag = np.array([1 if d in tp_covered else 0 for d, _ in ranked])
    obs_arr = np.array([c for _, c in ranked])
    total_obs = int(obs_arr.sum())

    # Rank buckets — head folded into a single 1-100 bucket as requested.
    buckets = [
        (1, 100),
        (101, 250),
        (251, 500),
        (501, 1000),
        (1001, 2000),
        (2001, n),
    ]
    bucket_labels = []
    avail_pct = []
    covered_n = []
    total_n = []
    for lo, hi in buckets:
        if lo > n:
            continue
        hi = min(hi, n)
        segment = slice(lo - 1, hi)
        cov = int(covered_flag[segment].sum())
        tot = hi - lo + 1
        bucket_labels.append(f"{lo:,}\u2013{hi:,}")
        avail_pct.append(100 * cov / tot)
        covered_n.append(cov)
        total_n.append(tot)

    fig, ax_a = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(bucket_labels))[::-1]
    bar_h = 0.65
    covered_arr = np.array(avail_pct)
    uncovered_arr = 100 - covered_arr
    ax_a.barh(y, covered_arr, height=bar_h,
              color=COLORS["primary"], edgecolor="white", linewidth=0.6,
              label="Policy available")
    ax_a.barh(y, uncovered_arr, left=covered_arr, height=bar_h,
              color="#d9d9d9", edgecolor="white", linewidth=0.6,
              label="Policy not available")

    for yi, pct in zip(y, avail_pct):
        if pct >= 20:
            ax_a.text(pct / 2, yi, f"{pct:.0f}%",
                      ha="center", va="center", color="white",
                      fontsize=9, fontweight="bold")
        else:
            ax_a.text(pct + 1.2, yi, f"{pct:.0f}%",
                      ha="left", va="center", color="#333333",
                      fontsize=8, fontweight="bold")

    ax_a.set_yticks(y)
    ax_a.set_yticklabels(bucket_labels, fontsize=9)
    ax_a.set_xlim(0, 102)
    ax_a.set_xlabel("")
    ax_a.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%" if x <= 100 else ""))
    ax_a.set_xticks([0, 25, 50, 75, 100])
    ax_a.set_ylabel("TP prevalence rank", fontsize=10)
    ax_a.grid(True, axis="x", alpha=0.22, linestyle=":")
    ax_a.set_axisbelow(True)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    # Put legend below the axis so it never collides with bars/labels
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                ncol=2, fontsize=9, frameon=False)

    paths: list[pathlib.Path] = []
    for ext in ("pdf", "png"):
        p = out_dir / f"tp_availability_by_prevalence.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# fig3: FP availability by Tranco rank
# ---------------------------------------------------------------------------
def fig3_fp_by_tranco_rank(results_path: pathlib.Path, out_dir: pathlib.Path) -> list[pathlib.Path]:
    fp_blacklist, _ = _load_blacklists(results_path.parent / "policy_quality_blacklist.json")

    # Collect (rank, qualifies) tuples — restrict universe to home_ok=True sites
    records: list[tuple[int, int]] = []
    with open(results_path) as fh:
        for ln in fh:
            r = json.loads(ln)
            if r.get("home_ok") is not True:
                continue
            rank = r.get("rank")
            if not isinstance(rank, int):
                continue
            et = (r.get("site_etld1") or "").lower()
            qualifies = (
                r.get("status") == "ok"
                and r.get("policy_is_english")
                and (r.get("first_party_policy_word_count") or 0) >= MIN_WORDS
                and et not in fp_blacklist
            )
            records.append((rank, 1 if qualifies else 0))

    records.sort(key=lambda x: x[0])
    ranks = np.array([r for r, _ in records])
    qual = np.array([q for _, q in records])

    # Log-spaced rank buckets (matches the TP prevalence figure's style)
    buckets = [
        (1, 100),
        (101, 500),
        (501, 1000),
        (1001, 2500),
        (2501, 5000),
        (5001, 10000),
        (10001, int(ranks.max()) if len(ranks) else 16100),
    ]

    bucket_labels = []
    avail_pct = []
    cov_n = []
    tot_n = []
    for lo, hi in buckets:
        mask = (ranks >= lo) & (ranks <= hi)
        tot = int(mask.sum())
        if tot == 0:
            continue
        cov = int(qual[mask].sum())
        bucket_labels.append(f"{lo:,}\u2013{hi:,}")
        avail_pct.append(100 * cov / tot)
        cov_n.append(cov)
        tot_n.append(tot)

    # Compact ACM single-column size.
    fig, ax_a = plt.subplots(figsize=(3.35, 2.5))

    y = np.arange(len(bucket_labels))[::-1]
    covered_arr = np.array(avail_pct)
    uncovered_arr = 100 - covered_arr
    bar_h = 0.7
    ax_a.barh(y, covered_arr, height=bar_h,
              color=COLORS["primary"], edgecolor="white", linewidth=0.5,
              label="Policy available")
    ax_a.barh(y, uncovered_arr, left=covered_arr, height=bar_h,
              color="#d9d9d9", edgecolor="white", linewidth=0.5,
              label="Policy not available")
    for yi, pct in zip(y, avail_pct):
        if pct >= 18:
            ax_a.text(pct / 2, yi, f"{pct:.0f}%",
                      ha="center", va="center", color="white",
                      fontsize=6.5, fontweight="bold")
        else:
            ax_a.text(pct + 1.2, yi, f"{pct:.0f}%",
                      ha="left", va="center", color="#333333",
                      fontsize=6.5, fontweight="bold")

    ax_a.set_yticks(y)
    ax_a.set_yticklabels(bucket_labels, fontsize=7)
    ax_a.set_xlim(0, 102)
    ax_a.set_xlabel("")
    ax_a.set_ylabel("Tranco rank", fontsize=7.5)
    ax_a.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%" if x <= 100 else ""))
    ax_a.set_xticks([0, 25, 50, 75, 100])
    ax_a.tick_params(axis="x", labelsize=7, pad=2)
    ax_a.tick_params(axis="y", pad=2)
    ax_a.grid(True, axis="x", alpha=0.22, linestyle=":", linewidth=0.5)
    ax_a.set_axisbelow(True)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                ncol=2, fontsize=6.5, frameon=False,
                handlelength=1.2, handletextpad=0.4, columnspacing=1.0)

    plt.tight_layout(pad=0.3)
    paths: list[pathlib.Path] = []
    for ext in ("pdf", "png"):
        p = out_dir / f"fp_availability_by_tranco_rank.{ext}"
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
    out.mkdir(parents=True, exist_ok=True)

    paths: list[pathlib.Path] = []
    print("[fig1] FP availability by category...")
    paths += fig1_fp_by_category(args.results.resolve(), out.resolve())
    for p in paths:
        print(f"  saved: {p}")

    print("[fig2] TP availability by prevalence rank...")
    paths2 = fig2_tp_by_prevalence(args.results.resolve(), out.resolve())
    for p in paths2:
        print(f"  saved: {p}")

    print("[fig3] FP availability by Tranco rank...")
    paths3 = fig3_fp_by_tranco_rank(args.results.resolve(), out.resolve())
    for p in paths3:
        print(f"  saved: {p}")


if __name__ == "__main__":
    main()
