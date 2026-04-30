#!/usr/bin/env python3
"""Two-axis line chart: median word count and median FKGL per group of
100 FPs sorted by Tranco rank ascending. Error bars show Q1-Q3.

Reads data/output/fp_length_readability.json, writes
CCS 2026/figures/RQ1/fp_length_readability.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-json", required=True, type=Path)
    ap.add_argument("--out",       required=True, type=Path)
    ap.add_argument("--xlabel",    default="Max Tranco rank per group (groups of 100 FPs)")
    ap.add_argument("--every",     type=int, default=3,
                    help="show every Nth x-tick label (others hidden)")
    ap.add_argument("--fkgl-ymin", type=float, default=10.0)
    ap.add_argument("--fkgl-ymax", type=float, default=16.0)
    ap.add_argument("--wc-ymin", type=float, default=None)
    ap.add_argument("--wc-ymax", type=float, default=None)
    ap.add_argument("--legend-outside", action="store_true",
                    help="place the legend above the plot instead of inside")
    ap.add_argument("--smooth-window", type=int, default=1,
                    help="centered moving-average window (1 = no smoothing)")
    ap.add_argument("--width",  type=float, default=3.5)
    ap.add_argument("--height", type=float, default=2.6)
    ap.add_argument("--color-wc",  default="#5e3c99")
    ap.add_argument("--color-fk",  default="#e66101")
    ap.add_argument("--marker-wc", default="^")
    ap.add_argument("--marker-fk", default="x")
    ap.add_argument("--ms",        type=float, default=3.5)
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family":   "sans-serif",
        "font.size":     8,
        "axes.spines.top":   False,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon":  True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "#bbb",
        "pdf.fonttype":    42,
        "ps.fonttype":     42,
    })

    d = json.loads(args.data_json.read_text())
    bks = d["buckets"]
    labels = [b["label"] for b in bks]
    x = np.arange(len(bks))

    # Use the central-10% band (P45-P55) for the error bars: gentler
    # than Q1-Q3, no min/max extremes.
    wc_med = np.array([b["wc_med"]    for b in bks])
    wc_lo  = np.array([b["wc_p45"]    for b in bks])
    wc_hi  = np.array([b["wc_p55"]    for b in bks])
    fk_med = np.array([b["fkgl_med"]  for b in bks])
    fk_lo  = np.array([b["fkgl_p45"]  for b in bks])
    fk_hi  = np.array([b["fkgl_p55"]  for b in bks])

    def _smooth(y: np.ndarray, w: int) -> np.ndarray:
        """Centered moving average with edge reflection."""
        if w <= 1 or len(y) < 3:
            return y
        pad = w // 2
        y_pad = np.concatenate([y[:pad][::-1], y, y[-pad:][::-1]])
        kernel = np.ones(w) / w
        return np.convolve(y_pad, kernel, mode="valid")

    sw = max(1, args.smooth_window)
    wc_med = _smooth(wc_med, sw)
    wc_lo  = _smooth(wc_lo,  sw)
    wc_hi  = _smooth(wc_hi,  sw)
    fk_med = _smooth(fk_med, sw)
    fk_lo  = _smooth(fk_lo,  sw)
    fk_hi  = _smooth(fk_hi,  sw)

    # Render at the size the paper will actually show, so labels stay
    # legible after \linewidth or \textwidth scaling.
    fig, ax1 = plt.subplots(figsize=(args.width, args.height))
    ax2 = ax1.twinx()

    c_wc = args.color_wc
    c_fk = args.color_fk

    ax1.errorbar(
        x, wc_med, yerr=[wc_med - wc_lo, wc_hi - wc_med],
        fmt="-", marker=args.marker_wc, color=c_wc, ecolor=c_wc,
        capsize=1.6, elinewidth=0.6, lw=1.0, ms=args.ms,
        markerfacecolor=c_wc, markeredgecolor=c_wc,
        label="Median Word Count",
    )
    ax2.errorbar(
        x, fk_med, yerr=[fk_med - fk_lo, fk_hi - fk_med],
        fmt="-", marker=args.marker_fk, color=c_fk, ecolor=c_fk,
        capsize=1.6, elinewidth=0.6, lw=1.0, ms=args.ms,
        markerfacecolor=c_fk, markeredgecolor=c_fk,
        label="Median FKGL Readability Score",
    )

    # Show every Nth label so they don't crowd; keep all data points
    every = max(1, args.every)
    keep_idx = list(range(0, len(labels), every))
    if (len(labels) - 1) not in keep_idx:
        keep_idx.append(len(labels) - 1)
    ax1.set_xticks([x[i] for i in keep_idx])
    ax1.set_xticklabels([labels[i] for i in keep_idx], rotation=90)
    ax1.set_xlabel(args.xlabel)
    ax1.set_ylabel("Median word count")
    ax2.set_ylabel("Median FKGL readability score")
    ax2.set_ylim(args.fkgl_ymin, args.fkgl_ymax)
    if args.fkgl_ymax - args.fkgl_ymin == 4:
        ax2.set_yticks([int(args.fkgl_ymin) + i for i in range(5)])
    if args.wc_ymin is not None or args.wc_ymax is not None:
        ax1.set_ylim(args.wc_ymin, args.wc_ymax)
    ax2.spines["top"].set_visible(False)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if args.legend_outside:
        ax1.legend(h1 + h2, l1 + l2, loc="lower center",
                   bbox_to_anchor=(0.5, 1.02), ncol=2,
                   handlelength=1.4, columnspacing=1.4, frameon=False)
    else:
        ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    ax1.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    fig.savefig(str(args.out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
