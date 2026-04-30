#!/usr/bin/env python3
"""Two figures in the v9 length-readability style:
  wordcount_gdpr_v4.pdf : X = log-binned word count, Y = median GDPR coverage
  fkgl_gdpr_v4.pdf      : X = binned FKGL,           Y = median GDPR coverage

Style mirrors fp/tp_length_readability_v9.pdf: connected median-marker
line with whiskers (P45-P55), centred 3-bin smoothing, legend above the
plot. FP series in blue circles, TP series in red diamonds.
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Default paths are relative to where the script is called from; can be
# overridden with --data-json / --out-dir.
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data/raw/wordcount_fkgl_gdpr_v4.json"
DEFAULT_OUT_DIR = ROOT / "notebooks"
DATA = DEFAULT_DATA
OUT_DIR = DEFAULT_OUT_DIR

FP_COLOR  = "#2166ac"
TP_COLOR  = "#b2182b"
FP_MARKER = "o"
TP_MARKER = "D"
MS        = 4.0
SMOOTH_W  = 3


def setup_style() -> None:
    plt.rcParams.update({
        "font.family":     "sans-serif",
        "font.size":       8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon":  False,
        "pdf.fonttype":    42,
        "ps.fonttype":     42,
    })


def _smooth(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(y) < 3:
        return y
    pad = w // 2
    y_pad = np.concatenate([y[:pad][::-1], y, y[-pad:][::-1]])
    kernel = np.ones(w) / w
    return np.convolve(y_pad, kernel, mode="valid")


def bucket_stats(x: np.ndarray, y: np.ndarray, edges: np.ndarray,
                 lo_pct: float = 25.0, hi_pct: float = 75.0, min_n: int = 5):
    centres, meds, lo, hi = [], [], [], []
    labels = []
    for i in range(len(edges) - 1):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if mask.sum() < min_n:
            continue
        ys = y[mask]
        centres.append(0.5 * (edges[i] + edges[i + 1]))
        meds.append(float(np.median(ys)))
        lo.append(float(np.percentile(ys, lo_pct)))
        hi.append(float(np.percentile(ys, hi_pct)))
        e = edges[i]
        labels.append(f"{int(e):,}" if e >= 100 else f"{e:.1f}")
    return (np.array(centres), np.array(meds), np.array(lo), np.array(hi), labels)


def plot_two_series(records, x_key, edges, xlabel, out: Path, log_x: bool,
                    lo_pct: float = 25.0, hi_pct: float = 75.0):
    fp = [r for r in records if r["role"] == "FP" and r.get(x_key) is not None]
    tp = [r for r in records if r["role"] == "TP" and r.get(x_key) is not None]
    fp_x = np.array([r[x_key] for r in fp])
    fp_y = np.array([r["n_gdpr"] for r in fp])
    tp_x = np.array([r[x_key] for r in tp])
    tp_y = np.array([r["n_gdpr"] for r in tp])

    fp_c, fp_m, fp_lo, fp_hi, _ = bucket_stats(fp_x, fp_y, edges, lo_pct, hi_pct)
    tp_c, tp_m, tp_lo, tp_hi, _ = bucket_stats(tp_x, tp_y, edges, lo_pct, hi_pct)

    fp_m  = _smooth(fp_m,  SMOOTH_W)
    fp_lo = _smooth(fp_lo, SMOOTH_W)
    fp_hi = _smooth(fp_hi, SMOOTH_W)
    tp_m  = _smooth(tp_m,  SMOOTH_W)
    tp_lo = _smooth(tp_lo, SMOOTH_W)
    tp_hi = _smooth(tp_hi, SMOOTH_W)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.errorbar(
        fp_c, fp_m, yerr=[fp_m - fp_lo, fp_hi - fp_m],
        fmt="-", marker=FP_MARKER, color=FP_COLOR, ecolor=FP_COLOR,
        capsize=1.6, elinewidth=0.6, lw=1.0, ms=MS,
        markerfacecolor=FP_COLOR, markeredgecolor=FP_COLOR,
        label="FP",
    )
    ax.errorbar(
        tp_c, tp_m, yerr=[tp_m - tp_lo, tp_hi - tp_m],
        fmt="-", marker=TP_MARKER, color=TP_COLOR, ecolor=TP_COLOR,
        capsize=1.6, elinewidth=0.6, lw=1.0, ms=MS,
        markerfacecolor=TP_COLOR, markeredgecolor=TP_COLOR,
        label="TP",
    )

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Median GDPR categories covered (/18)")
    ax.set_ylim(0, 18.5)
    ax.set_yticks(range(0, 19, 3))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=2, handlelength=1.4, columnspacing=1.4, frameon=False)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


import argparse


def main() -> None:
    setup_style()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-json", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir",  type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    global DATA, OUT_DIR
    DATA = args.data_json; OUT_DIR = args.out_dir
    d = json.loads(DATA.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # word count: log-spaced bins from 100 words to ~30k
    wc_edges = np.geomspace(200, 30000, 11)
    plot_two_series(d["records"], "word_count", wc_edges,
                    "policy word count", OUT_DIR / "wordcount_gdpr_v4.pdf",
                    log_x=True, lo_pct=25.0, hi_pct=75.0)
    plot_two_series(d["records"], "word_count", wc_edges,
                    "policy word count", OUT_DIR / "wordcount_gdpr_v4_iqr.pdf",
                    log_x=True, lo_pct=45.0, hi_pct=55.0)

    # FKGL: linear bins from grade 8 to grade 22
    fk_edges = np.linspace(8, 22, 11)
    plot_two_series(d["records"], "fkgl", fk_edges,
                    "FKGL", OUT_DIR / "fkgl_gdpr_v4.pdf",
                    log_x=False, lo_pct=25.0, hi_pct=75.0)
    plot_two_series(d["records"], "fkgl", fk_edges,
                    "FKGL", OUT_DIR / "fkgl_gdpr_v4_iqr.pdf",
                    log_x=False, lo_pct=45.0, hi_pct=55.0)


if __name__ == "__main__":
    main()
