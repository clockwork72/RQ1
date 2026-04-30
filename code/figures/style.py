"""Shared style configuration for publication-quality figures.

Matches IEEE S&P / USENIX / CCS visual standards:
- Serif font (Times New Roman / DejaVu Serif)
- 300 DPI output
- Clean, minimal design with no chartjunk
"""

from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
import os
FIGURE_DIR = pathlib.Path(
    os.environ.get("WPD_FIGURE_DIR")
    or pathlib.Path(__file__).resolve().parent / "output"
)

# ---------------------------------------------------------------------------
# Colormap — standard matplotlib Blues
# ---------------------------------------------------------------------------
BLUE_CMAP = plt.cm.Blues

# ---------------------------------------------------------------------------
# Named colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#2171b5",
    "dark": "#084594",
    "medium": "#4292c6",
    "light": "#c6dbef",
    "lightest": "#deebf7",
    "red": "#d62728",
    "green": "#2ca02c",
    "orange": "#ff7f0e",
    "grey": "#666666",
}

# ---------------------------------------------------------------------------
# Gradient helpers
# ---------------------------------------------------------------------------

def blue_gradient(n: int) -> list[str]:
    """Return *n* hex colours from dark blue to light blue."""
    cmap = BLUE_CMAP
    return [mpl.colors.to_hex(cmap(1.0 - i / max(n - 1, 1))) for i in range(n)]


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

def setup_style() -> None:
    """Apply publication rcParams (serif, 300 DPI, clean spines)."""
    plt.rcParams.update({
        # Font — serif for academic publications
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10,
        # Axes
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#cccccc",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.facecolor": "white",
        # Ticks
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        # Legend
        "legend.fontsize": 9,
        "legend.frameon": False,
        # Figure
        "figure.dpi": 300,
        "figure.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "savefig.facecolor": "white",
    })
