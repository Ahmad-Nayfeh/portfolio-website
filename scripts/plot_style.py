"""
scripts/plot_style.py — Nocturne plot style pack (May 2026 redesign).

Dark canvas with electric teal, amber, and magenta accents. Designed for
the Nocturne website theme.
"""
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

PALETTE = {
    "background": "#080812",
    "surface":    "#12121e",
    "border":     "#1e1e30",
    "ink":        "#e8e8ed",
    "muted":      "#8888a0",
    "teal":       "#00d4aa",
    "amber":      "#ffba08",
    "magenta":    "#ff3cac",
    "danger":     "#ff3366",
}

CYCLE = [
    PALETTE["teal"],
    PALETTE["amber"],
    PALETTE["magenta"],
    PALETTE["muted"],
    PALETTE["danger"],
    PALETTE["ink"],
]

_RCPARAMS = {
    "figure.facecolor": PALETTE["background"],
    "axes.facecolor":   PALETTE["background"],
    "savefig.facecolor": PALETTE["background"],
    "savefig.edgecolor": PALETTE["background"],
    "savefig.dpi":      200,
    "figure.dpi":       110,
    "figure.figsize":   (7.2, 4.5),
    "figure.constrained_layout.use": True,

    "font.family":      "sans-serif",
    "font.sans-serif":  ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.titlepad":    14,
    "axes.titlelocation": "left",
    "axes.labelsize":   11,
    "axes.labelpad":    8,
    "axes.labelcolor":  PALETTE["ink"],
    "xtick.labelsize":  9.5,
    "ytick.labelsize":  9.5,
    "xtick.color":      PALETTE["muted"],
    "ytick.color":      PALETTE["muted"],
    "text.color":       PALETTE["ink"],

    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  True,
    "axes.spines.bottom": True,
    "axes.edgecolor":   PALETTE["border"],
    "axes.linewidth":   0.8,
    "axes.grid":        True,
    "axes.grid.axis":   "y",
    "grid.color":       PALETTE["surface"],
    "grid.linewidth":   0.8,
    "grid.linestyle":   "-",
    "grid.alpha":       0.7,

    "xtick.direction":  "out",
    "ytick.direction":  "out",
    "xtick.major.size": 4,
    "ytick.major.size": 0,
    "xtick.major.width": 0.8,
    "xtick.minor.visible": False,

    "lines.linewidth":  1.8,
    "lines.markersize": 5.5,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",

    "patch.linewidth":  0.0,
    "patch.edgecolor":  PALETTE["background"],

    "legend.frameon":   False,
    "legend.fontsize":  10,
    "legend.handlelength": 1.4,
    "legend.borderaxespad": 0.6,

    "axes.prop_cycle":  cycler(color=CYCLE),

    "image.cmap":       "cividis",
}


def apply_editorial_style():
    mpl.rcParams.update(_RCPARAMS)
    try:
        import seaborn as sns
        sns.set_context("notebook", font_scale=1.0)
        sns.set_palette(CYCLE)
    except ImportError:
        pass


def figure(width=7.2, height=4.5, **kwargs):
    apply_editorial_style()
    fig, ax = plt.subplots(figsize=(width, height), **kwargs)
    return fig, ax


def annotate_callout(ax, xy, text, *, xytext=None, color=None):
    if color is None:
        color = PALETTE["teal"]
    if xytext is None:
        xytext = (xy[0] + 0.05 * abs(xy[0] or 1), xy[1])
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        color=color,
        fontsize=9.5,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=color, lw=0.9, alpha=0.7),
    )


def lead_color(idx=0):
    return CYCLE[idx % len(CYCLE)]


apply_editorial_style()
