"""scripts/plot_style.py

Editorial plot style pack — the matplotlib analog of the DALL-E cover
style suffix. Goal: every figure that ships in a blog post looks like it
was drawn for the same magazine.

Design choices (and why):

* Palette is locked to the SAME three-tone system the cover image uses:
  warm off-white background, deep navy as the primary, electric cobalt
  as the accent. A neutral gray tail gives us four-plus categorical
  series before we have to repeat. Picking different colors for plots
  vs. covers would split the visual identity.

* Spines are minimal. Top + right are off; left + bottom stay but are
  thin and dimmed. The grid is a soft horizontal-only line so the data
  carries the visual weight. Quanta Magazine and Distill.pub both use
  this pattern.

* Typography. Default serif for axis labels and titles (closer to
  scientific magazine style), monospace for ticks. We try DejaVu
  Serif / DejaVu Sans Mono — both are bundled with matplotlib so we
  never depend on system fonts being installed.

* Higher DPI (160) than the matplotlib default (100) — Retina-class
  blog readers won't see fuzzy lines.

This module is self-contained: importing it applies the style. We also
expose `PALETTE` and helper functions so demo code can reach for the
named colors directly.

Note: this module is also INLINED into the demo subprocess wrapper by
scripts/run_demo.py, so it intentionally avoids `from __future__ import
annotations` — that statement must be the first import in a file, and
the wrapper prepends a few setup lines (matplotlib backend selection)
before this content. Type hints that need PEP 604 / PEP 585 syntax are
written as strings to stay 3.10-safe without the future import.
"""
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# ---------------------------------------------------------------------------
# Palette — keep these EXACT hex values in lockstep with
# scripts/generate_cover.py:STYLE_SUFFIX. The editorial identity is one
# rule, applied to two surfaces (cover image and figures).
# ---------------------------------------------------------------------------

PALETTE = {
    "background": "#f5f0e8",   # warm parchment
    "ink":        "#1a2e1a",   # deep forest green — primary line / text
    "accent":     "#d4942a",   # warm amber — call-out series
    "muted":      "#6b8a6b",   # cool sage — secondary series, light text
    "soft":       "#d8d5c8",   # near-background tint — fills, gridlines
    "warm":       "#c87a4f",   # copper accent — used SPARINGLY
    "danger":     "#a1322f",   # earthy red — error/regret bars, threshold
                                # crossings. Also sparing.
}

# Categorical color cycle for line/scatter/bar series. The order matters:
# whatever's first gets the most visual weight, so we lead with navy,
# then cobalt, then muted slate. Soft + warm follow as fallbacks.
CYCLE = [
    PALETTE["ink"],
    PALETTE["accent"],
    PALETTE["muted"],
    PALETTE["warm"],
    PALETTE["danger"],
    PALETTE["soft"],
]


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------


_RCPARAMS = {
    # --- canvas -----------------------------------------------------------
    "figure.facecolor": PALETTE["background"],
    "axes.facecolor":   PALETTE["background"],
    "savefig.facecolor": PALETTE["background"],
    "savefig.edgecolor": PALETTE["background"],
    "savefig.dpi":      200,
    "figure.dpi":       110,
    "figure.figsize":   (7.5, 4.75),  # ~16:10, fits the blog content column
    "figure.constrained_layout.use": True,

    # --- typography -------------------------------------------------------
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif", "Georgia", "Cambria", "Times New Roman"],
    "font.size":        11.5,
    "axes.titlesize":   13.5,
    "axes.titleweight": "regular",
    "axes.titlepad":    14,
    "axes.titlelocation": "left",     # editorial titles align with the y-axis
    "axes.labelsize":   11.5,
    "axes.labelpad":    8,
    "axes.labelcolor":  PALETTE["ink"],
    "xtick.labelsize":  9.5,
    "ytick.labelsize":  9.5,
    "xtick.color":      PALETTE["muted"],
    "ytick.color":      PALETTE["muted"],
    "text.color":       PALETTE["ink"],

    # --- spines + grid ----------------------------------------------------
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  True,
    "axes.spines.bottom": True,
    "axes.edgecolor":   PALETTE["muted"],
    "axes.linewidth":   0.8,
    "axes.grid":        True,
    "axes.grid.axis":   "y",          # horizontal-only grid keeps things calm
    "grid.color":       PALETTE["soft"],
    "grid.linewidth":   0.8,
    "grid.linestyle":   "-",
    "grid.alpha":       0.7,

    # --- ticks ------------------------------------------------------------
    "xtick.direction":  "out",
    "ytick.direction":  "out",
    "xtick.major.size": 4,
    "ytick.major.size": 0,            # no tick marks on y; the grid does it
    "xtick.major.width": 0.8,
    "xtick.minor.visible": False,

    # --- lines + markers --------------------------------------------------
    "lines.linewidth":  1.8,
    "lines.markersize": 5.5,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",

    # --- patches (bars / fills) ------------------------------------------
    "patch.linewidth":  0.0,
    "patch.edgecolor":  PALETTE["background"],

    # --- legend -----------------------------------------------------------
    "legend.frameon":   False,
    "legend.fontsize":  10.5,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.8,
    "legend.borderaxespad": 0.6,

    # --- color cycle ------------------------------------------------------
    "axes.prop_cycle":  cycler(color=CYCLE),

    # --- images / colormaps ----------------------------------------------
    "image.cmap":       "cividis",   # perceptually uniform, navy-friendly
}


def apply_editorial_style():
    """Apply the editorial rcParams. Idempotent — safe to call multiple times.

    Demo code that wants to opt into the style explicitly can call this at
    the top of the file. The wrapper in scripts/run_demo.py also calls it
    automatically before user code runs, so most demos don't need to.
    """
    mpl.rcParams.update(_RCPARAMS)
    # Try to load seaborn ONLY if it's installed — we don't want a missing
    # optional dep to break a demo. When seaborn is present we lean on its
    # context for spacing improvements, but its color palette is overridden
    # by ours via rcParams above.
    try:
        import seaborn as sns  # noqa: WPS433 — optional import on purpose
        sns.set_context("notebook", font_scale=1.0)
        sns.set_palette(CYCLE)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Helpers — Claude's demo code can call these for common editorial moves
# ---------------------------------------------------------------------------


def figure(width=7.2, height=4.5, **kwargs):
    """Open a figure at the editorial default size with one axis ready.

    Equivalent to `fig, ax = plt.subplots(figsize=(w, h))` but with the
    style guaranteed to be applied first. Use this in demos to be sure
    the figure picks up the editorial defaults even if a user setting
    earlier in the demo overrode the rcParams.
    """
    apply_editorial_style()
    fig, ax = plt.subplots(figsize=(width, height), **kwargs)
    return fig, ax


def annotate_callout(ax, xy, text, *, xytext=None, color=None):
    """Drop a small labelled callout onto an axis.

    Editorial illustrations often pair a single highlighted data point
    with a short label. This wraps `ax.annotate` with sensible defaults
    so demo code stays readable.
    """
    if color is None:
        color = PALETTE["accent"]
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
    """Return the n-th color from the editorial cycle.

    Convenience for demo code that sets a single specific color rather
    than letting the cycle handle it (e.g. when shading a fill_between
    region the same color as a line).
    """
    return CYCLE[idx % len(CYCLE)]


def subplots(nrows=1, ncols=1, *, width=7.5, height=None, **kwargs):
    """Open a multi-panel figure at editorial default sizing.

    Use this instead of ``plt.subplots()`` when your demo has multiple
    subplots — it guarantees the editorial style is applied and sets a
    sensible total figure size. When *height* is omitted it scales
    proportionally from the single-panel default.

    Returns ``(fig, axes)`` where *axes* is a single Axes or array of
    Axes matching (nrows, ncols).
    """
    apply_editorial_style()
    if height is None:
        height = width * (4.75 / 7.5) * (nrows / max(ncols, 1))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)
    return fig, axes


def editorial_tick_params(ax, *, bottom=True, left=True):
    """Apply editorial tick defaults to an existing axis.

    Thin outward ticks on the specified sides, matching the style pack's
    conventions. Useful when building a figure from multiple subplots
    where some have shared axes that don't show ticks by default.
    """
    ax.tick_params(
        axis="x", direction="out", length=4, width=0.8,
        bottom=bottom, labelbottom=bottom,
    )
    ax.tick_params(
        axis="y", direction="out", length=4, width=0.8,
        left=left, labelleft=left,
    )


# Apply on import. Demo code that imports this module gets the style
# immediately — no explicit call required. We deliberately do this at
# module level (not under __main__) so the wrapper can drop a single
# `import plot_style` line and be done.
apply_editorial_style()
