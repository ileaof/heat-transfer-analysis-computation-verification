"""Publication-quality plotting helpers built on Matplotlib.

The Boundary Layer Laboratory produces figures intended for lecture slides and
textbook pages.  This module centralises a consistent visual style (fonts,
line widths, grid, colour cycle) and a few convenience wrappers so that every
module renders figures with the same look.

Nothing here is required to *compute* a boundary layer -- the solvers return
plain NumPy arrays -- but importing :func:`use_publication_style` once makes all
subsequent Matplotlib figures book-ready.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

#: A colour-blind-friendly qualitative palette (Wong, 2011, Nature Methods).
PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)


def use_publication_style() -> None:
    """Apply a clean, high-contrast Matplotlib style suitable for print.

    Sets serif fonts, sensible line/marker sizes, a light grid and the
    colour-blind-friendly :data:`PALETTE`.  Safe to call repeatedly.
    """
    mpl.rcParams.update(
        {
            "figure.figsize": (7.0, 5.0),
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "axes.linewidth": 1.1,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.fontsize": 11,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "mathtext.fontset": "cm",
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
        }
    )


def new_figure(
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    figsize: Optional[tuple[float, float]] = None,
):
    """Create a single-axes figure with labels already applied.

    Parameters
    ----------
    xlabel, ylabel, title : str
        Axis labels and title (Matplotlib mathtext accepted).
    figsize : tuple of float, optional
        Figure size in inches; defaults to the style's ``figure.figsize``.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The new figure and its axes.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax


def savefig(fig, path: str, formats: Sequence[str] = ("png",)) -> list[str]:
    """Save a figure to one or more formats and return the written paths.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str
        Base path *without* extension (e.g. ``"figures/blasius"``).
    formats : sequence of str
        File extensions to emit, e.g. ``("png", "pdf")``.

    Returns
    -------
    list of str
        The full paths that were written.
    """
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    written = []
    for ext in formats:
        full = f"{path}.{ext}"
        fig.savefig(full)
        written.append(full)
    return written


def annotate_value(ax, x: float, y: float, text: str, **kwargs) -> None:
    """Place a small boxed annotation at ``(x, y)`` in axes-fraction coords.

    A thin convenience wrapper around :meth:`Axes.annotate` with a consistent
    rounded box, used to label reference values on the plots.
    """
    ax.annotate(
        text,
        xy=(x, y),
        xycoords="axes fraction",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9),
        **kwargs,
    )


__all__ = [
    "PALETTE",
    "use_publication_style",
    "new_figure",
    "savefig",
    "annotate_value",
]
