"""Publication styling for the paper figures.

Matplotlib defaults are built for on-screen viewing and look wrong in a
two-column CVF paper: sans-serif type against a serif body, tick labels
that vanish once the figure is scaled into a column, and raster output
that goes soft at print resolution. This module fixes all three.

Three rules the figure code follows, all of which matter more than they
look:

1. **Size the figure to its final width and never scale it in LaTeX.**
   `\\includegraphics[width=\\columnwidth]{...}` on an oversized figure
   shrinks the text with it, which is the single most common reason
   figure labels end up unreadable in a submission. Build at COL_WIDTH or
   FULL_WIDTH and place at scale 1.
2. **Vector, not raster.** PDF keeps type sharp and selectable at any
   zoom. Reviewers do zoom.
3. **No title inside the figure.** In CVF style the caption carries the
   description; an in-figure title duplicates it and wastes vertical
   space that the data could be using.

Colour encodes disaster family rather than individual corruption. Nine
distinguishable colours is beyond what a reader can hold, and family is
the structure the taxonomy actually has, so family sets hue and position
within family sets line style. That also survives grayscale printing.
"""

from __future__ import annotations

# CVF two-column geometry, in inches. \columnwidth = 237.13594pt and
# \textwidth = 496.85625pt in the CVPR/WACV template, at 72.27pt/in.
COL_WIDTH = 3.28
FULL_WIDTH = 6.87

# Family hues: chosen to stay distinguishable for the common forms of
# colour-vision deficiency and to hold their ordering in grayscale.
FAMILY_COLORS = {
    "flood": "#2c7fb8",       # blue
    "wildfire": "#d95f02",    # orange
    "storm": "#5e3c99",       # purple
    "earthquake": "#8c6d31",  # brown
    "clean": "#4d4d4d",       # neutral
}

# Within-family distinction, applied in the corruption order the config
# declares so the assignment is stable across runs.
LINE_STYLES = ["-", "--", ":", "-."]

_SERIF_STACK = ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                "STIXGeneral", "DejaVu Serif"]


def apply() -> None:
    """Install the publication rcParams. Call before creating a figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": _SERIF_STACK,
        "mathtext.fontset": "stix",
        # Sized for a figure placed at COL_WIDTH in a 10pt paper: figure
        # text lands near caption size, which is what reads correctly.
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 9,
        # Hairlines: heavy default spines read as clutter at print size.
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.3,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        # Deliberately NOT "tight": a tight bbox silently grows the output
        # past the requested figsize whenever a legend sits outside the
        # axes, and the resulting figure then gets scaled down by LaTeX,
        # shrinking every label with it. Figures here reserve their own
        # margins instead, so the saved width is the width asked for.
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        # Keep text as text in the PDF rather than outlining it.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def family_style(family: str, index_in_family: int) -> dict:
    """Line style kwargs for one corruption: hue from its family,
    dash pattern from its position within that family."""
    return {
        "color": FAMILY_COLORS.get(family, "#4d4d4d"),
        "linestyle": LINE_STYLES[index_in_family % len(LINE_STYLES)],
    }


def save(fig, out_path, formats=("pdf", "png")) -> list:
    """Write the figure once per requested format, PDF first.

    PDF is what goes into the paper; PNG is for quick visual checks and
    for pasting into a slide or an email without a PDF viewer.
    """
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        p = out_path.with_suffix(f".{fmt}")
        fig.savefig(p, format=fmt)
        written.append(p)
    return written
