"""Visual verification artifacts: bbox overlays, corruption grids, sweep
heatmaps. These are the STOP-and-inspect checkpoints between phases."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

GREEN = (0, 220, 60)
RED = (0, 40, 230)      # BGR
YELLOW = (0, 210, 230)


def draw_boxes(img_rgb: np.ndarray, boxes: np.ndarray,
               color=GREEN, thickness: int | None = None,
               labels: list[str] | None = None) -> np.ndarray:
    """Returns a BGR image (ready for cv2.imwrite) with boxes drawn."""
    out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
    t = thickness or max(2, max(img_rgb.shape[:2]) // 1000)
    for i, (x1, y1, x2, y2) in enumerate(
            np.asarray(boxes, np.float32).reshape(-1, 4)):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, t)
        if labels:
            cv2.putText(out, labels[i], (int(x1), max(int(y1) - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, color, t)
    return out


def corruption_grid(clean_rgb: np.ndarray, suite, image_id: str,
                    out_path: str | Path, cell_width: int = 480) -> Path:
    """Rows = corruptions (top row = clean), columns = severities 1..3.
    The Phase 3 checkpoint artifact."""
    h, w = clean_rgb.shape[:2]
    cell_h = int(round(cell_width * h / w))

    def cell(img, label):
        small = cv2.resize(img, (cell_width, cell_h),
                           interpolation=cv2.INTER_AREA)
        small = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        cv2.rectangle(small, (0, 0), (cell_width, 26), (0, 0, 0), -1)
        cv2.putText(small, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return small

    blank = np.full((cell_h, cell_width, 3), 24, np.uint8)
    rows = [np.hstack([cell(clean_rgb, "clean"), blank, blank])]
    for name in suite.names():
        cells = [cell(suite.apply(clean_rgb, name, s, image_id),
                      f"{name} s{s}") for s in (1, 2, 3)]
        rows.append(np.hstack(cells))
    grid = np.vstack(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    return out_path


# ---------------------------------------------------------------- paper
# Everything below produces figures for the paper itself, as opposed to
# the verification artifacts above. See paperstyle.py for the sizing,
# typography, and colour rules these follow.

def _corruption_styles(df):
    """Stable (colour, linestyle) per corruption: hue by disaster family,
    dash pattern by position within the family. Sorted so the assignment
    does not shift between runs or between datasets."""
    from .paperstyle import family_style

    sub = (df[df["corruption"] != "clean"][["corruption", "family"]]
           .drop_duplicates().sort_values(["family", "corruption"]))
    styles, counts = {}, {}
    for _, r in sub.iterrows():
        fam = r["family"]
        i = counts.get(fam, 0)
        counts[fam] = i + 1
        styles[r["corruption"]] = family_style(fam, i)
    return styles


def severity_curves(ci_df, out_path, panels, metric="recall",
                    width=None, height=2.45, formats=("pdf", "png")):
    """Metric against severity, one line per corruption, bootstrap CI
    shaded. `panels` is a list of (model, dataset) drawn side by side on a
    shared y axis.

    Severity 0 on the x axis is the clean condition, so every line starts
    from the same point and the figure reads as divergence from a common
    baseline rather than four disconnected measurements.

    Laid out at full text width with the legend in reserved space on the
    right. Nine entries below the axes would cost more vertical space than
    the data uses, and stacking them in a right-hand column keeps the
    saved figure exactly `width` inches so it needs no scaling in LaTeX.
    """
    from . import paperstyle
    paperstyle.apply()
    import matplotlib.pyplot as plt

    width = width or paperstyle.FULL_WIDTH
    styles = _corruption_styles(ci_df)

    fig, axes = plt.subplots(1, len(panels), figsize=(width, height),
                             sharey=True, squeeze=False)
    axes = axes[0]

    handles, labels = [], []
    for ax, (model, dataset) in zip(axes, panels):
        d = ci_df[(ci_df["model"] == model) & (ci_df["dataset"] == dataset)]
        clean = d[d["corruption"] == "clean"]
        clean_y = float(clean[f"{metric}_mean"].iloc[0]) if len(clean) else None

        for corr in sorted(styles):
            sub = d[d["corruption"] == corr].sort_values("severity")
            if sub.empty:
                continue
            lead_x = [0] if clean_y is not None else []
            lead_y = [clean_y] if clean_y is not None else []
            xs = lead_x + sub["severity"].tolist()
            ys = lead_y + sub[f"{metric}_mean"].tolist()
            lo = lead_y + sub[f"{metric}_ci_lo"].tolist()
            hi = lead_y + sub[f"{metric}_ci_hi"].tolist()
            st = styles[corr]
            line, = ax.plot(xs, ys, marker="o", markersize=2.2,
                            markeredgewidth=0, **st)
            ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.13,
                            linewidth=0)
            if corr not in labels:
                handles.append(line)
                labels.append(corr)

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["clean", "1", "2", "3"])
        ax.set_xlabel("Severity")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.25, linewidth=0.4)
        ax.set_axisbelow(True)
        # Panel identity goes in a light inset label, not a title: the
        # caption names the figure, this only says which panel is which.
        ax.text(0.97, 0.96, f"{model}, {dataset}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="#333333")

    axes[0].set_ylabel(metric.capitalize())

    order = sorted(range(len(labels)), key=lambda i: labels[i])
    fig.legend([handles[i] for i in order],
               [labels[i].replace("_", " ") for i in order],
               loc="center left", bbox_to_anchor=(0.795, 0.52),
               ncol=1, fontsize=7, borderaxespad=0.0)
    # Reserved margins rather than tight_layout: the legend lives in the
    # right-hand strip, and the saved size must stay exactly `width`.
    fig.subplots_adjust(left=0.075, right=0.78, top=0.97, bottom=0.175,
                        wspace=0.09)
    return paperstyle.save(fig, out_path, formats)


def robustness_heatmap_panels(df, out_path, models=None, dataset=None,
                              metric="recall", formats=("pdf", "png")):
    """Relative drop as corruption (rows) by severity (columns), one panel
    per model.

    Deliberately not the 3-by-27 layout the verification heatmap uses: at
    column width those cells are far too narrow to annotate, and reading
    down a corruption is what the paper actually asks the reader to do.
    """
    from . import paperstyle
    from .evaluation.robustness import relative_drop
    paperstyle.apply()
    import matplotlib.pyplot as plt
    import numpy as np

    rpd = relative_drop(df, metric)
    if dataset:
        rpd = rpd[rpd["dataset"] == dataset]
    models = models or sorted(rpd["model"].unique())

    order = (rpd[["corruption", "family"]].drop_duplicates()
             .sort_values(["family", "corruption"])["corruption"].tolist())

    fig, axes = plt.subplots(
        1, len(models), figsize=(paperstyle.FULL_WIDTH, 2.9),
        squeeze=False)
    axes = axes[0]

    im = None
    for ax, model in zip(axes, models):
        d = rpd[rpd["model"] == model]
        grid = np.full((len(order), 3), np.nan)
        for i, corr in enumerate(order):
            for j, sev in enumerate((1, 2, 3)):
                cell = d[(d["corruption"] == corr) & (d["severity"] == sev)]
                if len(cell):
                    grid[i, j] = float(cell[f"rpd_{metric}"].iloc[0])

        im = ax.imshow(grid, cmap="magma_r", vmin=0.0, vmax=1.0,
                       aspect="auto")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["1", "2", "3"])
        ax.set_xlabel("Severity")
        ax.set_title(model, fontsize=8, pad=4)
        for i in range(len(order)):
            for j in range(3):
                v = grid[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if v > 0.55 else "#1a1a1a")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([c.replace("_", " ") for c in order], fontsize=7)
    for ax in axes[1:]:
        ax.set_yticks([])

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label(f"Relative {metric} drop", fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.outline.set_visible(False)
    return paperstyle.save(fig, out_path, formats)


def localization_decoupling(loc_df, ci_df, out_path, model, dataset=None,
                            width=None, formats=("pdf", "png")):
    """Recall against localization stability, one point per (corruption,
    severity), point area proportional to the number of instances detected
    in both conditions.

    Both axes are drawn on the same 0-to-1 range on purpose. The two
    quantities do fall together, so a zoomed stability axis would show a
    tidy positive correlation and imply the two failure modes degrade
    alike. They do not degrade alike, and the equal scaling is what makes
    that legible: recall sweeps almost the entire range while stability
    stays banded near the top, so severe corruption removes detections far
    faster than it displaces the ones that remain.
    """
    from . import paperstyle
    from .paperstyle import FAMILY_COLORS
    paperstyle.apply()
    import matplotlib.pyplot as plt

    loc = loc_df[loc_df["model"] == model]
    ci = ci_df[(ci_df["model"] == model) & (ci_df["corruption"] != "clean")]
    if dataset:
        loc = loc[loc["dataset"] == dataset]
        ci = ci[ci["dataset"] == dataset]

    merged = loc.merge(ci[["corruption", "severity", "recall_mean"]],
                       on=["corruption", "severity"], how="inner")
    merged = merged[merged["loc_stability_iou"].notna()]

    fig, ax = plt.subplots(figsize=(width or paperstyle.COL_WIDTH, 2.9))

    floor = float(merged["loc_stability_iou"].min())
    ax.axhspan(floor, 1.0, color="#000000", alpha=0.045, linewidth=0)
    ax.axhline(floor, color="#555555", linewidth=0.5, linestyle=(0, (3, 3)))
    ax.text(0.03, floor - 0.035, f"stability floor {floor:.2f}",
            fontsize=6.2, color="#444444", va="top")

    for fam in sorted(merged["family"].unique()):
        sub = merged[merged["family"] == fam]
        ax.scatter(sub["recall_mean"], sub["loc_stability_iou"],
                   s=sub["n_common"].clip(lower=20) / 14 + 7,
                   color=FAMILY_COLORS.get(fam, "#4d4d4d"), label=fam,
                   alpha=0.85, edgecolors="white", linewidths=0.4)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Localization stability (IoU)")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, loc="lower right", handletextpad=0.3,
              borderpad=0.3, labelspacing=0.25)
    fig.subplots_adjust(left=0.165, right=0.985, top=0.985, bottom=0.135)
    return paperstyle.save(fig, out_path, formats)


def corruption_panel(clean_rgb, suite, image_id, out_path, severity=3,
                     ncols=5, crop=None, formats=("pdf", "png")):
    """The taxonomy as one wide panel: the clean frame followed by all nine
    corruptions at a single severity.

    Distinct from `corruption_grid`, which is the tall Phase 3 verification
    artifact showing every severity. Ten cells at three severities is a
    full page of height; the paper needs a figure that sits across two
    columns, so this shows one severity and lets the caption name it.

    `crop` is (x1, y1, x2, y2) in pixels. Aerial frames are mostly terrain,
    and a crop around an annotated region shows the corruption at the scale
    that actually matters for detection rather than as a thumbnail.
    """
    from . import paperstyle
    paperstyle.apply()
    import matplotlib.pyplot as plt

    if crop is not None:
        x1, y1, x2, y2 = (int(v) for v in crop)
        clean_rgb = clean_rgb[y1:y2, x1:x2]

    names = list(suite.names())
    cells = [("clean", clean_rgb)]
    for n in names:
        cells.append((n, suite.apply(clean_rgb, n, severity, image_id)))

    nrows = int(np.ceil(len(cells) / ncols))
    h, w = clean_rgb.shape[:2]
    cell_w = paperstyle.FULL_WIDTH / ncols
    cell_h = cell_w * h / w
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(paperstyle.FULL_WIDTH,
                                      cell_h * nrows + 0.12 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, (label, img) in zip(axes, cells):
        ax.imshow(np.clip(img, 0, 1) if img.dtype != np.uint8 else img)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(label.replace("_", " "), fontsize=7, pad=1.5)
    for ax in axes[len(cells):]:
        ax.set_visible(False)

    fig.subplots_adjust(left=0.002, right=0.998, top=0.94, bottom=0.006,
                        wspace=0.03, hspace=0.16)
    return paperstyle.save(fig, out_path, formats)
