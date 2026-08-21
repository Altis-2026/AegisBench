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


def robustness_heatmap(df, out_path: str | Path, metric: str = "recall",
                       dataset: str | None = None) -> Path:
    """Heatmap of relative performance drop: rows = model, cols =
    corruption x severity. df is the master sweep table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .evaluation.robustness import relative_drop

    rpd = relative_drop(df, metric)
    if dataset:
        rpd = rpd[rpd["dataset"] == dataset]
    rpd["col"] = rpd["corruption"] + " s" + rpd["severity"].astype(str)
    pivot = rpd.pivot_table(index="model", columns="col",
                            values=f"rpd_{metric}")

    fig, ax = plt.subplots(
        figsize=(1.5 + 0.45 * len(pivot.columns), 1.5 + 0.6 * len(pivot)))
    im = ax.imshow(pivot.values, cmap="magma_r", vmin=0.0, vmax=1.0,
                   aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if v > 0.5 else "black")
    ax.set_title(f"Relative {metric} drop vs clean"
                 + (f" — {dataset}" if dataset else ""))
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def severity_curve(ci_df, out_path: str | Path, model: str,
                   dataset: str | None = None,
                   metric: str = "recall") -> Path:
    """One line per corruption, metric vs severity (0 = clean), with the
    bootstrap CI shaded. Reads a phase5_bootstrap_ci.py output CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = ci_df[ci_df["model"] == model]
    if dataset:
        df = df[df["dataset"] == dataset]

    clean = df[df["corruption"] == "clean"]
    clean_y = float(clean[f"{metric}_mean"].iloc[0]) if len(clean) else None

    corruptions = sorted(df[df["corruption"] != "clean"]["corruption"].unique())
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, corr in enumerate(corruptions):
        sub = df[df["corruption"] == corr].sort_values("severity")
        lead_x = [0] if clean_y is not None else []
        lead_y = [clean_y] if clean_y is not None else []
        xs = lead_x + sub["severity"].tolist()
        ys = lead_y + sub[f"{metric}_mean"].tolist()
        lo = lead_y + sub[f"{metric}_ci_lo"].tolist()
        hi = lead_y + sub[f"{metric}_ci_hi"].tolist()
        color = cmap(i % 10)
        ax.plot(xs, ys, marker="o", markersize=3, label=corr,
               color=color, linewidth=1.6)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["clean", "s1", "s2", "s3"])
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel(metric.capitalize())
    suffix = model + (f", {dataset}" if dataset else "")
    ax.set_title(f"{metric.capitalize()} vs severity ({suffix})")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5),
             frameon=False)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def localization_decoupling(loc_df, ci_df, out_path: str | Path, model: str,
                            dataset: str | None = None) -> Path:
    """Recall vs. localization stability, one point per (corruption,
    severity), point size = n_common. Shows the two failure modes are
    decoupled: recall collapses toward zero while stability stays high."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loc = loc_df[loc_df["model"] == model]
    ci = ci_df[(ci_df["model"] == model) & (ci_df["corruption"] != "clean")]
    if dataset:
        loc = loc[loc["dataset"] == dataset]
        ci = ci[ci["dataset"] == dataset]

    merged = loc.merge(ci[["corruption", "severity", "recall_mean"]],
                       on=["corruption", "severity"], how="inner")
    merged = merged[merged["loc_stability_iou"].notna()]

    families = sorted(merged["family"].unique())
    cmap = plt.get_cmap("Set2")
    fam_color = {f: cmap(i % 8) for i, f in enumerate(families)}

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for fam in families:
        sub = merged[merged["family"] == fam]
        ax.scatter(sub["recall_mean"], sub["loc_stability_iou"],
                   s=sub["n_common"].clip(lower=15) / 4 + 15,
                   color=fam_color[fam], label=fam, alpha=0.8,
                   edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Recall (detection survives)")
    ax.set_ylabel("Localization stability (IoU, clean vs. corrupted box)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.5, 1.02)
    suffix = model + (f", {dataset}" if dataset else "")
    ax.set_title(f"Recall collapses; localization does not ({suffix})")
    ax.legend(fontsize=8, frameon=False, title="Family", title_fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
