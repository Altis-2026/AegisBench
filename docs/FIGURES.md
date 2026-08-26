# Figure plan

Five figures for the main paper, plus supplementary. Every figure is
generated at exactly CVF column width (3.28 in) or full text width
(6.87 in) and saved as vector PDF, so it is placed at scale 1 and never
rescaled by LaTeX. Rescaling is the usual reason figure labels end up
unreadable in a submission: shrinking the graphic shrinks its type with it.

Generation commands are at the bottom. Draft captions are written to stand
alone, since many reviewers read figures and captions before the body.

---

## Figure 1, page 1 teaser. Qualitative failure

**Width:** column (or full width if you have room)
**Source:** `scripts/phase5_gallery.py`, already generated in
`results/sweep/gallery/`
**Shows:** one clean frame beside the same frame under `low_light`, ground
truth in cyan, detections above the frozen threshold in green. The
corrupted panel has the ground-truth box and no detection.

This is the strongest possible opening for this paper. The headline result
is that a detector stops seeing people, and an image of an empty box where
a person is annotated communicates that faster than any table. Prefer
RT-DETR on HERIDAL at severity 1, where recall is already 0.000: showing
total failure at the *mildest* severity is more striking than at the worst.

> **Figure 1.** Aerial person detection under mild low-light corruption.
> Left: a clean HERIDAL frame with ground-truth annotations (cyan) and
> detections retained at the model's frozen operating point (green).
> Right: the same frame under `low_light` at severity 1, the mildest level
> in our benchmark, with identical annotations. Every detection is lost.
> Across both datasets and all three detector architectures, recall under
> low light falls to exactly zero by severity 3.

Candidates already generated, ordered by how cleanly they will read at
print size: `ZRI_0004` (8 ground-truth boxes, 9 detections lost),
`ZRI_0005` (12 boxes, 17 lost), `BLI_0004` (18 boxes, 16 lost). The last is
the most dramatic but the busiest; consider it for supplementary instead.

---

## Figure 2, Benchmark Design. Corruption taxonomy panel

**Width:** full
**Source:** `phase5_figures.py taxonomy`
**Shows:** the clean frame plus all nine corruptions at one severity, in a
2 x 5 grid.

Use a real HERIDAL frame, not a synthetic one, and crop to a region
containing annotated people so the corruption is visible at the scale that
matters for detection rather than as a thumbnail of terrain. Severity 3
shows the taxonomy most clearly; the caption must state which severity is
shown, since showing only one is a deliberate simplification.

> **Figure 2.** The nine corruptions, shown at severity 3 on a cropped
> HERIDAL frame. Each models a documented optical signature of a disaster
> environment rather than a generic distortion: haze-family corruptions
> follow the Koschmieder scattering model with family-specific airlight
> chromaticity (neutral gray for wildfire smoke, brown mineral for
> post-disaster dust), low light is modeled in the linear photometric
> domain with signal-dependent sensor noise, and rain and glare are
> additive radiance effects. Severity is calibrated per corruption against
> a declared image statistic (Table 1).

The full three-severity grid belongs in supplementary. It is ten rows tall
and would consume most of a page here.

---

## Figure 3, Results. Severity curves with confidence intervals

**Width:** full
**Source:** `phase5_figures.py results`
**Shows:** recall against severity, one line per corruption, bootstrap 95%
interval shaded, starting from the shared clean baseline at x = 0.

This is where the spread becomes visible: `rain_streaks` stays nearly flat
while `low_light` falls off a cliff, and the reader sees that a single
"robustness" number would hide the entire finding. Colour encodes disaster
family and dash pattern separates corruptions within a family, so the
figure survives grayscale printing and matches the taxonomy structure.

> **Figure 3.** Recall against corruption severity for [MODEL] on [DATASET],
> with bootstrap 95% confidence intervals shaded (1,000 resamples of the
> test set per condition). All curves originate from the same clean
> baseline at left. Colour indicates disaster family; line style
> distinguishes corruptions within a family. Degradation is highly uneven:
> heavy rain costs roughly [X] points of recall at the highest severity,
> while low light drives recall to exactly zero.

Fill [X] from `ci_*.csv` and keep it consistent with the abstract.

---

## Figure 4, Results. Relative drop heatmap

**Width:** full
**Source:** `phase5_figures.py results --sweep-csv`
**Shows:** relative recall drop as corruption (rows) by severity (columns),
one panel per detector, cells annotated.

Laid out as corruption-by-severity rather than the 3 x 27 grid used for
internal verification: at print width those cells are too narrow to
annotate, and reading down a single corruption across models is what the
paper actually asks of the reader. The `low_light` row goes black in all
three panels, which is the cross-architecture claim in one glance.

> **Figure 4.** Relative recall drop, (clean − corrupted) / clean, for each
> detector on [DATASET]. Rows are grouped by disaster family and ordered to
> match Table 1. Darker cells indicate greater degradation. Measuring drop
> against each model's own clean baseline means a model is not rewarded for
> a weaker starting point. The `low_light` row saturates for all three
> architectures, the cross-architecture collapse examined in Section [X].

---

## Figure 5, Results. Localization stability decoupling

**Width:** column
**Source:** `phase5_figures.py results`
**Shows:** recall against localization stability, one point per
(corruption, severity), point area proportional to the number of instances
detected in both the clean and corrupted conditions.

**Both axes are drawn on the same 0-to-1 range deliberately, and the
caption should not undercut that.** The two quantities do fall together, so
a zoomed stability axis produces a tidy positive correlation and implies
the two failure modes degrade alike. They do not: recall sweeps almost the
entire range while stability stays banded near the top. Equal scaling is
what makes the difference in magnitude legible rather than flattering.

> **Figure 5.** Recall against localization stability for [MODEL] on
> [DATASET], one point per corruption and severity, point area
> proportional to the number of instances detected in both the clean and
> corrupted conditions. Both axes span the full unit range. As corruption
> worsens, points sweep left across the entire recall range while remaining
> within a narrow band above [FLOOR] on the stability axis: surviving
> detections stay close to where the same model placed them on clean
> imagery. The dominant failure mode is a detector ceasing to find a
> person, not a detector finding one and misplacing the box.

Read [FLOOR] off the figure; the script annotates it automatically.

---

## Supplementary figures

- The full three-severity corruption grid (`corruption_grid`, already at
  `results/phase3/heridal_grid.jpg`).
- Severity curves and decoupling scatters for the models not shown in the
  main paper. The script writes one per model automatically.
- The heatmap for the second dataset.
- Additional gallery pairs, including a partial-degradation condition such
  as `fire_warm_tint` at severity 2, which contrasts usefully against the
  total `low_light` collapse and shows the gallery is not cherry-picked for
  the most extreme case.

---

## Generating everything

```bash
# Results figures, per dataset. Writes PDF and PNG per model.
python scripts/phase5_figures.py results \
    --ci results/sweep/ci_heridal.csv \
    --localization results/sweep/localization_heridal.csv \
    --sweep-csv results/sweep/master_ci.csv \
    --dataset heridal --out results/figures

python scripts/phase5_figures.py results \
    --ci results/sweep/ci_sard.csv \
    --localization results/sweep/localization_sard.csv \
    --sweep-csv results/sweep/master_ci.csv \
    --dataset sard --out results/figures

# Taxonomy panel. Use a real frame and crop to an annotated region.
python scripts/phase5_figures.py taxonomy \
    --image <a clean HERIDAL frame> --severity 3 \
    --crop 1200,900,3000,2250 --out results/figures

# Qualitative gallery (already run for two conditions)
python scripts/phase5_gallery.py \
    --records data/heridal/records/test.json --dataset heridal \
    --pred-dir results/sweep/preds --model rtdetr \
    --corruption low_light --severity 1 --conf-thresh 0.65 \
    --out results/sweep/gallery --n 4
```

## Placing them in LaTeX

Place at scale 1. The figures are already the right width.

```latex
% full-width figure, spans both columns
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/severity_sard_yolo11.pdf}
  \caption{...}
  \label{fig:severity}
\end{figure*}

% single-column figure
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/decoupling_sard_yolo11.pdf}
  \caption{...}
  \label{fig:decoupling}
\end{figure}
```

Use the `.pdf` files, not the `.png`. The PNGs exist for quick viewing and
for pasting into email or slides.

## Before submitting, check each figure

1. Print the paper at 100% and confirm every axis label and tick is
   readable without zooming. This is the only check that matters, and it
   catches problems no on-screen review will.
2. Confirm no figure was scaled in LaTeX. If a figure looks small, change
   the width it is generated at rather than scaling it in the document.
3. View one figure in grayscale. Family colour plus dash pattern should
   still separate the lines.
4. Confirm every caption states what the reader should conclude, not only
   what is plotted.
