# Writing pack: everything needed for the remaining sections

Self-contained reference for drafting **Experimental Setup, Results,
Limitations, Conclusion**, and the missing **Related Work subsection 1**.
Every number here was read from a results file or from the code. Nothing is
estimated. Where a value still needs to be measured or confirmed it is
marked **[FILL]**, and those markers must not survive into the paper.

Companion documents: `docs/CORRUPTIONS.md` (physical grounding),
`docs/FIGURES.md` (figure plan and captions), `docs/DRAFT_REVIEW.md`
(corrections to the existing draft), `docs/DATASHEET.md` (provenance and
ethics language).

---

## 0. Read this first: one correction to a sentence already in the draft

The current Benchmark Design text says SARD "is evaluated directly at 1024
pixels without tiling." **That is not what the code does.**

`infer_records` in `src/aegisbench/inference.py` calls `tile_image`
unconditionally, with `tile_size=1024, overlap=256` for every dataset. The
sweep was run with those defaults. Measured directly:

| Dataset | Frame size | Tiles per frame |
| --- | --- | --- |
| HERIDAL | 4000 x 3000 | 20 |
| SARD | 1920 x 1080 | 6 |

SARD frames tile into a 3 x 2 arrangement (x origins 0, 768, 896; y origins
0, 56), with heavy overlap on the last column because the frame width is
not a clean multiple of the stride.

Four other files carry the same wrong claim: `docs/DATA.md` line 57,
`docs/DATASHEET.md` line 147, `docs/PAPER_GUIDE.md` line 181, and
`docs/RUNBOOK.md` line 82. They all predate the sweep and were never
reconciled against it.

**This does not invalidate any result and nothing needs rerunning.**
Detections are merged back to full-image coordinates with class-agnostic
NMS in both cases, and evaluation is always against the original full-image
ground truth. Tiling SARD is harmless and arguably preferable, since it
preserves native resolution. Only the description is wrong.

**Correct text to use:**

> Both datasets are processed with the same overlapping-tile pipeline:
> 1024-pixel tiles with 256 pixels of overlap. A 4000 x 3000 HERIDAL frame
> yields 20 tiles and a 1920 x 1080 SARD frame yields 6. Detection runs
> independently per tile, predicted boxes are mapped back into full-image
> coordinates, and overlapping predictions are merged with class-agnostic
> non-maximum suppression, so evaluation is always performed against the
> original full-resolution ground truth rather than against individual
> tiles. Applying one inference protocol to both datasets removes it as a
> confound when comparing degradation across capture regimes.

That last sentence turns the correction into a methodological strength,
which is accurate: a uniform pipeline is the better design.

---

## 1. Experimental setup

### 1.1 Detectors

Three architectures spanning the dominant detection paradigms. All
fine-tuned from pretrained weights, one training run per model per dataset.

| Model | Paradigm | Base weights | Framework |
| --- | --- | --- | --- |
| Faster R-CNN | Two-stage CNN | torchvision pretrained | torchvision |
| YOLOv11 (yolo11m) | Single-stage CNN | `yolo11m.pt` | Ultralytics |
| RT-DETR (rtdetr-l) | Transformer | `rtdetr-l.pt` | Ultralytics |

### 1.2 Training hyperparameters

Read directly from `configs/train_*.yaml`. Identical on both datasets.

| | YOLOv11 | RT-DETR | Faster R-CNN |
| --- | --- | --- | --- |
| Epochs | 80 | 80 | 26 |
| Batch size | 4 | 2 | 2 |
| Input resolution | 1024 | 1024 | 1024 |
| Initial LR | 0.01 | 0.0001 | 0.005 |
| Optimizer | auto (Ultralytics) | auto (Ultralytics) | SGD |
| Early-stopping patience | 20 | 20 | none |
| Mixed precision | yes | yes | yes |
| Seed | 17 | 17 | 17 |
| Dataloader workers | 4 | 4 | 4 |

Worth stating explicitly, since a reviewer will notice the asymmetry: the
learning rates differ by two orders of magnitude because they follow each
architecture's established fine-tuning practice (transformer detectors need
a much smaller LR than YOLO), and Faster R-CNN runs fewer epochs on a fixed
schedule rather than with early stopping. These are per-architecture
conventions, not tuning against the corrupted data, which never enters
training or threshold selection.

**Determinism.** `deterministic=True` is set for the Ultralytics runs, and
`seed_everything` seeds Python, NumPy, and torch.

### 1.3 Hardware

Single consumer GPU, 8 GB VRAM (RTX 5070, Blackwell / sm_120), with a
nightly CUDA 12.8 PyTorch build, which is what that architecture required at
the time. Worth stating: it shows the benchmark is reproducible on one
consumer card rather than requiring a cluster, which matters for a paper
arguing that this evaluation should be routine.

### 1.4 Datasets and splits

| | HERIDAL | SARD |
| --- | --- | --- |
| Regime | High-altitude orthophoto | Lower-altitude drone video |
| Frame size | ~4000 x 3000 | 1920 x 1080 |
| Test images | **101** | **862** |
| Train images | **[FILL]** | **[FILL]** |
| Val images | **[FILL]** | **[FILL]** |
| Source | FESB / IPSAR, free for research | IEEE DataPort |

Test counts are confirmed: they are the `n_images` column in
`ci_heridal.csv` and `ci_sard.csv`. Train and validation counts must be
measured from your own copies; `docs/DATA.md` is explicit that you should
never quote counts you did not measure.

**Split policy.**

- **HERIDAL:** the official test split is used untouched. Validation is
  carved deterministically as 15% of the official train split, used only
  for operating-point selection and early stopping.
- **SARD:** frames come from video, so consecutive frames are near
  duplicates. Splitting is **group-aware**: a whole video sequence goes to
  exactly one of train, validation, or test. Group identity comes from the
  filename prefix, and `phase1_prepare_sard.py` prints the discovered group
  table and refuses degenerate groupings. Without this, near-identical
  frames land in both train and test and every number is inflated. Say this
  explicitly; reviewers who know this dataset look for it.

### 1.5 Evaluation protocol summary

Already written in the draft; the numbers you need to state alongside it:

- **Frozen operating points**, selected once by maximizing F1 on clean
  validation data, then held fixed across all 28 conditions per model:

| Model | HERIDAL | SARD |
| --- | --- | --- |
| Faster R-CNN | 0.99 | 0.96 |
| RT-DETR | 0.65 | 0.60 |
| YOLOv11 | 0.45 | 0.30 |

- **Matching:** greedy, score-ordered, IoU >= 0.5, one prediction per
  ground-truth box.
- **Bootstrap:** 1,000 resamples per condition, resampled with replacement
  at image level, 2.5th and 97.5th percentiles.
- **Sweep size:** 3 models x (1 clean + 9 corruptions x 3 severities) x 2
  datasets = **168 conditions**.

### 1.6 Ethics and data statement

One short paragraph, required by the WACV ethics guidelines for work using
human-subject imagery:

> We use only publicly available research datasets, obtained from their
> original distributors under those datasets' own terms, and collect no new
> human-subjects data. SARD consists of staged imagery in which volunteers
> simulate lost or injured persons; HERIDAL contains aerial imagery of
> people in wilderness terrain. We add no new imagery and no new
> annotations of people, and annotate no identity, biometric, or
> demographic attribute: persons appear only as generic bounding boxes
> inherited from the source datasets. We redistribute no imagery. Because
> every corruption is deterministically seeded, the corrupted benchmark
> regenerates exactly from the clean datasets, so reproduction requires
> only the released code and the original datasets under their own
> licenses.

---

## 2. Results

### 2.1 Table 1 material: measured calibration statistics

Real measured values from `results/phase3/calibration.csv`, on a random
sample of **20 HERIDAL test images** per corruption per severity, reporting
the mean.

| Corruption | Family | Statistic | Direction | s1 | s2 | s3 | Monotonic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `water_glare` | flood | saturated fraction | increasing | 0.0116 | 0.0182 | 0.1895 | yes |
| `turbidity_cast` | flood | RMS contrast | decreasing | 0.1098 | 0.0701 | 0.0378 | yes |
| `inundation` | flood | occluded fraction | increasing | 0.0830 | 0.2235 | 0.3921 | yes |
| `smoke_haze` | wildfire | RMS contrast | decreasing | 0.1170 | 0.0776 | 0.0475 | yes |
| `fire_warm_tint` | wildfire | red / blue ratio | increasing | 1.6743 | 2.0721 | 2.4516 | yes |
| `rain_streaks` | storm | streak density | increasing | 0.1341 | 0.0906 | 0.1294 | **no** |
| `motion_blur` | storm | edge strength | decreasing | 0.0764 | 0.0572 | 0.0472 | yes |
| `low_light` | storm | mean luminance | decreasing | 0.2355 | 0.1533 | 0.1175 | yes |
| `dust_haze` | earthquake | RMS contrast | decreasing | 0.1094 | 0.0720 | 0.0489 | yes |

State the sample size in the caption. See section 4.4 for how to write the
`rain_streaks` exception.

A useful observation the numbers support, worth one sentence: `smoke_haze`
and `dust_haze` reach nearly identical RMS contrast at every severity
(0.0475 vs 0.0489 at s3), yet `dust_haze` is far more damaging to detection
(0.058 vs 0.391 recall at s3 on SARD/YOLOv11). Contrast reduction alone
therefore does not predict detection difficulty. Frame this as an
observation, not a mechanism: dust also differs in airlight chromaticity,
transmission-field granularity, and added grain, so the cause is not
isolated by this experiment.

### 2.2 Table 2 material: clean baselines

All real, from `master_ci.csv`.

| Model | Dataset | Recall | Precision | mAP@0.5 | mAP@[.5:.95] |
| --- | --- | --- | --- | --- | --- |
| Faster R-CNN | HERIDAL | 0.7448 | 0.9029 | 0.7853 | 0.5176 |
| RT-DETR | HERIDAL | 0.7923 | 0.8812 | 0.8349 | 0.5555 |
| YOLOv11 | HERIDAL | 0.7685 | 0.8548 | 0.8029 | 0.5096 |
| Faster R-CNN | SARD | 0.8788 | 0.9669 | 0.8962 | 0.6146 |
| RT-DETR | SARD | 0.8879 | 0.9721 | 0.9292 | 0.6333 |
| YOLOv11 | SARD | 0.8906 | 0.9550 | 0.9277 | 0.6010 |

**What to say about these.** Two points:

1. **Sanity against published work.** The commonly cited HERIDAL reference
   point is roughly 0.90 precision / 0.893 recall / 0.834 mAP@0.5 for
   YOLOv5L. Our HERIDAL numbers sit modestly below that on recall and
   comparable on precision and mAP, which is the expected spread across
   different architectures and input pipelines. Order-of-magnitude
   agreement is the claim to make, not a superiority claim. This preempts
   "how do we know the training is sound?" **[FILL]** Verify the exact
   reference figures and citation before quoting them.
2. **SARD is the easier task.** Clean recall is roughly 0.88 to 0.89 on
   SARD against 0.74 to 0.79 on HERIDAL, consistent with lower altitude and
   larger relative object size. This is exactly why every degradation
   result is reported as *relative* drop against each model's own clean
   score: raw corrupted metrics are not comparable across datasets that do
   not share a difficulty floor.

### 2.3 Table 3 material: the low-light collapse

**HERIDAL**, recall at the frozen operating point, bootstrap mean with 95%
interval, n = 101.

| Model | Threshold | Severity 1 | Severity 2 | Severity 3 |
| --- | --- | --- | --- | --- |
| RT-DETR | 0.65 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| YOLOv11 | 0.45 | 0.048 [0.027, 0.073] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Faster R-CNN | 0.99 | 0.452 [0.365, 0.540] | 0.110 [0.064, 0.163] | 0.000 [0.000, 0.000] |

**SARD**, n = 862. Point estimates from `master_ci.csv`; CIs shown where
recorded.

| Model | Threshold | Severity 1 | Severity 2 | Severity 3 |
| --- | --- | --- | --- | --- |
| Faster R-CNN | 0.96 | 0.419 | 0.077 | 0.000 |
| RT-DETR | 0.60 | 0.260 | 0.013 | 0.000 |
| YOLOv11 | 0.30 | 0.184 [0.157, 0.212] | 0.007 [0.003, 0.012] | 0.000 [0.000, 0.000] |

**The threshold-independence check, which is what makes this defensible.**
A reviewer's first objection to a zero is that it is an artifact of a badly
placed confidence threshold. mAP scans all thresholds, so it settles the
question. HERIDAL, low light, severity 1:

| Model | Clean mAP@0.5 | Severity 1 mAP@0.5 |
| --- | --- | --- |
| RT-DETR | 0.8349 | **0.0006** |
| YOLOv11 | 0.8029 | 0.0636 |
| Faster R-CNN | 0.7853 | 0.5366 |

If this were a calibration artifact, mAP would stay high while
frozen-threshold recall fell. It does not: mAP collapses in lockstep. The
detections are not below threshold, they are absent.

SARD low light, for the same argument on the second dataset:

| Model | s1 recall / mAP@0.5 | s2 recall / mAP@0.5 | s3 recall / mAP@0.5 |
| --- | --- | --- | --- |
| Faster R-CNN | 0.419 / 0.546 | 0.077 / 0.142 | 0.000 / 0.000 |
| RT-DETR | 0.260 / 0.421 | 0.013 / 0.049 | 0.000 / 0.001 |
| YOLOv11 | 0.184 / 0.247 | 0.007 / 0.018 | 0.000 / 0.000 |

**Two claims to make, and one not to.**

- Safe and cross-dataset: **Faster R-CNN is consistently the most robust of
  the three under low light**, on both datasets, at every severity where
  any model retains recall.
- Safe: total collapse by severity 3 replicates across both datasets and
  all three architectures.
- **Not safe:** "RT-DETR is the most fragile architecture." True on HERIDAL
  (zero at every severity, including the mildest) but it reverses on SARD,
  where RT-DETR outperforms YOLOv11 at severities 1 and 2. State the
  YOLOv11 versus RT-DETR ordering as dataset-dependent. A reviewer holding
  both tables will notice an unqualified claim.

Also worth one sentence: Faster R-CNN has the **strictest** threshold (0.99
on HERIDAL) and is nonetheless the most robust, so robustness ordering is
not explained by threshold strictness. That is the obvious alternative
explanation and it is worth closing off.

### 2.4 The severity spectrum

SARD, YOLOv11, bootstrap mean recall. Clean baseline 0.8906.

| Corruption | Family | s1 | s2 | s3 |
| --- | --- | --- | --- | --- |
| `rain_streaks` | storm | 0.893 | 0.883 | 0.855 |
| `inundation` | flood | 0.860 | 0.744 | 0.494 |
| `water_glare` | flood | 0.859 | 0.809 | 0.666 |
| `smoke_haze` | wildfire | 0.810 | 0.667 | 0.391 |
| `turbidity_cast` | flood | 0.796 | 0.576 | 0.191 |
| `motion_blur` | storm | 0.792 | 0.613 | 0.477 |
| `fire_warm_tint` | wildfire | 0.689 | 0.311 | 0.099 |
| `dust_haze` | earthquake | 0.673 | 0.272 | 0.058 |
| `low_light` | storm | 0.184 | 0.007 | 0.000 |

The `water_glare` s1 95% interval is [0.838, 0.880] if you want one worked
example of interval width in the text.

**The framing.** At matched severity 3 the spread runs from 0.855
(`rain_streaks`, a 3.6-point drop from clean) to 0.000 (`low_light`, total
loss). Rain barely registers; low light removes the capability entirely.
That range, produced by corruptions calibrated on comparable statistical
ladders, is the argument that this benchmark is diagnostic rather than
merely hard. A single averaged robustness score would hide all of it, which
is worth saying directly since averaging is the convention this work
departs from.

Note for the abstract: 0.8906 clean to 0.855 at s3 is **3.6 points**. The
abstract currently says "roughly four points." Either is defensible;
just keep the abstract and the Results section consistent.

### 2.5 Per-family aggregates

Mean relative recall drop, `(clean - corrupted) / clean`, averaged over
severities, from `robustness.py::summarize`.

**HERIDAL**

| Family | Faster R-CNN | RT-DETR | YOLOv11 |
| --- | --- | --- | --- |
| Storm | 0.784 | 0.874 | 0.829 |
| Earthquake | 0.534 | 0.568 | 0.571 |
| Wildfire | 0.335 | 0.459 | 0.405 |
| Flood | 0.301 | 0.231 | 0.246 |

**SARD**

| Family | Faster R-CNN | RT-DETR | YOLOv11 |
| --- | --- | --- | --- |
| Earthquake | 0.481 | 0.630 | 0.624 |
| Storm | 0.401 | 0.440 | 0.413 |
| Wildfire | 0.327 | 0.451 | 0.445 |
| Flood | 0.264 | 0.293 | 0.252 |

Storm is worst on HERIDAL for every model, pulled up by `low_light`.
Earthquake (`dust_haze`) edges it out on SARD. **Flood is the mildest
family on both datasets for every model without exception**, which is the
one family-level ordering that is fully consistent across the study.

### 2.6 Localization stability

SARD, YOLOv11. `loc_stability_iou` is IoU between the clean and corrupted
predicted boxes for instances detected in **both** conditions. `n_common`
is how many instances that was.

| Corruption | s1 | s2 | s3 | n at s1 | n at s3 |
| --- | --- | --- | --- | --- | --- |
| `rain_streaks` | 0.984 | 0.953 | 0.909 | 976 | 926 |
| `water_glare` | 0.973 | 0.942 | 0.906 | 938 | 729 |
| `inundation` | 0.960 | 0.930 | 0.884 | 938 | 542 |
| `smoke_haze` | 0.917 | 0.884 | 0.852 | 886 | 427 |
| `turbidity_cast` | 0.917 | 0.874 | 0.856 | 870 | 210 |
| `fire_warm_tint` | 0.896 | 0.870 | 0.860 | 751 | 109 |
| `dust_haze` | 0.886 | 0.864 | 0.822 | 734 | 64 |
| `motion_blur` | 0.878 | 0.819 | 0.779 | 864 | 523 |
| `low_light` | 0.833 | 0.804 | n/a | 202 | 0 |

Worked example for the text: `water_glare` severity 1 gives
`loc_stability_iou` 0.973, centre shift 0.0067 of the ground-truth box
diagonal, and an IoU-versus-ground-truth drop of only 0.0077 (0.833 clean
to 0.825 corrupted), over 938 commonly detected instances.

**The finding and how to state it carefully.** Stability declines from 0.98
to 0.78 across the full severity range while recall falls from 0.89 to
0.00. Both decline, so do not claim they are unrelated. The claim that the
data supports is about *magnitude*: recall spans nearly the entire unit
range while stability stays inside a narrow band near the top, never
falling below 0.78 even in conditions that remove almost every detection.
Meanwhile `n_common` tracks recall exactly (`low_light`: 202, then 8, then
0). The dominant failure mode is detections disappearing, not boxes
drifting.

**The `n/a` at `low_light` severity 3 is correct behaviour, not a gap.**
With zero instances detected in both conditions there is nothing to
average, and the evaluator returns NaN with `n_common = 0` rather than
silently reporting a misleading zero. Say so in the table caption.

**Operational implication, worth one sentence in Discussion.** If surviving
detections stay accurately placed, then for a deployed system the failure
mode is missed detections rather than false placement. An operator can
trust the boxes that appear; the risk is the ones that never do. That also
tells a mitigation what to target: sensitivity, not localization.

---

## 3. Related Work subsection 1, the missing one

This is the highest-priority remaining prose because it carries the
citations the paper cannot ship without and the differentiation a reviewer
will look for. Full details in `docs/DRAFT_REVIEW.md` sections 3 and 4.

**Must cite here:** the HERIDAL paper (Božić-Štulić, Marušić, Gotovac, IJCV
127(9):1256-1278, 2019), the SARD paper (Sambolek and Ivašić-Kos, IEEE
Access, 2021), and ideally SeaDronesSee (Varga et al., WACV 2022, pp.
2260-2270) and TinyPerson (Yu et al., WACV 2020, pp. 1257-1265), both of
which are WACV papers in adjacent SAR domains.

**The differentiation that must appear.** The SARD authors already released
a small corrupted-image supplement of their own, **SARD-Corr**: synthetic
fog, snow, ice, and motion blur. This is the closest prior attempt at this
paper's premise, and it lives inside a paper you already cite for the
dataset. Address it directly:

> The closest prior robustness evaluation on this data is the SARD-Corr
> supplement released with SARD itself, which adds fog, snow, ice, and
> motion-blur variants to a subset of the imagery. That supplement uses a
> small set of generic weather effects without a calibrated severity
> structure, and reports a single robustness check rather than a systematic
> sweep. We extend this into a benchmark: nine corruptions organized by
> disaster family, each at three severities calibrated against a declared
> and machine-checked image statistic, evaluated across three
> architecturally distinct detectors and two capture regimes with bootstrap
> confidence intervals and a second, localization-stability axis.

---

## 4. Limitations

Seven items. All are real, checked, and stated once. This is the right
length: padding the list buries the finding, and cutting any of these
leaves something a reviewer will find on their own.

### 4.1 Synthetic corruptions on real imagery

The established corruption-robustness methodology, following ImageNet-C and
Foggy Cityscapes. It is what makes controlled, per-condition attribution
possible at all, since real disaster imagery does not come with matched
clean counterparts of the same scene. It is also the benchmark's ceiling:
it measures response to modeled optics, not to live disaster footage. State
both halves in the same paragraph. Validation against real adverse-condition
imagery is the natural next step, and is hard precisely because these are
emergencies.

### 4.2 Single training seed

One training run per model per dataset. The bootstrap intervals quantify
variance from *which images are in the test set*, not variance from
training. Do not let the intervals imply more than they cover. Say this
plainly rather than leaving a reader to work it out.

### 4.3 Scope: two datasets, three detectors

Use the framing that makes this a design decision rather than an omission,
because that is what it is:

> We evaluate on two datasets spanning the two major real-world SAR-imagery
> regimes and three architecturally distinct detectors representative of
> the dominant paradigms. We prioritise methodological depth, physically
> grounded corruption modeling, group-aware splitting, frozen-threshold
> evaluation, and bootstrap confidence intervals, over dataset and model
> count, consistent with the goal of a rigorous diagnostic benchmark rather
> than a leaderboard survey.

### 4.4 The rain_streaks calibration proxy

Eight of nine corruptions pass automatic monotonicity verification;
`rain_streaks` does not (measured 0.1341, 0.0906, 0.1294). The declared
statistic in `configs/corruptions.yaml` is `streak_density`, which has no
closed-form image measure, so `edge_strength` is substituted as a proxy. The
proxy is confounded: additional streaks raise measured edge strength while
the veil blur applied in the same corruption (0.4, 0.8, 1.2 sigma per 1000
px, increasing with severity) suppresses it, and the two effects move in
opposite directions. The corruption's own generation parameters increase
monotonically by construction (streaks per megapixel 120, 350, 800; streak
length 12, 20, 30 per 1000 px), and downstream recall degrades in the
expected direction (0.893, 0.883, 0.855). This is a limitation of the proxy
statistic, not evidence that the corruption fails to intensify.

This disclosure makes the paper stronger, not weaker: it shows the
verification pipeline caught a problem in its own instrumentation rather
than rubber-stamping every corruption.

### 4.5 Monotonicity is verified on a sample, not exhaustively

Verification measures the statistic on a random sample of 20 test images per
corruption per severity and checks that the **mean** moves in the declared
direction; the unit-test suite runs the same check on synthetic imagery.
Averaging is deliberate, since the corruptions are stochastic and a single
image can deviate without the ladder being ill-formed. But the paper must
describe what was actually done rather than implying every image was
checked.

### 4.6 The inundation ripple

`inundation` is the one corruption with a geometric component: a
water-refraction ripple applied through pixel remapping. The ground-truth
box is not warped to follow it. The amplitude is bounded well below person
scale, reaching approximately 1.4 pixels at severity 3 on a 4000-pixel
frame, so residual misalignment is negligible, and the occlusion and murky
blend rather than the ripple carry the degradation. Disclose it as a minor
deviation from exact pixel alignment rather than leaving it for a reader to
discover.

### 4.7 No mitigation study

Disaster-aware training augmentation is the natural mitigation and the
pipeline for it exists in the released code, but evaluating it is out of
scope for this submission and is named as immediate future work. A
benchmark contribution stands on its own; a rushed, under-trained
mitigation result would be worse than none. Precedent worth one clause:
Michaelis et al. found stylization-based augmentation recovered substantial
robustness in the driving setting, which is what makes this the obvious
next experiment.

---

## 5. Conclusion

Short, three moves, no new numbers beyond the headline.

1. **Restate the gap.** Aerial SAR detection is benchmarked on clear-weather
   imagery and deployed into disasters.
2. **Restate the finding at its strongest.** Robustness is highly uneven:
   heavy rain is nearly free, while low light removes the capability
   entirely, to exactly zero recall, on both datasets and for all three
   architectures, confirmed threshold-independently. Surviving detections
   stay accurately placed, so the failure mode is blindness rather than
   imprecision.
3. **State the implication and the next step.** Systems intended for
   disaster response should be evaluated under disaster conditions, and
   current detectors have a specific, measurable, architecture-independent
   blind spot at exactly the night-time conditions where aerial search is
   most valuable and least substitutable by ground teams. Disaster-aware
   training augmentation is the natural mitigation and the immediate next
   experiment.

Avoid closing on a claim of generality the study does not support. Two
datasets and three detectors is the honest scope, and the conclusion is
stronger for staying inside it.

---

## 6. Checklist of everything still marked [FILL]

| Item | Where | How to get it |
| --- | --- | --- |
| HERIDAL train / val counts | Setup 1.4 | Measure from your prepared records |
| SARD train / val counts | Setup 1.4 | Same |
| Published HERIDAL reference figures and citation | Results 2.2 | Confirm from the HERIDAL literature |
| Whether AWOD is real or synthetic imagery | Related Work 2 | Open the WRRT-DETR paper; see `DRAFT_REVIEW.md` section 2 |
| Missing citations: HERIDAL, SARD, Faster R-CNN, RT-DETR, YOLOv11, Efron and Tibshirani | Bibliography | `DRAFT_REVIEW.md` section 3 |

Everything else in this document is measured and can be used as written.

---

## 7. Follow-ups: three points that need more precision

### 7.1 The literal calibration procedure, not a paraphrase

Read directly from `scripts/phase3_calibrate.py` lines 36 to 58. The
sampling and the check are both more specific than "20-image sample, by
mean" suggests.

**Sampling.** `picks` is drawn **once, before the loop over corruptions**:

```python
rng   = np.random.default_rng(stable_seed("phase3-calib", args.records))
picks = rng.choice(len(records), size=min(args.n, len(records)),
                   replace=False)
for name in suite.names():        # <- picks is already fixed here
```

Therefore:

- **20 images out of the 101-image HERIDAL test split**, since the run used
  `--records data/heridal/records/test.json --n 20`.
- Drawn **without replacement**, so 20 distinct images.
- **The same 20 images for every one of the nine corruptions and every one
  of the three severities.** They are not resampled per corruption. This is
  the right design: holding the image subset fixed means a severity ladder
  is compared against itself on identical content, and differences between
  corruptions are not confounded by having looked at different scenes.
- The draw is **deterministic**, seeded by a stable hash of the string
  `"phase3-calib"` and the records path, so rerunning selects the same 20
  images.

**What is checked.** For each `(corruption, severity)` pair the statistic is
computed on each of the 20 corrupted images and averaged. Monotonicity is
then asserted on the three resulting means:

```python
seq  = [means[1], means[2], means[3]]
mono = all(a < b for a, b in zip(seq, seq[1:]))   # or a > b, per direction
```

So the guarantee is: **the mean statistic over a fixed 20-image subset moves
in the declared direction across severities.** It is not a per-image
guarantee, and the paper must not imply one. Per-image monotonicity is the
wrong thing to require anyway, since every corruption is stochastic and a
single image can invert without the ladder being ill-formed.

**Sentence to use in Limitations:**

> Severity calibration is verified on a fixed random subset of 20 images
> drawn without replacement from the HERIDAL test split, held constant
> across all nine corruptions and all three severities. For each corruption
> and severity we measure the declared statistic on every image in that
> subset and verify that the subset mean moves in the declared direction.
> This is a guarantee about the mean over that subset, not about every
> individual image: the corruptions are stochastic, so an individual image
> can invert without the severity ladder being ill-formed. The same check
> runs on synthetic imagery in the unit-test suite.

**One improvement worth making, since the data already exists.**
`phase3_calibrate.py` writes a `std_value` column alongside `mean_value`
and `n` for every row. Reporting Table 1 as **mean ± std (n = 20)** rather
than mean alone directly answers the "only 20 images?" question by showing
the dispersion, and it costs nothing because the numbers are already in
`results/phase3/calibration.csv`. I do not have those standard deviations
here: the console output printed only the means, so read the `std_value`
column from the CSV when building the table.

### 7.2 The dust versus smoke gap, and the harder question behind it

The observation as stated invites a sharper follow-up than the one it
answers. If RMS contrast is the declared calibration statistic for both
corruptions, and two corruptions sitting at nearly identical values of it
(0.0475 and 0.0489 at severity 3) produce a 6.7-fold difference in recall
(0.391 versus 0.058), a reviewer may reasonably ask whether RMS contrast is
a good calibration statistic at all.

**The answer, which is worth putting in the paper rather than holding in
reserve, is that the statistic is not doing the job that question assumes.**
A calibration statistic here certifies that severity is ordered and
measurable *within* a single corruption. It is not a cross-corruption
difficulty normaliser, and the benchmark never claims that two corruptions
at matched statistic values should cost the same recall.

That distinction is worth one explicit sentence in the calibration
subsection, because it converts a potential objection into a design
statement:

> The calibration statistic makes each corruption's severity ladder
> measurable and falsifiable within that corruption. It is not a
> cross-corruption difficulty normaliser: two corruptions matched on the
> same statistic are not expected to be equally damaging, and we make no
> such claim.

**And the dust versus smoke gap is then a finding rather than an
embarrassment.** If the calibration statistic did predict detection
difficulty across corruptions, the benchmark would have little to reveal,
since measuring contrast would substitute for running the sweep. That two
corruptions built on the same scattering model, at nearly identical global
contrast, differ this much in cost is direct evidence that detection
difficulty in this domain is not reducible to global contrast reduction.
What separates them is what the taxonomy already distinguishes: airlight
chromaticity, the spatial granularity of the transmission field, and dust's
additional near-lens grain. This experiment does not isolate which of those
carries the effect, and the paper should say so.

**The honest scope note that goes with it.** Because the ladders are
calibrated within corruptions and not equalised across them, statements
like "dust is more damaging than smoke" are comparisons of the corruptions
*as modeled here*, at their respective severity 3, rather than claims about
matched physical severity in the world. Worth one clause so the comparison
is not over-read.

### 7.3 Table 1, ready to drop in

These supersede the `[Table 1 here]` placeholder currently in Benchmark
Design. All values are measured, from `results/phase3/calibration.csv`,
mean over the fixed 20-image subset described in 7.1.

| Corruption | Family | Optical mechanism | Statistic | s1 | s2 | s3 |
| --- | --- | --- | --- | --- | --- | --- |
| `water_glare` | Flood | Specular sun-glint saturating the sensor | saturated fraction ↑ | 0.0116 | 0.0182 | 0.1895 |
| `turbidity_cast` | Flood | Sediment-laden water: mud chromaticity blend, contrast compression | RMS contrast ↓ | 0.1098 | 0.0701 | 0.0378 |
| `inundation` | Flood | Semi-transparent standing water, ripple refraction | occluded fraction ↑ | 0.0830 | 0.2235 | 0.3921 |
| `smoke_haze` | Wildfire | Koschmieder scattering, neutral-gray airlight | RMS contrast ↓ | 0.1170 | 0.0776 | 0.0475 |
| `fire_warm_tint` | Wildfire | Low-CCT fire illumination, warm airlight haze | red / blue ratio ↑ | 1.6743 | 2.0721 | 2.4516 |
| `rain_streaks` | Storm | Additive directional streaks over a rain veil | streaks per megapixel ↑ † | 120 | 350 | 800 |
| `motion_blur` | Storm | Wind-induced platform shake, linear blur | edge strength ↓ | 0.0764 | 0.0572 | 0.0472 |
| `low_light` | Storm | Photon scaling in linear domain, shot + read noise | mean luminance ↓ | 0.2355 | 0.1533 | 0.1175 |
| `dust_haze` | Earthquake | Koschmieder scattering, brown mineral airlight, grain | RMS contrast ↓ | 0.1094 | 0.0720 | 0.0489 |

**On the `rain_streaks` row.** The table above reports its *generation
parameter* (streaks per megapixel, monotonic by construction) rather than a
measured image statistic, marked with a dagger. This is the honest
presentation: the declared statistic `streak_density` has no closed-form
image measure, the substituted `edge_strength` proxy is confounded by the
veil blur applied in the same corruption, and the measured proxy values
(0.1341, 0.0906, 0.1294) are not monotonic. Putting a non-monotonic
sequence into a table headed "calibrated severity" without room to explain
it invites exactly the wrong reading. Report the parameter in the table,
mark it, and explain the proxy properly in the calibration subsection and
in Limitations, where there is space (see 4.4).

If you prefer to report the measured proxy for consistency of column
meaning, that is defensible too, but then the dagger footnote must state
the non-monotonicity directly in the caption rather than deferring it.

**Caption:**

> **Table 1.** The nine corruptions, their disaster family, the optical
> mechanism each models, and the statistic defining its severity ladder.
> Arrows give the direction the statistic must move as severity increases.
> Values are means over a fixed random subset of 20 HERIDAL test images,
> held constant across all corruptions and severities; monotonicity is
> verified automatically on these means. † `rain_streaks` has no
> closed-form image statistic isolating streak density, so its ladder is
> defined by the generation parameter shown and audited visually; see
> Section [X].

**LaTeX skeleton:**

```latex
\begin{table*}[t]
\centering\small
\begin{tabular}{llllrrr}
\toprule
Corruption & Family & Optical mechanism & Statistic & s1 & s2 & s3 \\
\midrule
\texttt{water\_glare}    & Flood      & Specular sun-glint saturation          & saturated frac.\ $\uparrow$ & 0.0116 & 0.0182 & 0.1895 \\
\texttt{turbidity\_cast} & Flood      & Sediment-laden water, contrast loss    & RMS contrast $\downarrow$   & 0.1098 & 0.0701 & 0.0378 \\
\texttt{inundation}      & Flood      & Semi-transparent standing water        & occluded frac.\ $\uparrow$  & 0.0830 & 0.2235 & 0.3921 \\
\midrule
\texttt{smoke\_haze}     & Wildfire   & Koschmieder, neutral-gray airlight     & RMS contrast $\downarrow$   & 0.1170 & 0.0776 & 0.0475 \\
\texttt{fire\_warm\_tint}& Wildfire   & Low-CCT firelight, warm airlight haze  & red/blue ratio $\uparrow$   & 1.6743 & 2.0721 & 2.4516 \\
\midrule
\texttt{rain\_streaks}   & Storm      & Directional streaks over a rain veil   & streaks/MP $\uparrow$~$\dagger$ & 120 & 350 & 800 \\
\texttt{motion\_blur}    & Storm      & Wind-induced platform shake            & edge strength $\downarrow$  & 0.0764 & 0.0572 & 0.0472 \\
\texttt{low\_light}      & Storm      & Linear-domain photon scaling, noise    & mean luminance $\downarrow$ & 0.2355 & 0.1533 & 0.1175 \\
\midrule
\texttt{dust\_haze}      & Earthquake & Koschmieder, brown mineral airlight    & RMS contrast $\downarrow$   & 0.1094 & 0.0720 & 0.0489 \\
\bottomrule
\end{tabular}
\caption{...}
\label{tab:taxonomy}
\end{table*}
```

Add the standard deviations from the CSV's `std_value` column if you adopt
the mean ± std presentation recommended in 7.1.
