# Datasheet: the AegisBench benchmark suite

Structured after Gebru et al., *Datasheets for Datasets*. The WACV 2027
Evaluations and Datasets track directs authors to the NeurIPS 2026
Evaluations and Datasets guidelines, which expect this kind of structured
documentation. Include this file (as PDF or Markdown) in the supplementary
archive.

Fields marked **[VERIFY]** must be filled or confirmed by the authors
before submission. Do not submit this file with any `[VERIFY]` marker left
in place.

---

## What is actually being released

Read this section first, because it determines how every other answer below
is framed.

AegisBench does **not** redistribute images. It releases a *generator plus a
protocol plus results*:

| Released artifact | What it is |
| --- | --- |
| Corruption engine | Deterministic NumPy/OpenCV implementation of nine physically motivated corruptions (`src/aegisbench/corruptions/`) |
| Corruption parameter table | The canonical severity ladder with declared calibration statistics (`configs/corruptions.yaml`) |
| Evaluation harness | One shared evaluator: COCO mAP, fixed-operating-point P/R, size-stratified recall, relative drop, localization stability, bootstrap CIs (`src/aegisbench/evaluation/`) |
| Tiling pipeline | Overlapping-tile inference with exact box remapping and class-agnostic NMS merge (`src/aegisbench/tiling.py`) |
| Results tables | The full 168-condition sweep, bootstrap confidence intervals, and localization-stability tables (`results/sweep/*.csv`) |
| Archived predictions | Raw per-condition detections, so any downstream analysis can be recomputed without rerunning inference (`results/sweep/preds/`) |
| Protocol and runbook | The phased experimental protocol with per-phase verification checkpoints (`docs/RUNBOOK.md`) |

The corrupted evaluation images are **not** shipped and do not need to be.
Every stochastic element is seeded by a SHA-256 hash of
`(image_id, corruption, severity, global_seed)` with `global_seed = 20260707`,
so any user who obtains the underlying clean datasets under their own
licenses can regenerate the corrupted benchmark bit-identically on any
machine. This is a deliberate design choice: it makes the benchmark fully
reproducible without redistributing imagery we have no right to redistribute.

---

## Motivation

**For what purpose was the benchmark created?**
Aerial person detection is a core capability for search and rescue, but the
established benchmarks for it (HERIDAL, SARD) were collected in calm, clear
conditions, while real deployments occur during floods, wildfires, storms,
and at night. AegisBench was created to measure, systematically and
reproducibly, how far detection performance falls under the visual
conditions of the disasters that trigger those deployments, and which
specific conditions break which detector architectures.

**Who created it and who funded it?**
**[VERIFY]** Authors and funding sources. Leave blank in the anonymized
review copy; fill in for the camera-ready. Do not name a grant ID in the
double-blind version, since grant IDs are identifying.

---

## Composition

**What do the instances represent?**
AegisBench itself contributes evaluation *conditions* and *results*, not
image instances. One condition is a tuple
`(detector, dataset, corruption, severity)`. The benchmark defines
28 conditions per detector per dataset: one clean condition plus nine
corruptions at three severities each.

**How many instances are there?**
168 evaluation conditions: 3 detectors (Faster R-CNN, YOLOv11, RT-DETR)
x 28 conditions x 2 datasets. Each condition has a full metric row, a
bootstrap confidence-interval row (1,000 resamples), a localization-stability
row where applicable, and an archived prediction file.

**What underlying imagery does evaluation run on?**

| Dataset | Regime | Resolution | Test split used |
| --- | --- | --- | --- |
| HERIDAL | High-altitude orthophoto search imagery | ~4000x3000 | 101 labeled full-size images |
| SARD | Lower-altitude drone video frames | 1920x1080 | 862 frames |

**[VERIFY]** Confirm both counts against your own downloaded copies and
report the numbers you measured, never numbers quoted from elsewhere.

**Is any information missing?**
The corrupted images themselves are not distributed (see above). Ground
truth is inherited unchanged from the source datasets: corruptions are
appearance-only and do not move object content, so clean boxes remain
exactly valid on corrupted images.

**Does the benchmark contain data that could be offensive or sensitive?**
The underlying datasets contain photographs of people. SARD consists of
staged imagery in which actors simulate lost or injured persons. HERIDAL
contains aerial imagery of people in wilderness terrain. AegisBench adds no
new imagery and no new annotations of people. Persons are annotated only as
a single generic `person` class, with no identity, demographic, or attribute
labels of any kind.

---

## Collection process

**How was the data acquired?**
No new imagery was collected for this work. HERIDAL and SARD were obtained
from their original distributors under their own terms (see
`docs/DATA.md`). The contribution is the corruption engine, the evaluation
protocol, and the resulting measurements.

**What is the sampling strategy?**
No sampling: the benchmark evaluates the complete official test split of
each dataset under every condition.

**How were splits determined?**
- **HERIDAL:** the official test split is used untouched. A validation split
  is carved deterministically as 15% of the official train split, used only
  for operating-point selection and early stopping.
- **SARD:** frames originate from video, so consecutive frames are near
  duplicates. Splitting is **group-aware**: an entire video sequence is
  assigned to exactly one of train, validation, or test. Group identity is
  derived from filename prefix, and `scripts/phase1_prepare_sard.py` prints
  the discovered group table and refuses degenerate groupings. Without this
  guard, near-identical frames would appear in both train and test and
  inflate every reported number.

**Were people involved in data collection compensated?**
**[VERIFY]** This concerns the original dataset creators, not this work. If
you state anything here, state only what the original dataset papers
document.

---

## Preprocessing, cleaning, labeling

**Was any preprocessing done?**
- Annotations are parsed from PASCAL VOC XML into a common record schema.
- HERIDAL frames are cut into overlapping 1024 px tiles with 256 px overlap
  for training and inference. A ground-truth box is assigned to a tile if at
  least 30% of its original area is visible; clipped boxes under 4 px are
  dropped. With overlap at least as large as the largest person box, every
  box is fully contained in at least one tile, so no annotation is lost
  dataset-wide. The tiler verifies this per box and reports violations
  rather than silently dropping them.
- Detections are merged back to full-image coordinates with class-agnostic
  NMS, so evaluation is always against the original full-image ground truth
  and stays comparable to published full-image results.
- Both datasets use the same tiling pipeline (1024 px tiles, 256 px
  overlap): 20 tiles per HERIDAL frame, 6 per SARD frame. One inference
  protocol across both removes it as a confound when comparing
  degradation across capture regimes.

**Is the raw data available?**
Yes, from the original distributors. AegisBench modifies nothing about the
source annotations.

---

## Uses

**What has the benchmark been used for?**
The evaluation reported in the accompanying paper: a 168-condition
robustness sweep across three architecturally distinct detectors and two
datasets, with bootstrap confidence intervals and a localization-stability
analysis.

**What other tasks could it be used for?**
Evaluating any aerial person detector under the same conditions; ablating
training-time robustness interventions (the mitigation pipeline exists in
`scripts/phase6_mitigation.py` but was not run for this submission);
studying calibration under distribution shift using the archived
predictions; extending the corruption taxonomy to further disaster families.

**Is there anything that should NOT be done with it?**
- Do not treat results as a certification of operational safety. The
  benchmark measures response to *modeled* disaster optics, not live
  disaster footage.
- Do not compare raw corrupted metrics across the two datasets. They have
  different baseline difficulty (clean recall is roughly 0.88 to 0.89 on
  SARD versus 0.74 to 0.79 on HERIDAL), which is exactly why the benchmark
  reports relative drop against each model's own clean score.
- Do not tune confidence thresholds on corrupted data and report the result
  as robustness. The protocol freezes each model's operating point on clean
  validation data for this reason.

---

## Distribution

**How will it be distributed?**
Code, configuration, protocol, results tables, and archived predictions are
released as a public repository under the MIT License. Source imagery is
**not** redistributed; users obtain HERIDAL and SARD directly from their
original distributors under those datasets' own terms.

**[VERIFY]** For the anonymized review copy, host the code archive
anonymously (an anonymous repository service, or bundled in the
supplementary ZIP) and do not link a repository that identifies the authors.

**What license applies?**
The AegisBench code, configuration, and results are MIT licensed. HERIDAL
and SARD remain governed by their own licenses, which this project neither
alters nor sublicenses.

**Are there IP-based or other restrictions?**
Users must comply with the source datasets' terms. This project asserts no
rights over that imagery and redistributes none of it.

---

## Maintenance

**Who will maintain it?**
**[VERIFY]** Name the maintaining author or lab for the camera-ready. Leave
anonymous for review.

**How can the maintainer be contacted?**
**[VERIFY]** Contact address for the camera-ready. Omit for review.

**Will it be updated?**
**[VERIFY]** State your intent honestly. A reasonable and defensible
commitment: the repository will receive corrections and compatibility fixes,
and any change to `configs/corruptions.yaml` will be released under a new
version tag, since the corruption parameter table defines benchmark identity
and silently changing it would invalidate comparisons against published
numbers.

**How will versioning work?**
Every result row logs the git SHA, the SHA-256 hash of the corruption
config, the global seed, and a UTC timestamp, so any published number can be
traced to the exact code and parameter table that produced it.

---

## Responsible AI notes

**Human subjects.** The underlying datasets contain images of people.
No new human-subjects data was collected for this work, and no identity,
biometric, or demographic attributes are annotated, inferred, or released.
Persons appear only as generic `person` bounding boxes inherited from the
source datasets. **[VERIFY]** State in the paper that you use only publicly
available research datasets under their original terms, and confirm whether
your institution requires an ethics determination for secondary use of such
data.

**Foreseeable harms.** The clearest risk is misplaced confidence: a
practitioner could read strong clear-weather numbers as evidence that a
detector is fit for night or disaster operations. This benchmark exists
specifically to make that gap measurable, and the headline finding (complete
detection collapse under low light across all three architectures on both
datasets) argues directly against such deployment without mitigation.

**Known limitations.** Synthetic corruptions applied to real imagery, a
single training seed per model, two datasets, three detectors, and one
corruption (`rain_streaks`) whose severity ladder is defined by generation
parameters and visual audit rather than by a closed-form image statistic.
Each is stated explicitly in the paper's limitations section.
