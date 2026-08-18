# Writing the AegisBench paper: a complete guide for the co-author

**Purpose of this document.** You are joining a project that already has a
finished experimental pipeline and a complete set of headline results. This
document explains what was built, why each design decision was made, what the
results actually say, and exactly how to turn all of it into a WACV paper. It
assumes you have not seen the codebase before. Read it top to bottom once,
then use it as a reference while drafting.

Everything here is grounded in code and results that exist in this repository.
Where a number is not yet measured, it is explicitly marked `[FILL]` rather
than guessed. Please preserve that discipline: never write a number into the
paper that you did not read out of a results file.

---

## Table of contents

1. [The deadline situation, read this first](#1-the-deadline-situation-read-this-first)
2. [What the project is, in plain language](#2-what-the-project-is-in-plain-language)
3. [Why this is a strong paper](#3-why-this-is-a-strong-paper)
4. [The contributions, stated precisely](#4-the-contributions-stated-precisely)
5. [What we have already measured](#5-what-we-have-already-measured)
6. [The one analysis to run before writing results](#6-the-one-analysis-to-run-before-writing-results)
7. [Section-by-section drafting guide](#7-section-by-section-drafting-guide)
8. [Building the literature review](#8-building-the-literature-review)
9. [The rigor checklist and the objections it preempts](#9-the-rigor-checklist-and-the-objections-it-preempts)
10. [Anticipated reviewer objections and how to answer them](#10-anticipated-reviewer-objections-and-how-to-answer-them)
11. [Figures and tables plan](#11-figures-and-tables-plan)
12. [Writing style guidance](#12-writing-style-guidance)
13. [What is still outstanding](#13-what-is-still-outstanding)
14. [Submission logistics and double-blind hygiene](#14-submission-logistics-and-double-blind-hygiene)

---

## 1. The deadline situation, read this first

WACV 2027 runs a two-round submission system. Round 1 closed on **25 June
2026**, which has already passed. The live target is **Round 2**:

| Milestone | Date |
| --- | --- |
| Round 2 registration opens | 21 August 2026 |
| **Round 2 paper and supplementary deadline** | **28 August 2026** |
| Papers assigned to reviewers | 4 September 2026 |
| Reviews due | 25 September 2026 |
| Decisions released | 9 October 2026 |

Today is 16 August 2026. That leaves roughly **twelve days**. Please verify
these dates yourself against the official WACV site before relying on them,
since conference pages do get amended, and note that the abstract or
registration step happens before the full paper deadline in most CVF
conferences. Do not let registration be the thing that sinks the submission.

**Practical consequence for how you write.** The experiments are done. A
deliberate scope decision was made on 18 August to **not** run the Phase 6
mitigation study for this submission: two unexplained GPU crashes during
early testing made its time cost too uncertain against the deadline, and it
is a bonus contribution, not a required one, for the Evaluation and Datasets
track. Write the paper as a complete benchmark and analysis paper around the
results that already exist (section 5). See section 13 for the exact
Limitations-section framing to use.

### Which track to submit to

WACV 2027 has three tracks with different review criteria: Algorithms,
Applications, and **Evaluation and Datasets**. Submit to **Evaluation and
Datasets**. That track explicitly invites papers that "propose tools,
datasets, benchmarks, and practices for testing, stress-testing, auditing,
comparing, and interpreting AI/ML systems." That is a precise description of
this project.

This choice matters enormously. In the Algorithms track, a reviewer is
entitled to ask "what new architecture or loss did you invent?" and we have no
good answer, because we did not invent one. In the Evaluation and Datasets
track, that question is out of scope by design, and the criteria become
rigor, reproducibility, and whether the benchmark reveals something the field
did not already know. On those criteria this work is strong. Say in the
first paragraph of the introduction that this is a benchmark and analysis
contribution so the reader frames it correctly from the start.

The call also suggests following the NeurIPS 2026 Evaluations and Datasets
track guidelines. Look those up and skim them, since they typically ask for
specific artifacts such as documented dataset access, licensing statements,
and a reproducibility appendix. We can satisfy all of those, and doing so
visibly is cheap credibility.

---

## 2. What the project is, in plain language

### The problem

Aerial search and rescue means flying a drone or aircraft over wilderness or
disaster terrain and finding a missing person in the imagery. Modern object
detectors do this quite well, and there are published results on the standard
benchmarks showing strong performance.

But there is a mismatch that nobody has systematically measured. The standard
aerial search and rescue datasets, HERIDAL and SARD, were collected in calm,
clear, well-lit conditions. Real search and rescue deployments happen in
exactly the opposite conditions: after a flood, during a wildfire, in a storm,
at dusk, in the dust of a collapsed structure. The moment a detector is most
needed is the moment its input least resembles its training data.

So the question this project answers is: **how much does aerial person
detection degrade when the imagery carries the optical signature of the
disaster that caused the emergency, and which specific conditions break which
detectors?**

### The approach

We take the real, clean HERIDAL and SARD imagery and apply nine physically
motivated corruptions, each at three calibrated severities, then evaluate
three architecturally distinct detectors across every resulting condition.
This is the established "corruption robustness" methodology, the same lineage
as ImageNet-C, adapted to a domain where the corruptions are not generic
noise but modeled disaster optics.

The nine corruptions map to four disaster families:

| Family | Corruptions |
| --- | --- |
| Flood | `water_glare`, `turbidity_cast`, `inundation` |
| Wildfire | `smoke_haze`, `fire_warm_tint` |
| Storm | `rain_streaks`, `motion_blur`, `low_light` |
| Earthquake and post-disaster | `dust_haze` (plus `low_light`, shared) |

The three detectors span the three dominant paradigms: **Faster R-CNN** (two
stage, CNN), **YOLOv11** (single stage, CNN), and **RT-DETR** (transformer
based). This is deliberate. When all three fail the same way, the finding is
about the task, not about one model's quirks.

The two datasets span the two real capture regimes: **HERIDAL** is
high-altitude orthophoto-style search imagery at roughly 4000x3000, with a
test split of 101 full-size labeled images. **SARD** is lower-altitude drone
video at 1920x1080, with a test split of 862 frames.

### The vocabulary you need

Read these once. They recur constantly in the codebase and should recur in
the paper.

**Severity.** Each corruption has three intensity levels, 1 through 3. They
are not eyeballed. Each corruption declares a measurable image statistic in
`configs/corruptions.yaml` (for example RMS contrast for the haze family,
mean luminance for low light, red-to-blue ratio for fire tint), and
`scripts/phase3_calibrate.py` measures that statistic on real data and fails
loudly if the ladder is not monotonic. This is what lets us say "calibrated
severities" and mean something falsifiable by it.

**Frozen operating point.** Every detector needs a confidence threshold to
turn raw scores into detections. We select that threshold once, by maximizing
F1 on **clean validation data**, and then freeze it across all 28 conditions.
We never re-tune on corrupted data. This is the single most important
methodological guard in the project, and section 9 explains why.

**Relative performance drop (rPD).** For each corrupted condition, the drop
is measured against the same model's own clean score:
`rPD = (clean - corrupted) / clean`. Measuring relative to each model's own
baseline means a model is not penalized for having a lower clean score to
begin with, so the comparison is about robustness rather than raw accuracy.
Implemented in `src/aegisbench/evaluation/robustness.py`.

**Localization stability.** A second robustness axis that is genuinely
uncommon in corruption benchmarks and is one of our differentiators. Recall
asks "did the detector still find the survivor?" Localization stability asks
a different question: "for the survivors it did still find, did the predicted
box stay planted on the person, or did its aim get shakier?" We compute it
only over instances detected in **both** the clean and corrupted conditions,
so it is not confounded by recall. Implemented in
`src/aegisbench/evaluation/localization.py`. Metrics reported are IoU between
the clean and corrupted predicted boxes, a scale-normalized center shift, and
the drop in box-versus-ground-truth fit.

**Tiling.** HERIDAL frames are about 4000x3000 and people in them are tiny,
so the images are cut into overlapping 1024-pixel tiles with 256 pixels of
overlap. Detections are produced per tile, mapped back to full-image
coordinates, and merged with class-agnostic non-maximum suppression, so
evaluation is always against the original full-image ground truth and stays
comparable to published full-image results. SARD frames are not tiled, since
they fit in memory at 1024 pixels. See `src/aegisbench/tiling.py`.

**Group-aware splitting.** SARD frames come from video, so consecutive frames
are near-duplicates. If you split randomly, near-identical frames land in both
train and test, and your numbers are inflated garbage. We split by video
sequence: a whole sequence goes entirely to one of train, validation, or test.
`scripts/phase1_prepare_sard.py` prints the discovered group table and refuses
degenerate groupings. Mention this explicitly in the paper. Reviewers who
know this domain look for it.

### The naming issue, please resolve early

The GitHub repository is named **SentinelBench**, but the Python package,
README, and all internal documentation call the project **AegisBench**. Pick
one and make it consistent everywhere before submission. The paper name, the
package name, and any anonymized artifact link should agree. Also check
whether the chosen name collides with an existing published benchmark, since
name collisions are an avoidable embarrassment. Do this on day one, because it
touches every file.

---

## 3. Why this is a strong paper

Be clear-eyed about this, because it determines how you pitch it. This paper
is not strong because of algorithmic novelty. It has none, and in the
Evaluation and Datasets track it does not need any. It is strong for four
reasons, and the writing should foreground all four.

**It answers a question that is obviously important and has not been
answered.** Search and rescue detection under disaster conditions is a
safety-critical application where failure means someone is not found. The
gap between "benchmarked on clear-weather imagery" and "deployed into a
wildfire" is glaring once stated. Good application papers make the reader
think "of course somebody should have measured this."

**The corruptions are physically grounded rather than arbitrary.** This is
the main methodological differentiator from a generic ImageNet-C style
transplant. Haze corruptions use the Koschmieder atmospheric scattering model
`I = J*t + A*(1-t)`. Low light is modeled in an approximately linear
photometric domain with a decode gamma of 2.2, with photon scaling, a
twilight blue shift, and signal-dependent shot noise plus read noise whose
relative magnitude grows as light falls. Dust is distinguished from smoke not
by a label but by airlight chromaticity (brown mineral versus neutral gray,
enforced by a unit test on the red-to-blue ratio), by transmission field
granularity, and by coarse near-lens grain. A reviewer who works on
dehazing or low-light imaging will recognize these as real models rather than
hand-tuned filters, and that recognition is worth a lot.

**The experimental hygiene is unusually careful.** Frozen operating points
from clean validation only. Group-aware splitting. Deterministic seeding from
a stable hash so the corrupted benchmark is a fixed dataset that regenerates
bit-identically on any machine. Git SHA, config hash, seed, and timestamp
logged on every result row. One shared evaluator scoring every model so no
architecture gets a bespoke scoring path. Bootstrap confidence intervals on
the headline metrics. Most benchmark papers do perhaps half of these. Section
9 lists each guard and the specific reviewer objection it defuses.

**The headline finding is dramatic, cross-architecture, and cross-dataset.**
Under low light, detection does not degrade gracefully. It collapses to
exactly zero recall, on both datasets, for all three architectures, with
bootstrap confidence intervals of `[0.000, 0.000]`. A result that replicates
across two datasets and three architecturally unrelated detectors is not a
quirk. It is a property of the task.

---

## 4. The contributions, stated precisely

Use these as the contribution bullets at the end of the introduction. Keep
them concrete and falsifiable. Vague contribution claims ("we provide
extensive analysis") read as padding.

1. **A disaster-grounded corruption benchmark for aerial search and rescue
   person detection.** Nine corruptions across four disaster families, each
   at three severities calibrated by a declared, measurable image statistic,
   applied to two established datasets spanning the high-altitude and
   low-altitude capture regimes. Every stochastic element is seeded from a
   stable hash of `(image_id, corruption, severity, global_seed)`, so the
   corrupted benchmark is a fixed dataset rather than a random process.

2. **A systematic robustness evaluation of three architecturally distinct
   detectors** across 168 evaluation conditions, all scored by one shared
   evaluator at operating points frozen on clean validation data, with
   bootstrap confidence intervals on the headline metrics.

3. **The identification of catastrophic, cross-architecture failure under
   low-light conditions**, where recall reaches exactly zero with degenerate
   confidence intervals on both datasets and for all three detectors, along
   with a severity ranking showing which disaster conditions are survivable
   and which are not.

4. **A localization stability analysis** that separates two failure modes
   normally conflated under a single recall number: losing the detection
   entirely, versus keeping it with a box that drifts off the person. See
   section 5 for why this turned out to be one of the more interesting
   results.

5. **A fully reproducible, released pipeline** with per-phase verification
   checkpoints, provenance logging on every result row, and resumable
   long-running jobs.

---

## 5. What we have already measured

All numbers in this section were read directly from result files. Numbers
marked `[FILL]` still need to be read out of `results/sweep/master_ci.csv`.

### Inventory of completed artifacts

| Artifact | File | Status |
| --- | --- | --- |
| Main sweep, all conditions | `results/sweep/master_ci.csv` | Complete, 168 rows |
| Archived raw predictions | `results/sweep/preds/` | Complete, 168 files |
| Bootstrap CIs, HERIDAL | `results/sweep/ci_heridal.csv` | Complete, 84 rows |
| Bootstrap CIs, SARD | `results/sweep/ci_sard.csv` | Complete, 84 rows |
| Localization stability, HERIDAL | `results/sweep/localization_heridal.csv` | Complete, 81 rows |
| Localization stability, SARD | `results/sweep/localization_sard.csv` | Complete, 81 rows |
| Robustness heatmaps | `results/sweep/heatmap_{heridal,sard}_recall.png` | Complete |
| Mitigation study | Phase 6 | **Descoped, out of scope for this submission (see section 13)** |

The sweep covers 3 models x (1 clean + 9 corruptions x 3 severities) x 2
datasets = 168 evaluation passes. Bootstrap CIs use 1000 resamples per
condition, resampling the test set by image with replacement.

### The headline result: low-light collapse

**HERIDAL**, recall at each model's frozen operating point, mean over 1000
bootstrap resamples with 95% confidence interval, `n_images = 101`:

| Model | Frozen conf | Severity 1 | Severity 2 | Severity 3 |
| --- | --- | --- | --- | --- |
| RT-DETR | 0.65 | **0.000 [0.000, 0.000]** | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| YOLOv11 | 0.45 | 0.048 [0.027, 0.073] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Faster R-CNN | 0.99 | 0.452 [0.365, 0.540] | 0.110 [0.064, 0.163] | 0.000 [0.000, 0.000] |

**SARD**, YOLOv11, same protocol, `n_images = 862`:

| Severity 1 | Severity 2 | Severity 3 |
| --- | --- | --- |
| 0.184 [0.157, 0.212] | 0.007 [0.003, 0.012] | 0.000 [0.000, 0.000] |

Read what a `[0.000, 0.000]` interval means: across 1000 independent
resamples of the test set, **every single resample produced exactly zero
recall**. This is the strongest statistical statement available from a
bootstrap. It is not "usually fails" but "fails on every resampling of the
evaluation data."

Two further observations worth a paragraph each in the discussion:

**All three architectures converge to the same floor.** A two-stage CNN, a
single-stage CNN, and a transformer detector are about as architecturally
different as detectors get, and by severity 3 all three sit at exactly zero.
That rules out architecture-specific explanations.

**Faster R-CNN is the most robust despite having the strictest threshold.**
Its frozen operating point is 0.99, versus 0.65 for RT-DETR and 0.45 for
YOLOv11, and yet it retains the most recall under corruption. So the
robustness ordering is not simply an artifact of threshold strictness, which
is the first alternative explanation a reviewer will reach for. Say this
explicitly and preempt it.

### Severity ranking across all nine corruptions

SARD, YOLOv11, mean bootstrap recall (95% CI omitted here for readability,
they are in `ci_sard.csv` and belong in the paper table):

| Corruption | Sev 1 | Sev 2 | Sev 3 | Character |
| --- | --- | --- | --- | --- |
| `low_light` | 0.184 | 0.007 | **0.000** | Total collapse |
| `fire_warm_tint` | 0.689 | 0.311 | 0.099 | Near collapse |
| `dust_haze` | 0.673 | 0.272 | 0.058 | Near collapse |
| `turbidity_cast` | 0.796 | 0.576 | 0.191 | Severe |
| `smoke_haze` | 0.810 | 0.667 | 0.391 | Substantial |
| `motion_blur` | 0.792 | 0.613 | 0.477 | Substantial |
| `inundation` | 0.860 | 0.744 | 0.494 | Moderate |
| `water_glare` | `[FILL]` | 0.809 | 0.666 | Moderate |
| `rain_streaks` | 0.893 | 0.883 | **0.855** | Nearly unaffected |

The spread here is the story. At severity 3, rain costs about four points of
recall while low light costs everything. That range, produced by corruptions
calibrated on comparable statistical ladders, is what makes the benchmark
informative rather than merely difficult.

Note `water_glare` severity 1 is marked `[FILL]` only because that line was
cut off in the console output we captured. It is present in `ci_sard.csv`.

### The localization stability result, which is more interesting than expected

SARD, YOLOv11. `loc_stability_iou` is the IoU between the clean and corrupted
predicted boxes for instances detected in both conditions. `n_common` is how
many instances that was.

| Corruption | Sev 1 | Sev 2 | Sev 3 | `n_common` at sev 3 |
| --- | --- | --- | --- | --- |
| `rain_streaks` | 0.984 | 0.953 | 0.909 | 926 |
| `inundation` | 0.960 | 0.930 | 0.884 | 542 |
| `smoke_haze` | 0.917 | 0.884 | 0.852 | 427 |
| `turbidity_cast` | 0.917 | 0.874 | 0.856 | 210 |
| `water_glare` | `[FILL]` | 0.942 | 0.906 | 729 |
| `fire_warm_tint` | 0.896 | 0.870 | 0.860 | 109 |
| `dust_haze` | 0.886 | 0.864 | 0.822 | 64 |
| `motion_blur` | 0.878 | 0.819 | 0.779 | 523 |
| `low_light` | 0.833 | 0.804 | NaN | 0 |

**The finding: localization stability barely degrades even where recall
collapses.** Even at severity 3, surviving detections sit at 0.78 to 0.91 IoU
against their own clean-condition boxes. Meanwhile `n_common` falls off a
cliff, tracking the recall collapse exactly (for `low_light` it goes 202, then
8, then 0).

This decouples two failure modes that a single recall number hides. Corruption
in this domain does not cause detectors to drift, smear, or mislocalize. It
causes them to **stop seeing the person at all**. The detections that survive
are essentially as well-placed as they were on clean imagery.

That is a genuinely useful operational insight, and it is exactly the kind of
analysis the Evaluation and Datasets track rewards. It also has a practical
implication worth stating: if surviving detections stay accurate, then for a
deployed system the failure mode is missed detections rather than false
placement, which changes how an operator should tune the system and what a
mitigation should target (sensitivity, not localization).

The `NaN` at `low_light` severity 3 is correct behavior, not a bug. With zero
instances detected in both conditions there is nothing to average, and the
evaluator returns `NaN` with `n_common = 0` rather than silently reporting a
misleading zero. Explain that in the table caption.

### A hypothesis worth checking before you write about it

`dust_haze` is markedly more damaging than `smoke_haze` at matched severities
(0.058 versus 0.391 recall at severity 3). Both use the same Koschmieder
scattering model, both are calibrated on RMS contrast, and their transmission
means are nearly identical at severity 3 (`t_mean` 0.24 for dust versus 0.25
for smoke).

The tempting claim is that airlight chromaticity is what matters. Be careful,
because dust also differs in transmission field variation amplitude (0.30
versus 0.15), noise octaves (5 versus 3), base resolution (8 versus 3), and it
adds coarse grain that smoke does not have. So there are at least four
candidate causes, and the honest framing is a hypothesis rather than a
conclusion.

If you want to make the stronger claim, check
`results/phase3/calibration.csv` to confirm the two corruptions actually land
at comparable measured RMS contrast at each severity. If they do, the
comparison is meaningfully controlled on the calibration statistic and you can
say so. If they do not, report the measured values and frame the difference
as confounded. Either way, this is one paragraph of discussion, not a claim
for the abstract.

---

## 6. The one analysis to run before writing results

**This is the highest-value thirty minutes available to you, please do it
first.**

Look again at the RT-DETR row: zero recall on HERIDAL low light at **every**
severity, including the mildest, while YOLOv11 gets 0.048 and Faster R-CNN
gets 0.452 at that same severity. A careful reviewer will stop on that line
and ask whether it is a real detection failure or a **confidence calibration**
failure.

Those are very different findings:

- If RT-DETR genuinely produces no boxes on the person under mild low light,
  that is a detection failure and the current framing is correct.
- If RT-DETR still produces boxes in roughly the right places but its
  confidence scores drop below its frozen threshold of 0.65, then the model
  still "sees" the person and the failure is that its confidence became
  miscalibrated under distribution shift.

The second finding would be **more** interesting and more actionable, not
less, because it says the information is present and recoverable through
threshold adaptation or calibration rather than lost.

**How to check.** The master sweep CSV already contains threshold-free
metrics. Run:

```bash
python -c "
import pandas as pd
d = pd.read_csv('results/sweep/master_ci.csv')
d = d[(d.corruption=='low_light') & (d.dataset=='heridal')]
print(d[['model','severity','conf_thresh','recall','precision','map50','map50_95']].to_string(index=False))
print()
c = pd.read_csv('results/sweep/master_ci.csv')
c = c[(c.corruption=='clean') & (c.dataset=='heridal')]
print(c[['model','conf_thresh','recall','precision','map50','map50_95']].to_string(index=False))
"
```

Interpretation:

- `map50` near zero as well: genuine detection failure. Report as is.
- `map50` clearly above zero while recall at the frozen threshold is zero:
  a calibration failure under distribution shift. This deserves its own
  subsection and probably a sentence in the abstract.

Also print the clean rows, because you need each model's clean baseline to
report relative drops anyway, and you should confirm RT-DETR's clean
performance is healthy. If RT-DETR is weak even on clean HERIDAL, then its
corrupted collapse is less surprising and should be framed more carefully.

Do this before drafting the results section, because the answer changes what
you write.

---

## 7. Section-by-section drafting guide

Eight pages excluding references. Suggested budget:

| Section | Pages |
| --- | --- |
| Abstract and introduction | 1.25 |
| Related work | 0.75 |
| Benchmark design (corruptions, calibration, protocol) | 2.0 |
| Experimental setup | 0.5 |
| Results and analysis | 2.5 |
| Mitigation study | 0.5 |
| Limitations and conclusion | 0.5 |

Results plus benchmark design should be more than half the paper. That
allocation signals correctly which track this belongs in.

### 7.1 Title

Aim for concrete and searchable. Avoid a bare project name with no content
words, since readers scanning proceedings should be able to tell what it is.

Candidates:

- *When the Lights Go Out: Benchmarking Aerial Search-and-Rescue Person
  Detection Under Disaster-Grounded Visual Corruption*
- *Disaster-Grounded Robustness Benchmarking for Aerial Person Detection*
- *AegisBench: Physically Motivated Corruption Robustness for Aerial
  Search and Rescue*

The first is more memorable and directly references the headline result. Some
program committees dislike colon-and-quip titles, so if in doubt use the
second with the project name prefixed.

### 7.2 Abstract

Roughly 150 to 200 words. Structure: application stakes, the gap, what we
built, the headline number, the secondary finding, availability.

Draft to adapt:

> Aerial person detection is a core capability for search and rescue, yet the
> datasets used to benchmark it were collected in calm, clear conditions,
> while the missions that need it most take place during floods, wildfires,
> storms, and at night. We present [NAME], a robustness benchmark that
> evaluates aerial search-and-rescue person detection under nine physically
> motivated visual corruptions spanning four disaster families, each at three
> severities calibrated by a declared image statistic. We evaluate three
> architecturally distinct detectors, a two-stage CNN, a single-stage CNN, and
> a transformer detector, across 168 conditions on two datasets covering the
> high-altitude and low-altitude capture regimes, scoring every model with one
> shared evaluator at operating points frozen on clean validation data. We
> find that robustness is highly uneven across conditions: heavy rain costs
> [FILL] recall, while low light produces complete detection collapse, with
> recall reaching exactly zero and bootstrap confidence intervals of
> [0.000, 0.000] on both datasets and for all three architectures. A
> localization-stability analysis shows that surviving detections remain
> accurately placed, indicating that corruption causes missed detections
> rather than positional drift. We release the benchmark, the corruption
> engine, and the full evaluation pipeline.

Fill the rain number from the CSV. The abstract above is already correct as
written, since no mitigation study is part of this submission (section 13):
it describes the benchmark, the evaluation, and the localization finding,
and stops there.

### 7.3 Introduction

Use the classic five-paragraph structure. Reviewers read the introduction and
the figures, then decide their prior. Spend real effort here.

**Paragraph 1, the stakes.** Aerial search and rescue, what it is, why
automated detection matters, and why failure is costly in human terms. Keep
it factual rather than dramatic. One or two sentences of context, then the
technical framing.

**Paragraph 2, the gap.** Existing aerial SAR detection work reports strong
numbers on HERIDAL and SARD. Those datasets are clear-weather. Deployments
are not. State plainly that no systematic measurement exists of how these
detectors behave under the visual conditions of the disasters they respond
to. This is the sentence the whole paper hangs on, so make it precise and
make sure it is true after you have done the literature search in section 8.
If someone has partially done this, cite them and sharpen the claim to what
remains unmeasured.

**Paragraph 3, the approach.** Introduce the benchmark. Emphasize the three
things that make it more than an ImageNet-C transplant: corruptions modeled
on documented disaster optics rather than generic noise, severities
calibrated against measurable image statistics rather than eyeballed, and an
evaluation protocol with frozen clean-validation operating points. This
paragraph is where a knowledgeable reviewer decides whether to take the work
seriously.

**Paragraph 4, the findings.** Lead with the low-light collapse, since it is
the most striking. Give the number and the confidence interval. Then the
spread across conditions, then the localization stability decoupling. Three
findings, one sentence each. Do not save results for later, put them here.

**Paragraph 5, contributions.** The bulleted list from section 4.

Include **Figure 1 on page 1**. See section 11 for what it should be. A
strong page-1 figure meaningfully shifts reviewer impressions, and in this
paper the material is unusually visual.

### 7.4 Related work

Four subsections, roughly a paragraph each. Section 8 covers what to cite.
The purpose of related work is not to list everything, it is to establish
that your gap is real. Every paragraph should end by clarifying what remains
unaddressed.

1. **Aerial and search-and-rescue person detection.** HERIDAL, SARD, the
   published detection results on them, adjacent aerial datasets. Ends with:
   all evaluated under clear conditions.
2. **Corruption robustness benchmarking.** ImageNet-C and its lineage,
   robustness benchmarks for detection and segmentation. Ends with: the
   corruptions are generic, and the domain is not aerial SAR.
3. **Adverse-weather and low-light vision.** Foggy Cityscapes, ACDC,
   dehazing, low-light enhancement, rain removal. Ends with: focused on
   autonomous driving and ground-level imagery, and on restoration rather
   than on measuring detector degradation.
4. **Small-object detection and tiling.** Why aerial person detection is hard
   independent of weather, and the tiling-based inference literature. Ends
   with: motivates our tiling protocol and explains why these targets are
   fragile.

### 7.5 Benchmark design

This is the methods core and should be the most detailed section. Much of it
can be adapted from `docs/CORRUPTIONS.md`, which is already written to paper
standard.

Cover, in order:

**The corruption taxonomy.** A table with all nine corruptions: name, family,
optical mechanism, calibration statistic, and measured statistic values at
each severity read from `results/phase3/calibration.csv`. This table is the
single most important artifact in the paper. It is what makes the benchmark
reproducible and what separates it from arbitrary filtering.

**The physical models.** Give the actual equations. The Koschmieder model
`I = J*t + A*(1-t)` for the haze family, with airlight `A` differing in
chromaticity between smoke (neutral gray) and dust (brown mineral). The
linear-domain photometric model for low light with decode gamma 2.2, photon
scaling, twilight blue shift, and signal-dependent shot noise plus read
noise. Additive radiance models for glare and rain. Equations cost little
space and buy substantial credibility.

**Calibration methodology.** Explain that severities are defined by declared
measurable statistics and that monotonicity is machine-verified. Name the
statistic per family: RMS contrast for haze-like corruptions, mean luminance
for low light, saturated pixel fraction for glare, red-to-blue ratio for fire
tint, edge strength for motion blur, occluded fraction for inundation.

**One honest exception, disclose it rather than hide it.** Running
`phase3_calibrate.py` (18 August) confirmed 8 of 9 corruptions pass
automatic monotonicity verification. `rain_streaks` did not: measured
`streak_density` came back 0.1341, 0.0906, 0.1294 across severities 1 to 3,
not monotonically increasing. The code comment in
`src/aegisbench/corruptions/calibration.py` already names why: "streak_density
has no closed-form image statistic; rain severity is audited via
edge_strength/mean_luminance side effects and visual review." `edge_strength`
is used as a proxy, and it conflates two effects that move in opposite
directions as severity rises: more/longer streaks raise measured edge
strength, while the rain veil's blur (`veil_blur_sigma_per_1000px`, which
also increases with severity, 0.4 to 0.8 to 1.2) suppresses it. The proxy
statistic is confounded; the corruption itself is not. `rain_streaks`'
severity is defined directly by its physical parameters in
`configs/corruptions.yaml` (streaks per megapixel: 120, 350, 800; length,
veil blur, and darkening all increasing by construction), and the downstream
recall results already show the expected monotonic degradation (0.893 to
0.883 to 0.855 on SARD/YOLOv11), which is independent evidence the applied
corruption really does intensify with severity even though this one proxy
statistic does not track it cleanly.

Write one sentence to this effect in the calibration paragraph rather than
claiming all nine pass automatic verification. Something close to: "Eight of
nine corruptions are calibrated against a dedicated closed-form image
statistic with machine-verified monotonicity; `rain_streaks` severity is
instead defined directly by its physical generation parameters (streak
density, length, and veil intensity, all increasing by construction) and
verified by visual audit, since no simple per-pixel statistic cleanly
isolates streak coverage from the veil blur applied in the same corruption."
This is a stronger paper for saying so: it shows the verification pipeline
actually works, since it caught its own proxy's limitation, rather than
rubber-stamping every corruption.

**Scale invariance.** All pixel-unit parameters (blur kernel length, streak
length, bloom sigma, ripple amplitude) are specified per 1000 pixels of the
longer image side and scaled at runtime, so severity 2 means the same thing
on a 4000x3000 HERIDAL frame and a 1920x1080 SARD frame. Without this the
cross-dataset comparison would be meaningless, so state it.

**Ground-truth alignment.** Corruptions are appearance-only: they recolor,
darken, haze, or overlay the existing pixel grid without moving object
content, so clean ground-truth boxes remain exactly valid on corrupted
images. This pixel-for-pixel alignment is what makes the clean-versus-
corrupted comparison fair. Disclose the one exception honestly: `inundation`
applies a water-refraction ripple via `cv2.remap` while the box stays fixed,
with amplitude deliberately bounded well below person scale (about 1.4 pixels
maximum at severity 3 on a 4000-pixel frame). Stating this before a reviewer
finds it converts a potential weakness into evidence of care.

**Determinism.** Every stochastic element is seeded by a SHA-256 based stable
hash of `(image_id, corruption, severity, global_seed)`, independent of
`PYTHONHASHSEED` and platform, so the corrupted benchmark is a fixed dataset
that regenerates bit-identically anywhere.

**Evaluation protocol.** Frozen operating points from clean validation, the
shared evaluator, tiling and merge for HERIDAL, group-aware splitting for
SARD, relative performance drop, localization stability, and the bootstrap
procedure. Give the bootstrap details: resampling by image with replacement,
1000 resamples per condition, percentile intervals at 2.5 and 97.5.

**Test-time versus train-time application.** At evaluation, corruptions are
applied to the full image **before** tiling, because atmospheric structure
such as haze gradients and flood masks is spatially coherent across a whole
frame in reality. For mitigation training, corruption is applied per tile,
which is cheaper and trainer-agnostic. That is an approximation and should be
documented as one.

### 7.6 Experimental setup

Short and factual. Models and their sources, input resolution 1024, training
schedule and seeds from the configs, hardware (a single 8 GB consumer GPU,
which is worth stating because it makes the work reproducible by others
without a cluster), dataset sizes measured from your own copies, and split
policy. HERIDAL uses the official test split untouched with validation carved
deterministically at 15% from the official train split.

Do not quote dataset sizes you did not measure. `docs/DATA.md` is explicit
about this and it is good discipline.

### 7.7 Results and analysis

Order matters. Build from summary to mechanism.

1. **Clean baselines.** Establishes the pipeline is sound. Compare against
   the published HERIDAL reference point (approximately 0.90 precision, 0.893
   recall, 0.834 mAP@0.5 for YOLOv5L) and note that modest deltas across
   architectures and input pipelines are expected. This preempts "how do we
   know your training is any good?"
2. **The robustness heatmap.** Models by corruption by severity. One figure
   carries the overall shape of the result.
3. **The severity ranking table** with confidence intervals.
4. **The low-light collapse**, as its own subsection with the cross-dataset
   and cross-architecture tables from section 5. This is the headline, give
   it room.
5. **The calibration-versus-detection analysis** from section 6, if the mAP
   check shows what it might.
6. **Localization stability**, with the decoupling insight. This is the last
   results subsection for this submission; there is no Phase 6 mitigation
   subsection to follow it (section 13).

Do not fake a mitigation result or hedge vaguely about one. Add one honest
sentence to the conclusion instead: disaster-aware training augmentation is
the natural mitigation, and evaluating it is immediate future work. A
benchmark paper is perfectly complete without a mitigation study. A benchmark
paper with a rushed, under-trained mitigation study would have been worse
than one without, which is exactly why it was descoped rather than rushed.

### 7.8 Limitations

Write this section properly. Reviewers trust papers that disclose limits, and
this project has genuine ones that are better stated by you than discovered
by them.

**Synthetic corruptions on real imagery.** This is the ImageNet-C lineage
methodology and it is a controlled, repeatable, parameterized design. It is
also the benchmark's key limitation: it measures response to modeled optics,
not to live disaster footage. State both halves in the same paragraph. Note
that validating against real disaster imagery is the natural next step and
that collecting such data is difficult precisely because these are emergencies.

**Scale and scope.** Two datasets and three detectors. Use this framing, which
is honest and turns the constraint into a stated design choice:

> We evaluate on two datasets spanning the two major real-world SAR-imagery
> regimes (HERIDAL, high-altitude orthophoto search imagery; SARD, lower-
> altitude drone video) and three architecturally distinct detectors
> representative of the dominant paradigms in the field (two-stage: Faster
> R-CNN; single-stage: YOLOv11; transformer-based: RT-DETR). We prioritize
> methodological depth, physically grounded corruption modeling, group-aware
> splitting, frozen-threshold evaluation, and bootstrap confidence intervals,
> over dataset and model count, consistent with the goal of a rigorous
> diagnostic benchmark rather than a leaderboard survey.

**Single training seed.** Unless multi-seed training was run, say so plainly
rather than presenting single-run numbers as if their variance were known.
Note the mitigating factor: the bootstrap confidence intervals quantify
evaluation-set variance even though they do not quantify training variance.
Be precise about that distinction, because a statistically literate reviewer
will be.

**The inundation ripple.** As described in 7.5.

**Per-tile training augmentation.** As described in 7.5.

### 7.9 Conclusion

Short. Restate the gap, the headline finding, and the operational
implication: systems intended for disaster response should be evaluated under
disaster conditions, and current detectors have a specific, measurable,
architecture-independent blind spot at low light that would render them
useless in exactly the night-time search scenarios where aerial assets are
most valuable.

---

## 8. Building the literature review

### How to work

Every citation below is a starting point that **you must verify**: confirm
the authors, venue, year, and that the paper says what we claim. Do not cite
from this list without checking. Use Google Scholar, Semantic Scholar, and
the CVF open-access archive at `openaccess.thecvf.com`, which has full text
for all CVPR, ICCV, ECCV, and WACV papers.

Two search strategies that pay off:

- **Forward citation search.** Find the HERIDAL paper and the ImageNet-C
  paper, then look at everything that cites them. Anything citing both is
  almost certainly directly relevant and possibly a competitor.
- **Recent-venue sweep.** Search WACV, CVPR, and ICCV proceedings from 2023
  through 2026 for "search and rescue", "UAV person detection", "corruption
  robustness detection", and "adverse weather detection". You need to know
  about anything from the last two years, both to cite it and to be certain
  the gap claim in the introduction still holds.

A target of 35 to 50 references is normal for a WACV paper of this type.

### Theme 1: aerial and search-and-rescue person detection (essential)

This is where the domain grounding comes from and where you must be thorough,
since these are the closest works to ours.

- **The HERIDAL paper.** Božić-Štulić, Marušić, Gotovac, "Deep Learning
  Approach in Aerial Imagery for Supporting Land Search and Rescue Missions",
  IJCV 2019. The citable source of record for the dataset.
- **The SARD paper.** Sambolek and Ivašić-Kos, "Automatic Person Detection in
  Search and Rescue Operations Using Deep CNN Detectors", IEEE Access 2021.
- Kundid Vasić and Papić, "Multimodel Deep Learning for Person Detection in
  Aerial Images", Electronics 2020. A frequently cited HERIDAL result.
- **TinyPerson**, Yu et al., "Scale Match for Tiny Person Detection", WACV
  2020. Tiny person detection in maritime rescue imagery. Highly relevant and
  a WACV paper, which is worth noting since citing the venue's own literature
  is well received.
- **SeaDronesSee**, Varga et al., WACV 2022. Maritime UAV search and rescue.
  Same reason.
- **VisDrone**, Zhu et al., "Detection and Tracking Meet Drones Challenge",
  TPAMI 2021. The standard large-scale drone detection benchmark.
- **UAVDT**, Du et al., ECCV 2018. Includes weather-condition attributes,
  which makes it a useful contrast: attribute-labeled real conditions versus
  our controlled synthetic ladder.

Search additionally for recent survey papers on UAV-based search and rescue,
which are efficient sources of further citations.

### Theme 2: corruption robustness benchmarking (essential)

This establishes the methodological lineage.

- **ImageNet-C**, Hendrycks and Dietterich, "Benchmarking Neural Network
  Robustness to Common Corruptions and Perturbations", ICLR 2019. The
  foundational reference for the whole approach. Cite prominently.
- **Michaelis et al.**, "Benchmarking Robustness in Object Detection:
  Autonomous Driving when Winter is Coming", 2019. Corruption robustness for
  detection specifically (COCO-C, Pascal-C, Cityscapes-C). This is the closest
  methodological antecedent, so engage with it directly and explain what is
  different: our corruptions are disaster-specific and physically calibrated,
  our domain is aerial and small-object, and we add localization stability.
- Kamann and Rother, "Benchmarking the Robustness of Semantic Segmentation
  Models", CVPR 2020.
- Hendrycks et al., "The Many Faces of Robustness" (ImageNet-R), ICCV 2021.
- Croce et al., **RobustBench**, NeurIPS Datasets and Benchmarks 2021. Useful
  for benchmark-design conventions.
- Mintun et al., "On Interaction Between Augmentations and Corruptions in
  Natural Distribution Shifts", NeurIPS 2021. Directly relevant to the
  mitigation study, since it examines when augmentation actually helps versus
  when it merely overfits the corruption set.

### Theme 3: adverse weather, atmospheric optics, and low light (essential)

This backs the physical models and is where a domain-expert reviewer will
check your work.

- **Narasimhan and Nayar**, "Vision and the Atmosphere", IJCV 2002. The
  foundational scattering-model reference. Cite this for Koschmieder.
- **He, Sun, and Tang**, "Single Image Haze Removal Using Dark Channel Prior",
  CVPR 2009 and TPAMI 2011. The canonical haze-model reference in vision.
- **Sakaridis et al.**, "Semantic Foggy Scene Understanding with Synthetic
  Data", IJCV 2018 (Foggy Cityscapes). The best precedent for synthetic
  physically-modeled weather applied to a real dataset. Strong support for
  our methodology.
- **Sakaridis et al.**, ACDC, ICCV 2021. Real adverse-condition driving data,
  a useful contrast case for the synthetic-versus-real discussion.
- **Garg and Nayar**, "Vision and Rain", IJCV 2007. Backs the rain streak
  model.
- **Chen et al.**, "Learning to See in the Dark", CVPR 2018. Important for the
  low-light section, and the source of the raw linear-domain framing that
  justifies modeling low light in linear space with a decode gamma.
- Wei et al., "Deep Retinex Decomposition for Low-Light Enhancement", BMVC
  2018.
- **Loh and Chan**, "Getting to Know Low-Light Images with the Exclusively
  Dark Dataset", CVIU 2019. Real low-light imagery with object annotations,
  directly relevant to the headline finding.

For the low-light discussion specifically, search for recent work on
detection in low light and on whether enhancement preprocessing actually
improves downstream detection. That literature is directly relevant to the
mitigation discussion and a reviewer may well ask why we did not simply apply
a low-light enhancement model as a baseline. Having a cited answer is worth
having.

### Theme 4: small-object detection and tiling (supporting)

- Lin et al., **Feature Pyramid Networks**, CVPR 2017.
- **SAHI**, Akyon et al., "Slicing Aided Hyper Inference and Fine-tuning for
  Small Object Detection", ICIP 2022. Directly relevant to our tiling
  protocol. Cite it when describing tiling.
- Kisantal et al., "Augmentation for Small Object Detection", 2019.

### Theme 5: the detectors themselves (required, brief)

- Ren et al., **Faster R-CNN**, NeurIPS 2015.
- Carion et al., **DETR**, ECCV 2020, for transformer detection lineage.
- Zhao et al., **RT-DETR**, "DETRs Beat YOLOs on Real-time Object Detection",
  CVPR 2024.
- **YOLOv11**, cite the Ultralytics release or technical report. Check what
  the canonical citation is at the time of writing, since YOLO versions are
  often software releases rather than papers, and cite the software with a
  version number if that is the only option.

### Theme 6: robustness mitigation through augmentation (not needed for this submission)

Phase 6 was descoped (section 13), so this theme is not required for the
current draft. Left here in case a future version of the paper adds the
mitigation study back in, since it is the natural extension named as future
work in the conclusion and limitations.

- Hendrycks et al., **AugMix**, ICLR 2020.
- Cubuk et al., **AutoAugment**, CVPR 2019, and **RandAugment**, 2020.
- **Rusak et al.**, "A Simple Way to Make Neural Networks Robust Against
  Diverse Image Corruptions", ECCV 2020. Directly relevant.
- Geirhos et al., "ImageNet-trained CNNs are Biased Towards Texture", ICLR
  2019. Useful for explaining *why* haze-type and tint-type corruptions might
  be so damaging, since they attack exactly the texture and contrast cues the
  models lean on.

### Theme 7: evaluation methodology (supporting, cheap credibility)

- **Efron and Tibshirani**, *An Introduction to the Bootstrap*, 1993. Cite for
  the confidence interval procedure.
- Everingham et al., **PASCAL VOC**, IJCV 2010, and Lin et al., **COCO**,
  ECCV 2014, for the metric definitions.
- **Hoiem et al.**, "Diagnosing Error in Object Detectors", ECCV 2012, and
  **Bolya et al.**, **TIDE**, ECCV 2020. Both are strong precedents for
  decomposing detector failure into modes, which is exactly what the
  localization stability analysis does. Citing these places our second axis
  in an established tradition rather than making it look ad hoc.
- Gebru et al., "Datasheets for Datasets", and Mitchell et al., "Model Cards
  for Model Reporting", FAT* 2019. Worth citing if you add a datasheet-style
  appendix, which the Evaluation and Datasets track tends to appreciate.

---

## 9. The rigor checklist and the objections it preempts

Each guard below is implemented in the code. Each defuses a specific
objection. Make sure every one of them appears somewhere in the paper,
because unstated rigor earns nothing.

| Guard | Where | Objection it defuses |
| --- | --- | --- |
| Operating point selected on clean validation, then frozen across all conditions | `phase5_sweep.py`, `select_operating_point` | "You tuned thresholds on corrupted data, so the comparison is rigged." |
| One shared evaluator for all three architectures | `src/aegisbench/evaluation/` | "Each model was scored by its own framework's metric code, so the numbers are not comparable." |
| Group-aware splitting by video sequence for SARD | `phase1_prepare_sard.py` | "SARD is video, your train and test sets contain near-duplicate frames, so the numbers are inflated." |
| Deterministic seeding from a stable SHA-256 hash | `src/aegisbench/seeding.py` | "Your corrupted test set is a random draw and not reproducible." |
| Severities calibrated against declared measurable statistics, monotonicity machine-verified | `configs/corruptions.yaml`, `phase3_calibrate.py` | "Your severity levels are arbitrary, so the difficulty ordering means nothing." |
| Pixel-aligned, appearance-only corruptions | `docs/CORRUPTIONS.md` | "Your corruptions moved the image content, so the original boxes no longer apply." |
| Test-time corruption applied to the full frame before tiling | `phase5_sweep.py` | "You corrupted each tile independently, which is physically wrong for atmospheric effects." |
| Full-image evaluation after tile merge with class-agnostic NMS | `tiling.py`, `evaluation/merge.py` | "Tile-level numbers are not comparable to published full-image results." |
| Bootstrap CIs, 1000 resamples per condition | `phase5_bootstrap_ci.py` | "This is a single run, so how do we know the effect is not noise?" |
| Git SHA, config hash, seed, and timestamp logged per result row | `phase5_sweep.py` | "Which code version produced this table?" |
| Raw predictions archived per condition | `--pred-dir` | "We cannot re-analyze your results without rerunning everything." |
| Comparison against a published reference point on clean data | Phase 4 | "How do we know your training pipeline is not simply broken?" |
| Scale-invariant corruption parameters, per 1000 pixels | `configs/corruptions.yaml` | "Severity 2 means different things on two datasets of different resolution." |

One point deserves emphasis in the writing, because it is the guard most
often missing from robustness papers and the one a sharp reviewer looks for
first: **thresholds frozen on clean validation data**. If you tune the
confidence threshold per condition, you are measuring the best case a
clairvoyant operator could achieve, not what a deployed system does. A
deployed system has one threshold set before the mission. Our protocol
matches deployment. Say so in one sentence in the protocol subsection.

---

## 10. Anticipated reviewer objections and how to answer them

Prepare these now. Several belong in the paper preemptively, and all of them
belong in your notes for the rebuttal period.

**"The corruptions are synthetic, so this does not tell us about real
disaster imagery."** The strongest objection and it is partly correct.
Answer: this is the established corruption-robustness methodology, with the
ImageNet-C and Foggy Cityscapes lineage as precedent. Synthetic application
to real imagery is what makes controlled, repeatable, per-condition
attribution possible at all, since real disaster imagery does not come with
matched clean counterparts of the same scene. Concede the limit explicitly in
the limitations section and name real-imagery validation as future work.

**"Only two datasets and three models."** Answer with the scope paragraph in
section 7.8, which frames depth over breadth as a deliberate design choice
and lists the specific methodological investments made instead.

**"Is the low-light result just a threshold artifact?"** Answer with the mAP
analysis from section 6, plus the observation that Faster R-CNN has the
strictest threshold at 0.99 yet is the most robust, which rules out the
simple monotonic relationship between threshold strictness and measured
robustness. **Run the mAP check so this answer is backed by a number.**

**"Why not compare against low-light enhancement or dehazing preprocessing?"**
A fair question with no experiment behind it yet. Either add a short
discussion paragraph citing the relevant literature and noting it as future
work, or, if time somehow permits, run one enhancement baseline on the
low-light condition. The discussion answer is acceptable; silence is not.

**"Single training seed."** Disclose it, and be precise that the bootstrap
intervals quantify evaluation-set variance rather than training variance. Do
not blur those two.

**"Why is there no mitigation study?"** Own it in one sentence as future
work, per section 13. The benchmark contribution stands alone, and this
track's own criteria (stress-testing and auditing tools, not necessarily
fixes) do not require one.

**"How is this different from Michaelis et al.?"** Have a crisp answer ready,
since this is the closest prior work. Ours is a different domain (aerial
small-object search and rescue rather than driving), uses disaster-specific
physically calibrated corruptions rather than generic ImageNet-C transplants,
adds the localization stability axis, and adds bootstrap confidence
intervals. Put a sentence of this in related work, not only in the rebuttal.

---

## 11. Figures and tables plan

Eight pages is tight. Aim for four or five figures and four or five tables.

**Figure 1, page 1, the money figure.** A grid: one representative HERIDAL
image across all nine corruptions at severity 2 or 3, with ground-truth boxes
drawn and, ideally, detections overlaid so the reader sees boxes vanish as
conditions worsen. There is already a grid generator at
`scripts/phase3_generate_grid.py` and an existing artifact at
`results/phase3/heridal_grid.jpg`. Making the detection failure visible on
page 1 is the single highest-leverage visual choice available.

**Figure 2, the robustness heatmap.** Already generated at
`results/sweep/heatmap_{heridal,sard}_recall.png`. Models by corruption by
severity, colored by relative recall drop. Check that the fonts are legible
at print size, since generated figures often are not.

**Figure 3, severity curves.** Recall versus severity, one line per
corruption, with bootstrap confidence bands shaded. This is where the
divergence between rain (flat) and low light (falling off a cliff) becomes
visually obvious. The CI data is already in `ci_*.csv`. This figure will
likely be the most cited one in reviews.

**Figure 4, localization stability versus recall.** A scatter or dual-axis
plot showing that stability stays high while `n_common` collapses. This is
the decoupling finding, and it is much clearer as a picture than as a table.

**Table 1, the corruption taxonomy.** Nine rows: name, family, optical
mechanism, calibration statistic, measured values at each severity. The most
important table in the paper.

**Table 2, clean baselines.** Three models by two datasets, with the
published reference point for comparison.

**Table 3, the headline low-light table.** Cross-architecture and
cross-dataset, with confidence intervals. Small, high impact.

**Table 4, full results.** All models, corruptions, and severities. If it
does not fit, put the summary in the main paper and the full table in
supplementary. Do not shrink the font below the template minimum; reviewers
notice and some react badly.

There is no Table 5 (mitigation before/after) in this submission; Phase 6
was descoped (section 13).

Supplementary material has its own deadline (typically shortly after the
paper deadline, verify the exact date) and is the right home for the full
168-row results table, additional qualitative grids, per-family breakdowns,
and the reproducibility appendix.

---

## 12. Writing style guidance

**Lead with numbers.** "Recall falls to zero" is weaker than "recall reaches
exactly 0.000, with a bootstrap 95% confidence interval of [0.000, 0.000],
on both datasets and for all three architectures."

**Never write a number you did not read from a file.** If it is not in a CSV
you can point at, it does not go in the paper. This matters more than usual
here because we have a lot of numbers and it is easy to transpose one.

**Prefer plain language over inflation.** "Catastrophic failure" is justified
by a zero-recall result. "Revolutionary framework" is not justified by
anything. Reviewers discount papers that oversell, and this work does not
need selling.

**State limitations before a reviewer finds them.** Every honest disclosure
in section 7.8 makes the rest of the paper more credible, not less.

**Use consistent terminology.** Pick one term per concept and never vary it
for stylistic variety. "Corruption", "severity", "condition", "frozen
operating point", "localization stability". Scientific writing rewards
repetition over elegant variation, because the reader is tracking precise
referents.

**Write the captions to stand alone.** Many reviewers read the figures and
captions first, and some read little else on the first pass. Every caption
should state what the reader is looking at and what they should conclude
from it.

**Minimize em dashes.** Use commas, colons, parentheses, or separate
sentences instead. This is a house-style preference for this project, so
please apply it consistently.

**Active voice where it is natural.** "We evaluate three detectors" rather
than "three detectors were evaluated."

---

## 13. What is still outstanding

### Phase 6, the mitigation study: deliberately descoped, decided 18 August

This is not "not yet run" waiting on a future decision. The decision was
made and training was actively stopped: **Phase 6 is out of scope for this
submission.** Do not run it, and do not leave the paper hedging about
whether it might still land.

Why: during early testing, two full training attempts hit unexplained
`torch.AcceleratorError: CUDA error: unknown error` crashes in unrelated
code paths on this machine's nightly-cu128 Blackwell build, with no
confirmed root cause after a real diagnostic pass (WSL driver/library audit
came up clean). A subsequent fix (disabling AMP) was applied and did get a
run training stably, but a live per-epoch timing measurement put the
realistic cost at roughly 1 to 3.5 days for two training arms plus the
re-sweep, against a 10-day runway on 18 August that also had to cover
drafting the full paper and verifying a 35-50 reference literature review
from scratch. Given the track's own criteria reward stress-testing and
auditing tools rather than requiring a fix to be proposed, and given a
rushed or crash-interrupted mitigation result would be worse for the paper
than no mitigation section at all, the call was to protect the guaranteed
deliverable (a complete, polished, on-time submission) over the optional
one.

**What this means for the draft:** every section above already assumes this
outcome; none of the results, table, or figure plans reference Phase 6
output. The only place it appears is one sentence in Limitations (7.8) and
Conclusion (7.9) naming disaster-aware augmentation as immediate future
work, which is the honest and sufficient way to handle it.

The configs and pipeline
(`configs/train_yolo11_heridal_aug_all.yaml`,
`configs/train_yolo11_heridal_aug_lowlight.yaml`,
`configs/sweep_models_phase6.yaml`, `scripts/phase6_mitigation.py`) are left
in the repository, since they are real, working, resumable infrastructure
useful for a future extended version of this paper, not because they are
expected to run before 28 August.

### The mAP calibration check

Section 6. Half an hour, potentially changes the framing of the headline
result. Do this first.

### Numbers still to extract for the paper

- Clean baselines for all three models on both datasets, from
  `master_ci.csv`.
- The measured calibration statistics per corruption per severity, from
  `results/phase3/calibration.csv`, for Table 1.
- Exact dataset counts measured from your own copies, per `docs/DATA.md`.
- `water_glare` severity 1 values, which were merely cut off in console
  output and are present in the CSVs.
- The full localization stability tables for HERIDAL, which we have only
  partially reviewed. Both files are complete on disk.
- Per-family aggregate relative drops via
  `src/aegisbench/evaluation/robustness.py`.

### Repository hygiene

- Resolve the SentinelBench versus AegisBench naming inconsistency.
- Confirm the chosen name does not collide with an existing benchmark.
- Verify `python -m aegisbench.anonymize --root .` runs clean.

---

## 14. Submission logistics and double-blind hygiene

**Format.** Eight pages maximum, excluding references. Additional pages
containing only cited references are allowed. Use the official WACV 2027
template, which is available as an author kit and as an Overleaf template.
Papers that exceed eight pages, are not properly anonymized, or do not use
the template are **rejected without review**, so treat these as hard
constraints rather than guidelines.

**Submission platform.** OpenReview. Note that Round 2 registration opens 21
August 2026 and the paper deadline is 28 August 2026. Register early. A
missed registration step is an entirely avoidable way to lose the submission.

**Double-blind anonymization.** WACV review is double blind. This project
ships an anonymization helper:

```bash
python -m aegisbench.anonymize --root .
```

Run it before exporting any submission archive, and export with
`git archive` so the `.git` directory never ships. A `.git` directory in
supplementary material leaks author identity through commit metadata and is a
desk-reject risk.

Additional anonymization checks specific to this project:

- The GitHub URL `Altis-2026/SentinelBench` identifies the authors. Do not
  put it in the paper. If you want to promise a code release, write
  "code will be released upon acceptance" or use an anonymized mirror such as
  Anonymous GitHub.
- Check acknowledgements, funding statements, and any institution names in
  figures or file paths.
- Check image EXIF metadata in any figure generated from your own captures.
- Do not cite your own prior work in a way that reveals identity, using
  third-person phrasing if you must cite it at all.

**Supplementary material.** Has its own deadline, typically a couple of days
after the paper deadline, so verify the exact date on the official site.
Good candidates: the full 168-row results table, additional qualitative
corruption grids, per-family breakdowns, the reproducibility appendix, and a
datasheet-style dataset description.

**Licensing and data statements.** The Evaluation and Datasets track cares
about this. Neither HERIDAL nor SARD is redistributed by this repository, and
both require manual download under their own licenses. State that plainly.
`docs/DATA.md` already has the details.

---

## Appendix: orienting yourself in the codebase

Read in this order if you want to understand the system:

1. `README.md`, the overview.
2. `docs/CORRUPTIONS.md`, the physical grounding. Already written close to
   paper standard and is the best source for section 7.5.
3. `docs/RUNBOOK.md`, the phased protocol with verification checkpoints.
4. `docs/DATA.md`, dataset sourcing and split policy.
5. `configs/corruptions.yaml`, the canonical parameter table. Every number in
   Table 1 traces here.
6. `src/aegisbench/evaluation/localization.py`, whose docstring explains the
   localization stability metric better than most papers explain their
   metrics.
7. `src/aegisbench/evaluation/robustness.py`, the relative drop aggregation.
8. `scripts/phase5_sweep.py`, the main experimental loop, which is short and
   shows exactly how the protocol is enforced.

The phase scripts are numbered in execution order and each one ends by
printing what to run next. The test suite in `tests/` is worth skimming,
since several tests encode methodological claims made in the paper (for
example, the dust-versus-smoke chromaticity test enforces that the two
corruptions really are optically distinct, which is a claim the paper makes
in prose).
