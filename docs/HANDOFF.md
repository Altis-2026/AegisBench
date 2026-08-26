# AegisBench: paper handoff

Self-contained context for continuing work on the paper in a fresh
session, including one with an assistant that has no history of this
project. Paste or attach this file, plus the two it points to, and the
assistant has everything needed to help with LaTeX and prose.

---

## 1. What the paper is

**AegisBench: A Corruption-Robustness Benchmark for Aerial Human
Detection.** A benchmark and analysis paper. It applies nine physically
motivated visual corruptions, spanning four disaster families at three
calibrated severities each, to two established aerial search-and-rescue
person-detection datasets, and evaluates three architecturally distinct
detectors across all 168 resulting conditions.

The headline finding: robustness is highly uneven. Heavy rain costs about
3.6 points of recall at the highest severity, while low light drives
recall to exactly zero on both datasets and for all three architectures,
with bootstrap intervals degenerate at zero, confirmed
threshold-independently by mAP. A second axis, localization stability,
shows that surviving detections stay accurately placed, so the failure
mode is detectors ceasing to find people rather than mislocalizing them.

The paper proposes no new architecture and no mitigation. That is
appropriate for the target track and should not be treated as a gap.

## 2. Venue and hard requirements

**WACV 2027, Evaluations and Datasets track.** This track is new for
2027 and explicitly welcomes "analysis of strengths, limitations, or
failure modes of existing benchmarks." Reviewers assess whether the work
advances a sub-area and whether the benchmarking is scientifically
correct, **not** whether it proposes a novel algorithm.

| Requirement | Status |
| --- | --- |
| 8 pages max, excluding references | verified |
| Official WACV 2027 template required | verified |
| Double-blind anonymization required | verified |
| Over-length, non-template, or non-anonymous papers are **rejected without review** | verified |
| Paper deadline 28 Aug 2026 (AoE) | verified |
| Supplementary deadline 30 Aug 2026, PDF or ZIP, 200 MB max | verified |
| Submission via OpenReview | verified |

Full detail, including which requirements could not be verified against
an official source, is in `docs/SUBMISSION.md`.

## 3. Current status

Working draft is a Google Doc, exported to PDF at 30 pages of
single-column text. That is roughly 2 to 2.5 times the final two-column
budget, so cutting is required, not optional (see section 6).

| Section | Status |
| --- | --- |
| Abstract | written |
| Introduction paras 1 to 4 | **not written** |
| Introduction para 5 (contributions) | written |
| Related Work 1, aerial and SAR person detection | **not written, highest priority** |
| Related Work 2, corruption robustness | written |
| Related Work 3, adverse weather and low light | written, needs the largest cut |
| Related Work 4, small objects and tiling | written, trim FPN background |
| Benchmark Design | written, one correction outstanding |
| Evaluation Protocol | written, one duplicated sentence to delete |
| Experimental Setup | drafted with bracketed placeholders to fill |
| Ethical Considerations | written |
| Results | **not written**, all numbers measured and ready |
| Limitations | **not written**, full prose drafted in the writing pack |
| Conclusion | **not written** |
| Tables 1 to 7 | **all values measured**, LaTeX ready in `paper/aegisbench_body.tex` |
| Figures 1 to 5 | code written and tested; run the generators |
| Bibliography | 19 entries verified, **6 mandatory citations missing** |

## 4. Defects to fix in the existing draft

Ordered by severity. The first is a factual error about the paper's own
method and must be fixed.

### 4.1 The Koschmieder attribution is wrong (still present)

Benchmark Design, Physical Models currently says the three corruptions
built on the Koschmieder scattering model are `smoke_haze`, `dust_haze`,
and `turbidity_cast`.

Verified against the source: only `corruptions/wildfire.py` and
`corruptions/dust.py` call `koschmieder()`. The three are **`smoke_haze`,
`fire_warm_tint`, and `dust_haze`**. `turbidity_cast` lives in
`corruptions/flood.py`, which never imports it; it is an alpha blend
toward a mud chromaticity followed by contrast compression, with no
transmission map and no airlight term.

Corrected replacement text is in `docs/DRAFT_REVIEW.md` section 1.1. The
fix strengthens the section: it shows the taxonomy distinguishes genuinely
different optical processes instead of applying one model everywhere.

### 4.2 Duplicated sentence

In Evaluation Protocol, Tiling and Full-Image Evaluation, the sentence
beginning "Applying one uniform inference pipeline across both capture
regimes removes tiling itself as a confound" appears **twice**
consecutively. Delete one.

### 4.3 `rain_streaks` statistic wording

The text says "Its calibration statistic, edge strength, is intended as a
proxy for streak density." The declared statistic in
`configs/corruptions.yaml` is `streak_density`, and `edge_strength` is
the substituted implementation because streak density has no closed-form
image measure. Table 1 will list "streaks/MP", so the prose should match:
"Its declared statistic, streak density, has no closed-form image
measure, so edge strength is substituted as a proxy."

### 4.4 Six mandatory citations missing

The bibliography has 19 verified entries but is missing the papers the
work cannot ship without:

- **HERIDAL**: Božić-Štulić, Marušić, Gotovac, IJCV 127(9):1256-1278, 2019
- **SARD**: Sambolek and Ivašić-Kos, IEEE Access, 2021
- **Faster R-CNN**: Ren, He, Girshick, Sun, NeurIPS 2015
- **RT-DETR**: Zhao et al., CVPR 2024
- **YOLOv11**: cite the Ultralytics release with a version number
- **Efron and Tibshirani**, *An Introduction to the Bootstrap*, 1993

Strongly recommended additions: SeaDronesSee (Varga et al., WACV 2022),
and Hoiem et al. ECCV 2012 or TIDE (Bolya et al., ECCV 2020) to place the
localization-stability axis in an established failure-decomposition
tradition.

### 4.5 Working artifacts to strip

The draft carries headings that must not ship: "Claude Bib File",
"Related Work, subsection 3:" style working labels, "Tab 3", a duplicated
title block at the end, and a bare WACV URL. Replace with real section
headings.

### 4.6 One claim in Related Work to verify

The draft says WRRT-DETR built AWOD "around synthetic fog, flare, and low
light." Published descriptions say roughly 20,000 images *captured under*
those conditions, which reads as real imagery. The bibliography entry
itself is correct. Open the paper and confirm before keeping the word
"synthetic"; if AWOD is real capture, the paragraph's closing argument
does not apply to that work in the way the draft claims. See
`docs/DRAFT_REVIEW.md` section 2.

## 5. Target structure and page budget

Eight pages excluding references. Results plus Benchmark Design should be
more than half the paper: that allocation is what signals the correct
track.

| Section | Pages |
| --- | --- |
| Abstract and Introduction | 1.25 |
| Related Work (4 subsections) | 0.75 |
| Benchmark Design, including Table 1 and Fig. 2 | 2.0 |
| Evaluation Protocol | included above |
| Experimental Setup | 0.5 |
| Results and analysis | 3.0 |
| Limitations and Conclusion | 0.5 |

## 6. The cutting problem

This is the largest structural risk. Related Work subsections 2 to 4 run
roughly 1,800 words. At about 950 words per two-column page that is
around 1.9 pages, and subsection 1 is not yet written. The budget is
0.75 pages total. Benchmark Design is also running roughly 0.6 pages
long.

Combined, roughly two pages of overrun in an eight-page paper whose
three-page Results section does not exist yet. The arithmetic does not
close without cutting.

Priority order for cuts, from `docs/DRAFT_REVIEW.md` section 5:

1. **Related Work 3** is longest and most reducible. The restoration
   literature supports the physical models but is not competed with.
   Compress each work to a clause; keep Narasimhan and Nayar, Chen et
   al., and Garg and Nayar at sentence length since those three underlie
   the implementations. Target: halve it.
2. **Related Work 4**: keep the tiling justification and SAHI, cut the
   FPN and small-object background. Target: cut by a third.
3. **Related Work 2** is closest prior work; cut it least, but the
   sentence-by-sentence walk through UAV-C, HazyDet, and WRRT-DETR can
   compress to two sentences plus the pattern observation.

Per-sentence test: does this establish the gap this paper fills, or
explain a related paper for its own sake? Keep the first, cut the second.

## 7. Tables and figures

All tables are written with real measured values in
`paper/aegisbench_body.tex`, ready to compile.

| # | Content | Placement |
| --- | --- | --- |
| Table 1 | Corruption taxonomy and measured calibration statistics | Benchmark Design, full width |
| Table 2 | Training configuration | Experimental Setup |
| Table 3 | Frozen operating points | Experimental Setup |
| Table 4 | Clean baselines | Results |
| Table 5 | Severity spectrum, SARD/YOLOv11 | Results |
| Table 6 | Per-family relative drop, both datasets | Results |
| Table 7 | Low-light collapse, both datasets | Results |
| Table 8 | Threshold-independent mAP confirmation | Results |

| # | Content | Width | Status |
| --- | --- | --- | --- |
| Fig. 1 | Qualitative failure pair, page-1 teaser | column | generated, needs visual review |
| Fig. 2 | Corruption taxonomy panel | full | run the generator |
| Fig. 3 | Severity curves with CI bands | full | run the generator |
| Fig. 4 | Relative-drop heatmap | full | run the generator |
| Fig. 5 | Localization decoupling scatter | column | run the generator |

Generation commands and draft captions: `docs/FIGURES.md`. Figures are
produced at exact CVF column or text width and must be placed at scale 1;
do not rescale them in LaTeX, which shrinks their type.

## 8. Claims that must not be overstated

These are constraints on the prose, checked against the data.

- **Safe:** total detection collapse under low light by severity 3
  replicates across both datasets and all three architectures.
- **Safe:** Faster R-CNN is consistently the most robust of the three
  under low light, on both datasets, at every severity where any model
  retains recall.
- **Not safe:** "RT-DETR is the most fragile architecture." True on
  HERIDAL, but it reverses on SARD, where RT-DETR outperforms YOLOv11 at
  severities 1 and 2. State the ordering as dataset-dependent.
- **Localization stability is a claim about magnitude, not
  independence.** Both recall and stability decline. Recall spans nearly
  the entire unit range; stability never falls below 0.78. Do not say
  they are unrelated.
- **The calibration statistic is a within-corruption severity
  calibrator, not a cross-corruption difficulty normaliser.** Two
  corruptions matched on the same statistic are not expected to be
  equally damaging. The draft already states this correctly.
- **`smoke_haze` versus `dust_haze`** is an observation, not a mechanism.
  They reach near-identical RMS contrast yet differ 6.7-fold in cost, but
  they also differ in airlight chromaticity, transmission granularity,
  and grain, so this experiment does not isolate the cause.

## 9. Where everything lives

| Need | File |
| --- | --- |
| Every measured number, plus prose for Setup, Results, Limitations, Conclusion | `docs/WRITING_PACK.md` |
| Corrections to the existing draft, missing citations, cut plan | `docs/DRAFT_REVIEW.md` |
| Figure plan, captions, generation commands, LaTeX placement | `docs/FIGURES.md` |
| LaTeX skeleton with all tables filled | `paper/aegisbench_body.tex` |
| Submission requirements, anonymization, packaging | `docs/SUBMISSION.md` |
| Datasheet for the supplementary archive | `docs/DATASHEET.md` |
| Corruption physics and calibration methodology | `docs/CORRUPTIONS.md` |
| Dataset sourcing, layouts, split policy | `docs/DATA.md` |

For a fresh session, the minimum useful set is this file plus
`docs/WRITING_PACK.md` and `paper/aegisbench_body.tex`.

## 10. Immediate next actions

1. Apply the four corrections in section 4 to the working draft.
2. Write Related Work subsection 1, including the SARD-Corr
   differentiation. Text in `docs/WRITING_PACK.md` section 3.
3. Add the six missing citations.
4. Write Results from `docs/WRITING_PACK.md` section 2. All numbers are
   measured; none need computing.
5. Write Limitations and Conclusion from the same pack, sections 4 and 5.
6. Write Introduction paragraphs 1 to 4.
7. Cut Related Work by roughly 40 percent.
8. Run the figure generators and visually review the gallery images.
9. Move into the official WACV template and check the page count.
10. Run `python -m aegisbench.anonymize --root .`, then
    `python scripts/make_submission.py --out dist/`.

Steps 1 through 7 are prose. Step 8 takes minutes. Nothing on this list
requires further experiments.
