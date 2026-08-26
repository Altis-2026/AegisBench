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

**Updated 26 August**, in a remote session with no access to the working
Google Doc and no access to `data/` or `results/` (both gitignored, and
this was a fresh container clone). `paper/aegisbench_body.tex` previously
contained TODO markers and "ALREADY WRITTEN, paste here" placeholders
pointing at Google Doc content this session could not retrieve. Rather than
leave the LaTeX skeleton incomplete, every missing section was written
directly into the file from the fully-specified content in
`docs/WRITING_PACK.md` and `docs/PAPER_GUIDE.md`, and `paper/aegisbench.bib`
was built from scratch (no `.bib` file previously existed in the repo). The
table below reflects the file as it now stands, not the old Google Doc.

| Section | Status |
| --- | --- |
| Abstract | written, in `aegisbench_body.tex` |
| Introduction paras 1 to 4 | written |
| Introduction para 5 (contributions) | written, 4 bullets (reproducibility folded into 1 and 2) |
| Related Work 1, aerial and SAR person detection | written, includes the SARD-Corr differentiation |
| Related Work 2, corruption robustness | written, pre-cut to budget |
| Related Work 3, adverse weather and low light | written, pre-cut to budget |
| Related Work 4, small objects and tiling | written, pre-cut to budget (FPN background omitted) |
| Benchmark Design | written, Koschmieder correction applied |
| Evaluation Protocol | written, no duplicated sentence |
| Experimental Setup | written; **2 TODOs remain**, HERIDAL and SARD train/val image counts |
| Ethical Considerations | written, own section |
| Results | written, all 5 subsections |
| Limitations | written |
| Conclusion | written |
| Tables 1 to 7 | **all values measured**, LaTeX ready in `paper/aegisbench_body.tex` |
| Figures 1 to 5 | code written and tested; **still need to be run**, requires local `data/`/`results/` |
| Bibliography | `paper/aegisbench.bib` built, 25 entries, all 6 previously-mandatory ones added; a few entries flagged inline for pre-camera-ready verification (see section 4.4) |

**What this means practically:** the text of the paper is now complete
end to end except two numeric TODOs that require the prepared dataset
records (see section 10). The remaining work is mechanical: fill those two
numbers, spot-check the flagged citations, generate the figures somewhere
`data/` exists, and move into the official template.

## 4. Defects: status as of the 26 August rewrite

Everything in this section described defects in the old Google Doc draft.
Since the LaTeX body was rewritten from source material rather than pasted
from that doc, each item below is either resolved by construction or
still needs a human check. Kept for the record and so nobody re-opens a
closed item.

### 4.1 The Koschmieder attribution — resolved

`aegisbench_body.tex` now attributes the Koschmieder model correctly to
`smoke_haze`, `fire_warm_tint`, and `dust_haze` only; `turbidity_cast` is
described as a mud-chromaticity blend with contrast compression, no
transmission map, no airlight term.

### 4.2 Duplicated sentence — not applicable

The Evaluation Protocol section was written fresh; the tiling paragraph
appears once.

### 4.3 `rain_streaks` statistic wording — resolved

The Calibration and Severity subsection now reads "its declared statistic,
streak density, has no closed-form image measure, so edge strength is
substituted as a proxy," matching Table 1's "streaks/MP" row.

### 4.4 Six mandatory citations — resolved, but spot-check before camera-ready

`paper/aegisbench.bib` now exists (it did not before) with 25 entries,
including all 6 previously-missing mandatory ones (HERIDAL, SARD, Faster
R-CNN, RT-DETR, YOLOv11, Efron and Tibshirani) plus SeaDronesSee and
TIDE/Hoiem et al. for the localization-stability framing. Two entries are
flagged inline in the `.bib` file with `% VERIFY` comments and were **not**
independently re-confirmed against the primary source in this pass (no
live web access from that environment): the Sambolek and Ivašić-Kos (SARD)
volume/page numbers, and the exact title of the WRRT-DETR paper (only its
DOI and journal were previously confirmed). Check both before submitting.

### 4.5 Working artifacts to strip — not applicable

The rewritten file has no such headings; nothing to strip.

### 4.6 The WRRT-DETR/AWOD claim — resolved conservatively

The new Related Work 2 describes AWOD as "a maritime dataset of imagery
captured under fog, flare, and low-light conditions," using the
review's own finding (published descriptions favor real capture) rather
than asserting "synthetic." This sidesteps the unverified claim rather
than resolving it outright; if you do open the WRRT-DETR paper and confirm
one way or the other, the sentence can be sharpened.

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

## 6. The cutting problem — pre-empted, verify instead of cutting

This used to be the largest structural risk: the old Google Doc's Related
Work ran roughly 1.9 pages against a 0.75-page budget. The 26 August
rewrite wrote all four Related Work subsections directly to the cut
targets below rather than writing long and cutting afterward (roughly 200
/ 180 / 150 / 100 words for subsections 1 through 4). This has **not**
been checked against an actual page count in the WACV two-column template,
since no template or LaTeX install was available in that session. Treat
the estimate as pre-empted, not verified: once the paper is in the real
template (section 10), measure the actual Related Work length and confirm
it lands near 0.75 pages before assuming this risk is closed.

The original cut priorities are kept below in case the rewritten text
still runs long and needs further trimming:

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
| Fig. 2 | Corruption taxonomy panel | full | run the generator |
| Fig. 3 | Severity curves with CI bands | full | run the generator |
| Fig. 4 | Relative-drop heatmap | full | run the generator |
| Fig. 5 | Localization decoupling scatter | column | run the generator |

Note: `aegisbench_body.tex` currently places the qualitative clean/`low_light`
pair (`figures/gallery_lowlight.pdf`) inside the Results, Localization
Stability subsection rather than as a standalone page-1 teaser figure.
`docs/FIGURES.md` describes that pairing as a page-1 Figure 1; moving it
there (and renumbering Figures 2 to 5 accordingly) is a layout decision
nobody has made yet in the LaTeX, left alone in the 26 August rewrite to
avoid restructuring figure placement without being asked. Decide this
before the template pass.

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

`paper/aegisbench.bib` is new as of 26 August; the "where everything
lives" table above didn't previously have a row for it because it didn't
exist. For a fresh session, the minimum useful set is this file plus
`paper/aegisbench_body.tex` and `paper/aegisbench.bib` — the prose in the
`.tex` file is now self-contained and does not require re-reading
`docs/WRITING_PACK.md` unless you're auditing a specific number back to
its source.

## 10. Immediate next actions

All prose is written (section 3). What is left is data-dependent work
that requires the actual HERIDAL/SARD dataset and prior sweep outputs
under `data/` and `results/` — **run these on a machine that has them**,
not in an environment that only has the git-tracked repo:

1. Measure HERIDAL and SARD train/validation image counts from your
   prepared records and fill the two remaining `\TODO{n}` markers in
   Experimental Setup.
2. Spot-check the two `% VERIFY`-flagged entries in `paper/aegisbench.bib`
   (Sambolek and Ivašić-Kos page numbers; the WRRT-DETR title) against the
   primary sources.
3. Decide the Figure 1 placement question in section 7 above.
4. Run the figure generators (`docs/FIGURES.md`) and visually review the
   gallery images and the taxonomy panel crop.
5. Move into the official WACV template, paste in `aegisbench_body.tex`
   and `aegisbench.bib`, compile, and check the actual page count against
   the section 5 budget — confirm the Related Work pre-cut in section 6
   actually landed near 0.75 pages.
6. Run `python -m aegisbench.anonymize --root .`, then
    `python scripts/make_submission.py --out dist/`.

Nothing on this list requires further experiments or further writing;
every remaining step is measurement, verification, or packaging.
