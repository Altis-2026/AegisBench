# CV4EO @ WACV 2027 submission

Source: `paper/cv4eo/`. Build and check with `make -C paper/cv4eo`.

## Track: full paper, not short paper

The uploaded baseline was built on the short-paper template and came out
at 5 pages. Two things about the call for papers make the full track the
right home for it now:

* Short papers are capped at 4 pages excluding references, so the
  baseline was already over, and the mitigation and multi-seed results
  would have had to be cut to fit. Those are the two experiments the
  baseline's own "Ongoing Work" section promised, and they are now done.
* Short papers are **not** in the WACV 2027 Workshop Proceedings. Full
  papers are. For an archival benchmark release that is the difference
  between a citable artefact and a talk.

The short track's compensation is an invitation to an extended JSTARS
article. That remains available later and is not lost by submitting to
the full track now.

**Re-check the current deadline, page limit and submission site against
the CFP before submitting.** Last read: 8 pages excluding references and
appendix, double-blind, via OpenReview.

## What changed from the baseline

The baseline's measurement content is kept essentially intact, with its
numbers re-verified against the CSVs. What is new:

* `sec/2_related.tex` (new). The baseline had no related work, which a
  full paper needs. Positions AegisBench against ImageNet-C, Foggy
  Cityscapes and the optical models the corruptions are built from. No
  citation was added beyond the 15 already in `aegisbench.bib`.
* `sec/5_mitigation.tex` (new). The mitigation study: recovery at a
  held-out severity 3, and the trade-off on the 24 conditions the
  augmentation never trained for, which changes sign across
  architectures.
* Multi-seed subsection in `sec/4_results.tex` (new). Three seeds, all
  exactly zero at severity 3.
* `sec/4_ongoing.tex` is gone. Its "we are running these next" paragraph
  described experiments that have since finished, so leaving it would
  have been inaccurate. Its limitations and ethics content moved into
  `sec/6_discussion.tex`, extended with a fourth limitation (corruptions
  are applied one at a time, not composed).
* `sec/6_discussion.tex` adds a released-artefacts section, since the
  data release is part of what is being submitted.
* An explicit contributions list in the introduction.
* Results are now **point estimates** throughout, with bootstrap
  intervals reported separately. The baseline tables quoted bootstrap
  means, which differ in the third decimal (for example Faster R-CNN on
  HERIDAL at low-light severity 1 is 0.451 as a point estimate and 0.452
  as a bootstrap mean). Both are defensible; reporting one consistently
  and labelling it is what matters, and point estimates match the GRSL
  letter so the two papers cannot be read as disagreeing.

## Build gates

`make -C paper/cv4eo` fails, rather than warns, if any of these fail:

1. **Numbers.** `scripts/verify_cv4eo_numbers.py` parses the tables and
   the inline claims out of the `.tex` sources and re-derives all 193
   values from `results/`. Currently 193/193 agree.
2. **Length.** The body must end by page 8.
3. **Overfull boxes.** Must be zero.
4. **Anonymity.** Fails if any author name, institution or project name
   appears in the sources. CV4EO review is double-blind, which is the
   opposite of the GRSL letter's posture, so the two papers must not
   share a preamble.

Current status: 193 checks pass, body ends on page 8 (9 pages total, page
9 is references only), 0 overfull boxes, no identifying text.

## Before submitting

* Replace `\wacvPaperID{*****}` with the assigned OpenReview ID.
* Leave `\usepackage[review,applications]{wacv}` for submission; switch
  to the camera-ready line and fill in the real author block only if
  accepted.
* The `applications` option is a WACV main-conference track label and
  only affects the review-copy banner. It is set because the template
  errors out without one of the three track options.
* Confirm the figures are the intended versions. They were generated on
  the GPU machine from imagery this checkout does not contain and cannot
  be rebuilt here.

## Data release

`results/workshop/README.md` is the data card. Three summary tables
(`benchmark_168_conditions.csv`, `mitigation_sard_lowlight.csv`,
`multiseed_sard_yolo11.csv`) plus the raw sweep records, all regenerated
by `scripts/workshop_tables.py` and all consistent with the paper by
construction.

## Known open items

* The VisDrone real-night check is still the original 40 images.
  `scripts/visdrone_lowlight_check.py` scales it past 40 but has never
  been run; the paper states the 40-image limit plainly rather than
  overclaiming.
* No mitigation arm on HERIDAL, and one seed per architecture for the
  mitigation study. Stated as scope in the paper.
* `results/sweep/master.csv` is an older duplicate of `master_ci.csv`
  (it predates it and carries no `aug_lowlight` rows). Nothing reads it.
  It should probably be removed from the repo, but it is left in place
  here rather than deleted without asking.
