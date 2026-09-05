# IEEE GRSL submission

Everything specific to the *IEEE Geoscience and Remote Sensing Letters*
submission. This is a separate document from `docs/SUBMISSION.md`, which
covers the WACV version and still applies to it unchanged.

**The two versions have incompatible requirements.** Do not mix them.
The most important difference: WACV is double-blind and GRSL is not.

## How to read the verification column

- **[VERIFIED]** confirmed against an official IEEE or GRSS page by
  search on 5 September 2026.
- **[UNVERIFIED]** plausible and in circulation, but not confirmed
  against an official source. Check before relying on it.

---

## 1. What changed from the WACV process

| Item | WACV | GRSL | Status |
| --- | --- | --- | --- |
| Review model | Double-blind | **Single-blind** | [VERIFIED] |
| Anonymization | Required | **Must not anonymize** | [VERIFIED] |
| Page limit | 8 + refs | **5, everything included** | [VERIFIED] |
| Template | `wacv.sty` | **`IEEEtran.cls`, `[journal]`** | [VERIFIED] |
| Portal | OpenReview | **IEEE Author Portal** | [VERIFIED] |

### Do not run the anonymization scanner

`aegisbench.anonymize` and the GitHub-URL stripping in
`scripts/make_submission.py` exist for WACV's double-blind requirement.
GRSL review is single-blind: reviewers see author names and affiliation.
Running the scanner here would strip information the submission is
supposed to carry. Section 5 of `docs/SUBMISSION.md` does not apply.

Author names, affiliation, the acknowledgment, and the repository URL are
all in `paper/grsl/aegisbench_grsl.tex` from the first draft, and stay.

---

## 2. Requirements

| Requirement | Detail | Status |
| --- | --- | --- |
| Page limit | "The maximal number of pages of a GRSL article is 5." Over-length papers are redirected to TGRS or JSTARS rather than reviewed. | [VERIFIED] |
| Format | Two-column, single-spaced, official IEEE template | [VERIFIED] |
| Document class | `\documentclass[journal]{IEEEtran}`, IEEEtran V1.8a or later | [VERIFIED] |
| Portal | IEEE Author Portal, `ieee.atyponrex.com/journal/grsl`. Select "Letters" from the manuscript type dropdown. | [VERIFIED] |
| ORCID | Required before submission | [VERIFIED] |
| Page charges | Pages 1-3 free. Pages 4-5 are $230/page for non-members, $0 for GRSS members (submissions after 1 June 2025). | [VERIFIED] |
| Membership | Not required to submit | [VERIFIED] |
| Open access | Optional, $2645 | [VERIFIED] |
| First decision | ~30 days average | [VERIFIED] |
| Impact factor | 4.4 | [VERIFIED] |

### The portal is not ScholarOne

GRSL moved to the IEEE Author Portal (Atypon ReX). IEEE's migration
notice states that ScholarOne credentials do **not** carry over and that
a new account is needed. Papers already under review on ScholarOne finish
there; everything new goes through the Author Portal. Do not create a
ScholarOne account for this submission.

### Page charges are worth planning for

The letter is 5 pages, so pages 4 and 5 attract $460 in charges for
non-members. GRSS membership waives this entirely for submissions after
1 June 2025, and student membership costs well under that, so joining
first is the cheaper path. Confirm current rates at submission time.

---

## 3. GRSS Checklist for Authors

The society publishes a checklist at
<https://www.grss-ieee.org/publications/checklist-for-authors/> that
associate editors screen against. Items that bear directly on this paper:

- **Novelty and scope.** Must be within geoscience and remote sensing,
  with a clearly motivated contribution rather than an incremental
  improvement.
- **Real data.** The checklist states that evaluations should use real
  data and that **synthetic-only approaches are inadequate.** See the
  risk note below; this is the single largest scope exposure.
- **Modern datasets.** Avoid outdated benchmarks. HERIDAL (2019) and
  SARD (2021) are current for this task.
- **Baselines.** Compare against baseline and state-of-the-art methods.
- **Variance.** Report mean and standard deviation across runs where
  feasible. We report bootstrap intervals over the test set; we train one
  seed, and the Limitations section says so plainly.
- **Abstract.** Should carry motivation, contribution, data type, and
  quantitative findings. The current abstract does all four.
- **Introduction.** Must state the research gap and justify novelty
  explicitly.
- **Conclusion.** Must extend beyond the abstract with concrete future
  directions. Ours names three.
- **Figures.** Readable in print, vector preferred. Both figures are
  vector PDF built at exact column width and placed at scale 1.
- **Source code.** Should be public where possible. It is, and the letter
  links it.

### Open risk: the real-data expectation

AegisBench is built on modeled corruptions applied to real imagery. The
only real degraded-condition evidence is the 40-image VisDrone night
check. A reviewer applying the checklist literally can call this
synthetic-only.

Three things reduce the exposure, in descending order of value:

1. Expand the real night-imagery validation. Going from 40 to 100-150
   VisDrone images is cheap and strengthens an already-verified number.
2. Hand-curate real wilderness or wildfire drone imagery with boxes. A
   better domain match, but riskier to source inside a short budget.
3. Keep the framing honest. The letter already states the ceiling in
   Limitations rather than implying the gap is closed, which is the
   right posture if a reviewer raises it.

---

## 4. Building the letter

```
cd paper/grsl && make
```

The build fails if the paper exceeds 5 pages, if any line runs past its
column, or if any number in the tables disagrees with its source file in
`results/`. All three are hard gates, not warnings.

`scripts/grsl_figures.py` regenerates both figures from the committed
CSVs. `scripts/verify_grsl_numbers.py` runs the 193 number checks on its
own if you want them without a rebuild.

### Files

```
paper/grsl/
  aegisbench_grsl.tex     the letter
  aegisbench_grsl.bib     20 references, trimmed from the WACV set of 28
  IEEEtran.cls            official V1.8b from the IEEE author kit
  Makefile                build plus the three gates
  figures/                built by scripts/grsl_figures.py
```

---

## 5. Before submitting

- [ ] `make` passes all three gates
- [ ] Both authors have ORCID iDs
- [ ] GRSS membership decision made (affects the $460 page charge)
- [ ] Author Portal account created (not ScholarOne)
- [ ] Affiliation line and contact email confirmed correct
- [ ] Manuscript type set to "Letters" in the dropdown
- [ ] Read the full PDF once end to end, as a reviewer would
