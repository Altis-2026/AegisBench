# WACV 2027 submission package

Everything needed to assemble and ship the submission, and what each item
must look like. This supersedes the submission-related notes scattered
through `docs/PAPER_GUIDE.md`.

## How to read the verification column

Requirements below carry a status, because conference pages get amended and
because some widely-circulated "requirements" turn out to be secondhand
summaries rather than official text:

- **[VERIFIED]** confirmed against WACV or NeurIPS official pages by search
  on 21 August 2026.
- **[UNVERIFIED]** plausible and in circulation, but not confirmed against
  an official source. Check it yourself before relying on it.

Nothing here replaces reading the official pages directly:
[Author Guidelines](https://wacv.thecvf.com/Conferences/2027/AuthorGuides),
[Call for Papers](https://wacv.thecvf.com/Conferences/2027/CallForPapers),
[Reviewer Guidelines](https://wacv.thecvf.com/Conferences/2027/ReviewerGuidelines),
[Dates](https://wacv.thecvf.com/Conferences/2027/Dates).

---

## 1. Dates

| Milestone | Date | Status |
| --- | --- | --- |
| Round 2 registration opens | 21 Aug 2026 | [VERIFIED] |
| **Round 2 paper deadline** | **28 Aug 2026 (AoE)** | [VERIFIED] |
| Round 2 supplementary deadline | 30 Aug 2026 (AoE) | [VERIFIED] |
| Reviews and decisions to authors | 9 Oct 2026 | [VERIFIED] |
| Camera ready, both rounds | 2 Nov 2026 | [VERIFIED] |

Registration is a separate step that happens **before** the paper deadline.
A missed registration is the most avoidable way to lose a submission.

Note the supplementary deadline is two days **after** the paper deadline.
That is deliberate breathing room: the paper PDF must be final on the 28th,
but the supplementary archive can be finished on the 29th or 30th.

## 2. Track and subject area

**Track: Evaluations and Datasets.** [VERIFIED] This track is new for WACV
2027 and explicitly welcomes "analysis of strengths, limitations, or failure
modes of existing benchmarks," alongside new datasets and tools. That is a
precise description of this paper. Reviewers are directed to assess whether
the work meaningfully advances a sub-area and whether the benchmarking is
scientifically correct, not whether it proposes a novel architecture.

**Student paper:** answer according to whether the first author is a student
at the time of submission.

**Subject area:** pick from the dropdown WACV shows on the submission form.
It is used for reviewer matching only and has no effect on the review
itself. Choose the closest match to aerial/remote-sensing object detection,
or to datasets and evaluation, depending on what the list offers. Use
"other" only as a last resort.

## 3. Main paper PDF

| Requirement | Status |
| --- | --- |
| Maximum 8 pages excluding references; reference-only pages may exceed it | [VERIFIED] |
| Must use the official WACV 2027 template (LaTeX author kit or Overleaf) | [VERIFIED] |
| Must be anonymized for double-blind review | [VERIFIED] |
| Papers that are over length, not anonymized, or not using the template are **rejected without review** | [VERIFIED] |
| Maximum file size 20 MB | [UNVERIFIED] |
| Leave the paper ID field at the template default | [UNVERIFIED] |

The three rejection triggers are hard, automated, and unappealable. Treat
page count, template, and anonymization as pass/fail gates, not as style
guidance.

## 4. Supplementary material

| Requirement | Status |
| --- | --- |
| PDF or ZIP only, maximum 200 MB | [VERIFIED] |
| Separate deadline, 30 Aug 2026 | [VERIFIED] |
| Must be anonymous, same standard as the paper | [VERIFIED] |
| No template required; single-column PDF is fine | [UNVERIFIED] |

`scripts/make_submission.py` builds the ZIP. Contents:

```
aegisbench_supplementary/
  README.md                     entry point, tells a reviewer what is where
  DATASHEET.md                  the datasheet (docs/DATASHEET.md)
  RUNBOOK.md                    the phased protocol
  CORRUPTIONS.md                physical grounding and calibration
  DATA.md                       dataset sourcing, layout, split policy
  code/                         the full anonymized source tree
  results/
    master_ci.csv               all 168 conditions
    ci_heridal.csv              bootstrap CIs
    ci_sard.csv
    localization_heridal.csv    localization stability
    localization_sard.csv
    calibration.csv             measured severity statistics
    summary_recall.csv          per-family aggregates
  figures/                      heatmaps, severity curves, decoupling plots, gallery
```

Deliberately excluded: `docs/PAPER_GUIDE.md` (internal working document
that discusses the submission itself and names the repository),
`results/sweep/preds/` (large, and regenerable), the `.git` directory, and
any local dataset copies.

## 5. Anonymization

WACV desk-rejects on identity leaks. [VERIFIED] Run the scanner:

```bash
python -m aegisbench.anonymize --root .
```

It flags configured git identity, emails, non-allowlisted URLs, and any
terms listed in a local git-ignored `.anonymize-terms.txt`. Create that file
on your machine with your name, your institution, your username, and any
identifying domain, one term per line. It is git-ignored by design so the
terms themselves never enter the repository.

**Known gaps in the scanner, check these by hand:**

1. The URL regex requires a protocol prefix, so a bare
   `github.com/owner/repo` reference written without one is **not** caught.
   Grep for your repository owner and name directly.
2. It skips `results/`, `data/`, and `runs/`, so identifying content inside
   result filenames or figure metadata is not scanned.
3. It reads text files only. Check figure image metadata separately if any
   figure was produced from your own camera captures.

Also check by hand: acknowledgements, funding statements and grant IDs,
institution names in file paths or figure axes, and any repository link in
the paper. For a code release during review, use an anonymous mirror or the
supplementary ZIP; do not link a repository that identifies you.

## 6. Data and code availability

The Evaluations and Datasets track expects reviewable artifacts.
[VERIFIED] The Call for Papers directs authors to the NeurIPS 2026
Evaluations and Datasets guidelines.

**On Croissant metadata.** NeurIPS 2026 requires Croissant metadata with
Responsible AI fields for *dataset* submissions. [VERIFIED] Two things
matter for us:

1. WACV *encourages* following the NeurIPS guidelines; it is not confirmed
   that WACV's OpenReview form has a Croissant upload field at all.
   **[UNVERIFIED]** Check the actual submission form.
2. AegisBench is a benchmark suite and evaluation tool, not a redistributed
   image dataset. We ship a generator, a protocol, and results, and no
   imagery (see `docs/DATASHEET.md`). Croissant is designed to describe
   hosted datasets, so it may simply not apply here. If the form offers the
   field and you want to use it, the natural artifact to describe is the
   results bundle (the CSV tables), not the source imagery we have no right
   to redistribute.

Either way, `docs/DATASHEET.md` covers the substance the track cares about:
provenance, composition, licensing, intended use, limitations, and
responsible-AI notes.

## 7. What you need to do

Ordered, with the blocking items first.

**Before 21-28 August**

1. **Register the paper on OpenReview.** Separate from and earlier than the
   paper upload. Do this first.
2. **Create `.anonymize-terms.txt`** in the repository root with your name,
   institution, username, and any identifying domain, one per line. It is
   already git-ignored.
3. **Fill every `[VERIFY]` marker in `docs/DATASHEET.md`.** Several must
   stay blank for the anonymous review copy (authors, funding, maintainer
   contact); the rest need real answers, especially the measured dataset
   counts.
4. **Confirm the dataset counts** you actually measured, and use only those
   numbers in the paper.
5. **Generate the remaining figures** (seconds each, no GPU):
   ```bash
   python scripts/phase5_figures.py --ci results/sweep/ci_heridal.csv \
       --localization results/sweep/localization_heridal.csv \
       --dataset heridal --out results/sweep
   python scripts/phase5_figures.py --ci results/sweep/ci_sard.csv \
       --localization results/sweep/localization_sard.csv \
       --dataset sard --out results/sweep
   ```
6. **Look at the gallery images** in `results/sweep/gallery/` and confirm
   the boxes sit on real people and the corruption renders plausibly. This
   is the one open item that cannot be automated.
7. **Write the paper** using `docs/PAPER_GUIDE.md`, and add the
   human-subjects sentence from the datasheet to the experimental setup.
8. **Build the supplementary archive:**
   ```bash
   python scripts/make_submission.py --out dist/
   ```
   It refuses to build if the anonymization scan finds anything, so fix
   findings rather than bypassing it.
9. **Open the built ZIP and read its README** as if you were a reviewer who
   has never seen the project.
10. **Submit the paper PDF by 28 August**, then the supplementary archive by
    30 August.

**Check these by hand before uploading**

- Page count at or under 8, excluding references.
- Official template, unmodified margins and font sizes.
- No author names, affiliations, acknowledgements, grant IDs, or repository
  links anywhere in the PDF.
- Every number in the paper traceable to a results file.
- Figures legible at print size, not just on screen.
