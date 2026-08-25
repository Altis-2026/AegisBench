# Draft review, 21 August

Review of the paper draft (Abstract, Related Work subsections 2 to 4,
Benchmark Design, Evaluation Protocol, Contributions, bibliography) against
the codebase and against the WACV submission requirements.

Overall: this is good, careful writing. The prose is precise, the
methodological choices are explained rather than merely asserted, and the
Evaluation Protocol section in particular reads like something a reviewer
will trust. Two factual errors need correcting, one citation claim needs
verification, several required citations are missing, and there is a
structural length problem that will force cuts. All are fixable well inside
the remaining time.

---

## 1. Two factual errors, both verified against the code

### 1.1 The Koschmieder attribution is wrong

**Draft text (Physical Models):** "Three corruptions in the flood, wildfire,
and earthquake families (smoke_haze, dust_haze, and turbidity_cast) are
built on the Koschmieder atmospheric scattering model."

**What the code actually does.** Only two source files import or call
`koschmieder()`: `corruptions/wildfire.py` and `corruptions/dust.py`. The
three corruptions built on the scattering model are:

| Corruption | Family | Uses Koschmieder |
| --- | --- | --- |
| `smoke_haze` | wildfire | yes (`wildfire.py`) |
| `fire_warm_tint` | wildfire | yes (`wildfire.py`), applied after a white-balance shift |
| `dust_haze` | earthquake | yes (`dust.py`, importing from `wildfire.py`) |
| `turbidity_cast` | flood | **no** |

`turbidity_cast` lives in `corruptions/flood.py`, which never imports
`koschmieder`. It is an alpha blend toward a mud chromaticity followed by
contrast compression toward the scene mean and a saturation reduction:
`out = (1 - mud_alpha) * img + mud_alpha * mud`, then
`out = mean + contrast_factor * (out - mean)`. There is no transmission map
and no airlight term, so it is not a scattering model at all.

**Corrected text to use:**

> Three corruptions, `smoke_haze` and `fire_warm_tint` in the wildfire
> family and `dust_haze` in the earthquake family, are built on the
> Koschmieder atmospheric scattering model, I = J·t + A·(1 − t), where J is
> the clear-scene radiance, A is the atmospheric light, and t is the
> transmission map. `smoke_haze` and `dust_haze` differ principally in the
> chromaticity of A: smoke is modeled with a neutral gray airlight, dust
> with a brown mineral airlight, consistent with the documented spectral
> difference between combustion aerosol and suspended mineral particulate.
> This distinction is verified directly by a unit test on the red-to-blue
> channel ratio of the two corruptions. `fire_warm_tint` applies the same
> scattering model with a warm airlight and a higher-frequency transmission
> field, preceded by a white-balance shift toward orange that models
> low-colour-temperature firelight.
>
> The flood family is modeled differently, since sediment-laden water is
> not an atmospheric scattering process. `turbidity_cast` applies a blend
> toward a mud chromaticity, compresses contrast toward the scene mean, and
> reduces saturation, reproducing the appearance of imagery dominated by
> muddy water rather than by suspended aerosol.

This correction is worth making carefully rather than quickly. The physical
grounding of the corruption taxonomy is the paper's central methodological
claim, and a reviewer who works on dehazing will read this paragraph closely.
Getting `fire_warm_tint` in and `turbidity_cast` out actually strengthens
the section: it shows the taxonomy distinguishes between genuinely different
optical processes rather than applying one model everywhere.

### 1.2 The monotonicity verification claim overstates what is checked

**Draft text (Calibration and Severity):** "monotonicity of the statistic
across severities one through three is verified automatically for every
image in the benchmark."

**What the code actually does.** Verification happens in two places, neither
of which checks every image:

1. `scripts/phase3_calibrate.py` draws a random sample of `--n` test images
   (the run that produced `results/phase3/calibration.csv` used `--n 20`),
   computes the statistic on each, and then checks monotonicity of the
   **mean** across severities. It does not check any individual image.
2. `tests/test_corruptions.py::test_calibration_monotonicity` runs on a
   single synthetic textured image, averaged over six image identifiers,
   and explicitly skips `streak_density`.

So the claim is wrong in two directions at once: it is a sample, not the
full set, and it is the mean statistic that is checked, not the per-image
statistic. Averaging is the right design choice (the corruptions are
stochastic, so per-image monotonicity would fail spuriously), but the paper
has to describe what was done.

**Corrected text to use:**

> For eight of the nine corruptions this statistic is computed directly from
> the corrupted image. We verify the severity ladder by measuring the
> statistic on a random sample of N test images at each severity and
> confirming that its mean moves in the declared direction; the same check
> runs in the unit-test suite on synthetic imagery. We average across images
> rather than requiring per-image monotonicity because the corruptions are
> stochastic, so a single image can deviate without the ladder itself being
> ill-formed.

Substitute the actual N you used, and report it in the Table 1 caption
alongside the measured values. Stating the sample size is strictly better
than the vaguer original claim: it tells a reviewer exactly what was
checked, and it is a claim that survives scrutiny.

---

## 2. One citation claim to verify before it ships

**Draft text (Related Work 2):** "WRRT-DETR [CITE] takes a similar route in
the maritime domain, building the AWOD dataset around synthetic fog, flare,
and low light."

The bibliography entry itself checks out: Liu, B., Jin, J., Zhang, Y., and
Sun, C., *Drones* 9(5):369, 2025, DOI 10.3390/drones9050369. Verified.

The problem is the word **synthetic**. Published descriptions of AWOD say it
comprises roughly 20,000 images *captured under* three adverse weather
conditions (foggy, flare, low-light), and "captured under" reads as real
imagery, not synthesis. If AWOD is real capture, then the sentence
misdescribes it, and the paragraph's closing argument (that recent
drone-view robustness papers "synthesize a dataset around one or two
conditions and propose a fix") does not apply to WRRT-DETR in the way the
draft claims.

**Action:** open the paper and check. Then either keep "synthetic" if
confirmed, or reword. If AWOD turns out to be real imagery, the honest
framing is actually more interesting for us: it becomes an example of the
real-data alternative to our synthetic-on-real methodology, which is a
natural place to acknowledge our own limitation rather than a competitor to
differentiate from.

A second, smaller point in the same paragraph: the draft describes
WRRT-DETR as being "in the maritime domain," which is right for AWOD, but
the paper's own title is about drone-view detection generally. Keep the
maritime qualifier attached to the dataset rather than to the method.

---

## 3. Missing citations, including several that are mandatory

The bibliography has 20 entries and they are accurate. But it is missing
citations the paper cannot ship without.

**Mandatory, cannot submit without these:**

- **The HERIDAL paper.** Božić-Štulić, D., Marušić, Ž., Gotovac, S., "Deep
  Learning Approach in Aerial Imagery for Supporting Land Search and Rescue
  Missions," IJCV 127(9):1256-1278, 2019. You evaluate on this dataset;
  it must be cited.
- **The SARD paper.** Sambolek, S., Ivašić-Kos, M., "Automatic Person
  Detection in Search and Rescue Operations Using Deep CNN Detectors," IEEE
  Access, 2021. Same reason.
- **Faster R-CNN.** Ren, S., He, K., Girshick, R., Sun, J., NeurIPS 2015.
- **RT-DETR.** Zhao, Y. et al., "DETRs Beat YOLOs on Real-time Object
  Detection," CVPR 2024.
- **YOLOv11.** Cite the Ultralytics release with a version number if no
  paper exists; check what the canonical citation is at time of writing.

You cannot evaluate three detectors and two datasets without citing all
five. This is the single most urgent gap in the bibliography.

**Strongly recommended:**

- **Efron and Tibshirani**, *An Introduction to the Bootstrap*, 1993.
  Already flagged in your own "still needed" list.
- **SeaDronesSee.** Varga, L.A. et al., WACV 2022, pp. 2260-2270. Maritime
  UAV search and rescue, and a WACV paper. Its framing (vision systems for
  one environment have real-case data, another does not) is close to your
  own gap statement.
- **Hoiem et al.**, "Diagnosing Error in Object Detectors," ECCV 2012, and
  **Bolya et al.**, TIDE, ECCV 2020. Both decompose detector failure into
  modes, which is exactly what the localization-stability axis does. Citing
  them places your second axis in an established tradition instead of
  letting it look ad hoc. This is cheap credibility for one sentence.

**Worth checking:** a search surfaced "OWRT-DETR: A Novel Real-Time
Transformer Network for Small Object Detection in Open Water Search and
Rescue From UAV Aerial Imagery." Open-water UAV search and rescue with a
DETR variant is adjacent enough to your domain that you should look at it
and decide whether it belongs in Related Work subsection 1.

**Bibliography hygiene:**

- `he2009dark` and `he2011dark` are the conference and journal versions of
  the same work. Cite one, conventionally the TPAMI version, unless you have
  a specific reason for both.
- `koschmieder1924theorie` is typed as `@inproceedings` but is a journal
  article. Change to `@article` with the journal name. Citing the 1924
  original alongside Narasimhan and Nayar is good scholarship, keep it.

---

## 4. The missing Related Work subsection is the most important one

Subsections 2 (corruption robustness), 3 (adverse weather and low light),
and 4 (small objects and tiling) are written. **Subsection 1, aerial and
search-and-rescue person detection, is missing**, and it is the one that
matters most, for three reasons:

1. It is where HERIDAL and SARD get cited at all.
2. It is where the domain gap is established. Subsections 2 to 4 position
   the paper against *method* literature; subsection 1 is what makes the
   application case, and this is an applications-oriented venue.
3. It is where the **SARD-Corr** differentiation has to go. The SARD
   authors already shipped a small corrupted-image supplement (fog, snow,
   ice, motion blur, no calibrated severity ladder) inside the very paper
   you must cite for the dataset. That is the closest prior attempt at this
   paper's premise. Not addressing it directly is a findable gap; addressing
   it is a strong differentiation. See `docs/PAPER_GUIDE.md`, Theme 1, for
   the exact differentiation sentence.

Draft this subsection next, before anything else in Related Work.

---

## 5. The structural problem: Related Work is roughly three times its budget

Rough word counts from the draft:

| Section | Words | Approx. pages (two-column) | Budgeted |
| --- | --- | --- | --- |
| Related Work, subsections 2-4 | ~1,800 | ~1.9 | 0.75 total |
| Related Work, subsection 1 | not written | ~0.5 more | (included above) |
| Benchmark Design + Protocol | ~2,500 | ~2.6 | 2.0 |
| Contributions | ~400 | ~0.4 | (part of intro) |

At roughly 950 words per two-column page, the current Related Work is
heading for about 2.4 pages once subsection 1 is added. The budget is 0.75.
Benchmark Design is also running about 0.6 pages long.

Combined, that is roughly two pages of overrun in an eight-page paper whose
Results section is not written yet. Results needs about three pages. The
arithmetic does not close without cutting.

**This is not a criticism of the writing.** The Related Work is genuinely
good, and everything in it is relevant. But at WACV length, Related Work
earns its space only by establishing the gap, and the current version
explains related methods in more depth than that job requires.

**Where to cut, in priority order:**

1. **Subsection 3 (adverse weather and low light)** is the longest and the
   most reducible. The restoration and enhancement literature (Dark Channel
   Prior, Retinex, Exclusively Dark) supports your physical models but is
   not something you compete with. Compress each work to a clause. Keep
   Narasimhan and Nayar, Chen et al., and Garg and Nayar at sentence length,
   since those three actually underlie your corruption implementations.
   Target: cut by about half.
2. **Subsection 4 (small objects and tiling)** has a full paragraph on FPN
   and small-object difficulty that mostly restates known background. The
   part that must survive is the tiling protocol justification and SAHI.
   Target: cut by about a third.
3. **Subsection 2** is the closest prior work and should be cut least. But
   the sentence-by-sentence walk through UAV-C, HazyDet, and WRRT-DETR can
   compress into two sentences plus the pattern observation, which is the
   part that does the argumentative work.

A useful test for each sentence: does it establish the gap this paper
fills, or does it explain a related paper for its own sake? Keep the first,
cut the second. Move nothing to supplementary; related work belongs in the
main paper or nowhere.

---

## 6. Smaller notes

**The abstract.** It works as written. One optional improvement: the string
`[0.000, 0.000]` is table notation dropped into prose, and reads slightly
oddly mid-sentence. An alternative that keeps the rigor and adds the
resample count: "recall falls to exactly zero for all three architectures on
both datasets, and every one of 1,000 bootstrap resamples of the test set
agrees, the confidence interval is degenerate at zero." Your call; the
current version is not wrong.

**`rain_streaks` calibration wording.** The draft says "Its calibration
statistic, edge strength, is intended as a proxy for streak density."
Slightly off: the *declared* statistic in `configs/corruptions.yaml` is
`streak_density`, and edge strength is the substituted implementation
because streak density has no closed-form image measure. Table 1 will list
`streak density`, so the text should match: "Its declared statistic, streak
density, has no closed-form image measure, so edge strength is substituted
as a proxy."

**Contributions count.** Four contributions, cleanly stated. The
`PAPER_GUIDE.md` list had five, with released reproducible tooling as its
own item. Folding reproducibility into contributions one and two is a
defensible editorial choice; no change needed.

**"Four points of recall" in the abstract.** Check this against the CSV
before final submission. From `ci_sard.csv`, YOLOv11 rain_streaks severity 3
is 0.855 against a clean baseline of 0.8906, which is a 3.6-point drop.
"Roughly four points" is fair, but make sure the Results section states the
precise number and that the two are consistent.

---

## 7. What remains, in the order to write it

| Piece | Status | Notes |
| --- | --- | --- |
| Introduction paragraphs 1-4 | not written | Stakes, gap, approach, findings. Paragraph 5 (contributions) is done. See `PAPER_GUIDE.md` 7.3 |
| Related Work subsection 1 | not written | Highest priority. Carries HERIDAL, SARD, SARD-Corr differentiation |
| Related Work cuts | needed | Roughly 40% reduction across subsections 2-4 |
| Experimental Setup | not written | Models, hardware, splits, dataset counts you measured, and the human-subjects sentence from `DATASHEET.md` |
| Results | not written | ~3 pages. Order and content in `PAPER_GUIDE.md` 7.7 |
| Limitations | not written | Seven items, all already drafted in `PAPER_GUIDE.md` 7.8 and `DATASHEET.md` |
| Conclusion | not written | Short |
| Table 1 (taxonomy) | placeholder in text | Fill from `results/phase3/calibration.csv`, add the sample size N to the caption |
| Tables 2-4 | not built | Clean baselines, headline low-light, per-family. Numbers in `PAPER_GUIDE.md` 5 |
| Figures 1-4 | partly generated | Grid and heatmaps exist; severity curves and decoupling need `scripts/phase5_figures.py` |
| Qualitative gallery | generated, unreviewed | Open the PNGs in `results/sweep/gallery/` and choose two |
| Ethics statement | not written | One sentence, text in `DATASHEET.md` |

The two corrections in section 1 should be made now, while the Benchmark
Design section is fresh, rather than at the end.
