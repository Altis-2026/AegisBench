# Corruption taxonomy: physical grounding and calibration

Design principles:

1. **Physically motivated, not arbitrary noise.** Every corruption models a
   documented optical phenomenon of a disaster environment, organized by
   optical *mechanism* and mapped to disaster families. Haze-type
   corruptions use the Koschmieder scattering model
   `I = J·t + A·(1−t)`; low light is modeled in an approximately linear
   photometric domain with signal-dependent shot noise; glare and rain are
   additive radiance phenomena (screen blending / additive streaks).
2. **Calibrated severities.** Each corruption's three severities are
   defined by a measurable image statistic (declared in
   `configs/corruptions.yaml` under `calibration:`), not by eyeballed
   parameter values. `scripts/phase3_calibrate.py` measures the statistic
   on real data and fails if the ladder is not monotonic; the measured
   values are what the paper's taxonomy table reports.
3. **Deterministic.** Seeds derive from `(image_id, corruption, severity,
   global_seed)`; the corrupted benchmark is a fixed dataset.
4. **Synthetic-on-real, stated as such.** Corruptions are synthetic
   degradations applied to real clean imagery — the established
   corruption-robustness methodology (ImageNet-C lineage). This is a
   controlled, repeatable, parameterized design, and it is also the
   benchmark's key limitation: it measures response to modeled optics, not
   to live disaster footage. State both in the paper.

## The nine corruptions

| Corruption | Family | Optical mechanism | Severity statistic |
| --- | --- | --- | --- |
| `water_glare` | flood | specular sun-glint saturating the sensor; screen-blended glint cores + bloom | fraction of near-saturated pixels |
| `turbidity_cast` | flood | sediment-laden water: mud chromaticity blend, contrast compression, desaturation | RMS contrast |
| `inundation` | flood | semi-transparent standing water: smooth coverage mask, ripple refraction warp, murky blend, sparkle | occluded pixel fraction |
| `smoke_haze` | wildfire | Koschmieder scattering, neutral-gray airlight, low-frequency transmission field | RMS contrast |
| `fire_warm_tint` | wildfire | low-CCT fire illumination: warm WB shift + wispy warm-airlight haze | mean R / mean B |
| `rain_streaks` | storm | additive rain streaks with one wind direction/image + rain veil (blur, desat, darkening) | streak density (parameter-defined; visually audited) |
| `motion_blur` | storm | wind-induced platform shake: linear blur kernel, random direction/image | mean Sobel edge strength |
| `low_light` | storm (shared w/ earthquake) | photon scaling in linear domain, twilight blue-shift, shot + read noise | mean luminance |
| `dust_haze` | earthquake / post-disaster | Koschmieder with **brown** mineral airlight, patchier high-octave transmission, coarse near-lens grain | RMS contrast |

## Why dust is a corruption, not an "earthquake family" of one

Earthquakes have no unique optical signature; what degrades post-quake
aerial imagery is suspended dust, debris haze, and low light. Dust is
therefore included as one physically distinct corruption — distinguished
from wildfire smoke by airlight chromaticity (brown mineral vs. neutral
gray; enforced by a unit test on the R/B ratio), transmission-field
granularity (patchy turbulent plumes vs. diffuse layers), and coarse
particulate grain — rather than as a padded fourth family duplicating the
smoke model under a new label. `low_light` is shared framing: it covers
both storm dusk and night/dawn post-disaster response. The paper's framing
sentence: the corruption set covers the dominant visual failure modes
across flood, wildfire, storm, and earthquake response.

## Scale invariance

All pixel-unit parameters (blur kernel, streak length, bloom sigma, ripple
amplitude) are specified per 1000 px of the longer image side and scaled at
runtime, so severity means the same thing on a 4000x3000 HERIDAL frame and
a 640x640 SARD frame.

## Ground-truth / pixel alignment

Corruptions are appearance-only: they recolor, darken, haze, or overlay the
existing pixel grid without moving object content, so the ground-truth
boxes drawn on the clean grid remain exactly valid on the corrupted grid.
This pixel-for-pixel alignment is what makes the clean-vs-corrupted recall
comparison fair.

The one corruption with any geometric component is `inundation`, whose
water-refraction ripple displaces submerged pixels via `cv2.remap` while
the gt box stays fixed. The ripple amplitude is deliberately bounded well
below person scale (max ~1.4 px at severity 3 on a 4000 px frame — a few
percent of even the smallest survivor), so the residual gt/pixel
misalignment is negligible; the occlusion and murky-water blend, not the
ripple, carry the degradation. This bound is stated as a minor limitation
rather than hidden.

## Localization stability (second robustness axis)

Beyond recall (did we still find the survivor), the evaluator also reports
**localization stability** for survivors detected in both the clean and
corrupted conditions: the IoU between the clean and corrupted predicted
boxes, a scale-normalized center-shift, and the drop in box-vs-gt fit. This
separates two distinct failure modes — losing the detection entirely vs.
keeping it but with a box that drifts off the person — and directly
supports the failure-mode analysis in Section 5. See
`src/aegisbench/evaluation/localization.py`.

## Test-time vs. train-time application

At evaluation, corruptions are applied to the **full image before
tiling** — atmospheric structure (haze gradients, flood masks) is spatially
coherent across the whole frame, as in reality. For mitigation training,
corruption is applied per tile (cheap, trainer-agnostic); this is an
approximation and is documented as such.
