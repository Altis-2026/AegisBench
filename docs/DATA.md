# Datasets

Both datasets must be downloaded manually (registration/licensing); this
repo never redistributes them. After download, verify counts with the
Phase 1 scripts and treat THOSE numbers as ground truth for the paper —
never quote sizes you did not measure from your own copy.

## HERIDAL

* Source: the IPSAR research project page at FESB, University of Split
  (search "HERIDAL database"; hosted under fesb.unist.hr). Free for
  research use.
* Content: full-size aerial images (~4000x3000) of wilderness scenes with
  person annotations in PASCAL-VOC XML; an official train/test folder
  split (~1500-1600 train, ~101 test full-size labeled images — verify
  against your archive).
* Expected layout (adjust flags if your archive differs):

```
data/heridal/
  trainImages/            *.JPG
  trainImages/labels/     *.xml
  testImages/             *.JPG
  testImages/labels/      *.xml
```

* Split policy: official test split untouched; validation carved
  deterministically (15%) from official train for operating-point
  selection and early stopping.
* Cross-check: a community mirror of HERIDAL with YOLO-format labels
  exists on Roboflow. Useful for spot-checking label parsing (compare a
  handful of images' box counts), but the FESB original is the citable
  source of record.

## SARD

* Source: IEEE DataPort, "Search and Rescue Image Dataset for Person
  Detection" (Sambolek & Ivasic-Kos). Free with IEEE account.
* Content: 1920x1080 frames extracted from video of actors simulating
  injured/lost persons across terrains; PASCAL-VOC XML annotations.
* Layout:

```
data/sard/
  images/   *.jpg
  labels/   *.xml
```

* **Split policy (leakage guard):** frames from one video sequence are
  near-duplicates, so splitting is GROUP-aware — a whole sequence goes to
  exactly one of train/val/test. Group identity is derived from the
  filename prefix; `scripts/phase1_prepare_sard.py` prints the discovered
  group table and refuses degenerate groupings. Review that table against
  the actual filenames in your copy and adjust `--group-regex` if needed.
  This is the difference between a defensible benchmark and silently
  inflated numbers.
* SARD frames go through the same tiling pipeline as HERIDAL. At
  tile 1024 / overlap 256 a 1920x1080 frame yields 6 tiles (a 4000x3000
  HERIDAL frame yields 20). Detections merge back to full-image
  coordinates either way, so evaluation is always against the original
  full-image ground truth.

## Published reference point

For pipeline validation only (Phase 4): published YOLOv5L results on
HERIDAL of ~0.90 precision / 0.893 recall / 0.834 mAP@0.5. Our numbers
come from different architectures and input pipelines, so modest deltas
are expected; order-of-magnitude disagreement means a bug, not a finding.
Verify the exact citation from the HERIDAL literature when writing the
related-work section.
