#!/usr/bin/env python3
"""Real night-imagery spot check: SARD-trained YOLOv11, unmodified, run
against the lowest-luminance images in VisDrone's own test split.

This is new tooling. The original 40-image version of this check was not
scripted; this generalizes it to any sample size so the check can scale
past 40 without redoing the selection by hand each time.

VisDrone's own annotation format (one .txt per image, comma-separated
<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,
<truncation>,<occlusion>) is read directly; category 1 is "pedestrian"
and category 2 is "people" in the VisDrone-DET taxonomy, and both count
as a person for this check, matching what the original 40-image
selection used.

Two stages, run separately so the manual step is never accidentally
skipped:

  1. --rank: scores every test image by mean luminance and writes a
     ranked candidate list. Low luminance is a proxy for night/dusk, not
     a guarantee -- a dark shadow over a daytime scene ranks the same as
     a genuine night frame. The candidate list still needs a human to
     confirm each image actually depicts night or dusk before it is used,
     the same manual step the original 40-image check did. This script
     cannot do that step for you: automating "does this look like night"
     defeats the purpose of an independent human-verified check.

  2. --evaluate: takes a confirmed image list (one filename per line, you
     produce this by looking at the candidates from step 1) and runs the
     model against VisDrone's own annotations for exactly those images.

  python scripts/visdrone_lowlight_check.py --rank \
      --images /path/to/VisDrone2019-DET-test-dev/images \
      --out visdrone_candidates.txt --top 200

  # look at visdrone_candidates.txt, keep only genuine night/dusk frames,
  # save the confirmed subset as visdrone_confirmed.txt (one filename per line)

  python scripts/visdrone_lowlight_check.py --evaluate \
      --images /path/to/VisDrone2019-DET-test-dev/images \
      --labels /path/to/VisDrone2019-DET-test-dev/annotations \
      --confirmed visdrone_confirmed.txt \
      --weights runs/yolo11/yolo11_sard_clean/weights/best.pt \
      --conf-thresh 0.30
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PERSON_CATEGORIES = {1, 2}  # VisDrone-DET: 1=pedestrian, 2=people


def mean_luminance(path: Path) -> float:
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"could not read {path}")
    # BGR -> luminance, ITU-R BT.601 weights, consistent with how the
    # corruption engine's own mean_luminance calibration statistic is
    # computed (src/aegisbench/corruptions.py) so this ranking uses the
    # same definition of "dark" as the rest of the benchmark.
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    lum = 0.114 * b + 0.587 * g + 0.299 * r
    return float(lum.mean() / 255.0)


def cmd_rank(args) -> int:
    images_dir = Path(args.images)
    paths = sorted(images_dir.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no .jpg files found in {images_dir}")

    print(f"scoring {len(paths)} images by mean luminance...")
    scored = []
    for i, p in enumerate(paths):
        scored.append((mean_luminance(p), p.name))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(paths)}")

    scored.sort()
    top = scored[: args.top]

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        for lum, name in top:
            f.write(f"{lum:.4f}\t{name}\n")

    print(f"\nwrote {len(top)} candidates to {out_path}")
    print("Next: open these images and remove any that are not genuinely "
          "night or dusk (shadowed daytime scenes rank the same way). "
          "Save the filenames you keep, one per line, no luminance column, "
          "as the --confirmed file for --evaluate.")
    return 0


def _read_visdrone_annotation(path: Path) -> np.ndarray:
    """[x1, y1, x2, y2] boxes for pedestrian/people categories, VisDrone
    format. Ignored regions (category 0) and non-person categories are
    dropped; occluded and truncated instances are kept, since the SARD
    training data does not filter on either and dropping them here would
    make recall look better than the same model achieves on SARD."""
    boxes = []
    if not path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    for line in path.read_text().strip().splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        x, y, w, h = (float(v) for v in parts[:4])
        category = int(parts[5])
        if category not in PERSON_CATEGORIES or w <= 0 or h <= 0:
            continue
        boxes.append([x, y, x + w, y + h])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def cmd_evaluate(args) -> int:
    from aegisbench.evaluation.coco_eval import pr_at_threshold
    from aegisbench.models.ultralytics_wrapper import UltralyticsDetector
    import cv2

    confirmed = [ln.strip() for ln in Path(args.confirmed).read_text().splitlines()
                if ln.strip()]
    if not confirmed:
        raise SystemExit(f"{args.confirmed} is empty")
    print(f"{len(confirmed)} confirmed night/dusk images")

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    detector = UltralyticsDetector(args.weights)

    gt_records, dt_records = [], []
    n_gt_total = 0
    for name in confirmed:
        img_path = images_dir / name
        if not img_path.exists():
            print(f"  WARNING: missing image {img_path}, skipping")
            continue
        stem = img_path.stem
        gt_boxes = _read_visdrone_annotation(labels_dir / f"{stem}.txt")
        n_gt_total += len(gt_boxes)

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pred = detector.predict(img_rgb, conf=0.001, imgsz=args.imgsz)

        gt_records.append({"image_id": name, "boxes": gt_boxes})
        dt_records.append({"image_id": name,
                           "boxes": pred["boxes"], "scores": pred["scores"]})

    result = pr_at_threshold(gt_records, dt_records, args.conf_thresh)
    print(f"\nimages evaluated: {len(gt_records)}")
    print(f"ground-truth persons: {n_gt_total}")
    print(f"recall:    {result['recall']:.4f}  ({result['tp']}/{result['n_gt']})")
    print(f"precision: {result['precision']:.4f}")
    print(f"f1:        {result['f1']:.4f}")

    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n_images", "n_gt", "recall", "precision", "f1", "conf_thresh"])
            w.writerow([len(gt_records), result["n_gt"], result["recall"],
                       result["precision"], result["f1"], args.conf_thresh])
        print(f"\nwrote {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rank", action="store_true",
                      help="stage 1: rank test images by mean luminance")
    mode.add_argument("--evaluate", action="store_true",
                      help="stage 2: evaluate the model on a confirmed subset")

    ap.add_argument("--images", required=True, help="VisDrone test image directory")
    ap.add_argument("--labels", help="VisDrone test annotation directory (--evaluate)")
    ap.add_argument("--out", help="output path (candidate list or results CSV)")
    ap.add_argument("--top", type=int, default=100,
                    help="--rank: how many lowest-luminance candidates to write")
    ap.add_argument("--confirmed", help="--evaluate: manually confirmed filename list")
    ap.add_argument("--weights", help="--evaluate: detector checkpoint")
    ap.add_argument("--conf-thresh", type=float, default=0.30,
                    help="--evaluate: frozen operating point for this model/dataset")
    ap.add_argument("--imgsz", type=int, default=1024)
    args = ap.parse_args()

    if args.rank:
        if not args.out:
            raise SystemExit("--rank requires --out")
        return cmd_rank(args)
    if not (args.labels and args.confirmed and args.weights):
        raise SystemExit("--evaluate requires --labels, --confirmed, and --weights")
    return cmd_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
