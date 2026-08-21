#!/usr/bin/env python3
"""Phase 5e: qualitative failure gallery. Picks the examples where a model
had real detections on clean imagery and lost the most of them under one
corrupted condition, and renders clean vs. corrupted side by side with
ground truth and kept detections overlaid -- the collapse made visible,
not just tabulated.

Regenerates the corrupted image from the corruption engine (deterministic,
same seed as the sweep) rather than requiring corrupted images saved to
disk. Requires phase5_sweep.py to have been run with --pred-dir.

  python scripts/phase5_gallery.py \
      --records data/heridal/records/test.json --dataset heridal \
      --pred-dir results/sweep/preds --model yolo11 \
      --corruption low_light --severity 3 --conf-thresh 0.45 \
      --out results/sweep/gallery --n 4
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.corruptions import DEFAULT_CONFIG_PATH, CorruptionSuite
from aegisbench.datasets.common import load_records
from aegisbench.evaluation import load_predictions
from aegisbench.inference import load_rgb
from aegisbench.visualize import draw_boxes

GT_COLOR = (255, 210, 0)      # BGR, ground truth (cyan)
DET_COLOR = (0, 220, 60)      # BGR, a kept detection (green)


def _panel(img_rgb: np.ndarray, gt_boxes: np.ndarray, dt_boxes: np.ndarray,
          label: str) -> np.ndarray:
    out = draw_boxes(img_rgb, gt_boxes, color=GT_COLOR, thickness=3)
    for x1, y1, x2, y2 in np.asarray(dt_boxes, np.float32).reshape(-1, 4):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                     DET_COLOR, 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
               (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _kept(pred_by_id: dict, image_id: str, conf_thresh: float) -> np.ndarray:
    r = pred_by_id.get(image_id)
    if r is None:
        return np.zeros((0, 4), np.float32)
    boxes = np.asarray(r["boxes"], np.float32).reshape(-1, 4)
    scores = np.asarray(r["scores"], np.float32).reshape(-1)
    return boxes[scores >= conf_thresh]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--corruption", required=True)
    ap.add_argument("--severity", type=int, required=True)
    ap.add_argument("--conf-thresh", type=float, required=True,
                    help="this model's frozen operating point, from "
                         "master_ci.csv")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4,
                    help="how many worst-case examples to render")
    ap.add_argument("--cell-width", type=int, default=560)
    args = ap.parse_args()

    suite = CorruptionSuite(args.config)
    records = {str(r["image_id"]): r for r in load_records(args.records)}

    pred_dir = Path(args.pred_dir)
    clean_dt = {str(r["image_id"]): r for r in load_predictions(
        pred_dir / f"{args.model}_{args.dataset}_clean_s0.json")}
    corrupt_dt = {str(r["image_id"]): r for r in load_predictions(
        pred_dir / f"{args.model}_{args.dataset}_{args.corruption}"
                   f"_s{args.severity}.json")}

    candidates = []
    for image_id, rec in records.items():
        gt = np.asarray(rec["boxes"], np.float32).reshape(-1, 4)
        if len(gt) == 0:
            continue
        n_clean = len(_kept(clean_dt, image_id, args.conf_thresh))
        n_corrupt = len(_kept(corrupt_dt, image_id, args.conf_thresh))
        if n_clean > 0:
            candidates.append((n_clean - n_corrupt, image_id))
    candidates.sort(reverse=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for _, image_id in candidates:
        if written >= args.n:
            break
        rec = records[image_id]
        clean_rgb = load_rgb(rec["image_path"])
        corrupt_rgb = suite.apply(clean_rgb, args.corruption, args.severity,
                                  image_id)
        gt = np.asarray(rec["boxes"], np.float32).reshape(-1, 4)

        clean_panel = _panel(clean_rgb, gt,
                             _kept(clean_dt, image_id, args.conf_thresh),
                             "clean")
        corrupt_panel = _panel(
            corrupt_rgb, gt,
            _kept(corrupt_dt, image_id, args.conf_thresh),
            f"{args.corruption} s{args.severity}")

        h, w = clean_panel.shape[:2]
        ch = int(round(args.cell_width * h / w))
        clean_small = cv2.resize(clean_panel, (args.cell_width, ch))
        corrupt_small = cv2.resize(corrupt_panel, (args.cell_width, ch))
        gutter = np.full((ch, 8, 3), 255, np.uint8)
        pair = np.hstack([clean_small, gutter, corrupt_small])

        stem = Path(image_id).stem
        p = (out / f"gallery_{args.dataset}_{args.model}_{args.corruption}"
                   f"_s{args.severity}_{stem}.png")
        cv2.imwrite(str(p), pair)
        print(f"wrote {p}  (clean kept={_kept(clean_dt, image_id, args.conf_thresh).shape[0]}, "
             f"corrupted kept={_kept(corrupt_dt, image_id, args.conf_thresh).shape[0]}, "
             f"gt={len(gt)})")
        written += 1

    if written == 0:
        print("No qualifying examples found. Check that predictions exist "
             "for this exact model/dataset/corruption/severity combination "
             "(they must have been archived via --pred-dir during the "
             "sweep), and that at least one image had detections on clean "
             "that were lost under the corrupted condition.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
