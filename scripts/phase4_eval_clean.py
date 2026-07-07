#!/usr/bin/env python3
"""Phase 4b checkpoint: clean-condition baseline validation.

For one trained detector: select the F1-optimal confidence threshold on the
CLEAN validation split, freeze it, evaluate on the clean test split, and
print a comparison against the published reference point (user-provided:
YOLOv5L on HERIDAL ~ 0.90 precision / 0.893 recall / 0.834 mAP@0.5). If
clean numbers are far off that ballpark, the pipeline is broken — debug
BEFORE any corruption sweep.

  python scripts/phase4_eval_clean.py --kind yolo11 --weights runs/.../best.pt \
      --records data/heridal/records --dataset heridal --out results/phase4
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.datasets.common import load_records
from aegisbench.evaluation import (evaluate, save_predictions,
                                   select_operating_point)
from aegisbench.inference import infer_records
from aegisbench.models import load_detector

REFERENCE = {"heridal": {"precision": 0.90, "recall": 0.893, "map50": 0.834,
                         "source": "published YOLOv5L baseline "
                                   "(user-provided reference point)"}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True,
                    choices=["yolo11", "rtdetr", "fasterrcnn"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--imgsz", type=int, default=1024)
    args = ap.parse_args()

    rec_dir = Path(args.records)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    detector = load_detector(args.kind, args.weights)

    print("[1/3] inference on clean VAL (operating-point selection)")
    val = load_records(rec_dir / "val.json")
    val_dt = infer_records(val, detector, tile_size=args.tile,
                           overlap=args.overlap, imgsz=args.imgsz)
    conf = select_operating_point(val, val_dt)
    print(f"    frozen operating point: conf={conf:.2f}")

    print("[2/3] inference on clean TEST")
    test = load_records(rec_dir / "test.json")
    test_dt = infer_records(test, detector, tile_size=args.tile,
                            overlap=args.overlap, imgsz=args.imgsz)
    save_predictions(test_dt, out / f"{args.kind}_{args.dataset}_clean.json")

    print("[3/3] evaluation")
    metrics = evaluate(test, test_dt, conf)
    metrics["model"] = args.kind
    metrics["dataset"] = args.dataset
    result_path = out / f"{args.kind}_{args.dataset}_clean_metrics.json"
    result_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    ref = REFERENCE.get(args.dataset.lower())
    if ref:
        print(f"\nreference ({ref['source']}):")
        for k in ("precision", "recall", "map50"):
            delta = metrics[k] - ref[k]
            print(f"  {k:9s} ours={metrics[k]:.3f} ref={ref[k]:.3f} "
                  f"delta={delta:+.3f}")
        print("Interpretation: architectures/inputs differ, so a modest gap "
              "is fine; a recall of e.g. 0.4 means the pipeline is broken.")
    print(f"\nwrote {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
