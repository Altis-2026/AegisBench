#!/usr/bin/env python3
"""Parameter count and per-frame inference cost for the checklist's
efficiency item.

Loads each checkpoint, counts learnable parameters, and times a fixed
number of forward passes at the evaluation image size (1024, matching
every sweep in this project) after a warmup. No training, no dataset
needed -- just the checkpoint files and a GPU (falls back to CPU with a
note if none is available, but CPU timings should not be reported next
to the GPU numbers used for training/eval).

  python scripts/model_cost.py \
      --weights runs/yolo11/yolo11_sard_clean/weights/best.pt \
                runs/rtdetr/rtdetr_sard_clean/weights/best.pt \
                runs/fasterrcnn/fasterrcnn_sard_clean_best.pt \
      --imgsz 1024 --iters 50 --out results/workshop/model_cost.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def time_inference(predict_fn, img, iters: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        predict_fn(img)
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        predict_fn(img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def load_ultralytics(weights, imgsz):
    from aegisbench.models.ultralytics_wrapper import UltralyticsDetector
    det = UltralyticsDetector(weights)
    img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    n_params = count_params(det.model.model)
    return n_params, lambda im: det.predict(im, conf=0.001, imgsz=imgsz), img


def load_fasterrcnn(weights, imgsz):
    import torch
    from aegisbench.models.fasterrcnn import build_model
    model = build_model(num_classes=2)
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    n_params = count_params(model)
    img = torch.zeros((3, imgsz, imgsz), device=device)

    def predict(_img_ignored):
        with torch.no_grad():
            return model([img])
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    return n_params, predict, dummy


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="+", required=True,
                    help="one or more checkpoint paths")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", help="write results to this CSV")
    args = ap.parse_args()

    import torch
    device_note = "GPU" if torch.cuda.is_available() else "CPU (no GPU found)"
    print(f"running on: {device_note}\n")

    rows = []
    for w in args.weights:
        w = str(w)
        name = Path(w).stem
        if "fasterrcnn" in w.lower():
            n_params, predict_fn, img = load_fasterrcnn(w, args.imgsz)
        else:
            n_params, predict_fn, img = load_ultralytics(w, args.imgsz)
        ms = time_inference(predict_fn, img, args.iters) * 1000
        print(f"{name:30s}  {n_params/1e6:7.2f}M params  {ms:7.2f} ms/frame  "
              f"({1000/ms:5.1f} FPS)")
        rows.append({"checkpoint": w, "params_millions": round(n_params / 1e6, 3),
                     "ms_per_frame": round(ms, 3), "fps": round(1000 / ms, 2),
                     "device": device_note, "imgsz": args.imgsz})

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
