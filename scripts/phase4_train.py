#!/usr/bin/env python3
"""Phase 4a: fine-tune one detector from its config.

  python scripts/phase4_train.py --config configs/train_yolo11.yaml
  python scripts/phase4_train.py --config configs/train_rtdetr.yaml
  python scripts/phase4_train.py --config configs/train_fasterrcnn.yaml

If a run was interrupted (Ctrl+C, killed by system sleep, crash), resume
it from its last checkpoint instead of restarting from epoch 0 -- every
detector saves one after each epoch:

  python scripts/phase4_train.py --resume runs/rtdetr/rtdetr_heridal_clean/weights/last.pt
  python scripts/phase4_train.py --resume runs/fasterrcnn/fasterrcnn_sard_clean_last_ckpt.pt
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _is_fasterrcnn_ckpt(path: str | Path) -> bool:
    """Our own Faster R-CNN checkpoints are always named
    "..._last_ckpt.pt" (see fasterrcnn._run_training_loop); ultralytics
    always names its own "last.pt" regardless of architecture, so the
    suffix alone tells them apart without needing to load either file."""
    return Path(path).name.endswith("_last_ckpt.pt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-dir", default="runs")
    ap.add_argument("--resume", default=None,
                    help="path to a last.pt checkpoint from an "
                         "interrupted YOLO/RT-DETR run, to continue "
                         "instead of starting over")
    args = ap.parse_args()

    if bool(args.config) == bool(args.resume):
        raise SystemExit("pass exactly one of --config or --resume")

    if args.resume:
        resume_path = Path(args.resume)
        if _is_fasterrcnn_ckpt(resume_path):
            from aegisbench.models.fasterrcnn import resume_training
            best = resume_training(resume_path, resume_path.parent)
        else:
            from aegisbench.models.ultralytics_wrapper import resume_training
            best = resume_training(resume_path)
        print(f"\nbest weights: {best}")
        return 0

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if cfg["kind"] in ("yolo11", "rtdetr"):
        from aegisbench.models.ultralytics_wrapper import train_from_config
    elif cfg["kind"] == "fasterrcnn":
        from aegisbench.models.fasterrcnn import train_from_config
    else:
        raise SystemExit(f"unknown kind {cfg['kind']}")

    best = train_from_config(args.config, Path(args.run_dir) / cfg["kind"])
    print(f"\nbest weights: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
