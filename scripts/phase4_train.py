#!/usr/bin/env python3
"""Phase 4a: fine-tune one detector from its config.

  python scripts/phase4_train.py --config configs/train_yolo11.yaml
  python scripts/phase4_train.py --config configs/train_rtdetr.yaml
  python scripts/phase4_train.py --config configs/train_fasterrcnn.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", default="runs")
    args = ap.parse_args()

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
