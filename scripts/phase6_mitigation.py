#!/usr/bin/env python3
"""Phase 6: disaster-aware training augmentation (mitigation) + ablation.

Strategy: pre-generate corrupted variants of the TRAIN tiles (deterministic,
seeded per tile) so any trainer consumes them unmodified. Per source tile we
keep the clean copy and add `variants` corrupted copies, each with a
corruption drawn from the strategy's pool and a severity drawn uniformly
from --severities (default {1,2,3}).

Hold the evaluation severity out of training. Training on severity 1-2
and re-evaluating at severity 3 measures whether the augmentation
generalizes to a condition it never saw; training on all three and
re-evaluating at severity 3 measures recall of a seen condition, which
is a much weaker claim.

Ablation arms (--strategy):
  all        pool = every corruption          (the full mitigation)
  worst:<name>  pool = single named corruption (e.g. worst:smoke_haze)
  family:<fam>  pool = one family              (e.g. family:flood)

After generating, retrain with phase4_train.py pointing at the augmented
dataset yaml, then re-run phase5_sweep.py with the new weights; the
before/after delta per corruption is the mitigation result.

  python scripts/phase6_mitigation.py --tiles data/heridal/tiles \
      --out data/heridal/tiles_aug_all --strategy all --variants 1

The GRSL mitigation arm, targeting the low-light collapse on SARD and
holding severity 3 out of training:

  python scripts/phase6_mitigation.py --tiles data/sard/tiles \
      --out data/sard/tiles_aug_lowlight_s12 \
      --strategy worst:low_light --severities 1,2 --variants 1
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.corruptions import (DEFAULT_CONFIG_PATH, SEVERITIES,
                                    CorruptionSuite)
from aegisbench.datasets.common import write_dataset_yaml
from aegisbench.seeding import rng_for


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True,
                    help="phase 2 output dir (train/ + val/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strategy", default="all")
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--severities", default=",".join(str(s) for s in SEVERITIES),
                    help="comma-separated severities to draw from, e.g. '1,2' "
                         "to hold severity 3 out of training so the re-eval "
                         "at severity 3 measures generalization rather than "
                         "recall of a seen condition")
    args = ap.parse_args()

    severities = [int(s) for s in args.severities.split(",") if s.strip()]
    unknown = sorted(set(severities) - set(SEVERITIES))
    if unknown:
        raise SystemExit(f"unknown severities {unknown}, allowed {list(SEVERITIES)}")
    if not severities:
        raise SystemExit("empty severity pool")

    suite = CorruptionSuite(args.config)
    if args.strategy == "all":
        pool = suite.names()
    elif args.strategy.startswith("worst:"):
        pool = [args.strategy.split(":", 1)[1]]
    elif args.strategy.startswith("family:"):
        fam = args.strategy.split(":", 1)[1]
        pool = [n for n in suite.names() if suite.family(n) == fam]
    else:
        raise SystemExit(f"unknown strategy {args.strategy}")
    if not pool:
        raise SystemExit("empty corruption pool")
    print(f"strategy={args.strategy} pool={pool} severities={severities}")

    src = Path(args.tiles)
    out = Path(args.out)
    # Val stays CLEAN (operating-point selection must not shift with the
    # augmentation), train gets clean + corrupted copies.
    if (out / "val").exists():
        shutil.rmtree(out / "val")
    shutil.copytree(src / "val", out / "val")

    img_out = out / "train" / "images"
    lbl_out = out / "train" / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    src_images = sorted((src / "train" / "images").glob("*.jpg"))
    n_aug = 0
    for i, img_path in enumerate(src_images):
        lbl_path = src / "train" / "labels" / f"{img_path.stem}.txt"
        shutil.copy2(img_path, img_out / img_path.name)
        if lbl_path.exists():
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)

        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        for v in range(args.variants):
            rng = rng_for(img_path.stem, f"mitigation-{args.strategy}", v,
                          global_seed=suite.global_seed)
            name = pool[int(rng.integers(len(pool)))]
            sev = severities[int(rng.integers(len(severities)))]
            corrupted = suite.apply(img, name, sev,
                                    f"{img_path.stem}-aug{v}")
            stem = f"{img_path.stem}__aug{v}_{name}_s{sev}"
            cv2.imwrite(str(img_out / f"{stem}.jpg"),
                        cv2.cvtColor(corrupted, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_out / f"{stem}.txt")
            n_aug += 1
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(src_images)} tiles processed")

    write_dataset_yaml(out / "dataset.yaml", out,
                       "train/images", "val/images")
    print(f"\n{len(src_images)} clean + {n_aug} corrupted tiles -> {out}")
    print("Corruption is applied per-tile here (training augmentation); "
          "test-time corruption is always full-image before tiling.")
    print(f"Next: retrain via phase4_train.py with data_yaml={out}/"
          "dataset.yaml, then re-run phase5_sweep.py with the new weights.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
