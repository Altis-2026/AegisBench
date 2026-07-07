#!/usr/bin/env python3
"""Phase 3 checkpoint: corruption grid for eyeball plausibility review.

Rows = clean + each corruption, columns = severities 1..3. Severe smoke
must look like severe smoke, not noise — do not accept the engine until
this grid looks physically right.

  python scripts/phase3_generate_grid.py --image path/to/clean.jpg \
      --out results/phase3/grid.jpg
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.corruptions import DEFAULT_CONFIG_PATH, CorruptionSuite
from aegisbench.inference import load_rgb
from aegisbench.visualize import corruption_grid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = ap.parse_args()

    suite = CorruptionSuite(args.config)
    img = load_rgb(args.image)
    path = corruption_grid(img, suite, Path(args.image).stem, args.out)
    print(f"wrote {path} ({len(suite.names())} corruptions x 3 severities)")
    print("CHECKPOINT: confirm each cell is physically plausible before "
          "any corrupted evaluation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
