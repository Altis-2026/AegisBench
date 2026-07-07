#!/usr/bin/env python3
"""Generate SYNTHETIC aerial-like test images with known ground-truth boxes.

These are for pipeline verification only (tiling overlays, corruption
grids, unit tests) on machines without the real datasets. They are clearly
not real data and must never enter any reported experiment.

  python scripts/make_synthetic_samples.py --out data/synthetic --n 3
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.corruptions.noise_fields import value_noise
from aegisbench.seeding import stable_seed


def make_terrain(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Grass/scrub/rock terrain from layered noise, RGB uint8."""
    base = value_noise((h, w), rng, octaves=6, base_res=3)
    detail = value_noise((h, w), rng, octaves=4, base_res=60)
    grass = np.stack([0.30 + 0.15 * base, 0.42 + 0.20 * base,
                      0.18 + 0.10 * base], -1)
    rock = np.stack([0.45 + 0.1 * detail] * 3, -1) * \
        np.array([1.05, 1.0, 0.92])
    rock_mask = (value_noise((h, w), rng, octaves=3, base_res=5)
                 > 0.62)[..., None].astype(np.float32)
    img = grass * (1 - rock_mask) + rock * rock_mask
    img += 0.06 * (detail[..., None] - 0.5)
    # A dirt path for scale context.
    ys = np.arange(h)
    path_x = (w * 0.3 + w * 0.25
              * np.sin(ys / h * 3.0)).astype(int)
    for y in range(0, h, 2):
        cv2.circle(img, (int(path_x[y]), y), max(6, w // 90),
                   (0.52, 0.44, 0.33), -1)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def add_person(img: np.ndarray, rng: np.random.Generator,
               px_height: int) -> list[float]:
    """Paint a small person-like figure; returns [x1, y1, x2, y2]."""
    h, w = img.shape[:2]
    ph = px_height
    pw = max(4, int(ph * 0.45))
    x = int(rng.uniform(pw, w - 2 * pw))
    y = int(rng.uniform(ph, h - 2 * ph))
    shirt = [(220, 40, 40), (30, 60, 200), (230, 120, 20),
             (240, 220, 60)][int(rng.integers(4))]
    pants = (40, 40, 60)
    skin = (200, 160, 130)
    cv2.ellipse(img, (x, y + int(0.65 * ph)), (pw // 2, int(0.35 * ph)),
                0, 0, 360, pants, -1)
    cv2.ellipse(img, (x, y + int(0.3 * ph)), (pw // 2, int(0.3 * ph)),
                0, 0, 360, shirt, -1)
    cv2.circle(img, (x, y), max(2, ph // 6), skin, -1)
    pad = 2
    return [x - pw // 2 - pad, y - ph // 6 - pad,
            x + pw // 2 + pad, y + ph + pad]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--width", type=int, default=2000)
    ap.add_argument("--height", type=int, default=1500)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(args.n):
        rng = np.random.default_rng(stable_seed("synthetic-sample", i))
        img = make_terrain(args.height, args.width, rng)
        boxes = [add_person(img, rng,
                            int(rng.uniform(25, 55)))
                 for _ in range(int(rng.integers(4, 9)))]
        image_id = f"synthetic_{i:03d}"
        path = out / f"{image_id}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        records.append({"image_id": image_id, "image_path": str(path),
                        "width": args.width, "height": args.height,
                        "boxes": boxes})
        print(f"wrote {path} ({len(boxes)} synthetic persons)")

    with open(out / "records.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"wrote {out}/records.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
