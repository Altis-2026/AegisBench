# AegisBench runbook

The experimental protocol, phase by phase. **Each phase ends with a
checkpoint artifact. Inspect it and only continue when it passes.** Later
phases assume earlier checkpoints were verified — the sweep is meaningless
on top of a broken tiler or an implausible corruption.

Hardware assumption: one consumer GPU with 8 GB VRAM (tuned for an RTX
5070, Blackwell / sm_120). Everything except training and inference also
runs on CPU.

---

## Phase 0 — environment

```bash
bash scripts/setup_env_sm120.sh
python scripts/phase0_check_gpu.py
```

**Checkpoint:** the script prints the device, CUDA version, compiled arch
list, and a timed GPU matmul, ending in `PHASE 0: PASS`. Blackwell note:
a torch build that predates sm_120 imports fine but reports CUDA
unavailable — that is what the nightly cu128 wheel in the setup script
fixes. The check script, not the wheel label, is the source of truth.

Also run the CPU-side unit tests once per machine:

```bash
pytest
```

## Phase 1 — data

Download HERIDAL and SARD manually (licenses require it; see
`docs/DATA.md`), then:

```bash
python scripts/phase1_prepare_heridal.py \
    --train-images data/heridal/trainImages \
    --test-images  data/heridal/testImages \
    --out data/heridal/records
python scripts/phase1_prepare_sard.py \
    --images data/sard/images --labels data/sard/labels \
    --out data/sard/records
python scripts/phase1_visual_check.py \
    --records data/heridal/records/train.json \
    --out results/phase1/heridal --n 5
python scripts/phase1_visual_check.py \
    --records data/sard/records/train.json \
    --out results/phase1/sard --n 5
```

**Checkpoint:** (a) image/box counts match expectations (HERIDAL:
~1500-1600 train / ~101 test full-size images; verify exact numbers
against your downloaded archive — do not quote counts you didn't
measure); (b) every drawn box sits on a person; (c) for SARD, the printed
group table shows plausible video-sequence groups — this is the
train/test-leakage guard, do not wave it through.

## Phase 2 — tiling

```bash
python scripts/phase2_tile_dataset.py \
    --records data/heridal/records --out data/heridal/tiles \
    --tile 1024 --overlap 256
python scripts/phase2_visual_check.py \
    --records data/heridal/records/train.json \
    --out results/phase2 --n 5
```

Box rule (also documented in `src/aegisbench/tiling.py` for the paper):
clip to tile, keep if >=30% of the original box area is visible, drop
slivers under 4 px; with 256 px overlap >= the largest person box, every
box is fully contained in >=1 tile, so nothing is lost dataset-wide.

**Checkpoint:** the visual check prints a numeric round-trip PASS per
image and writes tile overlays (green = fully contained, yellow = clipped
edge copy). Zero containment violations expected; if the report lists
any, the overlap is smaller than some person box — bump `--overlap`.

SARD frames (640x640 in the Roboflow re-export evaluated here) go through
the same tiler, but are smaller than the 1024 px tile size, so each frame
passes through as a single full-frame tile rather than being split.

## Phase 3 — corruption engine

```bash
python scripts/phase3_generate_grid.py \
    --image <a real clean HERIDAL image> --out results/phase3/grid.jpg
python scripts/phase3_calibrate.py \
    --records data/heridal/records/test.json \
    --out results/phase3/calibration.csv --n 20
```

**Checkpoint:** (a) the grid looks physically plausible cell by cell —
severe smoke reads as smoke, not noise; dust reads browner and patchier
than smoke; inundation occludes rather than tints; (b) the calibration
audit prints `OK` (monotonic) for every corruption and the CSV numbers go
into the paper's taxonomy table.

## Phase 4 — clean baselines

```bash
python scripts/phase4_train.py --config configs/train_yolo11.yaml
python scripts/phase4_train.py --config configs/train_rtdetr.yaml
python scripts/phase4_train.py --config configs/train_fasterrcnn.yaml

python scripts/phase4_eval_clean.py --kind yolo11 \
    --weights runs/yolo11/yolo11_heridal_clean/weights/best.pt \
    --records data/heridal/records --dataset heridal --out results/phase4
# repeat for rtdetr / fasterrcnn, and for SARD records
```

**Checkpoint:** clean test metrics per model, with the printed comparison
against the published HERIDAL reference point (~0.90 P / 0.893 R /
0.834 mAP@0.5 for YOLOv5L). Modest deltas are expected across
architectures; large ones mean the pipeline is broken — debug before
Phase 5. VRAM notes: batch sizes in the configs fit 8 GB with AMP; if you
OOM, halve batch before touching imgsz.

## Phase 5 — stress-test sweep

```bash
# fill weight paths into configs/sweep_models.yaml first
python scripts/phase5_sweep.py --models configs/sweep_models.yaml \
    --records data/heridal/records --dataset heridal \
    --out results/sweep/master.csv --pred-dir results/sweep/preds
python scripts/phase5_sweep.py --models configs/sweep_models.yaml \
    --records data/sard/records --dataset sard \
    --out results/sweep/master.csv --pred-dir results/sweep/preds
python scripts/phase5_heatmap.py --csv results/sweep/master.csv \
    --out results/sweep
```

The sweep is resumable (already-logged cells are skipped) and runs
3 models x (1 clean + 9 corruptions x 3 severities) x 2 datasets =
168 evaluation passes, inference-only.

**Checkpoint:** the master CSV and per-dataset heatmaps of relative
recall drop; every later paper claim about "which corruption breaks which
model" must trace to a row of this CSV.

## Phase 6 — mitigation + ablation

```bash
# full mitigation arm
python scripts/phase6_mitigation.py --tiles data/heridal/tiles \
    --out data/heridal/tiles_aug_all --strategy all --variants 1
# ablation arms, e.g. worst corruption only (identify it from Phase 5):
python scripts/phase6_mitigation.py --tiles data/heridal/tiles \
    --out data/heridal/tiles_aug_worst --strategy worst:smoke_haze
# per-family arm:
python scripts/phase6_mitigation.py --tiles data/heridal/tiles \
    --out data/heridal/tiles_aug_flood --strategy family:flood
```

Copy `configs/train_yolo11.yaml`, point `data_yaml` at each augmented
set (new `run_name` per arm), retrain, then re-run the Phase 5 sweep with
the new weights appended to a copy of `sweep_models.yaml`. The
before/after recall per corruption is the mitigation table; the arms are
the ablation.

**Checkpoint:** per-corruption recovery table. Report honestly: partial
recovery is a finding, not a failure.

---

## Cross-cutting rules

* Never evaluate with a threshold tuned on corrupted data — operating
  points come from clean val only (phase4/5 scripts enforce this).
* Corruptions at test time are applied to the FULL image before tiling;
  training augmentation corrupts tiles (documented approximation).
* Multiple seeds: if GPU time allows, repeat Phase 4/6 training with
  2-3 seeds (edit `seed:` in the config; keep every run's CSV rows). If
  not, say so explicitly in the paper rather than presenting single-run
  numbers as stable.
* Before ANY submission export: `python -m aegisbench.anonymize --root .`
  and export with `git archive` so `.git` never ships.
