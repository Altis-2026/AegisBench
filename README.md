# AegisBench

A robustness benchmark for aerial search-and-rescue (SAR) person detection
under **physically motivated, disaster-grounded visual corruptions**, plus a
training-time mitigation study.

Existing SAR person-detection benchmarks are collected in calm, clear
conditions. AegisBench measures how modern detectors degrade when the same
imagery is subjected to the optical signatures of the disasters SAR systems
actually deploy into — flood (glare, turbidity, inundation), wildfire
(smoke haze, fire-warm tint), storm (rain, wind blur, low light), and
post-disaster dust — each at three calibrated severities.

## What's here

| Path | Purpose |
| --- | --- |
| `configs/corruptions.yaml` | The canonical corruption parameter table (9 corruptions x 3 severities, each calibrated by a measurable image statistic) |
| `src/aegisbench/corruptions/` | Deterministic corruption engine (pure numpy/OpenCV, no GPU needed) |
| `src/aegisbench/tiling.py` | Overlapping-tile pipeline for 4000x3000 frames with exact bbox remapping |
| `src/aegisbench/evaluation/` | One shared evaluator for every detector: COCO mAP, fixed-operating-point P/R, size-stratified recall, relative-drop robustness metrics |
| `src/aegisbench/models/` | YOLOv11 / RT-DETR (ultralytics) and Faster R-CNN (torchvision) behind one interface |
| `scripts/phase*.py` | The phased experimental protocol with stop-and-inspect checkpoints |
| `docs/RUNBOOK.md` | Exactly what to run, phase by phase, and what to verify at each checkpoint |
| `docs/CORRUPTIONS.md` | Physical grounding and calibration of every corruption |
| `docs/DATA.md` | How to obtain HERIDAL and SARD, expected layouts, split rules |

## Quickstart

```bash
# 1. Environment (GPU machine; see docs/RUNBOOK.md for the Blackwell note)
bash scripts/setup_env_sm120.sh
python scripts/phase0_check_gpu.py            # must print PHASE 0: PASS

# 2. Unit tests + a GPU-free demo of the corruption engine and tiling
pip install -e ".[dev]" && pytest
python scripts/make_synthetic_samples.py --out data/synthetic --n 3
python scripts/phase3_generate_grid.py \
    --image data/synthetic/synthetic_000.jpg --out results/phase3/grid.jpg
```

Then follow `docs/RUNBOOK.md` from Phase 1 (data) through Phase 6
(mitigation). Every phase ends with an artifact to inspect before the next
phase is allowed to run.

## Reproducibility

* Every stochastic element of every corruption is seeded by a stable hash
  of `(image_id, corruption, severity, global_seed)` — the corrupted test
  sets are fixed datasets, bit-identical on any machine.
* Training configs (`configs/train_*.yaml`) carry fixed seeds and are
  copied verbatim into each run directory.
* The sweep CSV logs git SHA, corruption-config hash, seed, and timestamp
  per row, and is resumable.
* One evaluator scores every model; operating points are selected on clean
  validation data once and frozen across all corruption conditions.

## Double-blind hygiene

Run `python -m aegisbench.anonymize --root .` before exporting any
submission archive, and export with `git archive` (never ship `.git`).
See the notes at the top of `src/aegisbench/anonymize.py`.
