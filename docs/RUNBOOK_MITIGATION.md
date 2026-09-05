# Mitigation runs: SARD low-light augmentation

Copy-paste sequence for the three mitigation arms that go into the GRSL
letter. These need the GPU box; they cannot run in a code-only checkout.

Everything here is prepared and committed. Nothing in this file has been
executed yet, so no result from it is in the paper.

## What the experiment is

Fine-tune each of the three clean SARD detectors on training tiles
augmented with `low_light` at **severities 1 and 2 only**, then
re-evaluate at **severity 3** through the unchanged frozen-threshold
pipeline. The before/after delta at severity 3 is the mitigation result.

Severity 3 is held out of training on purpose. Training on all three
severities and reporting severity 3 would measure recall of a condition
the model was trained on, which is a much weaker claim than
generalization to an unseen severity, and a reviewer will ask which one
it is.

Thresholds stay frozen at the clean-run values (0.30 YOLOv11, 0.60
RT-DETR, 0.96 Faster R-CNN). Re-tuning them on the mitigated models would
confound the augmentation with a threshold change.

## 0. Preconditions

```
ls runs/yolo11/yolo11_sard_clean/weights/best.pt
ls runs/rtdetr/rtdetr_sard_clean/weights/best.pt
ls runs/fasterrcnn/fasterrcnn_sard_clean_best.pt
ls data/sard/tiles/train/images | head -3
```

All four must exist. The Faster R-CNN path wants the `_best.pt`
state_dict, not `_last_ckpt.pt`; passing the checkpoint is rejected with a
message saying so.

## 1. Build the augmented training set (CPU, ~minutes)

```
python scripts/phase6_mitigation.py \
    --tiles data/sard/tiles \
    --out data/sard/tiles_aug_lowlight_s12 \
    --strategy worst:low_light \
    --severities 1,2 \
    --variants 1
```

Validation stays clean, which the script enforces: operating points must
not shift with the augmentation.

Check the output before training on it. Severity 3 must not appear:

```
ls data/sard/tiles_aug_lowlight_s12/train/images | grep -c '__aug0_low_light_s3' || echo "good, no s3"
ls data/sard/tiles_aug_lowlight_s12/train/images | wc -l
```

## 2. Train the three arms (GPU, sequential)

One GPU, so these run one after another. Rough order-of-magnitude only;
time them on the first run rather than trusting these.

```
python scripts/phase4_train.py --config configs/train_yolo11_sard_aug_lowlight.yaml
python scripts/phase4_train.py --config configs/train_rtdetr_sard_aug_lowlight.yaml
python scripts/phase4_train.py --config configs/train_fasterrcnn_sard_aug_lowlight.yaml
```

Each is a short fine-tune from a converged checkpoint (20, 20, and 8
epochs) rather than a fresh run, so the delta isolates the augmentation.

If a run dies partway, resume rather than restarting:

```
python scripts/phase4_train.py --resume runs/yolo11/yolo11_sard_aug_lowlight/weights/last.pt
```

If the `torch.AcceleratorError` CUDA crashes recur, the documented stable
fallback is `amp: false` with `batch: 2`, as recorded in
`configs/train_yolo11_heridal_aug_lowlight.yaml`. Changing batch changes
the effective learning-rate schedule, so write down which setting
produced the number you report.

## 3. Re-sweep at the frozen thresholds

```
python scripts/phase5_sweep.py \
    --models configs/sweep_models_sard_mitigation.yaml \
    --records data/sard/records \
    --dataset sard
```

Then recompute intervals so the mitigation numbers carry the same
bootstrap treatment as everything else in the paper:

```
python scripts/phase5_bootstrap_ci.py   # check its --help for the exact args
```

## 4. Get the numbers back to the paper

Commit the new CSV rows, exactly as the earlier results were committed:

```
git add -f results/sweep/*.csv
git commit -m "Add SARD low-light mitigation sweep results"
git push -u origin claude/grsl-reframe-mitigations-829rcz
```

That is the whole handoff. Once those rows are in the repo, the mitigation
table can be written and verified against them.

## 5. What goes in the paper

The letter currently states in Limitations and Conclusion that no
mitigation is reported. When these results land, that changes to a short
results subsection plus one table: clean, severity-3 before, severity-3
after, per detector, with bootstrap intervals.

Space is tight. The letter is at exactly 5 pages, so the mitigation
subsection has to be paid for out of existing text. The most compressible
material is the calibration discussion in Section II-C and the real-night
paragraph in Section IV-D.

`paper/grsl/Makefile` fails the build if the result goes over 5 pages, if
a line runs past its column, or if any number disagrees with its source
CSV. Add the new table's checks to `scripts/verify_grsl_numbers.py` at
the same time as the table itself, so the gate keeps covering everything.

## Honest note on what this can show

A single fine-tune arm per architecture, one seed, on one dataset, is
enough to demonstrate that the collapse is addressable. It is not enough
to claim a general fix. If all three arms recover substantial recall at
severity 3, the claim is "targeted augmentation recovers X to Y points of
recall at a severity never seen in training, on SARD". If they do not,
that is also a publishable result and a more interesting one, because it
argues the failure is architectural rather than a data-coverage gap. Do
not reach for a stronger claim than the runs support in either direction.
