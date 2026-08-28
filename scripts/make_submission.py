#!/usr/bin/env python3
"""Build the anonymized supplementary archive for submission.

Assembles code, protocol docs, result tables, and figures into one ZIP,
after refusing to proceed if the anonymization scan finds anything. See
docs/SUBMISSION.md for what the archive must contain and why.

  python scripts/make_submission.py --out dist/

The code tree is exported with `git archive`, so the .git directory and
every git-ignored file are excluded by construction. Results and figures
are git-ignored working-tree artifacts and are copied in explicitly.
"""

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Working-tree artifacts to include, (source, archive-relative dest).
# Missing entries are reported and skipped, not fatal: a partial archive
# with a clear manifest beats a build that dies on one absent file.
RESULT_FILES = [
    ("results/sweep/master_ci.csv", "results/master_ci.csv"),
    ("results/sweep/ci_heridal.csv", "results/ci_heridal.csv"),
    ("results/sweep/ci_sard.csv", "results/ci_sard.csv"),
    ("results/sweep/localization_heridal.csv",
     "results/localization_heridal.csv"),
    ("results/sweep/localization_sard.csv", "results/localization_sard.csv"),
    ("results/sweep/summary_recall.csv", "results/summary_recall.csv"),
    ("results/phase3/calibration.csv", "results/calibration.csv"),
]

# phase5_figures.py writes wherever --out points; both results/figures (the
# path used for the submitted figures) and results/sweep (the older default)
# are searched so a run from either layout produces a complete archive.
# PDFs are the vector originals used in the paper; PNGs are for quick
# viewing. Both ship.
FIGURE_GLOBS = [
    ("results/figures/*.pdf", "figures"),
    ("results/figures/*.png", "figures"),
    ("results/sweep/heatmap_*.png", "figures"),
    ("results/sweep/severity_*.png", "figures"),
    ("results/sweep/decoupling_*.png", "figures"),
    ("results/sweep/gallery/*.png", "figures/gallery"),
    ("results/phase3/heridal_grid.jpg", "figures"),
]

DOCS = [
    ("docs/DATASHEET.md", "DATASHEET.md"),
    ("docs/RUNBOOK.md", "RUNBOOK.md"),
    ("docs/CORRUPTIONS.md", "CORRUPTIONS.md"),
    ("docs/DATA.md", "DATA.md"),
]

# Internal working documents, excluded from the shipped code tree. These are
# drafting aids, not artifacts a reviewer should receive: PAPER_GUIDE names
# the repository (and so the authors), SUBMISSION discusses the submission
# process itself, and HANDOFF / DRAFT_REVIEW / WRITING_PACK / FIGURES are
# authoring notes carrying [FILL] markers, deadline strategy, and
# section-by-section drafting instructions. The four polished docs
# (DATASHEET, RUNBOOK, CORRUPTIONS, DATA) ship at the archive root instead,
# via DOCS above.
EXCLUDE_FROM_CODE = (
    "docs/PAPER_GUIDE.md",
    "docs/SUBMISSION.md",
    "docs/HANDOFF.md",
    "docs/DRAFT_REVIEW.md",
    "docs/WRITING_PACK.md",
    "docs/FIGURES.md",
)

README = """# AegisBench: supplementary material

Anonymous supplementary archive for the AegisBench submission: a robustness
benchmark for aerial search-and-rescue person detection under
disaster-grounded visual corruption.

## What is here

| Path | Contents |
| --- | --- |
| `DATASHEET.md` | Structured datasheet: provenance, composition, licensing, intended use, limitations, responsible-AI notes |
| `CORRUPTIONS.md` | Physical grounding and calibration methodology for all nine corruptions |
| `RUNBOOK.md` | The phased experimental protocol with per-phase verification checkpoints |
| `DATA.md` | How to obtain the source datasets, expected layouts, split policy |
| `code/` | Complete source: corruption engine, tiling pipeline, shared evaluator, detector wrappers, phase scripts, tests |
| `results/` | Every measurement reported in the paper, plus the full 168-condition sweep table |
| `figures/` | Robustness heatmaps, severity curves, decoupling scatters, the corruption taxonomy panel, and the qualitative failure gallery (`figures/gallery/`) |

## License

The AegisBench code, configuration, and results in this archive are released
under the MIT License. HERIDAL and SARD remain governed by their own
licenses, which this project neither alters nor sublicenses, and no imagery
from either is redistributed here.

## What is deliberately not here

**Source imagery.** AegisBench does not redistribute HERIDAL or SARD, whose
licenses require obtaining them from their original distributors. It is not
needed for verification: every stochastic element of every corruption is
seeded by a SHA-256 hash of `(image_id, corruption, severity, global_seed)`,
so a reader who obtains the clean datasets under their own licenses
regenerates the corrupted benchmark bit-identically. See `DATA.md`.

**Archived per-condition predictions.** Regenerable from the released code
and excluded to keep the archive small.

## Reproducing the reported numbers

`RUNBOOK.md` is the authoritative sequence. Briefly:

1. Obtain HERIDAL and SARD (`DATA.md`), then run the Phase 1 preparation
   scripts. For SARD, review the printed group table: splitting is
   group-aware by video sequence to prevent near-duplicate leakage.
2. Phase 2 tiles the high-resolution HERIDAL frames; Phase 3 verifies the
   corruption engine and re-measures the severity calibration statistics
   reported in the paper's taxonomy table.
3. Phase 4 trains the clean baselines; Phase 5 runs the 168-condition
   sweep, then the bootstrap confidence intervals and localization-stability
   analysis.

Each phase ends with an artifact to inspect before the next is allowed to
run.

## Results files

| File | Contents |
| --- | --- |
| `results/master_ci.csv` | The full sweep: one row per (detector, dataset, corruption, severity), with git SHA, corruption-config hash, seed, and timestamp per row |
| `results/ci_heridal.csv`, `results/ci_sard.csv` | Bootstrap 95% confidence intervals, 1,000 resamples per condition |
| `results/localization_heridal.csv`, `results/localization_sard.csv` | Localization stability over instances detected in both the clean and corrupted conditions |
| `results/calibration.csv` | Measured severity statistics underlying the corruption taxonomy table |
| `results/summary_recall.csv` | Per-family aggregate relative recall drop |

Every number in the paper traces to a row in one of these files.
"""


_FINDING_RE = re.compile(r"^  (\S+):\d+: ")


def run_anonymization_scan(root: Path) -> int:
    """Returns the BLOCKING finding count; -1 if the scanner could not run.

    The scanner walks the whole repo, including docs/PAPER_GUIDE.md and the
    other internal working documents in EXCLUDE_FROM_CODE that never ship
    (see export_code). A hit inside one of those files is real text on
    disk, but not a leak into anything a reviewer will see, so it does not
    block the build; it is still printed, just labeled, so nothing is
    silently hidden. A hit anywhere else still blocks the build exactly as
    before.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "aegisbench.anonymize", "--root", str(root)],
            cwd=root, capture_output=True, text=True, timeout=300,
            env={"PYTHONPATH": str(root / "src"), "PATH": ""})
    except Exception as exc:
        print(f"  could not run the scanner: {exc}")
        return -1

    out = (proc.stdout or "") + (proc.stderr or "")
    if "anonymization scan: CLEAN" in out:
        print(out.rstrip())
        return 0

    if not any(_FINDING_RE.match(line) for line in out.splitlines()):
        # The scanner produced output but no line matched the expected
        # "  path:line: message" format, so this isn't a normal
        # clean/findings report -- treat it as a scanner failure rather
        # than silently returning 0.
        print(out.rstrip())
        return -1

    blocking = 0
    for line in out.splitlines():
        m = _FINDING_RE.match(line)
        if m and m.group(1) in EXCLUDE_FROM_CODE:
            print(f"{line}  [expected: this file is never shipped]")
        else:
            print(line)
            if m:
                blocking += 1
    print(f"  {blocking} blocking finding(s) outside the excluded docs")
    return blocking


def export_code(root: Path, dest: Path) -> None:
    """git archive HEAD -> dest, so .git and git-ignored files never ship."""
    dest.mkdir(parents=True, exist_ok=True)
    tar = dest.parent / "_code.tar"
    subprocess.run(["git", "archive", "--format=tar", "-o", str(tar), "HEAD"],
                   cwd=root, check=True, timeout=300)
    shutil.unpack_archive(str(tar), str(dest), format="tar")
    tar.unlink()
    for rel in EXCLUDE_FROM_CODE:
        p = dest / rel
        if p.exists():
            p.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist",
                    help="directory to write the archive into")
    ap.add_argument("--name", default="aegisbench_supplementary")
    ap.add_argument("--force", action="store_true",
                    help="build even if the anonymization scan reports "
                         "findings. Only for findings you have personally "
                         "confirmed are false positives.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = out_dir / args.name

    print("== anonymization scan ==")
    findings = run_anonymization_scan(root)
    if findings != 0 and not args.force:
        print("\nREFUSING TO BUILD.")
        print("Fix the findings above, or rerun with --force if you have "
              "personally confirmed every one is a false positive.")
        print("WACV desk-rejects on identity leaks; this check is the last "
              "automated guard before upload.")
        return 1
    if findings != 0:
        print("\n--force given: building despite findings. You have asserted "
              "these are false positives.")

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    print("\n== code (git archive HEAD) ==")
    export_code(root, staging / "code")
    n_code = sum(1 for _ in (staging / "code").rglob("*") if _.is_file())
    print(f"  {n_code} files")

    print("\n== docs ==")
    (staging / "README.md").write_text(README)
    print("  README.md (generated)")
    for src, dst in DOCS:
        s = root / src
        if s.exists():
            shutil.copy2(s, staging / dst)
            print(f"  {dst}")
        else:
            print(f"  MISSING: {src}")

    print("\n== results ==")
    (staging / "results").mkdir(exist_ok=True)
    missing_results = []
    for src, dst in RESULT_FILES:
        s = root / src
        if s.exists():
            shutil.copy2(s, staging / dst)
            print(f"  {dst}")
        else:
            missing_results.append(src)
            print(f"  MISSING: {src}")

    print("\n== figures ==")
    n_figs = 0
    for pattern, dest_sub in FIGURE_GLOBS:
        d = staging / dest_sub
        d.mkdir(parents=True, exist_ok=True)
        for p in sorted(root.glob(pattern)):
            shutil.copy2(p, d / p.name)
            n_figs += 1
    print(f"  {n_figs} figure file(s)")
    if n_figs == 0:
        print("  none found. Generate them with scripts/phase5_figures.py "
              "and scripts/phase5_gallery.py before the final build.")

    print("\n== zip ==")
    zip_path = out_dir / f"{args.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(staging.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(staging.parent))
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"  {zip_path}  ({size_mb:.1f} MB)")

    limit = 200
    if size_mb > limit:
        print(f"\nOVER THE {limit} MB SUPPLEMENTARY LIMIT. Drop the largest "
              "figures or downsample the gallery images before uploading.")
        return 1
    print(f"  within the {limit} MB supplementary limit")

    if missing_results:
        print("\nMissing result files (archive built without them):")
        for m in missing_results:
            print(f"  {m}")

    print("\nNext: open the ZIP and read its README as a reviewer would, "
          "then upload it by the supplementary deadline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
