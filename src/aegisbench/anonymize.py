"""Double-blind hygiene scanner.

WACV desk-rejects submissions whose code archive leaks author identity.
This scanner walks the repository and flags:
  * terms listed in a LOCAL, GIT-IGNORED file `.anonymize-terms.txt`
    (one case-insensitive term per line: your name, employer, product
    names, domains — the terms themselves must never live in the repo);
  * the git author name/email configured for this checkout;
  * email addresses and URLs outside an allowlist of public dataset /
    library domains.

Run before exporting the submission archive:
    python -m aegisbench.anonymize --root .
Export with `git archive` (never ship the .git directory — its history and
config identify you).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

TEXT_EXTS = {".py", ".md", ".yaml", ".yml", ".txt", ".sh", ".cfg", ".toml",
             ".json", ".ipynb", ".tex", ".bib", ".html", ".csv"}
SKIP_DIRS = {".git", ".claude", "__pycache__", ".venv", "venv", "data",
             "runs", "results", ".pytest_cache", "node_modules"}
URL_ALLOWLIST = (
    "pytorch.org", "github.com/ultralytics", "docs.ultralytics.com",
    "ieee-dataport.org", "fesb.unist.hr", "cocodataset.org",
    "pypi.org", "arxiv.org", "opencv.org", "download.pytorch.org",
    "roboflow.com",
    # Conference, publisher, and badge domains: public and non-identifying,
    # so they must not force a --force build (see scripts/make_submission.py).
    "thecvf.com", "neurips.cc", "openreview.net", "img.shields.io",
    "doi.org", "link.springer.com", "ieeexplore.ieee.org",
    "openaccess.thecvf.com", "huggingface.co",
)
# TLD must be alphabetic so metric notation like "mAP@0.5" doesn't match.
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")
URL_RE = re.compile(r"https?://[^\s)\"'>]+")


def load_terms(root: Path) -> list[str]:
    f = root / ".anonymize-terms.txt"
    if not f.exists():
        return []
    return [t.strip().lower() for t in f.read_text().splitlines()
            if t.strip() and not t.startswith("#")]


def git_identity(root: Path) -> list[str]:
    terms = []
    for key in ("user.name", "user.email"):
        try:
            v = subprocess.run(["git", "config", key], cwd=root,
                               capture_output=True, text=True,
                               timeout=10).stdout.strip()
            if v:
                terms.append(v.lower())
        except Exception:
            pass
    return terms


def scan(root: str | Path) -> list[tuple[str, int, str]]:
    root = Path(root).resolve()
    terms = load_terms(root) + git_identity(root)
    findings: list[tuple[str, int, str]] = []
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTS:
            continue
        if path.name == ".anonymize-terms.txt":
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        rel = str(path.relative_to(root))
        for lineno, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            for term in terms:
                if term and term in low:
                    findings.append((rel, lineno, f"identity term: {term!r}"))
            for m in EMAIL_RE.finditer(line):
                if "noreply" not in m.group(0).lower():
                    findings.append((rel, lineno, f"email: {m.group(0)}"))
            for m in URL_RE.finditer(line):
                url = m.group(0)
                if not any(dom in url for dom in URL_ALLOWLIST):
                    findings.append((rel, lineno, f"non-allowlisted URL: "
                                     f"{url}"))
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    findings = scan(args.root)
    if not findings:
        print("anonymization scan: CLEAN")
        return 0
    print(f"anonymization scan: {len(findings)} finding(s)")
    for rel, lineno, msg in findings:
        print(f"  {rel}:{lineno}: {msg}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
