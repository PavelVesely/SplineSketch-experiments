#!/usr/bin/env python3
"""Create a reviewable archive of this working tree, excluding downloaded data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/splinesketch-artifact.zip")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Archive already exists: {args.output}; choose a new --output")
    files = [p for p in ROOT.iterdir() if p.is_file() and
             (p.suffix in (".py", ".java", ".jar", ".md", ".json", ".txt", ".sh") or
              p.name in ("Makefile", "Dockerfile", "LICENSE", ".dockerignore", ".gitignore"))]
    for directory in ("configs", "reproduction", "docs", "tests", "scripts", "java", "third_party", "plots"):
        files += [p for p in (ROOT / directory).rglob("*") if p.is_file() and
                  "__pycache__" not in p.parts and p.suffix not in (".pyc", ".class")]
    commit = None
    if shutil.which("git"):
        process = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
        commit = process.stdout.strip() if process.returncode == 0 else None
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            archive.write(p, "SplineSketch-experiments/" + str(p.relative_to(ROOT)))
        archive.writestr("SplineSketch-experiments/ARTIFACT-MANIFEST.json", json.dumps(
            dict(git_commit=commit, files=hashes), indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
