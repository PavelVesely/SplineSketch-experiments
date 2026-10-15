#!/usr/bin/env python3
"""Compile paper/ using plots regenerated from a specified profile's saved measurements."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reproduction.paper import build_paper


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=sorted(p.stem for p in (ROOT / "configs").glob("*.json")))
    parser.add_argument("--results", type=Path, help="Saved experiment directory (default: results/<profile>)")
    parser.add_argument("--output", type=Path, help="Build directory (default: <results>/paper)")
    args = parser.parse_args(argv)
    results = args.results or ROOT / "results" / args.profile
    output = args.output or results / "paper"
    try:
        pdf, report = build_paper(results, output)
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    if report["missing_panels"]:
        print(f"{len(report['missing_panels'])} panels have no measurements and show placeholders; see {output / 'paper-build.json'}.")
    if report["failed_measurements"] or report["numerical_failures"]:
        print(f"Run contains {report['failed_measurements']} failed measurements and "
              f"{report['numerical_failures']} numerical failures; these are omitted from curves.")
    print(f"Paper PDF: {pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
