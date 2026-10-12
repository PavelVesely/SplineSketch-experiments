"""Command-line interface; optional dependencies are imported only as needed."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .common import ROOT, digest, load_json
from .config import ABLATIONS, ALL_DATASETS, REAL, SKETCHES, plan, settings

SUITES = ["synthetic", "real", "update", "query", "merge", "heavy", "ablation"]


def parser():
    p = argparse.ArgumentParser(description="Reproduce SplineSketch experimental results (DOI 10.1145/3769827).")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="Create .venv, install pinned Python dependencies and compile Java (requires JDK 17)")
    commands.add_parser("build", help="Compile all Java programs")
    doctor = commands.add_parser("doctor", help="Check dependencies, bundled JAR hashes and compiled wrappers")
    doctor.add_argument("--datasets", nargs="+", choices=REAL)
    doctor.add_argument("--dataset-dir", type=Path, default=ROOT / "datasets")
    download = commands.add_parser("download", help="Fetch and validate official real-world inputs")
    download.add_argument("datasets", nargs="+", choices=REAL + ["all"])
    download.add_argument("--dataset-dir", type=Path, default=ROOT / "datasets")
    download.add_argument("--books-url", help="Override the Books mirror; upstream MD5 is still enforced")
    plot = commands.add_parser("plot", help="Regenerate figures/CSV without Java or original datasets")
    plot.add_argument("--output", type=Path, required=True, help="Existing experiment output directory")
    plot.add_argument("--figures", nargs="+", choices=["1", "4", "5", "6", "7", "8", "9", "10"])
    run = commands.add_parser("run", help="Generate inputs, execute measurements and plot (matching runs resume automatically)")
    run.add_argument("--profile", choices=["smoke", "paper"], default="smoke")
    run.add_argument("--config", type=Path, help="JSON overrides on top of the profile")
    run.add_argument("--suite", nargs="+", choices=SUITES, default=SUITES)
    run.add_argument("--datasets", nargs="+", help="Restrict workloads; see docs/REPRODUCIBILITY.md")
    run.add_argument("--ablations", nargs="+", choices=list(ABLATIONS))
    run.add_argument("--sketches", nargs="+", choices=SKETCHES + ["splinesketch_mg", "ddsketch"])
    run.add_argument("--sizes", nargs="+", type=int, help="Select sizes from the profile; custom mappings go in JSON")
    for argument in ("n", "queries", "jobs", "seed", "repeats", "merge-parts", "real-limit", "java-timeout", "max-plot-bytes"):
        run.add_argument("--" + argument, type=int)
    run.add_argument("--update-ns", nargs="+", type=int)
    run.add_argument("--query-counts", nargs="+", type=int)
    run.add_argument("--java-heap", help="Maximum Java heap per concurrent process (e.g. 1g, 100g)")
    run.add_argument("--output", type=Path, help="Default: results/<profile>")
    run.add_argument("--dataset-dir", type=Path, default=ROOT / "datasets")
    run.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/inputs")
    run.add_argument("--download", action="store_true", help="Fetch missing real-world inputs required by the selected tasks")
    run.add_argument("--dry-run", action="store_true", help="Print effective configuration/tasks without changing files")
    run.add_argument("--no-build", action="store_true", help="Use already compiled Java classes")
    return p


def validate(cfg):
    for key in ("n", "queries", "jobs", "repeats", "merge_parts", "java_timeout", "max_plot_bytes"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if cfg["real_limit"] is not None and (type(cfg["real_limit"]) is not int or cfg["real_limit"] <= 0):
        raise ValueError("real_limit must be null or a positive integer")
    for key in ("sizes", "moment_k", "kll_k", "update_ns", "query_counts"):
        if not cfg[key] or any(type(v) is not int or v <= 0 for v in cfg[key]):
            raise ValueError(f"{key} must contain positive integers")
        if len(set(cfg[key])) != len(cfg[key]):
            raise ValueError(f"Duplicate values in {key}")
    if len(cfg["sizes"]) != len(cfg["moment_k"]) or len(cfg["sizes"]) != len(cfg["kll_k"]):
        raise ValueError("sizes, moment_k and kll_k must have equal lengths")
    if min(cfg["sizes"]) < 6 or min(cfg["kll_k"]) < 8:
        raise ValueError("SplineSketch sizes must be >= 6 and KLL k must be >= 8")
    if not re.fullmatch(r"[1-9][0-9]*[kKmMgG]", cfg["java_heap"]):
        raise ValueError("java_heap must be a positive size, e.g. 1g")
    if set(cfg["sketches"]) - set(SKETCHES + ["splinesketch_mg", "ddsketch"]):
        raise ValueError("Unknown sketch in configuration")
    if not cfg["sketches"]:
        raise ValueError("At least one sketch is required")
    if set(cfg["real"]) - set(REAL):
        raise ValueError("Unknown real-world dataset in configuration")
    if set(cfg["synthetic"]) - (set(ALL_DATASETS) - set(REAL)):
        raise ValueError("Unknown synthetic dataset in configuration")
    for key in ("sketches", "synthetic", "real"):
        if len(set(cfg[key])) != len(cfg[key]):
            raise ValueError(f"Duplicate entries in {key}")
    if cfg["n"] > 2147483647 or any(n > 2147483647 for n in cfg["update_ns"]):
        raise ValueError("Java wrappers support at most 2**31-1 input values")


def doctor(args):
    from importlib.metadata import version
    from .java import PROGRAMS, check_java, program_directory
    info = check_java()
    print(json.dumps(info, indent=2))
    for name in ("numpy", "matplotlib", "joblib", "zstandard"):
        print(f"{name} {version(name)}")
    from packaging.requirements import Requirement
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate():
            if version(requirement.name) not in requirement.specifier:
                raise ValueError(f"Dependency version mismatch: {line}. Run python reproduce.py setup")
    manifest = load_json(ROOT / "dependency-manifest.json")
    for name, entry in manifest["jars"].items():
        if digest(ROOT / name) != entry["sha256"]:
            raise ValueError(f"Bundled dependency checksum mismatch: {name}")
    missing = [str((program_directory(program) / (program + ".class")).relative_to(ROOT))
               for program, _ in PROGRAMS.values()
               if not (program_directory(program) / (program + ".class")).exists()]
    if missing:
        raise ValueError(f"Missing compiled programs: {missing}. Run python reproduce.py build")
    if args.datasets:
        from .downloads import download_datasets
        # Existing inputs are validated. This command never initiates a download.
        from .data import input_paths
        absent = [str(p) for name in args.datasets for p in input_paths(name, args.dataset_dir) if not p.exists()]
        if absent:
            raise FileNotFoundError(f"Missing dataset files: {absent}")
        download_datasets(args.datasets, args.dataset_dir)
    print("Environment checks passed.")


def run(args):
    cfg = settings(args.profile, args.config)
    for key in cfg:
        value = getattr(args, key, None)
        if value is not None and key != "sizes":
            cfg[key] = value
    if args.sizes:
        indices = [cfg["sizes"].index(size) for size in args.sizes]
        for key in ("sizes", "moment_k", "kll_k"):
            cfg[key] = [cfg[key][i] for i in indices]
    validate(cfg)
    tasks = plan(cfg, args.suite, args.datasets, args.ablations)
    if not tasks:
        raise ValueError("No measurements selected. Select matching suites/datasets, or enable real datasets in the profile.")
    if args.dry_run:
        print(json.dumps(dict(config=cfg, measurements=len(tasks) * cfg["repeats"], tasks=tasks,
            limitations=["GKAdaptive merge is unavailable; see TODO.md."]), indent=2))
        return 0
    from .data import input_paths
    required = sorted({t["dataset"] for t in tasks if t["dataset"] in REAL})
    if args.download:
        from .downloads import download_datasets
        download_datasets(required, args.dataset_dir)
    missing = [str(p) for name in required for p in input_paths(name, args.dataset_dir) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}. Run python reproduce.py download {' '.join(required)} or add --download.")
    from .java import check_java, compile_java
    java = check_java()
    if not args.no_build:
        compile_java()
    from .runner import execute
    output = args.output or ROOT / "results" / args.profile
    print(f"{len(tasks) * cfg['repeats']} measurements, jobs={cfg['jobs']}, heap={cfg['java_heap']} per JVM. Output: {output}", flush=True)
    metadata = execute(cfg, tasks, output, args.dataset_dir, args.cache_dir, java)
    from .plots import plot_results
    plot_results(output)
    print(f"Results: {output}\nCounts: {metadata['counts']}")
    return 1 if metadata["counts"]["failed"] or metadata["counts"]["pending"] else 0


def setup():
    # The environment is local to the checkout and does not change system packages.
    if not (3, 9) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("Use Python 3.9–3.12 for the pinned artifact environment")
    subprocess.run([sys.executable, "-m", "venv", str(ROOT / ".venv")], check=True)
    executable = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(executable), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)
    subprocess.run([str(executable), str(ROOT / "reproduce.py"), "build"], check=True)
    subprocess.run([str(executable), str(ROOT / "reproduce.py"), "doctor"], check=True)
    print("Setup complete. Use .venv/bin/python reproduce.py run --profile smoke (Windows: .venv\\Scripts\\python.exe).")


def main(argv=None):
    args = parser().parse_args(argv)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    try:
        if args.command == "setup":
            setup()
        elif args.command == "doctor":
            doctor(args)
        elif args.command == "build":
            from .java import compile_java
            compile_java()
            print("Java compilation succeeded.")
        elif args.command == "download":
            from .downloads import download_datasets
            download_datasets(REAL if "all" in args.datasets else args.datasets, args.dataset_dir, args.books_url)
        elif args.command == "plot":
            from .plots import plot_results
            plot_results(args.output, args.figures)
            print(f"Figures and measurements.csv written to {args.output}")
        elif args.command == "run":
            return run(args)
    except (ImportError, RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Completed records are saved; rerun the same command to resume.", file=sys.stderr)
        return 130
    return 0


def legacy_main(suite):
    """Old script names now accept the same CLI options as the unified runner."""
    return main(["run", "--suite", suite] + sys.argv[1:])
