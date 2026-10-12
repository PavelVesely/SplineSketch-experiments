# SplineSketch experimental artifact

This repository accompanies **SplineSketch: Even More Accurate Quantiles with
Error Guarantees**, Aleksander Łukasiewicz, Jakub Tětek, and Pavel Veselý,
PACMMOD 3(6), Article 362 (2025), presented at SIGMOD 2026.
[Published paper](https://doi.org/10.1145/3769827).

The artifact builds the Java implementations, generates or downloads inputs,
executes experiments, saves raw rank estimates and measurements, and produces
the experimental figures in the published paper. Existing PDFs in `plots/` are
historical reference plots and are never overwritten. Newly reproduced figures
are saved under `results/<run>/figures/`.

## Start with a smoke run

Install **JDK 17** (`java` and `javac` on PATH) and **Python 3.8 or newer**, then:

```sh
git clone https://github.com/PavelVesely/SplineSketch-experiments.git
cd SplineSketch-experiments
python3 reproduce.py setup
.venv/bin/python reproduce.py run --profile smoke
```

The smoke profile runs 76 small measurements across all five main sketches,
SplineSketch+MG, and both published ablations. It needs no dataset downloads.
Expect about a minute and less than 2 GB RAM with one job; these are sanity
checks, not paper-scale accuracy or timing results. See the measured validation
report in [docs/VALIDATION.md](docs/VALIDATION.md).

On Windows use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.
The Python CLI constructs classpaths and subprocess arguments portably; the
Makefile and shell shortcut are intended for Linux/macOS. Native Windows and
the Docker image have not yet been validated on this checkout.

Alternatively, build the container and retain all output in mounted directories:

```sh
docker build -t splinesketch-artifact .
mkdir -p results datasets .cache
docker run --rm \
  -v "$PWD/results:/artifact/results" \
  -v "$PWD/datasets:/artifact/datasets" \
  -v "$PWD/.cache:/artifact/.cache" \
  splinesketch-artifact run --profile smoke
```

Setup uses dependency ranges so pip can select releases for Python 3.8–3.14,
including Python 3.13. There is no hard upper Python version limit; newer
interpreters depend on upstream package support. Local validation used Python
3.9; other versions need the same tests before relying on results.
For the historical Python 3.9–3.12 package versions, install
`requirements-pinned.txt` in a virtual environment. Actual direct dependency
versions are recorded in each run's metadata. To preserve the entire environment,
save `python -m pip freeze > requirements-local.txt` alongside your results.

The Dockerfile fixes Temurin to JDK 17.0.16+8 and pins Python packages. Its Ubuntu
OS packages are obtained from the distribution repositories at build time.
Archive the built image or its digest for a fully fixed OS image.

## Reproduce the published experiments

Preview the complete plan without creating files or downloading data:

```sh
.venv/bin/python reproduce.py run --profile paper --dry-run
```

On a server with sufficient memory and disk, one command fetches the three
real-world datasets, executes the full suite, and generates Figures 1 and 4–10:

```sh
.venv/bin/python reproduce.py run --profile paper --download
```

**The paper profile is large:** synthetic accuracy inputs contain 100 million
values; Books contains 800 million. It permits a 100 GB Java heap per job and
defaults to one job. Budget approximately 128 GB RAM and 200 GB free disk for
the full suite; 256 GB RAM gives more headroom for Books. These are conservative
planning estimates, not measured minima. Runtime is not yet measured for the
full suite. The paper used an AMD EPYC 7302 (3 GHz) server with 251 GB RAM and SSD.

All currently supported curves are automated. **GKAdaptive merge curves in
Figure 8 remain unavailable** because the repository has no GK merge
implementation. The author follow-up is isolated in [TODO.md](TODO.md); generated
merge panels explicitly label the missing curve.

Download just the required inputs, or run a selected experiment:

```sh
.venv/bin/python reproduce.py download hepmass power books
.venv/bin/python reproduce.py run --profile paper --suite real --datasets power \
  --output results/power
.venv/bin/python reproduce.py run --profile paper --suite synthetic merge \
  --datasets normal --output results/normal
.venv/bin/python reproduce.py run --profile paper --suite ablation \
  --ablations interpolations heuristicError --output results/ablations
```

Repeat the **same command** to resume completed measurements. Different
configurations, sources, datasets, or environments require a new output
directory. Regenerate plots and CSV without Java or original datasets:

```sh
.venv/bin/python reproduce.py plot --output results/paper
.venv/bin/python reproduce.py plot --output results/normal --figures 6 7 8
```

For complete figure mapping, expected trends, experiment protocols, workload
details, dataset preprocessing, resource estimates, and artifact submission
instructions, see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Configure or extend experiments

```sh
.venv/bin/python reproduce.py run --help
.venv/bin/python reproduce.py doctor
.venv/bin/python -m unittest discover -s tests -v
```

[configs/paper.json](configs/paper.json) contains published-scale settings.
[configs/smoke.json](configs/smoke.json) contains the small validation workload.
CLI options override either profile; `--config path.json` overlays JSON fields.
For example:

```sh
.venv/bin/python reproduce.py run --profile smoke --suite synthetic \
  --datasets normal uniform --n 100000 --queries 10000 --repeats 3 \
  --java-heap 2g --output results/small-comparison
```

The five original `run_experiments_*.py` filenames are compatibility entry points
to the corresponding configurable suite, using the smoke profile by default.
Use `--profile paper` for the original-scale workloads. The older exploratory
`scripts/test_merge.py` is not a validation test or the supported reproduction entry point.
It now raises an error when requesting unsupported GKAdaptive/MG merging.

Java sources and compiled classes live in `java/splinesketch/` (SplineSketch
variants) and `java/other_sketches/` (comparison sketches).

The Java prototypes are [SplineSketch.java](java/splinesketch/SplineSketch.java) (standard),
[SplineSketchMG.java](java/splinesketch/SplineSketchMG.java) (Misra–Gries filtering), and
[SplineSketchAdjustable.java](java/splinesketch/SplineSketchAdjustable.java) (ablations).
[spline_sketch_uniform.py](spline_sketch_uniform.py) is an exploratory Python
prototype; comparisons in the published paper use Java.

## Package the artifact

```sh
.venv/bin/python scripts/package_artifact.py
```

This creates `results/splinesketch-artifact.zip` with source, configuration,
bundled JARs, third-party notices, documentation, tests, historical plots and a
SHA-256 file manifest. It excludes environments, downloaded datasets, input
caches, and generated runs. Archive selected generated runs separately if desired.
Publish a versioned archive at a persistent public URL and link it in the ARI
submission. Packaging does not publish anything or claim an ACM badge.

The repository's original code is MIT licensed. Bundled dependencies and
datasets retain their own licenses; see [third_party/README.md](third_party/README.md)
and [dependency-manifest.json](dependency-manifest.json).
