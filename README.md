# SplineSketch experimental artifact

This repository accompanies **SplineSketch: Even More Accurate Quantiles with
Error Guarantees**, Aleksander Łukasiewicz, Jakub Tětek, and Pavel Veselý,
PACMMOD 3(6), Article 362 (2025), presented at SIGMOD 2026.
[Published paper](https://doi.org/10.1145/3769827).

The artifact builds the Java implementations, generates or downloads inputs,
executes experiments, saves raw rank estimates and measurements, and produces
the experimental figures in the published paper. Existing PDFs in `plots/` are
historical reference plots and are never overwritten. Newly reproduced figures
are saved under `results/<run>/figures/`. Finally, it is possible to build a version of the published paper with
the generated figures.





## Start with a smoke run

Install **Java 17 or newer** (`java` and `javac` on PATH) and **Python 3.8 or newer**, then:

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

For reduced workloads, choose one of the additional profiles:

| Profile | Synthetic accuracy input size | Queries | Java heap per job | Real datasets |
| --- | ---: | ---: | --- | --- |
| [`paper-small`](configs/paper-small.json) | 1 million | 1,000 | 20 GB | Power |
| [`paper-medium`](configs/paper-medium.json) | 10 million | 10,000 | 50 GB | HEPMASS, Power |

Run with `--profile paper-small` or `--profile paper-medium`; add `--dry-run`
to preview the workload or `--download` to fetch required datasets. These
profiles use the full selected real datasets; timing sweeps have their own
input sizes. The `paper-medium` configuration is larger than `paper-small`;
both are smaller than `paper`.

All curves are automated.

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

## Build the paper PDF

Install a TeX distribution providing `latexmk`, `pdflatex`, `bibtex`, and the
packages used by `paper/splineSketchCameraReady.tex` (for example, MacTeX or
TeX Live). After running a profile, build the paper with its saved measurements:

```sh
.venv/bin/python scripts/build_paper.py --profile paper-small
# Equivalent shortcut:
make paper PROFILE=paper-small
```

The PDF is written to `results/paper-small/paper/splineSketchCameraReady.pdf`.
The script regenerates individual plots with the filenames expected by the
TeX sources, stages the sources and static diagrams, and runs `latexmk` to
resolve the bibliography and cross-references. It uses saved records and needs
no Java or original datasets. It does not run experiments. Sources in `paper/`
and historical plots in `plots/` remain unchanged.

Use `--results path/to/run` for a run created with a custom `--output`, and
`--output path/to/build` to choose a different PDF build directory. Relative
paths are resolved from the current working directory; defaults are relative
to the repository. Missing datasets or suites produce labeled placeholder
panels, including when building from `smoke`. Failed measurements are excluded
from curves. `paper-build.json` records the effective run configuration,
missing panels, and failure counts; `latexmk.log` contains compiler output.
The paper's prose and captions retain the published workload descriptions,
so PDFs using reduced profiles are for comparison and inspection.

## Configure or extend experiments

```sh
.venv/bin/python reproduce.py run --help
.venv/bin/python reproduce.py doctor
.venv/bin/python -m unittest discover -s tests -v
```

[configs/paper.json](configs/paper.json) contains published-scale settings.
[configs/smoke.json](configs/smoke.json) contains the small validation workload.
[configs/paper-small.json](configs/paper-small.json) and
[configs/paper-medium.json](configs/paper-medium.json) provide additional
workloads, selectable with `--profile paper-small` or `--profile paper-medium`.
Outputs default to `results/<profile>`. Preview or run them with:

```sh
.venv/bin/python reproduce.py run --profile paper-small --dry-run
.venv/bin/python reproduce.py run --profile paper-medium --dry-run
.venv/bin/python reproduce.py run --profile paper-small --download
.venv/bin/python reproduce.py run --profile paper-medium --download
```

Configuration is applied in this order: the selected profile, fields from
`--config path.json`, then explicit CLI options. An override JSON can contain
just the fields you want to change; omitted fields keep their profile values.
Arrays replace the entire original array, rather than appending to it. Unknown
JSON fields are rejected. Outputs still default to `results/<profile>` when
using overrides, so choose a new `--output` for each changed configuration.

**Workloads and measurement counts.** These fields choose the data and the
amount of work in each measurement:

| JSON field | Meaning and how to change it |
| --- | --- |
| `synthetic` | Dataset names used by the `synthetic` suite. Add or remove names to change the accuracy comparison workloads; `[]` disables this suite's default workloads. This list does not control the dedicated timing, merge, heavy-item, or ablation suites. |
| `real` | Real datasets used by the `real` suite: `hepmass`, `power`, and/or `books`. `[]` skips real accuracy workloads. Including `hepmass` also enables its merge comparison. Inputs must already exist, or use `--download`. |
| `n` | Nominal number of input values for synthetic accuracy, query timing, Normal merge, heavy-item, and ablation experiments. Reducing it saves input preparation time, memory, and disk. It does not resize real datasets or the update timing sweep. The `normal_and_distinct_42` workload produces about `1.5 * n` actual values; normalization uses the actual count. |
| `queries` | Requested number of rank queries per accuracy, update, merge, heavy-item, or ablation measurement. The query timing suite uses `query_counts` instead. Queries are evenly spaced points from the sorted input, capped at the actual input length. More queries increase query time and the cost of evaluating errors. |
| `real_limit` | Maximum number of valid values loaded from each real dataset, including HEPMASS merge input. A positive integer selects a prefix; `null` uses the full dataset. For Books the prefix is selected before seeded shuffling. This does not reduce download sizes. To clear a limit inherited from `smoke`, put `"real_limit": null` in your override JSON. |
| `update_ns` | List of input lengths for the `update` suite on Normal data, producing update time versus stream length. For example, `[10000, 100000, 1000000]` gives three measurements per selected sketch and repeat. These lengths are independent of `n`. |
| `query_counts` | List of requested query batch sizes for the `query` suite, producing time per query versus batch size on `n` Normal values. Keep counts at or below `n` to avoid multiple requested counts being capped to the same actual count. |
| `merge_parts` | Number of consecutive chunks sketched separately and then merged in a binary tree in the `merge` suite. Use at least 2 and no more than the actual input length. More parts mean smaller chunks and more merges; time per merge divides the merge phase by `merge_parts - 1`. SplineSketch+MG merge comparisons are unavailable. |

The standard synthetic workloads are `normal`, `uniform`, `pareto`, `lognormal`,
`loguniform`, `signed_loguniform`, `signed_loguniform_extreme`,
`normal_with_1_small_change`, and `normal_with_1_large_change`. Additional
supported names are `gumbel`, `signed_lognormal`, `distinct_values_42`,
`distinct_values_5`, `distinct_values_150`, `normal_and_distinct_42`, and
`sorted_with_frequent`. Names and default ablation values are defined in
[reproduction/config.py](reproduction/config.py); distribution implementations
are in [IID_generators.py](IID_generators.py).

**Sketch selection and size sweeps.** The size controls are algorithm parameters;
plots compare the resulting measured serialized sizes in bytes.

| JSON field | Meaning and how to change it |
| --- | --- |
| `sketches` | Algorithms to compare: `splinesketch`, `kll`, `moments`, `tdigest`, `gk`, with optional `splinesketch_mg` and `ddsketch`. Remove algorithms to reduce measurements. The `heavy` suite always adds SplineSketch+MG; `ablation` uses the adjustable SplineSketch implementation regardless of this list. |
| `sizes` | List of nominal size parameters for accuracy, merge, heavy-item, and ablation sweeps. For SplineSketch this is the bucket count. t-digest uses compression `max(10, 5 * size / 3)`; GKAdaptive uses epsilon `3 / (2 * size)`; DDSketch uses relative accuracy `0.5 / size`. Larger values generally trade more space for better accuracy. |
| `moment_k` | MomentSketch parameter for each corresponding entry in `sizes`. For example, `sizes: [25, 100]` with `moment_k: [7, 15]` compares MomentSketch parameters 7 and 15 at those two sweep positions. High moment counts can cause numerical failures. |
| `kll_k` | KLL capacity parameter for each corresponding entry in `sizes`. Larger values generally use more memory and improve accuracy. Use values of at least 8. |

`sizes`, `moment_k`, and `kll_k` must be nonempty, contain no duplicates within
each list, and have equal lengths. Keep all three lists aligned when adding or
removing sweep positions, even if MomentSketch or KLL is not selected. `sizes`
must contain integers of at least 6; `moment_k` must contain positive integers.
The CLI's `--sizes 25 100` selects existing positions from the effective config
and automatically selects their MomentSketch/KLL mappings. To introduce new
sizes or change either mapping, use JSON.

The `update` and `query` suites use fixed parameters: size 100 for SplineSketch,
the t-digest/GKAdaptive/DDSketch mappings above at size 100, MomentSketch parameter
15, and KLL parameter 8. Editing the three size lists changes the other suites;
use `update_ns` and `query_counts` to change the timing sweeps. Update/query time
versus sketch size comes from the `synthetic` suite's Normal measurements.

**Repetition, resources, and plotting.**

| JSON field | Meaning and how to change it |
| --- | --- |
| `seed` | Master random seed, an integer in `[0, 2**32)`. Controls generated inputs, Books shuffling, and derived Java seeds. Change it for a different reproducible sample. |
| `repeats` | Number of fresh JVM measurements per task. Repeats reuse the same input and query points, with distinct per-repeat Java seeds. Plots aggregate repeated trials and show one standard deviation where nonzero. Increasing it multiplies the measurement count. |
| `jobs` | Maximum concurrent Java measurement processes. Input preparation remains serial. More jobs can shorten runs but increase memory use and contention, which can affect timing comparisons. |
| `java_heap` | Maximum Java heap **per process**, passed as `-Xmx`, such as `"1g"` or `"20g"`. Budget for up to `jobs * java_heap`, plus Python/input preparation memory and JVM overhead. Raising it allows larger measurements but does not reduce their work. |
| `java_timeout` | Positive integer timeout in seconds for each Java invocation, including loading its input and writing results. Timed-out measurements are recorded as failures. Increase it for large inputs or slower machines. |
| `max_plot_bytes` | Maximum serialized sketch size shown in plots whose x-axis is sketch bytes. Increasing it reveals larger measured sketches. It does not change the tasks, memory limits, saved raw measurements, or CSV export, and it does not filter timing panels whose x-axis is input/query count. |

Counts and resource limits must be positive integers; `update_ns` and
`query_counts` must be nonempty lists of distinct positive integers. Both `n`
and each `update_ns` entry are limited to `2**31 - 1` by the Java wrappers.
Most scalar fields have a CLI equivalent with underscores replaced by hyphens,
such as `--merge-parts`, `--java-heap`, and `--max-plot-bytes`. List equivalents
include `--sketches`, `--update-ns`, and `--query-counts`. The `synthetic`, `real`,
`moment_k`, and `kll_k` fields are configured through JSON.

**Choose suites and preview the run.** Suite selection is a CLI option, not a
JSON field. `--suite` defaults to all seven suites:

| Suite | Measurements selected |
| --- | --- |
| `synthetic` | Accuracy and update/query times across `synthetic` datasets, sketches, and size positions. |
| `real` | The same comparisons on the selected `real` datasets. |
| `update` | Normal input-length sweep from `update_ns`, at fixed sketch parameters. |
| `query` | Normal query-batch sweep from `query_counts`, at fixed sketch parameters. |
| `merge` | Normal and, when enabled, HEPMASS merge comparisons across sizes. |
| `heavy` | Mixed Normal/frequent-item and sorted frequent-item comparisons, including SplineSketch+MG. |
| `ablation` | Adjustable SplineSketch variants on Normal data, across sizes. Defaults to `interpolations` and `heuristicError`; use `--ablations` for other studies. |

`--datasets normal uniform` replaces the profile's synthetic dataset selection
for the `synthetic` suite, selects any named real datasets for `real`, and
filters the other suites to matching workloads. To exclude an experiment type,
choose `--suite` explicitly; emptying `synthetic` or `real` alone does not disable
the dedicated timing, heavy-item, merge, or ablation experiments.

For a quick comparison using only CLI overrides:

```sh
.venv/bin/python reproduce.py run --profile smoke --suite synthetic \
  --datasets normal uniform --n 100000 --queries 10000 --repeats 3 \
  --java-heap 2g --output results/small-comparison
```

For a custom size sweep and a limited real dataset, save this override as
`configs/local.json` (JSON requires double quotes and has no comments):

```json
{
  "synthetic": ["normal", "uniform"],
  "real": ["power"],
  "n": 100000,
  "queries": 1000,
  "real_limit": 100000,
  "sizes": [25, 50, 100],
  "moment_k": [7, 11, 15],
  "kll_k": [8, 16, 24],
  "update_ns": [10000, 100000],
  "query_counts": [100, 1000],
  "merge_parts": 100,
  "repeats": 3,
  "java_heap": "2g"
}
```

Preview the effective configuration, individual tasks, and total measurement
count before starting the run:

```sh
.venv/bin/python reproduce.py run --profile smoke --config configs/local.json \
  --suite synthetic real update query merge --download \
  --output results/custom-comparison --dry-run
```

Remove `--dry-run` to execute. The preview neither downloads data nor creates
files. In accuracy/merge sweeps, each dataset, size position, selected sketch,
and repeat adds a measurement; timing sweeps use one fixed size, and ablations
add one set per variant. Reducing datasets, size positions, or repeats reduces
the task count; reducing `n` or query counts reduces the work within each task.
Repeat an unchanged command to resume. A changed configuration, source, or
environment requires a new `--output` directory, including when only plotting
or resource settings in the run configuration change. To regenerate figures
from saved measurements, use `reproduce.py plot --output path/to/run`.

The five original `run_experiments_*.py` filenames are compatibility entry points
to the corresponding configurable suite, using the smoke profile by default.
Use `--profile paper` for the original-scale workloads. The older exploratory
`scripts/test_merge.py` is not a validation test or the supported reproduction entry point.

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
