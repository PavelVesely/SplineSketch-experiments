# Reproduction guide for the published paper

Reference: *SplineSketch: Even More Accurate Quantiles with Error Guarantees* [DOI 10.1145/3769827](https://doi.org/10.1145/3769827), PACMMOD 3(6),
Article 362, December 2025. The experiment mapping below follows the **published
26-page PDF**, not the differently numbered arXiv draft. The PDF is not bundled.

This artifact follows the [SIGMOD 2026 call](https://reproducibility.sigmod.org/2026/)
and [SIGMOD packaging guidelines](https://reproducibility.sigmod.org/): declared
dependencies, automatic setup, generated or downloadable inputs, configurable
experiments, raw measurements, plotting scripts, sanity checks, and documented
claims. Badge decisions require independent evaluation. No badge is claimed here.

## Environment and installation

- Working `java` and `javac` commands, each version 17 or newer, on PATH.
  No installation directory or vendor is required. Validation used Temurin
  17.0.16+8; compilation targets Java 17 even with a newer compiler.
- Python 3.8 or newer with venv and pip (including 3.13 and 3.14). Validation used Python 3.9.6 on Apple silicon
  macOS; the container uses Ubuntu 22.04 Python 3.10.
- Compatible dependency ranges in `requirements.txt`; pip selects releases
  for the interpreter. `requirements-pinned.txt` preserves the historical
  Python 3.9–3.12 direct/transitive package pins and is used by Docker.
- All Java dependencies are already bundled. `dependency-manifest.json` records
  SHA-256 identities and upstream projects. Snapshot JAR source commits are not
  known; the exact binary artifacts are retained rather than silently rebuilt.
- Headless Matplotlib uses the Agg backend and a writable local cache. No GUI,
  GPU, cloud credentials, or interactive plotting is needed.

`python3 reproduce.py setup` creates `.venv`, installs compatible dependencies, compiles Java,
and runs `doctor`. If prerequisites are missing, install them before retrying.
`make` remains available for Java-only compilation; the Python `build` command
also works without Make and builds all wrappers with explicit UTF-8 encoding.

No upper Python version is enforced; future versions require compatible upstream
packages. Only Python 3.9 has been tested locally. Preserve the exact environment
with `python -m pip freeze > requirements-local.txt`; run metadata records direct
dependency versions.

`doctor` validates Java runtime and compiler availability, Python package requirements, JAR hashes and compiled
wrapper presence. `doctor --datasets power` validates existing data without
initiating downloads. Native Windows uses the Python commands and portable
classpaths, but is not yet tested. `resource` memory counters are unavailable on
Windows and appear as null in run metadata.

## Paper-to-command mapping

Commands below assume `.venv/bin/python reproduce.py`; use the Windows equivalent
when needed. All `run` commands generate PDFs and PNGs in their output directory,
including placeholders for panels whose experiments were not selected.

| Figure | Published panels | Suites and selection | Generated figure |
|---|---|---|---|
| 1 | Average error: HEPMASS, Power, Books | `run --profile paper --suite real --download --output results/real` | `figures/figure-1.pdf` |
| 4 | Average error: Normal, Uniform, Pareto, Lognormal, Loguniform, Signed Loguniform, extreme Signed Loguniform, small/large Normal change | `run --profile paper --suite synthetic --output results/synthetic` | `figures/figure-4.pdf` |
| 5 | Maximum error: the three real inputs plus the nine Figure 4 inputs | `run --profile paper --suite real synthetic --download --output results/accuracy` | `figures/figure-5.pdf` |
| 6(a) | Update time versus Normal input length | `run --profile paper --suite update --output results/update` | `figures/figure-6.pdf`, left panel |
| 6(b) | Update time versus sketch bytes, Normal input | `run --profile paper --suite synthetic --datasets normal --output results/normal` | `figures/figure-6.pdf`, right panel |
| 7(a) | Query time versus batch query count | `run --profile paper --suite query --output results/query` | `figures/figure-7.pdf`, left panel |
| 7(b) | Query time versus sketch bytes, Normal input | Same Normal accuracy run as 6(b) | `figures/figure-7.pdf`, right panel |
| 8 | Maximum error on Normal/HEPMASS and merge time on Normal | `run --profile paper --suite merge --download --output results/merge` | `figures/figure-8.pdf` |
| 9 | Average errors on mixed and sorted frequent-item inputs; mixed-input update time | `run --profile paper --suite heavy --output results/heavy` | `figures/figure-9.pdf` |
| 10 | PCHIP versus linear; five heuristic error alternatives, on Normal | `run --profile paper --suite ablation --output results/ablation` | `figures/figure-10.pdf` |

Figures 2 and 3 illustrate the algorithm and do not derive from measurements.
The full command `run --profile paper --download` combines all suites in one
run and fills all supported experimental panels. Gumbel is also measured because
Section 5 lists it; supplementary PDFs are generated for workloads outside the
displayed panels. Additional parameter studies can be selected using:

```sh
.venv/bin/python reproduce.py run --profile paper --suite ablation \
  --ablations splitJoinRatio defaultBucketBoundMult minFracBucketBoundToSplit epochIncrFactor \
  --output results/parameter-studies
```

## Inputs and preprocessing

Synthetic accuracy and ablation inputs use nominal `n=100000000` and 100000
queries. Generators retain the repository's distributions, including the signed
loguniform generators' single randomly selected sign for an entire input. Normal
changes occur after n/2 items: `(0,1)` to `(1,1)` or `(100,10)` as `(mean,sigma)`.
The sorted frequent-item workload uses random integer frequencies from 1 to n/50,
trimming the last run of equal values to the desired length.

**Figure 9(a) implementation discrepancy, intentionally preserved:**
`normal_and_distinct_42(n)` generates n/2 Normal(0,1) values, another n−n/2
Normal(1,1) values, and n−n/2 samples from the integers −21 through 20. Thus,
nominal n=100000000 produces **150000000 actual items**, rather than the paper's
description of n/2 Normal values followed by n/2 frequent items. Normalized
errors and average update times use the actual 150000000 items, matching the
original experimental code. Both nominal and actual n are recorded in results.

| Dataset | Source and automatic command | Selected data and validation |
|---|---|---|
| HEPMASS | [UCI](https://archive.ics.uci.edu/dataset/347/hepmass); `download hepmass` | `all_train.csv.gz` (7 million rows), then `all_test.csv.gz` (3.5 million rows); decompress to `datasets/hepmass/`; column 2, header skipped; source order preserved. |
| Power | [UCI](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption); `download power` | Extract `datasets/household_power_consumption/household_power_consumption.txt`; semicolon delimiter; column 3 `Global_active_power`; header skipped; `?` and empty values skipped; source order preserved. |
| Books | [SOSD upstream script](https://github.com/learnedsystems/SOSD/blob/f52f4cba01dfcd37f1574551fccef00198863b88/scripts/download.sh); `download books` | Extract `datasets/SOSDdata/books_800M_uint64`; little-endian uint64 count header + 800 million values; validate file length/header and upstream MD5 `8708eb3e1757640ba18dcd3a0dbb53bc`; seeded permutation, then float64 conversion as in the original code. |

Power has **2075259 source rows, 25979 missing readings, and 2049280 usable
values**, verified against the official archive. The paper's n=2075259 is the
source-row count; the experiment uses the filtered actual count, as the original
loader did. Unexpected malformed rows or nonfinite values fail rather than being
silently discarded.

Download scripts need no logins. HEPMASS downloads only the two needed files,
not the complete 7.4 GB UCI collection. Downloads retain archives under
`datasets/archives/`, use `.part` files for interrupted transfers, verify archive
integrity while decompressing, and atomically publish validated files. Subsequent
downloads verify local SHA-256 hashes. UCI has no pinned upstream checksum in
this artifact: the initial accepted hashes are recorded in `datasets/downloads.json`
after row/column and compression checks. This is provenance tracking, not an
independent upstream authenticity check. Books additionally uses upstream MD5.

If the historical Books Dropbox link becomes unavailable, supply another mirror
with `download books --books-url URL`; the **same uncompressed MD5** is still
required. Or put the expected files in the paths above and run `doctor --datasets
hepmass power books`. Dataset directories can be changed with `--dataset-dir`.

`--real-limit N` takes the first N valid CSV values, or the first N Books values
before shuffling. Such runs are explicitly subset runs; they do not recreate the
full real-world results. Books float64 rounding is retained because all compared
Java wrappers consume doubles. Rank ground truth is computed on those same
rounded values, including equal values, rather than on the original uint64 keys.

## Measurement protocol

1. Input preparation runs serially, reducing peak memory. A local NumPy
   RandomState is seeded from the master seed, workload name and nominal n.
   Reordering suites or changing Java parallelism does not change the inputs.
2. Inputs are written as 17-digit text for the original Java wrappers. A sorted
   float64 NumPy copy supplies queries and exact inclusive ranks. Query selection
   uses indices `floor(i * actual_n / q)` for `i=0..q-1`, where q is the smaller
   of requested query count and actual n. This produces exactly q evenly spaced
   sorted data points. The old scripts used an integer stride that could produce
   more than the requested queries for nondivisible lengths.
3. Each measurement starts a fresh JVM, loads all input and queries into memory,
   then measures only sketch construction/consolidation or merges, followed by
   rank queries. JVM launch, input parsing, output writing and plotting are
   outside the measured Java interval. There is no added warmup phase; this
   retains the original wrapper protocol. The small-input initialization/JIT
   effect is therefore visible in the update-size experiment. File caches are
   not flushed; parsing is excluded from the reported Java timings.
4. Error is `abs(estimated_rank − inclusive_true_rank) / actual_n`; save its
   average and maximum. Java times are reported in ns and converted to μs per
   item/query, or per merge. Raw rank vectors are retained for independent checks.
5. Merge experiments build 10000 consecutive chunks and merge them pairwise in
   a binary tree. Report the measured merge phase divided by 9999. The final
   chunk contains any nondivisible remainder, including for MomentSketch.
6. Processes terminate after measurements, temporary rank files are cleaned up,
   and subsequent trials use fresh JVMs. Input caches and raw results persist.

The default size sweep is 15,20,25,32,50,75,100,125,150,175,200,250 SplineSketch
buckets. MomentSketch and KLL use the per-position k mappings in the profile.
t-digest uses `max(10,5*size/3)` compression with scale function K_0. GKAdaptive
uses epsilon `3/(2*size)`, preserving the supplied implementation. Fixed-size
timing sweeps use 100 SplineSketch buckets, the corresponding t-digest parameter,
MomentSketch k=15 and KLL k=8. `update_ns` and `query_counts` preserve the original
doubling sweeps (10000 and 1000 starts, respectively; 16 points each).

Sizes reflect the supplied wrappers: SplineSketch/t-digest use 16 bytes per
bucket/centroid; MomentSketch uses 16k+16; GKAdaptive uses 24 bytes per tuple;
MG includes its compressed heavy-hitter storage. KLL reports its library's
`getSerializedSizeBytes()` including serialization metadata, following the
repository wrapper, whereas the paper describes 8 bytes per stored item.

KLL's package-private RNG in the pinned DataSketches 6.0.0 JAR is seeded by the
wrapper via reflection before measurement. Per-task/repeat seeds are recorded.
Changing the JAR requires validating that hook. Repeats share the same input
but use distinct KLL seeds and fresh JVMs. Timing plots show means and ±1 standard
deviation across repeats; errors are averaged across repeats. Default repeats=1
matches the original scripts. All trial values remain in the CSV/JSON.

Numerical failures of high-moment MomentSketch fits are an expected limitation
discussed in the paper. Empty rank output with valid metrics and a numerical
diagnostic is recorded as `numerical_failure` and excluded from curves, with
diagnostics retained. Other subprocess errors, timeouts, malformed metrics,
rank-count mismatches, and out-of-range ranks are failures, never infinite-error
values masquerading as successful measurements. Runs return nonzero for such
failures. See `run.json` counts and `figure-report.json` before interpreting plots.

## Saved results, resumption and plotting

Each output directory contains:

- `run.json`: effective configuration, all tasks, source/JAR hashes, Git revision
  and dirty status, dependency/Java/OS versions, input provenance, status counts,
  timestamps and peak memory counters.
- `records/<task-hash>.json`: settings, nominal/actual input size, query count,
  actual sketch size, raw Java times, normalized errors, subprocess diagnostics,
  input/rank hashes and wall time for each trial.
- `raw/<task-hash>.npz`: inclusive true ranks and estimated ranks.
- `measurements.csv`: an export of successful and failed trial records.
- `figures/figure-{1,4,5,6,7,8,9,10}.{pdf,png}` and supplementary PDFs.
- `figure-report.json`: measurements that failed, numerical failures, curves
  present per panel, absent panels and any display floor used for zero errors.

Input caches live separately in `.cache/inputs/<content-key>/`. Their manifests
bind input to source file hashes, generator/loader hashes, seed and subset size.
Query files are keyed by requested count. Cached text, sorted data, queries and
ranks are verified before reuse; corrupted caches cause an explicit error.
Preparation and measurements are separate, so rank arrays need not be recopied
to every JVM input file.

A matching run resumes automatically; verified completed records are skipped,
failed trials are retried, and known numerical failures remain recorded. Use a
new output directory to repeat all timings or retry numerical failures. Runs
with a different configuration, source, environment or source data are rejected
in the same output directory. Do not run two processes against the same output
or input cache directory simultaneously; use separate `--output` and `--cache-dir`
when running independent sessions.

Plotting reads saved records without running Java or requiring original datasets.
It sorts **size and measurement together**, excludes failed points and sizes
above 4500 bytes (configurable), combines panels, and retains complete CSV data.
Zero error is displayed as a downward triangle at a documented positive floor
because logarithmic axes cannot display zero. The dotted purple `64/bytes` line
is the historical illustrative reference curve, not a verified bound on the
observed error. Plots resemble the paper's organization; fonts and layout need
not be pixel-identical. Missing panels/merge curves are visibly marked.

To independently recompute errors:

```python
import json
import numpy as np
from pathlib import Path

run = Path("results/paper")
record = next(json.loads(p.read_text()) for p in (run / "records").glob("*.json")
              if json.loads(p.read_text())["status"] == "ok")
raw = np.load(run / record["rank_artifact"])
errors = np.abs(raw["estimated_ranks"] - raw["true_ranks"]) / record["actual_n"]
assert np.isclose(errors.mean(), record["average_error"])
assert np.isclose(errors.max(), record["maximum_error"])
```

## Expected claims and resource requirements

Smoke results verify functionality only. At paper scale, assess the trends rather
than demanding identical timings on different CPUs:

- Figures 1,4,5: SplineSketch generally improves error with more space and often
  beats t-digest by factors of 2–20; MomentSketch can be more accurate at very
  small sizes but has numerical limitations as moments increase.
- Figures 6,7: initialization costs diminish on larger inputs; KLL and
  MomentSketch have fast updates, while MomentSketch rank queries are expensive.
- Figure 8: merge errors increase relative to streaming but preserve much of the
  comparative accuracy; SplineSketch merging is more expensive.
- Figure 9: MG filtering restores accuracy with frequent items once enough
  counters are available, with increased update cost. Preserve the documented
  Figure 9(a) workload discrepancy when comparing.
- Figure 10: PCHIP usually improves upon linear interpolation, and the normalized
  second derivative performs best among the heuristic alternatives.

Paper reference: AMD EPYC 7302, 3 GHz, 251 GB RAM, SSD. Java inputs are loaded
into boxed Double lists, so memory use is far above 8 bytes per value; Books also
requires a full permutation and sorted array. The original 100 GB MomentSketch
heap is exposed as the paper profile's 100 GB ceiling for **each** JVM. Jobs=1
avoids multiplying that requirement. A heap ceiling is not the amount allocated
at startup. Allow system/native/Python headroom beyond the ceiling. Increase
`--jobs` only if enough RAM is available and do not compare contended timings
with the sequential default without recording that difference.

Planning estimates: smoke <2 GB RAM and <100 MB cache/results; full suite
approximately 128–256 GB RAM and 200 GB free disk, depending on text precision,
archives and retained rank artifacts. Full-suite runtime and minimum resources
have not been benchmarked here. `run.json` records measured RSS peaks (Python and
largest child separately, not combined concurrent peak). Timing re-execution
requires a new output path; plotting alone is cheap. Inputs are reused across
suites and only one large dataset is prepared at a time.

For setup/running/cleanup checks, use `doctor`, the regression tests, the smoke
command, then verify `run.json` has zero failed/pending records. Confirm selected
figure panels list the expected curves in `figure-report.json`. Numerical failures
must be reviewed separately. Cleanup is optional: remove generated caches/runs
only after saving results you need. Dataset acquisition and large paper-scale
experiments are not run by `setup` or the tests.

## Submission and availability

`scripts/package_artifact.py` packages the current working tree and produces a
SHA-256 manifest; it refuses to overwrite an existing archive. It includes
historical reference PDFs, not the copyrighted paper, datasets, virtual
environment, compiled classes, or generated measurements. Include a separate
archive of selected result directories if submitting precomputed measurements.
The repository lacks the paper's LaTeX sources, so automatic recompilation of
the paper PDF is outside this artifact's scope; generated figure PDFs can be
included in those sources separately.

Publish the source archive under a persistent versioned URL (e.g. an archival
repository DOI), add it to the README and ARI submission, select the badges to be
evaluated, and provide access credentials if needed. The SIGMOD 2026 artifact
submission deadline listed on the call is October 15, 2026. Registration and
submission are author actions; this workflow prepares an archive but does not
register, upload, or publish it. Follow the live call for final submission details.
