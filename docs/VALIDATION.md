# Artifact validation

Prepared October 9, 2026. Tested locally on Apple silicon macOS, Python 3.9.6,
Temurin JDK 17.0.16+8, and the exact pinned dependencies in `requirements.txt`.
The JDK was downloaded to a temporary directory for validation; no system Java
installation or global Python package environment was changed.

## Completed checks

- All Java programs compile using both the Python `build` command and `make`.
- `doctor` passes dependency pins, Java/JAR and compiled-wrapper checks.
- `pip check` reports no broken requirements.
- **15 regression tests pass**, covering inclusive ranks with ties, deterministic
  generators/KLL, input cache corruption, query cache separation, CSV missing
  values, SOSD headers, size/measurement pairing, paths containing spaces, all
  eight Java wrappers, remainder handling in MomentSketch merge, actual-input
  normalization and merge counts, resumption/configuration-change rejection,
  and provenance capture without Git for containers/source archives.
- The complete **76-measurement smoke profile** passes with no failed, numerical
  or pending measurements. It runs accuracy, update/query timing, merging, both
  frequent-item workloads, MG, and the two published ablations. Figure 9(a)
  records actual n=3000 for nominal n=2000.
- All **10 synthetic workloads** run at nominal n=2000, q=100, sizes=15/100:
  98 successful measurements and two recorded MomentSketch numerical failures
  on extreme signed loguniform input. The diagnostic is the Apache Commons Math
  `NoBracketingException` with NaN endpoint values. There are no operational
  failures or pending trials. Fits that fail are excluded from curves.
- **32 additional parameter ablations** pass at smoke scale: split/join ratio,
  bucket bound multiplier, minimum splittable fraction, and epoch factor.
- Two repeats with two concurrent jobs pass for **10 measurements**, exercising
  repeat aggregation and concurrent Java invocation.
- The official Power archive downloads and extracts successfully. It has
  2075259 source rows, 25979 missing readings and 2049280 valid readings. Both a
  2000-value subset and the **full Power input** pass ten measurements each
  (sizes 15/100 for the full input, 100000 queries, all five main sketches).
- Saved rank vectors reproduce their reported normalized mean/max errors.
  Re-running a matching smoke command preserves existing measurement records;
  standalone plotting regenerates CSV and figures without Java execution.
- Source archive packaging and a fresh-directory build/run are checked before
  handoff. Generated source archives include a SHA-256 file manifest.

Generated local run directories are ignored by Git:

| Run | Directory | Peak Python / largest-child RSS |
|---|---|---|
| Final smoke | `results/smoke` | about 38 / 141 MiB |
| Full Power, two sizes | `results/validation/power-full` | about 108 / 507 MiB |
| All synthetic inputs at reduced scale | `results/validation/all-synthetic` | about 62 / 282 MiB |
| Supplementary parameter studies | `results/validation/parameters` | about 62 / 51 MiB |
| Two repeats/two jobs | `results/validation/repeats` | about 38 / 61 MiB |

These memory counters describe preparation/measurement only; figure generation
occurs afterward. Child counters are the largest individual child, not summed
concurrent memory. Timings on this laptop do not validate the paper's server
performance claims. Inspect each `run.json` for the actual environment and
individual records for measurement times.

## Remaining validation limits

- The 1264-measurement paper profile has been enumerated but **not executed in
  full**. Its 100-million-item synthetic inputs, complete HEPMASS and
  800-million-item Books input require a suitably equipped machine.
- HEPMASS's official training-file URL returns HTTP 200; full train/test
  acquisition and preprocessing have not been tested here. Books preprocessing
  has fixture coverage, but its full archive download/checksum has not been run.
- GKAdaptive merge requires the author follow-up in the separate `TODO.md`.
- Docker and native Windows were not available for validation. The native macOS
  path was exercised; container and Windows instructions remain unverified.
- The MomentSketch snapshot JARs do not identify their original source commits.
  Their exact bundled binaries are pinned by SHA-256.

The validation establishes functionality of the workflow, not independent
reproduction of every published claim. Reviewers should run the paper profile
and compare the trends described in `REPRODUCIBILITY.md`.
