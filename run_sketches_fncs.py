"""Backward-compatible sketch functions backed by the validated Java runner."""
import os
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np

from helper_funcs import load_floats_from_file
from reproduction.java import run_java
from spline_sketch_uniform import SplineSketchUniform


def run_program_parse_stdout_and_outputFile(command, output_file, n):
    """Legacy API; execute arguments directly, never through a shell."""
    try:
        process = subprocess.run(shlex.split(command) if isinstance(command, str) else command,
                                 check=True, text=True, capture_output=True)
        metrics = [int(line) for line in process.stdout.strip().splitlines()]
        if len(metrics) != 3 or metrics[0] <= 0 or min(metrics[1:]) < 0:
            raise ValueError(f"Expected size and two times, got {process.stdout}")
        with open(output_file) as stream:
            results = np.fromiter((int(line) for line in stream), dtype=np.int64)
        return results, *metrics
    finally:
        Path(output_file).unlink(missing_ok=True)


def _run(sketch, dataFile, queriesFile, n, size, parts, tmpPath, params=None):
    ranks, metrics, _ = run_java(sketch, dataFile, queriesFile, n, size, parts, params,
        heap=os.environ.get("SPLINESKETCH_JAVA_HEAP", "1g"),
        timeout=int(os.environ.get("SPLINESKETCH_JAVA_TIMEOUT", "86400")),
        tmp_dir=tmpPath)
    return ranks, *metrics


def run_splinesketch_java(dataFile, queriesFile, n, sketch_size, num_parts, print_info, tmpPath="./output_files/"):
    return _run("splinesketch", dataFile, queriesFile, n, sketch_size, num_parts, tmpPath)


def run_splinesketchMG_java(dataFile, queriesFile, n, sketch_size, num_parts, print_info, tmpPath="./output_files/"):
    return _run("splinesketch_mg", dataFile, queriesFile, n, sketch_size, num_parts, tmpPath)


def run_splinesketchAdjustable_java(dataFile, queriesFile, n, sketch_size, num_parts, print_info, tmpPath="./output_files/"):
    params = shlex.split(print_info) if isinstance(print_info, str) else print_info
    return _run("adjustable", dataFile, queriesFile, n, sketch_size, num_parts, tmpPath, params)


def run_tdigest(dataFile, queriesFile, n, sketch_size, num_parts, print_info, tmpPath="./output_files/"):
    return _run("tdigest", dataFile, queriesFile, n, sketch_size, num_parts, tmpPath)


def run_MomentSketch(dataFile, queriesFile, n, k, num_parts, print_info, tmpPath="./output_files/"):
    return _run("moments", dataFile, queriesFile, n, k, num_parts, tmpPath)


def run_kll(dataFile, queriesFile, n, k, num_parts, print_info, tmpPath="./output_files/"):
    return _run("kll", dataFile, queriesFile, n, k, num_parts, tmpPath)


def run_GK(dataFile, queriesFile, n, k, num_parts, print_info, tmpPath="./output_files/"):
    return _run("gk", dataFile, queriesFile, n, k, num_parts, tmpPath)


def run_DDSketch(dataFile, queriesFile, n, k, num_parts, print_info, tmpPath="./output_files/"):
    return _run("ddsketch", dataFile, queriesFile, n, k, num_parts, tmpPath)


def run_splineSketchUniform(dataFile, queriesFile, n, sketch_size, num_parts, print_info=""):
    if num_parts != 1:
        raise ValueError("Python sketch wrapper supports streaming only")
    data = load_floats_from_file(dataFile)
    queries = load_floats_from_file(queriesFile)
    start = time.perf_counter_ns()
    sketch = SplineSketchUniform(sketch_size, print_info)
    for x in data:
        sketch.update(x)
    sketch.consolidate()
    updated = time.perf_counter_ns()
    ranks = sketch.query(queries)
    queried = time.perf_counter_ns()
    return ranks, 12 * sketch_size + 8, updated - start, queried - updated
