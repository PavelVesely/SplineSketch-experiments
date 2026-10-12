"""Shell-free, portable invocation of the original Java benchmark wrappers."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

from .common import ROOT

PROGRAMS = {
    "splinesketch": ("SplineSketchProgram", []),
    "splinesketch_mg": ("SplineSketchMGProgram", []),
    "adjustable": ("SplineSketchAdjustableProgram", []),
    "tdigest": ("TDigestProgram", ["t-digest-3.3.jar"]),
    "kll": ("KLLProgram", ["datasketches-java-6.0.0.jar", "datasketches-memory-2.2.1.jar"]),
    "moments": ("MomentSketchProgram", ["msolver-1.0-SNAPSHOT.jar", "quantile-bench-1.0-SNAPSHOT.jar", "commons-math3-3.6.1.jar"]),
    "gk": ("GKProgram", []),
    "ddsketch": ("DDSketchProgram", ["sketches-java-0.8.3.jar"]),
}
LABELS = {"splinesketch": "SplineSketch", "splinesketch_mg": "SplineSketch+MG",
          "kll": "KLL", "moments": "MomentSketch", "tdigest": "t-digest",
          "gk": "GKAdaptive", "ddsketch": "DDSketch", "adjustable": "SplineSketch adjustable"}


class NumericalFailure(RuntimeError):
    pass


def java_version(executable):
    process = subprocess.run([executable, "-version"], capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"{executable} is unavailable. Install a JDK 17 or use Docker.\n{process.stderr}")
    return (process.stdout + process.stderr).strip()


def check_java():
    java, javac = shutil.which("java"), shutil.which("javac")
    if not java or not javac:
        raise RuntimeError("JDK 17 is required (both java and javac on PATH). Alternatively use Docker.")
    versions = {"java": java_version(java), "javac": java_version(javac)}
    if not re.search(r'version "17\.', versions["java"]) or not versions["javac"].startswith("javac 17."):
        raise RuntimeError("The artifact requires JDK 17. Install it or use the provided Dockerfile.")
    return versions


def compile_java():
    check_java()
    classpath = os.pathsep.join([str(ROOT)] + [str(p) for p in sorted(ROOT.glob("*.jar"))])
    sources = [str(p) for p in sorted(ROOT.glob("*.java"))]
    process = subprocess.run(["javac", "-encoding", "UTF-8", "-cp", classpath] + sources,
                             capture_output=True, text=True, cwd=ROOT)
    if process.returncode:
        raise RuntimeError(f"Java compilation failed:\n{process.stderr}")


def run_java(sketch, data, queries, n, parameter, parts=1, params=None,
             heap="1g", timeout=120, seed=42, tmp_dir=None):
    if not re.fullmatch(r"[1-9][0-9]*[kKmMgG]", heap):
        raise ValueError("Java heap must be a positive size, e.g. 1g or 100g")
    if n <= 0 or parts <= 0 or parts > n:
        raise ValueError("Merge partitions must satisfy 1 <= parts <= actual input size")
    if parts != 1 and sketch in ("gk", "splinesketch_mg"):
        raise ValueError(f"{LABELS[sketch]} merge is unavailable in this repository")
    program, jars = PROGRAMS[sketch]
    classpath = os.pathsep.join([str(ROOT)] + [str(ROOT / jar) for jar in jars])
    if tmp_dir:
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spline-", dir=tmp_dir) as directory:
        output = Path(directory) / "ranks.txt"
        parameter = max(10, 5 * parameter / 3) if sketch == "tdigest" else parameter
        command = ["java", f"-Xmx{heap}", f"-Dsplinesketch.seed={seed}",
                   "-cp", classpath, program, str(Path(data).resolve()),
                   str(Path(queries).resolve()), str(parameter), str(output), str(parts)]
        if params is not None:
            command.extend(str(p) for p in params)
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
        if process.returncode:
            raise RuntimeError(f"{program} exited {process.returncode}:\n{process.stderr[-8000:]}")
        try:
            metrics = [int(line) for line in process.stdout.strip().splitlines()]
            if len(metrics) != 3 or metrics[0] <= 0 or min(metrics[1:]) < 0:
                raise ValueError("invalid metrics")
            with output.open() as stream:
                ranks = np.fromiter((int(line) for line in stream), dtype=np.int64)
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Invalid {program} output: {process.stdout[-2000:]}\n{process.stderr[-8000:]}") from error
        with Path(queries).open() as stream:
            query_count = sum(1 for _ in stream)
        if len(ranks) != query_count or np.any(ranks < 0) or np.any(ranks > n):
            message = f"{program}: expected {query_count} ranks in [0,{n}], received {len(ranks)}.\n{process.stderr[-8000:]}"
            numerical_diagnostic = any(marker in process.stderr for marker in
                ("org.apache.commons.math3.exception.", "SingularMatrixException", "java.lang.ArithmeticException"))
            if sketch == "moments" and not len(ranks) and numerical_diagnostic and "OutOfMemoryError" not in process.stderr:
                raise NumericalFailure(message)
            raise RuntimeError(message)
        return ranks, metrics, {"command": command, "stderr": process.stderr}
