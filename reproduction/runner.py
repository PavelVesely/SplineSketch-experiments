"""Incremental experiment results, provenance, and safe resumption."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version
import os
from pathlib import Path
import platform
try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None
import shutil
import subprocess
import time

import numpy as np

from .common import ROOT, SCHEMA, digest, identity, load_json, write_json
from .data import prepare_data, prepare_queries
from .java import JAVA_DIRS, NumericalFailure, run_java


def now():
    return datetime.now(timezone.utc).isoformat()


def provenance():
    sources = [p for directory in JAVA_DIRS for p in directory.glob("*.java")] + [ROOT / "IID_generators.py"]
    sources += list((ROOT / "reproduction").glob("*.py"))
    sources += list((ROOT / "configs").glob("*.json")) + list(ROOT.glob("*.jar"))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(sources)}
    commit, dirty = None, None
    if shutil.which("git"):
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
        commit = revision.stdout.strip() if revision.returncode == 0 else None
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return dict(source_hashes=hashes, source_identity=identity(hashes),
                git_commit=commit, git_dirty=dirty,
                python=platform.python_version(), platform=platform.platform(),
                processor=platform.processor(), logical_cpus=os.cpu_count(),
                dependencies={name: version(name) for name in ("numpy", "matplotlib", "joblib", "zstandard")})


def measure(task, cfg, directory, data_info, query_info, truth, output):
    repeat = task["repeat"]
    java_seed = int(identity([cfg["seed"], task])[:8], 16)
    result = dict(task=task, schema=SCHEMA, input_key=data_info["key"],
                  input_sha256=data_info["files"]["data.txt"], actual_n=data_info["actual_n"],
                  actual_queries=query_info["actual_queries"], query_metadata=query_info,
                  java_seed=java_seed, started_at=now())
    start = time.perf_counter()
    try:
        ranks, metrics, process = run_java(task["sketch"], directory / "data.txt",
            directory / f"queries-{task['requested_queries']}.txt", data_info["actual_n"],
            task["parameter"], task["parts"], task["params"], cfg["java_heap"],
            cfg["java_timeout"], java_seed, output / "tmp")
        size, update_ns, query_ns = metrics
        # int64 arithmetic and actual N also handle Figure 9(a)'s extra segment.
        errors = np.abs(ranks - truth)
        count = data_info["actual_n"] if task["parts"] == 1 else task["parts"] - 1
        result.update(status="ok", serialized_bytes=size, update_or_merge_ns=update_ns,
            query_ns=query_ns, average_error=float(np.mean(errors) / data_info["actual_n"]),
            maximum_error=float(np.max(errors) / data_info["actual_n"]),
            update_or_merge_us=update_ns / (1000 * count),
            query_us=query_ns / (1000 * len(truth)), process=process)
        artifact = output / "raw" / (identity(task) + ".npz")
        np.savez_compressed(artifact, true_ranks=truth, estimated_ranks=ranks)
        result["rank_artifact"] = str(artifact.relative_to(output))
        result["rank_sha256"] = digest(artifact)
    except NumericalFailure as error:
        result.update(status="numerical_failure", error=str(error))
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        result.update(status="failed", error=str(error))
    result.update(wall_seconds=time.perf_counter() - start, completed_at=now(), repeat=repeat)
    return result


def execute(cfg, tasks, output, dataset_dir, cache_dir, java_info):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("raw", "records", "tmp"):
        (output / name).mkdir(exist_ok=True)
    expanded = [dict(task, repeat=repeat) for task in tasks for repeat in range(cfg["repeats"])]
    provenance_info = provenance()
    signature = identity(dict(config=cfg, tasks=expanded, code=provenance_info["source_identity"],
        python=provenance_info["python"], dependencies=provenance_info["dependencies"],
        platform=provenance_info["platform"], java=java_info,
        dataset_dir=str(Path(dataset_dir).resolve())))
    manifest = output / "run.json"
    if manifest.exists():
        metadata = load_json(manifest)
        if metadata["signature"] != signature:
            raise ValueError(f"{output} belongs to a different configuration, source, or environment. Use a new --output directory.")
    else:
        metadata = dict(schema=SCHEMA, signature=signature, config=cfg, tasks=expanded,
            provenance=provenance_info, java=java_info, started_at=now(), inputs={})
        write_json(manifest, metadata)

    groups = {}
    for task in expanded:
        key = (task["dataset"], None if task["dataset"] in ("hepmass", "power", "books") else task["nominal_n"])
        groups.setdefault(key, []).append(task)
    try:
        for (name, n), group in groups.items():
            print(f"Preparing {name}, nominal n={n or 'full source'}", flush=True)
            directory, data_info = prepare_data(name, n, cfg["seed"], dataset_dir, cache_dir, cfg["real_limit"])
            metadata["inputs"][data_info["key"]] = data_info
            write_json(manifest, metadata)
            prepared = {q: prepare_queries(directory, data_info, q) for q in
                        sorted({t["requested_queries"] for t in group})}
            pending = []
            for task in group:
                record = output / "records" / (identity(task) + ".json")
                if record.exists():
                    previous = load_json(record)
                    if previous["input_key"] != data_info["key"]:
                        raise ValueError("Source data changed since this run; use a new --output directory.")
                    if previous["status"] == "ok":
                        artifact = output / previous["rank_artifact"]
                        if not artifact.exists() or digest(artifact) != previous["rank_sha256"]:
                            raise ValueError(f"Raw ranks failed checksum validation: {artifact}")
                        continue
                    if previous["status"] == "numerical_failure":
                        continue
                pending.append(task)
            with ThreadPoolExecutor(max_workers=cfg["jobs"]) as executor:
                futures = {executor.submit(measure, t, cfg, directory, data_info,
                    prepared[t["requested_queries"]][0], prepared[t["requested_queries"]][1], output): t
                    for t in pending}
                for future in as_completed(futures):
                    result = future.result()
                    write_json(output / "records" / (identity(result["task"]) + ".json"), result)
                    task = result["task"]
                    print(f"{result['status']}: {task['suite']}/{name}/{task['sketch']} k={task['parameter']} repeat={task['repeat']}", flush=True)
        metadata["completed_at"] = now()
    finally:
        divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
        metadata["resources"] = dict(python_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor if resource else None,
            largest_child_peak_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / divisor if resource else None,
            free_disk_bytes=shutil.disk_usage(output).free)
        # Only planned records count; interrupted runs retain individual records.
        records = [load_json(output / "records" / (identity(t) + ".json")) for t in expanded
                   if (output / "records" / (identity(t) + ".json")).exists()]
        metadata["counts"] = {state: sum(r["status"] == state for r in records)
                              for state in ("ok", "failed", "numerical_failure")}
        metadata["counts"]["pending"] = len(expanded) - len(records)
        write_json(manifest, metadata)
    return metadata
