"""Seeded inputs, inclusive ranks, and independently verified input caches."""
import csv
import itertools
import os
from pathlib import Path

import numpy as np

import IID_generators as generators
from .common import ROOT, SCHEMA, digest, identity, load_json, write_json
from .config import REAL

EXPECTED_ROWS = {"hepmass": 10500000, "power": 2075259, "books": 800000000}


def input_paths(name, directory):
    directory = Path(directory)
    return {
        "hepmass": [directory / "hepmass/all_train.csv", directory / "hepmass/all_test.csv"],
        "power": [directory / "household_power_consumption/household_power_consumption.txt"],
        "books": [directory / "SOSDdata/books_800M_uint64"],
    }[name]


def derived_seed(seed, name, n):
    return int(identity([seed, name, n])[:8], 16)


def generate(name, n, seed):
    # Local RandomState preserves the original generator's sampling algorithm.
    return np.asarray(getattr(generators, name)(n, rng=np.random.RandomState(seed)), dtype=np.float64)


def read_real(name, directory, seed, limit=None):
    paths = input_paths(name, directory)
    stats = {"rows": 0, "missing": 0, "limited": limit is not None}
    if name == "books":
        path = paths[0]
        with path.open("rb") as stream:
            header = np.fromfile(stream, dtype="<u8", count=1)
        if len(header) != 1 or path.stat().st_size != 8 * (int(header[0]) + 1):
            raise ValueError(f"Invalid SOSD length header: {path}")
        if int(header[0]) != EXPECTED_ROWS[name] and limit is None:
            raise ValueError(f"Expected {EXPECTED_ROWS[name]} Books values, found {header[0]}")
        raw = np.memmap(path, mode="r", dtype="<u8", offset=8, shape=(int(header[0]),))
        # Full input is permuted before conversion, as in the original pipeline.
        # A limited run permutes only the prefix and is explicitly a subset run.
        data = np.array(raw[:limit] if limit else raw, copy=True)
        np.random.RandomState(seed).shuffle(data)
        stats["rows"] = len(data)
        return data.astype(np.float64), stats

    def values():
        for path in paths:
            with path.open(newline="") as stream:
                reader = csv.reader(stream, delimiter=";" if name == "power" else ",")
                header = next(reader, None)
                column = 2 if name == "power" else 1
                if not header or len(header) <= column:
                    raise ValueError(f"Invalid header in {path}")
                if name == "power" and header[2] != "Global_active_power":
                    raise ValueError(f"Unexpected Power column: {header[2]}")
                for row in reader:
                    stats["rows"] += 1
                    if len(row) <= column:
                        raise ValueError(f"Short row {stats['rows']} in {path}")
                    value = row[column].strip()
                    if name == "power" and value in ("", "?"):
                        stats["missing"] += 1
                        continue
                    number = float(value)
                    if not np.isfinite(number):
                        raise ValueError(f"Nonfinite value in {path}, row {stats['rows']}")
                    yield number

    iterator = values()
    if limit:
        iterator = itertools.islice(iterator, limit)
    data = np.fromiter(iterator, dtype=np.float64)
    if not limit and stats["rows"] != EXPECTED_ROWS[name]:
        raise ValueError(f"Expected {EXPECTED_ROWS[name]} {name} source rows; got {stats['rows']}")
    return data, stats


def select_queries(sorted_data, count):
    """Exactly min(count,n) sorted values, starting at index 0; ranks include ties."""
    count = min(count, len(sorted_data))
    indices = np.arange(count, dtype=np.int64) * len(sorted_data) // count
    queries = sorted_data[indices]
    return queries, np.searchsorted(sorted_data, queries, side="right")


def write_values(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    # Chunking avoids a second full text representation in memory.
    with temporary.open("w") as stream:
        for start in range(0, len(data), 100000):
            np.savetxt(stream, data[start:start + 100000], fmt="%.17g")
    os.replace(temporary, path)


def verify_cache(directory, metadata):
    for name, checksum in metadata["files"].items():
        path = directory / name
        if not path.is_file() or digest(path) != checksum:
            raise ValueError(f"Input cache failed checksum verification: {path}. Remove this cache directory and rerun.")


def prepare_data(name, nominal_n, seed, dataset_dir, cache_dir, limit):
    spec = dict(schema=SCHEMA, dataset=name, nominal_n=None if name in REAL else nominal_n,
                seed=seed, limit=limit if name in REAL else None,
                generators=digest(ROOT / "IID_generators.py"), loader=digest(Path(__file__)))
    if name in REAL:
        spec["sources"] = {str(p.resolve()): digest(p) for p in input_paths(name, dataset_dir)}
    key = identity(spec)
    directory = Path(cache_dir) / key
    manifest = directory / "data.json"
    if manifest.exists():
        metadata = load_json(manifest)
        verify_cache(directory, metadata)
        return directory, metadata
    directory.mkdir(parents=True, exist_ok=True)
    local_seed = derived_seed(seed, name, spec["nominal_n"])
    if name in REAL:
        data, stats = read_real(name, dataset_dir, local_seed, limit)
    else:
        data = generate(name, nominal_n, local_seed)
        stats = {"rows": len(data), "missing": 0, "limited": False}
    if not len(data) or not np.isfinite(data).all():
        raise ValueError(f"Empty or nonfinite input: {name}")
    write_values(directory / "data.txt", data)
    data.sort()
    np.save(directory / "sorted.npy", data)
    metadata = dict(spec=spec, key=key, actual_n=len(data), derived_seed=local_seed,
                    preprocessing=stats, files={filename: digest(directory / filename)
                    for filename in ("data.txt", "sorted.npy")})
    write_json(manifest, metadata)
    return directory, metadata


def prepare_queries(directory, data_metadata, count):
    manifest = directory / f"queries-{count}.json"
    if manifest.exists():
        metadata = load_json(manifest)
        verify_cache(directory, metadata)
    else:
        sorted_data = np.load(directory / "sorted.npy", mmap_mode="r")
        queries, ranks = select_queries(sorted_data, count)
        query_path = directory / f"queries-{count}.txt"
        rank_path = directory / f"ranks-{count}.npy"
        write_values(query_path, queries)
        np.save(rank_path, ranks)
        metadata = dict(data_key=data_metadata["key"], requested_queries=count,
                        actual_queries=len(queries), files={p.name: digest(p) for p in (query_path, rank_path)})
        write_json(manifest, metadata)
    return metadata, np.load(directory / f"ranks-{count}.npy", mmap_mode="r")
