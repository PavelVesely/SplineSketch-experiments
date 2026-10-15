"""Official inputs with resumable downloads, atomic extraction and provenance."""
import csv
import gzip
import math
import os
from pathlib import Path
import shutil
import struct
import time
import urllib.error
import urllib.request
import zipfile

from .common import digest, load_json, write_json

URLS = {
    "hepmass_train": "https://archive.ics.uci.edu/ml/machine-learning-databases/00347/all_train.csv.gz",
    "hepmass_test": "https://archive.ics.uci.edu/ml/machine-learning-databases/00347/all_test.csv.gz",
    "power": "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
    "books": "https://www.dropbox.com/s/y2u3nbanbnbmg7n/books_800M_uint64.zst?dl=1",
}
BOOKS_MD5 = "8708eb3e1757640ba18dcd3a0dbb53bc"


def fetch(url, path):
    """Resume only if the server honors Range; retries keep the .part file."""
    partial = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"User-Agent": "SplineSketch-reproduction/1",
            **({"Range": f"bytes={offset}-"} if offset else {})})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                resume = response.status == 206 and offset > 0
                if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an unexpected Content-Range")
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise ValueError(f"Dataset URL returned HTML rather than data: {url}")
                received = 0
                with partial.open("ab" if resume else "wb") as stream:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
                        received += len(block)
                length = response.headers.get("Content-Length")
                if length is not None and received != int(length):
                    raise OSError(f"Incomplete download: {received}/{length} bytes")
            os.replace(partial, path)
            return
        except (urllib.error.URLError, OSError) as error:
            if attempt == 2:
                raise RuntimeError(f"Download failed: {url}. Rerun to resume, or supply the file manually. {error}") from error
            time.sleep(attempt + 1)


def validate_csv(path, delimiter, column, expected_rows):
    rows, missing = 0, 0
    with path.open(newline="") as stream:
        reader = csv.reader(stream, delimiter=delimiter)
        header = next(reader, None)
        if not header or len(header) <= column:
            raise ValueError(f"Invalid CSV header in {path}")
        if delimiter == ";" and header[column] != "Global_active_power":
            raise ValueError(f"Unexpected Power column in {path}")
        for row in reader:
            rows += 1
            if len(row) <= column:
                raise ValueError(f"Short row in {path}: {rows}")
            value = row[column].strip()
            if delimiter == ";" and value in ("", "?"):
                missing += 1
            elif not math.isfinite(float(value)):
                raise ValueError(f"Nonfinite value in {path}: {rows}")
    if rows != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in {path}, found {rows}")
    return dict(source_rows=rows, missing_rows=missing, valid_values=rows - missing)


def validate_books(path):
    if path.stat().st_size != 8 * (800000000 + 1):
        raise ValueError(f"Incorrect Books file size: {path}")
    with path.open("rb") as stream:
        if struct.unpack("<Q", stream.read(8))[0] != 800000000:
            raise ValueError(f"Incorrect SOSD count header: {path}")
    if digest(path, "md5") != BOOKS_MD5:
        raise ValueError(f"Books checksum mismatch: {path}")
    return dict(source_rows=800000000, missing_rows=0, valid_values=800000000, upstream_md5=BOOKS_MD5)


def download_datasets(names, directory, books_url=None):
    directory = Path(directory).resolve()
    manifest = directory / "downloads.json"
    metadata = load_json(manifest) if manifest.exists() else {"files": {}}
    archives = directory / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    entries = []
    if "hepmass" in names:
        for split, count in (("train", 7000000), ("test", 3500000)):
            target = directory / f"hepmass/all_{split}.csv"
            entries.append((target, URLS[f"hepmass_{split}"], "gz", count))
    if "power" in names:
        entries.append((directory / "household_power_consumption/household_power_consumption.txt",
                        URLS["power"], "zip", 2075259))
    if "books" in names:
        entries.append((directory / "SOSDdata/books_800M_uint64", books_url or URLS["books"], "zst", 800000000))
    for target, url, kind, expected in entries:
        relative = str(target.relative_to(directory))
        known = metadata["files"].get(relative)
        if target.exists() and known:
            if digest(target) != known["sha256"]:
                raise ValueError(f"Checksum failed for {target}. Move the corrupt file aside and rerun.")
            print(f"Verified {target}", flush=True)
            continue
        if not target.exists():
            archive = archives / (target.name + "." + kind)
            if not archive.exists():
                print(f"Downloading {url}", flush=True)
                fetch(url, archive)
            temporary = target.with_name(target.name + ".tmp")
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with temporary.open("wb") as destination:
                    if kind == "gz":
                        with gzip.open(archive, "rb") as source:
                            shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
                    elif kind == "zip":
                        with zipfile.ZipFile(archive) as zipped:
                            matches = [name for name in zipped.namelist() if Path(name).name == target.name]
                            if len(matches) != 1:
                                raise ValueError(f"Cannot identify {target.name} in {archive}")
                            with zipped.open(matches[0]) as source:
                                shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
                    else:
                        import zstandard
                        with archive.open("rb") as source:
                            zstandard.ZstdDecompressor().copy_stream(source, destination)
                stats = validate_books(temporary) if kind == "zst" else validate_csv(
                    temporary, ";" if kind == "zip" else ",", 2 if kind == "zip" else 1, expected)
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()
        else:
            stats = validate_books(target) if kind == "zst" else validate_csv(
                target, ";" if kind == "zip" else ",", 2 if kind == "zip" else 1, expected)
        archive = archives / (target.name + "." + kind)
        metadata["files"][relative] = dict(url=url, sha256=digest(target), bytes=target.stat().st_size,
            archive_sha256=digest(archive) if archive.exists() else None, **stats)
        write_json(manifest, metadata)
        print(f"Validated {target}: {stats}", flush=True)
    return metadata
