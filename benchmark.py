#!/usr/bin/env python3
"""Benchmark OnPair+ (opp) against lz4, zstd, brotli, and fsst.

Usage: ./benchmark.py <dataset> [--limit N] [--repeats N]

Datasets (downloaded into ./data on first use):
  bitext   machine-translation text corpora (db.in.tum.de)
  dbtext   database string columns (db.in.tum.de)
  wiki9    first 10^9 bytes of English Wikipedia (enwik9, mattmahoney.net)

The measurements are done by the benchmark binary (benchmark/, built automatically): it loads each
file into memory, runs every compressor library in-process for the given number of repetitions
(keeping the best time), and verifies each roundtrip. This script only aggregates the results.
"""

import argparse
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
BENCHMARK_DIR = ROOT / "benchmark"
BENCHMARK_BINARY = BENCHMARK_DIR / "build" / "benchmark"

DATASET_URLS = {
    "bitext": "https://db.in.tum.de/~schmidt/data/bitext.tar.gz",
    "dbtext": "https://db.in.tum.de/~schmidt/data/dbtext.tar.gz",
    "wiki9": "https://mattmahoney.net/dc/enwik9.zip",
}


def download(url: str, target: Path) -> None:
    print(f"Downloading {url} ...")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    tmp.rename(target)


def ensure_dataset(name: str) -> list[Path]:
    """Download and extract the dataset if needed; return its benchmark files."""
    dataset_dir = DATA_DIR / name
    if not dataset_dir.is_dir():
        url = DATASET_URLS[name]
        archive = DATA_DIR / Path(url).name
        if not archive.exists():
            download(url, archive)
        print(f"Extracting {archive.name} ...")
        if archive.suffix == ".zip":
            dataset_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dataset_dir)
        else:
            with tarfile.open(archive) as tf:
                tf.extractall(DATA_DIR)
        if not dataset_dir.is_dir():
            sys.exit(f"error: extracting {archive} did not produce {dataset_dir}")
        archive.unlink()

    files = sorted(p for p in dataset_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    if not files:
        sys.exit(f"error: no files found in {dataset_dir}")
    return files


def ensure_benchmark_binary() -> Path:
    """Configure and build the benchmark project (bootstraps vcpkg on first use)."""
    for tool in ("cmake", "git"):
        if not shutil.which(tool):
            sys.exit(f"error: '{tool}' is required to build the benchmark. Please install it.")
    if not BENCHMARK_BINARY.exists():
        print("Building the benchmark binary (first build bootstraps vcpkg; this takes a while) ...")
    subprocess.run(["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"], cwd=BENCHMARK_DIR,
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["cmake", "--build", "build"], cwd=BENCHMARK_DIR, check=True,
                   stdout=subprocess.DEVNULL)
    return BENCHMARK_BINARY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=sorted(DATASET_URLS))
    parser.add_argument("--limit", type=int, default=None, help="benchmark only the first N files")
    parser.add_argument("--repeats", type=int, default=3, help="repetitions per measurement (best time wins)")
    args = parser.parse_args()

    binary = ensure_benchmark_binary()
    files = ensure_dataset(args.dataset)
    if args.limit:
        files = files[: args.limit]

    totals: dict[str, dict[str, float]] = {}
    total_bytes = sum(f.stat().st_size for f in files)
    print(f"\nBenchmarking {args.dataset}: {len(files)} file(s), "
          f"{total_bytes / (1 << 20):.1f} MiB total, best of {args.repeats} repetition(s)\n")

    for i, infile in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {infile.name}", flush=True)
        result = subprocess.run([str(binary), str(infile), str(args.repeats)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"error: benchmark failed on {infile}:\n{result.stderr}")
        for line in result.stdout.splitlines():
            name, original, compressed, ctime, dtime = line.split(",")
            entry = totals.setdefault(name, {"original": 0, "compressed": 0, "ctime": 0.0, "dtime": 0.0})
            entry["original"] += int(original)
            entry["compressed"] += int(compressed)
            entry["ctime"] += float(ctime)
            entry["dtime"] += float(dtime)

    print(f"\n{'compressor':<12} {'ratio':>8} {'enc MiB/s':>12} {'dec MiB/s':>12}")
    print("-" * 46)
    for name, t in totals.items():
        ratio = t["original"] / t["compressed"] if t["compressed"] else 0.0
        enc = t["original"] / (1 << 20) / t["ctime"] if t["ctime"] else 0.0
        dec = t["original"] / (1 << 20) / t["dtime"] if t["dtime"] else 0.0
        print(f"{name:<12} {ratio:>8.3f} {enc:>12.1f} {dec:>12.1f}")
    print("\nnote: in-memory library calls (lz4 default, zstd -3, brotli -q 5), best-of-repeats,"
          "\nno file I/O or process startup in the timings")


if __name__ == "__main__":
    main()
