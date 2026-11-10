"""Download and unpack the SMD DefDB dataset (Zenodo record 20001570).

The record publishes a single 825 MB RAR. Zenodo throttles each connection to
roughly 100 KB/s, so this fetch uses bounded parallel range requests, resumes
completed chunks across runs, verifies the published MD5, and extracts with the
bsdtar/libarchive shipped in Windows 11 (which reads RAR4/RAR5).

Usage:  .venv\\Scripts\\python.exe training\\fetch_smd_defdb.py [--force]
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data/datasets/smd_defdb"
ARCHIVE = STORE / "PCB_augmented.rar"
PARTS = STORE / "parts"
URL = "https://zenodo.org/records/20001570/files/PCB_augmented.rar?download=1"
SIZE = 825_641_526
MD5 = "4ccc072f1abd659089265066af1f9ff7"
CHUNK = 4 * 1024 * 1024
THREADS = 20
ATTEMPTS = 5
CLASSES = ["damaged_component", "excess_solder", "good_solder",
           "misaligned_component", "missing_component", "poor_solder",
           "solder_bridge"]


def extracted() -> bool:
    return all((STORE / name).is_dir() for name in CLASSES)


def chunk_path(index: int) -> Path:
    return PARTS / f"{index:05d}.part"


def expected_size(index: int) -> int:
    return min(CHUNK, SIZE - index * CHUNK)


def fetch(index: int) -> None:
    target = chunk_path(index)
    want = expected_size(index)
    if target.is_file() and target.stat().st_size == want:
        return
    start = index * CHUNK
    end = start + want - 1
    for attempt in range(ATTEMPTS):
        try:
            with requests.get(URL, headers={"Range": f"bytes={start}-{end}"},
                              timeout=180, stream=True) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"HTTP {response.status_code}")
                buffer = bytearray()
                for block in response.iter_content(256 * 1024):
                    buffer.extend(block)
                if len(buffer) != want:
                    raise RuntimeError(f"chunk {index}: {len(buffer)} != {want}")
                target.write_bytes(bytes(buffer))
                return
        except Exception as exc:  # noqa: BLE001 - retry any transport failure
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(2 * (attempt + 1))
            print(f"retry {index}: {exc}", flush=True)


def main() -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    if extracted():
        print("dataset already extracted")
        return
    bundled = STORE / "dataset.tar"
    if bundled.is_file():
        print("extracting bundled dataset.tar", flush=True)
        result = subprocess.run(["tar", "-xf", str(bundled), "-C", str(STORE)],
                                capture_output=True, text=True)
        if result.returncode == 0 and extracted():
            print("extracted to", STORE)
            return
        print("bundled tar extraction failed:", (result.stderr or result.stdout)[:300], flush=True)
    if ARCHIVE.is_file() and ARCHIVE.stat().st_size == SIZE:
        print("archive already present")
    else:
        if ARCHIVE.exists():
            ARCHIVE.unlink()
        PARTS.mkdir(exist_ok=True)
        total = (SIZE + CHUNK - 1) // CHUNK
        print(f"downloading {SIZE/2**20:.0f} MiB in {total} chunks via {THREADS} workers", flush=True)
        started = time.time()
        with ThreadPoolExecutor(THREADS) as pool:
            for done, _ in enumerate(pool.map(fetch, range(total)), 1):
                if done % 10 == 0 or done == total:
                    elapsed = time.time() - started
                    rate = done * CHUNK / 2**20 / max(elapsed, 1)
                    print(f"chunks {done}/{total}  {rate:.2f} MiB/s", flush=True)
        print("merging", flush=True)
        with ARCHIVE.open("wb") as stream:
            for index in range(total):
                stream.write(chunk_path(index).read_bytes())
        shutil.rmtree(PARTS)
    digest = hashlib.md5()
    with ARCHIVE.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != MD5:
        raise SystemExit(f"MD5 mismatch: {digest.hexdigest()}")
    print("md5 ok", flush=True)
    if extracted():
        print("already extracted")
        return
    print("extracting", flush=True)
    extract(ARCHIVE, STORE)
    if not extracted():
        raise SystemExit("archive extracted but class folders are missing")
    print("extracted to", STORE)


def extract(archive: Path, destination: Path) -> None:
    """Try libarchive first (Windows bsdtar), then common Linux RAR tools."""
    tools = [
        ["tar", "-xf", str(archive), "-C", str(destination)],
        ["bsdtar", "-xf", str(archive), "-C", str(destination)],
        ["7z", "x", "-y", f"-o{destination}", str(archive)],
        ["unrar", "x", "-o+", str(archive), str(destination) + os.sep],
        ["unar", "-o", str(destination), str(archive)],
    ]
    errors = []
    for command in tools:
        if not shutil.which(command[0]):
            continue
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and extracted():
            return
        errors.append(f"{command[0]}: {(result.stderr or result.stdout).strip()[:200]}")
    raise SystemExit("cannot extract RAR; install p7zip-full or unrar. Tried: " + " | ".join(errors))


if __name__ == "__main__":
    main()
