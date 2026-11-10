"""Build the ModelScope upload bundle (code + optional dataset)."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data/datasets/smd_defdb/dataset.tar"
ARCHIVE = ROOT / "data/datasets/smd_defdb/PCB_augmented.rar"
INCLUDE_FILES = ["app.py", "batch_report.py", "defect_classifier.py", "learning.py",
                 "learning_fit.py", "model_runtime.py", "platform_app.py", "requirements.txt", "README.md"]
INCLUDE_DIRS = ["static", "tests"]
TRAINING_FILES = ["fetch_smd_defdb.py", "prepare_smd_defdb.py", "train_smd_classifier.py",
                  "platform_fit.py", "yolo11n-cls.pt", "modelscope"]
SKIP = {"__pycache__", "runs", ".venv", "data", "logs"}


def add_tree(archive, folder, prefix=""):
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or any(part in SKIP for part in path.parts):
            continue
        if path.suffix in {".pyc", ".cache"}:
            continue
        archive.write(path, prefix + path.relative_to(folder).as_posix())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-only", action="store_true",
                        help="exclude the dataset; the instance downloads it from Zenodo")
    args = parser.parse_args()
    target = ROOT.parent / ("pcb-inspector-code.zip" if args.code_only
                            else "pcb-inspector-modelscope.zip")
    target.unlink(missing_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in INCLUDE_FILES:
            path = ROOT / name
            if path.is_file():
                archive.write(path, name)
        for name in INCLUDE_DIRS:
            add_tree(archive, ROOT / name, name + "/")
        archive.write(ROOT / "models/missing_connection_v7.pt", "models/missing_connection_v7.pt")
        archive.write(ROOT / "models/DATASET-LICENSE.md", "models/DATASET-LICENSE.md")
        for name in TRAINING_FILES:
            path = ROOT / "training" / name
            if path.is_file():
                archive.write(path, f"training/{name}")
            elif path.is_dir():
                add_tree(archive, path, f"training/{name}/")
        for name in ("board_ok.png", "board_missing.png"):
            path = ROOT / "training/missing_connection/images/train" / name
            if path.is_file():
                archive.write(path, f"training/missing_connection/images/train/{name}")
        archive.write(ROOT / "training/missing_connection/data.yaml",
                      "training/missing_connection/data.yaml")
        # Ship the dataset so the instance does not need Zenodo. Prefer the
        # plain tar (no RAR tool needed on Linux); fall back to the original RAR.
        if not args.code_only:
            source = DATASET if DATASET.is_file() else ARCHIVE
            if source.is_file():
                archive.write(source, f"data/datasets/smd_defdb/{source.name}",
                              compress_type=zipfile.ZIP_STORED)
            else:
                print("WARNING: dataset archive missing, bundle will download on the instance")
    print(target, f"{target.stat().st_size/2**20:.1f} MiB")


if __name__ == "__main__":
    main()
