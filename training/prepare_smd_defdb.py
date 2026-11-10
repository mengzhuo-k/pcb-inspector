"""Build a leakage-safe train/val classification tree from SMD DefDB.

The published archive numbers each class ``<class>_<i>.jpg`` and groups the ten
augmentations of one source photo in consecutive blocks of ten (i // 10). Some
blocks are byte-identical copies of one another, so blocks are first merged
whenever any image MD5 is shared. Splits are made over these merged groups,
never over individual images, so an augmented sibling can never appear in both
train and val.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data/datasets/smd_defdb"
CLASSES = ["damaged_component", "excess_solder", "good_solder",
           "misaligned_component", "missing_component", "poor_solder",
           "solder_bridge"]
SEED = 20261110
VAL_GROUPS = 2   # early-stopping only, of ~16-17 source groups per class
TEST_GROUPS = 3  # held out for the final metrics and threshold calibration


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def number(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def build() -> dict:
    files = []
    for cls in CLASSES:
        directory = STORE / cls
        if not directory.is_dir():
            raise SystemExit(f"missing class folder: {directory}")
        for path in sorted(directory.glob("*.jpg"), key=number):
            index = number(path)
            files.append(dict(class_name=cls, path=path, index=index, block=index // 10))
    uf = UnionFind()
    by_digest = defaultdict(list)
    for item in files:
        digest = hashlib.md5(item["path"].read_bytes()).hexdigest()
        by_digest[digest].append(item)
    for group in by_digest.values():
        for item in group[1:]:
            uf.union((group[0]["class_name"], group[0]["block"]),
                     (item["class_name"], item["block"]))
    for item in files:
        item["group"] = uf.find((item["class_name"], item["block"]))
    return dict(files=files, duplicates=sum(len(group) - 1 for group in by_digest.values()))


def assign_groups(files) -> dict:
    groups = defaultdict(list)
    for item in files:
        groups[item["group"]].append(item)
    split_by_group = {}
    by_class = defaultdict(list)
    for key in groups:
        by_class[key[0]].append(key)
    for offset, cls in enumerate(CLASSES):
        keys = sorted(by_class[cls])
        random.Random(SEED + offset).shuffle(keys)
        for position, key in enumerate(keys):
            if position < TEST_GROUPS:
                split_by_group[key] = "test"
            elif position < TEST_GROUPS + VAL_GROUPS:
                split_by_group[key] = "val"
            else:
                split_by_group[key] = "train"
    return split_by_group


def link(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    data = build()
    files = data["files"]
    split_by_group = assign_groups(files)
    destination = STORE / "cls"
    if destination.exists():
        shutil.rmtree(destination)
    summary = {}
    manifest = []
    for item in files:
        split = split_by_group[item["group"]]
        target_dir = destination / split / item["class_name"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{item['class_name']}_{item['index']:03d}.jpg"
        link(item["path"], target)
        summary.setdefault(split, defaultdict(int))[item["class_name"]] += 1
        manifest.append(dict(file=target.relative_to(destination).as_posix(),
                             source=item["path"].relative_to(STORE).as_posix(),
                             class_name=item["class_name"], group=f"{item['class_name']}#{item['group'][1]}",
                             split=split))
    (destination / "split.json").write_text(
        json.dumps(dict(seed=SEED, val_groups=VAL_GROUPS, test_groups=TEST_GROUPS,
                        policy="grouped by ten-image augmentation block, duplicate blocks merged",
                        counts={split: dict(counts) for split, counts in summary.items()},
                        files=manifest), ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("duplicate files merged:", data["duplicates"])
    for split in ("train", "val", "test"):
        counts = summary.get(split, {})
        print(split, sum(counts.values()), dict(counts))


if __name__ == "__main__":
    main()
