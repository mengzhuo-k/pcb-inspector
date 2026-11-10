"""Run the whole offline build: register -> PatchCore -> synthetic dataset -> YOLO.

Run:  python -m demo.build [--skip-yolo] [--epochs 30]
"""
from __future__ import annotations

import argparse
import sys

from . import patchcore, register, synth, train_yolo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-yolo", action="store_true", help="only registration + PatchCore (about 2 minutes on CPU)")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    print("== 1/4 four-point registration ==")
    register.main()
    print("== 2/4 PatchCore memory bank ==")
    patchcore.main()
    if args.skip_yolo:
        return
    print("== 3/4 synthetic defects ==")
    synth.main()
    print("== 4/4 YOLO on synthetic defects (CPU, roughly 15 minutes) ==")
    sys.argv = [sys.argv[0], "--epochs", str(args.epochs)]
    train_yolo.main()


if __name__ == "__main__":
    main()
