"""Step 3b: train a lightweight YOLO (nano) on the synthetic-defect tiles, CPU only.

Run:  python -m demo.train_yolo [--epochs 30]
"""
from __future__ import annotations

import argparse
import json
import shutil

from . import config as C


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--base", default="yolo11n.pt", help="pretrained nano checkpoint (downloaded once)")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    from ultralytics import YOLO, settings

    settings.update({"sync": False})  # no usage analytics
    data = C.YOLO_DATA / "data.yaml"
    if not data.exists():
        raise SystemExit("synthetic dataset missing; run `python -m demo.synth` first")
    C.YOLO_RUNS.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(C.OUT / args.base) if (C.OUT / args.base).exists() else args.base)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=16, device="cpu", workers=4,
                project=str(C.YOLO_RUNS), name="synth", exist_ok=True, patience=15, plots=False,
                degrees=3, flipud=0.0, fliplr=0.0, mosaic=0.5, hsv_h=0.01, hsv_s=0.4, hsv_v=0.4, verbose=False)
    best = C.YOLO_RUNS / "synth" / "weights" / "best.pt"
    shutil.copy(best, C.YOLO_WEIGHTS)
    metrics = YOLO(str(C.YOLO_WEIGHTS)).val(data=str(data), imgsz=args.imgsz, device="cpu", plots=False, verbose=False)
    report = {
        "note": "validation tiles are SYNTHETIC defects on two normal boards never used for training",
        "epochs": args.epochs,
        "map50": round(float(metrics.box.map50), 3),
        "map50_95": round(float(metrics.box.map), 3),
        "precision": round(float(metrics.box.mp), 3),
        "recall": round(float(metrics.box.mr), 3),
        "per_class_map50": {C.CLASSES[int(c)]: round(float(v), 3) for c, v in zip(metrics.box.ap_class_index, metrics.box.ap50)},
    }
    C.YOLO_REPORT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
