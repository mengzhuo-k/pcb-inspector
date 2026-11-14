"""Train and publish the SMD DefDB seven-class defect classifier.

Run after ``prepare_smd_defdb.py``:
    .venv\\Scripts\\python.exe training\\train_smd_classifier.py --epochs 40

The held-out ``test`` split is never seen during training or early stopping; it
is used once here for the final top-1/top-5 metrics, the confusion matrix, and
the accept/reject threshold stored in the model card consumed by the web app.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

# Set the CPU thread budget before torch/ultralytics import so OpenMP and
# oneDNN pick it up. The ModelScope launcher exports PCB_THREADS=8.
_THREADS = os.environ.get("PCB_THREADS")
if _THREADS:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, _THREADS)

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/datasets/smd_defdb/cls"
MODELS = ROOT / "models"
CLASSES = ["damaged_component", "excess_solder", "good_solder",
           "misaligned_component", "missing_component", "poor_solder",
           "solder_bridge"]
GOOD = "good_solder"
LABELS = {
    "damaged_component": "焊盘/元件损伤",
    "excess_solder": "多锡",
    "good_solder": "未见明显缺陷",
    "misaligned_component": "元件偏移",
    "missing_component": "缺件",
    "poor_solder": "虚焊/冷焊",
    "solder_bridge": "桥接",
}


def probabilities(model, imgsz, device="cpu"):
    rows = []
    for class_name in CLASSES:
        folder = DATA / "test" / class_name
        for path in sorted(folder.glob("*.jpg")):
            with Image.open(path) as image:
                result = model.predict(image.convert("RGB"), imgsz=imgsz,
                                       verbose=False, device=device)[0]
            values = result.probs.data.cpu().numpy()
            order = [result.names[i] for i in range(len(values))]
            rows.append(dict(path=path.name, truth=class_name,
                             probs={name: float(values[i]) for i, name in enumerate(order)}))
    return rows


def binary_curve(rows):
    truth = np.array([row["truth"] != GOOD for row in rows])
    scores = np.array([1 - row["probs"].get(GOOD, 0) for row in rows])
    curve = []
    for threshold in np.round(np.arange(0.05, 0.96, 0.01), 2):
        predicted = scores >= threshold
        tp = int(np.sum(predicted & truth))
        fp = int(np.sum(predicted & ~truth))
        fn = int(np.sum(~predicted & truth))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        curve.append(dict(threshold=float(threshold), precision=precision,
                          recall=recall, f1=f1))
    return curve


def choose_threshold(curve):
    # Fixed operating point. The test split is also used for calibration, so
    # picking the threshold that maximizes recall (the old code chose 0.05)
    # overfits this split and turns the gate into "P(good) > 0.95". 0.5 means
    # "more likely defective than good" and is stable across boards.
    return next(row for row in curve if abs(row["threshold"] - 0.5) < 1e-9)


def main():
    global DATA
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--model", default=str(ROOT / "training/yolo11n-cls.pt"))
    parser.add_argument("--name", default="smd_defdb_cls")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache", choices=["none", "ram", "disk"], default="none")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--version", default="smd-defdb-v1")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output-card", type=Path)
    parser.add_argument("--output-weights", type=Path)
    parser.add_argument("--platform-database", type=Path)
    parser.add_argument("--job-id")
    args = parser.parse_args()
    DATA = args.data.resolve()
    if not (DATA / "train").is_dir():
        raise SystemExit("run training/prepare_smd_defdb.py first")
    if args.threads:
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    torch.backends.mkldnn.enabled = True
    started = time.time()
    model = YOLO(args.model)
    result = model.train(data=str(DATA), epochs=args.epochs, imgsz=args.imgsz,
                         batch=args.batch, device=args.device, workers=args.workers,
                         cache=(args.cache if args.cache != "none" else False),
                         channels_last=args.channels_last,
                         project=str(ROOT / "training/runs"), name=args.name,
                         exist_ok=True, pretrained=True, seed=20261110,
                         deterministic=True, patience=args.patience,
                         cos_lr=True, plots=False, verbose=True)
    best = Path(result.save_dir) / "weights/best.pt"
    if not best.is_file():
        raise SystemExit("training produced no best.pt")
    if args.platform_database and args.job_id:
        import sqlite3
        with sqlite3.connect(args.platform_database, timeout=30) as conn:
            conn.execute("UPDATE jobs SET testing_started=? WHERE id=?",(time.time(),args.job_id))
    trained = YOLO(str(best))
    metrics = trained.val(data=str(DATA), split="val", imgsz=args.imgsz,
                          batch=args.batch, device=args.device, verbose=False, plots=False)
    rows = probabilities(trained, args.imgsz, args.device)
    top1 = sum(max(row["probs"], key=row["probs"].get) == row["truth"] for row in rows) / len(rows)
    top5 = sum(row["truth"] in sorted(row["probs"], key=row["probs"].get, reverse=True)[:5]
               for row in rows) / len(rows)
    matrix = {row["truth"]: {name: 0 for name in CLASSES} for row in rows}
    for row in rows:
        matrix[row["truth"]][max(row["probs"], key=row["probs"].get)] += 1
    per_class_recall = {}
    for name in CLASSES:
        support = sum(matrix[name].values())
        per_class_recall[name] = round(matrix[name][name] / support, 4) if support else None
    curve = binary_curve(rows)
    chosen = choose_threshold(curve)
    at = chosen
    card = dict(version=args.version,
                created=datetime.now().astimezone().isoformat(timespec="seconds"),
                weights=f"smd_defect_cls_{args.version.replace('smd-defdb-', '')}.pt",
                imgsz=args.imgsz, threshold=round(at["threshold"], 2),
                classes=CLASSES, labels=LABELS,
                metrics=dict(top1=round(top1, 4), top5=round(top5, 4),
                             val_top1=round(float(metrics.top1), 4),
                             val_top5=round(float(metrics.top5), 4),
                             test_images=len(rows),
                             defect_precision=round(at["precision"], 4),
                             defect_recall=round(at["recall"], 4),
                             defect_f1=round(at["f1"], 4),
                             defect_accuracy=round(float(np.mean(
                                 (np.array([1 - row["probs"].get(GOOD, 0) for row in rows]) >= at["threshold"])
                                 == np.array([row["truth"] != GOOD for row in rows]))), 4),
                             per_class_recall=per_class_recall,
                             confusion_matrix=matrix),
                dataset=dict(name="SMD DefDB (PCB SMD component)",
                             doi="10.5281/zenodo.20001570",
                             url="https://zenodo.org/records/20001570",
                             license="CC BY 4.0",
                             images=1190, classes=7,
                             split=("grouped by ten-image augmentation block, duplicate blocks merged; "
                                    "seed 20261110; train 810 / val 140 / test 240"),
                             train_images=810, val_images=140, test_images=len(rows)),
                training=dict(model=Path(args.model).name, epochs=args.epochs,
                              imgsz=args.imgsz, batch=args.batch, device=args.device,
                              patience=args.patience, optimizer="auto (AdamW)",
                              workers=args.workers, cache=args.cache,
                              channels_last=args.channels_last,
                              threads=args.threads or None,
                              seconds=round(time.time() - started, 1)))
    MODELS.mkdir(exist_ok=True)
    target = args.output_weights or MODELS / card["weights"]
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(best, target)
    output_card = args.output_card or MODELS / "smd_defect_cls_v1.json"
    output_card.parent.mkdir(parents=True,exist_ok=True)
    output_card.write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    run_card = Path(result.save_dir) / "metrics.json"
    run_card.write_text(json.dumps(dict(card=card, curve=curve, predictions=rows),
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(card["metrics"], ensure_ascii=False, indent=2))
    print("weights", target, target.stat().st_size)
    print("card", output_card)


if __name__ == "__main__":
    main()
