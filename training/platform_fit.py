"""Fit one platform training job and write metrics.json.

Invoked as a subprocess by platform_app.py so a training crash never takes the
web process down and the log can be streamed from a file.
"""
from __future__ import annotations

import json
import hashlib
import shutil
import sys
import traceback
from pathlib import Path

from ultralytics import YOLO


def evaluate_detector(model, dataset, split, imgsz, batch, device):
    names = list(model.names.values())
    support = [0] * len(names)
    images = list((dataset / "images" / split).glob("*"))
    normal_paths = []
    for image in images:
        label = dataset / "labels" / split / (image.stem + ".txt")
        if not label.is_file() or not label.read_text().strip():
            normal_paths.append(str(image))
    for path in (dataset / "labels" / split).glob("*.txt"):
        for line in path.read_text().splitlines():
            if line.strip():
                support[int(line.split()[0])] += 1
    per_class = {name: dict(precision=None, recall=None, ap50=None, ap=None,
                            support=support[index], status="未评估")
                 for index, name in enumerate(names)}
    result = dict(names=names, per_class=per_class, val_images=len(images),
                  instances=sum(support), evaluation_status="unavailable",
                  map50=None, map=None)
    split_hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path in images}
    other_hashes = {hashlib.sha256(path.read_bytes()).hexdigest()
                    for other in ("train", "val", "test") if other != split
                    for path in (dataset / "images" / other).glob("*")}
    if split_hashes & other_hashes:
        result["reason"] = "相同图片出现在多个划分，无法独立评估"
        return result
    if not images or not sum(support):
        result["reason"] = "缺少图片或缺陷标注，无法评估缺陷检出能力"
        return result
    metrics = model.val(data=str(dataset / "data.yaml"), split=split,
                        imgsz=imgsz, batch=batch, device=device, plots=True, verbose=False,
                        project=str(dataset.parent / "evaluation"), name=split, exist_ok=True)
    box = metrics.box
    for position, class_index in enumerate(box.ap_class_index):
        index = int(class_index)
        per_class[names[index]].update(
            precision=round(float(box.p[position]), 4), recall=round(float(box.r[position]), 4),
            ap50=round(float(box.ap50[position]), 4), ap=round(float(box.ap[position]), 4),
            status="已评估")
    matrix = metrics.confusion_matrix.matrix
    for index, name in enumerate(names):
        per_class[name]["missed"] = int(matrix[-1, index])
        per_class[name]["false_positive"] = int(matrix[index, -1])
    result["normal_images"] = len(normal_paths)
    result["normal_false_positive_images"] = 0
    if normal_paths:
        predictions = model.predict(normal_paths, imgsz=imgsz, batch=batch, device=device,
                                    conf=.25, stream=True, verbose=False)
        result["normal_false_positive_images"] = sum(len(prediction.boxes) > 0 for prediction in predictions)
    result.update(map50=round(float(box.map50), 4), map=round(float(box.map), 4),
                  evaluation_status="valid" if all(support) else "partial")
    return result


def main() -> int:
    job_dir = Path(sys.argv[1]).resolve()
    config = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    dataset = Path(config["dataset"])
    template = config["template"]
    epochs = int(config.get("epochs", 40))
    imgsz = int(config.get("imgsz", 640))
    batch = int(config.get("batch", 16))
    device = str(config.get("device", "cpu"))
    default_name = "yolo11n.pt" if template == "det" else "yolo11n-cls.pt"
    model_name = config.get("model") or default_name
    if not Path(model_name).is_file():
        bundled = Path(__file__).resolve().parent / default_name
        if bundled.is_file():
            model_name = str(bundled)
    try:
        if template == "det":
            model = YOLO(model_name)
            model.train(data=str(dataset / "data.yaml"), epochs=epochs, imgsz=imgsz,
                        batch=batch, device=device, project=str(job_dir), name="fit",
                        exist_ok=True, workers=0, seed=0, plots=False, verbose=True)
            best = job_dir / "fit/weights/best.pt"
            model = YOLO(str(best))
            result = evaluate_detector(model, dataset, "val", imgsz, batch, device)
            result["test"] = evaluate_detector(model, dataset, "test", imgsz, batch, device)
            if config.get("baseline"):
                try:
                    baseline = YOLO(config["baseline"])
                    if baseline.names != model.names:
                        result["comparison_reason"] = "线上模型类别顺序不同，无法在该测试集直接对比"
                    else:
                        result["baseline_version"] = config.get("baseline_version")
                        result["baseline_test"] = evaluate_detector(baseline, dataset, "test", imgsz, batch, device)
                except Exception as exc:
                    result["comparison_reason"] = "线上模型对比失败：" + str(exc)[:200]

        else:
            model = YOLO(model_name)
            model.train(data=str(dataset), epochs=epochs, imgsz=imgsz,
                        batch=batch, device=device, project=str(job_dir), name="fit",
                        exist_ok=True, workers=0, seed=0, plots=False, verbose=True)
            best = job_dir / "fit/weights/best.pt"
            model = YOLO(str(best))
            metrics = model.val(data=str(dataset), split="val", imgsz=imgsz,
                                batch=batch, device=device, plots=False, verbose=False)
            matrix = metrics.confusion_matrix.matrix.tolist()
            names = model.names
            per_class = {}
            for index, name in names.items():
                support = sum(row[index] for row in matrix)
                per_class[name] = dict(
                    recall=round(matrix[index][index] / support, 4) if support else 0.0)
            result = dict(top1=round(float(metrics.top1), 4),
                          top5=round(float(metrics.top5), 4),
                          per_class=per_class, names=list(names.values()),
                          confusion=matrix)
        shutil.copy2(best, job_dir / "best.pt")
        (job_dir / "metrics.json").write_text(
            json.dumps(dict(ok=True, metrics=result, threshold=0.25 if template == "det" else 0.5), ensure_ascii=False, indent=1),
            encoding="utf-8")
        print("metrics", json.dumps(result, ensure_ascii=False))
        return 0
    except Exception:
        (job_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
