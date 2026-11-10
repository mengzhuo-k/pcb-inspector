"""Image-level SMD defect classifier trained on Zenodo SMD DefDB.

The published dataset (record 20001570) labels whole-board photos with one of
seven classes and provides no bounding boxes. It therefore backs an *image-level*
recognition stage that runs next to the existing single-class detector: the
detector localises missing-joint candidates, while this classifier names the
dominant assembly defect on the board.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

# Dataset class -> operator-facing label. poor_solder covers both a joint that
# never wet (虚焊) and a cold joint (冷焊); the dataset does not separate them.
LABELS = {
    "damaged_component": "焊盘/元件损伤",
    "excess_solder": "多锡",
    "good_solder": "未见明显缺陷",
    "misaligned_component": "元件偏移",
    "missing_component": "缺件",
    "poor_solder": "虚焊/冷焊",
    "solder_bridge": "桥接",
}
DEFECT_CLASSES = {
    "damaged_component": "焊盘/元件损伤",
    "excess_solder": "多锡",
    "misaligned_component": "元件偏移",
    "missing_component": "缺件",
    "poor_solder": "虚焊/冷焊",
    "solder_bridge": "桥接",
}
GOOD_CLASS = "good_solder"
# Ordered by the dataset folder order used during training.
CLASSES = ["damaged_component", "excess_solder", "good_solder",
           "misaligned_component", "missing_component", "poor_solder",
           "solder_bridge"]
DEFAULT_THRESHOLD = 0.5


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DefectClassifier:
    """Load one completed classifier and expose thread-safe whole-board verdicts."""

    def __init__(self, weights_path, card_path=None, factory=YOLO,
                 imgsz=None, threshold=None):
        self.weights_path = Path(weights_path)
        self.card_path = Path(card_path) if card_path else None
        self.factory = factory
        self.lock = threading.RLock()
        self.active = None
        self.last_error = None
        self.imgsz = imgsz
        self.threshold = threshold
        try:
            self.active = self._load()
        except Exception as exc:  # noqa: BLE001 - keep the detector usable
            self.last_error = "缺陷分类模型加载失败：" + str(exc)[:300]

    def _card(self):
        if not self.card_path or not self.card_path.is_file():
            return {}
        try:
            return json.loads(self.card_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _load(self):
        if not self.weights_path.is_file():
            raise FileNotFoundError(self.weights_path.name)
        card = self._card()
        expected = card.get("classes") or CLASSES
        digest = _digest(self.weights_path)
        model = self.factory(str(self.weights_path))
        names = model.names
        loaded = list(names.values()) if isinstance(names, dict) else list(names)
        if getattr(model, "task", None) != "classify":
            raise ValueError("模型必须为图像分类模型")
        if set(loaded) != set(expected):
            raise ValueError("分类类别与模型卡不一致")
        imgsz = int(self.imgsz or card.get("imgsz") or 224)
        threshold = float(self.threshold if self.threshold is not None
                          else card.get("threshold", DEFAULT_THRESHOLD))
        if not 0 < threshold < 1:
            raise ValueError("分类阈值无效")
        probe = model.predict(Image.new("RGB", (imgsz, imgsz), "white"),
                              imgsz=imgsz, verbose=False)[0]
        if probe.probs is None or not len(probe.probs.data):
            raise ValueError("模型未返回分类概率")
        labels = card.get("labels") or LABELS
        return dict(model=model, version=card.get("version") or "smd-defdb",
                    label="SMD DefDB 七类缺陷识别 " + (card.get("version") or ""),
                    classes=loaded, labels=labels, imgsz=imgsz,
                    threshold=threshold, sha256=digest,
                    card=card,
                    loaded_at=datetime.now().astimezone().isoformat(timespec="seconds"))

    def available(self):
        return self.active is not None

    def status(self):
        with self.lock:
            if not self.active:
                return dict(ok=False, last_error=self.last_error)
            card = self.active["card"]
            metrics = card.get("metrics") or {}
            return dict(ok=True, version=self.active["version"],
                        label=self.active["label"], classes=self.active["classes"],
                        labels=self.active["labels"], threshold=self.active["threshold"],
                        imgsz=self.active["imgsz"], sha256=self.active["sha256"],
                        loaded_at=self.active["loaded_at"], metrics=metrics,
                        dataset=card.get("dataset"), last_error=self.last_error)

    def predict(self, image, threshold=None):
        """Return a whole-board verdict, or None when the classifier is unavailable.

        ``defect_score`` is 1 - P(good_solder); it is the calibrated quantity for
        the accept/reject gate. ``score`` is the top-1 class probability used for
        the defect-type label.
        """
        with self.lock:
            if not self.active:
                return None
            active = self.active
            result = active["model"].predict(image, imgsz=active["imgsz"],
                                             verbose=False)[0]
            probs = result.probs
            if probs is None:
                raise RuntimeError("分类模型未返回概率")
            values = probs.data.cpu().tolist()
            names = result.names
            ranking = sorted(
                (dict(key=names[i], label=active["labels"].get(names[i], names[i]),
                      score=round(float(value), 4))
                 for i, value in enumerate(values)),
                key=lambda item: item["score"], reverse=True)
            good = next((item["score"] for item in ranking if item["key"] == GOOD_CLASS), 0.0)
            top = ranking[0]
            gate = float(threshold if threshold is not None else active["threshold"])
            defect_score = round(1 - good, 4)
            is_defect = top["key"] in DEFECT_CLASSES and defect_score >= gate
            return dict(key=top["key"], label=top["label"], score=top["score"],
                        defect_score=defect_score, is_defect=is_defect,
                        threshold=gate, top=ranking[:3],
                        model_version=active["version"], model_sha256=active["sha256"])


def build_classifier(root=None):
    root = Path(root or ROOT)
    weights = Path(os.environ.get("PCB_DEFECT_WEIGHTS",
                                 root / "models/smd_defect_cls_v1.pt"))
    card = Path(os.environ.get("PCB_DEFECT_CARD",
                               root / "models/smd_defect_cls_v1.json"))
    threshold = os.environ.get("PCB_DEFECT_THRESHOLD")
    return DefectClassifier(weights, card,
                            threshold=float(threshold) if threshold else None)
