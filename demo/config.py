"""Shared paths and constants for the demo pipeline. Everything it writes stays under data/demo (gitignored)."""
from __future__ import annotations

import os
from pathlib import Path

# Keep every third-party library offline-quiet: no telemetry, no update checks.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("YOLO_OFFLINE", "True")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parent.parent
RAW_NORMAL = ROOT / "data" / "raw_upload2" / "dataset" / "normal"
RAW_ABNORMAL = ROOT / "data" / "raw_upload2" / "dataset" / "abnormal"

OUT = Path(os.environ.get("PCB_DEMO_DIR", ROOT / "data" / "demo"))
WARPED = OUT / "warped"            # registered images: warped/normal, warped/abnormal
REFERENCE = OUT / "reference.png"  # canonical golden view
REGISTRATION_REPORT = OUT / "registration.json"
PATCHCORE = OUT / "patchcore.pt"
PATCHCORE_REPORT = OUT / "patchcore_report.json"
YOLO_DATA = OUT / "yolo_dataset"
YOLO_RUNS = OUT / "yolo_runs"
YOLO_WEIGHTS = OUT / "yolo_synth.pt"
YOLO_REPORT = OUT / "yolo_report.json"
REVIEW_DIR = OUT / "reviews"
REVIEW_LOG = REVIEW_DIR / "reviews.csv"

# Canonical "standard view": board upright, terminal block at the bottom.
CANON_W, CANON_H = 768, 1472

# Golden photo and its four board corners (TL, TR, BR, BL) in full-resolution pixels.
# The bottom edge is taken at the feet of the terminal block so the block is inside the view.
REFERENCE_PHOTO_KEY = "150833_9"
REFERENCE_CORNERS = [(699, 235), (2495, 245), (2452, 3650), (695, 3650)]

# Synthetic defect classes (index = YOLO class id).
CLASSES = ["missing_solder", "solder_bridge", "missing_component", "component_offset", "pad_damage"]
CLASSES_ZH = {
    "missing_solder": "漏焊/空孔",
    "solder_bridge": "桥接",
    "missing_component": "缺件",
    "component_offset": "元件偏移",
    "pad_damage": "焊盘损伤",
}
REVIEW_CHOICES = [f"{CLASSES_ZH[c]} ({c})" for c in CLASSES] + ["虚焊/冷焊 (cold_joint)", "其他 (other)"]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
