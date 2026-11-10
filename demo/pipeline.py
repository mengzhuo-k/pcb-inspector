"""End-to-end inspection of one photo: register -> PatchCore heatmap -> YOLO categories -> verdict.

Decision logic (deliberately conservative, the inspector always signs):
  RETAKE  registration confidence too low - no result is forced
  PASS    anomaly score below the review level
  REVIEW  between review level and threshold
  NG      above the threshold (machine candidate; a human confirms)
YOLO was trained on synthetic defects only, so its boxes are *category hints*: a box counts as a
finding only when the anomaly map agrees at that place; the rest are listed as unconfirmed.
Where the anomaly map already flags a region, weaker YOLO boxes (conf >= 0.15) are accepted to name
it - the two models are independent, so agreement is worth more than either alone.
"""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config as C
from .register import Registrar, Registration, best_rotation, four_point_warp, imread, load_registrar, order_corners

REVIEW_RATIO = 0.8        # review band starts at 80 % of the anomaly threshold
YOLO_CONF = 0.35          # YOLO box shown as an unconfirmed hint (dashed)
YOLO_STRONG = 0.60        # YOLO box alone sends the board to review (the heatmap is weak on small bridges / pad damage)
YOLO_CONF_GATED = 0.15    # confidence needed where the anomaly map already says "abnormal here"
MAX_UNNAMED = 6           # at most this many unnamed review-level regions are listed (strongest first)
TILE = 768
TILE_OFFSETS = (0, (C.CANON_H - TILE) // 2, C.CANON_H - TILE)

SUGGESTIONS = {  # rule-based hints, shown as such in the UI
    "missing_solder": "检查波峰焊/手工焊是否漏工序，确认插件是否到位、助焊剂与预热是否正常。",
    "solder_bridge": "检查锡量与烙铁温度、引脚间距处钢网开口，必要时拖焊/吸锡返修。",
    "missing_component": "核对BOM与贴片/插件程序，检查供料器缺料与来料少件。",
    "component_offset": "检查贴装坐标与回流前移位、插件未压到位，必要时返修校正。",
    "pad_damage": "检查返修温度/次数与机械刮擦，评估焊盘是否可修复或报废。",
    "anomaly": "与标准板不一致，原因未知：请质检员确认并标注缺陷类型（会进入学习样本）。",
}
COLOURS = {"missing_solder": (255, 80, 60), "solder_bridge": (255, 170, 0), "missing_component": (200, 60, 255),
           "component_offset": (0, 190, 255), "pad_damage": (255, 60, 160), "anomaly": (255, 255, 255)}


@lru_cache(maxsize=1)
def _font(size: int = 24):
    for pattern in ("NotoSansCJK", "NotoSerifCJK", "wqy", "msyh", "simhei", "PingFang"):
        for root in ("/usr/share/fonts", "/System/Library/Fonts", "C:/Windows/Fonts"):
            hits = sorted(Path(root).rglob(f"*{pattern}*")) if Path(root).is_dir() else []
            if hits:
                try:
                    return ImageFont.truetype(str(hits[0]), size)
                except OSError:
                    continue
    return None


def label_text(cls: str) -> str:
    if cls == "anomaly":
        return "异常" if _font() else "anomaly"
    return C.CLASSES_ZH[cls] if _font() else cls


def draw_findings(bgr: np.ndarray, findings: list[dict], hints: list[dict] = ()) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = _font() or ImageFont.load_default()
    for hint in hints:
        x1, y1, x2, y2 = hint["box"]
        colour = COLOURS[hint["cls"]]
        for k in range(x1, x2, 10):  # dashed = unconfirmed category hint
            draw.line([(k, y1), (min(k + 5, x2), y1)], fill=colour, width=2)
            draw.line([(k, y2), (min(k + 5, x2), y2)], fill=colour, width=2)
        for k in range(y1, y2, 10):
            draw.line([(x1, k), (x1, min(k + 5, y2))], fill=colour, width=2)
            draw.line([(x2, k), (x2, min(k + 5, y2))], fill=colour, width=2)
    for number, finding in enumerate(findings, 1):
        x1, y1, x2, y2 = finding["box"]
        colour = COLOURS[finding["cls"]]
        draw.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], outline=colour, width=3)
        # small boxes (rows of holes) get the number only, otherwise the tags hide each other
        text = f"{number} {label_text(finding['cls'])}" if min(x2 - x1, y2 - y1) >= 40 else str(number)
        box = draw.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0] + 8, box[3] - box[1] + 10
        ty = y1 - th if y1 - th > 0 else y2
        draw.rectangle([x1, ty, x1 + tw, ty + th], fill=colour)
        draw.text((x1 + 4, ty + 1), text, fill=(0, 0, 0), font=font)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def _overlap_of(small, big) -> float:
    ix = max(0, min(small[2], big[2]) - max(small[0], big[0]))
    iy = max(0, min(small[3], big[3]) - max(small[1], big[1]))
    return ix * iy / ((small[2] - small[0]) * (small[3] - small[1]) + 1e-9)


def region_name(box) -> str:
    """Coarse 3x4 grid name in the standard view, e.g. 'B3' (column A-C, row 1-4 from the top)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return "ABC"[min(2, int(cx / C.CANON_W * 3))] + str(min(4, int(cy / C.CANON_H * 4) + 1))


class Inspector:
    """Loads whatever models exist; each stage degrades gracefully if its model is missing."""

    def __init__(self):
        self.registrar: Registrar = load_registrar()
        self.patchcore = None
        self.yolo = None
        if C.PATCHCORE.exists():
            from .patchcore import PatchCore
            self.patchcore = PatchCore.load()
        if C.YOLO_WEIGHTS.exists():
            from ultralytics import YOLO, settings
            settings.update({"sync": False})
            self.yolo = YOLO(str(C.YOLO_WEIGHTS))

    # ---------- stages ----------
    def register(self, bgr: np.ndarray, manual_corners=None, rotate: int | None = None):
        """`rotate` None = pick the orientation that matches the golden view."""
        if manual_corners is not None:
            corners = order_corners(manual_corners)
            if rotate is None:
                rotate = best_rotation(bgr, corners, self.registrar.reference)
            warped = four_point_warp(bgr, corners, rotate=rotate)
            return warped, Registration(True, corners.tolist(), 0, 0, 0.0, "manual four-point")
        return self.registrar.warp(bgr)

    def detect_categories(self, warped: np.ndarray, conf: float = YOLO_CONF) -> list[dict]:
        if self.yolo is None:
            return []
        tiles = [np.ascontiguousarray(warped[y:y + TILE]) for y in TILE_OFFSETS]
        boxes = []
        for y0, result in zip(TILE_OFFSETS, self.yolo.predict(tiles, imgsz=640, conf=conf, device="cpu", verbose=False)):
            for xyxy, score, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
                x1, y1, x2, y2 = xyxy
                boxes.append({"box": [int(x1), int(y1 + y0), int(x2), int(y2 + y0)], "conf": float(score), "cls": C.CLASSES[int(cls)]})
        kept = []
        for box in sorted(boxes, key=lambda b: -b["conf"]):  # NMS across overlapping tiles
            if all(_iou(box["box"], k["box"]) < 0.5 for k in kept):
                kept.append(box)
        return kept

    # ---------- full pipeline ----------
    def inspect(self, bgr: np.ndarray, manual_corners=None, rotate: int | None = None, sensitivity: float = 1.0) -> dict:
        """`sensitivity` > 1 lowers the anomaly threshold (more boards go to review)."""
        started = time.time()
        warped, reg = self.register(bgr, manual_corners, rotate)
        out = {"registration": reg.to_json(), "verdict": "RETAKE", "findings": [], "hints": [], "warped": warped,
               "annotated": None, "heatmap": None, "score": None, "threshold": None, "timing": {}}
        out["timing"]["register"] = round(time.time() - started, 2)
        if warped is None:
            return out
        return self.inspect_warped(warped, sensitivity, out)

    def inspect_warped(self, warped: np.ndarray, sensitivity: float = 1.0, out: dict | None = None) -> dict:
        out = out or {"registration": None, "findings": [], "hints": [], "warped": warped, "timing": {}}
        findings, hints = [], []
        t = time.time()
        detections = self.detect_categories(warped, YOLO_CONF_GATED if self.patchcore is not None else YOLO_CONF)
        out["timing"]["yolo"] = round(time.time() - t, 2)
        if self.patchcore is not None:
            from .patchcore import boxes_from_map, overlay
            t = time.time()
            result = self.patchcore.score(warped)
            out["timing"]["patchcore"] = round(time.time() - t, 2)
            threshold = result["threshold"] / sensitivity
            review_level = threshold * REVIEW_RATIO
            norm = result["map"]
            named_boxes = []
            for d in detections:
                x1, y1, x2, y2 = d["box"]
                local = float(norm[y1:y2, x1:x2].max()) if x2 > x1 and y2 > y1 else 0.0
                if local > review_level:  # both models agree -> a named finding
                    findings.append({"box": d["box"], "cls": d["cls"], "conf": d["conf"], "anomaly": local,
                                     "level": "NG" if local > threshold else "REVIEW", "source": "PatchCore+YOLO"})
                    named_boxes.append(d["box"])
                elif d["conf"] >= YOLO_STRONG:  # confident YOLO alone -> a human should look
                    findings.append({"box": d["box"], "cls": d["cls"], "conf": d["conf"], "anomaly": local, "level": "REVIEW", "source": "YOLO"})
                    named_boxes.append(d["box"])
                elif d["conf"] >= YOLO_CONF:  # weak YOLO alone -> unconfirmed hint (dashed)
                    hints.append({**d, "anomaly": local})
            unnamed = []
            for region in boxes_from_map(norm, review_level):
                covered = sum(_overlap_of(region["box"], b) for b in named_boxes)
                if covered < 0.5:  # not already explained by named findings
                    unnamed.append({"box": region["box"], "cls": "anomaly", "conf": None, "anomaly": region["score"],
                                    "level": "NG" if region["score"] > threshold else "REVIEW", "source": "PatchCore"})
            strong = [u for u in unnamed if u["level"] == "NG"]
            findings += strong + [u for u in unnamed if u["level"] != "NG"][:max(0, MAX_UNNAMED - len(strong))]
            score = result["score"]
            verdict = "NG" if score > threshold else "REVIEW" if score > review_level or findings else "PASS"
            out.update(score=score, threshold=threshold, heatmap=overlay(warped, norm, threshold))
        else:  # no anomaly model yet: YOLO alone, everything goes to review
            findings = [{"box": d["box"], "cls": d["cls"], "conf": d["conf"], "anomaly": None, "level": "REVIEW", "source": "YOLO"} for d in detections]
            verdict = "REVIEW" if findings else "PASS"
            out.update(score=None, threshold=None, heatmap=None)
        findings.sort(key=lambda f: (f["level"] != "NG", -(f["anomaly"] or 0)))
        for f in findings:
            f["region"] = region_name(f["box"])
            f["suggestion"] = SUGGESTIONS[f["cls"]]
        out.update(verdict=verdict, findings=findings, hints=hints, annotated=draw_findings(warped, findings, hints))
        out["timing"]["total"] = round(sum(out["timing"].values()), 2)
        return out


def main() -> None:
    """CLI smoke run over all raw photos."""
    from .register import list_images
    inspector = Inspector()
    for folder in (C.RAW_NORMAL, C.RAW_ABNORMAL):
        for path in list_images(folder):
            r = inspector.inspect(imread(path))
            score = "-" if r["score"] is None else f"{r['score']:.2f}/{r['threshold']:.2f}"
            kinds = ", ".join(f"{f['cls']}@{f['region']}" for f in r["findings"][:6])
            print(f"{folder.name:8s} {path.name[-16:]:16s} {r['verdict']:6s} score={score} findings={len(r['findings'])} hints={len(r['hints'])} t={r['timing']}  {kinds}")


if __name__ == "__main__":
    main()
