"""Step 1: four-point perspective transform into one standard view.

Two ways to get the four corners of the board in a photo:
  * manual  - the operator clicks TL, TR, BR, BL (used for the golden photo and as the UI fallback);
  * auto    - SIFT matches against the golden view give a homography; the canonical rectangle is
              projected back into the photo, which yields the four corners without clicking.
Both end in the same cv2.getPerspectiveTransform + cv2.warpPerspective call.

Run:  python -m demo.register
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from . import config as C

MATCH_SCALE = 1.5          # the golden view is matched at 1.5x canonical size for more keypoints
QUERY_MAX_SIDE = 2000
MIN_INLIERS = 40           # below this the photo is rejected ("retake")
MIN_INLIER_RATIO = 0.25


def imread(path: Path | str) -> np.ndarray:
    """cv2.imread that survives non-ASCII file names."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def imwrite(path: Path | str, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"cannot encode image: {path}")
    buf.tofile(str(path))


def order_corners(points) -> np.ndarray:
    """Order four arbitrary clicks as TL, TR, BR, BL (by angle around the centroid).

    This fixes the click order only; which physical board corner is 'top' still depends on the
    photo orientation, so the caller rotates the result if needed (see `four_point_warp(rotate=)`).
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    pts = pts[np.argsort(angles)]            # clockwise in image coordinates (y down)
    start = int(np.argmin(pts.sum(axis=1)))  # top-left has the smallest x + y
    return np.roll(pts, -start, axis=0)


def four_point_warp(image: np.ndarray, corners, size=(C.CANON_W, C.CANON_H), rotate: int = 0) -> np.ndarray:
    """Warp the quadrilateral `corners` (TL, TR, BR, BL) onto the canonical rectangle.

    `rotate` (0-3) cycles the corner order, i.e. turns the result by multiples of 90 degrees.
    """
    w, h = size
    src = np.roll(np.asarray(corners, dtype=np.float32).reshape(4, 2), -rotate, axis=0)
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_AREA if max(image.shape[:2]) > 2 * h else cv2.INTER_LINEAR)


def best_rotation(image: np.ndarray, corners, reference: np.ndarray) -> int:
    """Which of the four 90-degree turns of a manual warp looks most like the golden view?"""
    def thumb(bgr):
        small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (96, 184), interpolation=cv2.INTER_AREA).astype(np.float32)
        return (small - small.mean()) / (small.std() + 1e-6)
    target = thumb(reference)
    scores = [float((thumb(four_point_warp(image, corners, rotate=k)) * target).mean()) for k in range(4)]
    return int(np.argmax(scores))


@dataclass
class Registration:
    ok: bool
    corners: list | None      # TL, TR, BR, BL in the photo, full resolution
    inliers: int
    matches: int
    inlier_ratio: float
    reason: str = ""

    def to_json(self) -> dict:
        return asdict(self)


class Registrar:
    """Finds the four board corners of a new photo by matching it to the golden view."""

    def __init__(self, reference: np.ndarray):
        self.reference = reference
        big = cv2.resize(reference, None, fx=MATCH_SCALE, fy=MATCH_SCALE, interpolation=cv2.INTER_CUBIC)
        self.sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.02)
        self.ref_kp, self.ref_desc = self.sift.detectAndCompute(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY), None)
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))

    def locate(self, image: np.ndarray) -> Registration:
        scale = min(1.0, QUERY_MAX_SIDE / max(image.shape[:2]))
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
        kp, desc = self.sift.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), None)
        if desc is None or len(kp) < 50:
            return Registration(False, None, 0, 0, 0.0, "too few keypoints (blurred or empty photo?)")
        pairs = self.matcher.knnMatch(desc, self.ref_desc, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < MIN_INLIERS:
            return Registration(False, None, 0, len(good), 0.0, "too few matches with the golden board")
        src = np.float32([kp[m.queryIdx].pt for m in good]) / scale
        dst = np.float32([self.ref_kp[m.trainIdx].pt for m in good]) / MATCH_SCALE
        H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 4.0, maxIters=20000, confidence=0.999)
        if H is None:
            return Registration(False, None, 0, len(good), 0.0, "no consistent homography")
        inliers = int(mask.sum())
        ratio = inliers / len(good)
        rect = np.float32([[0, 0], [C.CANON_W - 1, 0], [C.CANON_W - 1, C.CANON_H - 1], [0, C.CANON_H - 1]])
        corners = cv2.perspectiveTransform(rect[None], np.linalg.inv(H))[0]
        reason = ""
        if inliers < MIN_INLIERS or ratio < MIN_INLIER_RATIO:
            reason = f"alignment confidence too low ({inliers} inliers, ratio {ratio:.2f})"
        elif not _plausible_quad(corners, image.shape):
            reason = "implausible board outline"
        return Registration(not reason, corners.round(1).tolist(), inliers, len(good), round(ratio, 3), reason)

    def warp(self, image: np.ndarray) -> tuple[np.ndarray | None, Registration]:
        reg = self.locate(image)
        return (four_point_warp(image, reg.corners) if reg.ok else None), reg


def _plausible_quad(corners: np.ndarray, shape) -> bool:
    quad = corners.astype(np.float32)
    if not cv2.isContourConvex(quad):
        return False
    area = cv2.contourArea(quad)
    h, w = shape[:2]
    if not 0.02 * w * h < area < 1.5 * w * h:
        return False
    margin = 0.35 * max(w, h)  # corners may fall a little outside a tightly framed photo
    return bool((quad[:, 0] > -margin).all() and (quad[:, 0] < w + margin).all()
                and (quad[:, 1] > -margin).all() and (quad[:, 1] < h + margin).all())


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in C.IMAGE_SUFFIXES) if folder.is_dir() else []


def build_reference() -> np.ndarray:
    matches = [p for p in list_images(C.RAW_NORMAL) if C.REFERENCE_PHOTO_KEY in p.name]
    if not matches:
        raise SystemExit(f"golden photo *{C.REFERENCE_PHOTO_KEY}* not found in {C.RAW_NORMAL}")
    reference = four_point_warp(imread(matches[0]), C.REFERENCE_CORNERS)
    imwrite(C.REFERENCE, reference)
    return reference


def load_registrar() -> Registrar:
    if not C.REFERENCE.exists():
        build_reference()
    return Registrar(imread(C.REFERENCE))


def main() -> None:
    registrar = Registrar(build_reference())
    report = {}
    for kind, folder in (("normal", C.RAW_NORMAL), ("abnormal", C.RAW_ABNORMAL)):
        for index, path in enumerate(list_images(folder)):
            image = imread(path)
            warped, reg = registrar.warp(image)
            name = f"{kind}_{index:02d}.png"
            report[name] = {"source": path.name, **reg.to_json()}
            status = "ok" if reg.ok else f"REJECTED: {reg.reason}"
            print(f"{name}  inliers={reg.inliers:4d}/{reg.matches:4d}  {status}")
            if warped is not None:
                imwrite(C.WARPED / kind / name, warped)
    C.REGISTRATION_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(r["ok"] for r in report.values())
    print(f"registered {ok}/{len(report)} photos -> {C.WARPED}")


if __name__ == "__main__":
    main()
