"""Step 3a: synthesise defects with OpenCV on registered normal boards -> YOLO dataset.

Five classes that can be drawn credibly from a top-down photo.  Cold / insufficient joints are NOT
synthesised: they are mostly invisible in a top-down photo and a drawn version would be fiction.

Run:  python -m demo.synth
"""
from __future__ import annotations

import json
import shutil

import cv2
import numpy as np

from . import config as C
from .register import imread, imwrite, list_images

TILE = 768
VAL_BOARDS = 2            # the last N normal boards are used only for validation tiles
TRAIN_PER_BOARD, VAL_PER_BOARD = 44, 40


def green_mask(bgr: np.ndarray) -> np.ndarray:
    """Bare solder-mask area of the board (1 = free green surface)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 60, 35), (95, 255, 215))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) // 255


def component_boxes(free: np.ndarray, lo: int = 20, hi: int = 170) -> list[tuple[int, int, int, int]]:
    """Blobs that are not bare board and have component-like size."""
    count, _, stats, _ = cv2.connectedComponentsWithStats((1 - free).astype(np.uint8), connectivity=8)
    boxes = []
    for x, y, w, h, area in stats[1:count]:
        if lo <= w <= hi and lo <= h <= hi and area > 0.45 * w * h and x > 4 and y > 4 and x + w < free.shape[1] - 4 and y + h < free.shape[0] - 4:
            boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def _free_spot(free: np.ndarray, rng, w: int, h: int, ratio: float = 0.85):
    integral = cv2.integral(free)
    for _ in range(60):
        x = int(rng.integers(6, free.shape[1] - w - 6))
        y = int(rng.integers(6, free.shape[0] - h - 6))
        inside = integral[y + h, x + w] - integral[y, x + w] - integral[y + h, x] + integral[y, x]
        if inside >= ratio * w * h:
            return x, y
    return None


def _board_colour(img, x, y, w, h):
    patch = img[max(0, y - 6):y + h + 6, max(0, x - 6):x + w + 6].reshape(-1, 3)
    greenness = patch[:, 1].astype(int) - patch[:, [0, 2]].max(1)
    return np.median(patch[greenness >= np.percentile(greenness, 60)], axis=0)


def _soften(img, x1, y1, x2, y2, sigma=0.9):
    x1, y1 = max(0, x1), max(0, y1)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (0, 0), sigma)


def add_missing_solder(img, free, rng):
    """A row/cluster of empty plated through-holes: bright annular ring, dark hole, no solder cone.

    Half of the rows are dropped on bare board, half anywhere (so they also replace existing joints,
    which is what a skipped soldering step looks like).
    """
    n = int(rng.integers(1, 9))
    r = int(rng.integers(7, 12))
    pitch = int(r * rng.uniform(2.3, 3.2))
    vertical = rng.random() < 0.5
    w, h = (2 * r + 6, pitch * (n - 1) + 2 * r + 6) if vertical else (pitch * (n - 1) + 2 * r + 6, 2 * r + 6)
    spot = _free_spot(free, rng, w, h, 0.7 if rng.random() < 0.5 else 0.3)
    if spot is None:
        return []
    x, y = spot
    ring = np.array([rng.uniform(70, 150), rng.uniform(150, 215), rng.uniform(165, 235)])  # tinned / gold pad, BGR
    board = _board_colour(img, x, y, w, h)
    hole_ratio = rng.uniform(0.42, 0.66)
    outline = rng.random() < 0.4
    labels = []
    for i in range(n):
        cx = x + r + 3 + (0 if vertical else i * pitch)
        cy = y + r + 3 + (i * pitch if vertical else 0)
        cv2.circle(img, (cx, cy), r + 2, board, -1, cv2.LINE_AA)  # clear whatever was there (e.g. a solder cone)
        cv2.circle(img, (cx, cy), r, ring * rng.uniform(0.8, 1.1), -1, cv2.LINE_AA)
        if outline:
            cv2.circle(img, (cx - 1, cy - 1), r, (235, 240, 240), 1, cv2.LINE_AA)
        hole = board * rng.uniform(0.15, 0.5) if rng.random() < 0.5 else rng.uniform(8, 45, 3)
        cv2.circle(img, (cx, cy), max(2, int(r * hole_ratio)), tuple(float(v) for v in hole), -1, cv2.LINE_AA)
        _soften(img, cx - r - 3, cy - r - 3, cx + r + 4, cy + r + 4, rng.uniform(0.7, 1.3))
        labels.append((0, cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2))
    return labels


def add_solder_bridge(img, free, rng, comps):
    """A shiny solder blob bridging neighbouring pins, placed on the edge of a component."""
    if not comps:
        return []
    x, y, w, h = comps[int(rng.integers(len(comps)))]
    side = int(rng.integers(4))
    cx = int(x + w * rng.uniform(0.2, 0.8)) if side < 2 else (x if side == 2 else x + w)
    cy = (y if side == 0 else y + h) if side < 2 else int(y + h * rng.uniform(0.2, 0.8))
    a, b = int(rng.integers(9, 20)), int(rng.integers(5, 10))
    angle = float(rng.uniform(0, 180))
    base = rng.uniform(150, 200)
    blob = np.zeros(img.shape[:2], np.uint8)
    cv2.ellipse(blob, (cx, cy), (a, b), angle, 0, 360, 255, -1, cv2.LINE_AA)
    cv2.ellipse(blob, (cx + int(rng.integers(-5, 6)), cy + int(rng.integers(-5, 6))), (b, b), 0, 0, 360, 255, -1, cv2.LINE_AA)
    shade = np.full(img.shape, base, np.float32) + rng.normal(0, 10, img.shape[:2])[..., None]
    cv2.ellipse(shade, (cx - 2, cy - 2), (max(2, a // 2), max(1, b // 3)), angle, 0, 360, (250, 250, 250), -1, cv2.LINE_AA)
    alpha = cv2.GaussianBlur(blob, (0, 0), 1.2)[..., None] / 255.0
    img[:] = (img * (1 - alpha) + np.clip(shade, 0, 255) * alpha).astype(np.uint8)
    ys, xs = np.nonzero(blob)
    return [(1, xs.min() - 2, ys.min() - 2, xs.max() + 3, ys.max() + 3)]


def _erase(img, x, y, w, h):
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.rectangle(mask, (x - 2, y - 2), (x + w + 2, y + h + 2), 255, -1)
    img[:] = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)


def add_missing_component(img, free, rng, comps):
    """Remove a part: inpaint the body and leave two bare tinned pads."""
    if not comps:
        return []
    x, y, w, h = comps[int(rng.integers(len(comps)))]
    colour = _board_colour(img, x, y, w, h)
    _erase(img, x, y, w, h)
    img[y:y + h, x:x + w] = (0.5 * img[y:y + h, x:x + w] + 0.5 * colour).astype(np.uint8)
    pad = rng.uniform(170, 225)
    if w >= h:
        pw = max(4, w // 4)
        pads = [(x + 1, y + 2, x + pw, y + h - 2), (x + w - pw, y + 2, x + w - 1, y + h - 2)]
    else:
        ph = max(4, h // 4)
        pads = [(x + 2, y + 1, x + w - 2, y + ph), (x + 2, y + h - ph, x + w - 2, y + h - 1)]
    for x1, y1, x2, y2 in pads:
        cv2.rectangle(img, (x1, y1), (x2, y2), (pad * 0.95, pad, pad), -1)
    _soften(img, x - 3, y - 3, x + w + 4, y + h + 4, 1.1)
    return [(2, x - 2, y - 2, x + w + 2, y + h + 2)]


def add_component_offset(img, free, rng, comps):
    """Cut a part out and put it back shifted and rotated."""
    if not comps:
        return []
    x, y, w, h = comps[int(rng.integers(len(comps)))]
    part = img[y:y + h, x:x + w].copy()
    _erase(img, x, y, w, h)
    angle = float(rng.choice([-1, 1]) * rng.uniform(8, 28))
    shift = rng.choice([-1, 1], 2) * rng.uniform(0.15, 0.4, 2) * (w, h)
    size = int(np.hypot(w, h)) + 4
    canvas = np.zeros((size, size, 3), np.uint8)
    alpha = np.zeros((size, size), np.uint8)
    ox, oy = (size - w) // 2, (size - h) // 2
    canvas[oy:oy + h, ox:ox + w] = part
    alpha[oy:oy + h, ox:ox + w] = 255
    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    canvas = cv2.warpAffine(canvas, M, (size, size))
    alpha = cv2.GaussianBlur(cv2.warpAffine(alpha, M, (size, size)), (0, 0), 0.8)
    px, py = int(x + w / 2 + shift[0] - size / 2), int(y + h / 2 + shift[1] - size / 2)
    if px < 0 or py < 0 or px + size > img.shape[1] or py + size > img.shape[0]:
        img[y:y + h, x:x + w] = part
        return []
    a = alpha[..., None] / 255.0
    img[py:py + size, px:px + size] = (img[py:py + size, px:px + size] * (1 - a) + canvas * a).astype(np.uint8)
    ys, xs = np.nonzero(alpha > 100)
    x1, y1 = min(x, px + xs.min()), min(y, py + ys.min())
    x2, y2 = max(x + w, px + xs.max()), max(y + h, py + ys.max())
    return [(3, x1 - 2, y1 - 2, x2 + 2, y2 + 2)]


def add_pad_damage(img, free, rng):
    """Lifted / scratched solder mask exposing copper: irregular copper-coloured patch with a dark rim."""
    w, h = int(rng.integers(18, 56)), int(rng.integers(12, 40))
    spot = _free_spot(free, rng, w, h, 0.6)
    if spot is None:
        return []
    x, y = spot
    k = int(rng.integers(6, 11))
    angles = np.sort(rng.uniform(0, 2 * np.pi, k))
    radii = rng.uniform(0.55, 1.0, k)
    pts = np.stack([x + w / 2 + np.cos(angles) * radii * w / 2, y + h / 2 + np.sin(angles) * radii * h / 2], 1).astype(np.int32)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)
    copper = np.array([rng.uniform(40, 90), rng.uniform(95, 150), rng.uniform(170, 225)])
    texture = np.clip(copper + rng.normal(0, 14, img.shape[:2])[..., None], 0, 255)
    alpha = cv2.GaussianBlur(mask, (0, 0), 1.0)[..., None] / 255.0 * rng.uniform(0.8, 1.0)
    img[:] = (img * (1 - alpha) + texture * alpha).astype(np.uint8)
    cv2.polylines(img, [pts], True, (25, 40, 35), 1, cv2.LINE_AA)
    _soften(img, x - 3, y - 3, x + w + 4, y + h + 4, 0.8)
    return [(4, int(pts[:, 0].min()) - 2, int(pts[:, 1].min()) - 2, int(pts[:, 0].max()) + 3, int(pts[:, 1].max()) + 3)]


def random_tile(board: np.ndarray, rng) -> np.ndarray:
    """A 768x768 crop with small geometric + photometric jitter (imitates registration/lighting error)."""
    h, w = board.shape[:2]
    y0 = int(rng.integers(0, h - TILE + 1))
    src = np.float32([[0, y0], [w, y0], [w, y0 + TILE], [0, y0 + TILE]]) + rng.uniform(-10, 10, (4, 2)).astype(np.float32)
    dst = np.float32([[0, 0], [TILE, 0], [TILE, TILE], [0, TILE]])
    tile = cv2.warpPerspective(board, cv2.getPerspectiveTransform(src, dst), (TILE, TILE), borderMode=cv2.BORDER_REFLECT)
    gain, bias = rng.uniform(0.8, 1.2), rng.uniform(-15, 15)
    tile = np.clip(tile.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        tile = cv2.GaussianBlur(tile, (0, 0), rng.uniform(0.5, 1.2))
    return tile


def synthesize(tile: np.ndarray, rng, n_defects: int | None = None):
    """Return (image with defects, [(cls, x1, y1, x2, y2), ...])."""
    img = tile.copy()
    free = green_mask(img)
    comps = component_boxes(free)
    labels = []
    for _ in range(int(rng.integers(2, 6)) if n_defects is None else n_defects):
        kind = int(rng.integers(len(C.CLASSES)))
        taken = [l[1:] for l in labels]
        # keep parts that are still untouched so two defects never overlap
        comps_free = [c for c in comps if not any(c[0] < t[2] + 8 and c[0] + c[2] > t[0] - 8 and c[1] < t[3] + 8 and c[1] + c[3] > t[1] - 8 for t in taken)]
        mask = free.copy()
        for t in taken:
            mask[max(0, t[1] - 8):t[3] + 8, max(0, t[0] - 8):t[2] + 8] = 0
        if kind == 0:
            new = add_missing_solder(img, mask, rng)
        elif kind == 1:
            new = add_solder_bridge(img, mask, rng, comps_free)
        elif kind == 2:
            new = add_missing_component(img, mask, rng, comps_free)
        elif kind == 3:
            new = add_component_offset(img, mask, rng, comps_free)
        else:
            new = add_pad_damage(img, mask, rng)
        labels += new
    h, w = img.shape[:2]
    clipped = [(c, max(0, x1), max(0, y1), min(w, x2), min(h, y2)) for c, x1, y1, x2, y2 in labels]
    return img, [l for l in clipped if l[3] - l[1] > 4 and l[4] - l[2] > 4]


def yolo_lines(labels, size: int = TILE) -> str:
    return "".join(f"{c} {(x1 + x2) / 2 / size:.6f} {(y1 + y2) / 2 / size:.6f} {(x2 - x1) / size:.6f} {(y2 - y1) / size:.6f}\n"
                   for c, x1, y1, x2, y2 in labels)


def main(seed: int = 0) -> None:
    boards = list_images(C.WARPED / "normal")
    if len(boards) <= VAL_BOARDS:
        raise SystemExit("not enough registered normal boards; run `python -m demo.register` first")
    if C.YOLO_DATA.exists():
        shutil.rmtree(C.YOLO_DATA)
    rng = np.random.default_rng(seed)
    counts = {split: dict.fromkeys(C.CLASSES, 0) for split in ("train", "val")}
    images = {"train": 0, "val": 0}
    for index, path in enumerate(boards):
        split = "val" if index >= len(boards) - VAL_BOARDS else "train"
        board = imread(path)
        for k in range(VAL_PER_BOARD if split == "val" else TRAIN_PER_BOARD):
            tile = random_tile(board, rng)
            image, labels = synthesize(tile, rng, n_defects=0 if k % 8 == 0 else None)  # 1 in 8 stays clean
            stem = f"{path.stem}_{k:03d}"
            imwrite(C.YOLO_DATA / "images" / split / f"{stem}.jpg", image)
            label_path = C.YOLO_DATA / "labels" / split / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(yolo_lines(labels))
            images[split] += 1
            for l in labels:
                counts[split][C.CLASSES[l[0]]] += 1
    (C.YOLO_DATA / "data.yaml").write_text(
        f"path: {C.YOLO_DATA.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(C.CLASSES)))
    summary = {"images": images, "instances": counts, "val_boards": [p.name for p in boards[-VAL_BOARDS:]]}
    (C.YOLO_DATA / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
