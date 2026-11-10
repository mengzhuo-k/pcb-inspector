"""Generate editable solder/lead location proposals from a board reference."""
import cv2
import numpy as np

from pcb_alignment import read_image


def detect_sites(path):
    image = read_image(path, limit=1920)
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(image.astype(np.int16))
    board = ((green > red * 1.12) & (green > blue * 1.02) & (green > 45)).astype(np.uint8) * 255
    board = cv2.morphologyEx(board, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))
    contours, _ = cv2.findContours(board, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未找到电路板区域，请使用清楚的正面标准图")
    board[:] = 0
    cv2.drawContours(board, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    board = cv2.erode(board, np.ones((7, 7), np.uint8))
    # Broad metallic areas survive opening. Thin silkscreen strokes tend to disappear.
    metal = ((hsv[:, :, 1] < 105) & (hsv[:, :, 2] > 110) & (green - red < 35)
             & (board > 0)).astype(np.uint8) * 255
    metal = cv2.morphologyEx(metal, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(metal)
    proposals = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if not (8 <= area <= 1800 and 3 <= w <= 65 and 3 <= h <= 65 and max(w, h) / min(w, h) < 6):
            continue
        component = (labels[y:y+h, x:x+w] == index).astype(np.uint8)
        thickness = cv2.distanceTransform(component, cv2.DIST_L2, 3).max()
        if thickness < 1.8 or area / (w * h) < .2:
            continue
        proposals.append([max(0, x-3), max(0, y-3), min(width, x+w+3), min(height, y+h+3)])
    # Circular exposed pads are useful locations even when their centers are dark.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 16, param1=100, param2=24,
                               minRadius=5, maxRadius=16)
    if circles is not None:
        for x, y, radius in circles[0]:
            x, y, radius = round(float(x)), round(float(y)), round(float(radius))
            if not (radius <= x < width-radius and radius <= y < height-radius and board[y, x]):
                continue
            roi = metal[y-radius:y+radius+1, x-radius:x+radius+1]
            if np.count_nonzero(roi) < radius * 2:
                continue
            proposals.append([x-radius-2, y-radius-2, x+radius+2, y+radius+2])
    # Keep one proposal per overlapping metal patch or circular pad.
    kept = []
    for rect in sorted(proposals, key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True):
        area = (rect[2]-rect[0])*(rect[3]-rect[1])
        duplicate = False
        for other in kept:
            intersection = max(0, min(rect[2], other[2])-max(rect[0], other[0])) * max(0, min(rect[3], other[3])-max(rect[1], other[1]))
            if intersection / area > .55:
                duplicate = True
                break
        if not duplicate:
            kept.append(rect)
    kept.sort(key=lambda r: (round((r[1]+r[3])/24), r[0]))
    return [dict(id=f"S{index:04}", x1=rect[0]/width, y1=rect[1]/height,
                 x2=rect[2]/width, y2=rect[3]/height, origin="auto")
            for index, rect in enumerate(kept, 1)]
