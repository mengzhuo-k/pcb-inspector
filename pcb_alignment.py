"""Planar board registration with a high-resolution image for annotation."""
import base64
import json

import cv2
import numpy as np
from PIL import Image


def read_image(path, limit=1600):
    image = (cv2.cvtColor(np.asarray(path.convert("RGB")), cv2.COLOR_RGB2BGR)
             if isinstance(path, Image.Image) else cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR))
    if image is None:
        raise ValueError("无法读取原图，请重新上传")
    scale = min(1, limit / max(image.shape[:2]))
    return cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image


def validate_matrix(value):
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("对齐参数无效") from None
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("对齐参数无效")
    if np.linalg.cond(matrix) > 1e8:
        raise ValueError("对齐参数退化，请重新选择参照点")
    return matrix


def validate_points(value):
    try:
        points = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        raise ValueError("请选择四个对应的板面位置") from None
    if points.shape != (4, 2) or not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1):
        raise ValueError("请选择四个对应的板面位置")
    if not cv2.isContourConvex(points) or abs(cv2.contourArea(points)) < .005:
        raise ValueError("四点请按顺时针顺序选择，并分布在板面四周")
    return points


def validate_registration(matrix):
    corners=np.array([[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=float) @ matrix.T
    depths=corners[:,2]
    if (depths > 0).any() and (depths < 0).any() or np.min(np.abs(depths)) < 1e-6:
        raise ValueError("板面匹配发生翻折，请重新拍摄完整板面")
    points=np.float32(corners[:,:2]/depths[:,None])
    if not np.isfinite(points).all() or not cv2.isContourConvex(points) or cv2.contourArea(points,oriented=True)<=0:
        raise ValueError("板面匹配方向异常，请重新拍摄或使用手动四点对齐")


def align(source_path, reference_path, source_points=None, reference_points=None, matrix=None):
    source, reference = read_image(source_path), read_image(reference_path)
    sh, sw = source.shape[:2]
    rh, rw = reference.shape[:2]
    matches_count = inliers_count = 0
    if matrix is not None:
        transform = validate_matrix(matrix)
        method = "saved"
    elif source_points is not None or reference_points is not None:
        src, dst = validate_points(source_points), validate_points(reference_points)
        if cv2.contourArea(src, oriented=True) * cv2.contourArea(dst, oriented=True) <= 0:
            raise ValueError("两张图的四个点需要按相同顺序选择")
        transform = validate_matrix(cv2.getPerspectiveTransform(src, dst))
        method = "manual"
    else:
        detector = cv2.SIFT_create(nfeatures=6000)
        sk, sd = detector.detectAndCompute(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), None)
        rk, rd = detector.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None)
        if sd is None or rd is None or len(rd) < 2:
            raise ValueError("可匹配的板面细节不足，请使用手动四点对齐")
        pairs = cv2.BFMatcher().knnMatch(sd, rd, k=2)
        matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .7 * pair[1].distance]
        matches_count = len(matches)
        if matches_count < 12:
            raise ValueError("对应位置不足，请使用手动四点对齐")
        src = np.float32([sk[m.queryIdx].pt for m in matches])
        dst = np.float32([rk[m.trainIdx].pt for m in matches])
        pixel_matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if pixel_matrix is None or mask is None:
            raise ValueError("自动对齐失败，请使用手动四点对齐")
        inliers = mask.ravel().astype(bool)
        inliers_count = int(inliers.sum())
        if inliers_count < 12 or inliers_count / matches_count < .4:
            raise ValueError("匹配结果不稳定，请使用手动四点对齐")
        for points, area in ((src[inliers], sw * sh), (dst[inliers], rw * rh)):
            if cv2.contourArea(cv2.convexHull(points)) / area < .03:
                raise ValueError("匹配位置过于集中，请选择板面四周的对应点")
        transform = validate_matrix(np.diag([1 / rw, 1 / rh, 1]) @ pixel_matrix @ np.diag([sw, sh, 1]))
        method = "auto"
    validate_registration(transform)
    # Matrix maps normalized original coordinates to normalized reference coordinates.
    source = read_image(source_path, limit=4096)
    sh, sw = source.shape[:2]
    scale = max(sw, sh, rw, rh) / max(rw, rh)
    rw, rh = round(rw * scale), round(rh * scale)
    pixel_matrix = np.diag([rw, rh, 1]) @ transform @ np.diag([1 / sw, 1 / sh, 1])
    preview = cv2.warpPerspective(source, pixel_matrix, (rw, rh))
    ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("对齐预览生成失败")
    return dict(matrix=transform.tolist(), method=method, matches=matches_count,
                inliers=inliers_count, width=rw, height=rh,
                preview="data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"))



def project_labels(labels, matrix):
    """Freeze detection labels in the coordinate system of the aligned JPEG."""
    frame = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    projected = []
    for label in labels:
        polygon = json.loads(label["polygon"] or "[]") or [
            [label["x1"], label["y1"]], [label["x2"], label["y1"]],
            [label["x2"], label["y2"]], [label["x1"], label["y2"]]]
        points = cv2.perspectiveTransform(np.float32([polygon]), np.asarray(matrix, dtype=float))[0]
        area, clipped = cv2.intersectConvexConvex(points, frame)
        if area > 0 and clipped is not None:
            coords = clipped.reshape(-1, 2).clip(0, 1)
            x1, y1 = coords.min(axis=0).tolist()
            x2, y2 = coords.max(axis=0).tolist()
            projected.append(dict(label, x1=x1, y1=y1, x2=x2, y2=y2, polygon=json.dumps(coords.tolist())))
    return projected
