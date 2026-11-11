"""Inspect trained solder-site regions in the aligned reference coordinate system."""
import math

import cv2
import numpy as np


def prepare_sites(image, sites, matrix=None):
    validated = []
    identifiers = set()
    for site in sites:
        key = str(site["id"])
        rect = [float(site[name]) for name in ("x1", "y1", "x2", "y2")]
        if key in identifiers or not all(math.isfinite(v) and 0 <= v <= 1 for v in rect) or rect[0] >= rect[2] or rect[1] >= rect[3]:
            raise ValueError("点位模板无效，请在训练平台重新保存")
        identifiers.add(key)
        validated.append(dict(id=key, rect=rect))
    if not validated:
        raise ValueError("点位模板为空，请在模型训练平台保存需要检查的点位")
    if matrix is not None:
        inverse = np.linalg.inv(np.asarray(matrix, dtype=float))
        uncovered = []
        for site in validated:
            x1, y1, x2, y2 = site["rect"]
            points = cv2.perspectiveTransform(np.float32([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]]), inverse)[0]
            if not np.isfinite(points).all() or (points < -.003).any() or (points > 1.003).any():
                uncovered.append(site["id"])
        if uncovered:
            raise ValueError("照片未覆盖模板点位 " + "、".join(uncovered[:8]) + "，请拍摄完整板面")
    width, height = image.size
    regions = [(0, 0, width, height)]
    for site in validated:
        x1,y1,x2,y2 = site["rect"]
        # Share contextual crops between nearby sites; the full-board pass retains
        # the scale used by existing board-level detector training.
        size = min(max(width, height), max(192, math.ceil(max((x2-x1)*width, (y2-y1)*height)*4)))
        left = max(0, min(width-size, round((x1+x2)*width/2-size/2)))
        top = max(0, min(height-size, round((y1+y2)*height/2-size/2)))
        region = (left, top, min(width,left+size), min(height,top+size))
        if any(a <= x1*width-2 and b <= y1*height-2 and c >= x2*width+2 and d >= y2*height+2
               for a,b,c,d in regions[1:]):
            continue
        if region not in regions:
            regions.append(region)
    return validated, regions


def collect_site_defects(results, regions, size, sites, rules, version):
    """Only template-covered predictions affect the verdict; merge crop duplicates."""
    width,height = size
    findings = {}
    if len(results) != len(regions):
        raise RuntimeError("点位识别结果不完整，请重试")
    for result,(left,top,right,bottom) in zip(results, regions):
        for x1,y1,x2,y2,score,cls in result.boxes.data.cpu().tolist():
            rect = [(left+x1)/width,(top+y1)/height,(left+x2)/width,(top+y2)/height]
            if not all(math.isfinite(v) for v in [*rect,score]):
                continue
            area = max(0,rect[2]-rect[0])*max(0,rect[3]-rect[1])
            if not area:
                continue
            overlaps = []
            for site in sites:
                a,b,c,d = site["rect"]
                overlap = max(0,min(rect[2],c)-max(rect[0],a))*max(0,min(rect[3],d)-max(rect[1],b))
                if overlap:
                    overlaps.append((site,overlap))
            if sum(value for _,value in overlaps)/area < .35:
                continue
            key = result.names[int(cls)]
            if key not in rules:
                continue
            for site,overlap in overlaps:
                if overlap/area < .1:
                    continue
                token = (site["id"],key)
                if token in findings and findings[token]["score"] >= score:
                    continue
                findings[token] = dict(key=key,label=rules[key][0],score=round(score,4),
                    model=version,rect=site["rect"],site_id=site["id"],detected_rect=rect)
    return list(findings.values())
