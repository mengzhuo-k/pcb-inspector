"""Step 2: PatchCore-style anomaly detection trained on registered normal boards only.

Every cell of the backbone feature map describes one image patch (stride 8 px, receptive field of a
few dozen px).  The memory bank stores the patch features of all normal boards.  Because the boards
are registered to one standard view, the bank is *position-aware*: a test patch is compared only
with normal patches from the same neighbourhood (+-`radius` cells).  A global bank would accept an
empty through-hole in the solder zone because unpopulated holes exist elsewhere on a good board.

No gradient training is involved; "training" = feature extraction, so it runs on a CPU in minutes.

Run:  python -m demo.patchcore
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .register import imread, imwrite, list_images

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PROJ_DIM = 256
RADIUS = 2            # neighbourhood in feature cells (8 px each) that absorbs residual misalignment
GAMMAS = (1.0,)       # optional photometric copies of each normal board (measured: no gain, 3x slower)
MARGIN = 1.15         # threshold = MARGIN x worst leave-one-out score of a normal board


class Extractor:
    """ImageNet wide_resnet50_2, layer2 + layer3, locally averaged, randomly projected to 256-d."""

    def __init__(self, projection: torch.Tensor | None = None):
        from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

        net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).eval()
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        if projection is None:
            gen = torch.Generator().manual_seed(0)
            projection = torch.randn(PROJ_DIM, 512 + 1024, generator=gen) / PROJ_DIM ** 0.5
        self.projection = projection

    @torch.inference_mode()
    def __call__(self, bgr: np.ndarray, gamma: float = 1.0) -> torch.Tensor:
        """BGR uint8 canonical image -> [D, H/8, W/8] float tensor."""
        rgb = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 255
        if gamma != 1.0:
            rgb = rgb ** gamma
        x = self.stem((rgb - MEAN) / STD)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.interpolate(F.avg_pool2d(f3, 3, 1, 1), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        feats = torch.cat([f2, f3], 1)
        return torch.einsum("pc,bchw->bphw", self.projection, feats)[0].contiguous()


@torch.inference_mode()
def nearest_distance(query: torch.Tensor, bank: torch.Tensor, radius: int = RADIUS) -> torch.Tensor:
    """Distance of every query patch to its nearest normal patch within +-radius cells.

    query [D,H,W], bank [N,D,H,W] -> [H,W]
    """
    q_sq = (query ** 2).sum(0)
    padded = F.pad(bank, (radius, radius, radius, radius), value=1e3)  # far-away padding never wins
    padded_sq = (padded ** 2).sum(1)
    h, w = query.shape[-2:]
    best = torch.full((h, w), float("inf"))
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            window = padded[:, :, dy:dy + h, dx:dx + w]
            dist = q_sq[None] + padded_sq[:, dy:dy + h, dx:dx + w] - 2 * torch.einsum("dhw,ndhw->nhw", query, window)
            best = torch.minimum(best, dist.amin(0))
    return best.clamp_min(0).sqrt()


def to_score_map(dist: torch.Tensor) -> np.ndarray:
    """Upsample a patch-distance map to canonical resolution and smooth it."""
    up = F.interpolate(dist[None, None], size=(C.CANON_H, C.CANON_W), mode="bilinear", align_corners=False)[0, 0]
    return cv2.GaussianBlur(up.numpy(), (0, 0), 4)


class PatchCore:
    def __init__(self, bank: torch.Tensor, tolerance: np.ndarray, threshold: float, projection: torch.Tensor, meta: dict):
        self.bank = bank                # [N, D, H, W]
        self.tolerance = tolerance      # [H, W] normal variation seen at each position (leave-one-out)
        self.threshold = threshold      # on the normalised score
        self.meta = meta
        self.extractor = Extractor(projection)

    # ---------- inference ----------
    def score(self, bgr: np.ndarray) -> dict:
        raw = to_score_map(nearest_distance(self.extractor(bgr), self.bank))
        norm = raw / self.tolerance
        return {
            "raw": raw,
            "map": norm,                               # > threshold means "unlike any normal board here"
            "score": float(norm.max()),
            "threshold": self.threshold,
            "is_anomalous": bool(norm.max() > self.threshold),
            "boxes": boxes_from_map(norm, self.threshold),
        }

    # ---------- persistence ----------
    def save(self, path: Path = C.PATCHCORE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"bank": self.bank.half(), "tolerance": torch.from_numpy(self.tolerance),
                    "threshold": self.threshold, "projection": self.extractor.projection, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: Path = C.PATCHCORE) -> "PatchCore":
        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls(state["bank"].float(), state["tolerance"].numpy(), float(state["threshold"]), state["projection"], state["meta"])


def boxes_from_map(norm: np.ndarray, threshold: float, min_area: int = 150) -> list[dict]:
    """Connected regions above the threshold -> boxes in canonical pixels, strongest first."""
    mask = (norm > threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        pad = 12
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(norm.shape[1], x + w + pad), min(norm.shape[0], y + h + pad)
        boxes.append({"box": [int(x1), int(y1), int(x2), int(y2)], "score": float(norm[y:y + h, x:x + w].max()), "area": int(area)})
    return sorted(boxes, key=lambda b: -b["score"])


def overlay(bgr: np.ndarray, norm: np.ndarray, threshold: float, boxes: list[dict] | None = None) -> np.ndarray:
    """Heatmap overlay: transparent below ~60 % of the threshold, red at/above it."""
    level = np.clip(norm / (threshold * 1.5), 0, 1)
    heat = cv2.applyColorMap((level * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = np.clip((norm / threshold - 0.6) / 0.4, 0, 1)[..., None] * 0.6
    out = (bgr * (1 - alpha) + heat * alpha).astype(np.uint8)
    for b in boxes or []:
        x1, y1, x2, y2 = b["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(out, f"{b['score']:.2f}", (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return out


def fit(normal_paths: list[Path], log=print) -> tuple[PatchCore, dict]:
    """Build the memory bank and calibrate tolerance + threshold by leave-one-board-out."""
    extractor = Extractor()
    started = time.time()
    feats = []  # per board: [len(GAMMAS), D, H, W]
    for path in normal_paths:
        image = imread(path)
        feats.append(torch.stack([extractor(image, g) for g in GAMMAS]))
        log(f"features {path.name}  ({time.time() - started:.0f}s)")

    # Leave-one-out: how unlike the *other* normal boards is each board, position by position?
    loo_maps = []
    for i in range(len(feats)):
        others = torch.cat([f for j, f in enumerate(feats) if j != i])
        loo_maps.append(to_score_map(nearest_distance(feats[i][0], others)))
    loo = np.stack(loo_maps)
    # Tolerance = worst normal deviation seen around each position, with a floor so that quiet
    # areas are not over-sensitive.  Tall parts (parallax) and glue/coating get a high tolerance.
    worst = cv2.dilate(loo.max(0), np.ones((33, 33), np.uint8))
    tolerance = np.maximum(cv2.GaussianBlur(worst, (0, 0), 8), float(np.median(loo)) * 1.5).astype(np.float32)
    # By construction every normal board is <= ~1 everywhere; re-score with a second leave-one-out
    # pass over the tolerance itself so the threshold is not calibrated on its own training data.
    loo_scores = []
    for i in range(len(feats)):
        rest = np.delete(loo, i, axis=0).max(0)
        tol_i = np.maximum(cv2.GaussianBlur(cv2.dilate(rest, np.ones((33, 33), np.uint8)), (0, 0), 8), float(np.median(loo)) * 1.5)
        loo_scores.append(float((loo[i] / tol_i).max()))
    threshold = round(MARGIN * max(1.0, float(np.percentile(loo_scores, 75))), 3)

    meta = {"normals": [p.name for p in normal_paths], "backbone": "wide_resnet50_2 layer2+layer3",
            "radius_cells": RADIUS, "gammas": list(GAMMAS), "loo_scores": [round(s, 3) for s in loo_scores]}
    model = PatchCore(torch.cat(feats), tolerance, threshold, extractor.projection, meta)
    model._fit_cache = (feats, loo, normal_paths)
    return model, {"loo_scores": dict(zip(meta["normals"], meta["loo_scores"])), "threshold": threshold,
                   "fit_seconds": round(time.time() - started, 1)}


def synthetic_sensitivity(model: PatchCore, per_board: int = 6) -> dict:
    """How often does the heatmap reach the review level on an injected defect?

    Leave-one-out: defects are injected into board i and scored against a bank and a tolerance map
    built from the other boards, i.e. as if board i were a new board.  Synthetic defects only.
    """
    from .synth import synthesize
    feats, loo, paths = model._fit_cache
    floor = float(np.median(loo)) * 1.5
    review = model.threshold * 0.8
    hits = {name: [0, 0] for name in C.CLASSES}
    for i, path in enumerate(paths):
        image, labels = synthesize(imread(path), np.random.default_rng(100 + i), n_defects=per_board)
        others = torch.cat([f for j, f in enumerate(feats) if j != i])
        raw = to_score_map(nearest_distance(model.extractor(image), others))
        rest = cv2.dilate(np.delete(loo, i, axis=0).max(0), np.ones((33, 33), np.uint8))
        norm = raw / np.maximum(cv2.GaussianBlur(rest, (0, 0), 8), floor)
        for cls, x1, y1, x2, y2 in labels:
            hits[C.CLASSES[cls]][1] += 1
            hits[C.CLASSES[cls]][0] += bool(norm[y1:y2, x1:x2].max() > review)
    return {name: {"found": found, "injected": total} for name, (found, total) in hits.items()}


def main() -> None:
    normals = list_images(C.WARPED / "normal")
    if len(normals) < 3:
        raise SystemExit("need at least 3 registered normal boards; run `python -m demo.register` first")
    model, report = fit(normals)
    model.save()
    out = C.OUT / "patchcore_preview"
    report["abnormal"] = {}
    for path in list_images(C.WARPED / "abnormal"):
        image = imread(path)
        started = time.time()
        result = model.score(image)
        report["abnormal"][path.name] = {"score": round(result["score"], 3), "is_anomalous": result["is_anomalous"],
                                         "boxes": result["boxes"][:10], "seconds": round(time.time() - started, 1)}
        imwrite(out / f"{path.stem}_heatmap.jpg", overlay(image, result["map"], result["threshold"], result["boxes"]))
    report["synthetic_sensitivity_leave_one_out"] = synthetic_sensitivity(model)
    C.PATCHCORE_REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
