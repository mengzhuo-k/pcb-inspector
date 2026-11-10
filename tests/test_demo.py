"""Data-free tests for the demo pipeline (registration maths, synthesis labels, PatchCore search)."""
import unittest

try:
    import cv2
    import numpy as np
    import torch
except ImportError:  # the demo has its own requirements file
    cv2 = None


def board(seed=0, w=384, h=736):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (70, 130, 40)  # solder-mask green (BGR)
    for _ in range(60):
        x, y = int(rng.integers(10, w - 60)), int(rng.integers(10, h - 60))
        bw, bh = int(rng.integers(22, 50)), int(rng.integers(22, 50))
        cv2.rectangle(img, (x, y), (x + bw, y + bh), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
        cv2.putText(img, str(int(rng.integers(100, 999))), (x, y + bh), 0, 0.5, (255, 255, 255), 1)
    return img


@unittest.skipIf(cv2 is None, "demo dependencies not installed")
class DemoTests(unittest.TestCase):
    def test_order_corners_any_click_order(self):
        from demo.register import order_corners
        tl, tr, br, bl = (10, 12), (200, 20), (190, 400), (5, 390)
        for clicks in ([br, tl, bl, tr], [tr, br, bl, tl], [bl, tr, tl, br]):
            self.assertEqual(order_corners(clicks).tolist(), [list(map(float, p)) for p in (tl, tr, br, bl)])

    def test_four_point_warp_undoes_perspective(self):
        from demo.register import four_point_warp
        ref = board()
        h, w = ref.shape[:2]
        quad = np.float32([[120, 80], [560, 140], [520, 900], [60, 820]])
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]), quad)
        photo = cv2.warpPerspective(ref, M, (700, 1000))
        back = four_point_warp(photo, quad, size=(w, h))
        self.assertLess(float(np.abs(back.astype(int) - ref).mean()), 12)
        turned = four_point_warp(photo, quad, size=(w, h), rotate=2)  # 180 degrees
        self.assertLess(float(np.abs(turned[::-1, ::-1].astype(int) - ref).mean()), 12)
        from demo.register import best_rotation
        for k in range(4):  # corners clicked starting from any corner: the right turn is recovered
            self.assertEqual(best_rotation(photo, np.roll(quad, k, axis=0), cv2.resize(ref, (768, 1472))), k)

    def test_auto_registration_finds_rotated_board_and_rejects_noise(self):
        from demo import config as C
        from demo.register import Registrar
        ref = cv2.resize(board(1), (C.CANON_W, C.CANON_H), interpolation=cv2.INTER_CUBIC)
        registrar = Registrar(ref)
        quad = np.float32([[900, 150], [880, 800], [80, 760], [120, 100]])  # board lying on its side
        rect = np.float32([[0, 0], [C.CANON_W - 1, 0], [C.CANON_W - 1, C.CANON_H - 1], [0, C.CANON_H - 1]])
        photo = cv2.warpPerspective(ref, cv2.getPerspectiveTransform(rect, quad), (1000, 900), borderValue=(200, 200, 200))
        warped, reg = registrar.warp(photo)
        self.assertTrue(reg.ok, reg.reason)
        self.assertLess(float(np.abs(np.float32(reg.corners) - quad).max()), 6)
        self.assertLess(float(np.abs(warped.astype(int) - ref).mean()), 15)
        noise = np.random.default_rng(0).integers(0, 255, (600, 800, 3), dtype=np.uint8)
        self.assertFalse(registrar.warp(noise)[1].ok)

    def test_synthesis_labels_are_valid_and_change_pixels(self):
        from demo import config as C
        from demo.synth import synthesize, yolo_lines
        tile = cv2.resize(board(2), (768, 768))
        seen = set()
        for seed in range(6):
            image, labels = synthesize(tile, np.random.default_rng(seed), n_defects=5)
            self.assertEqual(image.shape, tile.shape)
            for cls, x1, y1, x2, y2 in labels:
                seen.add(cls)
                self.assertTrue(0 <= x1 < x2 <= 768 and 0 <= y1 < y2 <= 768)
                self.assertGreater(int(np.abs(image[y1:y2, x1:x2].astype(int) - tile[y1:y2, x1:x2]).sum()), 0)
            for line in yolo_lines(labels).splitlines():
                values = [float(v) for v in line.split()]
                self.assertTrue(all(0 <= v <= 1 for v in values[1:]))
        self.assertEqual(seen, set(range(len(C.CLASSES))))
        clean, none = synthesize(tile, np.random.default_rng(0), n_defects=0)
        self.assertEqual(none, [])
        self.assertTrue((clean == tile).all())

    def test_position_aware_nearest_distance(self):
        from demo.patchcore import nearest_distance
        gen = torch.Generator().manual_seed(0)
        bank = torch.randn(3, 8, 12, 10, generator=gen)
        self.assertLess(float(nearest_distance(bank[1], bank, radius=1).max()), 1e-2)
        shifted = torch.roll(bank[0], shifts=2, dims=2)  # 2 cells to the right
        inner = (slice(2, 10), slice(4, 8))
        self.assertLess(float(nearest_distance(shifted, bank[:1], radius=2)[inner].max()), 1e-2)
        self.assertGreater(float(nearest_distance(shifted, bank[:1], radius=1)[inner].min()), 0.5)

    def test_boxes_from_map(self):
        from demo.patchcore import boxes_from_map
        norm = np.zeros((200, 100), np.float32)
        norm[50:80, 20:50] = 2.0
        norm[150:153, 60:63] = 5.0  # too small, ignored
        boxes = boxes_from_map(norm, 1.0)
        self.assertEqual(len(boxes), 1)
        x1, y1, x2, y2 = boxes[0]["box"]
        self.assertTrue(x1 <= 20 and y1 <= 50 and x2 >= 50 and y2 >= 80)


if __name__ == "__main__":
    unittest.main()
