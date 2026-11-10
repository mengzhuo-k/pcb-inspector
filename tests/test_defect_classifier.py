import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from defect_classifier import CLASSES, DefectClassifier


class FakeValues:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeModel:
    task = "classify"
    names = {index: name for index, name in enumerate(CLASSES)}
    probabilities = [0.05, 0.05, 0.8, 0.02, 0.02, 0.03, 0.03]

    def __init__(self, path):
        pass

    def predict(self, image, **kwargs):
        return [SimpleNamespace(probs=SimpleNamespace(data=FakeValues(self.probabilities)),
                                names=self.names)]


class RejectingModel(FakeModel):
    task = "detect"


class BadLabelModel(FakeModel):
    names = {index: f"class{index}" for index in range(7)}


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        FakeModel.probabilities = [0.05, 0.05, 0.8, 0.02, 0.02, 0.03, 0.03]
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.weights = self.root / "weights.pt"
        self.weights.write_bytes(b"checkpoint")
        self.card = self.root / "card.json"
        self.card.write_text('{"version":"test","threshold":0.5,"labels":{}}', encoding="utf-8")

    def classifier(self, factory=FakeModel, **kwargs):
        return DefectClassifier(self.weights, self.card, factory=factory, **kwargs)

    def test_good_board_is_accepted(self):
        classifier = self.classifier()
        self.assertTrue(classifier.available())
        verdict = classifier.predict(Image.new("RGB", (64, 64)))
        self.assertFalse(verdict["is_defect"])
        self.assertEqual(verdict["key"], "good_solder")
        self.assertAlmostEqual(verdict["defect_score"], 0.2, places=4)
        self.assertEqual(len(verdict["top"]), 3)
        self.assertEqual(classifier.status()["classes"], CLASSES)
        self.assertEqual(classifier.status()["sha256"],
                         hashlib.sha256(b"checkpoint").hexdigest())

    def test_defect_top1_above_gate_is_rejected(self):
        FakeModel.probabilities = [0.02, 0.02, 0.2, 0.02, 0.02, 0.7, 0.02]
        classifier = self.classifier()
        verdict = classifier.predict(Image.new("RGB", (64, 64)))
        self.assertTrue(verdict["is_defect"])
        self.assertEqual(verdict["key"], "poor_solder")
        self.assertEqual(verdict["label"], "虚焊/冷焊")
        self.assertAlmostEqual(verdict["defect_score"], 0.8, places=4)

    def test_defect_top1_below_gate_is_not_rejected(self):
        FakeModel.probabilities = [0.02, 0.02, 0.4, 0.02, 0.02, 0.45, 0.02]
        classifier = self.classifier(threshold=0.8)
        verdict = classifier.predict(Image.new("RGB", (64, 64)))
        self.assertFalse(verdict["is_defect"])
        self.assertEqual(verdict["key"], "poor_solder")
        self.assertAlmostEqual(verdict["defect_score"], 0.6, places=4)

    def test_threshold_override(self):
        FakeModel.probabilities = [0.02, 0.02, 0.4, 0.02, 0.02, 0.45, 0.02]
        classifier = self.classifier(threshold=0.55)
        self.assertTrue(classifier.predict(Image.new("RGB", (64, 64)))["is_defect"])

    def test_wrong_task_or_classes_leave_classifier_unavailable(self):
        classifier = self.classifier(factory=RejectingModel)
        self.assertFalse(classifier.available())
        self.assertIn("分类模型", classifier.last_error)
        self.assertIsNone(classifier.predict(Image.new("RGB", (64, 64))))
        classifier = self.classifier(factory=BadLabelModel)
        self.assertFalse(classifier.available())
        self.assertIn("类别", classifier.last_error)

    def test_missing_weights_are_reported_without_raising(self):
        classifier = DefectClassifier(self.root / "absent.pt", self.card, factory=FakeModel)
        self.assertFalse(classifier.available())
        self.assertIn("加载失败", classifier.last_error)


if __name__ == "__main__":
    unittest.main()
