import hashlib
import io
import sqlite3
import subprocess
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import learning
from model_runtime import ModelRuntime


class ModelStub:
    names = {0:"missing_connection"}
    task = "detect"

    def __init__(self, path):
        pass

    def predict(self, image, **kwargs):
        return [SimpleNamespace(boxes=[])]


class TrainingPublishTests(unittest.TestCase):
    def test_completed_training_publishes_and_failed_training_keeps_active_version(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            with patch.object(learning, "STORE", store):
                learning.initialize()
                fallback = store/"fallback.pt"
                fallback.write_bytes(b"baseline")
                digest = hashlib.sha256(fallback.read_bytes()).hexdigest()
                runtime = ModelRuntime(store, fallback, digest, factory=ModelStub)

                def finish_training(command, **kwargs):
                    folder = Path(command[2])
                    self.assertTrue((folder/"data.yaml").is_file())
                    self.assertTrue(list((folder/"labels/train").glob("*.txt")))
                    weights = folder/"fit/weights/best.pt"
                    weights.parent.mkdir(parents=True)
                    weights.write_bytes(b"trained checkpoint")

                with patch.object(learning.subprocess, "run", side_effect=finish_training):
                    self.assertTrue(learning.run_pending())
                    self.assertFalse(learning.run_pending())
                runtime.refresh()
                self.assertEqual(runtime.status()["version"], "v1")
                image = io.BytesIO()
                Image.new("RGB", (32, 32), "white").save(image, "PNG")
                image.seek(0)
                response = learning.app.test_client().post("/api/upload", data={
                    "image": (image, "new.png"), "mode": "normal", "split": "train",
                })
                self.assertEqual(response.status_code, 200)
                with patch.object(learning.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "fit")):
                    self.assertTrue(learning.run_pending())
                runtime.refresh()
                self.assertEqual(runtime.status()["version"], "v1")
                with closing(sqlite3.connect(store/"library.sqlite")) as conn:
                    self.assertEqual(conn.execute("SELECT state FROM jobs WHERE id=2").fetchone()[0], "失败")
