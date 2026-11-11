"""Inspection uploads reach training without relying on successful inference."""
import io
import hashlib
import os
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from PIL import Image

storage = tempfile.TemporaryDirectory()
os.environ["PCB_DATA"] = storage.name
os.environ["PCB_PLATFORM_DATA"] = str(Path(storage.name) / "platform")
os.environ["PCB_AUTO_TRAIN"] = "0"

import app
import platform_app
from model_runtime import Detection


def image_bytes(color="white", format="PNG"):
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(stream, format, **({"quality": 94} if format == "JPEG" else {}))
    return stream.getvalue()


def detection(image, station="", inspection_context=None):
    result = Detection([], "test", "test")
    result.alignment = None
    return result


class InspectionImportTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()
        self.training = platform_app.app.test_client()
        self.station = "QA-" + uuid.uuid4().hex
        self.client.post("/api/training-import-setting", json={"enabled": True})

    def sources(self):
        with platform_app.db() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM images WHERE camera=?", (self.station,))]

    def upload(self, endpoint="/api/preview", **fields):
        return self.client.post(endpoint, data={"image": (io.BytesIO(image_bytes(tuple(hashlib.sha256(self.station.encode()).digest()[:3]))), "board.png"),
                                               "station": self.station, **fields})

    @patch.object(app, "detect", side_effect=detection)
    def test_upload_preview_and_record_share_one_unlabeled_sample(self, _):
        self.assertEqual(self.upload().status_code, 200)
        self.assertEqual(self.upload("/api/records", batch="B-01").status_code, 200)
        samples = self.sources()
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["state"], "new")
        self.assertEqual(samples[0]["review_status"], "pending")
        detail = self.training.get("/api/images/" + samples[0]["id"]).json
        self.assertEqual(detail["labels"], [])
        self.assertEqual(self.training.get("/api/inspection-workspace").json["id"], samples[0]["project_id"])

    @patch.object(app, "detect", side_effect=RuntimeError("inference unavailable"))
    def test_inference_failure_does_not_discard_import(self, _):
        with self.assertLogs(app.app.logger, level="ERROR"):
            self.assertEqual(self.upload().status_code, 500)
        self.assertEqual(len(self.sources()), 1)

    @patch.object(app, "detect", side_effect=detection)
    def test_live_preview_only_imports_when_captured(self, _):
        self.assertEqual(self.upload(live_preview="1").status_code, 200)
        self.assertFalse(self.sources())
        self.assertEqual(self.upload("/api/records").status_code, 200)
        self.assertEqual(len(self.sources()), 1)

    @patch.object(app, "detect", side_effect=detection)
    def test_existing_image_preview_does_not_add_training_image(self, _):
        self.assertEqual(self.upload(preview_only="1").status_code,200)
        self.assertFalse(self.sources())

    def test_batch_is_imported_before_queue_runs_and_retries_are_idempotent(self):
        bid = self.client.post("/api/batches", json={"station": self.station, "name": "B-02",
            "filenames": ["original.png"]}).json["id"]
        endpoint = f"/api/batches/{bid}/items/0"
        self.assertEqual(self.upload(endpoint).status_code, 200)
        sample = self.sources()[0]
        self.assertEqual(sample["filename"], "original.png")
        self.assertEqual(sample["batch"], "B-02")
        self.assertEqual(sample["state"], "new")
        self.assertEqual(self.upload(endpoint).status_code, 200)
        self.assertEqual(len(self.sources()), 1)

    @patch.object(app, "detect", side_effect=detection)
    def test_failed_delivery_is_persisted_and_retried(self, _):
        with patch.object(platform_app, "import_inspection_image", side_effect=OSError("busy")):
            with self.assertLogs(app.app.logger, level="ERROR"):
                self.assertEqual(self.upload().status_code, 200)
        self.assertFalse(self.sources())
        with app.db() as conn:
            queued = conn.execute("SELECT * FROM training_imports WHERE station=?", (self.station,)).fetchone()
            self.assertIsNotNone(queued["image"])
            self.assertIsNone(queued["image_id"])
        app.flush_training_imports()
        self.assertEqual(len(self.sources()), 1)
        with app.db() as conn:
            queued = conn.execute("SELECT * FROM training_imports WHERE station=?", (self.station,)).fetchone()
            self.assertIsNone(queued["image"])
            self.assertEqual(queued["error"], "")

    def test_history_backfill_uses_original_and_respects_training_deletion(self):
        original = image_bytes("red", "JPEG")
        with app.db() as conn:
            rid = app.save_record(conn, Image.new("RGB", (32, 32), "white"), [], self.station, "history")
            conn.execute("UPDATE records SET original_image=? WHERE id=?", (original, rid))
        app.backfill_training_imports()
        sample = self.sources()[0]
        self.assertEqual(Path(sample["path"]).read_bytes(), original)
        with app.db() as conn:
            conn.execute("DELETE FROM training_imports WHERE station=?", (self.station,))
        self.assertEqual(self.training.delete("/api/images/" + sample["id"]).status_code, 200)
        app.backfill_training_imports()
        self.assertFalse(self.sources())

    def test_library_pagination_reaches_images_beyond_500(self):
        pid = self.training.post("/api/projects", json={"name": "paged"}).json["id"]
        with platform_app.db() as conn:
            conn.executemany("INSERT INTO images(id,project_id,filename,path,created) VALUES(?,?,?,?,?)",
                             [(f"page-{i}", pid, f"{i}.jpg", "unused", "same-time") for i in range(505)])
        pages = [self.training.get(f"/api/projects/{pid}/images?limit=200&offset={offset}").json
                 for offset in (0, 200, 400)]
        self.assertEqual([len(page) for page in pages], [200, 200, 105])
        self.assertEqual(len({row["id"] for page in pages for row in page}), 505)

    @patch.object(app, "detect", side_effect=detection)
    def test_switch_excludes_new_uploads_and_history_until_explicitly_reimported(self, _):
        response = self.client.post("/api/training-import-setting", json={"enabled": False})
        self.assertEqual(response.json, {"enabled": False})
        self.assertFalse(app.app.test_client().get("/api/training-import-setting").json["enabled"])
        self.assertEqual(self.upload("/api/records", batch="excluded").status_code, 200)
        self.assertFalse(self.sources())
        self.client.post("/api/training-import-setting", json={"enabled": True})
        app.backfill_training_imports()
        self.assertFalse(self.sources())
        self.assertEqual(self.upload().status_code, 200)
        self.assertEqual(len(self.sources()), 1)

    def test_switch_excludes_batches_but_allows_direct_training_upload(self):
        self.client.post("/api/training-import-setting", json={"enabled": False})
        bid = self.client.post("/api/batches", json={"station": self.station, "name": "excluded",
            "filenames": ["board.png"]}).json["id"]
        self.assertEqual(self.upload(f"/api/batches/{bid}/items/0").status_code, 200)
        self.assertFalse(self.sources())
        pid = self.training.get("/api/inspection-workspace").json["id"]
        direct = self.training.post(f"/api/projects/{pid}/images", data={
            "files": (io.BytesIO(image_bytes()), "direct.png")})
        self.assertEqual(direct.status_code, 200)
        self.assertEqual(self.training.get("/api/images/" + direct.json["saved"][0]).json["filename"], "direct.png")
        self.assertEqual(self.client.post("/api/training-import-setting", json={"enabled": "false"}).status_code, 400)

    @patch.object(app, "detect", side_effect=detection)
    def test_z_completed_batches_keep_import_policy_across_jpeg_reencoding(self, _):
        image = Image.frombytes("RGB", (32, 32), bytes((i * 67 + i // 13) % 256 for i in range(32 * 32 * 3)))
        stream = io.BytesIO()
        image.save(stream, "PNG")
        stations = []
        app.start_worker()
        for enabled in (False, True):
            station = self.station + str(enabled)
            stations.append(station)
            self.client.post("/api/training-import-setting", json={"enabled": enabled})
            bid = self.client.post("/api/batches", json={"station": station, "name": "reencoded",
                "filenames": ["pattern.png"]}).json["id"]
            response = self.client.post(f"/api/batches/{bid}/items/0", data={
                "image": (io.BytesIO(stream.getvalue()), "pattern.png")})
            self.assertEqual(response.status_code, 200)
            with patch.object(app.platform_runtime,"alignment_reference",return_value=None):
                self.client.post(f"/api/batches/{bid}/start")
            for attempt in range(100):
                batch = self.client.get(f"/api/batches/{bid}").json
                if batch["state"] == "ready":
                    break
                time.sleep(.05)
            self.assertEqual(batch["items"][0]["state"], "done")
        app.backfill_training_imports()
        with platform_app.db() as conn:
            counts = [conn.execute("SELECT count(*) FROM images WHERE camera=?", (station,)).fetchone()[0]
                      for station in stations]
        self.assertEqual(counts, [0, 1])


if __name__ == "__main__":
    unittest.main()
