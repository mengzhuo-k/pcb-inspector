"""Batch queue regressions with controlled detector results and isolated storage."""
import io
import os
import sys
import tempfile
import time
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from PIL import Image

storage = tempfile.TemporaryDirectory()
os.environ["PCB_DATA"] = storage.name
os.environ["PCB_PLATFORM_DATA"] = str(Path(storage.name) / "platform")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


def image_bytes(red):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (255, 0, 0) if red else (255, 255, 255)).save(buffer, "PNG")
    return buffer.getvalue()


def controlled_detection(image, station=""):
    if image.getpixel((0, 0))[1] < 100:
        return [dict(key="missing_connection", label="疑似漏焊", score=.8, rect=[.1, .1, .4, .4])]
    return []


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = patch.object(app, "detect", controlled_detection)
        cls.detector.start()
        app.start_worker()

    @classmethod
    def tearDownClass(cls):
        cls.detector.stop()

    def setUp(self):
        self.client = app.app.test_client()

    def create_batch(self, count):
        response = self.client.post("/api/batches", json={
            "station": "QA", "name": "QA-BATCH", "filenames": [f"{i:03}.png" for i in range(count)],
        })
        self.assertEqual(response.status_code, 200)
        return response.json["id"]

    def upload(self, bid, index, red=False, data=None):
        with app.app.test_client() as client:
            return client.post(f"/api/batches/{bid}/items/{index}", data={
                "image": (io.BytesIO(data if data is not None else image_bytes(red)), "pcb.png"),
            })

    def wait_ready(self, bid):
        for _ in range(200):
            batch = self.client.get(f"/api/batches/{bid}").json
            if batch["state"] == "ready":
                return batch
            time.sleep(.05)
        self.fail("Batch detection did not finish")

    def test_model_hints_keep_automatic_verdict_and_corrections_are_idempotent(self):
        from model_runtime import Detection
        import platform_app
        project = platform_app.app.test_client().post("/api/projects",json={"name":"Review","station":"QA"}).json["id"]
        with app.db() as conn:
            rid = app.save_record(conn,Image.new("RGB",(32,32),"white"),
                Detection([],None,None,issues=["工位模型未加载"]),"QA","Review")
        detail=self.client.get("/api/records/"+rid).json
        self.assertEqual(detail["decision"],"合格")
        self.assertFalse(detail["review_required"])
        self.assertEqual(self.client.post("/api/records/"+rid+"/review",data={
            "reviewer":"QA","decision":"合格","revision":0}).status_code,400)
        self.assertEqual(detail["issues"],["工位模型未加载"])
        payload=dict(project_id=project,reviewer="QA",revision=0,
                     boxes=[dict(key="solder_bridge",rect=[.1,.1,.4,.4])])
        first=self.client.post("/api/records/"+rid+"/corrections",json=payload)
        second=self.client.post("/api/records/"+rid+"/corrections",json=payload)
        self.assertEqual(first.status_code,200)
        self.assertEqual(first.json["image_id"],second.json["image_id"])
        conflict = dict(payload,reviewer="Other")
        self.assertEqual(self.client.post("/api/records/"+rid+"/corrections",json=conflict).status_code,409)
        iid=first.json["image_id"]
        sample=platform_app.app.test_client().get("/api/images/"+iid).json
        self.assertEqual(sample["review_status"],"approved")
        self.assertEqual(sample["labels"][0]["class"],"solder_bridge")
        with platform_app.db() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM feedback WHERE image_id=?",(iid,)).fetchone()[0],1)
        self.assertEqual(self.client.get("/api/records/"+rid).json["boxes"],[])

    def test_200_images_auto_decisions_sampling_corrections_and_deletion(self):
        bid = self.create_batch(200)
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(lambda i: self.upload(bid, i, red=i % 2 == 0), range(200)))
        self.assertTrue(all(r.status_code == 200 for r in responses))
        self.assertEqual(self.upload(bid, 0).status_code, 200)
        self.assertEqual(self.client.post(f"/api/batches/{bid}/start").status_code, 200)
        batch = self.wait_ready(bid)
        self.assertIsNone(batch["completed_at"])
        self.assertEqual(batch["summary"]["completed"], 200)
        self.assertEqual(batch["summary"]["passed"], 100)
        self.assertEqual(batch["summary"]["rejected"], 100)
        self.assertEqual(batch["summary"]["rate"], 50)
        listed = next(r for r in self.client.get("/api/batches").json if r["id"] == bid)
        self.assertEqual(listed["rate"], 50)
        self.assertEqual(listed["rejected"], 100)
        self.assertEqual(sum(r["batch_id"] == bid for r in self.client.get("/api/batches/rejections").json),100)
        unselected = next(item for item in batch["items"] if item["decision"] == "合格")
        self.assertFalse(self.client.get("/api/records/"+unselected["record_id"]).json["review_required"])
        self.assertEqual(self.client.post("/api/records/"+unselected["record_id"]+"/review",
            data=dict(reviewer="QA",decision="合格",revision=0)).status_code,400)
        sampled_before_completion = self.client.post(f"/api/batches/{bid}/sample", json={"mode":"percent","value":10})
        self.assertEqual(sampled_before_completion.status_code, 200)
        submitted = self.client.post(f"/api/batches/{bid}/finish").json
        self.assertEqual(submitted["state"], "reviewing")
        self.assertIsNone(submitted["completed_at"])
        manifest = self.client.post(f"/api/batches/{bid}/report").json
        self.assertEqual(self.client.get(manifest["download_url"]).data[:4], b"%PDF")
        self.assertEqual(self.client.get(f"/api/batches/{bid}").json["state"], "reviewing")
        finished = self.client.post(f"/api/batches/{bid}/reports/{manifest['id']}/complete").json
        self.assertEqual(finished["state"], "completed")
        self.assertTrue(finished["completed_at"])
        repeated = self.client.post(f"/api/batches/{bid}/finish").json
        self.assertEqual(repeated["completed_at"], finished["completed_at"])
        self.assertEqual(self.client.post(f"/api/batches/{bid}/start").json["state"], "completed")
        self.assertEqual(self.client.post(f"/api/batches/{bid}/retry").json["state"], "completed")
        rejected = [r for r in self.client.get("/api/batches/rejections").json if r["batch_id"] == bid]
        self.assertEqual(len(rejected), 100)
        self.assertEqual({r["ordinal"] for r in rejected}, set(range(0, 200, 2)))
        with app.db() as conn:
            conn.execute("UPDATE batches SET state='importing' WHERE id=?", (bid,))
        self.assertFalse(any(r["batch_id"] == bid for r in self.client.get("/api/batches/rejections").json))
        with app.db() as conn:
            conn.execute("UPDATE batches SET state='completed' WHERE id=?", (bid,))
        with app.db() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM records WHERE batch_id=?", (bid,)).fetchone()[0], 200)

        batch = self.client.post(f"/api/batches/{bid}/sample", json={"mode": "percent", "value": 10}).json
        sampled = [r for r in batch["items"] if r["sampled"]]
        self.assertEqual(len(sampled), 10)
        self.assertTrue(all(r["auto_decision"] == "合格" for r in sampled))
        rid = sampled[0]["record_id"]
        data = dict(revision=0, reviewer="QA", decision="不合格", comment="抽样发现漏焊")
        self.assertEqual(self.client.post(f"/api/records/{rid}/review", data=data).status_code, 200)
        self.assertEqual(self.client.post(f"/api/records/{rid}/review", data=data).status_code, 409)
        batch = self.client.get(f"/api/batches/{bid}").json
        self.assertEqual(batch["summary"]["rate"], 49.5)
        self.assertEqual(batch["summary"]["auto_rate"], 50)
        self.assertEqual(batch["summary"]["sample_reviewed"], 1)
        self.assertEqual(batch["state"], "reviewing")
        self.assertEqual(self.client.post(f"/api/batches/{bid}/reports/{manifest['id']}/complete").status_code, 409)

        data.update(revision=1, decision="合格")
        self.assertEqual(self.client.post(f"/api/records/{rid}/review", data=data).status_code, 200)
        self.assertEqual(len(self.client.get(f"/api/records/{rid}").json["reviews"]), 2)
        batch = self.client.post(f"/api/batches/{bid}/sample", json={"mode":"percent","value":100}).json
        self.assertEqual(batch["summary"]["sampled"], 100)
        self.assertEqual(batch["summary"]["sample_reviewed"], 0)
        self.assertEqual(self.client.delete(f"/api/records/{rid}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/records/{rid}/image").status_code, 404)
        batch = self.client.get(f"/api/batches/{bid}").json
        self.assertEqual(batch["summary"]["deleted"], 1)
        self.assertEqual(batch["summary"]["completed"], 199)
        self.assertEqual(batch["summary"]["sampled"], 99)

    def test_whole_board_classifier_marks_batch_rejected(self):
        verdict = dict(key="poor_solder", label="虚焊/冷焊", score=.82, defect_score=.9,
                       is_defect=True, threshold=.5,
                       top=[dict(key="poor_solder", label="虚焊/冷焊", score=.82),
                            dict(key="solder_bridge", label="桥接", score=.1)],
                       model_version="smd-defect-test", model_sha256="sha")

        def classified(image, station=""):
            return app.Detection([], "v-test", "sha", classification=verdict)

        bid = self.create_batch(1)
        self.upload(bid, 0)
        with patch.object(app, "detect", classified):
            self.client.post(f"/api/batches/{bid}/start")
            batch = self.wait_ready(bid)
        self.assertEqual(batch["summary"]["rejected"], 1)
        self.assertEqual(batch["summary"]["defects"], {"虚焊/冷焊": 1})
        record = self.client.get(f"/api/records/{batch['items'][0]['record_id']}").json
        self.assertEqual(record["decision"], "不合格")
        self.assertEqual(record["boxes"], [])
        self.assertTrue(record["classification"]["is_defect"])
        self.assertEqual(record["analysis"][0]["label"], "虚焊/冷焊")

    def test_cannot_finish_while_inference_is_running(self):
        bid = self.create_batch(1)
        self.upload(bid, 0)
        entered = threading.Event()
        release = threading.Event()

        def slow_detection(image, station=""):
            entered.set()
            release.wait(5)
            return controlled_detection(image)

        with patch.object(app, "detect", slow_detection):
            self.client.post(f"/api/batches/{bid}/start")
            try:
                self.assertTrue(entered.wait(3))
                self.assertEqual(self.client.post(f"/api/batches/{bid}/finish").status_code, 409)
            finally:
                release.set()
            batch = self.wait_ready(bid)
        self.assertEqual(batch["state"], "ready")
        self.assertIsNone(batch["completed_at"])

    def test_partial_upload_invalid_image_and_retry(self):
        bid = self.create_batch(3)
        self.upload(bid, 0)
        self.assertEqual(self.client.post(f"/api/batches/{bid}/finish").status_code, 409)
        self.assertEqual(self.client.post(f"/api/batches/{bid}/start").status_code, 409)
        self.assertEqual(self.upload(bid, 1, data=b"corrupt").status_code, 400)
        self.upload(bid, 2, red=True)
        self.assertEqual(self.client.post(f"/api/batches/{bid}/finish").status_code, 409)
        with patch.object(app, "detect", side_effect=RuntimeError("temporary failure")):
            self.client.post(f"/api/batches/{bid}/start")
            batch = self.wait_ready(bid)
        self.assertEqual(batch["summary"]["failed"], 3)
        self.assertIsNone(batch["summary"]["rate"])
        self.assertEqual(self.client.post(f"/api/batches/{bid}/finish").json["state"], "reviewing")
        self.client.post(f"/api/batches/{bid}/retry")
        batch = self.wait_ready(bid)
        self.assertIsNone(batch["completed_at"])
        self.assertEqual(batch["summary"]["completed"], 2)
        self.assertEqual(batch["summary"]["failed"], 1)
        self.assertEqual(batch["summary"]["rate"], 50)
        self.upload(bid, 1)
        self.client.post(f"/api/batches/{bid}/start")
        batch = self.wait_ready(bid)
        self.assertEqual(batch["summary"]["completed"], 3)
        self.assertEqual(batch["summary"]["rate"], 66.7)

    def test_report_failure_and_stale_snapshot_do_not_complete_batch(self):
        bid = self.create_batch(1)
        self.upload(bid, 0)
        self.client.post(f"/api/batches/{bid}/start")
        batch = self.wait_ready(bid)
        self.assertEqual(self.client.post(f"/api/batches/{bid}/report").status_code, 409)
        self.client.post(f"/api/batches/{bid}/finish")
        with patch.object(app, "build_pdf", side_effect=RuntimeError("report failure")):
            self.assertEqual(self.client.post(f"/api/batches/{bid}/report").status_code, 500)
        self.assertEqual(self.client.get(f"/api/batches/{bid}").json["state"], "reviewing")
        self.client.post(f"/api/batches/{bid}/sample",json={"mode":"count","value":1})
        first = self.client.post(f"/api/batches/{bid}/report").json
        duplicate = self.client.post(f"/api/batches/{bid}/report").json
        self.assertEqual(first["id"], duplicate["id"])
        rid = batch["items"][0]["record_id"]
        self.client.post(f"/api/records/{rid}/review", data={
            "reviewer":"QA", "decision":"不合格", "revision":0,
        })
        self.assertEqual(self.client.post(f"/api/batches/{bid}/reports/{first['id']}/complete").status_code, 409)
        second = self.client.post(f"/api/batches/{bid}/report").json
        self.assertNotEqual(first["id"], second["id"])
        finished = self.client.post(f"/api/batches/{bid}/reports/{second['id']}/complete").json
        self.assertEqual(finished["state"], "completed")
        repeated = self.client.post(f"/api/batches/{bid}/reports/{second['id']}/complete").json
        self.assertEqual(repeated["completed_at"], finished["completed_at"])
        self.client.delete(f"/api/records/{rid}")
        self.assertEqual(self.client.get(f"/api/batches/{bid}").json["state"], "reviewing")


if __name__ == "__main__":
    unittest.main()
