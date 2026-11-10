import io
import os
import tempfile
import zipfile
import unittest
from pathlib import Path

storage = tempfile.TemporaryDirectory()
os.environ["PCB_PLATFORM_DATA"] = storage.name

import platform_app

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data/datasets/smd_defdb"


class PlatformTests(unittest.TestCase):
    def setUp(self):
        self.client = platform_app.app.test_client()
        self.project = self.client.post("/api/projects", json={
            "name": "UNIT", "board_model": "B1", "station": "S1",
        }).json["id"]

    def upload(self, name, board="A", batch="B1", cls="good_solder"):
        if cls == "good_solder":
            path = SAMPLES / "good_solder/good_solder_0.jpg"
        else:
            path = SAMPLES / "poor_solder/poor_solder_0.jpg"
        with path.open("rb") as stream:
            response = self.client.post(f"/api/projects/{self.project}/images", data={
                "files": (stream, name), "batch": batch, "board_serial": board,
                "primary_class": cls,
            }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200, response.get_json())
        return response.json["saved"][0]

    def label(self, image_id, cls="poor_solder_virtual", boxes=True):
        response = self.client.post(f"/api/images/{image_id}/labels", json={
            "boxes": [{"class": cls, "x1": .1, "y1": .1, "x2": .5, "y2": .5}] if boxes else [],
            "primary_class": cls, "annotator": "QA",
        })
        self.assertEqual(response.status_code, 200, response.get_json())

    def test_project_upload_label_review_dataset_and_feedback(self):
        ids = [self.upload(f"{index}.jpg", board=f"A{index}", cls="poor_solder_virtual")
               for index in range(4)]
        for image_id in ids:
            self.label(image_id)
        response = self.client.post(f"/api/projects/{self.project}/datasets", json={
            "template": "det", "split_policy": "board", "val_ratio": .25, "test_ratio": .25,
            "seed": 1, "created_by": "QA",
        })
        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.json
        self.assertGreaterEqual(payload["counts"]["train"], 1)
        self.assertGreaterEqual(payload["counts"]["val"], 1)
        detail = self.client.get(f"/api/datasets/{payload['id']}").json
        splits = {item["split"] for item in detail["items"]}
        self.assertEqual(splits, {"train", "val", "test"})

        feedback = self.client.post(f"/api/projects/{self.project}/feedback", data={
            "image": (io_for_sample(), "bad.jpg"), "predicted": "虚焊 0.61", "note": "实际桥接",
        }, content_type="multipart/form-data")
        self.assertEqual(feedback.status_code, 200)
        converted = self.client.post(f"/api/feedback/{feedback.json['id']}/convert", json={})
        self.assertEqual(converted.status_code, 200)
        self.assertEqual(self.client.get(f"/api/projects/{self.project}").json["images"], 5)

    def test_rejects_unapproved_and_bad_labels(self):
        image_id = self.upload("x.jpg")
        response = self.client.post(f"/api/projects/{self.project}/datasets", json={"template": "det"})
        self.assertEqual(response.status_code, 400)
        response = self.client.post(f"/api/images/{image_id}/labels", json={
            "boxes": [{"class": "not-a-class", "x1": .1, "y1": .1, "x2": .2, "y2": .2}],
        })
        self.assertEqual(response.status_code, 400)
        response = self.client.post(f"/api/images/{image_id}/labels", json={
            "boxes": [{"class": "poor_solder_virtual", "x1": .6, "y1": .1, "x2": .2, "y2": .2}],
        })
        self.assertEqual(response.status_code, 400)

    def test_saved_labels_are_stored_without_review(self):
        image_id = self.upload("auto.jpg", cls="poor_solder_virtual")
        response = self.client.post(f"/api/images/{image_id}/labels", json={
            "boxes": [{"class": "poor_solder_virtual", "x1": .1, "y1": .1, "x2": .5, "y2": .5}],
            "primary_class": "poor_solder_virtual", "annotator": "QA",
        })
        self.assertEqual(response.status_code, 200)
        detail = self.client.get(f"/api/images/{image_id}").json
        self.assertEqual(detail["review_status"], "approved")
        self.assertEqual(detail["reviewer"], "自动")
        dataset = self.client.post(f"/api/projects/{self.project}/datasets", json={"template": "det"})
        self.assertEqual(dataset.status_code, 200)

    def test_accuracy_returns_reference_when_no_model(self):
        response = self.client.get(f"/api/projects/{self.project}/accuracy")
        self.assertEqual(response.status_code, 200)
        payload = response.json
        self.assertIn("参考模型", payload["source"])
        self.assertTrue(payload["rows"])
        self.assertTrue(any(row["key"] == "poor_solder" for row in payload["rows"]))

    def test_export_model_bundle(self):
        card = {"template": "det", "classes": ["missing_connection"]}
        with platform_app.db() as conn:
            conn.execute("""INSERT INTO models(id,project_id,job_id,version,sha256,metrics,
                threshold,weights,card,published,created) VALUES(?,?,?,?,?,?,?,?,?,0,?)""",
                         ("export1", self.project, "j1", "v9", "sha", '{"map50":0.5}', .25,
                          str(ROOT / "models/missing_connection_v7.pt"),
                          __import__("json").dumps(card), platform_app.now()))
        response = self.client.get("/api/models/export1/export")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/zip")
        archive = zipfile.ZipFile(io.BytesIO(response.data))
        names = set(archive.namelist())
        self.assertIn("weights/best.pt", names)
        self.assertIn("model_card.json", names)
        self.assertIn("classes.txt", names)
        self.assertIn("README.txt", names)

    def test_publish_and_monitor(self):
        with platform_app.db() as conn:
            conn.execute("""INSERT INTO models(id,project_id,job_id,version,sha256,metrics,
                threshold,weights,card,published,created) VALUES(?,?,?,?,?,?,?,?,?,0,?)""",
                         ("m1", self.project, "j1", "v1", "sha", "{}", .5,
                          str(ROOT / "models/missing_connection_v7.pt"), "{}",
                          platform_app.now()))
        response = self.client.post("/api/models/m1/publish", json={"station": "S1"})
        self.assertEqual(response.status_code, 200)
        monitor = self.client.get(f"/api/projects/{self.project}/monitor").json
        self.assertEqual(monitor["deployment"]["version"], "v1")
        self.assertEqual(monitor["deployment"]["station"], "S1")


    def test_delete_saved_image_keeps_frozen_dataset_reproducible(self):
        iid = self.upload("delete-frozen.jpg")
        self.label(iid)
        did = self.client.post(f"/api/projects/{self.project}/datasets",
                               json={"template":"det"}).json["id"]
        detail = self.client.get(f"/api/images/{iid}").json
        path = Path(detail["path"])
        response = self.client.delete(f"/api/images/{iid}")
        self.assertEqual(response.status_code,200)
        self.assertTrue(response.json["retained_in_versions"])
        self.assertTrue(path.exists())
        self.assertEqual(self.client.get(f"/api/images/{iid}").status_code,404)
        self.assertEqual(self.client.get(f"/api/projects/{self.project}/images").json,[])
        with tempfile.TemporaryDirectory() as folder:
            dataset=platform_app.build_det_dataset(did,Path(folder))
            self.assertTrue((dataset/"images/train"/f"{iid}.jpg").is_file())
            self.assertTrue((dataset/"labels/train"/f"{iid}.txt").read_text())

    def test_delete_unversioned_image_removes_file(self):
        iid = self.upload("delete.jpg")
        self.label(iid)
        path = Path(self.client.get(f"/api/images/{iid}").json["path"])
        response = self.client.delete(f"/api/images/{iid}")
        self.assertEqual(response.status_code,200)
        self.assertFalse(response.json["retained_in_versions"])
        self.assertFalse(path.exists())
        self.assertEqual(self.client.delete(f"/api/images/{iid}").status_code,404)

    def test_dataset_snapshot_survives_label_edits(self):
        iid = self.upload("frozen.jpg")
        self.label(iid)
        did = self.client.post(f"/api/projects/{self.project}/datasets",
                               json={"template": "det"}).json["id"]
        self.label(iid, cls="solder_bridge")
        with tempfile.TemporaryDirectory() as folder:
            dataset = platform_app.build_det_dataset(did, Path(folder))
            label = next((dataset / "labels/train").glob("*.txt")).read_text()
            self.assertEqual(label.split()[0], str(platform_app.classes_for("det").index("poor_solder_virtual")))
        detail = self.client.get(f"/api/datasets/{did}").json
        self.assertEqual(detail["items"][0]["primary_class"], "poor_solder_virtual")

    def test_publish_keeps_other_station_and_can_roll_back(self):
        import json
        for mid in ("station-v1", "station-v2"):
            with platform_app.db() as conn:
                conn.execute("""INSERT INTO models(id,project_id,version,weights,sha256,card,metrics)
                    VALUES(?,?,?,?,?,?,?)""", (mid,self.project,mid,"unused","sha",
                    json.dumps({"template":"det"}),"{}"))
        for mid, station in (("station-v1","A"),("station-v1","B"),("station-v2","B")):
            self.assertEqual(self.client.post(f"/api/models/{mid}/publish",json={"station":station}).status_code,200)
        with platform_app.db() as conn:
            active = dict(conn.execute("SELECT station,model_id FROM deployments WHERE project_id=? AND active=1",
                                       (self.project,)).fetchall())
        self.assertEqual(active, {"A":"station-v1","B":"station-v2"})
        self.client.post("/api/models/station-v1/publish",json={"station":"B"})
        with platform_app.db() as conn:
            self.assertEqual(conn.execute("SELECT model_id FROM deployments WHERE project_id=? AND station='B' AND active=1",
                                         (self.project,)).fetchone()[0], "station-v1")

    def test_policy_and_unavailable_evaluation(self):
        import json
        response=self.client.post(f"/api/projects/{self.project}/policy",json={
            "station":"A","required_classes":["missing_connection"],"require_classifier":False})
        self.assertEqual(response.json["required_classes"],["missing_connection"])
        self.assertFalse(response.json["require_classifier"])
        with platform_app.db() as conn:
            conn.execute("INSERT INTO models(id,project_id,metrics,card) VALUES(?,?,?,?)",
                         ("unavailable",self.project,json.dumps({"evaluation_status":"unavailable"}),"{}"))
        self.assertEqual(self.client.post("/api/models/unavailable/publish",json={"station":"A"}).status_code,400)


def io_for_sample():
    from io import BytesIO
    return BytesIO((SAMPLES / "poor_solder/poor_solder_0.jpg").read_bytes())


if __name__ == "__main__":
    unittest.main()
