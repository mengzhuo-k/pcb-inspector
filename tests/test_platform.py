import io
import os
import tempfile
import zipfile
import unittest
from pathlib import Path

storage = tempfile.TemporaryDirectory()
os.environ["PCB_PLATFORM_DATA"] = storage.name
os.environ["PCB_AUTO_TRAIN"] = "0"

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

    def test_annotation_quality_holds_image_out_of_training(self):
        iid = self.upload("retake.jpg")
        detail = self.client.get(f"/api/images/{iid}").json
        boxes = [dict(x1=.1, y1=.1, x2=.5, y2=.5, **{"class":"solder_bridge"})]
        response = self.client.post(f"/api/images/{iid}/labels", json={
            "boxes": boxes, "revision": detail["revision"],
            "annotation_view": {"quality_flags": ["glare", "occlusion"], "quality_note": "引脚根部补拍"},
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["review_status"], "retake")
        self.assertEqual(len(response.json["labels"]), 1)
        self.assertEqual(response.json["annotation_view"]["quality_note"], "引脚根部补拍")
        self.assertEqual(self.client.get(f"/api/projects/{self.project}/summary").json["retake"], 1)
        self.assertEqual(self.client.post(f"/api/projects/{self.project}/datasets", json={"template": "det"}).status_code, 400)
        response = self.client.post(f"/api/images/{iid}/labels", json={
            "boxes": boxes, "annotation_view": {"quality_flags": []},
        })
        self.assertEqual(response.json["review_status"], "approved")
        self.assertEqual(self.client.post(f"/api/projects/{self.project}/datasets", json={"template": "det"}).status_code, 200)

    def test_reference_remains_available_for_manual_alignment(self):
        ref = self.upload("reference.jpg")
        other = self.upload("other.jpg")
        endpoint = f"/api/projects/{self.project}/alignment-reference"
        self.client.post(endpoint,json={"image_id":ref,"only_if_empty":True})
        self.assertEqual(self.client.post(endpoint,json={"image_id":other,"only_if_empty":True}).status_code,409)
        self.label(ref)
        self.assertEqual(self.client.delete(f"/api/images/{ref}").status_code,200)
        with self.client.get(endpoint+"/file?id="+ref) as response:
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.mimetype,"image/jpeg")
        points = [[.1,.1],[.9,.1],[.9,.9],[.1,.9]]
        response = self.client.post(f"/api/images/{other}/alignment", json={"reference_id":ref,
            "source_points":points,"reference_points":points,"include_preview":False})
        self.assertEqual(response.status_code,200)
        self.assertEqual(self.client.get(f"/api/images/{other}").json["alignment"]["reference_id"],ref)

    def test_alignment_preserves_original_box_coordinates(self):
        import numpy as np
        iid = self.upload("alignment.jpg")
        self.assertEqual(self.client.post(f"/api/projects/{self.project}/alignment-reference", json={"image_id": iid}).status_code, 200)
        self.label(iid)
        original = self.client.get(f"/api/images/{iid}").json
        source = [[.1,.1],[.8,.15],[.9,.8],[.15,.9]]
        target = [[.1,.1],[.9,.1],[.9,.9],[.1,.9]]
        result = self.client.post(f"/api/images/{iid}/alignment", json={
            "source_points": source, "reference_points": target,
        })
        self.assertEqual(result.status_code, 200, result.json)
        self.assertTrue(result.json["preview"].startswith("data:image/jpeg;base64,"))
        matrix = np.array(result.json["matrix"])
        for src, dst in zip(source, target):
            projected = matrix @ [*src, 1]
            np.testing.assert_allclose(projected[:2] / projected[2], dst, atol=1e-6)
        import base64
        jpeg = base64.b64decode(result.json["preview"].split(",")[1])
        with self.client.get(f"/api/images/{iid}/file") as image:
            self.assertEqual(image.data, jpeg)
        detail = self.client.get(f"/api/images/{iid}").json
        self.assertEqual(detail["labels"], original["labels"])
        self.assertEqual(detail["review_status"], "approved")
        self.assertEqual(self.client.post(f"/api/images/{iid}/labels", json={"boxes":[]}).status_code,409)
        self.assertEqual(detail["labels"][0]["x1"], .1)
        self.assertEqual(detail["alignment"]["matrix"], result.json["matrix"])
        self.assertEqual(self.client.post(f"/api/images/{iid}/alignment", json={}).status_code, 409)
        did = self.client.post(f"/api/projects/{self.project}/datasets", json={"template":"det"}).json["id"]
        item = platform_app.frozen_items(did)[0]
        self.assertEqual(Path(item["path"]).read_bytes(), jpeg)
        corners = np.array([[.1,.1,1],[.5,.1,1],[.5,.5,1],[.1,.5,1]]) @ matrix.T
        coords = corners[:, :2] / corners[:, 2:]
        self.assertAlmostEqual(item["labels"][0]["x2"], coords[:,0].max(), places=5)

    def test_saved_legacy_alignment_is_materialized_once(self):
        ref = self.upload("reference.jpg")
        self.client.post(f"/api/projects/{self.project}/alignment-reference", json={"image_id":ref})
        iid = self.upload("legacy.jpg")
        matrix = [[.8,0,.1],[0,.8,.1],[0,0,1]]
        saved = self.client.post(f"/api/images/{iid}/labels", json={"boxes":[], "annotation_view":{
            "alignment":{"reference_id":ref, "matrix":matrix}}}).json
        platform_app.initialize()
        migrated = self.client.get(f"/api/images/{iid}").json
        self.assertNotEqual(migrated["path"], saved["path"])
        self.assertEqual(migrated["alignment"]["matrix"], matrix)
        platform_app.initialize()
        self.assertEqual(self.client.get(f"/api/images/{iid}").json, migrated)

    def test_applying_solder_template_preserves_selected_sites(self):
        from unittest.mock import patch
        reference = self.upload("selected-reference.jpg")
        self.client.post(f"/api/projects/{self.project}/alignment-reference", json={"image_id": reference})
        endpoint = f"/api/projects/{self.project}/solder-template"
        sites = [{"id": "H01", "x1": .4, "y1": .4, "x2": .42, "y2": .42, "origin": "manual"}]
        saved = self.client.post(endpoint, json={"action": "save", "reference_id": reference, "revision": 0, "sites": sites}).json
        with patch("solder_sites.detect_sites", side_effect=AssertionError("Selected sites must be preserved")):
            applied = self.client.post(endpoint, json={"action": "detect", "reference_id": reference, "revision": saved["revision"]})
        self.assertEqual(applied.status_code, 200, applied.json)
        self.assertEqual(applied.json["sites"], saved["sites"])

    def test_annotation_concurrent_save_and_invalid_settings(self):
        iid = self.upload("concurrent.jpg")
        self.label(iid)
        self.assertEqual(self.client.post(f"/api/images/{iid}/labels", json={"boxes": [], "revision": 0}).status_code, 409)
        self.assertEqual(len(self.client.get(f"/api/images/{iid}").json["labels"]), 1)
        for settings in [{"quality_flags": ["unknown"]}, {"alignment": {"matrix": [[0]*3]*3}}, {"alignment_disabled": "yes"}]:
            response = self.client.post(f"/api/images/{iid}/labels", json={"boxes": [], "annotation_view": settings})
            self.assertEqual(response.status_code, 400)
        result = self.client.post(f"/api/images/{iid}/labels", json={"boxes": [], "annotation_view": {"alignment_disabled": True}})
        self.assertEqual(result.status_code, 409)

    def test_aligned_label_corners_survive_save_reload_and_dataset_export(self):
        iid = self.upload("aligned-corners.jpg")
        polygon = [[.2,.1],[.5,.15],[.6,.5],[.15,.4]]
        box = {"class": "solder_bridge", "x1": .1, "y1": .1, "x2": .8, "y2": .8,
               "polygon": polygon}
        response = self.client.post(f"/api/images/{iid}/labels", json={"boxes": [box]})
        self.assertEqual(response.status_code, 200, response.json)
        detail = self.client.get(f"/api/images/{iid}").json
        self.assertEqual(detail["labels"][0]["polygon"], polygon)
        self.assertEqual(detail["labels"][0]["x1"], .15)
        self.assertEqual(detail["labels"][0]["x2"], .6)
        did = self.client.post(f"/api/projects/{self.project}/datasets", json={"template": "det"}).json["id"]
        with tempfile.TemporaryDirectory() as folder:
            dataset = platform_app.build_det_dataset(did, Path(folder))
            values = (dataset / "labels/train" / f"{iid}.txt").read_text().split()
            self.assertAlmostEqual(float(values[3]), .45)
            self.assertAlmostEqual(float(values[4]), .4)
        for invalid in [[[.2,.2]] * 4, [[.2,.2],[.5,.5],[.5,.2],[.2,.5]], [[-1,0],[1,0],[1,1],[0,1]]]:
            response = self.client.post(f"/api/images/{iid}/labels", json={"boxes": [{**box, "polygon": invalid}]})
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get(f"/api/images/{iid}").json["labels"][0]["polygon"], polygon)

    def test_solder_template_is_shared_and_does_not_create_defect_labels(self):
        reference = self.upload("reference.jpg")
        other = self.upload("angled.jpg")
        self.client.post(f"/api/projects/{self.project}/alignment-reference", json={"image_id": reference})
        endpoint = f"/api/projects/{self.project}/solder-template"
        sites = [{"id": "S0001", "x1": .1, "y1": .2, "x2": .15, "y2": .25}]
        result = self.client.post(endpoint, json={"action": "save", "reference_id": reference, "revision": 0, "sites": sites})
        self.assertEqual(result.status_code, 200, result.json)
        self.assertEqual(result.json["revision"], 1)
        self.assertEqual(self.client.get(endpoint).json["sites"][0]["id"], "S0001")
        self.assertEqual(self.client.get(f"/api/images/{other}").json["labels"], [])
        stale = self.client.post(endpoint, json={"action": "save", "reference_id": reference, "revision": 0, "sites": []})
        self.assertEqual(stale.status_code, 409)
        duplicate = self.client.post(endpoint, json={"action": "save", "reference_id": reference, "revision": 1, "sites": sites * 2})
        self.assertEqual(duplicate.status_code, 400)
        invalid = self.client.post(endpoint, json={"action": "save", "reference_id": reference, "revision": 1,
            "sites": [{**sites[0], "x1": -.1}]})
        self.assertEqual(invalid.status_code, 400)

    def test_solder_template_is_invalidated_when_reference_changes(self):
        reference = self.upload("reference.jpg")
        other = self.upload("new-reference.jpg")
        ref_endpoint = f"/api/projects/{self.project}/alignment-reference"
        endpoint = f"/api/projects/{self.project}/solder-template"
        self.client.post(ref_endpoint, json={"image_id": reference})
        self.assertEqual(self.client.post(endpoint, json={"action": "save", "reference_id": reference, "revision": 0, "sites": []}).status_code, 200)
        self.client.post(ref_endpoint, json={"image_id": other})
        self.assertIsNone(self.client.get(endpoint).json)
        self.assertEqual(self.client.post(endpoint, json={"action": "save", "reference_id": reference, "revision": 1, "sites": []}).status_code, 409)
        self.assertEqual(self.client.post(endpoint, json={"action": "save", "reference_id": other, "revision": 0, "sites": []}).status_code, 200)
        self.client.delete(f"/api/images/{other}")
        self.assertEqual(self.client.get(endpoint).json["reference_id"], other)

    def test_reference_and_localization_survive_deleting_reference_image(self):
        reference = self.upload("reference-to-delete.jpg")
        other = self.upload("processed.jpg")
        self.client.post(f"/api/projects/{self.project}/alignment-reference", json={"image_id": reference})
        sites = [{"id": "H01", "x1": .2, "y1": .2, "x2": .25, "y2": .25}]
        endpoint = f"/api/projects/{self.project}/solder-template"
        self.client.post(endpoint, json={"action": "save", "reference_id": reference, "revision": 0, "sites": sites})
        self.assertEqual(self.client.delete(f"/api/images/{reference}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/projects/{self.project}/alignment-reference").json["id"], reference)
        self.assertEqual(self.client.get(endpoint).json["sites"][0]["id"], "H01")
        result = self.client.post(f"/api/images/{other}/alignment", json={"reference_id": reference,
            "matrix": [[1,0,0],[0,1,0],[0,0,1]]})
        self.assertEqual(result.status_code, 200, result.json)
        detail = self.client.get(f"/api/images/{other}").json
        saved = self.client.post(f"/api/images/{other}/labels", json={"boxes": detail["labels"],
            "revision": detail["revision"], "annotation_view": {"alignment": {
                "reference_id": reference, "matrix": result.json["matrix"]}}})
        self.assertEqual(saved.status_code, 200, saved.json)
        self.assertEqual(saved.json["review_status"], "approved")

    def test_processing_filters_include_legacy_unprocessed_images(self):
        new = self.upload("new.jpg")
        processed = self.upload("processed.jpg")
        self.label(processed)
        legacy = self.upload("legacy.jpg")
        with platform_app.db() as conn:
            conn.execute("UPDATE images SET state='retake',review_status='retake' WHERE id=?", (legacy,))
        unprocessed = self.client.get(f"/api/projects/{self.project}/images?processing=unprocessed").json
        self.assertEqual({row["id"] for row in unprocessed}, {new, legacy})
        done = self.client.get(f"/api/projects/{self.project}/images?processing=processed").json
        self.assertEqual([row["id"] for row in done], [processed])

    def test_dedicated_reference_upload_does_not_add_training_image(self):
        result = self.client.post(f"/api/projects/{self.project}/alignment-reference",
            data={"image": (io_for_sample(), "standard.jpg")}, content_type="multipart/form-data")
        self.assertEqual(result.status_code, 200, result.json)
        self.assertEqual(self.client.get(f"/api/projects/{self.project}/images").json, [])
        with platform_app.db() as conn:
            source = platform_app.board_reference(conn, self.project)
        self.assertTrue(Path(source["path"]).is_file())
        self.assertEqual(source["id"], result.json["id"])

    def test_accuracy_returns_reference_when_no_model(self):
        response = self.client.get(f"/api/projects/{self.project}/accuracy")
        self.assertEqual(response.status_code, 200)
        payload = response.json
        self.assertIn("参考模型", payload["source"])
        self.assertTrue(payload["rows"])
        self.assertTrue(any(row["key"] == "poor_solder" for row in payload["rows"]))

    def test_model_update_waits_for_defect_labels(self):
        from unittest.mock import patch
        with patch.object(platform_app, "AUTO_TRAIN", True), patch.object(platform_app.QUEUE, "put") as queued:
            self.upload("normal.jpg")
            with platform_app.db() as conn:
                conn.execute("UPDATE model_updates SET changed=0 WHERE project_id=?", (self.project,))
            platform_app.prepare_model_update(self.project)
            status = self.client.get(f"/api/projects/{self.project}/model-update").json
            self.assertEqual(status["state"], "waiting")
            self.assertIn("缺陷标注", status["message"])
            queued.assert_not_called()

    def test_deleting_image_requests_automatic_model_update(self):
        from unittest.mock import patch
        with patch.object(platform_app, "AUTO_TRAIN", True), patch.object(platform_app.QUEUE, "put") as queued:
            iid = self.upload("delete-auto.jpg")
            before = self.client.get(f"/api/projects/{self.project}/model-update").json
            self.assertEqual(before["state"], "waiting")
            self.assertEqual(self.client.delete(f"/api/images/{iid}").status_code, 200)
            after = self.client.get(f"/api/projects/{self.project}/model-update").json
            self.assertEqual(after["state"], "pending")
            self.assertEqual(after["revision"], before["revision"] + 1)
            self.assertEqual(after["images"], 0)
            queued.assert_not_called()

    def test_model_update_queues_once_and_repeats_after_new_changes(self):
        from unittest.mock import patch
        with patch.object(platform_app, "AUTO_TRAIN", True), patch.object(platform_app.QUEUE, "put") as queued:
            ids = [self.upload(f"auto-{i}.jpg", board=f"auto-board-{i}") for i in range(4)]
            self.label(ids[0])
            for iid in ids[1:]:
                self.label(iid, cls="good_solder", boxes=False)
            with platform_app.db() as conn:
                conn.execute("UPDATE model_updates SET changed=0 WHERE project_id=?", (self.project,))
            platform_app.prepare_model_update(self.project)
            first = self.client.get(f"/api/projects/{self.project}/model-update").json
            self.assertEqual(first["state"], "queued")
            self.assertEqual(queued.call_count, 1)
            platform_app.prepare_model_update(self.project)
            self.assertEqual(queued.call_count, 1)
            self.label(self.upload("additional.jpg"), cls="solder_bridge")
            with platform_app.db() as conn:
                conn.execute("UPDATE jobs SET state='done' WHERE id=?", (first["job_id"],))
            platform_app.finish_model_update(first["job_id"])
            changed = self.client.get(f"/api/projects/{self.project}/model-update").json
            self.assertEqual(changed["state"], "pending")
            self.assertEqual(changed["trained_revision"], first["target_revision"])
            with platform_app.db() as conn:
                conn.execute("UPDATE model_updates SET changed=0 WHERE project_id=?", (self.project,))
            platform_app.prepare_model_update(self.project)
            second = self.client.get(f"/api/projects/{self.project}/model-update").json
            self.assertEqual(second["state"], "queued")
            self.assertNotEqual(first["job_id"], second["job_id"])
            self.assertEqual(queued.call_count, 2)
            with platform_app.db() as conn:
                conn.execute("UPDATE jobs SET state='failed',error='模拟训练失败' WHERE id=?", (second["job_id"],))
            platform_app.finish_model_update(second["job_id"])
            platform_app.prepare_model_update(self.project)
            self.assertEqual(queued.call_count, 2)
            self.assertEqual(self.client.get(f"/api/projects/{self.project}/model-update").json["state"], "failed")

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
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["retained_in_versions"])
        self.assertTrue(path.exists())
        self.assertEqual(self.client.get(f"/api/images/{iid}").status_code, 404)
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
        self.assertEqual(self.client.post(f"/api/images/{iid}/labels", json={"boxes":[]}).status_code, 409)
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
