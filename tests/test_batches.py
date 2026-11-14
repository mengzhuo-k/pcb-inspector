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


def controlled_detection(image, station="", inspection_context=None):
    if image.getpixel((0, 0))[1] < 100:
        return [dict(key="missing_connection", label="疑似漏焊", score=.8, rect=[.1, .1, .4, .4])]
    return []


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = patch.object(app, "detect", controlled_detection)
        cls.detector.start()
        cls.reference=patch.object(app.platform_runtime,"alignment_reference",return_value=None)
        cls.reference.start()
        app.start_worker()

    @classmethod
    def tearDownClass(cls):
        cls.detector.stop()
        cls.reference.stop()

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
        self.assertTrue(detail["review_available"])
        self.assertEqual(self.client.post("/api/records/"+rid+"/review",data={
            "reviewer":"QA","decision":"合格","revision":99}).status_code,409)
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

    def test_gallery_summary_counts_distinct_sites_and_keeps_all_defect_types(self):
        from model_runtime import Detection
        bid=self.create_batch(3)
        boxes=[dict(key=key,label=label,site_id=site,score=.8,rect=[.1,.1,.4,.4])
               for key,label,site in [('missing_connection','疑似漏焊','H01'),
                                     ('missing_connection','疑似漏焊','H01'),
                                     ('solder_bridge','桥接','H01'),
                                     ('missing_connection','疑似漏焊','H02')]]
        for ordinal,findings in enumerate([boxes,[]]):
            detection=Detection(findings,'v-test','sha')
            detection.inspection=dict(mode='solder_sites',checked_sites=26,template_revision=1)
            with app.db() as conn:
                rid=app.save_record(conn,Image.new('RGB',(32,32),'white'),detection,'QA','QA-BATCH',batch_id=bid,filename=f'{ordinal}.png')
                conn.execute("UPDATE batch_items SET state='done',record_id=? WHERE batch_id=? AND ordinal=?",(rid,bid,ordinal))
        response=self.client.get('/api/batches/'+bid)
        self.assertEqual(response.status_code,200)
        first,normal,pending=response.json['items']
        self.assertEqual(first['defect_summary'],dict(abnormal_sites=2,checked_sites=26,unlocated_count=0,defects=[
            dict(key='missing_connection',label='疑似漏焊',count=2,point_based=True),
            dict(key='solder_bridge',label='桥接',count=1,point_based=True)]))
        self.assertEqual(normal['defect_summary'],dict(abnormal_sites=0,checked_sites=26,unlocated_count=0,defects=[]))
        self.assertIsNone(pending['defect_summary'])
        self.assertNotIn('detection_boxes',first)
        self.assertEqual(self.client.get('/api/records/'+first['record_id']).json['boxes'],boxes)

    def test_legacy_unlocated_defects_are_not_reported_as_template_sites(self):
        boxes=[dict(key='missing_connection',label='疑似漏焊')]*2
        summary=app.summarize_point_defects(boxes)
        self.assertIsNone(summary['abnormal_sites'])
        self.assertIsNone(summary['checked_sites'])
        self.assertEqual(summary['unlocated_count'],2)
        self.assertEqual(summary['defects'][0]['count'],2)
        self.assertFalse(summary['defects'][0]['point_based'])

    def test_review_changes_sync_image_metadata_and_batch_counts_in_both_directions(self):
        bid=self.create_batch(1)
        self.upload(bid,0,red=True)
        self.client.post(f'/api/batches/{bid}/start')
        before=self.wait_ready(bid)
        original=before['items'][0]['defect_summary']
        rid=before['items'][0]['record_id']
        reviewed=self.client.post(f'/api/records/{rid}/review',data={
            'decision':'合格','reviewer':'QA甲','revision':0,'comment':'确认可接受'})
        self.assertEqual(reviewed.status_code,200,reviewed.json)
        passed=self.client.get(f'/api/batches/{bid}').json
        item=passed['items'][0]
        self.assertEqual(item['decision'],'合格')
        self.assertEqual(item['auto_decision'],'不合格')
        self.assertEqual(item['reviewer'],'QA甲')
        self.assertEqual(item['comment'],'确认可接受')
        self.assertTrue(item['signed'])
        self.assertEqual(item['manual'],[])
        self.assertEqual(item['defect_summary'],original)
        self.assertEqual((passed['summary']['passed'],passed['summary']['rejected'],passed['summary']['rate']),(1,0,100))
        self.assertEqual(passed['summary']['auto_rate'],0)
        reviewed=self.client.post(f'/api/records/{rid}/review',data={
            'decision':'不合格','reviewer':'QA乙','revision':1,'manual':'冷焊','comment':'补充确认冷焊'})
        self.assertEqual(reviewed.status_code,200,reviewed.json)
        rejected=self.client.get(f'/api/batches/{bid}').json
        item=rejected['items'][0]
        self.assertEqual((item['decision'],item['reviewer'],item['manual'],item['revision']),('不合格','QA乙',['冷焊'],2))
        self.assertEqual(item['defect_summary'],original)
        self.assertEqual((rejected['summary']['passed'],rejected['summary']['rejected'],rejected['summary']['rate']),(0,1,0))
        listed=next(batch for batch in self.client.get('/api/batches').json if batch['id']==bid)
        self.assertEqual((listed['passed'],listed['rejected'],listed['reviewed']),(0,1,1))

    def test_batch_process_parameters_flow_into_analysis_and_report(self):
        batch=self.client.post("/api/batches",json={"station":"PROCESS-BATCH","name":"Process check",
            "filenames":["board.png"],"temp":244,"reflow_seconds":54,"solder_batch":"SP-BATCH"}).json
        bid=batch["id"]
        self.assertEqual(self.upload(bid,0,red=True).status_code,200)
        self.client.post(f"/api/batches/{bid}/start")
        ready=self.wait_ready(bid)
        record=self.client.get("/api/records/"+ready["items"][0]["record_id"]).json
        self.assertEqual(record["temp"],244)
        self.assertEqual(record["process_parameters"]["solder_batch"],"SP-BATCH")
        self.assertIn("54 秒",record["analysis"][0]["evidence"])
        self.client.post(f"/api/batches/{bid}/finish")
        report=self.client.post(f"/api/batches/{bid}/report")
        self.assertEqual(report.status_code,200)
        self.assertTrue(self.client.get(report.json["download_url"]).data.startswith(b"%PDF"))
        with app.db() as conn:
            snapshot,_=app.report_snapshot(conn,bid)
            self.assertIn("SP-BATCH",snapshot["items"][0]["process_parameters"])

    def test_employee_can_review_any_image_or_archive_without_reviews(self):
        bid = self.create_batch(2)
        self.upload(bid, 0)
        self.upload(bid, 1, red=True)
        self.client.post(f"/api/batches/{bid}/start")
        batch = self.wait_ready(bid)
        for item in batch["items"]:
            record = self.client.get("/api/records/"+item["record_id"]).json
            self.assertFalse(record["review_required"])
            self.assertTrue(record["review_available"])
            self.assertFalse(record["signed"])
        finished = self.client.post(f"/api/batches/{bid}/finish")
        self.assertEqual(finished.status_code, 200)
        self.assertEqual(finished.json["state"], "completed")
        self.assertEqual([r["decision"] for r in finished.json["items"]], ["合格", "不合格"])
        frozen = self.client.get("/api/records/"+batch["items"][0]["record_id"]).json
        self.assertFalse(frozen["review_available"])

        bid = self.create_batch(1)
        self.upload(bid, 0)
        self.client.post(f"/api/batches/{bid}/start")
        item = self.wait_ready(bid)["items"][0]
        self.assertFalse(item["sampled"])
        response = self.client.post("/api/records/"+item["record_id"]+"/review",
            data=dict(reviewer="QA",decision="不合格",revision=0))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["decision"], "不合格")
        self.assertTrue(response.json["signed"])
        self.assertFalse(response.json["review_required"])

    def test_200_images_auto_decisions_sampling_and_corrections(self):
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
        self.assertEqual(sum(r["batch_id"] == bid for r in self.client.get("/api/batches/rejections").json),0)
        unselected = next(item for item in batch["items"] if item["decision"] == "合格")
        self.assertFalse(self.client.get("/api/records/"+unselected["record_id"]).json["review_required"])
        self.assertTrue(self.client.get("/api/records/"+unselected["record_id"]).json["review_available"])
        sampled_before_completion = self.client.post(f"/api/batches/{bid}/sample", json={"mode":"percent","value":10})
        self.assertEqual(sampled_before_completion.status_code, 200)
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
        self.assertEqual(batch["state"], "ready")

        data.update(revision=1, decision="合格")
        self.assertEqual(self.client.post(f"/api/records/{rid}/review", data=data).status_code, 200)
        self.assertEqual(len(self.client.get(f"/api/records/{rid}").json["reviews"]), 2)
        batch = self.client.post(f"/api/batches/{bid}/sample", json={"mode":"percent","value":100}).json
        self.assertEqual(batch["summary"]["sampled"], 100)
        self.assertEqual(batch["summary"]["sample_reviewed"], 0)
        self.assertEqual(self.client.delete(f"/api/records/{rid}").status_code, 405)
        self.assertEqual(self.client.get(f"/api/records/{rid}/image").status_code, 200)
        batch = self.client.get(f"/api/batches/{bid}").json
        self.assertEqual(batch["summary"]["deleted"], 0)
        self.assertEqual(batch["summary"]["completed"], 200)
        self.assertEqual(batch["summary"]["sampled"], 100)

        finished=self.client.post(f"/api/batches/{bid}/finish").json
        self.assertEqual(finished["state"],"completed")
        self.assertTrue(finished["completed_at"])
        self.assertEqual(self.client.post(f"/api/batches/{bid}/finish").json,finished)
        cases=[item for item in self.client.get("/api/batches/rejections").json if item["batch_id"]==bid]
        self.assertEqual(len(cases),100)
        self.assertEqual(self.client.post(f"/api/batches/{bid}/sample",json={"value":10}).status_code,409)
        self.assertEqual(self.client.post(f"/api/batches/{bid}/start").status_code,409)
        self.assertEqual(self.client.post(f"/api/batches/{bid}/retry").status_code,409)
        self.assertEqual(self.upload(bid,0).status_code,409)

    def test_whole_board_classifier_marks_batch_rejected(self):
        verdict = dict(key="poor_solder", label="虚焊/冷焊", score=.82, defect_score=.9,
                       is_defect=True, threshold=.5,
                       top=[dict(key="poor_solder", label="虚焊/冷焊", score=.82),
                            dict(key="solder_bridge", label="桥接", score=.1)],
                       model_version="smd-defect-test", model_sha256="sha")

        def classified(image, station="", inspection_context=None):
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

        def slow_detection(image, station="", inspection_context=None):
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

        self.assertEqual(self.client.post(f"/api/batches/{bid}/finish").json["state"],"completed")
        self.assertEqual(self.client.post(f"/api/batches/{bid}/retry").status_code,409)

    def test_finalization_freezes_records_and_archives_rejected_images_readonly(self):
        bid=self.create_batch(2)
        self.upload(bid,0,red=True)
        self.upload(bid,1,red=False)
        self.client.post(f"/api/batches/{bid}/start")
        batch=self.wait_ready(bid)
        good=batch["items"][1]["record_id"]
        bad=batch["items"][0]["record_id"]
        self.client.post(f"/api/batches/{bid}/sample",json={"mode":"count","value":1})
        self.assertEqual(self.client.post(f"/api/records/{good}/review",data={
            "reviewer":"QA","decision":"不合格","revision":0}).status_code,200)
        before=self.client.get(f"/api/records/{bad}").json
        finished=self.client.post(f"/api/batches/{bid}/finish").json
        self.assertEqual(finished["state"],"completed")
        cases=[r for r in self.client.get("/api/batches/rejections").json if r["batch_id"]==bid]
        self.assertEqual({c["record_id"] for c in cases},{bad,good})
        frozen=self.client.get(f"/api/records/{bad}").json
        self.assertTrue(frozen["readonly"])
        self.assertFalse(frozen["review_required"])
        self.assertEqual(self.client.post(f"/api/records/{bad}/review",data={
            "reviewer":"QA","decision":"合格","revision":0}).status_code,409)
        import platform_app
        project=platform_app.app.test_client().post("/api/projects",json={"name":"archive-lock"}).json["id"]
        self.assertEqual(self.client.post(f"/api/records/{bad}/corrections",json={
            "project_id":project,"reviewer":"QA","boxes":[],"revision":0}).status_code,409)
        case=self.client.get(f"/api/exceptions/{bad}").json
        self.assertTrue(case["readonly"])
        self.assertFalse(case["review_required"])
        reviewed=self.client.post(f"/api/exceptions/{bad}",data={
            "reviewer":"异常QA","decision":"合格","revision":case["revision"]})
        self.assertEqual(reviewed.status_code,405)
        self.assertEqual(self.client.get(f"/api/exceptions/{bad}").json,case)
        self.assertEqual(self.client.get(f"/api/records/{bad}").json,frozen)
        self.assertEqual(self.client.get(f"/api/batches/{bid}").json,finished)
        self.assertEqual(self.client.post(f"/api/exceptions/{bad}",data={
            "reviewer":"并发QA","decision":"不合格","revision":case["revision"]}).status_code,405)
        manifest=self.client.post(f"/api/batches/{bid}/report").json
        self.assertTrue(self.client.get(manifest["download_url"]).data.startswith(b"%PDF"))
        self.assertEqual(self.client.get(f"/api/records/{bad}/image").status_code,200)

    def test_rejected_library_only_includes_final_rejections_from_completed_batches(self):
        bid=self.create_batch(3)
        self.upload(bid,0,red=True)
        self.upload(bid,1,red=True)
        self.upload(bid,2,red=False)
        self.client.post(f"/api/batches/{bid}/start")
        batch=self.wait_ready(bid)
        corrected,bad,good=[item["record_id"] for item in batch["items"]]
        self.assertFalse([r for r in self.client.get("/api/batches/rejections").json if r["batch_id"]==bid])
        self.assertEqual(self.client.get(f"/api/exceptions/{bad}").status_code,404)
        self.assertEqual(self.client.post(f"/api/records/{corrected}/review",data={
            "reviewer":"QA","decision":"合格","revision":0}).status_code,200)
        self.assertEqual(self.client.post(f"/api/batches/{bid}/finish").status_code,200)
        cases=[r for r in self.client.get("/api/batches/rejections").json if r["batch_id"]==bid]
        self.assertEqual([r["record_id"] for r in cases],[bad])

        self.assertTrue(cases[0]["completed_at"])
        self.assertEqual(self.client.get(f"/api/exceptions/{corrected}").status_code,404)
        self.assertEqual(self.client.get(f"/api/exceptions/{good}").status_code,404)
        # An old snapshot of an automatic false alarm must stay out of the library.
        with app.db() as conn:
            record=self.client.get(f"/api/records/{corrected}").json
            record["ordinal"]=0
            import json
            conn.execute("INSERT INTO exception_cases(record_id,batch_id,snapshot) VALUES(?,?,?)",
                         (corrected,bid,json.dumps(record)))
        cases=[r for r in self.client.get("/api/batches/rejections").json if r["batch_id"]==bid]
        self.assertEqual([r["record_id"] for r in cases],[bad])

    def test_gallery_image_cannot_be_deleted(self):
        bid=self.create_batch(2)
        self.upload(bid,0,red=True)
        self.upload(bid,1,red=True)
        self.client.post(f"/api/batches/{bid}/start")
        batch=self.wait_ready(bid)
        bad=batch["items"][0]["record_id"]
        self.client.post(f"/api/batches/{bid}/finish")
        self.assertEqual(self.client.delete(f"/api/exceptions/{bad}").status_code,405)
        self.assertEqual(self.client.get(f"/api/exceptions/{bad}").status_code,200)
        self.assertEqual(self.client.get(f"/api/records/{bad}/image").status_code,200)
        self.assertEqual(self.client.get(f"/api/records/{bad}").json["decision"],"不合格")
        with app.db() as conn:
            app.capture_exceptions(conn,bid)
        self.client.post(f"/api/batches/{bid}/finish")
        cases=[r for r in self.client.get("/api/batches/rejections").json if r["batch_id"]==bid]
        self.assertEqual(len(cases),2)
        self.assertIn(bad,{item["record_id"] for item in cases})

    def test_single_record_delete_is_unavailable_before_and_after_archive(self):
        bid = self.create_batch(1)
        self.upload(bid, 0, red=True)
        self.client.post(f"/api/batches/{bid}/start")
        batch = self.wait_ready(bid)
        rid = batch["items"][0]["record_id"]
        self.assertEqual(self.client.delete(f"/api/records/{rid}").status_code, 405)
        self.assertEqual(self.client.get(f"/api/records/{rid}").status_code, 200)
        self.client.post(f"/api/batches/{bid}/finish")
        report = self.client.post(f"/api/batches/{bid}/report").json
        self.assertEqual(self.client.delete(f"/api/records/{rid}").status_code, 405)
        for endpoint in (f"/api/records/{rid}", f"/api/records/{rid}/image", f"/api/exceptions/{rid}", report["download_url"]):
            self.assertEqual(self.client.get(endpoint).status_code, 200)
        self.assertEqual(self.client.get(f"/api/batches/{bid}").json["summary"]["completed"], 1)

    def test_delete_importing_and_ready_batches(self):
        importing=self.create_batch(2)
        self.upload(importing,0)
        self.assertEqual(self.client.delete(f"/api/batches/{importing}").status_code,200)
        self.assertEqual(self.client.get(f"/api/batches/{importing}").status_code,404)
        self.assertEqual(self.upload(importing,1).status_code,404)
        ready=self.create_batch(1)
        self.upload(ready,0)
        self.client.post(f"/api/batches/{ready}/start")
        rid=self.wait_ready(ready)["items"][0]["record_id"]
        self.assertEqual(self.client.delete(f"/api/batches/{ready}").status_code,200)
        self.assertEqual(self.client.get(f"/api/records/{rid}").status_code,404)

    def test_delete_running_batch_does_not_recreate_records_after_inference(self):
        entered,release=threading.Event(),threading.Event()
        bid=self.create_batch(2)
        self.upload(bid,0)
        self.upload(bid,1)
        def blocked(image,station="",inspection_context=None):
            entered.set()
            if not release.wait(10):raise RuntimeError("test detector timed out")
            return controlled_detection(image,station,inspection_context)
        try:
            with patch.object(app,"detect",side_effect=blocked):
                self.client.post(f"/api/batches/{bid}/start")
                self.assertTrue(entered.wait(5))
                self.assertEqual(self.client.delete(f"/api/batches/{bid}").status_code,200)
                release.set()
                # A later queued batch proves that the worker has discarded the deleted result.
                later=self.create_batch(1)
                self.upload(later,0)
                self.client.post(f"/api/batches/{later}/start")
                self.wait_ready(later)
        finally:
            release.set()
        with app.db() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM records WHERE batch_id=?",(bid,)).fetchone()[0],0)
            self.assertEqual(conn.execute("SELECT count(*) FROM batch_items WHERE batch_id=?",(bid,)).fetchone()[0],0)
        self.assertEqual(self.client.get(f"/api/batches/{bid}").status_code,404)

    def test_quality_archive_delete_removes_batch_images_reports_and_gallery(self):
        bid=self.create_batch(2)
        self.upload(bid,0,red=True)
        self.upload(bid,1)
        self.client.post(f"/api/batches/{bid}/start")
        batch=self.wait_ready(bid)
        ids=[item["record_id"] for item in batch["items"]]
        self.client.post(f"/api/records/{ids[0]}/review",data={"reviewer":"QA","decision":"不合格","revision":0})
        self.client.post(f"/api/batches/{bid}/finish")
        report=self.client.post(f"/api/batches/{bid}/report").json
        self.assertEqual(self.client.delete(f"/api/batches/{bid}").status_code,200)
        self.assertEqual(self.client.get(f"/api/batches/{bid}").status_code,404)
        self.assertEqual(self.client.get(report["download_url"]).status_code,404)
        for rid in ids:
            self.assertEqual(self.client.get(f"/api/records/{rid}/image").status_code,404)
            self.assertEqual(self.client.get(f"/api/exceptions/{rid}").status_code,404)
        with app.db() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM reviews WHERE record_id IN (?,?)",ids).fetchone()[0],0)
            self.assertEqual(conn.execute("SELECT count(*) FROM batch_items WHERE batch_id=?",(bid,)).fetchone()[0],0)
        self.assertEqual(self.client.delete(f"/api/batches/{bid}").status_code,404)




    def test_batch_pins_one_reference_and_template_revision_for_all_images(self):
        bid=self.create_batch(2)
        self.upload(bid,0,red=True)
        self.upload(bid,1,red=True)
        path=Path(storage.name)/"pinned-reference.jpg"
        Image.new("RGB",(32,32),"white").save(path)
        reference=dict(id="reference",path=str(path),project_id="project",template_revision=1,
            sites=[dict(id="H01",x1=.1,y1=.1,x2=.2,y2=.2)])
        entered,release=threading.Event(),threading.Event()
        seen=[]
        def detect(image,station="",inspection_context=None):
            seen.append(inspection_context["reference"]["template_revision"])
            entered.set();release.wait(4)
            return controlled_detection(image,station)
        with patch.object(app.platform_runtime,"alignment_reference",return_value=reference) as resolver,patch.object(app,"detect",detect):
            response=self.client.post(f"/api/batches/{bid}/start")
            self.assertEqual(response.status_code,200,response.json)
            try:
                self.assertTrue(entered.wait(3))
                resolver.return_value={**reference,"template_revision":2}
            finally:
                release.set()
            batch=self.wait_ready(bid)
        self.assertEqual(seen,[1,1])
        self.assertEqual(batch["inspection_context"]["reference"]["template_revision"],1)

    def test_reinspection_uses_original_pixels_and_preserves_completed_archive(self):
        bid=self.create_batch(1)
        self.upload(bid,0,red=True)
        original=self.client.get(f"/api/batches/{bid}/items/0/image")
        self.assertEqual(original.status_code,200)
        self.client.post(f"/api/batches/{bid}/start")
        batch=self.wait_ready(bid)
        rid=batch["items"][0]["record_id"]
        self.client.post(f"/api/batches/{bid}/finish")
        import json
        wrong=io.BytesIO();Image.new("RGB",(80,50),"blue").save(wrong,"JPEG")
        with app.db() as conn:
            conn.execute("UPDATE records SET image=?,alignment=? WHERE id=?",(wrong.getvalue(),json.dumps({"reference_id":"old"}),rid))
        before=self.client.get(f"/api/batches/{bid}").json
        path=Path(storage.name)/"reinspection-reference.jpg"
        Image.new("RGB",(32,32),"white").save(path)
        reference=dict(id="current",path=str(path),project_id="project",template_revision=3,
            sites=[dict(id="H01",x1=.1,y1=.1,x2=.2,y2=.2)])
        observed=[]
        def detect(image,station="",inspection_context=None):
            observed.append((image.size,image.getpixel((0,0)),inspection_context["reference"]["id"]))
            return controlled_detection(image,station)
        with patch.object(app.platform_runtime,"alignment_reference",return_value=reference),patch.object(app,"detect",detect):
            response=self.client.post(f"/api/batches/{bid}/reinspect")
            self.assertEqual(response.status_code,200,response.json)
            new_id=response.json["id"]
            copied=self.wait_ready(new_id)
        self.assertNotEqual(new_id,bid)
        self.assertEqual(copied["source_batch_id"],bid)
        self.assertEqual(observed[0][0],(32,32))
        self.assertLess(observed[0][1][1],10)
        self.assertEqual(observed[0][2],"current")
        self.assertEqual(self.client.get(f"/api/batches/{bid}").json,before)
        self.assertEqual(self.client.get(f"/api/records/{rid}/image").data,wrong.getvalue())


class InspectionAlignmentTests(unittest.TestCase):
    def test_marked_thumbnail_uses_saved_corrected_image_without_inference_or_mutation(self):
        client=app.app.test_client()
        boxes=app.Detection([dict(key='missing_connection',label='疑似漏焊',score=.8,
                                 site_id='H01',rect=[.2,.3,.4,.5])],'v-saved','sha')
        boxes.image=Image.new('RGB',(320,180),(20,90,60))
        boxes.original_image=Image.new('RGB',(180,320),(90,20,60))
        boxes.alignment=dict(reference_id='saved-reference')
        boxes.inspection=dict(mode='solder_sites',checked_sites=26,total_sites=26,
            sites=[dict(id='H01',rect=[.2,.3,.4,.5])])
        with app.db() as conn:
            rid=app.save_record(conn,boxes.original_image,boxes,'QA','Saved result')
        before=client.get('/api/records/'+rid).json
        plain=client.get('/api/records/'+rid+'/image').data
        with patch.object(app,'detect',side_effect=AssertionError('saved images must not be inferred again')):
            marked=client.get('/api/records/'+rid+'/image?marked=1')
        self.assertEqual(marked.status_code,200)
        self.assertEqual(client.get('/api/records/'+rid+'/image?marked=1',headers={
            'If-None-Match':marked.headers['ETag']}).status_code,304)
        with Image.open(io.BytesIO(marked.data)) as image,Image.open(io.BytesIO(plain)) as saved:
            self.assertEqual(image.size,(320,180))
            self.assertGreater(image.getpixel((75,54))[0],saved.getpixel((75,54))[0]+40)
            # Interior pixels retain the corrected board rather than the portrait original.
            self.assertLess(abs(image.getpixel((250,100))[1]-saved.getpixel((250,100))[1]),5)
        original=client.get('/api/records/'+rid+'/image?original=1').data
        self.assertEqual(client.get('/api/records/'+rid+'/image?original=1&marked=1').data,original)
        self.assertEqual(client.get('/api/records/'+rid+'/image').data,plain)
        self.assertEqual(client.get('/api/records/'+rid).json,before)

    def test_process_parameters_auto_form_and_signature(self):
        client = app.app.test_client()
        with patch.object(app, "detect", return_value=app.Detection([], "v-test", "sha", classification={"is_defect":False})):
            response = client.post("/api/records",data={"image":(io.BytesIO(image_bytes(False)),"board.png"),
                "station":"PROCESS-QA","temp":"243.5","reflow_seconds":"52","solder_batch":"SP-01"})
        self.assertEqual(response.status_code,200,response.json)
        record=response.json
        self.assertTrue(record["batch"].startswith("巡检-"))
        self.assertEqual(record["process_parameters"]["reflow_seconds"],52)
        self.assertEqual(record["temp"],243.5)
        self.assertFalse(record["review_required"])
        self.assertTrue(record["review_available"])
        self.assertEqual(record["analysis"],[])
        signed=client.post("/api/records/"+record["id"]+"/review",data={
            "reviewer":"质检员","decision":"合格","revision":record["revision"]})
        self.assertEqual(signed.status_code,200)
        self.assertTrue(signed.json["signed"])
        self.assertEqual(signed.json["process_parameters"]["solder_batch"],"SP-01")
        invalid=client.post("/api/records",data={"image":(io.BytesIO(image_bytes(False)),"board.png"),
            "station":"PROCESS-QA","temp":"nan"})
        self.assertEqual(invalid.status_code,400)

    def test_preview_and_saved_record_use_corrected_pixels_and_coordinates(self):
        from unittest.mock import Mock
        import pcb_alignment
        source = Image.new("RGB", (80, 60), "red")
        reference = Path(storage.name) / "reference.jpg"
        Image.new("RGB", (100, 50), "blue").save(reference)
        aligned = pcb_alignment.align(source, reference, matrix=[[1,0,0],[0,1,0],[0,0,1]])
        width, height = aligned["width"], aligned["height"]
        result = Mock()
        result.names = {0: "solder_bridge"}
        result.pcb_issues = []
        result.boxes.data.cpu.return_value.tolist.return_value = [[width*.2,height*.3,width*.4,height*.5,.9,0]]
        client = app.app.test_client()
        def upload():
            buffer = io.BytesIO()
            source.save(buffer, "PNG")
            buffer.seek(0)
            return {"image":(buffer,"source.png"),"station":"QA","batch":"alignment"}
        with patch.object(app.platform_runtime, "alignment_reference", return_value={"id":"ref","path":str(reference)}), \
             patch.object(app.platform_runtime, "policy", return_value={"required_classes":[],"require_classifier":False}), \
             patch.object(app.platform_runtime, "predict", return_value=(result,"v-new","sha")) as predictor, \
             patch.object(app.defect_classifier, "predict", return_value=None), \
             patch.object(app.defect_classifier, "status", return_value={}), \
             patch("pcb_alignment.align", return_value=aligned):
            preview = client.post("/api/preview", data=upload())
            self.assertEqual(preview.status_code, 200)
            self.assertEqual(preview.json["alignment"]["reference_id"], "ref")
            self.assertEqual(predictor.call_args.args[0].size, (width,height))
            for actual, expected in zip(preview.json["boxes"][0]["rect"], [.2,.3,.4,.5]):
                self.assertAlmostEqual(actual, expected)
            saved = client.post("/api/records", data=upload()).json
            self.assertEqual(saved["boxes"], preview.json["boxes"])
            self.assertEqual(saved["alignment"], preview.json["alignment"])
            iid = saved["id"]
            with Image.open(io.BytesIO(client.get(f"/api/records/{iid}/image").data)) as picture:
                self.assertEqual(picture.size, (width,height))
            with Image.open(io.BytesIO(client.get(f"/api/records/{iid}/image?original=1").data)) as picture:
                self.assertEqual(picture.size, source.size)
            with patch("pcb_alignment.align", side_effect=ValueError("too few points")):
                self.assertEqual(client.post("/api/preview",data=upload()).status_code,400)


if __name__ == "__main__":
    unittest.main()
