import io
import base64
import hashlib
import numpy as np
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image
from site_inspection import prepare_sites, collect_site_defects

storage = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["PCB_DATA"] = storage.name
os.environ["PCB_PLATFORM_DATA"] = str(Path(storage.name)/"platform")
os.environ["PCB_AUTO_TRAIN"] = "0"

import app
import pcb_alignment


def result(boxes, name="solder_bridge"):
    tensor = SimpleNamespace(tolist=lambda: boxes)
    return SimpleNamespace(names={0:name},boxes=SimpleNamespace(data=SimpleNamespace(cpu=lambda: tensor)),pcb_issues=[])


class SiteGeometryTests(unittest.TestCase):
    def test_context_crops_cover_template_and_reject_invisible_sites(self):
        sites=[dict(id="H01",x1=.2,y1=.2,x2=.22,y2=.24)]
        image=Image.new("RGB",(1000,600))
        prepared,regions=prepare_sites(image,sites,[[1,0,0],[0,1,0],[0,0,1]])
        self.assertEqual(prepared[0]["rect"],[.2,.2,.22,.24])
        self.assertEqual(regions[0],(0,0,1000,600))
        self.assertGreater(len(regions),1)
        with self.assertRaisesRegex(ValueError,"未覆盖"):
            prepare_sites(image,sites,[[1,0,.5],[0,1,0],[0,0,1]])
        with self.assertRaisesRegex(ValueError,"为空"):
            prepare_sites(image,[])

    def test_maps_crop_coordinates_and_discards_predictions_outside_sites(self):
        sites=[dict(id="H01",rect=[.12,.14,.16,.22])]
        results=[result([[20,20,60,60,.9,0],[100,120,140,160,.95,0],[0,0,200,200,.99,0]])]
        boxes=collect_site_defects(results,[(100,50,300,250)],(1000,500),sites,app.RULES,"v1")
        self.assertEqual(len(boxes),1)
        self.assertEqual(boxes[0]["site_id"],"H01")
        self.assertEqual(boxes[0]["rect"],sites[0]["rect"])
        self.assertEqual(boxes[0]["detected_rect"],[.12,.14,.16,.22])

    def test_bridge_can_span_adjacent_sites_and_crop_duplicates_merge(self):
        sites=[dict(id="H01",rect=[.1,.1,.2,.2]),dict(id="H02",rect=[.21,.1,.31,.2])]
        boxes=collect_site_defects([result([[100,100,310,200,.8,0]]),result([[100,100,310,200,.9,0]])],
            [(0,0,1000,1000)]*2,(1000,1000),sites,app.RULES,"v1")
        self.assertEqual({box["site_id"] for box in boxes},{"H01","H02"})
        self.assertEqual([box["score"] for box in boxes],[.9,.9])
        with self.assertRaisesRegex(RuntimeError,"不完整"):
            collect_site_defects([],[(0,0,1000,1000)],(1000,1000),sites,app.RULES,"v1")


    def test_real_quarter_turns_align_to_reference_orientation_and_site_coordinates(self):
        pixels=np.random.default_rng(17).integers(0,256,(400,600),dtype=np.uint8)
        reference=Image.fromarray(np.repeat(pixels[:,:,None],3,axis=2))
        path=Path(storage.name)/"rotation-reference.png"
        reference.save(path)
        for angle in (90,180):
            aligned=pcb_alignment.align(reference.rotate(angle,expand=True),path)
            image=Image.open(io.BytesIO(base64.b64decode(aligned["preview"].split(",")[1]))).convert("RGB")
            self.assertEqual(image.size,reference.size)
            point=np.asarray(aligned["matrix"]) @ [.25,.25,1]
            expected=[.75,.25] if angle==90 else [.75,.75]
            np.testing.assert_allclose(point[:2]/point[2],expected,atol=.005)
            import cv2
            actual_blur=cv2.GaussianBlur(np.asarray(image),(9,9),0).astype(float)
            expected_blur=cv2.GaussianBlur(np.asarray(reference),(9,9),0).astype(float)
            self.assertLess(np.abs(actual_blur-expected_blur).mean(),8)
            sites,regions=prepare_sites(image,[dict(id="H01",x1=.2,y1=.2,x2=.3,y2=.3)],aligned["matrix"])
            self.assertEqual(len(sites),1)
        with self.assertRaisesRegex(ValueError,"方向异常"):
            pcb_alignment.align(reference,path,matrix=[[-1,0,1],[0,1,0],[0,0,1]])
        with self.assertRaisesRegex(ValueError,"翻折"):
            pcb_alignment.align(reference,path,matrix=[[1,0,0],[0,1,0],[2,0,-.5]])



class SiteInspectionTests(unittest.TestCase):
    def setUp(self):
        self.client=app.app.test_client()
        options=patch.object(app.defect_classifier,"status",return_value={})
        options.start();self.addCleanup(options.stop)
        self.original=Image.new("RGB",(400,240),"green")
        self.reference=Path(storage.name)/"reference.jpg"
        self.original.save(self.reference)
        self.aligned=pcb_alignment.align(self.original,self.reference,matrix=[[1,0,0],[0,1,0],[0,0,1]])
        self.template=dict(id="reference",path=str(self.reference),project_id="project",template_revision=7,
            sites=[dict(id="H01",x1=.2,y1=.2,x2=.25,y2=.3),dict(id="H02",x1=.6,y1=.6,x2=.65,y2=.7)])

    def upload(self):
        stream=io.BytesIO()
        self.original.save(stream,"PNG")
        return dict(image=(io.BytesIO(stream.getvalue()),"board.png"),station="QA",live_preview="1")

    def predict(self,images,**kwargs):
        # A target-site defect and a high-confidence background false positive.
        predictions=[result([[80,48,100,72,.9,0],[300,10,330,30,.99,0]])]
        predictions += [result([]) for _ in images[1:]]
        return predictions,"v-sites","sha-sites"

    def test_preview_and_record_use_corrected_template_scope_without_board_classifier(self):
        with patch.object(app.platform_runtime,"alignment_reference",return_value=self.template), \
             patch("pcb_alignment.align",return_value=self.aligned), \
             patch.object(app.platform_runtime,"predict_many",side_effect=self.predict) as predictor, \
             patch.object(app.defect_classifier,"predict",side_effect=AssertionError("whole board classifier must not decide template sites")):
            response=self.client.post("/api/preview",data=self.upload())
            self.assertEqual(response.status_code,200,response.json)
            self.assertIsNone(response.json["classification"])
            self.assertEqual(response.json["inspection"]["checked_sites"],2)
            self.assertEqual(response.json["inspection"]["template_revision"],7)
            self.assertEqual([box["site_id"] for box in response.json["boxes"]],["H01"])
            self.assertEqual(predictor.call_args.kwargs["project_id"],"project")
            saved=self.client.post("/api/records",data=self.upload()).json
            self.assertEqual(saved["inspection"],response.json["inspection"])
            self.assertEqual(saved["decision"],"不合格")
            self.assertEqual(saved["model_version"],"v-sites")

    def test_opt_in_classifier_combines_board_verdict_with_template_detection(self):
        classification=dict(key="poor_solder",label="虚焊/冷焊",score=.9,defect_score=.95,is_defect=True,threshold=.5,top=[])
        with patch.object(app.platform_runtime,"alignment_reference",return_value=self.template), \
             patch("pcb_alignment.align",return_value=self.aligned), \
             patch.object(app.platform_runtime,"predict_many",side_effect=self.predict), \
             patch.object(app.defect_classifier,"status",return_value=dict(use_with_sites=True,allow_unaligned=True)), \
             patch.object(app.defect_classifier,"predict",return_value=classification):
            response=self.client.post("/api/preview",data=self.upload())
            self.assertEqual(response.status_code,200,response.json)
            self.assertEqual(response.json["classification"]["key"],"poor_solder")
            self.assertEqual(response.json["inspection"]["checked_sites"],2)

    def test_authoritative_board_classifier_controls_verdict_and_keeps_point_candidates(self):
        classification=dict(key="good_solder",label="未见明显缺陷",score=.9,defect_score=.1,is_defect=False,
            threshold=.5,top=[],authoritative=True,model_version="v-cls",model_sha256="sha-cls")
        with patch.object(app.platform_runtime,"alignment_reference",return_value=self.template), \
             patch("pcb_alignment.align",return_value=self.aligned), \
             patch.object(app.platform_runtime,"predict_many",side_effect=self.predict), \
             patch.object(app.defect_classifier,"status",return_value=dict(use_with_sites=True,allow_unaligned=True)), \
             patch.object(app.defect_classifier,"predict",return_value=classification):
            response=self.client.post("/api/records",data=self.upload())
            self.assertEqual(response.status_code,200,response.json)
            self.assertEqual(response.json["decision"],"合格")
            self.assertEqual(response.json["model_version"],"v-cls")
            self.assertEqual(response.json["classification"]["point_model_version"],"v-sites")
            self.assertEqual(len(response.json["boxes"]),1)

    def test_opt_in_multi_board_classifier_can_run_without_matching_reference(self):
        classification=dict(key="good_solder",label="未见明显缺陷",score=.9,defect_score=.1,is_defect=False,threshold=.5,top=[])
        with patch.object(app.platform_runtime,"alignment_reference",return_value=self.template), \
             patch("pcb_alignment.align",side_effect=ValueError("different board")), \
             patch.object(app.platform_runtime,"predict",return_value=(result([]),"v-det","sha")), \
             patch.object(app.defect_classifier,"status",return_value=dict(use_with_sites=True,allow_unaligned=True)), \
             patch.object(app.defect_classifier,"predict",return_value=classification):
            response=self.client.post("/api/preview",data=self.upload())
            self.assertEqual(response.status_code,200,response.json)
            self.assertIsNone(response.json["inspection"])
            self.assertEqual(response.json["classification"]["key"],"good_solder")
            self.assertTrue(any("未匹配示例图" in issue for issue in response.json["issues"]))

    def test_empty_template_or_alignment_failure_does_not_silently_run_whole_board(self):
        with patch.object(app.platform_runtime,"alignment_reference",return_value={**self.template,"sites":[]}), \
             patch("pcb_alignment.align",return_value=self.aligned), \
             patch.object(app.platform_runtime,"predict_many") as predictor:
            response=self.client.post("/api/preview",data=self.upload())
            self.assertEqual(response.status_code,400)
            predictor.assert_not_called()
        with patch.object(app.platform_runtime,"alignment_reference",return_value=self.template), \
             patch("pcb_alignment.align",side_effect=ValueError("not matched")):
            self.assertEqual(self.client.post("/api/preview",data=self.upload()).status_code,400)


    def test_pinned_template_does_not_reselect_project_and_changed_reference_is_rejected(self):
        reference={**self.template,"sha256":hashlib.sha256(self.reference.read_bytes()).hexdigest()}
        with patch.object(app.platform_runtime,"alignment_reference",side_effect=AssertionError("must use pinned reference")), \
             patch("pcb_alignment.align",return_value=self.aligned), \
             patch.object(app.platform_runtime,"predict_many",side_effect=self.predict):
            detected=app.detect(self.original,"QA",inspection_context={"reference":reference})
            self.assertEqual(detected.inspection["template_revision"],7)
        self.reference.write_bytes(b"changed")
        from werkzeug.exceptions import BadRequest
        with self.assertRaises(BadRequest):
            app.detect(self.original,"QA",inspection_context={"reference":reference})



if __name__ == "__main__":
    unittest.main()
