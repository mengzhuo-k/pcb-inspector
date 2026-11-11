import hashlib
import io
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from PIL import Image

storage=tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["PCB_PLATFORM_DATA"]=storage.name
os.environ["PCB_AUTO_TRAIN"]="0"

import platform_app


def picture_bytes(color="red",format="PNG"):
    stream=io.BytesIO()
    Image.new("RGB",(64,64),color).save(stream,format,**({"quality":94} if format=="JPEG" else {}))
    return stream.getvalue()


class DeduplicationTests(unittest.TestCase):
    def setUp(self):
        self.client=platform_app.app.test_client()
        self.pid=self.client.post("/api/projects",json={"name":"Dedup"}).json["id"]

    def upload(self,content=None,name="board.png",**fields):
        return self.client.post(f"/api/projects/{self.pid}/images",data={
            "files":(io.BytesIO(content or picture_bytes()),name),**fields}).json

    def count(self):
        return self.client.get(f"/api/projects/{self.pid}").json["images"]

    def test_repeated_files_in_one_upload_and_renamed_files_are_skipped(self):
        result=self.client.post(f"/api/projects/{self.pid}/images",data={"files":[
            (io.BytesIO(picture_bytes()),"first.png"),(io.BytesIO(picture_bytes()),"renamed.png")]}).json
        self.assertEqual(len(result["saved"]),1)
        self.assertEqual(result["duplicate_count"],1)
        with patch.object(platform_app,"mark_model_update") as changed:
            repeated=self.upload(name="another.png",batch="different",board_serial="different")
            self.assertEqual(repeated["saved"],[])
            self.assertEqual(repeated["duplicates"][0]["id"],result["saved"][0])
            changed.assert_not_called()
        self.assertEqual(self.count(),1)
        self.assertEqual(len(list((platform_app.IMAGES/self.pid).glob("*.jpg"))),1)

    def test_equal_pixels_in_different_formats_and_same_name_with_different_pixels(self):
        first=self.upload()["saved"][0]
        self.assertEqual(self.upload(picture_bytes(format="BMP"),"board.bmp")["duplicates"][0]["id"],first)
        self.assertEqual(len(self.upload(picture_bytes("blue"),"board.png")["saved"]),1)
        self.assertEqual(self.count(),2)

    def test_auto_and_direct_imports_share_existing_annotations_across_stations(self):
        iid=self.upload()["saved"][0]
        self.client.post(f"/api/images/{iid}/labels",json={"boxes":[
            {"class":"solder_bridge","x1":.1,"y1":.1,"x2":.3,"y2":.3}],"annotator":"QA"})
        before=self.client.get(f"/api/images/{iid}").json
        with platform_app.db() as conn:
            conn.execute("INSERT OR REPLACE INTO platform_settings VALUES('inspection_project',?)",(self.pid,))
        for key,station in (("source-a","S1"),("source-b","S2")):
            self.assertEqual(platform_app.import_inspection_image(key,picture_bytes(format="JPEG"),"camera.jpg",station),iid)
        self.assertEqual(self.client.get(f"/api/images/{iid}").json,before)
        self.assertEqual(self.count(),1)

    def test_direct_upload_after_auto_import_is_skipped(self):
        with platform_app.db() as conn:
            conn.execute("INSERT OR REPLACE INTO platform_settings VALUES('inspection_project',?)",(self.pid,))
        iid=platform_app.import_inspection_image("auto-"+self.pid,picture_bytes(format="JPEG"),"auto.jpg")
        result=self.upload()
        self.assertEqual(result["saved"],[])
        self.assertEqual(result["duplicates"][0]["id"],iid)

    def test_alignment_does_not_lose_original_identity(self):
        import pcb_alignment
        iid=self.upload()["saved"][0]
        row=self.client.get(f"/api/images/{iid}").json
        aligned=pcb_alignment.align(row["path"],row["path"],matrix=[[.8,0,.1],[0,.8,.1],[0,0,1]])
        with platform_app.db() as conn:
            platform_app.persist_alignment(conn,row,"reference",aligned)
        self.assertEqual(self.upload()["duplicates"][0]["id"],iid)
        self.assertEqual(self.count(),1)

    def test_existing_aligned_library_is_indexed_from_original_files(self):
        iid=self.upload()["saved"][0]
        original=platform_app.IMAGES/self.pid/(iid+".jpg")
        aligned=original.with_name(iid+"-aligned.jpg")
        aligned.write_bytes(picture_bytes("blue","JPEG"))
        with platform_app.db() as conn:
            conn.execute("UPDATE images SET path=?,sha256=? WHERE id=?",(str(aligned),hashlib.sha256(aligned.read_bytes()).hexdigest(),iid))
            conn.execute("DELETE FROM image_fingerprints WHERE image_id=?",(iid,))
        platform_app.index_existing_images()
        self.assertEqual(self.upload()["duplicates"][0]["id"],iid)

    def test_deleted_images_can_be_explicitly_imported_again(self):
        iid=self.upload()["saved"][0]
        self.assertEqual(self.client.delete(f"/api/images/{iid}").status_code,200)
        replacement=self.upload()["saved"][0]
        self.assertNotEqual(replacement,iid)
        self.assertEqual(self.count(),1)

    def test_concurrent_uploads_only_create_one_image(self):
        def upload(index):
            with platform_app.app.test_client() as client:
                return client.post(f"/api/projects/{self.pid}/images",data={
                    "files":(io.BytesIO(picture_bytes()),f"{index}.png")}).json
        with ThreadPoolExecutor(max_workers=6) as pool:
            results=list(pool.map(upload,range(6)))
        self.assertEqual(sum(len(row["saved"]) for row in results),1)
        self.assertEqual(sum(row["duplicate_count"] for row in results),5)
        self.assertEqual(self.count(),1)

    def test_feedback_reuses_existing_training_image(self):
        iid=self.upload()["saved"][0]
        feedback=self.client.post(f"/api/projects/{self.pid}/feedback",data={
            "image":(io.BytesIO(picture_bytes()),"feedback.png")}).json["id"]
        result=self.client.post(f"/api/feedback/{feedback}/convert",json={}).json
        self.assertEqual(result["id"],iid)
        self.assertTrue(result["duplicate"])
        self.assertEqual(self.count(),1)


if __name__=="__main__":
    unittest.main()
