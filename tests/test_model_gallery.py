"""Model galleries use immutable dataset membership and annotations."""
import hashlib
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from PIL import Image

storage = tempfile.TemporaryDirectory()
os.environ['PCB_PLATFORM_DATA'] = storage.name
os.environ['PCB_AUTO_TRAIN'] = '0'
import platform_app as platform


class ModelGalleryTests(unittest.TestCase):
    def setUp(self):
        self.client = platform.app.test_client()
        self.pid = self.client.post('/api/projects',json={'name':'Gallery','station':'QA'}).json['id']
        self.did, self.mid, self.jid = (uuid.uuid4().hex for _ in range(3))
        with platform.db() as conn:
            conn.execute('INSERT INTO datasets(id,project_id,template) VALUES(?,?,?)',(self.did,self.pid,'det'))
            conn.execute('INSERT INTO jobs(id,project_id,dataset_id,state) VALUES(?,?,?,?)',(self.jid,self.pid,self.did,'done'))
            conn.execute('INSERT INTO models(id,project_id,job_id,version,card) VALUES(?,?,?,?,?)',
                         (self.mid,self.pid,self.jid,'v1',json.dumps({'template':'det','name':'Test model'})))
        self.ids = [self.add_image(split) for split in ('train','val','test')]
        self.extra = self.add_image(None)
        with platform.db() as conn:
            platform.freeze_dataset(conn,self.did)
        self.url = f'/api/models/{self.mid}/images'

    def add_image(self,split):
        iid = uuid.uuid4().hex
        path = platform.IMAGES / self.pid / (iid+'.jpg')
        path.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(32,32),'white').save(path,'JPEG')
        with platform.db() as conn:
            conn.execute('INSERT INTO images(id,project_id,filename,path,width,height,sha256,state,review_status,revision) VALUES(?,?,?,?,?,?,?,?,?,?)',
                (iid,self.pid,iid+'.jpg',str(path),32,32,hashlib.sha256(path.read_bytes()).hexdigest(),'labeled','approved',1))
            conn.execute('INSERT INTO labels(image_id,class,x1,y1,x2,y2,polygon) VALUES(?,?,?,?,?,?,?)',
                (iid,'missing_connection',.1,.1,.3,.3,'[]'))
            if split:
                conn.execute('INSERT INTO dataset_items VALUES(?,?,?)',(self.did,iid,split))
        return iid

    def test_gallery_contains_all_splits_and_only_model_dataset(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json['total'],3)
        self.assertEqual(response.json['splits'],{'train':1,'val':1,'test':1})
        self.assertEqual(response.json['model']['name'],'Test model')
        self.assertEqual({item['id'] for item in response.json['items']},set(self.ids))
        self.assertEqual(self.client.get(self.url+'/'+self.extra).status_code,404)
        self.assertEqual(self.client.get(self.url+'/'+self.extra+'/file').status_code,404)
        detail = self.client.get(self.url+'/'+self.ids[0]).json
        self.assertTrue(detail['readonly'])
        self.assertIsNone(detail['alignment'])
        self.assertEqual(detail['labels'][0]['class'],'missing_connection')

    def test_deleted_current_image_and_changed_labels_keep_training_snapshot(self):
        iid = self.ids[0]
        with platform.db() as conn:
            conn.execute('UPDATE labels SET class=? WHERE image_id=?',('solder_bridge',iid))
        self.assertEqual(self.client.get(self.url+'/'+iid).json['labels'][0]['class'],'missing_connection')
        self.assertEqual(self.client.delete('/api/images/'+iid).status_code,200)
        self.assertEqual(self.client.get('/api/images/'+iid).status_code,404)
        self.assertEqual(self.client.get(self.url).json['total'],3)
        with self.client.get(self.url+'/'+iid+'/file?w=160') as preview:
            self.assertEqual(preview.status_code,200)
            self.assertEqual(preview.mimetype,'image/jpeg')
        self.assertEqual(self.client.post(self.url+'/'+iid,json={}).status_code,405)

    def test_missing_or_changed_files_are_reported_without_using_current_images(self):
        iid = self.ids[0]
        with platform.db() as conn:
            path = Path(conn.execute('SELECT path FROM images WHERE id=?',(iid,)).fetchone()[0])
        path.write_bytes(b'changed')
        self.assertEqual(self.client.get(self.url+'/'+iid+'/file').status_code,409)
        path.unlink()
        self.assertEqual(self.client.get(self.url+'/'+iid+'/file').status_code,404)
        self.assertEqual(self.client.get(self.url).json['total'],3)

    def test_untraceable_model_has_empty_gallery(self):
        mid = uuid.uuid4().hex
        with platform.db() as conn:
            conn.execute('INSERT INTO models(id,project_id,version,card) VALUES(?,?,?,?)',(mid,self.pid,'v0','{}'))
        gallery = self.client.get(f'/api/models/{mid}/images')
        self.assertEqual(gallery.status_code,200)
        self.assertEqual(gallery.json['items'],[])
        self.assertTrue(gallery.json['message'])
        self.assertEqual(self.client.get('/api/models/missing/images').status_code,404)

    def test_gallery_is_not_limited_to_200_images(self):
        with platform.db() as conn:
            row = conn.execute('SELECT * FROM dataset_snapshots WHERE dataset_id=? LIMIT 1',(self.did,)).fetchone()
            for index in range(205):
                conn.execute('INSERT INTO dataset_snapshots VALUES(?,?,?,?,?)',
                    (self.did,'extra-'+str(index),'train',row['metadata'],row['labels']))
        self.assertEqual(self.client.get(self.url).json['total'],208)
