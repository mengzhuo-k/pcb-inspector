"""Model selection and explicit deletion, using isolated storage and fake inference."""
import hashlib
import io
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image

storage = tempfile.TemporaryDirectory()
os.environ['PCB_PLATFORM_DATA'] = storage.name
os.environ['PCB_AUTO_TRAIN'] = '0'

import platform_app as platform
from model_runtime import PlatformRuntime


class Detector:
    task = 'detect'
    names = dict(enumerate(key for key, _ in platform.DET_CLASSES))

    def __init__(self, path):
        self.path = path

    def predict(self, image, **kwargs):
        return [SimpleNamespace(boxes=[])]


class ModelHistoryTests(unittest.TestCase):
    def setUp(self):
        self.client = platform.app.test_client()
        self.pid = self.client.post('/api/projects', json={'name': 'History', 'station': 'S1'}).json['id']
        self.selection = f'/api/projects/{self.pid}/model-selection'
        self.factory = patch('ultralytics.YOLO', Detector)
        self.factory.start()
        self.addCleanup(self.factory.stop)

    def model(self, version, template='det', path=None):
        mid = uuid.uuid4().hex
        path = path or platform.MODELS / self.pid / version / 'best.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('checkpoint-' + mid).encode())
        with platform.db() as conn:
            conn.execute('''INSERT INTO models(id,project_id,version,sha256,weights,card,metrics,created)
                VALUES(?,?,?,?,?,?,?,?)''', (mid, self.pid, version,
                hashlib.sha256(path.read_bytes()).hexdigest(), str(path),
                json.dumps({'template': template}), '{}', platform.now()))
        return mid, path

    def activate(self, mid):
        with platform.db() as conn:
            platform.deploy_updated_detector(conn, self.pid, mid, 'S1',automatic=False)

    def test_history_is_unique_and_manual_choice_persists_across_training_and_restart(self):
        old, _ = self.model('v1')
        current, _ = self.model('v2')
        self.activate(current)
        listing = self.client.get(f'/api/projects/{self.pid}/models').json
        self.assertEqual(len(listing), 2)
        self.assertEqual(set(listing[0]['active_stations']), {'', 'S1'})
        response = self.client.post(self.selection, json={'model_id': old})
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(response.json, {'mode': 'manual', 'current_model': {'id': old, 'version': 'v1'}})
        new, _ = self.model('v3')
        with platform.db() as conn:
            self.assertFalse(platform.deploy_updated_detector(conn, self.pid, new, 'S1'))
            self.assertEqual({r[0] for r in conn.execute(
                'SELECT model_id FROM deployments WHERE project_id=? AND active=1', (self.pid,))}, {old})
        platform.initialize()
        self.assertEqual(self.client.get(self.selection).json['current_model']['id'], old)
        self.assertEqual(len(self.client.get(f'/api/projects/{self.pid}/models').json), 3)
        self.assertEqual(self.client.get(f'/api/projects/{self.pid}/model-update').json['selection']['mode'], 'manual')
        runtime = PlatformRuntime(platform.DB, factory=Detector)
        self.assertEqual(runtime.status('S1', self.pid)['version'], 'v1')
        response = self.client.post(self.selection, json={'follow_latest': True})
        self.assertEqual(response.json['current_model']['id'], old)
        self.assertEqual(response.json['mode'], 'manual')
        self.assertEqual(runtime.status('S1', self.pid)['version'], 'v1')
        # Only explicit selection can change the default; a legacy toggle cannot choose latest.
        self.client.post(self.selection,json={'model_id':new})
        self.assertEqual(runtime.status('S1', self.pid)['version'], 'v3')
        newer, _ = self.model('v4')
        with platform.db() as conn:
            self.assertFalse(platform.deploy_updated_detector(conn, self.pid, newer, 'S1'))
        response = self.client.post(self.selection, json={'follow_latest': False})
        self.assertEqual(response.json['current_model']['id'], new)
        self.assertEqual(response.json['mode'], 'manual')

    def test_switch_updates_all_project_stations_and_default(self):
        old, _ = self.model('v1')
        current, _ = self.model('v2')
        self.activate(current)
        with platform.db() as conn:
            conn.execute('INSERT INTO deployments(project_id,station,model_id,active) VALUES(?,?,?,1)',
                         (self.pid, 'extra-station', current))
        self.assertEqual(self.client.post(self.selection, json={'model_id': old}).status_code, 200)
        with platform.db() as conn:
            active = conn.execute('SELECT station,model_id FROM deployments WHERE project_id=? AND active=1', (self.pid,)).fetchall()
        self.assertEqual({r['station'] for r in active}, {'', 'S1', 'extra-station'})
        self.assertEqual({r['model_id'] for r in active}, {old})

    def test_bad_weights_missing_files_and_wrong_task_keep_current_model(self):
        current, _ = self.model('v1')
        self.activate(current)
        bad, path = self.model('v2')
        path.write_bytes(b'tampered')
        self.assertEqual(self.client.post(self.selection, json={'model_id': bad}).status_code, 400)
        path.unlink()
        self.assertEqual(self.client.post(self.selection, json={'model_id': bad}).status_code, 400)
        cls, _ = self.model('v3', template='cls')
        self.assertEqual(self.client.post(self.selection, json={'model_id': cls}).status_code, 400)
        valid, _ = self.model('v4')
        with patch.object(Detector, 'task', 'classify'):
            self.assertEqual(self.client.post(self.selection, json={'model_id': valid}).status_code, 400)
        self.assertEqual(self.client.get(self.selection).json['current_model']['id'], current)
        self.assertEqual(self.client.get(self.selection).json['mode'], 'manual')
        other = self.client.post('/api/projects', json={'name': 'other'}).json['id']
        self.assertEqual(self.client.post(f'/api/projects/{other}/model-selection', json={'model_id': current}).status_code, 404)
        self.assertEqual(self.client.post(self.selection, json={'follow_latest': 'yes'}).status_code, 400)

    def test_delete_only_requested_inactive_version_and_keep_inference_history(self):
        current, current_path = self.model('v1')
        old, old_path = self.model('v2')
        self.activate(old)
        self.activate(current)
        with platform.db() as conn:
            conn.execute('INSERT INTO jobs(id,project_id,state,model_id,weights) VALUES(?,?,?,?,?)',
                         ('history-' + old, self.pid, 'done', old, str(old_path)))
        self.assertEqual(self.client.delete('/api/models/' + current).status_code, 409)
        response = self.client.delete('/api/models/' + old)
        self.assertEqual(response.status_code, 200, response.json)
        self.assertTrue(response.json['weights_removed'])
        self.assertFalse(old_path.exists())
        self.assertTrue(current_path.exists())
        self.assertEqual(self.client.get('/api/models/' + old).status_code, 404)
        self.assertEqual(self.client.get('/api/models/' + current).status_code, 200)
        with platform.db() as conn:
            self.assertEqual(conn.execute('SELECT state FROM jobs WHERE id=?', ('history-' + old,)).fetchone()[0], 'done')
            self.assertEqual(conn.execute('SELECT count(*) FROM deployments WHERE model_id=?', (old,)).fetchone()[0], 0)
            self.assertEqual(platform.next_model_version(conn, self.pid), 'v3')
            self.assertEqual(platform.next_model_version(conn, self.pid), 'v4')
        self.assertEqual(self.client.delete('/api/models/' + old).status_code, 404)

    def test_training_reference_prevents_deletion_and_external_weights_are_preserved(self):
        mid, path = self.model('v1')
        with platform.db() as conn:
            conn.execute('INSERT INTO jobs(id,project_id,state,config) VALUES(?,?,?,?)',
                         ('training-' + mid, self.pid, 'running', json.dumps({'model': str(path)})))
        self.assertEqual(self.client.delete('/api/models/' + mid).status_code, 409)
        self.assertTrue(path.exists())
        external, external_path = self.model('v2', path=Path(storage.name) / 'external' / self.pid / 'best.pt')
        self.assertEqual(self.client.delete('/api/models/' + external).status_code, 200)
        self.assertTrue(external_path.exists())

    def test_no_models_require_explicit_selection_and_version_skips_existing_file(self):
        self.assertEqual(self.client.get(self.selection).json,{'mode':'manual','current_model':None})
        response = self.client.post(self.selection, json={'follow_latest': True})
        self.assertEqual(response.status_code,400)
        path = platform.MODELS / self.pid / 'v1' / 'best.pt'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'orphaned weight')
        with platform.db() as conn:
            self.assertEqual(platform.next_model_version(conn, self.pid), 'v2')
        self.assertEqual(path.read_bytes(), b'orphaned weight')

    def test_legacy_follow_latest_settings_migrate_to_current_version_without_switching(self):
        current,_=self.model('v1')
        self.activate(current)
        newer,_=self.model('v2')
        with platform.db() as conn:
            conn.execute("UPDATE project_model_selection SET mode='automatic',model_id=NULL WHERE project_id=?",(self.pid,))
            self.assertFalse(platform.deploy_updated_detector(conn,self.pid,newer,'S1'))
        platform.initialize()
        self.assertEqual(self.client.get(self.selection).json,{
            'mode':'manual','current_model':{'id':current,'version':'v1'}})
        with platform.db() as conn:
            selection=conn.execute('SELECT mode,model_id FROM project_model_selection WHERE project_id=?',(self.pid,)).fetchone()
            self.assertEqual(tuple(selection),('manual',current))
        self.assertEqual(self.client.post(self.selection,json={'follow_latest':True}).json['current_model']['id'],current)

    def test_first_detector_is_default_and_subsequent_training_cannot_replace_it(self):
        first,_=self.model('v1')
        with platform.db() as conn:
            self.assertTrue(platform.deploy_updated_detector(conn,self.pid,first,'S1'))
        second,_=self.model('v2')
        with platform.db() as conn:
            self.assertFalse(platform.deploy_updated_detector(conn,self.pid,second,'S1'))
        self.assertEqual(self.client.get(self.selection).json['current_model']['id'],first)
        self.assertEqual(len(self.client.get(f'/api/projects/{self.pid}/models').json),2)

    def dataset(self):
        """A small real, frozen dataset with independent positive images."""
        for index in range(4):
            stream=io.BytesIO()
            Image.new('RGB',(64,64),(index*50,40,30)).save(stream,'PNG')
            stream.seek(0)
            result=self.client.post(f'/api/projects/{self.pid}/images',data={
                'files':(stream,f'board-{index}.png'),'board_serial':f'board-{index}',
            },content_type='multipart/form-data')
            iid=result.json['saved'][0]
            saved=self.client.post(f'/api/images/{iid}/labels',json={'boxes':[
                {'class':'missing_connection','x1':.1,'y1':.1,'x2':.5,'y2':.5}]})
            self.assertEqual(saved.status_code,200,saved.json)
        result=self.client.post(f'/api/projects/{self.pid}/datasets',json={
            'template':'det','split_policy':'board','val_ratio':.25,'test_ratio':.25})
        self.assertEqual(result.status_code,200,result.json)
        return result.json['id']

    def test_select_training_base_is_persistent_separate_and_protected(self):
        base,_=self.model('v1')
        current,_=self.model('v2')
        self.activate(current)
        endpoint=f'/api/projects/{self.pid}/training-base'
        with patch.object(platform.QUEUE,'put') as queued:
            result=self.client.post(endpoint,json={'model_id':base})
            self.assertEqual(result.json['model']['id'],base)
            queued.assert_not_called()
        self.assertEqual(self.client.get(self.selection).json['current_model']['id'],current)
        platform.initialize()
        self.assertEqual(self.client.get(endpoint).json['model']['id'],base)
        listing=self.client.get(f'/api/projects/{self.pid}/models').json
        self.assertEqual([row['id'] for row in listing if row['is_training_base']],[base])
        self.assertEqual(self.client.delete('/api/models/'+base).status_code,409)
        self.assertEqual(self.client.post(endpoint,json={'model_id':None}).json,{'model':None})
        self.assertEqual(self.client.delete('/api/models/'+base).status_code,200)

    def test_improvement_queues_selected_weights_and_saves_new_version_with_origin(self):
        base,path=self.model('v1')
        original=path.read_bytes()
        current,_=self.model('v2')
        self.activate(current)
        self.client.post(self.selection,json={'model_id':current})
        endpoint=f'/api/projects/{self.pid}/training-base'
        self.client.post(endpoint,json={'model_id':base})
        did=self.dataset()
        with patch.object(platform.QUEUE,'put') as queued:
            response=self.client.post(f'/api/datasets/{did}/train',json={})
            self.assertEqual(response.status_code,200,response.json)
            jid=response.json['id']
            queued.assert_called_once_with(jid)
        config=self.client.get('/api/jobs/'+jid).json['config']
        self.assertEqual(config['base_model_id'],base)
        self.assertEqual(config['model'],str(path))
        self.assertEqual(config['base_model_sha256'],hashlib.sha256(original).hexdigest())
        # A queued job keeps its own parent even if the next training base changes.
        self.client.post(endpoint,json={'model_id':current})
        self.assertEqual(self.client.delete('/api/models/'+base).status_code,409)

        def fit(command,**kwargs):
            job_dir=Path(command[2])
            running=json.loads(Path(command[3]).read_text(encoding='utf-8'))
            self.assertEqual(Path(running['model']).name,'initial.pt')
            self.assertEqual(Path(running['model']).read_bytes(),original)
            (job_dir/'best.pt').write_bytes(original+b'-improved')
            (job_dir/'metrics.json').write_text(json.dumps({'metrics':{'map50':.8},'threshold':.25}))
            return SimpleNamespace(wait=lambda:0)

        with patch.object(platform.subprocess,'Popen',side_effect=fit):
            platform.run_job(jid)
        job=self.client.get('/api/jobs/'+jid).json
        self.assertEqual(job['state'],'done',job.get('error'))
        child=self.client.get('/api/models/'+job['model_id']).json
        self.assertEqual(child['version'],'v3')
        self.assertEqual(child['card']['trained_from'],{
            'id':base,'version':'v1','sha256':hashlib.sha256(original).hexdigest()})
        self.assertEqual(path.read_bytes(),original)
        self.assertNotEqual(Path(child['weights']).read_bytes(),original)
        self.assertEqual(self.client.get(self.selection).json['current_model']['id'],current)

    def test_automatic_updates_use_training_base_and_reset_restores_initial_model(self):
        base,_=self.model('v1')
        endpoint=f'/api/projects/{self.pid}/training-base'
        self.client.post(endpoint,json={'model_id':base})
        self.dataset()
        with patch.object(platform,'AUTO_TRAIN',True),patch.object(platform.QUEUE,'put') as queued:
            self.client.post(f'/api/projects/{self.pid}/model-update',json={})
            with platform.db() as conn:
                conn.execute('UPDATE model_updates SET changed=0 WHERE project_id=?',(self.pid,))
            platform.prepare_model_update(self.pid)
            queued.assert_called_once()
            status=self.client.get(f'/api/projects/{self.pid}/model-update').json
        config=self.client.get('/api/jobs/'+status['job_id']).json['config']
        self.assertEqual(config['base_model_id'],base)
        self.assertEqual(status['training_base']['model']['id'],base)
        self.client.post(endpoint,json={'model_id':None})
        did=self.client.get('/api/jobs/'+status['job_id']).json['dataset_id']
        with patch.object(platform.QUEUE,'put'):
            result=self.client.post(f'/api/datasets/{did}/train',json={})
        reset_config=self.client.get('/api/jobs/'+result.json['id']).json['config']
        self.assertNotIn('base_model_id',reset_config)
        self.assertEqual(reset_config['model'],platform.TEMPLATES['det']['model'])

    def test_invalid_base_and_tampered_queued_parent_never_fall_back_to_initial(self):
        base,path=self.model('v1')
        endpoint=f'/api/projects/{self.pid}/training-base'
        other=self.client.post('/api/projects',json={'name':'other'}).json['id']
        self.assertEqual(self.client.post(f'/api/projects/{other}/training-base',json={'model_id':base}).status_code,404)
        self.assertEqual(self.client.post(endpoint,json={}).status_code,400)
        self.assertEqual(self.client.post(endpoint,json={'model_id':123}).status_code,400)
        self.client.post(endpoint,json={'model_id':base})
        did=self.dataset()
        with patch.object(platform.QUEUE,'put'):
            jid=self.client.post(f'/api/datasets/{did}/train',json={}).json['id']
        path.write_bytes(b'corrupted-parent')
        with patch.object(platform.subprocess,'Popen') as process:
            platform.run_job(jid)
            process.assert_not_called()
        job=self.client.get('/api/jobs/'+jid).json
        self.assertEqual(job['state'],'failed')
        self.assertIn('校验失败',job['error'])
        self.assertEqual(self.client.post(f'/api/datasets/{did}/train',json={}).status_code,400)


if __name__ == '__main__':
    unittest.main()
