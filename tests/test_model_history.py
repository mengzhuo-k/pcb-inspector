"""Model selection and explicit deletion, using isolated storage and fake inference."""
import hashlib
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
            platform.deploy_updated_detector(conn, self.pid, mid, 'S1')

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
        self.assertEqual(response.json['current_model']['id'], new)
        self.assertEqual(response.json['mode'], 'automatic')
        self.assertEqual(runtime.status('S1', self.pid)['version'], 'v3')
        newer, _ = self.model('v4')
        with platform.db() as conn:
            self.assertTrue(platform.deploy_updated_detector(conn, self.pid, newer, 'S1'))
        response = self.client.post(self.selection, json={'follow_latest': False})
        self.assertEqual(response.json['current_model']['id'], newer)
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
        self.assertEqual(self.client.get(self.selection).json['mode'], 'automatic')
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

    def test_no_models_automatic_mode_and_version_skips_existing_file(self):
        response = self.client.post(self.selection, json={'follow_latest': True})
        self.assertEqual(response.json, {'mode': 'automatic', 'current_model': None})
        path = platform.MODELS / self.pid / 'v1' / 'best.pt'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'orphaned weight')
        with platform.db() as conn:
            self.assertEqual(platform.next_model_version(conn, self.pid), 'v2')
        self.assertEqual(path.read_bytes(), b'orphaned weight')


if __name__ == '__main__':
    unittest.main()
