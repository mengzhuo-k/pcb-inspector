import hashlib
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from contextlib import closing

from model_runtime import ModelRuntime


class FakeModel:
    gate = None
    entered = None

    def __init__(self, filename):
        self.marker = Path(filename).read_text()
        if self.marker == "corrupt":
            raise ValueError("corrupt checkpoint")
        self.names = {0:"other"} if self.marker == "wrong-class" else {0:"missing_connection"}
        self.task = "detect"

    def predict(self, image, **kwargs):
        if isinstance(image, str) and image == "hold":
            self.entered.set()
            self.gate.wait(5)
        return [SimpleNamespace(boxes=[], marker=self.marker)]


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.fallback = self.root/"fallback.pt"
        self.fallback.write_text("baseline")
        self.digest = hashlib.sha256(self.fallback.read_bytes()).hexdigest()
        with closing(sqlite3.connect(self.root/"library.sqlite")) as conn, conn:
            conn.execute("CREATE TABLE jobs(id INTEGER PRIMARY KEY,state TEXT)")

    def publish(self, version, marker, state="完成"):
        weights = self.root/f"v{version}/fit/weights/best.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_text(marker)
        with closing(sqlite3.connect(self.root/"library.sqlite")) as conn, conn:
            conn.execute("INSERT OR REPLACE INTO jobs VALUES (?,?)",(version,state))

    def runtime(self):
        return ModelRuntime(self.root,self.fallback,self.digest,factory=FakeModel)

    def test_only_completed_compatible_weights_activate_and_repair_is_retried(self):
        self.publish(1,"one")
        self.publish(2,"two")
        self.publish(3,"three","训练中")
        runtime=self.runtime()
        self.assertEqual(runtime.status()["version"],"v2")
        self.publish(3,"wrong-class")
        runtime.refresh()
        self.assertEqual(runtime.status()["version"],"v2")
        self.assertIn("v3",runtime.status()["last_error"])
        self.publish(3,"corrupt")
        runtime.refresh()
        self.assertEqual(runtime.predict("image")[0].marker,"two")
        self.publish(3,"repaired")
        runtime.refresh()
        self.assertEqual(runtime.status()["version"],"v3")
        self.assertIsNone(runtime.status()["last_error"])
        self.publish(4,"failed","失败")
        runtime.refresh()
        self.assertEqual(runtime.status()["version"],"v3")

    def test_inflight_prediction_keeps_its_model_and_provenance(self):
        runtime=self.runtime()
        FakeModel.entered=threading.Event()
        FakeModel.gate=threading.Event()
        output=[]
        predicting=threading.Thread(target=lambda:output.append(runtime.predict("hold")))
        predicting.start()
        self.assertTrue(FakeModel.entered.wait(2))
        self.publish(1,"next")
        updating=threading.Thread(target=runtime.refresh)
        updating.start()
        FakeModel.gate.set()
        predicting.join(5)
        updating.join(5)
        self.assertFalse(predicting.is_alive())
        self.assertFalse(updating.is_alive())
        self.assertEqual(output[0][0].marker,"baseline")
        self.assertEqual(output[0][2],self.digest)
        self.assertEqual(runtime.status()["version"],"v1")
        self.assertEqual(runtime.predict("next")[0].marker,"next")


class PlatformRuntimeTests(unittest.TestCase):
    def test_stations_load_independently_and_report_corrupt_update(self):
        from model_runtime import PlatformRuntime
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            database=root/"platform.sqlite"
            with closing(sqlite3.connect(database)) as conn, conn:
                conn.executescript("""
                    CREATE TABLE models(id TEXT,project_id TEXT,version TEXT,weights TEXT,sha256 TEXT,card TEXT,threshold REAL);
                    CREATE TABLE deployments(id INTEGER PRIMARY KEY,project_id TEXT,station TEXT,model_id TEXT,active INTEGER);
                    CREATE TABLE runtime_status(project_id TEXT,station TEXT,loaded_model TEXT,error TEXT,updated TEXT,PRIMARY KEY(project_id,station));
                """)
                for index, station in enumerate(("A","B"),1):
                    path=root/(station+".pt")
                    path.write_text(station)
                    conn.execute("INSERT INTO models VALUES(?,?,?,?,?,?,?)",
                        (station,"project",station,str(path),hashlib.sha256(path.read_bytes()).hexdigest(),'{"template":"det"}',.25))
                    conn.execute("INSERT INTO deployments VALUES(?,?,?,?,1)",(index,"project",station,station))
            runtime=PlatformRuntime(database,factory=FakeModel)
            self.assertEqual(runtime.predict("image",station="A")[0].marker,"A")
            self.assertEqual(runtime.predict("image",station="B")[0].marker,"B")
            self.assertEqual(runtime.predict("image",station="A")[0].marker,"A")
            with self.assertRaises(LookupError):
                runtime.predict("image",station="missing")
            with closing(sqlite3.connect(database)) as conn, conn:
                conn.execute("INSERT INTO deployments VALUES(3,'project','','A',1)")
            self.assertEqual(runtime.predict("image",station="SMT-01")[0].marker,"A")
            self.assertEqual(runtime.predict("image",station="B")[0].marker,"B")
            with closing(sqlite3.connect(database)) as conn, conn:
                conn.execute("UPDATE models SET id='bad',sha256='bad' WHERE id='B'")
                conn.execute("UPDATE deployments SET model_id='bad' WHERE station='B'")
            with self.assertRaises(LookupError):
                runtime.predict("image",station="B")
            self.assertFalse(runtime.status("B")["ok"])
            self.assertEqual(runtime.predict("image",station="A")[0].marker,"A")
