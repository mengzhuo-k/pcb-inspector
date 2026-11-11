"""Load completed training versions and swap the active detector atomically."""
import hashlib
import sqlite3
import threading
from contextlib import closing
from datetime import datetime
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


class Detection(list):
    def __init__(self, boxes, model_version, model_sha256, classification=None, issues=None):
        super().__init__(boxes)
        self.model_version = model_version
        self.model_sha256 = model_sha256
        # Whole-board defect classification from the SMD DefDB model, if any.
        self.classification = classification
        self.issues = issues or []


class ModelRuntime:
    def __init__(self, training_store, fallback_path, fallback_sha256, factory=YOLO):
        self.store = Path(training_store).resolve()
        self.fallback_path = Path(fallback_path).resolve()
        self.fallback_sha256 = fallback_sha256
        self.factory = factory
        self.lock = threading.RLock()
        self.reload_lock = threading.Lock()
        self.active = None
        self.pending_version = None
        self.last_error = None
        self.failed = {}
        self.monitor_started = False
        self.refresh()

    def completed_versions(self):
        database = self.store / "library.sqlite"
        if not database.is_file():
            return []
        with closing(sqlite3.connect(database.as_uri()+"?mode=ro", uri=True, timeout=5)) as conn:
            return [row[0] for row in conn.execute(
                "SELECT id FROM jobs WHERE state='完成' ORDER BY id DESC")]

    def load(self, path, version, training_version, expected_hash=None):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_hash and digest != expected_hash:
            raise ValueError("权重校验失败")
        candidate = self.factory(str(path))
        names = candidate.names
        labels = list(names.values()) if isinstance(names, dict) else list(names)
        if candidate.task != "detect" or labels != ["missing_connection"]:
            raise ValueError("模型必须为单类别 missing_connection 目标检测模型")
        result = candidate.predict(Image.new("RGB", (64, 64)), imgsz=960, conf=.25, verbose=False)[0]
        if result.boxes is None:
            raise ValueError("模型未返回目标检测结果")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("权重在加载期间发生变化")
        return dict(model=candidate, version=version, training_version=training_version,
                    label="YOLO11n 漏焊 "+version, sha256=digest,
                    loaded_at=datetime.now().astimezone().isoformat(timespec="seconds"))

    def refresh(self):
        if not self.reload_lock.acquire(blocking=False):
            return
        problems = []
        try:
            try:
                versions = self.completed_versions()
            except (sqlite3.Error, OSError) as exc:
                versions = []
                problems.append("训练版本读取失败："+str(exc))
            for version in versions:
                if self.active and self.active["training_version"] is not None and version <= self.active["training_version"]:
                    break
                path = self.store / f"v{version}/fit/weights/best.pt"
                try:
                    stat = path.stat()
                    token = (version, stat.st_mtime_ns, stat.st_size)
                except OSError:
                    token = (version, None, None)
                if token in self.failed:
                    problems.append(self.failed[token])
                    continue
                self.pending_version = f"v{version}"
                try:
                    candidate = self.load(path, f"v{version}", version)
                except Exception as exc:
                    error = f"v{version} 加载失败："+str(exc)[:300]
                    self.failed[token] = error
                    problems.append(error)
                    continue
                with self.lock:
                    self.active = candidate
                break
            if self.active is None:
                candidate = self.load(self.fallback_path, "v7", None, self.fallback_sha256)
                with self.lock:
                    self.active = candidate
            self.last_error = problems[0] if problems else None
        finally:
            self.pending_version = None
            self.reload_lock.release()

    def predict(self, image, **kwargs):
        self.refresh()
        with self.lock:
            active = self.active
            result = active["model"].predict(image, **kwargs)[0]
            return result, active["label"], active["sha256"]

    def predict_many(self, images, **kwargs):
        self.refresh()
        with self.lock:
            active = self.active
            results = []
            for image in images:
                results.append(active["model"].predict(image, **kwargs)[0])
            return results, active["label"], active["sha256"]

    def status(self):
        with self.lock:
            info = {key:value for key,value in self.active.items() if key != "model"}
        info.update(pending_version=self.pending_version, last_error=self.last_error)
        return info

    def start_monitor(self):
        with self.lock:
            if self.monitor_started:
                return
            self.monitor_started = True

        def monitor():
            event = threading.Event()
            while True:
                self.refresh()
                event.wait(3)

        threading.Thread(target=monitor, daemon=True, name="pcb-model-monitor").start()


class PlatformRuntime:
    """Cache published detectors by project and station, including load status."""

    def __init__(self, database, factory=YOLO):
        self.database = Path(database)
        self.factory = factory
        self.lock = threading.RLock()
        self.models = {}
        self.errors = {}
        self.targets = {}
        self.monitor_started = False

    def _published(self, station="", project_id=None):
        if not self.database.is_file():
            return None
        with closing(sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True, timeout=5)) as conn:
            conn.row_factory = sqlite3.Row
            query = """SELECT m.*,d.station FROM deployments d
                JOIN models m ON m.id=d.model_id
                WHERE d.active=1 AND d.station=?"""
            params = [station]
            if project_id:
                query += " AND d.project_id=?"
                params.append(project_id)
            rows = conn.execute(query + " ORDER BY d.id DESC", params).fetchall()
            if not rows and station:
                rows = conn.execute(query + " ORDER BY d.id DESC", ["", *params[1:]]).fetchall()
            rows = [row for row in rows if json_loads(row["card"]).get("template", "det") == "det"]
            if len({row["project_id"] for row in rows}) > 1:
                raise ValueError("该工位绑定了多个项目，请为工位指定唯一项目")
            return rows[0] if rows else None

    def refresh(self, station="", project_id=None):
        key = (project_id, station)
        with self.lock:
            try:
                row = self._published(station, project_id)
                if row is None:
                    self.models.pop(key, None)
                    self.errors.pop(key, None)
                    return
                self.targets[key] = row["project_id"]
                active = self.models.get(key)
                if active and active["id"] == row["id"]:
                    self.errors.pop(key, None)
                    return
                path = Path(row["weights"])
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != row["sha256"]:
                    raise ValueError("权重校验失败")
                candidate = self.factory(str(path))
                labels = list(candidate.names.values())
                if candidate.task != "detect" or not labels:
                    raise ValueError("模型必须为目标检测模型")
                result = candidate.predict(Image.new("RGB", (64, 64)), imgsz=960,
                                           conf=.25, verbose=False)[0]
                if result.boxes is None:
                    raise ValueError("模型未返回目标检测结果")
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError("权重在加载期间发生变化")
                self.models[key] = dict(id=row["id"], model=candidate, version=row["version"],
                    station=station, project_id=row["project_id"], classes=labels,
                    sha256=digest, threshold=row["threshold"],
                    metrics=(json_loads(row["metrics"]) or {}) if "metrics" in row.keys() else {})
                self.errors.pop(key, None)
            except Exception as exc:
                self.errors[key] = str(exc)[:300]
            finally:
                pid = self.targets.get(key)
                if pid:
                    try:
                        with closing(sqlite3.connect(self.database, timeout=5)) as conn, conn:
                            active = self.models.get(key, {})
                            conn.execute("INSERT OR REPLACE INTO runtime_status VALUES(?,?,?,?,?)",
                                (pid, station, active.get("id"), self.errors.get(key),
                                 datetime.now().astimezone().isoformat(timespec="seconds")))
                    except sqlite3.Error:
                        pass

    def alignment_reference(self, station):
        row = self._published(station)
        if not self.database.is_file():
            return None
        with closing(sqlite3.connect(self.database, timeout=5)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN")
            if row is None:
                candidates = conn.execute("SELECT id AS project_id FROM projects WHERE station=?", (station,)).fetchall()
                if not candidates:
                    candidates = conn.execute("SELECT id AS project_id FROM projects").fetchall()
                if len(candidates) != 1:
                    if candidates:
                        raise ValueError("请为当前工位指定唯一的训练项目后再使用点位模板")
                    return None
                row = candidates[0]
            reference = conn.execute("""SELECT s.image_id AS id,s.path FROM alignment_sources s
                JOIN alignment_references r ON r.project_id=s.project_id AND r.image_id=s.image_id
                WHERE s.project_id=?""", (row["project_id"],)).fetchone()
            if reference is None:
                reference = conn.execute("""SELECT i.id,i.path FROM images i JOIN alignment_references r
                    ON r.image_id=i.id WHERE r.project_id=?""", (row["project_id"],)).fetchone()
            if reference is None:
                return None
            result = dict(reference, project_id=row["project_id"])
            template = conn.execute("""SELECT reference_id,revision,sites FROM solder_templates
                WHERE project_id=? AND reference_id=?""", (row["project_id"], reference["id"])).fetchone()
            if template:
                result.update(template_revision=template["revision"],sites=json_loads(template["sites"]))
            return result

    def policy(self, station):
        default = dict(required_classes=["missing_connection", "poor_solder_virtual",
            "poor_solder_cold", "solder_bridge", "pad_damage", "component_shift",
            "missing_component"], require_classifier=True)
        if not self.database.is_file():
            return default
        try:
            row = self._published(station)
            if row is None:
                return default
            with closing(sqlite3.connect(self.database, timeout=5)) as conn:
                policy = conn.execute("SELECT required_classes,require_classifier FROM station_policies WHERE project_id=? AND station=?",
                                      (row["project_id"], station)).fetchone()
            if policy:
                return dict(required_classes=json_loads(policy[0]) if isinstance(json_loads(policy[0]), list) else [],
                            require_classifier=bool(policy[1]))
        except (sqlite3.Error, ValueError):
            pass
        return default

    def predict(self, image, station="", project_id=None, **kwargs):
        key = (project_id, station)
        self.refresh(station, project_id)
        with self.lock:
            if key in self.errors:
                raise LookupError("工位模型加载失败：" + self.errors[key])
            active = self.models.get(key)
            if not active:
                raise LookupError("工位尚未发布检测模型")
            kwargs.setdefault("conf", active["threshold"])
            result = active["model"].predict(image, **kwargs)[0]
            metrics = active.get("metrics", {})
            per_class = metrics.get("per_class", {})
            policy = self.policy(station)
            if metrics.get("evaluation_status"):
                missing = [name for name in policy["required_classes"]
                           if not per_class.get(name, {}).get("support")]
                result.pcb_issues = ["必检类别缺少验证样本：" + "、".join(missing)] if missing else []
            test_status = metrics.get("test", {}).get("evaluation_status")
            if metrics.get("evaluation_status") and (metrics["evaluation_status"] == "unavailable"
                    or test_status not in ("valid", "partial") or not metrics.get("map50")):
                result.pcb_issues.append("新模型缺少有效的检测验证或独立测试结果，请人工复核")
            return result, active["version"], active["sha256"]

    def predict_many(self, images, station="", project_id=None, **kwargs):
        # Hold the same model through all regions so a model update cannot mix versions.
        with self.lock:
            first,version,digest = self.predict(images[0],station=station,project_id=project_id,**kwargs)
            active = self.models[(project_id,station)]
            kwargs.setdefault("conf", active["threshold"])
            results = [first]
            for image in images[1:]:
                result = active["model"].predict(image,**kwargs)[0]
                result.pcb_issues = list(getattr(first,"pcb_issues",[]))
                results.append(result)
            return results,version,digest

    def status(self, station="", project_id=None):
        self.refresh(station, project_id)
        key = (project_id, station)
        with self.lock:
            active = self.models.get(key)
            info = {k: v for k, v in (active or {}).items() if k != "model"}
            info.update(ok=bool(active) and key not in self.errors,
                        last_error=self.errors.get(key))
            return info

    def start_monitor(self):
        # Requests refresh their own station before inference.
        self.monitor_started = True


def json_loads(value):
    import json
    try:
        return json.loads(value) if value else {}
    except (TypeError, ValueError):
        return {}
