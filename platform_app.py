"""PCB defect model-training platform.

Implements the nine-stage loop: capture/upload -> image store -> labeling ->
review -> dataset version -> training -> evaluation -> registry -> deploy ->
monitoring/feedback.  Inspection runtime stays in app.py; this module owns the
dataset and model lifecycle.
"""
from __future__ import annotations

import io
import hashlib
import json
import zipfile
import os
import queue
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PCB_PLATFORM_DATA", ROOT / "data/platform")).resolve()
IMAGES = DATA / "images"
MODELS = DATA / "models"
RUNS = DATA / "runs"
LOGS = DATA / "logs"
DB = DATA / "platform.sqlite"

DET_CLASSES = [
    ("missing_connection", "漏焊/缺焊"),
    ("poor_solder_virtual", "虚焊"),
    ("poor_solder_cold", "冷焊"),
    ("solder_bridge", "桥接"),
    ("pad_damage", "焊盘损伤"),
    ("component_shift", "元件偏移"),
    ("missing_component", "缺件"),
]
CLS_CLASSES = [("good_solder", "合格")] + DET_CLASSES
ALL_LABELS = dict(CLS_CLASSES)
ALL_LABELS.update({
    "damaged_component": "焊盘/元件损伤",
    "excess_solder": "多锡",
    "good_solder": "未见明显缺陷",
    "misaligned_component": "元件偏移",
    "missing_component": "缺件",
    "poor_solder": "虚焊/冷焊",
    "solder_bridge": "桥接",
})
TEMPLATES = {
    "det": dict(label="多类别缺陷检测", classes=[key for key, _ in DET_CLASSES],
                model=str(ROOT / "training/yolo11n.pt"), imgsz=640,
                hint="在图上画缺陷框；无框图片作为负样本"),
    "cls": dict(label="整板缺陷分类", classes=[key for key, _ in CLS_CLASSES],
                model=str(ROOT / "training/yolo11n-cls.pt"), imgsz=224,
                hint="每张图标一个整板主类型"),
}

app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
QUEUE: "queue.Queue[str]" = queue.Queue()
PROCS: dict[str, subprocess.Popen] = {}
WORKER_STARTED = False
WORKER_LOCK = threading.Lock()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@contextmanager
def db():
    conn = sqlite3.connect(DB, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def json_value(value, default=None):
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def initialize() -> None:
    for folder in (DATA, IMAGES, MODELS, RUNS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB)) as conn, conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects(
            id TEXT PRIMARY KEY, name TEXT NOT NULL, board_model TEXT DEFAULT '',
            station TEXT DEFAULT '', note TEXT DEFAULT '', created TEXT);
        CREATE TABLE IF NOT EXISTS images(
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL, filename TEXT,
            path TEXT NOT NULL, width INTEGER, height INTEGER,
            batch TEXT DEFAULT '', board_serial TEXT DEFAULT '',
            camera TEXT DEFAULT '', lighting TEXT DEFAULT '',
            captured_at TEXT DEFAULT '', sha256 TEXT, state TEXT DEFAULT 'new',
            primary_class TEXT, review_status TEXT DEFAULT 'pending',
            reviewer TEXT DEFAULT '', reviewed_at TEXT DEFAULT '',
            note TEXT DEFAULT '', revision INTEGER DEFAULT 0, created TEXT);
        CREATE INDEX IF NOT EXISTS images_project ON images(project_id, created);
        CREATE TABLE IF NOT EXISTS labels(
            id INTEGER PRIMARY KEY AUTOINCREMENT, image_id TEXT NOT NULL,
            class TEXT NOT NULL, x1 REAL, y1 REAL, x2 REAL, y2 REAL,
            subtype TEXT DEFAULT '', annotator TEXT DEFAULT '', created TEXT);
        CREATE INDEX IF NOT EXISTS labels_image ON labels(image_id);
        CREATE TABLE IF NOT EXISTS datasets(
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL, version INTEGER,
            template TEXT NOT NULL, split_policy TEXT DEFAULT 'group',
            counts TEXT, warnings TEXT, created_by TEXT, created TEXT);
        CREATE TABLE IF NOT EXISTS dataset_items(
            dataset_id TEXT, image_id TEXT, split TEXT, PRIMARY KEY(dataset_id, image_id));
        CREATE TABLE IF NOT EXISTS jobs(
            id TEXT PRIMARY KEY, dataset_id TEXT, project_id TEXT, template TEXT,
            config TEXT, state TEXT, log TEXT, metrics TEXT, weights TEXT,
            model_id TEXT, error TEXT, created TEXT, started TEXT, finished TEXT);
        CREATE INDEX IF NOT EXISTS jobs_project ON jobs(project_id, created);
        CREATE TABLE IF NOT EXISTS models(
            id TEXT PRIMARY KEY, project_id TEXT, job_id TEXT, version TEXT,
            sha256 TEXT, metrics TEXT, threshold REAL, weights TEXT,
            card TEXT, published INTEGER DEFAULT 0, created TEXT);
        CREATE TABLE IF NOT EXISTS deployments(
            id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, station TEXT,
            model_id TEXT, active INTEGER DEFAULT 1, created TEXT);
        CREATE TABLE IF NOT EXISTS feedback(
            id TEXT PRIMARY KEY, project_id TEXT, path TEXT, predicted TEXT,
            note TEXT, state TEXT DEFAULT 'open', image_id TEXT, created TEXT);
        """)
    with db() as conn:
        conn.execute("UPDATE jobs SET state='failed', error='平台重启，任务中断' WHERE state='running'")
        # No review stage in the employee flow: labels saved before this change
        # are treated as stored samples.
        conn.execute("""UPDATE images SET review_status='approved',
            reviewer=CASE WHEN reviewer='' THEN '自动' ELSE reviewer END,
            reviewed_at=CASE WHEN reviewed_at='' THEN created ELSE reviewed_at END
            WHERE (state='labeled' OR id LIKE 'review-%') AND review_status='pending'""")
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS dataset_snapshots(
            dataset_id TEXT, image_id TEXT, split TEXT, metadata TEXT, labels TEXT,
            PRIMARY KEY(dataset_id,image_id));
        CREATE TABLE IF NOT EXISTS runtime_status(
            project_id TEXT, station TEXT, loaded_model TEXT, error TEXT, updated TEXT,
            PRIMARY KEY(project_id,station));
        CREATE TABLE IF NOT EXISTS station_policies(
            project_id TEXT, station TEXT, required_classes TEXT,
            require_classifier INTEGER DEFAULT 1,
            PRIMARY KEY(project_id,station));
        """)
        for dataset in conn.execute("SELECT id FROM datasets").fetchall():
            if not conn.execute("SELECT 1 FROM dataset_snapshots WHERE dataset_id=?", (dataset["id"],)).fetchone():
                freeze_dataset(conn, dataset["id"], legacy=True)
    start_worker()


def project(pid):
    with db() as conn:
        row = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    if not row:
        abort(404, "项目不存在")
    return row


def image(iid):
    with db() as conn:
        row = conn.execute("SELECT * FROM images WHERE id=?", (iid,)).fetchone()
        labels = conn.execute("SELECT * FROM labels WHERE image_id=? ORDER BY id", (iid,)).fetchall()
    if not row:
        abort(404, "图片不存在")
    result = dict(row)
    result["labels"] = [dict(item) for item in labels]
    result["url"] = f"api/images/{iid}/file"
    return result


def counts_for(pid):
    with db() as conn:
        row = conn.execute("""SELECT
            count(*) AS images,
            sum(CASE WHEN review_status='pending' THEN 1 ELSE 0 END) AS pending,
            sum(CASE WHEN review_status='approved' THEN 1 ELSE 0 END) AS approved,
            sum(CASE WHEN review_status='rejected' THEN 1 ELSE 0 END) AS rejected
            FROM images WHERE project_id=?""", (pid,)).fetchone()
        jobs = conn.execute("SELECT count(*) FROM jobs WHERE project_id=?", (pid,)).fetchone()[0]
        datasets = conn.execute("SELECT count(*) FROM datasets WHERE project_id=?", (pid,)).fetchone()[0]
        models = conn.execute("SELECT count(*) FROM models WHERE project_id=?", (pid,)).fetchone()[0]
    return dict(images=row["images"] or 0, pending=row["pending"] or 0,
                approved=row["approved"] or 0, rejected=row["rejected"] or 0,
                jobs=jobs, datasets=datasets, models=models)


def classes_for(template: str):
    return TEMPLATES[template]["classes"]


@app.get("/")
def home():
    return app.send_static_file("platform.html")


@app.get("/api/meta")
def meta():
    return jsonify(classes=[dict(key=key, label=label) for key, label in CLS_CLASSES],
                   det_classes=[dict(key=key, label=label) for key, label in DET_CLASSES],
                   templates=TEMPLATES)


# ---------------------------------------------------------------- projects

@app.get("/api/projects")
def projects():
    with db() as conn:
        rows = conn.execute("SELECT * FROM projects ORDER BY created DESC").fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item.update(counts_for(item["id"]))
        result.append(item)
    return jsonify(result)


@app.post("/api/projects")
def create_project():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip()
    if not name or len(name) > 80:
        abort(400, "请填写项目名称，最多 80 字")
    pid = uuid.uuid4().hex
    with db() as conn:
        conn.execute("INSERT INTO projects(id,name,board_model,station,note,created) VALUES(?,?,?,?,?,?)",
                     (pid, name, str(payload.get("board_model", ""))[:80],
                      str(payload.get("station", ""))[:80],
                      str(payload.get("note", ""))[:500], now()))
    return jsonify(id=pid)


@app.get("/api/projects/<pid>")
def project_detail(pid):
    row = project(pid)
    result = dict(row)
    result.update(counts_for(pid))
    with db() as conn:
        result["recent_images"] = [dict(item) for item in conn.execute(
            "SELECT id,filename,state,review_status,primary_class,created FROM images WHERE project_id=? ORDER BY created DESC LIMIT 12", (pid,))]
        result["datasets"] = [dict(item) for item in conn.execute(
            "SELECT id,version,template,counts,created FROM datasets WHERE project_id=? ORDER BY created DESC LIMIT 8", (pid,))]
        result["jobs"] = [dict(item) for item in conn.execute(
            "SELECT id,template,state,created FROM jobs WHERE project_id=? ORDER BY created DESC LIMIT 8", (pid,))]
        deployed = conn.execute("""SELECT d.model_id,d.station,m.version,m.metrics FROM deployments d
            JOIN models m ON m.id=d.model_id WHERE d.project_id=? AND d.active=1
            ORDER BY d.id DESC LIMIT 1""", (pid,)).fetchone()
    result["deployment"] = dict(deployed) if deployed else None
    return jsonify(result)


# ------------------------------------------------------------------- images

@app.post("/api/projects/<pid>/images")
def upload_images(pid):
    project(pid)
    files = request.files.getlist("files") or request.files.getlist("image")
    if not files:
        abort(400, "请选择图片")
    fields = dict(batch=request.form.get("batch", "")[:80],
                  board_serial=request.form.get("board_serial", "")[:80],
                  camera=request.form.get("camera", "")[:80],
                  lighting=request.form.get("lighting", "")[:120],
                  captured_at=request.form.get("captured_at", "")[:40],
                  note=request.form.get("note", "")[:500])
    primary = request.form.get("primary_class")
    if primary and primary not in ALL_LABELS:
        abort(400, "整板类型无效")
    saved = []
    for upload in files:
        try:
            with Image.open(upload.stream) as raw:
                if raw.width * raw.height > 24_000_000:
                    abort(400, f"{upload.filename} 超过 2400 万像素")
                img = ImageOps.exif_transpose(raw).convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            abort(400, f"无法读取图片：{upload.filename}")
        iid = uuid.uuid4().hex
        folder = IMAGES / pid
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{iid}.jpg"
        img.save(path, "JPEG", quality=94)
        import hashlib
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with db() as conn:
            conn.execute("""INSERT INTO images(id,project_id,filename,path,width,height,
                batch,board_serial,camera,lighting,captured_at,sha256,state,primary_class,
                review_status,note,created) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                         (iid, pid, (upload.filename or "image.jpg")[:200], str(path),
                          img.width, img.height, fields["batch"], fields["board_serial"],
                          fields["camera"], fields["lighting"], fields["captured_at"],
                          digest, "new", primary or None, "pending", fields["note"], now()))
        saved.append(iid)
    return jsonify(saved=saved)


@app.get("/api/projects/<pid>/summary")
def project_summary(pid):
    project(pid)
    with db() as conn:
        row = conn.execute("""SELECT count(*) AS total,
            sum(CASE WHEN state='new' THEN 1 ELSE 0 END) AS new,
            sum(CASE WHEN state='labeled' AND review_status='pending' THEN 1 ELSE 0 END) AS waiting_review,
            sum(CASE WHEN review_status='approved' THEN 1 ELSE 0 END) AS approved,
            sum(CASE WHEN review_status='rejected' THEN 1 ELSE 0 END) AS rejected
            FROM images WHERE project_id=?""", (pid,)).fetchone()
        feedback = conn.execute("SELECT count(*) FROM feedback WHERE project_id=? AND state='open'",
                                (pid,)).fetchone()[0]
    return jsonify(dict(total=row["total"] or 0, new=row["new"] or 0,
                        waiting_review=0,
                        approved=row["approved"] or 0, rejected=row["rejected"] or 0,
                        feedback_open=feedback))


@app.get("/api/projects/<pid>/images")
def list_images(pid):
    project(pid)
    clauses = ["project_id=?"]
    values = [pid]
    for key, column in (("state", "state"), ("review", "review_status"), ("class", "primary_class")):
        value = request.args.get(key)
        if value:
            clauses.append(f"{column}=?")
            values.append(value)
    needle = request.args.get("q", "").strip()
    if needle:
        clauses.append("(filename LIKE ? OR batch LIKE ? OR board_serial LIKE ?)")
        values += [f"%{needle}%"] * 3
    limit = min(max(request.args.get("limit", 200, type=int), 1), 500)
    rows = None
    with db() as conn:
        rows = conn.execute(f"""SELECT i.*, (SELECT count(*) FROM labels l WHERE l.image_id=i.id) AS label_count
            FROM images i WHERE {' AND '.join(clauses)} ORDER BY i.created DESC LIMIT ?""",
                            (*values, limit)).fetchall()
    return jsonify([dict(row) for row in rows])


@app.get("/api/images/<iid>")
def image_detail(iid):
    return jsonify(image(iid))


@app.delete("/api/images/<iid>")
def delete_image(iid):
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM images WHERE id=?", (iid,)).fetchone()
        if not row:
            abort(404, "图片不存在或已删除")
        frozen = conn.execute("SELECT 1 FROM dataset_snapshots WHERE image_id=? LIMIT 1", (iid,)).fetchone()
        path = Path(row["path"]).resolve()
        if not path.is_relative_to(IMAGES.resolve()):
            abort(400, "图片路径超出图库")
        conn.execute("DELETE FROM labels WHERE image_id=?", (iid,))
        conn.execute("DELETE FROM feedback WHERE image_id=?", (iid,))
        conn.execute("DELETE FROM images WHERE id=?", (iid,))
    if not frozen:
        path.unlink(missing_ok=True)
    return jsonify(ok=True, retained_in_versions=bool(frozen))


@app.get("/api/images/<iid>/file")
def image_file(iid):
    row = image(iid)
    path = Path(row["path"])
    if not path.is_file():
        abort(404, "图片文件不存在")
    width = request.args.get("w", type=int)
    if width and width > 0:
        with Image.open(path) as source:
            source = source.convert("RGB")
            source.thumbnail((min(width, 1600), min(width, 1600) * 3))
            buffer = io.BytesIO()
            source.save(buffer, "JPEG", quality=85)
        buffer.seek(0)
        return send_file(buffer, mimetype="image/jpeg")
    return send_file(path, mimetype="image/jpeg")


@app.post("/api/images/<iid>/labels")
def save_labels(iid):
    row = image(iid)
    payload = request.get_json(silent=True) or {}
    boxes = payload.get("boxes", [])
    if not isinstance(boxes, list) or len(boxes) > 500:
        abort(400, "标注框数量无效")
    for box in boxes:
        if not isinstance(box, dict):
            abort(400, "标注框格式无效")
        key = box.get("class")
        if key not in ALL_LABELS:
            abort(400, "缺陷类型无效")
        try:
            x1, y1, x2, y2 = (float(box[name]) for name in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError):
            abort(400, "标注坐标无效")
        if not all(0 <= value <= 1 for value in (x1, y1, x2, y2)) or x1 >= x2 or y1 >= y2:
            abort(400, "标注框面积无效")
    primary = payload.get("primary_class") or None
    if primary and primary not in dict(CLS_CLASSES):
        abort(400, "整板类型无效")
    annotator = str(payload.get("annotator", ""))[:80]
    note = str(payload.get("note", ""))[:500]
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if not conn.execute("SELECT 1 FROM images WHERE id=?", (iid,)).fetchone():
            abort(404, "图片已删除，请刷新图库")
        conn.execute("DELETE FROM labels WHERE image_id=?", (iid,))
        conn.executemany("""INSERT INTO labels(image_id,class,x1,y1,x2,y2,subtype,annotator,created)
            VALUES(?,?,?,?,?,?,?,?,?)""",
                         [(iid, box["class"], float(box["x1"]), float(box["y1"]),
                           float(box["x2"]), float(box["y2"]),
                           str(box.get("subtype", ""))[:40], annotator, now()) for box in boxes])
        conn.execute("""UPDATE images SET state='labeled',review_status='approved',
            reviewer='自动',reviewed_at=?,primary_class=?,note=?,revision=revision+1 WHERE id=?""",
                     (now(), primary, note, iid))
    return jsonify(image(iid))


def freeze_dataset(conn, did, legacy=False):
    items = conn.execute("""SELECT i.*,d.split FROM dataset_items d
        JOIN images i ON i.id=d.image_id WHERE d.dataset_id=?""", (did,)).fetchall()
    dataset = conn.execute("SELECT * FROM datasets WHERE id=?", (did,)).fetchone()
    counts = dict(train=0, val=0, test=0, boxes=0, classes={},
                  splits={split: dict(images=0, boards=[], normal=0, boxes=0, classes={})
                          for split in ("train", "val", "test")})
    warnings = []
    hashes = {}
    for item in items:
        metadata = dict(item)
        metadata["sha256"] = hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() if Path(item["path"]).is_file() else item["sha256"]
        hashes.setdefault(metadata["sha256"], set()).add(item["split"])
        labels = [dict(row) for row in conn.execute(
            "SELECT * FROM labels WHERE image_id=?", (item["id"],))]
        conn.execute("INSERT INTO dataset_snapshots VALUES(?,?,?,?,?)",
                     (did, item["id"], item["split"], json.dumps(metadata), json.dumps(labels)))
        split = item["split"]
        counts[split] += 1
        stats = counts["splits"].setdefault(split, dict(images=0, boards=[], normal=0, boxes=0, classes={}))
        stats["images"] += 1
        board = item["board_serial"] or item["batch"] or item["id"]
        if board not in stats["boards"]:
            stats["boards"].append(board)
        valid = [label for label in labels if label["class"] in classes_for("det")]
        stats["normal"] += int(not valid)
        stats["boxes"] += len(valid)
        counts["boxes"] += len(valid)
        for label in valid:
            key = label["class"]
            stats["classes"][key] = stats["classes"].get(key, 0) + 1
        key = item["primary_class"] or "unlabeled"
        counts["classes"][key] = counts["classes"].get(key, 0) + 1
    for split, stats in counts["splits"].items():
        stats["boards"] = len(stats["boards"])
        if dataset["template"] == "det" and not stats["boxes"]:
            warnings.append(f"{split} 没有缺陷框，无法评估缺陷检出能力")
    counts["duplicate_cross_split"] = sum(len(splits) > 1 for splits in hashes.values())
    if counts["duplicate_cross_split"]:
        warnings.append("相同图片跨越多个划分，评估将标记为不可用；请去重或重新分组")
    if legacy:
        warnings.append("历史版本于本次升级冻结当前标注，升级前标注无法追溯")
    counts["class_order"] = classes_for(dataset["template"])
    conn.execute("UPDATE datasets SET counts=?,warnings=? WHERE id=?",
                 (json.dumps(counts, ensure_ascii=False), json.dumps(warnings, ensure_ascii=False), did))
    return counts, warnings


def frozen_items(did):
    with db() as conn:
        rows = conn.execute("SELECT * FROM dataset_snapshots WHERE dataset_id=?", (did,)).fetchall()
    items = []
    for row in rows:
        item = json.loads(row["metadata"])
        item["labels"] = json.loads(row["labels"])
        if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("数据集图片已变化，请创建新版本")
        items.append(item)
    return items


def split_groups(rows, policy, seed, test_ratio, val_ratio):
    groups = {}
    for row in rows:
        if policy == "board":
            key = row["board_serial"] or row["batch"] or row["id"]
        elif policy == "batch":
            key = row["batch"] or row["board_serial"] or row["id"]
        else:
            key = row["id"]
        groups.setdefault(key, []).append(row["id"])
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    count = len(keys)
    test_n = round(count * test_ratio) if count >= 4 else 0
    val_n = round(count * val_ratio) if count >= 3 else 0
    test_n = max(1, test_n) if count >= 4 and test_ratio > 0 else 0
    val_n = max(1, val_n) if count - test_n >= 2 and val_ratio > 0 else 0
    if count - test_n - val_n < 1:
        test_n = val_n = 0
    assigned = {}
    for index, key in enumerate(keys):
        if index < test_n:
            split = "test"
        elif index < test_n + val_n:
            split = "val"
        else:
            split = "train"
        for iid in groups[key]:
            assigned[iid] = split
    return assigned, len(keys)


@app.post("/api/projects/<pid>/datasets")
def create_dataset(pid):
    project(pid)
    payload = request.get_json(silent=True) or {}
    template = payload.get("template", "det")
    if template not in TEMPLATES:
        abort(400, "模型模板无效")
    policy = payload.get("split_policy", "board")
    if policy not in ("board", "batch", "random"):
        abort(400, "划分方式无效")
    try:
        seed = int(payload.get("seed", 20261110))
        val_ratio = float(payload.get("val_ratio", 0.2))
        test_ratio = float(payload.get("test_ratio", 0.2))
    except (TypeError, ValueError):
        abort(400, "划分参数无效")
    if not (0 <= val_ratio < 0.5 and 0 <= test_ratio < 0.5):
        abort(400, "划分比例无效")
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute("""SELECT id,board_serial,batch,primary_class,sha256
            FROM images WHERE project_id=? AND review_status='approved'
            ORDER BY created""", (pid,)).fetchall()
        if not rows:
            abort(400, "没有已保存标注的图片，请先标注并保存")
        rows = [dict(row) for row in rows]
        if template == "cls":
            rows = [row for row in rows if row["primary_class"]]
            if not rows:
                abort(400, "分类模板需要为图片填写整板类型")
        assigned, group_count = split_groups(rows, policy, seed, test_ratio, val_ratio)
        version = (conn.execute("SELECT max(version) FROM datasets WHERE project_id=?", (pid,)).fetchone()[0] or 0) + 1
        did = uuid.uuid4().hex
        conn.execute("""INSERT INTO datasets(id,project_id,version,template,split_policy,counts,warnings,created_by,created)
            VALUES(?,?,?,?,?,'{}','[]',?,?)""",
                     (did, pid, version, template, policy,
                      str(payload.get("created_by", ""))[:80], now()))
        conn.executemany("INSERT INTO dataset_items(dataset_id,image_id,split) VALUES(?,?,?)",
                         [(did, iid, split) for iid, split in assigned.items()])
        counts, warnings = freeze_dataset(conn, did)
    return jsonify(id=did, version=version, counts=counts, warnings=warnings, groups=group_count)


@app.get("/api/projects/<pid>/datasets")
def list_datasets(pid):
    project(pid)
    with db() as conn:
        rows = conn.execute("SELECT * FROM datasets WHERE project_id=? ORDER BY created DESC", (pid,)).fetchall()
    return jsonify([dict(row) for row in rows])


@app.get("/api/datasets/<did>")
def dataset_detail(did):
    with db() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id=?", (did,)).fetchone()
        if not row:
            abort(404, "数据集不存在")
    result = dict(row)
    result["items"] = [dict(id=item["id"], filename=item["filename"],
        primary_class=item["primary_class"], board_serial=item["board_serial"],
        batch=item["batch"], split=item["split"], label_count=len(item["labels"]))
        for item in frozen_items(did)]
    return jsonify(result)


# ------------------------------------------------------------------ training

@app.post("/api/datasets/<did>/train")
def start_training(did):
    payload = request.get_json(silent=True) or {}
    with db() as conn:
        dataset = conn.execute("SELECT * FROM datasets WHERE id=?", (did,)).fetchone()
        if not dataset:
            abort(404, "数据集不存在")
        project_id = dataset["project_id"]
        template = dataset["template"]
        for split in ("train", "val"):
            count = conn.execute("SELECT count(*) FROM dataset_items WHERE dataset_id=? AND split=?",
                                 (did, split)).fetchone()[0]
            if not count:
                abort(400, f"数据集缺少 {split} 划分")
        if template == "det":
            stats = json_value(dataset["counts"], {}).get("splits", {})
            if not stats.get("train", {}).get("boxes"):
                abort(400, "训练集没有缺陷框，请补充标注并创建新版本")
        try:
            epochs = int(payload.get("epochs", 40))
            imgsz = int(payload.get("imgsz", TEMPLATES[template]["imgsz"]))
            batch = int(payload.get("batch", 16))
        except (TypeError, ValueError):
            abort(400, "训练参数无效")
        if not (1 <= epochs <= 500 and 64 <= imgsz <= 1280 and 1 <= batch <= 128):
            abort(400, "训练参数超出范围")
        config = dict(epochs=epochs, imgsz=imgsz, batch=batch,
                      device=str(payload.get("device", "cpu")),
                      model=str(payload.get("model") or TEMPLATES[template]["model"]),
                      classes=json_value(dataset["counts"], {}).get("class_order") or classes_for(template),
                      auto_publish=bool(payload.get("auto_publish", False)),
                      station=str(payload.get("station", ""))[:80])
        local_model = ROOT / "training" / Path(config["model"]).name
        if not Path(config["model"]).is_file() and local_model.is_file():
            config["model"] = str(local_model)
        jid = uuid.uuid4().hex
        conn.execute("""INSERT INTO jobs(id,dataset_id,project_id,template,config,state,created)
            VALUES(?,?,?,?,?,'queued',?)""",
                     (jid, did, project_id, template, json.dumps(config), now()))
    QUEUE.put(jid)
    return jsonify(id=jid)


@app.get("/api/jobs/<jid>")
def job_detail(jid):
    with db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
    if not row:
        abort(404, "训练任务不存在")
    result = dict(row)
    log = LOGS / f"{jid}.log"
    result["log_tail"] = tail(log, 120)
    result["metrics"] = json_value(result.get("metrics"))
    result["config"] = json_value(result.get("config"), {})
    return jsonify(result)


@app.get("/api/projects/<pid>/jobs")
def list_jobs(pid):
    project(pid)
    with db() as conn:
        rows = conn.execute("SELECT * FROM jobs WHERE project_id=? ORDER BY created DESC LIMIT 50", (pid,)).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["metrics"] = json_value(item.get("metrics"))
        result.append(item)
    return jsonify(result)


@app.post("/api/jobs/<jid>/cancel")
def cancel_job(jid):
    proc = PROCS.get(jid)
    if proc and proc.poll() is None:
        proc.terminate()
    with db() as conn:
        conn.execute("UPDATE jobs SET state='cancelled',finished=? WHERE id=? AND state IN ('queued','running')",
                     (now(), jid))
    return jsonify(ok=True)


# -------------------------------------------------------------------- models

@app.get("/api/projects/<pid>/models")
def list_models(pid):
    project(pid)
    with db() as conn:
        rows = conn.execute("""SELECT m.*, d.station AS deployed_station
            FROM models m LEFT JOIN deployments d ON d.model_id=m.id AND d.active=1
            WHERE m.project_id=? ORDER BY m.created DESC""", (pid,)).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["metrics"] = json_value(item.get("metrics"))
        item["card"] = json_value(item.get("card"))
        item["template"] = (item["card"] or {}).get("template")
        result.append(item)
    return jsonify(result)


@app.get("/api/models/<mid>")
def model_detail(mid):
    with db() as conn:
        row = conn.execute("SELECT * FROM models WHERE id=?", (mid,)).fetchone()
    if not row:
        abort(404, "模型不存在")
    result = dict(row)
    result["metrics"] = json_value(result.get("metrics"))
    result["card"] = json_value(result.get("card"))
    return jsonify(result)


@app.get("/api/models/<mid>/export")
def export_model(mid):
    row = model_detail(mid).get_json()
    path = Path(row["weights"])
    if not path.is_file():
        abort(404, "权重文件不存在")
    card = row.get("card") or {}
    classes = card.get("classes") or []
    labels = {key: ALL_LABELS.get(key, key) for key in classes}
    meta = dict(version=row["version"], template=card.get("template"),
                classes=classes, labels=labels, threshold=row.get("threshold"),
                metrics=row.get("metrics"), card=card, sha256=row.get("sha256"))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(path, "weights/best.pt")
        archive.writestr("model_card.json", json.dumps(meta, ensure_ascii=False, indent=2))
        archive.writestr("classes.txt", "\n".join(classes))
        archive.writestr("labels.json", json.dumps(labels, ensure_ascii=False, indent=2))
        readme = f"""PCB 焊接缺陷模型导出包

版本: {row['version']}
模板: {card.get('template') or 'det'}
类别: {', '.join(classes) or '未记录'}
阈值: {row.get('threshold')}
SHA256: {row.get('sha256')}

文件:
  weights/best.pt   YOLO 权重
  model_card.json   模型卡、指标、类别、阈值
  classes.txt       类别顺序
  labels.json       类别中文名

加载:
  from ultralytics import YOLO
  model = YOLO("weights/best.pt")
  result = model.predict("board.jpg", imgsz=640, conf=0.25)

指标:
  {json.dumps(row.get('metrics'), ensure_ascii=False)}

该权重由焊视训练平台从当前项目已保存的缺陷框训练生成。
"""
        archive.writestr("README.txt", readme)
    buffer.seek(0)
    return send_file(buffer, mimetype="application/zip", as_attachment=True,
                     download_name=f"pcb-{row['version']}.zip")


@app.get("/api/models/<mid>/file")
def model_file(mid):
    row = model_detail(mid)
    path = Path(row["weights"])
    if not path.is_file():
        abort(404, "权重文件不存在")
    return send_file(path, as_attachment=True, download_name=f"pcb-{row['version']}.pt")


@app.post("/api/models/<mid>/publish")
def publish_model(mid):
    payload = request.get_json(silent=True) or {}
    with db() as conn:
        row = conn.execute("SELECT * FROM models WHERE id=?", (mid,)).fetchone()
        if not row:
            abort(404, "模型不存在")
        card = json_value(row["card"], {})
        if card.get("template", "det") != "det":
            abort(400, "识别工位目前支持目标检测模型")
        station = str(payload.get("station", "")).strip()[:80]
        if not station:
            abort(400, "请指定工位")
        metrics = json_value(row["metrics"], {})
        if metrics.get("evaluation_status") == "unavailable":
            abort(400, "模型缺少有效验证结果，请补充验证集后重新训练")
        if metrics.get("evaluation_status") and metrics.get("test", {}).get("evaluation_status") not in ("valid", "partial"):
            abort(400, "模型缺少独立测试结果，请补充测试集后重新训练")
        conn.execute("UPDATE deployments SET active=0 WHERE project_id=? AND station=?",
                     (row["project_id"], station))
        conn.execute("INSERT INTO deployments(project_id,station,model_id,active,created) VALUES(?,?,?,1,?)",
                     (row["project_id"], station, mid, now()))
        conn.execute("UPDATE models SET published=1 WHERE id=?", (mid,))
    return jsonify(ok=True)


# ---------------------------------------------------------------- monitoring

@app.get("/api/projects/<pid>/accuracy")
def accuracy(pid):
    project(pid)
    with db() as conn:
        model = conn.execute("""SELECT m.* FROM deployments d JOIN models m ON m.id=d.model_id
            WHERE d.project_id=? AND d.active=1 ORDER BY d.id DESC LIMIT 1""", (pid,)).fetchone()
    if model:
        card = json_value(model["card"], {}) or {}
        metrics = json_value(model["metrics"], {}) or {}
        template = card.get("template") or "det"
        classes = card.get("classes") or metrics.get("names") or []
        labels = {**ALL_LABELS, **(card.get("labels") or {})}
        source = f"模型 {model['version']}"
        note = metrics.get("reason") or "基于模型验证集；未覆盖的类别显示未评估，样本越少，数字波动越大。"
    else:
        card_path = ROOT / "models/smd_defect_cls_v1.json"
        if not card_path.is_file():
            return jsonify(source=None, template=None, overall={}, rows=[],
                           note="还没有已发布模型，也没有本机参考模型。")
        card = json.loads(card_path.read_text(encoding="utf-8"))
        metrics = card.get("metrics") or {}
        template = "cls"
        classes = card.get("classes") or []
        labels = {**ALL_LABELS, **(card.get("labels") or {})}
        source = "本机 SMD DefDB 参考模型"
        note = "这是系统内置的整板分类模型，不是本工位训练的模型；仅供参考。"
    rows = accuracy_rows(template, metrics, classes, labels)
    overall = {}
    for key in ("top1", "top5", "map50", "map", "defect_precision", "defect_recall"):
        if metrics.get(key) is not None:
            overall[key] = round(float(metrics[key]), 4)
    return jsonify(source=source, template=template, overall=overall,
                   rows=rows, note=note)


def accuracy_rows(template, metrics, classes, labels):
    per = metrics.get("per_class") or {}
    recall_map = metrics.get("per_class_recall") or {
        key: value.get("recall") for key, value in per.items() if isinstance(value, dict)}
    confusion = metrics.get("confusion_matrix") or metrics.get("confusion")
    names = metrics.get("names") or classes or list(recall_map)
    rows = []
    if isinstance(confusion, dict):
        for key in names:
            row = confusion.get(key)
            if not isinstance(row, dict):
                continue
            support = sum(row.values())
            correct = row.get(key, 0)
            predicted = sum(confusion.get(true, {}).get(key, 0) for true in names)
            rows.append(dict(key=key, label=labels.get(key, key),
                             accuracy=round(correct / support, 4) if support else None,
                             precision=round(correct / predicted, 4) if predicted else None,
                             support=support, correct=correct))
    elif isinstance(confusion, list) and confusion:
        order = metrics.get("names") or classes or []
        for index, key in enumerate(order):
            if index >= len(confusion):
                break
            support = sum(row[index] for row in confusion)
            correct = confusion[index][index] if index < len(confusion[index]) else 0
            predicted = sum(confusion[index])
            rows.append(dict(key=key, label=labels.get(key, key),
                             accuracy=round(correct / support, 4) if support else None,
                             precision=round(correct / predicted, 4) if predicted else None,
                             support=support, correct=correct))
    else:
        for key in names:
            item = per.get(key) or {}
            accuracy = item.get("recall", recall_map.get(key))
            rows.append(dict(key=key, label=labels.get(key, key),
                             accuracy=round(float(accuracy), 4) if accuracy is not None else None,
                             precision=round(float(item["precision"]), 4) if item.get("precision") is not None else None,
                             support=item.get("support"), correct=None))
    return rows


@app.route("/api/projects/<pid>/policy", methods=["GET", "POST"])
def station_policy(pid):
    item = project(pid)
    payload = request.get_json(silent=True) or {}
    station = str(payload.get("station") or request.args.get("station") or item["station"]).strip()
    if not station and request.method == "POST":
        abort(400, "请填写工位")
    with db() as conn:
        if request.method == "POST":
            required = payload.get("required_classes", [])
            if not isinstance(required, list) or any(key not in classes_for("det") for key in required):
                abort(400, "必检缺陷类型无效")
            conn.execute("INSERT OR REPLACE INTO station_policies VALUES(?,?,?,?)",
                (pid, station, json.dumps(required), int(bool(payload.get("require_classifier", True)))))
        row = conn.execute("SELECT * FROM station_policies WHERE project_id=? AND station=?",
                           (pid, station)).fetchone()
    return jsonify(station=station, required_classes=json_value(row["required_classes"]) if row else classes_for("det"),
                   require_classifier=bool(row["require_classifier"]) if row else True)


@app.get("/api/projects/<pid>/monitor")
def monitor(pid):
    project(pid)
    with db() as conn:
        deployment = conn.execute("""SELECT d.*,m.version,m.metrics,m.sha256 FROM deployments d
            JOIN models m ON m.id=d.model_id WHERE d.project_id=? AND d.active=1
            ORDER BY d.id DESC LIMIT 1""", (pid,)).fetchone()
        feedback = conn.execute("SELECT * FROM feedback WHERE project_id=? ORDER BY created DESC LIMIT 100",
                                (pid,)).fetchall()
        counts = conn.execute("""SELECT
            sum(CASE WHEN review_status='pending' THEN 1 ELSE 0 END) AS pending,
            sum(CASE WHEN review_status='approved' THEN 1 ELSE 0 END) AS approved
            FROM images WHERE project_id=?""", (pid,)).fetchone()
    result = dict(feedback_count=len(feedback), open=sum(item["state"] == "open" for item in feedback))
    result["deployment"] = None
    if deployment:
        item = dict(deployment)
        item["metrics"] = json_value(item.get("metrics"))
        result["deployment"] = item
    with db() as conn:
        result["deployments"] = [dict(row) for row in conn.execute("""SELECT d.*,m.version,r.loaded_model,r.error AS load_error,r.updated AS loaded_at
            FROM deployments d JOIN models m ON m.id=d.model_id
            LEFT JOIN runtime_status r ON r.project_id=d.project_id AND r.station=d.station
            WHERE d.project_id=? AND d.active=1 ORDER BY d.id DESC""", (pid,))]
    result["feedback"] = [dict(item) for item in feedback]
    result["counts"] = dict(counts)
    return jsonify(result)


@app.post("/api/projects/<pid>/feedback")
def add_feedback(pid):
    project(pid)
    upload = request.files.get("image")
    if not upload:
        abort(400, "请选择图片")
    fid = uuid.uuid4().hex
    folder = IMAGES / pid / "feedback"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{fid}.jpg"
    try:
        with Image.open(upload.stream) as raw:
            img = ImageOps.exif_transpose(raw).convert("RGB")
            img.save(path, "JPEG", quality=94)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        abort(400, "无法读取图片")
    with db() as conn:
        conn.execute("INSERT INTO feedback(id,project_id,path,predicted,note,state,created) VALUES(?,?,?,?,?,'open',?)",
                     (fid, pid, str(path), str(request.form.get("predicted", ""))[:120],
                      str(request.form.get("note", ""))[:500], now()))
    return jsonify(id=fid)


@app.get("/api/projects/<pid>/feedback")
def list_feedback(pid):
    project(pid)
    with db() as conn:
        rows = conn.execute("SELECT * FROM feedback WHERE project_id=? ORDER BY created DESC", (pid,)).fetchall()
    return jsonify([dict(row) for row in rows])


@app.post("/api/feedback/<fid>/convert")
def convert_feedback(fid):
    with db() as conn:
        row = conn.execute("SELECT * FROM feedback WHERE id=?", (fid,)).fetchone()
        if not row:
            abort(404, "反馈不存在")
        if row["state"] == "converted":
            return jsonify(ok=True)
        with Image.open(row["path"]) as source:
            img = source.convert("RGB")
            iid = uuid.uuid4().hex
            target = IMAGES / row["project_id"] / f"{iid}.jpg"
            img.save(target, "JPEG", quality=94)
        import hashlib
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        conn.execute("""INSERT INTO images(id,project_id,filename,path,width,height,sha256,
            state,review_status,note,created) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                     (iid, row["project_id"], f"feedback-{fid[:8]}.jpg", str(target),
                      img.width, img.height, digest, "new", "pending",
                      "由运行反馈转入：" + row["note"], now()))
        conn.execute("UPDATE feedback SET state='converted',image_id=? WHERE id=?", (iid, fid))
    return jsonify(id=iid)


# ------------------------------------------------------------------- worker

def tail(path: Path, lines: int = 100) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            content = stream.readlines()
        return "".join(content[-lines:])
    except OSError:
        return ""


def build_det_dataset(did: str, job_dir: Path) -> Path:
    items = frozen_items(did)
    if not items:
        raise RuntimeError("数据集为空")
    with db() as conn:
        counts = json_value(conn.execute("SELECT counts FROM datasets WHERE id=?", (did,)).fetchone()[0], {})
    classes = counts.get("class_order") or classes_for("det")
    dataset = job_dir / "dataset"
    for split in ("train", "val", "test"):
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)
    with db() as conn:
        for item in items:
            split = item["split"]
            target = dataset / "images" / split / f"{item['id']}.jpg"
            shutil.copy2(item["path"], target)
            rows = item["labels"]
            lines = []
            for label in rows:
                if label["class"] not in classes:
                    continue
                index = classes.index(label["class"])
                width = max(label["x2"] - label["x1"], 1e-6)
                height = max(label["y2"] - label["y1"], 1e-6)
                lines.append(f"{index} {(label['x1']+label['x2'])/2:.6f} "
                             f"{(label['y1']+label['y2'])/2:.6f} {width:.6f} {height:.6f}")
            (dataset / "labels" / split / f"{item['id']}.txt").write_text("\n".join(lines), encoding="ascii")
    import yaml
    (dataset / "data.yaml").write_text(yaml.safe_dump(dict(
        path=str(dataset), train="images/train", val="images/val",
        test="images/test",
        names=classes)), encoding="utf-8")
    return dataset


def build_cls_dataset(did: str, job_dir: Path) -> Path:
    items = frozen_items(did)
    dataset = job_dir / "dataset"
    classes = classes_for("cls")
    for split in ("train", "val", "test"):
        for name in classes:
            (dataset / split / name).mkdir(parents=True, exist_ok=True)
    for item in items:
        split = item["split"]
        name = item["primary_class"]
        if name not in classes:
            continue
        shutil.copy2(item["path"], dataset / split / name / f"{item['id']}.jpg")
    return dataset


def run_job(jid: str) -> None:
    job_dir = RUNS / jid
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True)
    log_path = LOGS / f"{jid}.log"
    with db() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
    if not job:
        return
    config = json_value(job["config"], {})
    try:
        with db() as conn:
            conn.execute("UPDATE jobs SET state='running',started=? WHERE id=?", (now(), jid))
        dataset_dir = (build_det_dataset if job["template"] == "det" else build_cls_dataset)(
            job["dataset_id"], job_dir)
        config.update(dataset=str(dataset_dir), job_dir=str(job_dir), template=job["template"])
        if job["template"] == "det":
            with db() as conn:
                baseline = conn.execute("""SELECT m.* FROM deployments d
                    JOIN models m ON m.id=d.model_id WHERE d.project_id=? AND d.station=? AND d.active=1
                    ORDER BY d.id DESC LIMIT 1""", (job["project_id"], config.get("station", ""))).fetchone()
            if baseline:
                baseline_path = job_dir / "baseline.pt"
                shutil.copy2(baseline["weights"], baseline_path)
                config["baseline"] = str(baseline_path)
                config["baseline_version"] = baseline["version"]
        config_path = job_dir / "config.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen([sys.executable, str(ROOT / "training/platform_fit.py"),
                                     str(job_dir), str(config_path)],
                                    stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
            PROCS[jid] = proc
            code = proc.wait()
        best = job_dir / "best.pt"
        metrics_path = job_dir / "metrics.json"
        with db() as conn:
            current = conn.execute("SELECT state FROM jobs WHERE id=?", (jid,)).fetchone()
        if current and current["state"] == "cancelled":
            return
        if code != 0 or not best.is_file() or not metrics_path.is_file():
            error_file = job_dir / "error.txt"
            error = (error_file.read_text(encoding="utf-8", errors="replace")
                     if error_file.is_file() else (tail(log_path, 20) or f"训练进程退出码 {code}"))
            with db() as conn:
                conn.execute("UPDATE jobs SET state='failed',error=?,finished=? WHERE id=?",
                             (error[-1000:], now(), jid))
            return
        metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
        with db() as conn:
            count = conn.execute("SELECT count(*) FROM models WHERE project_id=?", (job["project_id"],)).fetchone()[0]
            version = f"v{count+1}"
            mid = uuid.uuid4().hex
            folder = MODELS / job["project_id"] / version
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / "best.pt"
            shutil.copy2(best, target)
            import hashlib
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            card = dict(project=job["project_id"], template=job["template"], version=version,
                        classes=config.get("classes"), metrics=metadata.get("metrics"),
                        threshold=metadata.get("threshold", 0.5),
                        dataset=job["dataset_id"], created=now(),
                        source="pcb-inspector training platform")
            conn.execute("""INSERT INTO models(id,project_id,job_id,version,sha256,metrics,
                threshold,weights,card,published,created) VALUES(?,?,?,?,?,?,?,?,?,0,?)""",
                         (mid, job["project_id"], jid, version, digest,
                          json.dumps(metadata.get("metrics"), ensure_ascii=False),
                          float(metadata.get("threshold", 0.5)), str(target),
                          json.dumps(card, ensure_ascii=False), now()))
            if (config.get("auto_publish") and config.get("classes") == classes_for("det")
                    and float((metadata.get("metrics") or {}).get("map50") or 0) > 0
                    and (metadata.get("metrics") or {}).get("test", {}).get("evaluation_status") in ("valid", "partial")):
                station = config.get("station") or conn.execute(
                    "SELECT station FROM projects WHERE id=?", (job["project_id"],)).fetchone()[0] or ""
                conn.execute("UPDATE deployments SET active=0 WHERE project_id=? AND station=?", (job["project_id"], station))
                conn.execute("INSERT INTO deployments(project_id,station,model_id,active,created) VALUES(?,?,?,1,?)",
                             (job["project_id"], station, mid, now()))
                conn.execute("UPDATE models SET published=1 WHERE id=?", (mid,))
            conn.execute("""UPDATE jobs SET state='done',metrics=?,weights=?,model_id=?,
                finished=? WHERE id=?""",
                         (json.dumps(metadata.get("metrics"), ensure_ascii=False), str(target),
                          mid, now(), jid))
    except Exception as exc:  # noqa: BLE001
        with db() as conn:
            conn.execute("UPDATE jobs SET state='failed',error=?,finished=? WHERE id=?",
                         (str(exc)[:1000], now(), jid))
    finally:
        PROCS.pop(jid, None)


def worker():
    while True:
        jid = QUEUE.get()
        try:
            run_job(jid)
        except Exception:  # noqa: BLE001
            time.sleep(1)


def start_worker():
    global WORKER_STARTED
    with WORKER_LOCK:
        if WORKER_STARTED:
            return
        WORKER_STARTED = True
        try:
            with db() as conn:
                queued = [row[0] for row in conn.execute("SELECT id FROM jobs WHERE state='queued'")]
        except sqlite3.Error:
            queued = []
        for jid in queued:
            QUEUE.put(jid)
        threading.Thread(target=worker, daemon=True, name="pcb-platform-worker").start()


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(409)
@app.errorhandler(413)
def error(exc):
    return jsonify(error="上传文件超过 32 MB" if exc.code == 413 else exc.description), exc.code


initialize()

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PCB_PLATFORM_PORT", 8767))
    serve(app, host="127.0.0.1", port=port)
