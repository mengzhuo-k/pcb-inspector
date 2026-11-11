import base64
import io
import json
import math
import os
import sqlite3
import threading
import uuid
import secrets
import hashlib
import re
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, redirect
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import HTTPException
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
from model_runtime import Detection, ModelRuntime, PlatformRuntime
from batch_report import build_pdf
from defect_classifier import build_classifier

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PCB_DATA", ROOT / "data"))
DATA.mkdir(parents=True, exist_ok=True)
app = Flask(__name__, static_folder=str(ROOT / "static"))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
MISSING_WEIGHTS = ROOT / "models/missing_connection_v7.pt"
TRAIN_STORE = Path(os.environ.get("PCB_TRAIN_DATA", ROOT / "data/learning")).resolve()
model_runtime = ModelRuntime(TRAIN_STORE, MISSING_WEIGHTS,
    "da412cec50cee7a0a30faaa9508d54a0891a079909ee82e3fee716255a1df0a7")
# Multi-class detector published by the training platform, if any.
platform_runtime = PlatformRuntime(
    Path(os.environ.get("PCB_PLATFORM_DATA", ROOT / "data/platform")) / "platform.sqlite")
# Whole-board SMD defect classifier trained on Zenodo SMD DefDB.
defect_classifier = build_classifier(ROOT)

# Label, calibrated validation threshold, possible cause, follow-up.
RULES = {
    "missing_connection": ("疑似漏焊", .25, "应焊接的孔位或焊盘可能未完成焊接", "对照同板型正常样板逐处复查，必要时测量通断"),
    "insufficient_solder": ("少锡", .297, "印刷转移量偏低或润湿不足", "核查钢网开口、焊膏印刷厚度与润湿情况"),
    "excess_solder": ("多锡", .349, "局部焊膏沉积过量", "复测印刷体积并检查刮刀压力"),
    "solder_bridge": ("焊锡桥接", .417, "相邻引脚间焊膏连印或贴装偏差", "显微复查引脚间距，检查钢网清洁度与贴装坐标"),
    "solder_void": ("焊锡空洞", .249, "挥发物排出不足或焊料覆盖异常", "结合少锡候选交叉复查，必要时做 X 光检测"),
    "missing_component": ("缺件", .381, "漏贴或回流前掉件", "核对 BOM、供料报警和贴片机记录"),
    "shifted_component": ("元件偏移", .389, "贴装定位或回流受力不均", "核对基准点、贴装坐标及两端焊膏量"),
    "burn": ("烧伤", .224, "局部热应力或电气异常", "核对炉温实测曲线并检查异常供电记录"),
    "physical_damage": ("物理损伤", .214, "搬运、分板或装夹应力", "放大检查焊盘与基材并追溯装夹工序"),
    "oxidation": ("氧化", .228, "存储暴露或表面污染", "检查物料存储时间、湿度及可焊性"),
    # Whole-board classifier classes from SMD DefDB.
    "poor_solder": ("虚焊/冷焊", .5, "润湿不足、预热不够或焊盘氧化，冷焊外观多为暗淡粗糙",
                    "核对回流曲线与焊膏活性，冷焊需结合外观、剖面或拉力抽检确认"),
    "poor_solder_virtual": ("虚焊", .25, "焊料未充分润湿或假焊，电气连接不可靠",
                            "对照正常焊点复查润湿角，必要时测量通断"),
    "poor_solder_cold": ("冷焊", .25, "回流温度不足或预热不均，焊点未形成可靠金属间化合物",
                         "核对炉温曲线与预热区，结合外观、剖面或拉力抽检"),
    "pad_damage": ("焊盘损伤", .25, "焊盘翘起、脱落或基材损伤",
                   "放大检查焊盘与基材，追溯分板、返修与装夹工序"),
    "component_shift": ("元件偏移", .25, "贴装定位偏移或回流时焊料表面张力不均",
                        "核对贴装坐标、基准点与两端焊膏量"),
    "misaligned_component": ("元件偏移", .5, "贴装定位偏移或回流时焊料表面张力不均",
                             "核对贴装坐标、基准点与两端焊膏量"),
    "damaged_component": ("焊盘/元件损伤", .5, "搬运、分板、返修或装夹造成焊盘翘起、元件破损",
                          "放大检查焊盘与元件本体，追溯分板、返修和装夹工序"),
}
MANUAL = ["漏焊", "虚焊", "冷焊", "桥接", "多锡", "焊盘损伤", "元件偏移", "缺件", "氧化", "烧伤"]


@contextmanager
def db():
    conn = sqlite3.connect(DATA / "records.sqlite", timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


with db() as conn:
    conn.execute("""CREATE TABLE IF NOT EXISTS records
        (id TEXT PRIMARY KEY, created TEXT, station TEXT, batch TEXT, temp REAL,
         note TEXT, boxes TEXT, analysis TEXT, image BLOB, decision TEXT,
         reviewer TEXT, signed TEXT, manual TEXT, comment TEXT)""")
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(records)")}
    for name, kind in [("batch_id", "TEXT"), ("filename", "TEXT"), ("auto_decision", "TEXT"),
                       ("revision", "INTEGER DEFAULT 0"), ("model_version", "TEXT"), ("model_sha256", "TEXT"),
                       ("alignment", "TEXT"), ("original_image", "BLOB"), ("process_parameters", "TEXT"),
                       ("classification", "TEXT"), ("issues", "TEXT DEFAULT '[]'"), ("corrections", "TEXT"),
                       ("training_source_key", "TEXT"), ("inspection", "TEXT")]:
        if name not in columns:
            conn.execute(f"ALTER TABLE records ADD COLUMN {name} {kind}")
    conn.execute("""CREATE TABLE IF NOT EXISTS batches (
        id TEXT PRIMARY KEY, created TEXT, station TEXT, name TEXT, state TEXT)""")
    if "completed_at" not in {row["name"] for row in conn.execute("PRAGMA table_info(batches)")}:
        conn.execute("ALTER TABLE batches ADD COLUMN completed_at TEXT")
    if "process_parameters" not in {row["name"] for row in conn.execute("PRAGMA table_info(batches)")}:
        conn.execute("ALTER TABLE batches ADD COLUMN process_parameters TEXT")
    for column in ("inspection_context","source_batch_id"):
        if column not in {row["name"] for row in conn.execute("PRAGMA table_info(batches)")}:
            conn.execute(f"ALTER TABLE batches ADD COLUMN {column} TEXT")
    conn.execute("""CREATE TABLE IF NOT EXISTS batch_items (
        batch_id TEXT, ordinal INTEGER, filename TEXT, state TEXT, error TEXT,
        image BLOB, record_id TEXT, sampled INTEGER DEFAULT 0, sample_revision INTEGER DEFAULT 0,
        PRIMARY KEY(batch_id, ordinal))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS reviews (
        record_id TEXT, revision INTEGER, created TEXT, reviewer TEXT, decision TEXT, comment TEXT,
        PRIMARY KEY(record_id, revision))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS reports (
        id TEXT PRIMARY KEY,batch_id TEXT,created TEXT,fingerprint TEXT,filename TEXT,
        document BLOB,completed_at TEXT,UNIQUE(batch_id,fingerprint))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS training_imports(
        source_key TEXT PRIMARY KEY,image BLOB,filename TEXT,station TEXT,batch TEXT,
        created TEXT,image_id TEXT,error TEXT DEFAULT '',skipped INTEGER DEFAULT 0)""")
    if "skipped" not in {row["name"] for row in conn.execute("PRAGMA table_info(training_imports)")}:
        conn.execute("ALTER TABLE training_imports ADD COLUMN skipped INTEGER DEFAULT 0")
    conn.execute("CREATE TABLE IF NOT EXISTS inspection_settings(key TEXT PRIMARY KEY,value TEXT NOT NULL)")

with db() as conn:
    conn.execute("""CREATE TABLE IF NOT EXISTS exception_cases(
        record_id TEXT PRIMARY KEY,batch_id TEXT NOT NULL,snapshot TEXT NOT NULL,
        review_data TEXT DEFAULT '{}',revision INTEGER DEFAULT 0)""")
    if "deleted" not in {row["name"] for row in conn.execute("PRAGMA table_info(exception_cases)")}:
        conn.execute("ALTER TABLE exception_cases ADD COLUMN deleted INTEGER DEFAULT 0")


def require_open_batch(conn, bid):
    if not bid:
        return
    row = conn.execute("SELECT state FROM batches WHERE id=?", (bid,)).fetchone()
    if not row:
        abort(404, "批量任务不存在")
    if row["state"] == "completed":
        abort(409, "批次已完成归档，仅供查看")


def capture_exceptions(conn, bid):
    rows = conn.execute("""SELECT r.*,i.ordinal FROM records r
        JOIN batch_items i ON i.record_id=r.id WHERE r.batch_id=? AND i.state='done'
        AND r.decision='不合格'""", (bid,)).fetchall()
    for row in rows:
        snapshot = document({key:row[key] for key in COLS.split(",")})
        snapshot["ordinal"] = row["ordinal"]
        snapshot["reviews"] = [dict(item) for item in conn.execute(
            "SELECT * FROM reviews WHERE record_id=? ORDER BY revision", (row["id"],))]
        conn.execute("INSERT OR IGNORE INTO exception_cases(record_id,batch_id,snapshot) VALUES(?,?,?)",
                     (row["id"], bid, json.dumps(snapshot, ensure_ascii=False)))


with db() as conn:
    for row in conn.execute("SELECT id,boxes,classification,decision,auto_decision FROM records WHERE decision='待复核' OR auto_decision='待复核'").fetchall():
        classification = json.loads(row["classification"]) if row["classification"] else {}
        verdict = "不合格" if json.loads(row["boxes"] or "[]") or classification.get("is_defect") else "合格"
        conn.execute("""UPDATE records SET decision=CASE WHEN decision='待复核' THEN ? ELSE decision END,
            auto_decision=CASE WHEN auto_decision='待复核' THEN ? ELSE auto_decision END,
            revision=revision+1 WHERE id=?""", (verdict, verdict, row["id"]))


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def training_import_enabled(conn):
    row = conn.execute("SELECT value FROM inspection_settings WHERE key='training_import_enabled'").fetchone()
    return not row or row[0] == "1"


@app.get("/api/training-import-setting")
def get_training_import_setting():
    with db() as conn:
        return jsonify(enabled=training_import_enabled(conn))


@app.post("/api/training-import-setting")
def set_training_import_setting():
    payload = request.get_json(silent=True) or {}
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        abort(400, "入库开关必须为开启或关闭")
    with db() as conn:
        conn.execute("INSERT OR REPLACE INTO inspection_settings VALUES('training_import_enabled',?)",
                     ("1" if enabled else "0",))
    if enabled:
        work_ready.set()
    return jsonify(enabled=enabled)


def queue_training_import(conn, content, filename, station="", batch="", created=None, historical=False, source_key=None):
    key = source_key or hashlib.sha256(station.encode("utf-8") + b"\0" + content).hexdigest()
    if historical and conn.execute("SELECT 1 FROM training_imports WHERE source_key=?", (key,)).fetchone():
        return key
    enabled = training_import_enabled(conn)
    conn.execute("""INSERT INTO training_imports(
        source_key,image,filename,station,batch,created,skipped) VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(source_key) DO UPDATE SET image=excluded.image,skipped=0,
        filename=excluded.filename,batch=excluded.batch
        WHERE training_imports.skipped=1 AND excluded.skipped=0""",
                 (key, content if enabled else None, filename, station, batch, created or now(), int(not enabled)))
    return key


def import_to_training(img, filename, station="", batch=""):
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=94)
    with db() as conn:
        key = queue_training_import(conn, buffer.getvalue(), filename, station, batch)
    flush_training_imports()
    return key


training_import_lock = threading.Lock()


def flush_training_imports():
    if not training_import_lock.acquire(blocking=False):
        return
    try:
        import platform_app
        while True:
            with db() as conn:
                if not training_import_enabled(conn):
                    return
                row = conn.execute("SELECT * FROM training_imports WHERE image_id IS NULL AND skipped=0 ORDER BY rowid LIMIT 1").fetchone()
            if not row:
                return
            try:
                iid = platform_app.import_inspection_image(row["source_key"], row["image"],
                    row["filename"], row["station"], row["batch"], row["created"])
            except Exception as exc:
                app.logger.exception("图片自动入库失败，后台将重试")
                with db() as conn:
                    conn.execute("UPDATE training_imports SET error=? WHERE source_key=?", (str(exc), row["source_key"]))
                return
            with db() as conn:
                conn.execute("UPDATE training_imports SET image_id=?,image=NULL,error='' WHERE source_key=?",
                             (iid, row["source_key"]))
    finally:
        training_import_lock.release()


def backfill_training_imports():
    # Read one image at a time so historic libraries do not exhaust memory.
    for query in (
        "SELECT COALESCE(original_image,image) AS image,filename,station,batch,created,training_source_key FROM records",
        """SELECT i.image,i.filename,b.station,b.name AS batch,b.created,NULL AS training_source_key FROM batch_items i
            JOIN batches b ON b.id=i.batch_id WHERE i.image IS NOT NULL""",
    ):
        with db() as conn:
            for row in conn.execute(query):
                if row["image"]:
                    queue_training_import(conn, row["image"], row["filename"],
                                          row["station"] or "", row["batch"] or "", row["created"],
                                          historical=True, source_key=row["training_source_key"])
    flush_training_imports()


def snapshot():
    upload = request.files.get("image")
    if not upload:
        abort(400, "请选择图片")
    try:
        with Image.open(upload.stream) as raw:
            if raw.width * raw.height > 24000000:
                abort(400, "图片超过 2400 万像素")
            return ImageOps.exif_transpose(raw).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        abort(400, "图片无法解码，请使用 JPG、PNG 或 WebP")


def detect(img, station="", inspection_context=None):
    original = img
    alignment = None
    reference = None
    try:
        reference = (inspection_context.get("reference") if inspection_context is not None
                     else platform_runtime.alignment_reference(station))
        if reference:
            if reference.get("sha256") and hashlib.sha256(Path(reference["path"]).read_bytes()).hexdigest() != reference["sha256"]:
                raise ValueError("标准图文件已变化，请按当前模板重新检测该批次")
            from pcb_alignment import align
            result = align(img, reference["path"])
            with Image.open(io.BytesIO(base64.b64decode(result["preview"].split(",", 1)[1]))) as corrected:
                img = corrected.convert("RGB")
            alignment = dict(reference_id=reference["id"], matrix=result["matrix"], method=result["method"],
                project_id=reference.get("project_id"),matches=result.get("matches"),inliers=result.get("inliers"))
    except (ValueError, OSError, sqlite3.Error) as exc:
        abort(400, "板面对齐失败，请使用清晰完整的板面图片：" + str(exc))
    if reference and "sites" in reference:
        return detect_template_sites(img, original, station, reference, alignment)
    issues = []
    policy = platform_runtime.policy(station)
    classification = None
    try:
        classification = defect_classifier.predict(img)
    except Exception:  # noqa: BLE001 - a classifier fault must not stop detection
        app.logger.exception("Defect classification failed")
        if policy["require_classifier"]:
            issues.append("整板分类执行失败")
    if classification is None and policy["require_classifier"] and not issues:
        issues.append("整板分类模型未加载")
    try:
        result, version, sha256 = platform_runtime.predict(
            img, station=station, imgsz=960, verbose=False)
        issues.extend(getattr(result, "pcb_issues", []))
        missing = set(policy["required_classes"]) - set(result.names.values())
        if missing:
            issues.append("模型未覆盖必检类型：" + "、".join(sorted(missing)))
        boxes = []
        for x1, y1, x2, y2, score, cls in result.boxes.data.cpu().tolist():
            key = result.names[int(cls)]
            rule = RULES.get(key, (key, .25, "模型检出焊接缺陷", "对照正常样板复查"))
            boxes.append(dict(key=key, label=rule[0], score=round(score, 4), model=version,
                              rect=[x1/img.width, y1/img.height, x2/img.width, y2/img.height]))
        detection = Detection(boxes, version, sha256, classification=classification, issues=issues)
        detection.image = img
        detection.original_image = original
        detection.alignment = alignment
        return detection
    except LookupError as exc:
        issues.append(str(exc))
    except Exception:
        app.logger.exception("Platform inference failed")
        issues.append("工位检测模型执行失败")
    issues.append("当前使用备用漏焊模型，检测范围仅包含漏焊定位")
    try:
        result, version, sha256 = model_runtime.predict(
            img, imgsz=960, conf=RULES["missing_connection"][1], verbose=False)
    except Exception:
        app.logger.exception("Fallback inference failed")
        issues.append("备用漏焊模型执行失败")
        raise RuntimeError("检测模型执行失败，请重试")
    boxes = []
    for x1, y1, x2, y2, score, cls in result.boxes.data.cpu().tolist():
        key = result.names[int(cls)]
        if key == "missing_connection":
            boxes.append(dict(key=key, label=RULES[key][0], score=round(score, 4), model=version,
                              rect=[x1/img.width, y1/img.height, x2/img.width, y2/img.height]))
    detection = Detection(boxes, version, sha256, classification=classification, issues=issues)
    detection.image = img
    detection.original_image = original
    detection.alignment = alignment
    return detection


def detect_template_sites(img, original, station, reference, alignment):
    from site_inspection import prepare_sites, collect_site_defects
    try:
        sites,regions = prepare_sites(img, reference["sites"], alignment["matrix"])
    except (ValueError, KeyError, TypeError) as exc:
        abort(400, str(exc))
    images = [img.crop(region) for region in regions]
    issues = []
    try:
        results,version,digest = platform_runtime.predict_many(images,station=station,
            project_id=reference.get("project_id"),imgsz=640,verbose=False)
    except Exception:
        app.logger.exception("点位检测模型不可用，尝试备用漏焊模型")
        issues.append("当前使用备用漏焊模型检查点位，仅支持漏焊识别")
        results,version,digest = model_runtime.predict_many(images,imgsz=640,
            conf=RULES["missing_connection"][1],verbose=False)
    for result in results:
        issues.extend(getattr(result,"pcb_issues",[]))
    missing = set(platform_runtime.policy(station)["required_classes"]) - set(results[0].names.values())
    if missing:
        issues.append("点位模型未覆盖必检类型：" + "、".join(sorted(missing)))
    boxes = collect_site_defects(results,regions,img.size,sites,RULES,version)
    detection = Detection(boxes,version,digest,issues=list(dict.fromkeys(issues)))
    detection.image = img
    detection.original_image = original
    detection.alignment = alignment
    detection.inspection = dict(mode="solder_sites",project_id=reference.get("project_id"),
        reference_id=reference["id"],template_revision=reference["template_revision"],
        total_sites=len(sites),checked_sites=len(sites),regions=len(regions),sites=sites)
    return detection


def field(name, limit, required=False):
    value = request.form.get(name, "").strip()
    if len(value) > limit or (required and not value):
        abort(400, "请检查字段：" + name)
    return value


@app.get("/")
def index():
    return redirect("/platform/")


@app.post("/api/preview")
def preview():
    img = snapshot()
    station = request.form.get("station", "")[:80]
    if request.form.get("live_preview") != "1" and request.form.get("preview_only") != "1":
        import_to_training(img, request.files["image"].filename, station, request.form.get("batch", "")[:80])
    boxes = detect(img, station)
    preview = None
    if boxes.alignment:
        buffer = io.BytesIO()
        boxes.image.save(buffer, "JPEG", quality=94)
        preview = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return jsonify(boxes=boxes, model_version=boxes.model_version,
                   classification=boxes.classification, issues=boxes.issues,
                   image=preview, alignment=boxes.alignment,inspection=getattr(boxes,"inspection",None))


@app.get("/api/model/status")
def model_status():
    return jsonify(model_info())


def model_info():
    info = model_runtime.status()
    info["classifier"] = defect_classifier.status()
    station = request.args.get("station", "")
    info["platform"] = platform_runtime.status(station)
    try:
        reference = platform_runtime.alignment_reference(station)
        info["inspection"] = dict(mode="solder_sites",total_sites=len(reference["sites"]),
            template_revision=reference["template_revision"]) if reference and "sites" in reference else dict(mode="whole_board")
    except (ValueError, OSError, sqlite3.Error):
        info["inspection"] = dict(mode="unavailable")
    return info


def process_parameters(payload):
    result = {}
    for key, label, maximum in (("temp", "回流峰值温度", 500), ("reflow_seconds", "回流时间", 3600)):
        raw = payload.get(key)
        if raw not in (None, ""):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                abort(400, label + "须为数字")
            if not math.isfinite(value) or not 0 <= value <= maximum:
                abort(400, label + "超出有效范围")
            result[key] = value
    paste = str(payload.get("solder_batch", "")).strip()
    if len(paste) > 80:
        abort(400, "焊膏批号最多 80 字")
    if paste:
        result["solder_batch"] = paste
    return result


@app.post("/api/records")
def create():
    station = field("station", 80, True)
    batch = field("batch", 80) or "巡检-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    note = field("note", 2000)
    parameters = process_parameters(request.form)
    temp = parameters.get("temp")
    img = snapshot()
    training_source_key = import_to_training(img, request.files["image"].filename, station, batch)
    boxes = detect(img, station)
    with db() as conn:
        rid = save_record(conn, img, boxes, station, batch, temp, note,
                          filename=request.files["image"].filename, process=parameters,
                          training_source_key=training_source_key)
    return detail(rid)


def save_record(conn, img, boxes, station, batch, temp=None, note="", batch_id=None, filename=None, process=None, training_source_key=None):
    buffer = io.BytesIO()
    getattr(boxes, "image", img).save(buffer, "JPEG", quality=94)
    rid = uuid.uuid4().hex
    process = process or {}
    history = conn.execute("""SELECT boxes,temp,decision FROM records WHERE station=?
        ORDER BY created DESC LIMIT 100""", (station,)).fetchall()
    analysis = []
    for key in dict.fromkeys(b["key"] for b in boxes):
        similar = [r for r in history if any(b["key"] == key for b in json.loads(r["boxes"]))]
        evidence = f"同工位最近 {len(history)} 份记录中，{len(similar)} 份检出同类缺陷。"
        temps = [r["temp"] for r in similar if r["temp"] is not None]
        if temp is not None:
            evidence += f"本次回流峰值 {temp:g} °C。"
            if temps:
                evidence += f"历史同类 {len(temps)} 份有温度记录，均值 {sum(temps)/len(temps):.1f} °C。"
        if process.get("reflow_seconds") is not None:
            evidence += f"本次回流时间 {process['reflow_seconds']:g} 秒。"
        if process.get("solder_batch"):
            evidence += f"焊膏批号 {process['solder_batch']}。"
        if not history:
            evidence += "当前工位尚无历史记录可对照。"
        rule = RULES.get(key, (key, .25, "模型检出缺陷", "请对照原图复查"))
        analysis.append(dict(label=rule[0], cause=rule[2], action=rule[3], evidence=evidence))
    classification = getattr(boxes, "classification", None)
    if classification and classification.get("is_defect"):
        key = classification["key"]
        rule = RULES.get(key)
        label = classification["label"]
        cause = rule[2] if rule else "整板图像分类检出焊接缺陷"
        action = rule[3] if rule else "对照同板型正常样板复查，确认缺陷类型与位置"
        alternatives = "、".join(f"{item['label']} {item['score']:.0%}" for item in classification.get("top", [])[1:])
        evidence = (f"整板图像识别：{label} {classification['score']:.0%}，"
                    f"缺陷概率 {classification['defect_score']:.0%}（阈值 {classification['threshold']:.0%}）。"
                    + (f" 次选：{alternatives}。" if alternatives else "")
                    + " 整板分类仅给出类型，不定位具体元件。")
        analysis.append(dict(label=label, cause=cause, action=action, evidence=evidence))
    for finding in analysis:
        finding["basis"] = "模型检出与工艺规则"
        finding["history_samples"] = len(history)
    defective = bool(boxes) or bool(classification and classification["is_defect"])
    issues = getattr(boxes, "issues", [])
    decision = "不合格" if defective else "合格"
    conn.execute("""INSERT INTO records
        (id,created,station,batch,temp,note,boxes,analysis,image,decision,manual,comment,
         batch_id,filename,auto_decision,revision,model_version,model_sha256,classification)
        VALUES (?,?,?,?,?,?,?,?,?,?,'[]','',?,?,?,0,?,?,?)""",
        (rid, now(), station, batch, temp, note, json.dumps(boxes), json.dumps(analysis),
         buffer.getvalue(), decision, batch_id, filename, decision,
         getattr(boxes, "model_version", None), getattr(boxes, "model_sha256", None),
         json.dumps(classification) if classification else None))
    conn.execute("UPDATE records SET issues=? WHERE id=?", (json.dumps(issues, ensure_ascii=False), rid))
    original = io.BytesIO()
    getattr(boxes, "original_image", img).save(original, "JPEG", quality=94)
    conn.execute("UPDATE records SET alignment=?,original_image=?,process_parameters=?,training_source_key=? WHERE id=?",
                 (json.dumps(getattr(boxes, "alignment", None)), original.getvalue(),
                  json.dumps(process, ensure_ascii=False), training_source_key, rid))
    conn.execute("UPDATE records SET inspection=? WHERE id=?",
                 (json.dumps(getattr(boxes,"inspection",None),ensure_ascii=False),rid))
    return rid


COLS = "id,created,station,batch,temp,note,boxes,analysis,decision,reviewer,signed,manual,comment,batch_id,filename,auto_decision,revision,model_version,model_sha256,classification,issues,corrections,alignment,process_parameters,inspection"


def document(row):
    obj = dict(row)
    for key in ["boxes", "analysis", "manual", "issues"]:
        obj[key] = json.loads(obj[key] or "[]")
    obj["classification"] = json.loads(obj["classification"]) if obj.get("classification") else None
    obj["process_parameters"] = json.loads(obj.get("process_parameters") or "{}")
    obj["alignment"] = json.loads(obj["alignment"]) if obj.get("alignment") else None
    obj["corrections"] = json.loads(obj["corrections"]) if obj.get("corrections") else None
    obj["inspection"] = json.loads(obj["inspection"]) if obj.get("inspection") else None
    return obj


@app.get("/api/records")
def records():
    with db() as conn:
        rows = conn.execute("""SELECT id,created,station,batch,decision,signed,reviewer,
            batch_id,filename,auto_decision,revision,model_version,classification,
            json_array_length(boxes) AS count,
            COALESCE((SELECT state='completed' FROM batches WHERE id=records.batch_id),0) AS readonly,
            COALESCE((SELECT sampled FROM batch_items WHERE record_id=records.id),0) AS sampled,
            COALESCE((SELECT sample_revision FROM batch_items WHERE record_id=records.id),0) AS sample_revision
            FROM records ORDER BY rowid DESC""").fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["classification"] = json.loads(item["classification"]) if item["classification"] else None
        result.append(item)
    return jsonify(result)


@app.get("/api/records/<rid>")
def detail(rid):
    with db() as conn:
        row = conn.execute(f"SELECT {COLS},(original_image IS NOT NULL) AS original_available FROM records WHERE id=?", (rid,)).fetchone()
        history = conn.execute("SELECT * FROM reviews WHERE record_id=? ORDER BY revision DESC", (rid,)).fetchall()
        sample = conn.execute("SELECT sampled,sample_revision FROM batch_items WHERE record_id=?", (rid,)).fetchone()
        batch = conn.execute("SELECT state FROM batches WHERE id=?", (row["batch_id"],)).fetchone() if row else None
    if not row:
        abort(404)
    result = document(row)
    result["reviews"] = [dict(item) for item in history]
    result["sampled"] = bool(sample and sample["sampled"])
    result["sample_revision"] = sample["sample_revision"] if sample else 0
    result["readonly"] = bool(batch and batch["state"] == "completed")
    result["review_required"] = not result["readonly"] and (not row["batch_id"] or result["sampled"] or row["decision"] == "不合格" or row["auto_decision"] == "不合格")
    return jsonify(result)


@app.get("/api/records/<rid>/image")
def photo(rid):
    with db() as conn:
        original=request.args.get('original')=='1'
        column = "COALESCE(original_image,image)" if original else "image"
        row = conn.execute(f"SELECT {column},boxes,inspection FROM records WHERE id=?", (rid,)).fetchone()
    if not row:
        abort(404)
    content=row[0]
    boxes=json.loads(row[1] or '[]')
    inspection=json.loads(row[2] or 'null')
    sites=inspection.get('sites',[]) if inspection else []
    marked=request.args.get('marked')=='1' and not original and bool(boxes or sites)
    digest=hashlib.sha256(content)
    if marked:
        digest.update(('marked-v2:'+str(row[1])+str(row[2])).encode('utf-8'))
    tag=digest.hexdigest()
    if request.if_none_match.contains_weak(tag):
        response=app.response_class(status=304)
        response.set_etag(tag)
        response.cache_control.no_cache=True
        return response
    if marked:
        # Thumbnails use the saved corrected pixels and detections, never a new inference.
        with Image.open(io.BytesIO(content)) as source:
            picture=source.convert('RGB')
        drawing=ImageDraw.Draw(picture,'RGBA')
        width,height=picture.size
        def rectangle(values):
            try:
                x1,y1,x2,y2=map(float,values)
            except (TypeError,ValueError):
                return None
            if not all(math.isfinite(value) for value in (x1,y1,x2,y2)) or x2<=x1 or y2<=y1:
                return None
            return (max(0,min(width-1,round(x1*width))),max(0,min(height-1,round(y1*height))),
                    max(0,min(width-1,round(x2*width))),max(0,min(height-1,round(y2*height))))
        dash=max(3,round(max(width,height)/320))
        point_width=max(1,round(max(width,height)/960))
        for site in sites:
            rect=rectangle(site.get('rect'))
            if rect is None:
                continue
            left,top,right,bottom=rect
            drawing.rectangle(rect,fill=(56,189,248,8))
            for start in range(left,right+1,dash*2):
                for y in (top,bottom):
                    drawing.line((start,y,min(start+dash,right),y),fill=(56,189,248,255),width=point_width)
            for start in range(top,bottom+1,dash*2):
                for x in (left,right):
                    drawing.line((x,start,x,min(start+dash,bottom)),fill=(56,189,248,255),width=point_width)
        for box in boxes:
            rect=rectangle(box.get('rect'))
            if rect is not None:
                drawing.rectangle(rect,outline=(245,158,11,153),width=max(2,round(max(width,height)/320)))
        buffer=io.BytesIO()
        picture.save(buffer,'JPEG',quality=94)
        content=buffer.getvalue()
    return send_file(io.BytesIO(content),mimetype='image/jpeg',etag=tag,conditional=True)


@app.delete("/api/records/<rid>")
def delete_record(rid):
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        record = conn.execute("SELECT batch_id FROM records WHERE id=?", (rid,)).fetchone()
        if not record:
            abort(404, "检验单已删除或不存在")
        require_open_batch(conn, record["batch_id"])
        cursor = conn.execute("DELETE FROM records WHERE id=?", (rid,))
        if not cursor.rowcount:
            abort(404, "检验单已删除或不存在")
        conn.execute("DELETE FROM reviews WHERE record_id=?", (rid,))
        conn.execute("UPDATE batch_items SET state='deleted',record_id=NULL,sampled=0 WHERE record_id=?", (rid,))

    return jsonify(deleted=rid)


@app.post("/api/records/<rid>/review")
def review(rid):
    reviewer = field("reviewer", 80, True)
    decision = field("decision", 10)
    manual = request.form.getlist("manual")
    comment = field("comment", 2000)
    if decision not in ["合格", "不合格"] or any(m not in MANUAL for m in manual) or (manual and decision == "合格"):
        abort(400, "请核对复查结论与补标项")
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("""SELECT revision,signed,batch_id,decision,auto_decision,
            COALESCE((SELECT sampled FROM batch_items WHERE record_id=records.id),0) AS sampled
            FROM records WHERE id=?""", (rid,)).fetchone()
        if not row:
            abort(404, "检验单已删除或不存在")
        require_open_batch(conn, row["batch_id"])
        if not (not row["batch_id"] or row["sampled"] or row["decision"] == "不合格" or row["auto_decision"] == "不合格"):
            abort(400, "该图片不属于抽样清单或当前不合格图片")
        try:
            revision = int(request.form.get("revision", 0))
        except ValueError:
            abort(400, "复查版本无效")
        if row["revision"] != revision or (row["signed"] and "revision" not in request.form):
            abort(409, "其他员工已修改此单，请重新打开后复查")
        signed = now()
        cursor = conn.execute("""UPDATE records SET decision=?,reviewer=?,signed=?,manual=?,comment=?
            ,revision=revision+1 WHERE id=? AND revision=?""",
            (decision, reviewer, signed, json.dumps(manual), comment, rid, revision))
        if not cursor.rowcount:
            abort(409, "其他员工已修改此单，请重新打开后复查")
        conn.execute("INSERT INTO reviews VALUES (?,?,?,?,?,?)", (rid, revision+1, signed, reviewer, decision, comment))

    return detail(rid)


work_ready = threading.Event()


@app.post("/api/records/<rid>/corrections")
def save_corrections(rid):
    import platform_app
    payload = request.get_json(silent=True) or {}
    pid = payload.get("project_id")
    platform_app.project(pid)
    boxes = payload.get("boxes")
    if not isinstance(boxes, list) or len(boxes) > 500:
        abort(400, "纠错框数量无效")
    cleaned = []
    for box in boxes:
        try:
            key = box["key"]
            x1, y1, x2, y2 = map(float, box["rect"])
        except (TypeError, KeyError, ValueError):
            abort(400, "纠错框格式无效")
        if key not in platform_app.classes_for("det"):
            abort(400, "请选择训练平台支持的缺陷类别")
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x1, y1, x2, y2)) or x1 >= x2 or y1 >= y2:
            abort(400, "纠错框坐标无效")
        cleaned.append(dict(key=key, rect=[x1, y1, x2, y2]))
    reviewer = str(payload.get("reviewer", "")).strip()[:80]
    if not reviewer:
        abort(400, "请填写纠错人姓名")
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM records WHERE id=?", (rid,)).fetchone()
        if not row:
            abort(404, "检验单不存在")
        require_open_batch(conn, row["batch_id"])
        if payload.get("revision") != row["revision"]:
            abort(409, "检验单已更新，请重新打开后纠错")
        correction = dict(boxes=cleaned, reviewer=reviewer, project_id=pid)
        previous = json.loads(row["corrections"]) if row["corrections"] else None
        if previous != correction and previous != payload.get("previous_corrections"):
            abort(409, "其他员工已更新纠错框，请重新打开后修改")
        token = hashlib.sha256((rid + json.dumps(correction, sort_keys=True)).encode()).hexdigest()
        iid = "review-" + token
        folder = platform_app.IMAGES / pid
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / (iid + ".jpg")
        with platform_app.db() as training:
            exists = training.execute("SELECT id FROM images WHERE id=?", (iid,)).fetchone()
            if not exists:
                target.write_bytes(row["image"])
                with Image.open(io.BytesIO(row["image"])) as source:
                    width, height = source.size
                predicted = dict(record_id=rid, station=row["station"], batch=row["batch"],
                                 model_version=row["model_version"], boxes=json.loads(row["boxes"]),
                                 classification=json.loads(row["classification"]) if row["classification"] else None)
                training.execute("""INSERT INTO images(id,project_id,filename,path,width,height,batch,
                    board_serial,sha256,state,review_status,note,created)
                    VALUES(?,?,?,?,?,?,?,?,?,'labeled','approved',?,?)""",
                    (iid, pid, row["filename"] or rid + ".jpg", str(target), width, height,
                     row["batch"], rid, hashlib.sha256(row["image"]).hexdigest(),
                     "检验纠错：" + reviewer + "；来源检验单 " + rid, now()))
                for box in cleaned:
                    training.execute("""INSERT INTO labels(image_id,class,x1,y1,x2,y2,annotator,created)
                        VALUES(?,?,?,?,?,?,?,?)""",
                        (iid, box["key"], *box["rect"], reviewer, now()))
                training.execute("""INSERT INTO feedback(id,project_id,path,predicted,note,state,image_id,created)
                    VALUES(?,?,?,?,?,'converted',?,?)""",
                    (iid, pid, str(target), json.dumps(predicted, ensure_ascii=False),
                     "检验纠错，由 " + reviewer + " 提交，标注已保存", iid, now()))
        conn.execute("UPDATE records SET corrections=? WHERE id=?",
                     (json.dumps(correction, ensure_ascii=False), rid))
    return jsonify(ok=True, image_id=iid, project_id=pid)


@app.post("/api/batches")
def new_batch():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        abort(400, "批次参数无效")
    station = str(payload.get("station", "")).strip()
    name = str(payload.get("name", "")).strip()
    filenames = payload.get("filenames")
    if not station or not name or len(station) > 80 or len(name) > 80:
        abort(400, "请填写工位和批次，最多 80 字")
    if not isinstance(filenames, list) or not 1 <= len(filenames) <= 200:
        abort(400, "每批请选择 1 至 200 张图片")
    if any(not isinstance(n, str) or not n or len(n) > 300 for n in filenames):
        abort(400, "文件名无效或过长")
    parameters = process_parameters(payload)
    bid = uuid.uuid4().hex
    with db() as conn:
        conn.execute("INSERT INTO batches(id,created,station,name,state) VALUES (?,?,?,?, 'importing')", (bid, now(), station, name))
        conn.execute("UPDATE batches SET process_parameters=? WHERE id=?", (json.dumps(parameters), bid))
        conn.executemany("""INSERT INTO batch_items
            (batch_id,ordinal,filename,state) VALUES (?,?,?,'awaiting_upload')""",
            [(bid, i, n) for i, n in enumerate(filenames)])
    return batch_detail(bid)


@app.get("/api/batches")
def batch_list():
    with db() as conn:
        rows = conn.execute("""SELECT b.*,
            count(i.ordinal) AS total,
            sum(CASE WHEN i.state IN ('done','failed','deleted') THEN 1 ELSE 0 END) AS processed,
            sum(CASE WHEN i.state='done' THEN 1 ELSE 0 END) AS completed,
            sum(CASE WHEN i.state='done' AND r.decision='合格' THEN 1 ELSE 0 END) AS passed,
            sum(CASE WHEN i.state='done' AND r.decision='不合格' THEN 1 ELSE 0 END) AS rejected,
            sum(CASE WHEN i.state='done' AND r.signed IS NOT NULL THEN 1 ELSE 0 END) AS reviewed,
            sum(CASE WHEN i.state='failed' THEN 1 ELSE 0 END) AS failed
            FROM batches b LEFT JOIN batch_items i ON b.id=i.batch_id
            LEFT JOIN records r ON i.record_id=r.id
            GROUP BY b.id ORDER BY b.created DESC,b.rowid DESC""").fetchall()
    result = []
    for row in rows:
        item = dict(row)
        decided = item["passed"] + item["rejected"]
        item["rate"] = round(100*item["passed"]/decided, 1) if decided else None
        result.append(item)
    return jsonify(result)


@app.get("/api/batches/rejections")
def rejected_items():
    with db() as conn:
        rows = conn.execute("""SELECT e.*,b.name,b.station,b.state,b.completed_at FROM exception_cases e
            JOIN batches b ON b.id=e.batch_id WHERE b.state='completed'
            AND e.deleted=0
            AND json_extract(e.snapshot,'$.decision')='不合格'
            ORDER BY b.completed_at DESC,b.rowid DESC,e.rowid""").fetchall()
    result = []
    for row in rows:
        snapshot = json.loads(row["snapshot"])
        result.append(dict(record_id=row["record_id"],exception_id=row["record_id"],
            batch_id=row["batch_id"],batch_name=row["name"],station=row["station"],
            filename=snapshot["filename"],ordinal=snapshot["ordinal"],
            completed_at=row["completed_at"],
            decision=snapshot["decision"],auto_decision=snapshot["auto_decision"],
            classification=snapshot["classification"],count=len(snapshot["boxes"])))
    return jsonify(result)


@app.route("/api/exceptions/<rid>",methods=["GET","DELETE"])
def exception_case(rid):
    with db() as conn:
        if request.method == "DELETE":
            conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("""SELECT e.* FROM exception_cases e JOIN batches b ON b.id=e.batch_id
            WHERE e.record_id=? AND b.state='completed' AND e.deleted=0
            AND json_extract(e.snapshot,'$.decision')='不合格'""", (rid,)).fetchone()
        if not row:
            abort(404, "已完成批次的不合格图片不存在")
        if request.method == "DELETE":
            conn.execute("UPDATE exception_cases SET deleted=1 WHERE record_id=?", (rid,))
            return jsonify(deleted=rid)
        snapshot = json.loads(row["snapshot"])
        availability=conn.execute("SELECT original_image IS NOT NULL FROM records WHERE id=?",(rid,)).fetchone()
        snapshot["original_available"]=int(bool(availability and availability[0]))
    snapshot.update(id=rid,exception_mode=True,readonly=True,review_required=False)
    return jsonify(snapshot)


def summarize_point_defects(boxes, inspection=None):
    """Count distinct abnormal sites, keeping multiple defect types at one site."""
    sites=set()
    groups={}
    unlocated=0
    for index,box in enumerate(boxes):
        site=box.get('site_id')
        site=str(site) if site not in (None,'') else None
        if site is not None:
            sites.add(site)
        else:
            unlocated+=1
        key=box.get('key') or box.get('label') or 'unknown'
        group=groups.setdefault(key,dict(key=key,label=box.get('label') or RULES.get(key,(key,))[0],
                                         tokens=set(),point_based=True))
        group['tokens'].add(('site',site) if site is not None else ('box',index))
        group['point_based']=group['point_based'] and site is not None
    templated=bool(inspection and inspection.get('mode')=='solder_sites')
    defects=[dict(key=group['key'],label=group['label'],count=len(group['tokens']),point_based=group['point_based'])
             for group in groups.values()]
    defects.sort(key=lambda entry:(-entry['count'],entry['key']))
    return dict(abnormal_sites=len(sites) if sites or (templated and not boxes) else None,
                checked_sites=inspection.get('checked_sites') if templated else None,
                defects=defects,unlocated_count=unlocated)


@app.get("/api/batches/<bid>")
def batch_detail(bid):
    with db() as conn:
        batch = conn.execute("SELECT * FROM batches WHERE id=?", (bid,)).fetchone()
        rows = conn.execute("""SELECT i.ordinal,i.filename,i.state,i.error,i.record_id,i.sampled,
            i.sample_revision,(i.image IS NOT NULL) AS retryable,r.decision,r.auto_decision,r.signed,r.reviewer,r.revision,r.model_version,
            r.classification,r.alignment,r.inspection,r.manual,r.comment,r.boxes AS detection_boxes,json_array_length(r.boxes) AS count
            FROM batch_items i LEFT JOIN records r ON i.record_id=r.id
            WHERE i.batch_id=? ORDER BY i.ordinal""", (bid,)).fetchall()
    if not batch:
        abort(404, "批量任务不存在")
    result = dict(batch)
    items = []
    for row in rows:
        item = dict(row)
        item["classification"] = json.loads(item["classification"]) if item["classification"] else None
        item["alignment"] = json.loads(item["alignment"]) if item["alignment"] else None
        item["inspection"] = json.loads(item["inspection"]) if item["inspection"] else None
        item['manual']=json.loads(item['manual']) if item['manual'] else []
        boxes=json.loads(item.pop('detection_boxes') or '[]')
        item['defect_summary']=summarize_point_defects(boxes,item['inspection']) if item['state']=='done' else None
        items.append(item)
    done = [r for r in items if r["state"] == "done"]
    passed = sum(r["decision"] == "合格" for r in done)
    auto_passed = sum(r["auto_decision"] == "合格" for r in done)
    decided = sum(r["decision"] in ("合格", "不合格") for r in done)
    auto_decided = sum(r["auto_decision"] in ("合格", "不合格") for r in done)
    sampled = [r for r in done if r["sampled"]]
    summary = dict(total=len(items), completed=len(done),
                   processed=sum(r["state"] in ("done", "failed", "deleted") for r in items),
                   uploaded=sum(r["state"] != "awaiting_upload" for r in items),
                   failed=sum(r["state"] == "failed" for r in items),
                   deleted=sum(r["state"] == "deleted" for r in items),
                   passed=passed, rejected=sum(r["decision"] == "不合格" for r in done),
                   auto_passed=auto_passed,
                   rate=round(100*passed/decided, 1) if decided else None,
                   auto_rate=round(100*auto_passed/auto_decided, 1) if auto_decided else None,
                   sampled=len(sampled),
                   sample_reviewed=sum(r["revision"] > r["sample_revision"] for r in sampled))
    defects = {}
    for item in done:
        classification = item.get("classification")
        if classification and classification.get("is_defect"):
            label = classification["label"]
            defects[label] = defects.get(label, 0) + 1
    summary["defects"] = defects
    result["inspection_context"] = json.loads(result["inspection_context"]) if result["inspection_context"] else None
    result.update(items=items, summary=summary)
    return jsonify(result)


@app.post("/api/batches/<bid>/items/<int:ordinal>")
def upload_item(bid, ordinal):
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_open_batch(conn, bid)
        item = conn.execute("SELECT state FROM batch_items WHERE batch_id=? AND ordinal=?", (bid, ordinal)).fetchone()
    if not item:
        abort(404, "图片条目不存在")
    if item["state"] in ("queued", "processing", "done", "deleted"):
        return jsonify(state=item["state"])
    try:
        img = snapshot()
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=94)
    except HTTPException as exc:
        with db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            require_open_batch(conn, bid)
            conn.execute("""UPDATE batch_items SET state='failed',error=?
                WHERE batch_id=? AND ordinal=? AND state IN ('awaiting_upload','failed')""",
                (exc.description, bid, ordinal))
        raise
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_open_batch(conn, bid)
        conn.execute("""UPDATE batch_items SET state='queued',image=?,error=NULL
            WHERE batch_id=? AND ordinal=? AND state IN ('awaiting_upload','failed')""",
            (buffer.getvalue(), bid, ordinal))
        batch = conn.execute("SELECT station,name FROM batches WHERE id=?", (bid,)).fetchone()
        filename = conn.execute("SELECT filename FROM batch_items WHERE batch_id=? AND ordinal=?", (bid, ordinal)).fetchone()[0]
        queue_training_import(conn, buffer.getvalue(), filename, batch["station"], batch["name"])
        conn.execute("UPDATE batches SET state='importing',completed_at=NULL WHERE id=? AND state IN ('ready','reviewing')", (bid,))
    flush_training_imports()
    work_ready.set()
    return jsonify(state="queued")


@app.get("/api/batches/<bid>/items/<int:ordinal>/image")
def batch_item_original(bid,ordinal):
    with db() as conn:
        row=conn.execute("""SELECT i.state,i.image,r.original_image,r.image AS detected_image,r.alignment
            FROM batch_items i LEFT JOIN records r ON r.id=i.record_id
            WHERE i.batch_id=? AND i.ordinal=?""",(bid,ordinal)).fetchone()
    if not row or row["state"]=='deleted':
        abort(404,"图片不存在")
    content=row["original_image"] or row["image"]
    if not content and not json.loads(row["alignment"] or "null"):
        content=row["detected_image"]
    if not content:
        abort(404,"原图尚未上传或已不可用")
    return send_file(io.BytesIO(content),mimetype="image/jpeg")


@app.post("/api/batches/<bid>/start")
def start_batch(bid):
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_open_batch(conn, bid)
        batch = conn.execute("SELECT * FROM batches WHERE id=?", (bid,)).fetchone()
        if not batch:
            abort(404, "批量任务不存在")
        if conn.execute("SELECT 1 FROM batch_items WHERE batch_id=? AND state='awaiting_upload'", (bid,)).fetchone():
            abort(409, "部分图片尚未上传，请选择原始图片继续导入")
        if batch["state"] not in ("reviewing","completed"):
            pending = conn.execute("SELECT 1 FROM batch_items WHERE batch_id=? AND state IN ('queued','processing')", (bid,)).fetchone()
            if pending and not batch["inspection_context"]:
                context = current_inspection_context(batch["station"])
                conn.execute("UPDATE batches SET inspection_context=? WHERE id=?",(json.dumps(context,ensure_ascii=False),bid))
            conn.execute("UPDATE batches SET state=?,completed_at=NULL WHERE id=?", ("running" if pending else "ready", bid))
    work_ready.set()
    return batch_detail(bid)


def current_inspection_context(station):
    try:
        reference = platform_runtime.alignment_reference(station)
        if reference:
            reference = dict(reference)
            reference["sha256"] = hashlib.sha256(Path(reference["path"]).read_bytes()).hexdigest()
        return dict(reference=reference,captured_at=now())
    except (ValueError,OSError,sqlite3.Error) as exc:
        abort(400,"无法应用当前标准图和点位模板："+str(exc))


@app.post("/api/batches/<bid>/reinspect")
def reinspect_batch(bid):
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        batch=conn.execute("SELECT * FROM batches WHERE id=?",(bid,)).fetchone()
        if not batch:
            abort(404,"批次不存在")
        rows=conn.execute("""SELECT i.ordinal,i.filename,i.state,r.alignment,
            (r.original_image IS NOT NULL OR i.image IS NOT NULL) AS has_original,
            (r.image IS NOT NULL) AS has_detected
            FROM batch_items i LEFT JOIN records r ON r.id=i.record_id
            WHERE i.batch_id=? AND i.state!='deleted' ORDER BY i.ordinal""",(bid,)).fetchall()
        if any(row["state"] in ("awaiting_upload","queued","processing") for row in rows):
            abort(409,"请等待当前批次全部上传和检测结束")
        if not rows:
            abort(400,"该批次没有可重新检测的图片")
        context=current_inspection_context(batch["station"])
        if not context["reference"] or not context["reference"].get("sites"):
            abort(400,"请先在模型训练平台设置标准图并保存点位模板")
        for row in rows:
            if not row["has_original"] and (not row["has_detected"] or json.loads(row["alignment"] or "null")):
                abort(409,"图片缺少原图，请重新上传："+row["filename"])
        new_id=uuid.uuid4().hex
        conn.execute("""INSERT INTO batches(id,created,station,name,state,inspection_context,source_batch_id)
            VALUES(?,?,?,?,'running',?,?)""",(new_id,now(),batch["station"],batch["name"][:68]+" · 重新检测",
                json.dumps(context,ensure_ascii=False),bid))
        for index,row in enumerate(rows):
            conn.execute("""INSERT INTO batch_items(batch_id,ordinal,filename,state,image)
                SELECT ?,?,i.filename,'queued',COALESCE(r.original_image,i.image,r.image)
                FROM batch_items i LEFT JOIN records r ON r.id=i.record_id
                WHERE i.batch_id=? AND i.ordinal=?""",(new_id,index,bid,row["ordinal"]))
    work_ready.set()
    return batch_detail(new_id)


@app.post("/api/batches/<bid>/retry")
def retry_batch(bid):
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_open_batch(conn, bid)
        batch=conn.execute("SELECT * FROM batches WHERE id=?", (bid,)).fetchone()
        if not batch:
            abort(404, "批量任务不存在")
        retried = conn.execute("""UPDATE batch_items SET state='queued',error=NULL
            WHERE batch_id=? AND state='failed' AND image IS NOT NULL""", (bid,))
        if retried.rowcount:
            if not batch["inspection_context"]:
                conn.execute("UPDATE batches SET inspection_context=? WHERE id=?",
                    (json.dumps(current_inspection_context(batch["station"]),ensure_ascii=False),bid))
            conn.execute("UPDATE batches SET state='running',completed_at=NULL WHERE id=?", (bid,))
    work_ready.set()
    return batch_detail(bid)


@app.post("/api/batches/<bid>/finish")
def finish_batch(bid):
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        batch = conn.execute("SELECT state FROM batches WHERE id=?", (bid,)).fetchone()
        if not batch:
            abort(404, "批量任务不存在")
        if batch["state"] == "completed":
            return batch_detail(bid)
        pending = conn.execute("""SELECT 1 FROM batch_items WHERE batch_id=?
            AND state IN ('awaiting_upload','queued','processing')""", (bid,)).fetchone()
        if pending:
            abort(409, "图片尚未全部上传或检测完成，请等待处理结束")
        capture_exceptions(conn, bid)
        conn.execute("UPDATE batches SET state='completed',completed_at=? WHERE id=?", (now(), bid))
    return batch_detail(bid)


@app.delete("/api/batches/<bid>")
def delete_archived_batch(bid):
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT state FROM batches WHERE id=?", (bid,)).fetchone()
        if not row:
            abort(404,"批次不存在或已删除")
        if row["state"] != "completed":
            abort(409,"请先完成批次归档，再从质量档案删除")
        conn.execute("""DELETE FROM reviews WHERE record_id IN
            (SELECT id FROM records WHERE batch_id=? UNION SELECT record_id FROM batch_items WHERE batch_id=?)""", (bid,bid))
        conn.execute("DELETE FROM exception_cases WHERE batch_id=?", (bid,))
        conn.execute("DELETE FROM reports WHERE batch_id=?", (bid,))
        conn.execute("DELETE FROM records WHERE batch_id=? OR id IN (SELECT record_id FROM batch_items WHERE batch_id=?)", (bid,bid))
        conn.execute("DELETE FROM batch_items WHERE batch_id=?", (bid,))
        conn.execute("DELETE FROM batches WHERE id=?", (bid,))
    return jsonify(deleted=bid)


def report_snapshot(conn, bid):
    batch = conn.execute("SELECT id,name,station,created,state FROM batches WHERE id=?", (bid,)).fetchone()
    if not batch:
        abort(404, "批量任务不存在")
    if batch["state"] not in ("reviewing", "completed"):
        abort(409, "请先完成批次归档，再导出报告")
    rows = conn.execute("""SELECT i.ordinal,i.filename,i.state,i.error,i.record_id,i.sampled,i.sample_revision,
        r.decision,r.auto_decision,r.signed,r.reviewer,r.revision,r.comment,r.model_version,
        r.model_sha256,r.classification,r.temp,r.process_parameters,r.analysis,json_array_length(r.boxes) AS count
        FROM batch_items i LEFT JOIN records r ON r.id=i.record_id
        WHERE i.batch_id=? ORDER BY i.ordinal""", (bid,)).fetchall()
    if any(r["state"] in ("awaiting_upload", "queued", "processing") for r in rows):
        abort(409, "图片仍在处理，请等待检测结束")
    snapshot = dict(batch={key:batch[key] for key in ("id","name","station","created")},
                    items=[dict(row) for row in rows])
    fingerprint = hashlib.sha256(json.dumps(snapshot,sort_keys=True,ensure_ascii=False).encode("utf-8")).hexdigest()
    return snapshot, fingerprint


@app.post("/api/batches/<bid>/report")
def create_report(bid):
    with db() as conn:
        conn.execute("BEGIN")
        snapshot, fingerprint = report_snapshot(conn, bid)
        cached = conn.execute("SELECT id,filename FROM reports WHERE batch_id=? AND fingerprint=?", (bid,fingerprint)).fetchone()
    if cached:
        report_id = cached["id"]
        filename = cached["filename"]
    else:
        created = now()
        try:
            document = build_pdf(snapshot, created)
        except Exception:
            app.logger.exception("Batch report generation failed")
            return jsonify(error="报告生成失败，批次尚未完成，请重试"), 500
        report_id = uuid.uuid4().hex
        safe_name = re.sub(r"[^\w.-]+","_",snapshot["batch"]["name"])[:60]
        filename = f"PCB_{safe_name}_{report_id[:8]}.pdf"
        with db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _, current = report_snapshot(conn, bid)
            if current != fingerprint:
                abort(409, "报告生成期间检验结果发生变化，请重新导出")
            conn.execute("""INSERT OR IGNORE INTO reports
                (id,batch_id,created,fingerprint,filename,document) VALUES (?,?,?,?,?,?)""",
                (report_id,bid,created,fingerprint,filename,document))
            saved = conn.execute("SELECT id,filename FROM reports WHERE batch_id=? AND fingerprint=?", (bid,fingerprint)).fetchone()
            report_id, filename = saved["id"], saved["filename"]
    return jsonify(id=report_id,filename=filename,download_url=f"/api/batches/{bid}/reports/{report_id}")


@app.get("/api/batches/<bid>/reports/<report_id>")
def download_report(bid, report_id):
    with db() as conn:
        report = conn.execute("SELECT filename,document FROM reports WHERE id=? AND batch_id=?", (report_id,bid)).fetchone()
    if not report:
        abort(404, "报告不存在")
    return send_file(io.BytesIO(report["document"]),mimetype="application/pdf",
                     as_attachment=True,download_name=report["filename"])


@app.post("/api/batches/<bid>/reports/<report_id>/complete")
def complete_export(bid, report_id):
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        report = conn.execute("SELECT fingerprint FROM reports WHERE id=? AND batch_id=?", (report_id,bid)).fetchone()
        if not report:
            abort(404, "报告不存在")
        _, fingerprint = report_snapshot(conn, bid)
        if fingerprint != report["fingerprint"]:
            abort(409, "检验结果已变化，请重新导出最新报告完成批次")
        capture_exceptions(conn, bid)
        completed_at = now()
        conn.execute("UPDATE batches SET state='completed',completed_at=? WHERE id=? AND state!='completed'", (completed_at,bid))
        conn.execute("UPDATE reports SET completed_at=COALESCE(completed_at,?) WHERE id=?", (completed_at,report_id))
    return batch_detail(bid)


@app.post("/api/batches/<bid>/sample")
def sample_batch(bid):
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        abort(400, "抽样参数无效")
    try:
        value = int(payload.get("value", 10))
        if float(payload.get("value", 10)) != value:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        abort(400, "请填写有效抽样数量")
    mode = payload.get("mode", "percent")
    if mode not in ("percent", "count") or value < 1 or (mode == "percent" and value > 100):
        abort(400, "抽样比例须为 1 至 100，张数须为正整数")
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_open_batch(conn, bid)
        batch = conn.execute("SELECT state FROM batches WHERE id=?", (bid,)).fetchone()
        if not batch:
            abort(404, "批量任务不存在")
        if batch["state"] not in ("ready", "reviewing", "completed"):
            abort(409, "请在本批次图片检测结束后抽样")
        candidates = conn.execute("""SELECT i.ordinal,r.revision FROM batch_items i
            JOIN records r ON i.record_id=r.id
            WHERE i.batch_id=? AND i.state='done' AND r.auto_decision='合格'""", (bid,)).fetchall()
        count = math.ceil(len(candidates)*value/100) if mode == "percent" else value
        if not candidates:
            abort(400, "本批次没有机器初判合格的图片可供抽样")
        if count > len(candidates):
            abort(400, f"机器初判合格图片仅 {len(candidates)} 张，请减少抽样数量")
        chosen = secrets.SystemRandom().sample(candidates, count)
        conn.execute("UPDATE batch_items SET sampled=0 WHERE batch_id=?", (bid,))
        conn.executemany("""UPDATE batch_items SET sampled=1,sample_revision=?
            WHERE batch_id=? AND ordinal=?""", [(r["revision"], bid, r["ordinal"]) for r in chosen])
    return batch_detail(bid)


def batch_worker():
    while True:
        work_ready.wait(timeout=2)
        work_ready.clear()
        flush_training_imports()
        while True:
            with db() as conn:
                conn.execute("BEGIN IMMEDIATE")
                item = conn.execute("""SELECT i.*,b.station,b.name,b.process_parameters,b.inspection_context FROM batch_items i
                    JOIN batches b ON i.batch_id=b.id
                    WHERE i.state='queued' AND b.state='running'
                    ORDER BY b.created,i.ordinal LIMIT 1""").fetchone()
                if not item:
                    conn.execute("""UPDATE batches SET state='ready' WHERE state='running'
                        AND NOT EXISTS (SELECT 1 FROM batch_items i WHERE i.batch_id=batches.id
                        AND i.state IN ('awaiting_upload','queued','processing'))""")
                    break
                conn.execute("UPDATE batch_items SET state='processing' WHERE batch_id=? AND ordinal=?",
                             (item["batch_id"], item["ordinal"]))
            try:
                with Image.open(io.BytesIO(item["image"])) as image:
                    img = image.convert("RGB")
                context=json.loads(item["inspection_context"]) if item["inspection_context"] else None
                boxes = detect(img, item["station"],inspection_context=context) if context is not None else detect(img,item["station"])
                with db() as conn:
                    rid = save_record(conn, img, boxes, item["station"], item["name"],
                                      batch_id=item["batch_id"], filename=item["filename"],
                                      temp=json.loads(item["process_parameters"] or "{}").get("temp"),
                                      training_source_key=hashlib.sha256(item["station"].encode("utf-8") + b"\0" + item["image"]).hexdigest(),
                                      process=json.loads(item["process_parameters"] or "{}"))
                    conn.execute("""UPDATE batch_items SET state='done',record_id=?,image=NULL,error=NULL
                        WHERE batch_id=? AND ordinal=?""", (rid, item["batch_id"], item["ordinal"]))
            except Exception as exc:
                app.logger.exception("Batch inference failed")
                with db() as conn:
                    message=exc.description if isinstance(exc,HTTPException) else '检测失败，可点击重试'
                    conn.execute("""UPDATE batch_items SET state='failed',error=?
                        WHERE batch_id=? AND ordinal=?""", (message,item["batch_id"],item["ordinal"]))


def start_worker():
    backfill_training_imports()
    with db() as conn:
        for batch in conn.execute("SELECT id FROM batches WHERE state='completed'").fetchall():
            capture_exceptions(conn, batch["id"])
        conn.execute("UPDATE batch_items SET state='queued' WHERE state='processing'")
    threading.Thread(target=batch_worker, daemon=True, name="pcb-batch-worker").start()
    work_ready.set()


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(409)
@app.errorhandler(413)
def error(exc):
    return jsonify(error="上传文件超过 16 MB" if exc.code == 413 else exc.description), exc.code


if __name__ == "__main__":
    import sys
    import urllib.request
    import webbrowser
    from waitress import serve
    from werkzeug.middleware.dispatcher import DispatcherMiddleware
    port = int(os.getenv("PCB_PORT", "8765"))
    if "--open-training" in sys.argv:
        address = f"http://127.0.0.1:{port}/"
        try:
            with urllib.request.urlopen(address+"api/model/status", timeout=2) as response:
                running = bool(json.load(response).get("version"))
        except (OSError, ValueError):
            running = False
        if running:
            webbrowser.open(address+"?page=training")
            sys.exit(0)
        threading.Timer(1, webbrowser.open, args=(address+"?page=training",)).start()
    import learning
    import platform_app
    learning.initialize()
    learning.deployment_status = model_info
    learning.start_worker()
    model_runtime.start_monitor()
    platform_runtime.start_monitor()
    start_worker()
    combined = DispatcherMiddleware(app, {"/training": learning.app, "/platform": platform_app.app})
    serve(combined, host=os.getenv("PCB_HOST", "127.0.0.1"), port=port)
