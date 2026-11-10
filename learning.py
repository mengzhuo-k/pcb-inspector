"""Local continual training for the fixed PCB layout."""
import csv
import hashlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, request, send_file
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
SEED = ROOT / "training/missing_connection"
BASE = SEED / "runs/trial_fit-2/weights/best.pt"
STORE = Path(os.environ.get("PCB_TRAIN_DATA", ROOT / "data/learning")).resolve()
STORE.mkdir(parents=True, exist_ok=True)
app = Flask(__name__, static_folder=str(ROOT / "static"))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ROWS = [(449,[857,917,947,977,1006,1036,1066,1095,1125,1156]),(507,[917,947]),
        (565,[917]),(594,[977,1007]),(624,[917]),(684,[856,947,978,1039,1069,1131,1163])]
SITES = [[x-12,y-12,x+12,y+12] for y,xs in ROWS for x in xs]
SITES += [[1008,295,1090,390],[890,816,998,906],[1087,862,1193,957]]
NAMES = [f"H{i:02}" for i in range(1,24)] + ["上方焊盘","下方左焊盘","下方右焊盘"]
reference = Image.open(SEED / "images/train/board_missing.png").convert("RGB")
wake = threading.Event()
worker_started = False
worker_start_lock = threading.Lock()
deployment_status = None


def connect():
    db = sqlite3.connect(STORE/"library.sqlite", timeout=30)
    db.row_factory = sqlite3.Row
    return db


def unpack(img):
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue(), hashlib.sha256(f"{img.size}:".encode()+img.tobytes()).hexdigest()


def norm_sites():
    w,h = reference.size
    return [[x/w,y/h,r/w,b/h] for x,y,r,b in SITES]


def initialize():
    with closing(connect()) as db, db:
        db.executescript("""
        BEGIN IMMEDIATE;
        CREATE TABLE IF NOT EXISTS samples(id TEXT PRIMARY KEY,name TEXT,image BLOB,
          boxes TEXT,sites TEXT,split TEXT,ready INTEGER,rev INTEGER);
        CREATE TABLE IF NOT EXISTS jobs(id INTEGER PRIMARY KEY,rev INTEGER,state TEXT,
          started INTEGER,ntrain INTEGER,nval INTEGER,parent TEXT,note TEXT DEFAULT '');
        """)
        if "deleted" not in [r["name"] for r in db.execute("PRAGMA table_info(samples)")]:
            db.execute("ALTER TABLE samples ADD COLUMN deleted INTEGER NOT NULL DEFAULT 0")
        if not db.execute("SELECT 1 FROM samples LIMIT 1").fetchone():
            for i,name in enumerate(["board_ok","board_missing"],1):
                img = Image.open(SEED/f"images/train/{name}.png").convert("RGB")
                data, key = unpack(img)
                boxes = norm_sites() if i==2 else []
                db.execute("INSERT INTO samples(id,name,image,boxes,sites,split,ready,rev) VALUES(?,?,?,?,?,?,?,?)",
                           (key,name+".png",data,json.dumps(boxes),"[]","train",1,i))


def features(gray):
    return cv2.SIFT_create(3500).detectAndCompute(gray,None)


ref_points, ref_desc = features(cv2.cvtColor(np.asarray(reference),cv2.COLOR_RGB2GRAY))


def align(img):
    # Match stationary board details. Require support around the inspection sites.
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    scale = min(1, 1920/max(img.size))
    if scale < 1:
        gray = cv2.resize(gray,None,fx=scale,fy=scale)
    points, desc = features(gray)
    if desc is None or len(desc)<2:
        return []
    good = [p[0] for p in cv2.BFMatcher().knnMatch(ref_desc,desc,k=2)
            if len(p)==2 and p[0].distance < .68*p[1].distance]
    if len(good)<25:
        return []
    src = np.float32([ref_points[m.queryIdx].pt for m in good])
    dst = np.float32([points[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src,dst,cv2.RANSAC,3)
    if H is None or mask.sum()<25 or mask.mean()<.45:
        return []
    hull = cv2.convexHull(src[mask.ravel()==1])
    output = []
    h,w = gray.shape
    for x,y,r,b in SITES:
        if cv2.pointPolygonTest(hull,((x+r)/2,(y+b)/2),False)<0:
            return []
        corners = np.float32([[[x,y],[r,y],[r,b],[x,b]]])
        poly = cv2.perspectiveTransform(corners,H)[0]/[w,h]
        rect = [*poly.min(axis=0),*poly.max(axis=0)]
        if not all(np.isfinite(v) and 0<=v<=1 for v in rect) or min(rect[2]-rect[0],rect[3]-rect[1])<.001:
            return []
        output.append(rect)
    return output


def next_rev(db):
    return db.execute("SELECT MAX(COALESCE((SELECT MAX(rev) FROM samples),0),"
                      "COALESCE((SELECT MAX(rev) FROM jobs),0))+1").fetchone()[0]


@app.get("/")
def home():
    return app.send_static_file("learning.html")


@app.get("/api/state")
def state():
    with closing(connect()) as db:
        samples=[dict(r) for r in db.execute("SELECT id,name,split,ready,rev,json_array_length(boxes) AS count FROM samples WHERE deleted=0 ORDER BY rev DESC")]
        jobs=[dict(r) for r in db.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT 30")]
    for job in jobs:
        if job["state"] == "完成" and job["nval"]:
            results = STORE / f"v{job['id']}/fit/results.csv"
            try:
                with results.open(encoding="utf-8") as stream:
                    final = list(csv.DictReader(stream))[-1]
                job["note"] = f"末轮验证 P {float(final['metrics/precision(B)']):.2f} · R {float(final['metrics/recall(B)']):.2f}"
            except (OSError, ValueError, KeyError, IndexError):
                job["note"] += " · 验证指标暂不可读"
    return jsonify(samples=samples,jobs=jobs,names=NAMES,
                   inference=deployment_status() if deployment_status else None)


@app.post("/api/upload")
def upload():
    f=request.files.get("image")
    mode=request.form.get("mode")
    split=request.form.get("split","train")
    if not f or mode not in ("normal","same","partial") or split not in ("train","val"):
        abort(400,"请选择样本类型和图片")
    try:
        with Image.open(f.stream) as raw:
            if raw.width*raw.height>24000000:
                abort(400,"图像超过 2400 万像素")
            img=ImageOps.exif_transpose(raw).convert("RGB")
    except (OSError,UnidentifiedImageError,Image.DecompressionBombError):
        abort(400,"无法读取图片")
    data,key=unpack(img)
    with closing(connect()) as db:
        if db.execute("SELECT 1 FROM samples WHERE id=? AND deleted=0",(key,)).fetchone():
            return jsonify(id=key,duplicate=True)
    sites=align(img) if mode!="normal" else []
    ready=mode=="normal" or (mode=="same" and bool(sites))
    boxes=sites if ready and mode=="same" else []
    with closing(connect()) as db, db:
        db.execute("BEGIN IMMEDIATE")
        inserted=db.execute("""INSERT INTO samples(id,name,image,boxes,sites,split,ready,rev)
                   VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                   name=excluded.name,boxes=excluded.boxes,sites=excluded.sites,
                   split=excluded.split,ready=excluded.ready,rev=excluded.rev,deleted=0
                   WHERE samples.deleted=1""",
                   (key,(f.filename or "PCB")[:200],data,json.dumps(boxes),json.dumps(sites),
                    split,int(ready),next_rev(db))).rowcount
    wake.set()
    return jsonify(id=key,duplicate=not inserted,ready=bool(ready))


@app.get("/api/sample/<key>")
def detail(key):
    with closing(connect()) as db:
        row=db.execute("SELECT name,boxes,sites,ready,rev FROM samples WHERE id=? AND deleted=0",(key,)).fetchone()
    if not row: abort(404)
    return jsonify(name=row["name"],boxes=json.loads(row["boxes"]),sites=json.loads(row["sites"]),rev=row["rev"])


@app.get("/api/image/<key>")
def image(key):
    with closing(connect()) as db:
        row=db.execute("SELECT image FROM samples WHERE id=? AND deleted=0",(key,)).fetchone()
    if not row: abort(404)
    return send_file(io.BytesIO(row[0]),mimetype="image/png")


@app.post("/api/sample/<key>")
def review(key):
    body=request.get_json() or {}
    boxes=body.get("boxes")
    if not isinstance(boxes,list) or len(boxes)>1000: abort(400,"框数量无效")
    for box in boxes:
        if not isinstance(box,list) or len(box)!=4 or not all(type(v) in (int,float) and 0<=v<=1 for v in box):
            abort(400,"标注坐标无效")
        if box[0]>=box[2] or box[1]>=box[3]: abort(400,"标注框面积无效")
    with closing(connect()) as db, db:
        db.execute("BEGIN IMMEDIATE")
        if not db.execute("UPDATE samples SET boxes=?,ready=1,rev=? WHERE id=? AND rev=? AND deleted=0",
                          (json.dumps(boxes),next_rev(db),key,body.get("rev"))).rowcount:
            abort(409,"样本已更新，请重新打开")
    wake.set()
    return jsonify(ok=True)


@app.delete("/api/sample/<key>")
def delete_sample(key):
    body = request.get_json() or {}
    with closing(connect()) as db, db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT rev,deleted FROM samples WHERE id=?", (key,)).fetchone()
        if not row or row["deleted"]:
            return jsonify(ok=True)
        if row["rev"] != body.get("rev"):
            abort(409, "图片标注已更新，请刷新后再删除")
        db.execute("UPDATE samples SET deleted=1,rev=? WHERE id=?", (next_rev(db), key))
    wake.set()
    return jsonify(ok=True)


@app.get("/api/model/<int:number>")
def model(number):
    with closing(connect()) as db:
        row=db.execute("SELECT state FROM jobs WHERE id=?",(number,)).fetchone()
    if not row or row[0]!="完成": abort(404)
    return send_file(STORE/f"v{number}/fit/weights/best.pt",as_attachment=True,download_name=f"pcb-missing-v{number}.pt")


@app.post("/api/retry")
def retry():
    with closing(connect()) as db, db:
        db.execute("UPDATE jobs SET state='待重试' WHERE id=(SELECT MAX(id) FROM jobs) AND state='失败'")
    wake.set()
    return jsonify(ok=True)


def run_pending():
    with closing(connect()) as db, db:
        db.execute("BEGIN IMMEDIATE")
        rev=db.execute("SELECT MAX(rev) FROM samples WHERE ready=1").fetchone()[0]
        last=db.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT 1").fetchone()
        if last and last["rev"]==rev and last["state"]!="待重试": return False
        rows=db.execute("SELECT id,split,boxes FROM samples WHERE ready=1 AND deleted=0 ORDER BY rev").fetchall()
        if not any(r["split"]=="train" and json.loads(r["boxes"]) for r in rows):
            return False
        previous=db.execute("SELECT id FROM jobs WHERE state='完成' ORDER BY id DESC LIMIT 1").fetchone()
        parent=STORE/f"v{previous[0]}/fit/weights/best.pt" if previous else BASE
        nt=sum(r["split"]=="train" for r in rows)
        nv=len(rows)-nt
        number=db.execute("INSERT INTO jobs(rev,state,started,ntrain,nval,parent) VALUES(?,?,?,?,?,?)",
                          (rev,"训练中",int(time.time()),nt,nv,f"v{previous[0]}" if previous else "原始 YOLO11n")).lastrowid
    folder=STORE/f"v{number}"
    try:
        folder.mkdir()
        for row in rows:
            split=row["split"]
            im=folder/"images"/split
            lab=folder/"labels"/split
            im.mkdir(parents=True,exist_ok=True)
            lab.mkdir(parents=True,exist_ok=True)
            with closing(connect()) as image_db:
                image_data=image_db.execute("SELECT image FROM samples WHERE id=?",(row["id"],)).fetchone()[0]
            (im/(row["id"]+".png")).write_bytes(image_data)
            lines=[f"0 {(x+r)/2} {(y+b)/2} {r-x} {b-y}" for x,y,r,b in json.loads(row["boxes"])]
            (lab/(row["id"]+".txt")).write_text("\n".join(lines),encoding="ascii")
        import yaml
        (folder/"data.yaml").write_text(yaml.safe_dump(dict(path=folder.as_posix(),train="images/train",
            val="images/val" if nv else "images/train",names=["missing_connection"])),encoding="utf-8")
        with (folder/"train.log").open("w",encoding="utf-8") as log:
            subprocess.run([sys.executable,str(ROOT/"learning_fit.py"),str(folder),str(parent)],
                           stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        if not (folder/"fit/weights/best.pt").is_file(): raise RuntimeError("缺少输出权重")
        state,note="完成","独立验证集" if nv else "仅训练集回测"
    except Exception as exc:
        state,note="失败",f"{exc} · 详见 v{number}/train.log"
    with closing(connect()) as db, db:
        db.execute("UPDATE jobs SET state=?,note=? WHERE id=?",(state,note,number))
    return True


def worker():
    import msvcrt
    with (STORE/"worker.lock").open("a+b") as guard:
        guard.seek(0)
        while True:
            try:
                msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
                break
            except OSError:
                wake.wait(5)
                wake.clear()
        with closing(connect()) as db, db:
            db.execute("UPDATE jobs SET state='待重试',note='服务重启' WHERE state='训练中'")
        while True:
            if not run_pending():
                wake.wait(5)
                wake.clear()
                time.sleep(1)


def start_worker():
    global worker_started
    with worker_start_lock:
        if worker_started:
            return
        worker_started = True
        threading.Thread(target=worker, daemon=True, name="pcb-training-worker").start()


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(409)
@app.errorhandler(413)
def error(exc):
    return jsonify(error=exc.description),exc.code


if __name__=="__main__":
    from waitress import serve
    port = int(os.environ.get("PCB_TRAIN_PORT",8766))
    if "--open" in sys.argv:
        import webbrowser
        from urllib.request import urlopen
        address = f"http://127.0.0.1:{port}/"
        try:
            with urlopen(address+"api/state",timeout=2) as response:
                running = "samples" in json.load(response)
        except (OSError, ValueError):
            running = False
        if running:
            webbrowser.open(address)
            sys.exit(0)
        threading.Timer(1,webbrowser.open,args=(address,)).start()
    initialize()
    start_worker()
    serve(app,host="127.0.0.1",port=port)

