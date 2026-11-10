"""Generate a model from the labels saved in the PCB platform.

The employee UI intentionally has no training pages. This CLI is the operator
entry point: it creates a detection dataset from all saved boxes in one project,
starts a training job, waits for it, and optionally publishes the result.

Examples:
    .venv\\Scripts\\python.exe training\\generate_model.py --project "aa·cc" --epochs 20
    .venv\\Scripts\\python.exe training\\generate_model.py --project "aa·cc" --publish --station SMT-01
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import requests

DEFAULT_BASE = "http://127.0.0.1:8765/platform"


def call(base, method, path, **kwargs):
    response = requests.request(method, base + path, timeout=60, **kwargs)
    try:
        data = response.json()
    except ValueError:
        data = {}
    if not response.ok:
        raise RuntimeError(data.get("error") or f"HTTP {response.status_code}")
    return data


def pick_project(base, name):
    projects = call(base, "GET", "/api/projects")
    if not projects:
        raise RuntimeError("平台还没有项目，请先在 /platform 新建项目并上传、框缺陷")
    if not name:
        print("可用项目：")
        for project in projects:
            print(f"  {project['id']}  {project['name']}  {project.get('station','')}")
        raise RuntimeError("请用 --project 指定项目名称或 id")
    for project in projects:
        if name in (project["id"], project["name"]):
            return project
    for project in projects:
        if name.lower() in project["name"].lower():
            return project
    raise RuntimeError("找不到项目：" + name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--project", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="training/yolo11n.pt")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--station", default="")
    args = parser.parse_args()
    base = args.base.rstrip("/")
    project = pick_project(base, args.project)
    print(f"项目：{project['name']}（{project['id']}）")
    try:
        dataset = call(base, "POST", f"/api/projects/{project['id']}/datasets", json={
            "template": "det", "split_policy": "random",
            "val_ratio": .2, "test_ratio": .2, "seed": 20261110,
            "created_by": "generate_model",
        })
    except RuntimeError as exc:
        raise SystemExit(f"无法创建训练集：{exc}\n请先在平台上框出并保存足够的缺陷图片。")
    print(f"训练集 v{dataset['version']}：train {dataset['counts']['train']} / "
          f"val {dataset['counts']['val']} / test {dataset['counts']['test']}")
    job = call(base, "POST", f"/api/datasets/{dataset['id']}/train", json={
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "device": args.device, "model": args.model,
    })
    jid = job["id"]
    print(f"训练任务 {jid} 已启动，轮询进度…")
    last = ""
    while True:
        detail = call(base, "GET", f"/api/jobs/{jid}")
        tail = (detail.get("log_tail") or "").strip()
        if tail and tail != last:
            print(tail[-600:])
            last = tail
        if detail["state"] in ("done", "failed", "cancelled"):
            break
        time.sleep(5)
    if detail["state"] != "done":
        raise SystemExit(f"训练失败：{detail.get('error') or detail['state']}")
    models = call(base, "GET", f"/api/projects/{project['id']}/models")
    model = next((item for item in models if item.get("job_id") == jid), None)
    if not model:
        raise SystemExit("训练完成，但模型没有注册")
    print(f"模型已生成：{model['version']}  权重 {model['weights']}")
    print("指标：" + json.dumps(model.get("metrics"), ensure_ascii=False))
    if args.publish:
        station = args.station or project.get("station") or ""
        call(base, "POST", f"/api/models/{model['id']}/publish", json={"station": station})
        print(f"已发布到工位：{station or '未填写'}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print("错误：" + str(exc), file=sys.stderr)
        raise SystemExit(1)
