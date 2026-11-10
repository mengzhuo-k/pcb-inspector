"""Step 4: Gradio web demo - few-shot cold start + synthetic augmentation + human review.

Run:  python -m demo.app_gradio        (http://127.0.0.1:7860)
Env:  PCB_DEMO_HOST=0.0.0.0 to reach it from a phone on the same LAN, PCB_DEMO_PORT=7860.
Everything stays on this machine: share links and analytics are disabled.
"""
from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

from . import config as C  # sets the offline / no-telemetry environment before gradio is imported
import gradio as gr

from .pipeline import COLOURS, Inspector, SUGGESTIONS, draw_findings, _iou, _overlap_of
from .register import imread, imwrite, list_images
from .synth import synthesize

INSPECTOR: Inspector | None = None
MAX_SIDE = 2000
LOG_FIELDS = ["time", "work_order", "board_sn", "machine_verdict", "score", "threshold", "machine_findings",
              "inspector", "decision", "defect_types", "note", "registration", "image"]
ROTATIONS = {"自动": None, "0°": 0, "90°": 1, "180°": 2, "270°": 3}
LOG_HEADERS = {"time": "时间", "work_order": "工单", "board_sn": "板号", "machine_verdict": "机器判定", "score": "异常分", "threshold": "阈值",
               "machine_findings": "机器发现", "inspector": "质检员", "decision": "复核结论", "defect_types": "缺陷类型", "note": "备注", "registration": "对齐"}
VERDICT_STYLE = {
    "PASS": ("#0f9d58", "合格 PASS", "未发现超出正常波动的区域，可按抽检规则放行。"),
    "REVIEW": ("#f29900", "待复核 REVIEW", "存在可疑区域，请质检员重点查看标出的位置。"),
    "NG": ("#d93025", "疑似不合格 NG", "存在明显异于标准板的区域，请质检员确认并签字。"),
    "RETAKE": ("#5f6368", "需重拍 RETAKE", "与标准板对齐置信度不足，不强行给出结果。请重拍，或到「手动四点校正」页点选四角。"),
}

CSS = """
.gradio-container {max-width: 1500px !important}
#hero {background: linear-gradient(120deg,#0b3d2e 0%,#11694b 55%,#1b9a6c 100%); border-radius: 16px; padding: 22px 28px; color: #fff}
#hero h1 {color:#fff; font-size: 26px; margin: 0 0 6px 0; letter-spacing: .5px}
#hero p {color:#d7f5e6; margin: 0; font-size: 14px}
.chips {display:flex; gap:10px; flex-wrap:wrap; margin-top:14px}
.chip {background: rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.25); border-radius: 999px; padding: 4px 12px; font-size: 12.5px; color:#fff}
.chip b {color:#b9ffd9}
.verdict {border-radius: 14px; padding: 16px 20px; color: #fff; box-shadow: 0 4px 14px rgba(0,0,0,.12)}
.verdict .big {font-size: 28px; font-weight: 700; letter-spacing: 1px}
.verdict .sub {opacity: .95; font-size: 14px; margin-top: 4px}
.meter {height: 10px; border-radius: 6px; background: rgba(255,255,255,.3); margin-top: 12px; position: relative}
.meter > i {display:block; height:100%; border-radius:6px; background:#fff}
.meter > u {position:absolute; top:-4px; width:2px; height:18px; background:#111}
.facts {display:flex; gap:18px; flex-wrap:wrap; margin-top:10px; font-size: 12.5px; opacity:.95}
.kpis {display:grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr)); gap:12px}
.kpi {border:1px solid var(--border-color-primary); border-radius: 12px; padding: 14px 16px; background: var(--background-fill-secondary)}
.kpi .n {font-size: 26px; font-weight: 700; color: var(--body-text-color)}
.kpi .l {font-size: 12.5px; color: var(--body-text-color-subdued)}
.note {font-size: 12.5px; color: var(--body-text-color-subdued)}
.steps {display:flex; gap:8px; flex-wrap:wrap; margin: 6px 0 2px}
.step {border:1px solid var(--border-color-primary); border-radius: 10px; padding: 6px 12px; font-size: 13px; background: var(--background-fill-secondary)}
.step b {color:#11875d}
"""


def inspector() -> Inspector:
    global INSPECTOR
    if INSPECTOR is None:
        INSPECTOR = Inspector()
    return INSPECTOR


def _read_json(path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _rgb(bgr):
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _shrink(bgr: np.ndarray) -> np.ndarray:
    scale = MAX_SIDE / max(bgr.shape[:2])
    return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr


# ---------------------------------------------------------------- header
def hero_html() -> str:
    pc, yolo = _read_json(C.PATCHCORE_REPORT), _read_json(C.YOLO_REPORT)
    normals = len(list_images(C.WARPED / "normal"))
    chips = [f"<span class='chip'>标准板样本 <b>{normals}</b> 张（仅正常板）</span>"]
    chips.append(f"<span class='chip'>PatchCore 阈值 <b>{pc['threshold']}</b></span>" if pc else "<span class='chip'>PatchCore 未训练</span>")
    chips.append(f"<span class='chip'>YOLO 合成验证集 mAP50 <b>{yolo['map50']}</b></span>" if yolo else "<span class='chip'>YOLO 未训练</span>")
    chips.append("<span class='chip'>纯 CPU · 本地运行 · 数据不出厂</span>")
    return ("<div id='hero'><h1>PCB 焊接质检 · 小样本冷启动演示</h1>"
            "<p>四点透视校正 → PatchCore 异常热力图（只学正常板）→ 合成缺陷训练的轻量 YOLO 给出类别 → 质检员复核签字</p>"
            f"<div class='chips'>{''.join(chips)}</div></div>")


# ---------------------------------------------------------------- inspection
def verdict_html(result: dict | None) -> str:
    if result is None:
        return "<div class='verdict' style='background:#5f6368'><div class='big'>等待检测</div><div class='sub'>上传/拍摄一张板子照片，或点击下方样例。</div></div>"
    colour, title, text = VERDICT_STYLE[result["verdict"]]
    reg = result.get("registration") or {}
    facts = []
    if reg.get("reason") == "manual four-point":
        facts.append("对齐：手动四点")
    elif reg:
        facts.append(f"对齐内点 {reg.get('inliers', 0)}/{reg.get('matches', 0)}")
    meter = ""
    if result.get("score") is not None:
        score, threshold = result["score"], result["threshold"]
        full = threshold * 2
        meter = (f"<div class='meter'><i style='width:{min(100, score / full * 100):.0f}%'></i>"
                 f"<u style='left:40%' title='复核线'></u><u style='left:50%' title='阈值'></u></div>")
        facts.append(f"异常分 {score:.2f} / 阈值 {threshold:.2f}（复核线 {threshold * 0.8:.2f}）")
    if result.get("findings"):
        facts.append(f"标出 {len(result['findings'])} 处")
    if result.get("hints"):
        facts.append(f"YOLO 未确认提示 {len(result['hints'])} 处（虚线）")
    if result.get("timing", {}).get("total"):
        facts.append(f"耗时 {result['timing']['total']:.1f}s (CPU)")
    if result["verdict"] == "RETAKE" and reg.get("reason"):
        facts.append(reg["reason"])
    return (f"<div class='verdict' style='background:{colour}'><div class='big'>{title}</div><div class='sub'>{text}</div>"
            f"{meter}<div class='facts'>{''.join(f'<span>{f}</span>' for f in facts)}</div></div>")


def legend_html() -> str:
    items = [("anomaly", "未知异常")] + [(c, C.CLASSES_ZH[c]) for c in C.CLASSES]
    spans = "".join(f"<span style='display:inline-flex;align-items:center;gap:5px;margin-right:14px'>"
                    f"<i style='width:12px;height:12px;border-radius:3px;border:1px solid #8886;background:rgb{COLOURS[k]}'></i>{name}</span>" for k, name in items)
    return f"<div class='note'>{spans}（图上小框只标序号，对应下表）</div>"


def findings_table(result: dict | None) -> list[list]:
    rows = []
    for i, f in enumerate((result or {}).get("findings", []), 1):
        name = "未知异常" if f["cls"] == "anomaly" else C.CLASSES_ZH[f["cls"]]
        rows.append([i, name, f["region"], f["level"], "-" if f["anomaly"] is None else round(f["anomaly"], 2),
                     "-" if f["conf"] is None else round(f["conf"], 2), f["source"], f["suggestion"]])
    return rows


def _choices_from(result: dict) -> list[str]:
    picked = []
    for f in result.get("findings", []):
        if f["cls"] in C.CLASSES:
            choice = C.REVIEW_CHOICES[C.CLASSES.index(f["cls"])]
            if choice not in picked:
                picked.append(choice)
    return picked


def _present(result: dict):
    """Outputs shared by automatic and manual inspection."""
    slim = {k: v for k, v in result.items() if k not in ("heatmap",)}
    decision = {"PASS": "合格", "NG": "不合格", "REVIEW": None, "RETAKE": None}[result["verdict"]]
    slider = None
    if result.get("warped") is not None and result.get("heatmap") is not None:
        slider = (_rgb(result["warped"]), _rgb(result["heatmap"]))
    return (verdict_html(result), _rgb(result.get("annotated")), slider, findings_table(result), slim,
            gr.update(value=decision), gr.update(value=_choices_from(result)), "")


def run_inspection(image_rgb, sensitivity):
    if image_rgb is None:
        raise gr.Error("请先上传或拍摄一张照片 / upload a photo first")
    bgr = _shrink(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return _present(inspector().inspect(bgr, sensitivity=sensitivity))


def load_sample(evt: gr.SelectData):
    path = SAMPLES[evt.index][0]
    return _rgb(_shrink(imread(path)))


def save_review(result, name, decision, defect_types, note, work_order, board_sn):
    if not result or result.get("warped") is None:
        raise gr.Error("没有可归档的检测结果（需重拍的照片不归档）")
    if not (name or "").strip():
        raise gr.Error("请填写质检员姓名后再签字")
    if not decision:
        raise gr.Error("请选择复核结论")
    if decision != "合格" and not defect_types:
        raise gr.Error("判为不合格/返工时请至少勾选一种缺陷类型（这就是后续训练用的标签）")
    C.REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now()
    image_name = f"{stamp:%Y%m%d_%H%M%S}_{int(time.time() * 1000) % 1000:03d}.jpg"
    imwrite(C.REVIEW_DIR / image_name, result["annotated"])
    imwrite(C.REVIEW_DIR / image_name.replace(".jpg", "_view.png"), result["warped"])  # clean registered image = future training sample
    reg = result.get("registration") or {}
    row = {
        "time": stamp.isoformat(timespec="seconds"), "work_order": (work_order or "").strip(), "board_sn": (board_sn or "").strip(),
        "machine_verdict": result["verdict"], "score": "" if result.get("score") is None else round(result["score"], 3),
        "threshold": "" if result.get("threshold") is None else round(result["threshold"], 3),
        "machine_findings": "; ".join(f"{f['cls']}@{f['region']}" for f in result.get("findings", [])),
        "inspector": name.strip(), "decision": decision, "defect_types": "; ".join(defect_types or []), "note": (note or "").strip(),
        "registration": "manual" if reg.get("reason") == "manual four-point" else f"auto:{reg.get('inliers', 0)}", "image": image_name,
    }
    new = not C.REVIEW_LOG.exists()
    with C.REVIEW_LOG.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)
    return f"✅ 已归档：{row['time']} · {name.strip()} 签字 · 结论「{decision}」 · 机器判定 {result['verdict']}"


# ---------------------------------------------------------------- manual four-point
def _draw_points(bgr: np.ndarray, points: list) -> np.ndarray:
    canvas = bgr.copy()
    radius = max(6, canvas.shape[1] // 120)
    for i, (x, y) in enumerate(points):
        cv2.circle(canvas, (int(x), int(y)), radius, (0, 0, 255), -1)
        cv2.circle(canvas, (int(x), int(y)), radius + 3, (255, 255, 255), 2)
        cv2.putText(canvas, str(i + 1), (int(x) + radius + 4, int(y) + radius), cv2.FONT_HERSHEY_SIMPLEX, radius / 8, (0, 255, 255), 3)
    if len(points) == 4:
        from .register import order_corners
        cv2.polylines(canvas, [order_corners(points).astype(np.int32)], True, (0, 255, 255), 3)
    return canvas


def manual_load(image_rgb):
    if image_rgb is None:
        raise gr.Error("请先在「检测工位」页上传照片")
    bgr = _shrink(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return _rgb(bgr), bgr, [], "依次点击板子的四个角（顺序不限）。"


def manual_click(evt: gr.SelectData, source, points):
    if source is None:
        raise gr.Error("请先点击「载入当前照片」")
    points = (points + [list(evt.index)])[-4:] if len(points) < 4 else [list(evt.index)]
    hint = f"已选 {len(points)}/4 个角。" + (" 可以点击「校正并检测」。" if len(points) == 4 else "")
    return _rgb(_draw_points(source, points)), points, hint


def manual_clear(source):
    return (_rgb(source) if source is not None else None), [], "已清空，请重新点击四个角。"


def manual_inspect(source, points, rotate, sensitivity):
    if source is None or len(points) != 4:
        raise gr.Error("需要正好四个角点")
    result = inspector().inspect(source, manual_corners=points, rotate=ROTATIONS[rotate], sensitivity=sensitivity)
    return (*_present(result), gr.Tabs(selected="inspect"))


# ---------------------------------------------------------------- synthetic lab
def run_lab(board_name, seed, count, sensitivity):
    path = C.WARPED / "normal" / board_name
    if not path.exists():
        raise gr.Error("请选择一块标准板")
    rng = np.random.default_rng(int(seed))
    image, labels = synthesize(imread(path), rng, n_defects=int(count))
    truth = [{"box": [x1, y1, x2, y2], "cls": C.CLASSES[c]} for c, x1, y1, x2, y2 in labels]
    result = inspector().inspect_warped(image, sensitivity)
    rows, located, named, heat = [], 0, 0, 0
    for i, t in enumerate(truth, 1):
        hits = [f for f in result["findings"] if _iou(f["box"], t["box"]) > 0.1 or _overlap_of(t["box"], f["box"]) > 0.5]
        by_heat = any(f["source"] != "YOLO" for f in hits)
        hint_hits = [h for h in result["hints"] if _iou(h["box"], t["box"]) > 0.3]
        found = bool(hits)
        right = any(f["cls"] == t["cls"] for f in hits + hint_hits)
        located += found
        heat += by_heat
        named += right
        seen = "、".join(sorted({("未知异常" if f["cls"] == "anomaly" else C.CLASSES_ZH[f["cls"]]) for f in hits + hint_hits})) or "-"
        rows.append([i, C.CLASSES_ZH[t["cls"]], "✅" if found else ("仅YOLO提示" if hint_hits else "❌ 漏检"), seen, "✅" if right else "—"])
    extra = sum(1 for f in result["findings"] if not any(_iou(f["box"], t["box"]) > 0.1 or _overlap_of(t["box"], f["box"]) > 0.5 for t in truth))
    total = max(1, len(truth))
    summary = (f"<div class='kpis'><div class='kpi'><div class='n'>{len(truth)}</div><div class='l'>注入缺陷数</div></div>"
               f"<div class='kpi'><div class='n'>{located}/{len(truth)}</div><div class='l'>被系统标出（{located / total:.0%}）</div></div>"
               f"<div class='kpi'><div class='n'>{heat}/{len(truth)}</div><div class='l'>其中热力图也同意</div></div>"
               f"<div class='kpi'><div class='n'>{named}/{len(truth)}</div><div class='l'>YOLO 类别正确（{named / total:.0%}）</div></div>"
               f"<div class='kpi'><div class='n'>{extra}</div><div class='l'>多余标记（误报）</div></div>"
               f"<div class='kpi'><div class='n'>{result['verdict']}</div><div class='l'>整板判定</div></div></div>")
    truth_img = draw_findings(image, [{**t} for t in truth])
    return _rgb(truth_img), _rgb(result["annotated"]), _rgb(result.get("heatmap")), summary, rows


# ---------------------------------------------------------------- records dashboard
def load_records() -> pd.DataFrame:
    if not C.REVIEW_LOG.exists():
        return pd.DataFrame(columns=LOG_FIELDS)
    return pd.read_csv(C.REVIEW_LOG, encoding="utf-8-sig", dtype=str).fillna("")


def dashboard():
    df = load_records()
    total = len(df)
    passed = int((df["decision"] == "合格").sum()) if total else 0
    machine_ok = df["machine_verdict"].map({"PASS": "合格", "NG": "不合格"}) if total else pd.Series(dtype=str)
    decided = machine_ok.notna().sum() if total else 0
    agree = int((machine_ok == df["decision"].replace({"返工": "不合格"})).sum()) if total else 0
    auto_pass = int((df["machine_verdict"] == "PASS").sum()) if total else 0
    kpis = [(total, "已复核签字（块）"), (f"{passed / total:.0%}" if total else "-", "人工判定合格率"),
            (f"{agree}/{decided}" if decided else "-", "机器明确判定与人工一致"),
            (f"{auto_pass / total:.0%}" if total else "-", "机器判 PASS 占比（可降为抽检）")]
    html = "<div class='kpis'>" + "".join(f"<div class='kpi'><div class='n'>{n}</div><div class='l'>{l}</div></div>" for n, l in kpis) + "</div>"
    counts: dict[str, int] = {}
    for cell in df["defect_types"] if total else []:
        for item in filter(None, (s.strip() for s in cell.split(";"))):
            counts[item.split(" (")[0]] = counts.get(item.split(" (")[0], 0) + 1
    pareto = pd.DataFrame(sorted(counts.items(), key=lambda kv: -kv[1]), columns=["缺陷类型", "数量"])
    table = (df.drop(columns=["image"]).iloc[::-1] if total else df.drop(columns=["image"])).rename(columns=LOG_HEADERS)
    download = gr.update(value=str(C.REVIEW_LOG) if C.REVIEW_LOG.exists() else None, visible=C.REVIEW_LOG.exists())
    return html, pareto, table, download


# ---------------------------------------------------------------- about
def about_markdown() -> str:
    pc, yolo, reg = _read_json(C.PATCHCORE_REPORT), _read_json(C.YOLO_REPORT), _read_json(C.REGISTRATION_REPORT)
    lines = ["### 这套演示实际做了什么（实测数字，未美化）", ""]
    if reg:
        ok = sum(1 for r in reg.values() if r["ok"])
        lines.append(f"- **配准**：{ok}/{len(reg)} 张原始照片（横/竖/倒置/强倾斜、白底/绿底）自动找到四角并校正到 {C.CANON_W}×{C.CANON_H} 标准视角；最少内点 {min(r['inliers'] for r in reg.values())}。")
    if pc:
        loo = pc.get("loo_scores", {})
        over = sum(1 for v in loo.values() if v > pc["threshold"])
        lines.append(f"- **PatchCore**：只用 {len(loo)} 张正常板建库。留一法（每张正常板用其余板打分）得分 {sorted(loo.values())}，阈值 {pc['threshold']}；"
                     f"**{over}/{len(loo)} 张正常板在留一法下会被误判**（它们本身不典型：一张带三防漆、一张强倾斜）。")
        for name, item in pc.get("abnormal", {}).items():
            lines.append(f"- **唯一的真实缺陷图** `{name}`：得分 {item['score']}，{'高于' if item['is_anomalous'] else '低于'}阈值。这是 n=1，只能说明“能演示”，不能说明检出率。")
    sens = pc.get("synthetic_sensitivity_leave_one_out")
    if sens:
        text = "、".join(f"{C.CLASSES_ZH[k]} {v['found']}/{v['injected']}" for k, v in sens.items())
        lines.append(f"- **热力图对注入缺陷的灵敏度**（留一法、合成缺陷、达到复核线的比例）：{text}。空孔和缺件基本都能亮，小的桥接/焊盘损伤/偏移只有一半左右，所以高置信度的 YOLO 框会单独送复核。")
    if yolo:
        lines.append("- 合成空孔的外观是看过那张真实缺陷图之后设计的，所以 YOLO 在该图上认出空孔不是独立验证。")
        lines.append(f"- **YOLO（合成缺陷）**：在从未参与训练的两块正常板的**合成**验证图上 mAP50={yolo['map50']}、P={yolo['precision']}、R={yolo['recall']}。"
                     f"分类别 mAP50：{yolo.get('per_class_map50', {})}。合成→真实存在域差，所以它的框只有在热力图同意时才算发现，否则只画虚线提示。")
    lines += ["", "### 边界", "",
              "- 库里只有 9 张正常板，演示页里的“正常样例”都在库内，判 PASS 是必然的，不代表新板的误报率。",
              "- 虚焊/冷焊俯拍多数不可见，没有合成、也不声称能检出；复核时由质检员勾选。",
              "- 高元件（变压器、电容、继电器）有视差，容差图会自动放宽这些位置，代价是那里的缺陷更难发现。",
              "- 根因建议是规则表，不是模型结论。", "",
              "### 上线前要做的", "",
              "- 固定机位与光源，每种板采 30–50 张正常板；用复核页积累真实缺陷标签，每类几十张后再训练真实数据的检测器。",
              "- 阈值在留出的正常板 + 真实缺陷板上重新标定，并报告漏检/误报。"]
    return "\n".join(lines)


# ---------------------------------------------------------------- UI
SAMPLES: list[tuple[str, str]] = []


def _collect_samples() -> list[tuple[str, str]]:
    """Small local thumbnails for the sample strip (written under data/demo, never uploaded)."""
    samples = []
    thumbs = C.OUT / "thumbs"
    for label, folder in (("缺陷板", C.RAW_ABNORMAL), ("正常板", C.RAW_NORMAL)):
        for i, path in enumerate(list_images(folder), 1):
            thumb = thumbs / f"{folder.name}_{i:02d}.jpg"
            if not thumb.exists():
                image = imread(path)
                scale = 320 / max(image.shape[:2])
                imwrite(thumb, cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
            samples.append((str(path), str(thumb), f"{label} {i}"))
    return samples


def build() -> gr.Blocks:
    global SAMPLES
    samples = _collect_samples()
    SAMPLES = [(s[0], s[2]) for s in samples]
    boards = [p.name for p in list_images(C.WARPED / "normal")]

    with gr.Blocks(title="PCB 焊接质检演示", analytics_enabled=False) as demo:
        gr.HTML(hero_html())
        state = gr.State(None)
        manual_src = gr.State(None)
        manual_pts = gr.State([])

        with gr.Tabs() as tabs:
            # ---------------- inspect
            with gr.Tab("① 检测工位", id="inspect"):
                gr.HTML("<div class='steps'><span class='step'><b>1</b> 拍照/上传</span><span class='step'><b>2</b> 自动四点校正</span>"
                        "<span class='step'><b>3</b> 热力图 + 缺陷类别</span><span class='step'><b>4</b> 质检员复核签字</span></div>")
                with gr.Row():
                    with gr.Column(scale=4):
                        photo = gr.Image(label="板子照片（任意角度，手机可直接拍照）", sources=["upload", "webcam", "clipboard"], type="numpy", height=360)
                        gallery = gr.Gallery(value=[(s[1], s[2]) for s in samples], label="样例（点击载入并检测；正常板均在标准库内）", columns=5, rows=2,
                                             height=250, allow_preview=False, object_fit="cover")
                        sensitivity = gr.Slider(0.7, 1.5, value=1.0, step=0.05, label="灵敏度（>1 更敏感：漏检更少、送复核更多）")
                        go = gr.Button("开始检测", variant="primary", size="lg")
                        with gr.Row():
                            work_order = gr.Textbox(label="工单号", placeholder="例如 WO-2311", scale=1)
                            board_sn = gr.Textbox(label="板号/序列号", placeholder="可扫码枪输入", scale=1)
                    with gr.Column(scale=7):
                        verdict = gr.HTML(verdict_html(None))
                        with gr.Row():
                            annotated = gr.Image(label="标注结果（实线=发现，虚线=YOLO 未确认提示）", height=640, interactive=False)
                            slider = gr.ImageSlider(label="标准视角 ⟷ 异常热力图（拖动对比）", height=640, interactive=False)
                        gr.HTML(legend_html())
                        table = gr.Dataframe(headers=["#", "缺陷", "区域", "级别", "异常分", "YOLO置信度", "来源", "规则建议"],
                                             label="发现列表", interactive=False, wrap=True)
                with gr.Accordion("④ 质检员复核签字", open=True):
                    with gr.Row():
                        reviewer = gr.Textbox(label="质检员", placeholder="姓名/工号", scale=1)
                        decision = gr.Radio(["合格", "不合格", "返工"], label="复核结论", scale=2)
                        defect_types = gr.CheckboxGroup(C.REVIEW_CHOICES, label="缺陷类型（已按机器结果预选，可改）", scale=4)
                    note = gr.Textbox(label="备注（手机上可直接用输入法语音听写）", lines=1)
                    with gr.Row():
                        sign = gr.Button("复核签字并归档", variant="primary")
                        saved = gr.Markdown("")

            # ---------------- manual four-point
            with gr.Tab("② 手动四点校正", id="manual"):
                gr.Markdown("自动对齐失败（判为“需重拍”）或换了新板型时使用：点四个角 → 透视变换到标准视角 → 送检。")
                with gr.Row():
                    with gr.Column(scale=6):
                        manual_view = gr.Image(label="点击四个角", interactive=False, height=620)
                    with gr.Column(scale=3):
                        manual_hint = gr.Markdown("先在「检测工位」页上传照片，再点「载入当前照片」。")
                        manual_load_btn = gr.Button("载入当前照片")
                        rotate = gr.Radio(list(ROTATIONS), value="自动", label="朝向（自动=与标准板最像的方向；端子排应朝下）")
                        manual_clear_btn = gr.Button("清空角点")
                        manual_go = gr.Button("校正并检测", variant="primary")

            # ---------------- synthetic lab
            with gr.Tab("③ 合成缺陷实验", id="lab"):
                gr.Markdown("在一块正常板上用 OpenCV **现场注入**缺陷，再让系统去找。这就是 YOLO 训练数据的来源；"
                            "`normal_07/08` 没有参与 YOLO 训练。注意：这些板都在 PatchCore 的正常库里，所以这是偏有利的测试。")
                with gr.Row():
                    lab_board = gr.Dropdown(boards, value=boards[-2] if len(boards) > 1 else None, label="标准板")
                    lab_seed = gr.Number(value=7, precision=0, label="随机种子")
                    lab_count = gr.Slider(1, 8, value=5, step=1, label="注入缺陷数")
                    lab_go = gr.Button("注入并检测", variant="primary")
                lab_summary = gr.HTML()
                with gr.Row():
                    lab_truth = gr.Image(label="注入的缺陷（真值）", height=640, interactive=False)
                    lab_found = gr.Image(label="系统检出", height=640, interactive=False)
                    lab_heat = gr.Image(label="异常热力图", height=640, interactive=False)
                lab_table = gr.Dataframe(headers=["#", "注入类型", "是否标出", "系统给出的类别", "类别正确"], interactive=False)

            # ---------------- records
            with gr.Tab("④ 记录与看板", id="records") as records_tab:
                kpi_html = gr.HTML()
                with gr.Row():
                    pareto = gr.BarPlot(x="缺陷类型", y="数量", title="缺陷类型排列图（来自人工复核结论）", height=280, sort="-y", scale=2)
                    with gr.Column(scale=1):
                        refresh = gr.Button("刷新")
                        download = gr.DownloadButton("导出复核记录 CSV", visible=False)
                        gr.HTML("<p class='note'>记录保存在本机 data/demo/reviews/，同时保存校正后的原图，可直接作为后续训练样本。</p>")
                records = gr.Dataframe(label="复核记录（最新在前）", interactive=False, wrap=False)

            # ---------------- about
            with gr.Tab("⑤ 模型说明与边界", id="about"):
                gr.Markdown(about_markdown())

        outputs = [verdict, annotated, slider, table, state, decision, defect_types, saved]
        go.click(run_inspection, [photo, sensitivity], outputs)
        gallery.select(load_sample, None, photo).then(run_inspection, [photo, sensitivity], outputs)
        sign.click(save_review, [state, reviewer, decision, defect_types, note, work_order, board_sn], saved)
        manual_load_btn.click(manual_load, photo, [manual_view, manual_src, manual_pts, manual_hint])
        manual_view.select(manual_click, [manual_src, manual_pts], [manual_view, manual_pts, manual_hint])
        manual_clear_btn.click(manual_clear, manual_src, [manual_view, manual_pts, manual_hint])
        manual_go.click(manual_inspect, [manual_src, manual_pts, rotate, sensitivity], outputs + [tabs])
        lab_go.click(run_lab, [lab_board, lab_seed, lab_count, sensitivity], [lab_truth, lab_found, lab_heat, lab_summary, lab_table])
        dash_outputs = [kpi_html, pareto, records, download]
        records_tab.select(dashboard, None, dash_outputs)
        refresh.click(dashboard, None, dash_outputs)
        demo.load(dashboard, None, dash_outputs)
    return demo


def main() -> None:
    inspector()  # load models before the first click
    build().launch(server_name=os.environ.get("PCB_DEMO_HOST", "127.0.0.1"), server_port=int(os.environ.get("PCB_DEMO_PORT", "7860")),
                   share=False, theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"), css=CSS,
                   allowed_paths=[str(C.OUT)], footer_links=[])


if __name__ == "__main__":
    main()
