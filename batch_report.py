"""Generate a printable batch report from an immutable inspection snapshot."""
import io
import json
import os
import threading
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle

build_lock = threading.Lock()


def build_pdf(snapshot, exported_at):
    with build_lock:
        return _build_pdf(snapshot, exported_at)


def _build_pdf(snapshot, exported_at):
    font = "PCBReport"
    if font not in pdfmetrics.getRegisteredFontNames():
        path = Path(os.environ.get("PCB_REPORT_FONT", "C:/Windows/Fonts/simsun.ttc"))
        if path.is_file():
            pdfmetrics.registerFont(TTFont(font, str(path)))
        else:
            font = "STSong-Light"
            pdfmetrics.registerFont(UnicodeCIDFont(font))
    body = ParagraphStyle("Body", fontName=font, fontSize=10, leading=15,
                          spaceAfter=8, wordWrap="CJK", textColor=colors.black)
    small = ParagraphStyle("Cell", parent=body, fontSize=9, leading=13, spaceAfter=0)
    title = ParagraphStyle("ReportTitle", parent=body, fontSize=21, leading=29, spaceAfter=12)
    heading = ParagraphStyle("Section", parent=body, fontSize=13, leading=20,
                             spaceBefore=14, spaceAfter=8, keepWithNext=True)
    width = A4[0]-84

    def paragraph(value, style=small):
        return Paragraph(escape(str(value if value is not None else "—")).replace("\n", "<br/>"), style)

    def table(headers, rows, widths):
        data = [[paragraph(value) for value in headers]]
        data += [[paragraph(value) for value in row] for row in rows]
        result = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E8EDF2")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LINEBELOW",(0,0),(-1,0),.7,colors.HexColor("#8A98A6")),
            ("LINEBELOW",(0,1),(-1,-1),.35,colors.HexColor("#D7DDE3")),
            ("LEFTPADDING",(0,0),(-1,-1),6),
            ("RIGHTPADDING",(0,0),(-1,-1),6),
            ("TOPPADDING",(0,0),(-1,-1),7),
            ("BOTTOMPADDING",(0,0),(-1,-1),7),
        ]))
        return result

    def rate(passed, count):
        return f"{100*passed/count:.1f}%" if count else "—"

    batch = snapshot["batch"]
    items = snapshot["items"]
    for item in items:
        raw = item.get("classification")
        item["classification"] = json.loads(raw) if isinstance(raw, str) and raw else None
    done = [item for item in items if item["state"] == "done"]
    rejected = [item for item in done if item["decision"] == "不合格"]
    passed = sum(item["decision"] == "合格" for item in done)
    auto_passed = sum(item["auto_decision"] == "合格" for item in done)
    failed = sum(item["state"] == "failed" for item in items)
    deleted = sum(item["state"] == "deleted" for item in items)
    reviewed = sum(bool(item["signed"]) for item in done)
    decided = sum(item["decision"] in ("合格", "不合格") for item in done)
    auto_decided = sum(item["auto_decision"] in ("合格", "不合格") for item in done)
    serial = lambda item: f"#{item['ordinal']+1:03}"

    def identify(item):
        classification = item.get("classification")
        if not classification:
            return "未记录"
        if classification.get("is_defect"):
            return f"{classification['label']} {classification['score']:.0%}"
        return "未见明显缺陷"

    defect_counts = {}
    for item in done:
        classification = item.get("classification")
        if classification and classification.get("is_defect"):
            label = classification["label"]
            defect_counts[label] = defect_counts.get(label, 0) + 1
    defect_summary = "、".join(f"{label} {count} 张" for label, count in defect_counts.items()) or "未识别到整板缺陷类型"

    story = [paragraph("PCB 焊接缺陷批次检验报告", title),
             paragraph(batch["name"], body),
             paragraph(f"本批次导入 {len(items)} 张图片，成功检测 {len(done)} 张。当前合格 {passed} 张，"
                       f"不合格 {len(rejected)} 张，合格率 {rate(passed, decided)}。"
                       f"整板缺陷类型：{defect_summary}。"
                       f"本报告记录导出时的机器初判与员工复查结论，供批次归档和不合格追溯。", body)]
    story.append(table(["项目","内容","项目","内容"],[
        ["批次名称",batch["name"],"工位",batch["station"]],
        ["批次创建时间",batch["created"],"报告生成时间",exported_at],
        ["机器初判合格率",rate(auto_passed,auto_decided),"当前合格率",rate(passed,decided)],
        ["已人工复查",f"{reviewed} 张","检测失败及已删除",f"{failed} 张失败 / {deleted} 张已删除"],
    ],[90,width/2-90,90,width/2-90]))
    story.append(paragraph("合格率以成功检测且有当前结论的图片为分母，检测失败和已删除图片单独计数。"
                           "常规图片采用自动检测结论，抽样图片与不合格图片记录人工复查结果。序号对应批次原始导入顺序。"
                           "漏焊候选来自目标检测模型；整板缺陷类型来自图像级分类模型，只给出类型、不定位具体元件。", body))
    story.append(paragraph("异常序号清单", heading))
    if rejected:
        rows = []
        for item in rejected:
            review = f"{item['reviewer']}\n{item['signed']}" if item["signed"] else "未人工复查"
            rows.append([serial(item),item["filename"],item["count"],
                         identify(item),item["auto_decision"],item["decision"],review])
        story.append(table(["序号","图片文件名","候选数","整板识别","机器初判","当前结论","人工复查"],
                           rows,[34,145,38,78,50,50,width-395]))
    else:
        story.append(paragraph("本批次当前没有异常图片。",body))
    story.append(paragraph("逐图检验记录", heading))
    states = {"failed":"检测失败","deleted":"已删除"}
    rows = []
    for item in items:
        result = item["decision"] if item["state"] == "done" else states.get(item["state"],item["state"])
        review = item["reviewer"] if item["signed"] else "未复查"
        if item["state"] != "done":
            review = item["error"] or "—"
        rows.append([serial(item),item["filename"],result,identify(item),item["model_version"] or "未记录",review])
    story.append(table(["序号","图片文件名","当前结果","整板识别","检测模型","复查人员或异常"],
                       rows,[34,130,55,78,90,width-387]))
    notes = [item for item in items if item["comment"]]
    if notes:
        story.append(paragraph("员工复查意见",heading))
        for item in notes:
            story.append(paragraph(serial(item)+" "+item["filename"],heading))
            story.append(paragraph(item["comment"],body))
    story.append(Spacer(1,12))
    story.append(paragraph("批次编号 "+batch["id"],small))
    output = io.BytesIO()
    document = SimpleDocTemplate(output,pagesize=A4,leftMargin=42,rightMargin=42,
                                 topMargin=42,bottomMargin=42,
                                 title="PCB 焊接缺陷批次检验报告",author="焊视质量检测系统")

    def footer(canvas, doc):
        canvas.setFont(font,9)
        canvas.setFillColor(colors.HexColor("#66717D"))
        canvas.drawCentredString(A4[0]/2,24,f"第 {doc.page} 页")

    document.build(story,onFirstPage=footer,onLaterPages=footer)
    return output.getvalue()
