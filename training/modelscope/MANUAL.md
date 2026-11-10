# 手动放到魔搭并训练（不需要 SSH）

三种方式，按省事程度排序。**推荐方案 A**：只上传 18 MB 代码，数据集由实例自己下载。

---

## 方案 A：只传代码，实例自动下载数据集（推荐）

### 1. 上传代码包

在魔搭控制台启动你的 8 核 32G 实例，打开 **JupyterLab**：

1. 左侧文件浏览器进入 `/mnt/workspace`（或你的用户目录）。
2. 点上方 **Upload Files**（↑ 图标），选择本机
   `C:\Users\32537\Desktop\开发\pcb-inspector-code.zip`（18.2 MB）。
3. 上传完成后，在 JupyterLab 里 **File → New → Terminal** 打开终端。

### 2. 解压并一键训练

```bash
cd ~
unzip -q pcb-inspector-code.zip -d pcb-inspector
cd pcb-inspector
bash training/modelscope/train_all.sh
```

脚本会自动：

1. 安装 CPU 版 torch（若镜像里没有）和训练依赖；
2. 从 Zenodo 多线程下载 825 MB 数据集并校验 MD5；
3. 用 `tar` 解压（Linux 自带，不需要 7z/unrar）；
4. 按原图组划分 810/140/240；
5. 8 线程、4 个 dataloader worker、batch 64、RAM 缓存训练 40 轮；
6. 在独立测试集上算 top1/top5/混淆矩阵/阈值；
7. 生成 `models/smd_defect_cls_v1.pt`、`.json` 和 `smd_defect_cls_v1.tar.gz`。

### 3. 看进度、取权重

```bash
tail -f logs/train.log          # 实时日志
cat training/runs/smd_defdb_cls/results.csv
```

训练结束后在 JupyterLab 文件浏览器里下载：

```text
pcb-inspector/models/smd_defect_cls_v1.pt
pcb-inspector/models/smd_defect_cls_v1.json
```

或者下载打包好的 `smd_defect_cls_v1.tar.gz`。

---

## 方案 B：连数据集一起上传（离线可用）

适合实例访问 Zenodo 很慢、或你不想在实例上下载的情况。

1. 上传本机的 `pcb-inspector-modelscope.zip`（813 MB，已包含 `dataset.tar`）。
2. 终端执行同样的命令：

```bash
cd ~
unzip -q pcb-inspector-modelscope.zip -d pcb-inspector
cd pcb-inspector
bash training/modelscope/train_all.sh
```

`train_all.sh` 发现 `data/datasets/smd_defdb/dataset.tar` 后会直接解压，不再联网。

> JupyterLab 上传 800 MB 可能较慢或超时。若失败，用方案 C 把数据放魔搭数据集库，
> 或者用方案 A。

---

## 方案 C：先把数据集传到魔搭“数据集库”，实例再拉取

适合你希望数据集在魔搭上长期保存、多个实例复用。

### 1. 创建数据集仓库

魔搭网页 → **数据集** → **创建数据集**，例如：

```text
<你的用户名>/smd-defdb
```

### 2. 本机上传 `dataset.tar`（833 MB）

在本机 PowerShell：

```powershell
cd C:\Users\32537\Desktop\开发\pcb-inspector
.\.venv\Scripts\python.exe -m pip install modelscope
set MODELSCOPE_API_KEY=你的访问令牌
.\.venv\Scripts\python.exe -c "from modelscope.hub.api import HubApi; import os; api=HubApi(); api.login(access_token=os.environ['MODELSCOPE_API_KEY']); api.upload_file(path_or_fileobj=r'data/datasets/smd_defdb/dataset.tar', path_in_repo='dataset.tar', repo_id='<你的用户名>/smd-defdb', repo_type='dataset', commit_message='add SMD DefDB')"
```

令牌在魔搭网页右上角头像 → **访问令牌**（`https://modelscope.cn/my/myaccesstoken`）。

### 3. 实例上下载并训练

```bash
cd ~/pcb-inspector
python -m pip install modelscope
python - <<'PY'
import os, shutil
from modelscope.hub.file_download import dataset_file_download
path = dataset_file_download(dataset_id="<你的用户名>/smd-defdb", file_path="dataset.tar")
os.makedirs("data/datasets/smd_defdb", exist_ok=True)
shutil.copy(path, "data/datasets/smd_defdb/dataset.tar")
print("saved", "data/datasets/smd_defdb/dataset.tar")
PY
bash training/modelscope/train_all.sh
```

更简单的做法：在 JupyterLab 文件浏览器里把下载到的 `dataset.tar` 拖到
`pcb-inspector/data/datasets/smd_defdb/`，再运行 `train_all.sh`。

---

## 常用参数

```bash
# 先跑 1 轮测试环境/速度
EPOCHS=1 bash training/modelscope/train_all.sh

# 指定线程/worker（默认 nproc / 4）
PCB_THREADS=8 WORKERS=4 EPOCHS=40 bash training/modelscope/train_all.sh

# 强制重训（已有 smd_defect_cls_v1.pt 时）
FORCE=1 bash training/modelscope/train_all.sh

# 试 channels-last
CHANNELS_LAST=1 bash training/modelscope/train_all.sh
```

日志和中间权重在：

```text
training/runs/smd_defdb_cls/
├── results.csv
├── weights/best.pt
├── weights/last.pt
└── metrics.json
```

---

## 常见问题

| 现象 | 处理 |
|---|---|
| `unzip: command not found` | `python -m zipfile -e pcb-inspector-code.zip pcb-inspector` |
| `tar` 解压 RAR 报错 | 用方案 B 的 `dataset.tar`，不要用 RAR |
| torch 下载慢 | 镜像若已自带 torch 会自动跳过；否则换 `pip install torch --index-url https://download.pytorch.org/whl/cpu`，或配 `PIP_INDEX_URL` 国内源 |
| Zenodo 下载慢/超时 | 用方案 B 或 C |
| 没有 root，装不了 p7zip | 用方案 B/C 的 tar，不需要 7z |
| 训练中断 | 重新执行 `train_all.sh`；已完成的 `cls/` 划分会复用，数据集不会重复下载 |
| 想停 | 终端 `Ctrl+C` |

---

## 重要说明

- 这版模型在独立测试集上的**七类 top1 约 67%**，二分类“有无缺陷”精确率 100%、召回 86.7%。
  100% 只代表 30 张合格测试图没有误报，不代表新板型精度。
- 焊盘损伤、桥接的类型召回仍偏低；训练脚本会把这些指标写进 `smd_defect_cls_v1.json`。
- 现场使用前，必须用**不同板型、不同光照、不同批次**的图片重新验证；仅靠这套 119 张原图的增强数据不够。
