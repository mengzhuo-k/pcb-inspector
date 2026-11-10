# 魔搭 CPU 实例训练（8 核 / 32 GB）

本目录把 SMD DefDB 七类缺陷分类模型的训练压缩成一条命令。训练只依赖
`training/` 下的三个脚本和数据集，不依赖本机 Windows 环境。

## 0. 上传代码与数据

本目录的 `train_all.sh` 可直接使用包内自带的数据集压缩包（825 MB，已校验 MD5），
实例不需要访问 Zenodo。上传包已把 `data/datasets/smd_defdb/PCB_augmented.rar`
包含在内，因此完整包约 840 MB。

```powershell
cd C:\Users\32537\Desktop\开发\pcb-inspector
.venv\Scripts\python.exe training\build_modelscope_bundle.py
scp -P <端口> ..\pcb-inspector-modelscope.zip <用户名>@101.200.150.64:~/
```

魔搭 DSW/Notebook 也可以在终端里 `git clone` 或拖拽上传。若只想传输代码，
可在实例上删除 `data/datasets/smd_defdb/PCB_augmented.rar`，脚本会自动从 Zenodo 下载。

## 1. 一键训练

```bash
cd ~/pcb-inspector
bash training/modelscope/train_all.sh
```

脚本会：

1. 安装 CPU 版 torch（若未安装）和 `training/modelscope/requirements.txt`；
2. 多线程下载 `PCB_augmented.rar`（825 MB，MD5 校验）并用 `bsdtar` 解压；
3. 按“10 张一组的原图组”划分 810/140/240 训练、早停验证、独立测试集；
4. 以 8 线程、8 个 dataloader worker、batch 64、RAM 缓存训练 40 轮；
5. 在从未参与训练/早停的测试集上计算 top1/top5、混淆矩阵和判定阈值；
6. 产出 `models/smd_defect_cls_v1.pt`、`.json` 和 `smd_defect_cls_v1.tar.gz`。

预计 8 核 CPU 约 40–70 分钟（取决于实例的单核性能）。日志在 `logs/train.log`。

## 2. 取回权重

```powershell
scp -P <端口> <用户名>@101.200.150.64:~/pcb-inspector/smd_defect_cls_v1.tar.gz .
```

解压到本机 `pcb-inspector/models/`，重启 `start.cmd` 即可加载。

## 3. 调参

```bash
# 先测 1 轮，比较线程/worker 组合
PCB_THREADS=8 WORKERS=4 EPOCHS=1 bash training/modelscope/train_all.sh
# 正式训练
PCB_THREADS=8 WORKERS=4 EPOCHS=40 bash training/modelscope/train_all.sh
```

`CHANNELS_LAST=1` 可试 oneDNN 的 channels-last 路径；`FORCE=1` 强制重训。

## 注意

- 若实例无法访问 `download.pytorch.org`，先 `pip install torch --index-url` 换成魔搭/阿里云镜像，
  或预装任意 CPU 版 torch；`train_all.sh` 检测到已安装 torch 就跳过该步。
- 若 Zenodo 下载过慢，可在本机下载 `PCB_augmented.rar` 后 `scp` 到
  `data/datasets/smd_defdb/PCB_augmented.rar`，脚本会直接校验并解压。
- 8 核实例的最终速度可能与本机 16 线程相当；脚本会打印每轮耗时，便于对比。
