#!/usr/bin/env bash
# One-shot training on the 8-core / 32 GB ModelScope CPU instance.
#
#   bash training/modelscope/train_all.sh
#
# Tunables (environment variables):
#   PCB_THREADS   torch/OpenMP threads, default: nproc
#   EPOCHS        default 40        BATCH    default 64
#   WORKERS       dataloader workers, default 4
#   FORCE=1       retrain even when a published classifier exists
#   CHANNELS_LAST=1  enable channels-last (benchmark before keeping it)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

THREADS="${PCB_THREADS:-$(nproc)}"
export PCB_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1

PY="${PYTHON:-python3}"
echo "== host =="
echo "cores: $(nproc)  threads: $THREADS  mem: $(free -g | awk '/^Mem:/{print $2}') GiB"
echo "python: $($PY -V)  $($PY -c 'import sys;print(sys.executable)')"

if [ ! -f data/datasets/smd_defdb/dataset.tar ] \
   && ! command -v 7z >/dev/null 2>&1 && ! command -v unrar >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
    echo "== installing p7zip for RAR extraction =="
    apt-get update -qq && apt-get install -y -qq p7zip-full || echo "p7zip install skipped"
  fi
fi

if ! $PY -c "import torch" >/dev/null 2>&1; then
  echo "== installing CPU torch =="
  $PY -m pip install --upgrade pip
  $PY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
echo "== installing requirements =="
$PY -m pip install -r training/modelscope/requirements.txt

if [ "${FORCE:-0}" != "1" ] && [ -f models/smd_defect_cls_v1.pt ]; then
  echo "published classifier already exists; set FORCE=1 to retrain"
  exit 0
fi

echo "== dataset =="
if [ -d data/datasets/smd_defdb/cls/train ]; then
  echo "prepared train/val/test split already exists"
else
  $PY training/fetch_smd_defdb.py
  $PY training/prepare_smd_defdb.py
fi

ARGS=(--epochs "${EPOCHS:-40}" --imgsz 224 --batch "${BATCH:-64}"
      --workers "${WORKERS:-4}" --threads "$THREADS" --cache ram)
if [ "${CHANNELS_LAST:-0}" = "1" ]; then ARGS+=(--channels-last); fi

echo "== training =="
$PY -u training/train_smd_classifier.py "${ARGS[@]}" 2>&1 | tee logs/train.log

echo "== publish =="
ls -lh models/smd_defect_cls_v1.pt models/smd_defect_cls_v1.json
tar -czf smd_defect_cls_v1.tar.gz models/smd_defect_cls_v1.pt models/smd_defect_cls_v1.json
echo "done: $ROOT/smd_defect_cls_v1.tar.gz"
