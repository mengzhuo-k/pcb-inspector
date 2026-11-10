import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

if __name__ == "__main__":
    folder, weights = map(Path, sys.argv[1:])
    os.environ["YOLO_OFFLINE"] = "true"
    torch.set_num_threads(4)
    YOLO(str(weights)).train(data=str(folder/"data.yaml"), project=str(folder), name="fit",
        device="cuda:0" if torch.cuda.is_available() else "cpu", imgsz=960, epochs=24,
        batch=2, nbs=2, workers=0, mosaic=0, close_mosaic=0, scale=.12, translate=.04,
        optimizer="AdamW", lr0=.0004, plots=False, verbose=False)
