"""Train YOLOv8. Usage: python src/train.py data/xxx/data.yaml [--weights yolov8n.pt] [--epochs 50]"""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("--weights", default="yolov8n.pt")
ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--name", default="fabric"); ap.add_argument("--device", default=None)
a = ap.parse_args()
YOLO(a.weights).train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, name=a.name, seed=0, device=a.device, plots=True)
