"""Populate the inspection log by running the model on a folder of images (real predictions only).
Usage: python src/seed_history.py IMAGE_DIR [--source "validation batch"] [--conf 0.25]"""
import argparse, sys
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).parent))
import app

ap = argparse.ArgumentParser(); ap.add_argument("folder"); ap.add_argument("--source", default="validation batch"); ap.add_argument("--conf", type=float, default=0.25)
a = ap.parse_args(); n = 0
for p in sorted(Path(a.folder).iterdir()):
    img = cv2.imread(str(p))
    if img is not None: app.log_inspection(img, p.name, a.source, a.conf); n += 1
print(f"logged {n} inspections")
