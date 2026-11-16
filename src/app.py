"""FastAPI service: /predict (inference), plus a results + live-detection dashboard at /.
Set MODEL_PATH to the trained weights (default results/best.pt)."""
import json, os
from pathlib import Path
import cv2, numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Fabric defect detector")
_model = None

def model():
    global _model
    if _model is None: _model = YOLO(os.environ.get("MODEL_PATH", str(ROOT/"results"/"best.pt")))
    return _model

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.25):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400, "Not a valid image")
    r = model().predict(img, conf=conf, verbose=False)[0]
    return {"width": img.shape[1], "height": img.shape[0],
            "defects": [{"box": [round(v, 1) for v in b], "confidence": round(c, 3), "class": r.names[int(k)]}
                        for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]}

@app.get("/api/summary")
def summary():
    p = ROOT/"results"/"summary.json"
    if not p.exists(): raise HTTPException(404, "Run src/make_summary.py first")
    return json.loads(p.read_text())

@app.get("/api/errors")
def errors():
    d = ROOT/"results"/"errors"
    return [{"file": f.name, "kind": f.name.split("_")[0]} for f in sorted(d.glob("*.jpg"))] if d.exists() else []

@app.get("/")
def index(): return FileResponse(Path(__file__).parent/"static"/"index.html")

(ROOT/"results"/"errors").mkdir(parents=True, exist_ok=True)
app.mount("/errors", StaticFiles(directory=ROOT/"results"/"errors"), name="errors")
