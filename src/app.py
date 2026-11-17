"""Fabric inspection system: model API + inspection log + dashboard.
Run: uvicorn src.app:app --port 8000   (MODEL_PATH, DB_PATH optional)"""
import csv, io, json, os, sqlite3, time
from datetime import datetime, timezone
from pathlib import Path
import cv2, numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DB = Path(os.environ.get("DB_PATH", ROOT/"results"/"inspections.db"))
THUMBS = DB.parent/"inspections"
app = FastAPI(title="Fabric Inspection System")
_model = None

def model():
    global _model
    if _model is None: _model = YOLO(os.environ.get("MODEL_PATH", str(ROOT/"results"/"best.pt")))
    return _model

DEMO = ROOT/"results"/"demo"

def _seed_demo():
    """First start only: copy the bundled demo log (real predictions on the validation split) if no log exists yet."""
    if "DB_PATH" in os.environ or DB.exists() or not (DEMO/"inspections.db").exists(): return
    import shutil
    DB.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(DEMO/"inspections.db", DB)
    if (DEMO/"inspections").is_dir(): shutil.copytree(DEMO/"inspections", THUMBS, dirs_exist_ok=True)

def db():
    _seed_demo()
    DB.parent.mkdir(parents=True, exist_ok=True); THUMBS.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS inspections(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, source TEXT, filename TEXT,
        verdict TEXT, n_defects INTEGER, max_conf REAL, classes TEXT, conf_used REAL, ms REAL, boxes TEXT)""")
    return c

def decode(data):
    if len(data) > 15_000_000: raise HTTPException(413, "Image too large (max 15 MB)")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400, "Not a valid image")
    return img

def run(img, conf):
    t0 = time.perf_counter(); r = model().predict(img, conf=conf, verbose=False)[0]
    ms = (time.perf_counter()-t0)*1000
    return [{"box": [round(v, 1) for v in b], "confidence": round(c, 3), "class": r.names[int(k)]}
            for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())], ms

def log_inspection(img, filename, source, conf):
    defects, ms = run(img, conf)
    verdict = "FAIL" if defects else "PASS"
    with db() as c:
        cur = c.execute("INSERT INTO inspections(ts,source,filename,verdict,n_defects,max_conf,classes,conf_used,ms,boxes) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), source, filename, verdict, len(defects),
             max([d["confidence"] for d in defects], default=0), ",".join(sorted({d["class"] for d in defects})), conf, round(ms, 1), json.dumps(defects)))
        iid = cur.lastrowid
    vis = img.copy(); lw = max(2, img.shape[1]//250)
    for d in defects:
        x0, y0, x1, y1 = map(int, d["box"]); cv2.rectangle(vis, (x0, y0), (x1, y1), (31, 50, 196), lw)
    s = 480/vis.shape[1]; vis = cv2.resize(vis, (480, int(vis.shape[0]*s)))
    cv2.imwrite(str(THUMBS/f"{iid}.jpg"), vis)
    return {"id": iid, "verdict": verdict, "defects": defects, "ms": round(ms, 1), "width": img.shape[1], "height": img.shape[0]}

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.25):
    img = decode(await file.read()); defects, _ = run(img, conf)
    return {"width": img.shape[1], "height": img.shape[0], "defects": defects}

@app.post("/api/inspect")
async def inspect(file: UploadFile = File(...), conf: float = 0.25, source: str = "upload"):
    return log_inspection(decode(await file.read()), file.filename or "image", source, conf)

@app.get("/api/history")
def history(limit: int = 200, verdict: str = ""):
    q, a = "SELECT id,ts,source,filename,verdict,n_defects,max_conf,classes,conf_used,ms FROM inspections", []
    if verdict in ("PASS", "FAIL"): q += " WHERE verdict=?"; a.append(verdict)
    with db() as c: return [dict(r) for r in c.execute(q+" ORDER BY id DESC LIMIT ?", a+[limit])]

@app.get("/api/history.csv")
def history_csv():
    with db() as c: rows = [dict(r) for r in c.execute("SELECT id,ts,source,filename,verdict,n_defects,max_conf,classes,conf_used,ms FROM inspections ORDER BY id")]
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=["id", "ts", "source", "filename", "verdict", "n_defects", "max_conf", "classes", "conf_used", "ms"])
    w.writeheader(); w.writerows(rows)
    return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=inspections.csv"})

@app.get("/api/stats")
def stats():
    with db() as c:
        rows = [dict(r) for r in c.execute("SELECT verdict,n_defects,classes,ms,source,boxes FROM inspections ORDER BY id")]
    n = len(rows); fail = sum(r["verdict"] == "FAIL" for r in rows); cls = {}
    for r in rows:
        for d in json.loads(r["boxes"]): cls[d["class"]] = cls.get(d["class"], 0)+1
    src = {}
    for r in rows: src[r["source"]] = src.get(r["source"], 0)+1
    return {"inspections": n, "failed": fail, "passed": n-fail, "defects": sum(r["n_defects"] for r in rows),
            "avg_ms": round(sum(r["ms"] for r in rows)/n, 1) if n else 0, "by_class": cls, "by_source": src,
            "recent": [r["n_defects"] for r in rows[-40:]]}

@app.get("/api/thumb/{iid}")
def thumb(iid: int):
    p = THUMBS/f"{iid}.jpg"
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p)

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

for sub in ("errors", "training"): (ROOT/"results"/sub).mkdir(parents=True, exist_ok=True)
app.mount("/errors", StaticFiles(directory=ROOT/"results"/"errors"), name="errors")
app.mount("/training", StaticFiles(directory=ROOT/"results"/"training"), name="training")
