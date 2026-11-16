import io, sys, os, tempfile
from pathlib import Path
import cv2, numpy as np
os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp())/"t.db")
from fastapi.testclient import TestClient
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import app as appmod

class _T:
    def __init__(self, v): self.v = v
    def tolist(self): return self.v
class _Boxes: xyxy = _T([[1.0, 2.0, 30.0, 40.0]]); conf = _T([0.9]); cls = _T([0])
class _R: boxes = _Boxes(); names = {0: "defect"}
class _M:
    def predict(self, *a, **k): return [_R()]

def png(): return io.BytesIO(cv2.imencode(".png", np.zeros((64, 64, 3), np.uint8))[1].tobytes())

def test_health(): assert TestClient(appmod.app).get("/health").json() == {"status": "ok"}

def test_predict_returns_boxes(monkeypatch):
    monkeypatch.setattr(appmod, "model", lambda: _M())
    r = TestClient(appmod.app).post("/predict", files={"file": ("a.png", png(), "image/png")})
    assert r.status_code == 200 and r.json()["defects"][0]["class"] == "defect"

def test_predict_rejects_non_image():
    assert TestClient(appmod.app).post("/predict", files={"file": ("a.txt", io.BytesIO(b"no"), "text/plain")}).status_code == 400

def test_inspect_logs_and_stats(monkeypatch):
    monkeypatch.setattr(appmod, "model", lambda: _M()); c = TestClient(appmod.app)
    r = c.post("/api/inspect", files={"file": ("a.png", png(), "image/png")}); assert r.json()["verdict"] == "FAIL"
    s = c.get("/api/stats").json(); assert s["inspections"] >= 1 and s["by_class"]["defect"] >= 1
    assert c.get(f"/api/thumb/{r.json()['id']}").status_code == 200
    assert "a.png" in c.get("/api/history.csv").text
