import io, sys
from pathlib import Path
import cv2, numpy as np
from fastapi.testclient import TestClient
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import app as appmod

class _Boxes:
    xyxy = type("T", (), {"tolist": lambda s: [[1.0, 2.0, 30.0, 40.0]]})()
    conf = type("T", (), {"tolist": lambda s: [0.9]})()
    cls = type("T", (), {"tolist": lambda s: [0]})()
class _R: boxes = _Boxes(); names = {0: "defect"}
class _M:
    def predict(self, *a, **k): return [_R()]

def test_health():
    assert TestClient(appmod.app).get("/health").json() == {"status": "ok"}

def test_predict_returns_boxes(monkeypatch):
    monkeypatch.setattr(appmod, "model", lambda: _M())
    ok, buf = cv2.imencode(".png", np.zeros((64, 64, 3), np.uint8))
    r = TestClient(appmod.app).post("/predict", files={"file": ("a.png", io.BytesIO(buf.tobytes()), "image/png")})
    assert r.status_code == 200 and r.json()["defects"][0]["class"] == "defect"

def test_predict_rejects_non_image():
    r = TestClient(appmod.app).post("/predict", files={"file": ("a.txt", io.BytesIO(b"nope"), "text/plain")})
    assert r.status_code == 400
