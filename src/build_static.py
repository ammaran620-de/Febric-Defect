"""Build the GitHub Pages site into docs/ (static dashboard + in-browser ONNX inference).
Usage: python src/build_static.py        (needs results/best.onnx, results/demo/, onnxruntime-web via npm)
Export the model first:  yolo export model=results/best.pt format=onnx imgsz=416 opset=12 dynamic=True simplify=True"""
import json, shutil, sqlite3, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; DOCS = ROOT/"docs"; SITE = ROOT/"src"/"static_site"
ORT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/ortw/node_modules/onnxruntime-web/dist")

def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(src, dst)

# 1) dashboard html, rewritten for static hosting
h = (ROOT/"src"/"static"/"index.html").read_text(encoding="utf-8")
for a, b in [("showImg('/api/thumb/'+r.dataset.id)", 'showImg(THUMB(r.dataset.id))'), ('src="/api/thumb/${x.id}"', 'src="${THUMB(x.id)}"'), ('src="/errors/${e.file}"', 'src="errors/${e.file}"'),
             ('src="/training/', 'src="training/'), ('href="/api/history.csv"', 'href="#csv"'),]:
    assert a in h, a; h = h.replace(a, b)
banner = ('<p class="note">Static demo. The detector runs in your browser, so images stay on your device. '
          'The log starts with 80 real validation predictions and resets when you reload. '
          '<a class="link" href="https://github.com/ammaran620-de/Febric-Defect">Source code</a></p>')
h = h.replace("<main>", "<main>" + banner, 1)
h = h.replace("<script>\nconst $=", '<script src="vendor/ort.wasm.min.js"></script>\n<script src="static-api.js"></script>\n<script>\nconst $=', 1)
assert "static-api.js" in h
(DOCS/"index.html").write_text(h, encoding="utf-8")
copy(SITE/"static-api.js", DOCS/"static-api.js"); copy(ROOT/"src"/"static"/"fonts"/"archivo.woff2", DOCS/"fonts"/"archivo.woff2"); (DOCS/".nojekyll").write_text("")

# 2) runtime + model
for f in ["ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]: copy(ORT/f, DOCS/"vendor"/f)
copy(ROOT/"results"/"best.onnx", DOCS/"model"/"best.onnx")

# 3) data: demo history (with boxes), summary, error list, thumbnails, training plots
con = sqlite3.connect(ROOT/"results"/"demo"/"inspections.db"); con.row_factory = sqlite3.Row
rows = [dict(r) for r in con.execute("SELECT * FROM inspections ORDER BY id")]
for r in rows: r["boxes"] = json.loads(r["boxes"])
(DOCS/"data").mkdir(exist_ok=True)
(DOCS/"data"/"history.json").write_text(json.dumps(rows)); copy(ROOT/"results"/"summary.json", DOCS/"data"/"summary.json")
errs = sorted((ROOT/"results"/"errors").glob("*.jpg"))
(DOCS/"data"/"errors.json").write_text(json.dumps([{"file": f.name, "kind": f.name.split("_")[0]} for f in errs]))
for f in errs: copy(f, DOCS/"errors"/f.name)
for f in (ROOT/"results"/"demo"/"inspections").glob("*.jpg"): copy(f, DOCS/"demo"/"thumbs"/f.name)
for f in ["confusion_matrix.png", "results.png"]: copy(ROOT/"results"/"training"/f, DOCS/"training"/f)
print("built docs/:", len(rows), "demo rows,", len(errs), "error images")
