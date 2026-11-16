"""Build results/summary.json (per-class metrics + class counts) for the dashboard.
Usage: python src/make_summary.py WEIGHTS DATA_YAML"""
import json, sys
from pathlib import Path
import yaml
from ultralytics import YOLO

w, data = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(data)); root = Path(cfg["path"]); names = cfg["names"]
m = YOLO(w).val(data=data, verbose=False).box
per = [{"class": names[i], "precision": round(float(m.class_result(k)[0]), 3), "recall": round(float(m.class_result(k)[1]), 3),
        "map50": round(float(m.class_result(k)[2]), 3), "map5095": round(float(m.class_result(k)[3]), 3)} for k, i in enumerate(m.ap_class_index)]
counts = {s: {n: 0 for n in names.values()} for s in ("train", "val")}
images = {}
for s in counts:
    images[s] = len(list((root/"images"/s).iterdir()))
    for f in (root/"labels"/s).glob("*.txt"):
        for ln in f.read_text().split("\n"):
            if ln.strip(): counts[s][names[int(ln.split()[0])]] += 1
out = json.loads(Path("results/metrics.json").read_text()) if Path("results/metrics.json").exists() else {}
out.update({"per_class": per, "class_counts": counts, "images": images, "dataset": "TILDA-400 subset", "model": "YOLOv8n (COCO-pretrained, 40 epochs, 416 px)"})
Path("results/summary.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2)[:600])
