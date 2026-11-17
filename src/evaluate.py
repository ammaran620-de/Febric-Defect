"""Compare YOLO vs the OpenCV baseline on the val split with the same IoU-0.5 matching.

Two scores are written, and they answer different questions:
  yolo, baseline      class-agnostic localization: a box counts if it overlaps a labelled defect, whatever
                      class it names. The OpenCV baseline has no classes, so this is the fair comparison.
  yolo_class_aware    YOLO only: the box must also name the labelled defect's class.
Writes results/metrics.json and results/errors/ (missed + false-positive images) for error analysis.
Usage: python src/evaluate.py runs/detect/fabric/weights/best.pt data/xxx/data.yaml"""
import argparse, json, sys
from pathlib import Path
import cv2, yaml
sys.path.insert(0, str(Path(__file__).parent))
from baseline import detect as baseline_detect
from ultralytics import YOLO

from metrics import iou, match_boxes as match, match_detections, prf, check_class_names, IOU_THRESHOLD  # noqa: E402,F401

def gt_boxes(lbl, W, H):
    return [b for b, _ in gt_labelled(lbl, W, H)]

def gt_labelled(lbl, W, H):
    """[(box_xyxy, class_id)] from a YOLO label file."""
    out = []
    if lbl.exists():
        for ln in lbl.read_text().split("\n"):
            if ln.strip():
                c, x, y, w, h = ln.split()[:5]; x, y, w, h = map(float, (x, y, w, h))
                out.append((((x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H), int(c)))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("weights"); ap.add_argument("data"); ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.data)); root = Path(cfg["path"])
    vdir = root/cfg["val"]; ldir = root/str(cfg["val"]).replace("images", "labels")
    model = YOLO(a.weights); check_class_names(model.names, cfg["names"])
    names = {int(k): v for k, v in cfg["names"].items()}
    res = {"yolo": [0, 0, 0], "baseline": [0, 0, 0]}
    aware = {"tp": 0, "fp": 0, "fn": 0, "wrong_class": 0, "duplicate": 0}; per = {n: [0, 0, 0] for n in names.values()}
    agnostic_dup = 0; images = 0
    err = Path("results/errors"); err.mkdir(parents=True, exist_ok=True)
    for p in sorted(vdir.iterdir()):
        img = cv2.imread(str(p))
        if img is None: continue
        images += 1
        H, W = img.shape[:2]; lab = gt_labelled(ldir/f"{p.stem}.txt", W, H); gt = [b for b, _ in lab]
        r = model.predict(img, conf=a.conf, verbose=False)[0]
        yolo = list(zip([tuple(b) for b in r.boxes.xyxy.tolist()], r.boxes.conf.tolist(), [int(c) for c in r.boxes.cls.tolist()]))
        preds = {"yolo": [b for b, _, _ in yolo], "baseline": baseline_detect(img)}
        for k, pr in preds.items():
            tp, fp, fn = match(pr, gt)
            for i, v in enumerate((tp, fp, fn)): res[k][i] += v
            if k == "yolo" and (fp or fn):
                vis = img.copy()
                for g in gt: cv2.rectangle(vis, tuple(map(int, g[:2])), tuple(map(int, g[2:])), (0, 255, 0), 2)
                for q in pr: cv2.rectangle(vis, tuple(map(int, q[:2])), tuple(map(int, q[2:])), (0, 0, 255), 2)
                cv2.imwrite(str(err/f"{'FN' if fn else 'FP'}_{p.name}"), vis)
        agnostic_dup += match_detections(yolo, lab, class_aware=False)["duplicate"]
        m = match_detections(yolo, lab, class_aware=True)
        for k in aware: aware[k] += m[k]
        for c, v in m["per_class"].items():
            for j, key in enumerate(("tp", "fp", "fn")): per[names[c]][j] += v[key]
    out = {k: prf(*v) for k, v in res.items()}
    out["yolo"]["duplicate_fp"] = agnostic_dup
    out["yolo_class_aware"] = {**prf(aware["tp"], aware["fp"], aware["fn"]), "wrong_class_fp": aware["wrong_class"], "duplicate_fp": aware["duplicate"]}
    out["yolo_class_aware_per_class"] = [{"class": n, **prf(*v)} for n, v in per.items()]
    out["evaluation"] = {"images": images, "conf": a.conf, "iou": IOU_THRESHOLD, "negative_images": 0,
                         "yolo_baseline": "class-agnostic localization: a box counts if it overlaps a labelled defect at IoU >= 0.5, class ignored",
                         "yolo_class_aware": "box must also name the labelled defect's class; a wrong-class box is a false positive and the defect stays a miss",
                         "caveat": "the validation split has no defect-free images, so a factory false-alarm rate is not measured"}
    out["ultralytics_val"] = {k: round(float(v), 4) for k, v in zip(("precision", "recall", "mAP50", "mAP50-95"), model.val(data=a.data, verbose=False).box.mean_results()[:4])}
    out["note"] = "green=ground truth, red=YOLO prediction in results/errors"
    Path("results/metrics.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))

if __name__ == "__main__": main()
