"""Compare YOLO vs the OpenCV baseline on the val split with the same IoU-0.5 matching.
Writes results/metrics.json and results/errors/ (missed + false-positive images) for error analysis.
Usage: python src/evaluate.py runs/detect/fabric/weights/best.pt data/xxx/data.yaml"""
import argparse, json, sys
from pathlib import Path
import cv2, yaml
sys.path.insert(0, str(Path(__file__).parent))
from baseline import detect as baseline_detect
from ultralytics import YOLO

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix*iy; u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/u if u else 0.0

def gt_boxes(lbl, W, H):
    out = []
    if lbl.exists():
        for ln in lbl.read_text().split("\n"):
            if ln.strip():
                _, x, y, w, h = map(float, ln.split()[:5])
                out.append(((x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H))
    return out

def match(pred, gt, t=0.5):
    used, tp = set(), 0
    for p in pred:
        best, bi = 0, None
        for i, g in enumerate(gt):
            if i not in used and iou(p, g) > best: best, bi = iou(p, g), i
        if bi is not None and best >= t: used.add(bi); tp += 1
    return tp, len(pred)-tp, len(gt)-tp

def prf(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.0; r = tp/(tp+fn) if tp+fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2*p*r/(p+r), 4) if p+r else 0.0, "tp": tp, "fp": fp, "fn": fn}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("weights"); ap.add_argument("data"); ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.data)); root = Path(cfg["path"])
    vdir = root/cfg["val"]; ldir = root/str(cfg["val"]).replace("images", "labels")
    model = YOLO(a.weights); res = {"yolo": [0, 0, 0], "baseline": [0, 0, 0]}
    err = Path("results/errors"); err.mkdir(parents=True, exist_ok=True)
    for p in sorted(vdir.iterdir()):
        img = cv2.imread(str(p))
        if img is None: continue
        H, W = img.shape[:2]; gt = gt_boxes(ldir/f"{p.stem}.txt", W, H)
        preds = {"yolo": [tuple(b) for b in model.predict(img, conf=a.conf, verbose=False)[0].boxes.xyxy.tolist()],
                 "baseline": baseline_detect(img)}
        for k, pr in preds.items():
            tp, fp, fn = match(pr, gt)
            for i, v in enumerate((tp, fp, fn)): res[k][i] += v
            if k == "yolo" and (fp or fn):
                vis = img.copy()
                for g in gt: cv2.rectangle(vis, tuple(map(int, g[:2])), tuple(map(int, g[2:])), (0, 255, 0), 2)
                for q in pr: cv2.rectangle(vis, tuple(map(int, q[:2])), tuple(map(int, q[2:])), (0, 0, 255), 2)
                cv2.imwrite(str(err/f"{'FN' if fn else 'FP'}_{p.name}"), vis)
    out = {k: prf(*v) for k, v in res.items()}
    out["ultralytics_val"] = {k: round(float(v), 4) for k, v in zip(("precision", "recall", "mAP50", "mAP50-95"), model.val(data=a.data, verbose=False).box.mean_results()[:4])}
    out["note"] = "green=ground truth, red=YOLO prediction in results/errors"
    Path("results/metrics.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))

if __name__ == "__main__": main()
