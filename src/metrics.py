"""Detection matching shared by evaluate.py, make_summary.py and the tests.

Two scoring modes use the same greedy matcher:

* class-agnostic (class_aware=False): a prediction is a true positive if its box overlaps a not yet
  matched labelled defect at IoU >= iou_thr. The predicted class is ignored. This is the only mode the
  OpenCV baseline can take part in, because it does not predict a class.
* class-aware (class_aware=True): the overlapping labelled defect must also have the same class. A box on
  the right spot with the wrong class is a false positive and leaves the labelled defect as a miss.

Predictions are matched in order of falling confidence. Each labelled defect can be matched once, so a
second box on the same defect is a duplicate false positive. Both thresholds are inclusive: a box with
IoU exactly equal to iou_thr matches, and a box with confidence exactly equal to the cut-off is kept.
(Ultralytics' own predict(conf=...) uses a strict ">" internally, so evaluate.py scores whatever the model
returns and applies filter_conf only for the confidence sweep.)
"""
from collections import defaultdict

IOU_THRESHOLD = 0.5


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u else 0.0


def filter_conf(preds, conf):
    """Keep detections whose confidence is at least `conf`. preds: [(box, confidence, class)]."""
    return [p for p in preds if p[1] >= conf]


def match_detections(preds, gts, iou_thr=IOU_THRESHOLD, class_aware=False):
    """Match predictions to labelled defects.

    preds: [(box_xyxy, confidence, class_id)]   gts: [(box_xyxy, class_id)]
    Returns a dict with tp, fp, fn, wrong_class, duplicate and per_class {class: {tp, fp, fn}}.
    fp counts every unmatched prediction; wrong_class and duplicate say why some of them are false
    positives (the rest simply sit where no labelled defect is). In class-agnostic mode wrong_class is 0.
    Per-class counts are only meaningful in class-aware mode: fp is keyed by the predicted class and fn
    by the labelled class.
    """
    used = set(); tp = wrong = dup = 0
    per = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for box, _conf, cls in sorted(preds, key=lambda p: -p[1]):
        best, bi = 0.0, None
        for i, (g, gcls) in enumerate(gts):
            if i in used or (class_aware and gcls != cls):
                continue
            o = iou(box, g)
            if o >= iou_thr and o > best:
                best, bi = o, i
        if bi is not None:
            used.add(bi); tp += 1; per[cls]["tp"] += 1
            continue
        per[cls]["fp"] += 1
        same = any(iou(box, g) >= iou_thr and (not class_aware or gcls == cls) for g, gcls in gts)
        other = class_aware and any(iou(box, g) >= iou_thr and gcls != cls for g, gcls in gts)
        if same: dup += 1
        elif other: wrong += 1
    for i, (_g, gcls) in enumerate(gts):
        if i not in used: per[gcls]["fn"] += 1
    return {"tp": tp, "fp": len(preds) - tp, "fn": len(gts) - tp, "wrong_class": wrong, "duplicate": dup,
            "per_class": {k: dict(v) for k, v in per.items()}}


def match_boxes(pred_boxes, gt_boxes, t=IOU_THRESHOLD):
    """Class-agnostic (tp, fp, fn) for plain box lists (used by the OpenCV baseline and the sweep)."""
    r = match_detections([(b, 1.0, 0) for b in pred_boxes], [(g, 0) for g in gt_boxes], t)
    return r["tp"], r["fp"], r["fn"]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def check_class_names(model_names, data_names):
    """Refuse to score a model against a dataset whose class ids mean something else."""
    m = {int(k): v for k, v in dict(model_names).items()}; d = {int(k): v for k, v in dict(data_names).items()}
    if m != d:
        raise ValueError(f"model classes {m} do not match dataset classes {d}; refusing to score")
