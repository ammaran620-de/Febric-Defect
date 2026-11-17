import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from metrics import iou, filter_conf, match_detections, match_boxes, prf, check_class_names

G = (0, 0, 10, 10)          # a labelled defect, area 100
HOLE, OIL = 0, 2


def pred(box, conf=0.9, cls=HOLE): return (box, conf, cls)
def gt(box, cls=HOLE): return (box, cls)


def test_exact_match_is_true_positive_in_both_modes():
    for aware in (False, True):
        r = match_detections([pred(G)], [gt(G)], class_aware=aware)
        assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 0)


def test_miss_is_false_negative_and_nothing_else():
    r = match_detections([], [gt(G)])
    assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 1)


def test_box_on_empty_background_is_false_positive():
    r = match_detections([pred((50, 50, 60, 60))], [gt(G)])
    assert (r["tp"], r["fp"], r["fn"], r["wrong_class"], r["duplicate"]) == (0, 1, 1, 0, 0)


def test_wrong_class_counts_as_hit_only_when_class_is_ignored():
    p, g = [pred(G, cls=OIL)], [gt(G, cls=HOLE)]
    agnostic = match_detections(p, g, class_aware=False)
    aware = match_detections(p, g, class_aware=True)
    assert (agnostic["tp"], agnostic["fp"], agnostic["fn"], agnostic["wrong_class"]) == (1, 0, 0, 0)
    assert (aware["tp"], aware["fp"], aware["fn"], aware["wrong_class"]) == (0, 1, 1, 1)


def test_wrong_class_does_not_steal_a_correct_match():
    # a wrong-class box and a right-class box on the same defect: the right one is the true positive
    p = [pred(G, conf=0.95, cls=OIL), pred(G, conf=0.60, cls=HOLE)]
    r = match_detections(p, [gt(G, cls=HOLE)], class_aware=True)
    assert (r["tp"], r["fp"], r["fn"], r["wrong_class"]) == (1, 1, 0, 1)


def test_duplicate_boxes_on_one_defect_give_one_tp_and_one_fp():
    p = [pred(G, conf=0.9), pred((0, 0, 10, 9), conf=0.8)]
    for aware in (False, True):
        r = match_detections(p, [gt(G)], class_aware=aware)
        assert (r["tp"], r["fp"], r["fn"], r["duplicate"]) == (1, 1, 0, 1)


def test_higher_confidence_box_is_matched_first_regardless_of_input_order():
    good, loose = pred(G, conf=0.9), pred((0, 0, 10, 6), conf=0.5)   # loose box IoU 0.6
    for order in ([good, loose], [loose, good]):
        r = match_detections(order, [gt(G)])
        assert (r["tp"], r["fp"], r["duplicate"]) == (1, 1, 1)


def test_negative_image_without_predictions_is_clean():
    r = match_detections([], [])
    assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 0)


def test_negative_image_with_a_prediction_is_a_false_alarm():
    r = match_detections([pred(G)], [])
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 0)


def test_iou_threshold_boundary_is_inclusive():
    assert iou(G, (0, 0, 10, 5)) == 0.5
    assert match_detections([pred((0, 0, 10, 5))], [gt(G)], iou_thr=0.5)["tp"] == 1
    just_below = (0, 0, 10, 4.99)
    assert iou(G, just_below) < 0.5
    r = match_detections([pred(just_below)], [gt(G)], iou_thr=0.5)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 1)


def test_confidence_cutoff_is_inclusive():
    preds = [pred(G, conf=0.25), pred(G, conf=0.2499), pred(G, conf=0.5)]
    kept = filter_conf(preds, 0.25)
    assert [p[1] for p in kept] == [0.25, 0.5]


def test_per_class_counts_in_class_aware_mode():
    gts = [gt(G, HOLE), gt((20, 20, 30, 30), OIL), gt((40, 40, 50, 50), OIL)]
    preds = [pred(G, cls=HOLE), pred((20, 20, 30, 30), cls=OIL), pred((40, 40, 50, 50), cls=HOLE)]
    r = match_detections(preds, gts, class_aware=True)
    assert r["per_class"][HOLE] == {"tp": 1, "fp": 1, "fn": 0}
    assert r["per_class"][OIL] == {"tp": 1, "fp": 0, "fn": 1}
    assert (r["tp"], r["fp"], r["fn"], r["wrong_class"]) == (2, 1, 1, 1)


def test_match_boxes_is_the_class_agnostic_helper():
    assert match_boxes([G, (50, 50, 60, 60)], [G, (100, 100, 110, 110)]) == (1, 1, 1)


def test_prf_handles_zero_denominators():
    assert prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}
    assert prf(86, 11, 15)["f1"] == pytest.approx(172 / 198, abs=1e-6)


def test_class_name_mismatch_is_rejected():
    check_class_names({0: "hole", 1: "oil"}, {0: "hole", 1: "oil"})
    with pytest.raises(ValueError):
        check_class_names({0: "hole", 1: "oil"}, {0: "hole", 1: "stain"})


def test_prf_is_not_rounded_before_display():
    # 86/101 is 0.851485: rounding to four places first (0.8515) and again to three gave 0.852.
    r = prf(86, 11, 15)
    assert r["recall"] == pytest.approx(86 / 101, abs=1e-6)
    assert f"{r['recall']:.3f}" == "0.851"
    assert f"{prf(84, 13, 17)['f1']:.3f}" == "0.848"


def test_stored_summary_matches_its_own_counts():
    import json
    s = json.loads((Path(__file__).parents[1] / "results" / "summary.json").read_text(encoding="utf-8"))
    rows = [s["yolo"], s["baseline"], s["yolo_class_aware"], *s["sweep"]]
    for row in rows:
        expect = prf(row["tp"], row["fp"], row["fn"])
        for key in ("precision", "recall", "f1"):
            assert row[key] == pytest.approx(expect[key], abs=1e-6), (row.get("conf"), key)
