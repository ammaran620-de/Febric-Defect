"""The GitHub Pages demo reads data/*.json next to index.html. A .gitignore rule for `data/` once kept
those files out of the repository, which leaves the hosted dashboard empty. These tests fail if that
happens again, and keep the published summary in step with results/summary.json."""
import json, re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_static_site_data_files_are_present_and_valid():
    api = (ROOT / "src" / "static_site" / "static-api.js").read_text(encoding="utf-8")
    names = re.findall(r"'data/([a-z]+\.json)'", api)
    assert sorted(names) == ["errors.json", "history.json", "summary.json"]
    for n in names:
        assert json.loads((ROOT / "docs" / "data" / n).read_text(encoding="utf-8")), n


def test_published_summary_matches_results_summary():
    assert json.loads((ROOT / "docs" / "data" / "summary.json").read_text(encoding="utf-8")) == \
        json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))


def test_both_scores_account_for_every_labelled_defect():
    s = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
    labelled = sum(s["class_counts"]["val"].values())
    assert labelled == 101
    assert s["yolo"]["tp"] + s["yolo"]["fn"] == labelled
    assert s["yolo_class_aware"]["tp"] + s["yolo_class_aware"]["fn"] == labelled
    assert s["yolo_class_aware"]["tp"] <= s["yolo"]["tp"]            # requiring the class cannot add hits
    assert s["evaluation"]["negative_images"] == 0                    # no defect-free images in this split


def test_font_ships_with_its_licence():
    assert (ROOT / "docs" / "fonts" / "archivo.woff2").exists()
    assert "SIL OPEN FONT LICENSE" in (ROOT / "docs" / "fonts" / "OFL.txt").read_text(encoding="utf-8")
