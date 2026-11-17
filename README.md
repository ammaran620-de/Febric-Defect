<div align="center">

# Fabric Defect Inspection

**YOLOv8n defect detection for textile images, with a FastAPI service, an audit log and a dashboard.**

[Live demo](https://ammaran620-de.github.io/Febric-Defect/) · [Full write-up](docs/README_full.md) · [Architecture](docs/ARCHITECTURE.md)

</div>

![Dashboard overview](docs/dashboard-overview.png)

A fine-tuned YOLOv8n detector looks at a fabric image and returns defect boxes. Any box above the confidence threshold makes the verdict FAIL. Every result is saved with its verdict, classes, confidence, time and an annotated thumbnail, so a quality manager can review it later.

The live demo runs the model in your browser with ONNX Runtime. Nothing is uploaded.

## Results

Validation set: 80 images, 101 labelled defects, 4 classes (hole, objects, oil spot, thread error). Confidence 0.25, IoU 0.5. These are two different scores; do not mix them.

**Class-agnostic localization.** A box counts if it overlaps a labelled defect, whatever class it names. This is the only scoring the OpenCV baseline can take part in, because it predicts no class, so it is the fair comparison.

| | YOLOv8n | OpenCV baseline |
|---|---:|---:|
| Precision | 0.887 | 0.140 |
| Recall | 0.851 | 0.069 |
| F1 | 0.869 | 0.093 |
| True positives, false positives, misses | 86, 11, 15 | 7, 43, 94 |

**Class-aware (YOLOv8n only).** The box must also name the labelled defect's class. A box in the right place with the wrong class is a false positive and the defect stays a miss.

| | YOLOv8n |
|---|---:|
| Precision | 0.866 |
| Recall | 0.832 |
| F1 | 0.848 |
| True positives, false positives, misses | 84, 13, 17 |

Four of the 13 class-aware false positives are boxes in the right place with the wrong class. The Ultralytics validator reports mAP50 0.880 (which does require the class). Per-class counts are in `results/summary.json`.

At the class-agnostic score the model finds 86 of 101 defects. At confidence 0.55, class-agnostic precision rises to 0.97 and recall drops to 0.64. Which class is missed most depends on how you measure: the validator's per-class recall is lowest for thread errors (0.70), while at the fixed 0.25 threshold with the class required the misses are 4 of 23 holes, 7 of 33 objects, 1 of 20 oil spots and 5 of 25 thread errors. The baseline is a simple blur and threshold, so it shows that a learned model helps, not that this is state of the art.

Two limits on these numbers. The `best.pt` checkpoint is the epoch Ultralytics kept for its best validation fitness, so it was chosen with these same 80 images in view, and the figures are likely a little optimistic. Every validation image contains at least one labelled defect, so none of these precision figures is a factory false-alarm rate on good fabric.

## Run it

```bash
git clone https://github.com/ammaran620-de/Febric-Defect.git
cd Febric-Defect
pip install -r requirements.txt
python -m uvicorn src.app:app --port 8000
```

On Windows, `run.bat` does the same. Then open http://localhost:8000. Run the tests with `python -m pytest -q`. GitHub Actions runs them on every push.

## What it includes

- Inspect one image, run a batch of images, or use the browser webcam preview (the preview is not saved to the log)
- A SQLite log with CSV export, and a history page
- A model page with metrics, a threshold sweep and a confusion matrix
- An error-review page that shows ground truth next to each prediction
- Three themes: Light, Blue and Dark

## Limits

- **Timing.** The model time shown for each inspection is the duration of the `predict` call only. The first inspection after the server starts is much slower because of one-time initialisation (in one cold run: 1218 ms, then 50 and 44 ms). Judge latency from later calls.
- 400 public images, one train and validation split, no separate test set and no defect-free images. The real false-alarm rate is not measured.
- Lab-style images with even lighting. Expect lower numbers on a real mill without retraining.
- No login and no industrial camera integration (the webcam option is a browser preview only). The Docker image has not been tested.

## Data and licences

The code is MIT. The images come from the TILDA textile database through a public subset that states no licence, so the dataset and the screenshots that show it are **not** covered by the MIT licence. YOLOv8 comes from Ultralytics (AGPL-3.0), and the bundled weights are derived from it, so they are not MIT either. `THIRD_PARTY_NOTICES.md` lists each item and one open licensing question. The Archivo font used by the dashboard is under the SIL Open Font License 1.1 (notice in `docs/fonts/OFL.txt`).

## Author

Ammara Noor, BS Computer Engineering, NUTECH, Islamabad. [GitHub](https://github.com/ammaran620-de)
