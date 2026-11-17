<div align="center">

# Fabric Defect Inspection

**YOLOv8n defect detection for textile images, with a FastAPI service, an audit log and a dashboard.**

[Live demo](https://ammaran620-de.github.io/Febric-Defect/) · [Full write-up](docs/README_full.md) · [Architecture](docs/ARCHITECTURE.md)

</div>

![Dashboard overview](docs/dashboard-overview.png)

A fine-tuned YOLOv8n detector looks at a fabric image and returns defect boxes. Any box above the confidence threshold makes the verdict FAIL. Every result is saved with its verdict, classes, confidence, time and an annotated thumbnail, so a quality manager can review it later.

The live demo runs the model in your browser with ONNX Runtime. Nothing is uploaded.

## Results

Held-out validation set: 80 images, 101 labelled defects, 4 classes (hole, objects, oil spot, thread error). Confidence 0.25, IoU 0.5.

| | YOLOv8n | OpenCV baseline |
|---|---:|---:|
| Precision | 0.887 | 0.140 |
| Recall | 0.852 | 0.069 |
| F1 | 0.869 | 0.093 |
| mAP50 | 0.880 | n/a |

The model finds 86 of 101 defects, misses 15 and raises 11 false alarms. Thread errors are missed most often (recall 0.70). At confidence 0.55, precision rises to 0.97 and recall drops to 0.64. The baseline is a simple blur and threshold, so it shows that a learned model helps, not that this is state of the art.

## Run it

```bash
git clone https://github.com/ammaran620-de/Febric-Defect.git
cd Febric-Defect
pip install -r requirements.txt
python -m uvicorn src.app:app --port 8000
```

On Windows, `run.bat` does the same. Then open http://localhost:8000. Run the tests with `python -m pytest -q`. GitHub Actions runs them on every push.

## What it includes

- Inspect one image or a webcam stream, and batch inspection of many images
- A SQLite log with CSV export, and a history page
- A model page with metrics, a threshold sweep and a confusion matrix
- An error-review page that shows ground truth next to each prediction
- Three themes: Light, Blue and Dark

## Limits

- **Timing.** The model time shown for each inspection is the duration of the `predict` call only. The first inspection after the server starts is much slower because of one-time initialisation (in one cold run: 1218 ms, then 50 and 44 ms). Judge latency from later calls.
- 400 public images, one train and validation split, no separate test set and no defect-free images. The real false-alarm rate is not measured.
- Lab-style images with even lighting. Expect lower numbers on a real mill without retraining.
- No login, no camera integration, and the Docker image has not been tested.

## Data and licences

The code is MIT. The images come from the TILDA textile database through a public subset that states no licence, so the dataset and the screenshots that show it are **not** covered by the MIT licence. YOLOv8 comes from Ultralytics (AGPL-3.0).

## Author

Ammara Noor, BS Computer Engineering, NUTECH, Islamabad. [GitHub](https://github.com/ammaran620-de)
