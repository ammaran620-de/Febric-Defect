# Third-party notices

`LICENSE` (MIT) covers the original code, documentation and recorded results in this repository. It does not cover the items below.

## Model weights

`results/best.pt` and `docs/model/best.onnx` are YOLOv8n weights fine-tuned from the COCO-pretrained `yolov8n.pt` checkpoint with the Ultralytics training code. Ultralytics publishes its code and pretrained weights under AGPL-3.0 (the `ultralytics` package on PyPI declares AGPL-3.0; checked 2026-11-17), with a paid enterprise licence as the alternative. Treat the two weight files as derivatives under AGPL-3.0, not as MIT.

## Dataset images

The images come from the TILDA textile database through a public subset that states no licence. The 80 thumbnails in `docs/demo/thumbs/`, the files in `results/demo/inspections/` and `results/errors/`, and the dashboard screenshots in `docs/` show those images and are not covered by MIT. I could not find a licence for the subset, so I have not assumed one; do not redistribute the images without checking with the dataset's owners.

## Bundled web assets

| Path | What it is | Licence |
|---|---|---|
| `docs/vendor/` | ONNX Runtime Web 1.31.1 build files, used by the in-browser demo | MIT, Microsoft; the notice is in the header of `ort.wasm.min.js` |
| `docs/fonts/archivo.woff2`, `src/static/fonts/` | Archivo font | SIL Open Font License 1.1, notice in `docs/fonts/OFL.txt` |

## Python dependencies

`ultralytics` (AGPL-3.0) is imported by the FastAPI service and the training and evaluation scripts in `src/`. Other dependencies keep their own licences and are installed from `requirements.txt`, not bundled.

## Open question

I have not settled whether MIT code that imports the AGPL-3.0 `ultralytics` package, and is published beside AGPL-derived weights, can stay MIT as a whole when someone redistributes it. This is a legal question, not a technical one. The options are to keep the current arrangement with this notice, to relicense the repository as AGPL-3.0, or to remove the weight files and the `ultralytics` dependency. Nothing has been changed pending that decision, and the project is not offered as cleared for commercial redistribution.
