"""Synthetic fabric images with random defects, in YOLO format.
ONLY for smoke-testing the pipeline. Never report results from this data."""
import sys, random
from pathlib import Path
import cv2, numpy as np

def make(out, n=40, size=320, seed=0):
    rng = random.Random(seed); out = Path(out)
    for split, cnt in (("train", n), ("val", max(8, n // 4))):
        (out/"images"/split).mkdir(parents=True, exist_ok=True)
        (out/"labels"/split).mkdir(parents=True, exist_ok=True)
        for i in range(cnt):
            base = np.tile(np.array([[120, 140], [140, 120]], np.uint8), (size//2, size//2))
            img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            img = cv2.add(img, np.random.RandomState(rng.randint(0, 9999)).randint(0, 12, img.shape, np.uint8))
            lines = []
            for _ in range(rng.randint(0, 2)):
                w, h = rng.randint(20, 70), rng.randint(10, 40)
                x, y = rng.randint(0, size-w), rng.randint(0, size-h)
                cls = rng.randint(0, 1)
                cv2.rectangle(img, (x, y), (x+w, y+h), (30, 30, 30) if cls == 0 else (230, 230, 230), -1)
                lines.append(f"{cls} {(x+w/2)/size:.6f} {(y+h/2)/size:.6f} {w/size:.6f} {h/size:.6f}")
            cv2.imwrite(str(out/"images"/split/f"{i:04d}.jpg"), img)
            (out/"labels"/split/f"{i:04d}.txt").write_text("\n".join(lines))
    (out/"data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: dark_defect\n  1: light_defect\n")

if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "data/synth")
