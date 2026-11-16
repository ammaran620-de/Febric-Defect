"""Convert defect masks (white = defect) to YOLO boxes. Single class by default.
Usage: python src/mask2yolo.py IMG_DIR MASK_DIR OUT_DIR [--val-frac 0.2]
Masks must share the image file stem (mask may carry a '_mask' suffix). Check a few
outputs by eye before training: mask conventions differ between datasets."""
import argparse, random
from pathlib import Path
import cv2

def boxes_from_mask(mask, min_area=30):
    _, b = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(b)
    return [tuple(stats[i][:4]) for i in range(1, n) if stats[i][4] >= min_area]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("img_dir"); ap.add_argument("mask_dir"); ap.add_argument("out")
    ap.add_argument("--val-frac", type=float, default=0.2); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); out = Path(a.out)
    imgs = sorted(p for p in Path(a.img_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    random.Random(a.seed).shuffle(imgs); nval = max(1, int(len(imgs)*a.val_frac))
    for k, p in enumerate(imgs):
        split = "val" if k < nval else "train"
        for s in ("images", "labels"): (out/s/split).mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(p)); H, W = img.shape[:2]
        mp = next((m for m in (Path(a.mask_dir)/f"{p.stem}{s}{e}" for s in ("", "_mask") for e in (".png", ".jpg", ".bmp")) if m.exists()), None)
        lines = []
        if mp is not None:
            m = cv2.imread(str(mp), 0)
            for x, y, w, h in boxes_from_mask(m):
                lines.append(f"0 {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}")
        cv2.imwrite(str(out/"images"/split/p.name), img)
        (out/"labels"/split/f"{p.stem}.txt").write_text("\n".join(lines))
    (out/"data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: defect\n")

if __name__ == "__main__": main()
