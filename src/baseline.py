"""Classical OpenCV baseline: flag regions that differ from a blurred copy of the image."""
import cv2, numpy as np

def detect(img_bgr, k=31, thr=35, min_area=80):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g, cv2.medianBlur(g, k))
    _, m = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return [(x, y, x+w, y+h) for x, y, w, h, a in st[1:] if a >= min_area]
