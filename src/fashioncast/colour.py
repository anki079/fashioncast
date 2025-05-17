import cv2
import numpy as np
from pathlib import Path
from .constants import CACHE_ROOT


def hue_histogram(img_bgr: np.ndarray, bins: int = 12) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]  # 0–179
    hist, _ = np.histogram(h, bins=bins, range=(0, 180), density=False)
    return hist.astype(np.float32) / hist.sum()


def process_one(img_path: str):
    out_file = CACHE_ROOT / "colour" / (Path(img_path).stem + ".npy")
    if out_file.exists():
        return
    out_file.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(img_path)
    np.save(out_file, hue_histogram(img))
