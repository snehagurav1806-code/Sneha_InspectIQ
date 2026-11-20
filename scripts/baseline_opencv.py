"""
baseline_opencv.py – Simple mean-image anomaly detector used as a comparison baseline.

Compares each test image to the per-pixel mean of the good_train set using absolute
pixel difference.  Outputs baseline_scores.csv in the same format as scores.csv so the
Streamlit comparison tab can read it directly.
"""

import sys, os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from config import (IMAGE_SIZE, GOOD_TRAIN, GOOD_VAL, DEFECT_TEST,
                    BASELINE_DIFF_THRESHOLD, BASELINE_CSV, RESULTS_DIR)


def load_images(folder: Path) -> list[np.ndarray]:
    """Return list of float32 BGR images resized to IMAGE_SIZE."""
    imgs = []
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in exts:
            continue
        img = cv2.imread(str(f))
        if img is None:
            continue
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
        imgs.append(img)
    return imgs


def compute_mean_image(imgs: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(imgs, axis=0), axis=0)


def score_image(img: np.ndarray, mean_img: np.ndarray) -> float:
    """Anomaly score = mean absolute pixel difference (0-255 range), normalised to [0,1]."""
    diff = np.abs(img - mean_img)
    return float(diff.mean() / 255.0)


def evaluate(split: str, folder: Path, label: int, defect_type: str,
             mean_img: np.ndarray, records: list) -> None:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in exts:
            continue
        img = cv2.imread(str(f))
        if img is None:
            continue
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
        score = score_image(img, mean_img)
        records.append({
            "image":       f.name,
            "split":       split,
            "defect_type": defect_type,
            "label":       label,   # 0 = good, 1 = defect
            "score":       round(score, 6),
        })


if __name__ == "__main__":
    print("Building mean good image from good_train …")
    train_imgs = load_images(GOOD_TRAIN)
    if not train_imgs:
        print("ERROR: No images found in", GOOD_TRAIN)
        sys.exit(1)
    mean_img = compute_mean_image(train_imgs)
    print(f"  Mean image computed from {len(train_imgs)} images.")

    records: list = []

    print("Scoring good_val …")
    evaluate("good_val", GOOD_VAL, label=0, defect_type="none",
             mean_img=mean_img, records=records)

    print("Scoring defect_test …")
    for dtype_dir in sorted(DEFECT_TEST.iterdir()):
        if dtype_dir.is_dir():
            evaluate("defect_test", dtype_dir, label=1, defect_type=dtype_dir.name,
                     mean_img=mean_img, records=records)

    df = pd.DataFrame(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(BASELINE_CSV, index=False)
    print(f"\nSaved {len(df)} rows → {BASELINE_CSV}")

    # Quick summary
    good = df[df["label"] == 0]["score"]
    bad  = df[df["label"] == 1]["score"]
    print(f"\nBaseline score summary:")
    print(f"  Good  : mean={good.mean():.4f}, std={good.std():.4f}")
    print(f"  Defect: mean={bad.mean():.4f},  std={bad.std():.4f}")

