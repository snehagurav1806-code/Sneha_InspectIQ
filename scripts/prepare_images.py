"""
prepare_images.py – Resize and centre-crop every image in the data folders to
                    the configured IMAGE_SIZE, and rename them sequentially.
Run this once after adding raw photos (or after make_demo_data.py for a sanity check).
"""

import sys, os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import shutil
from pathlib import Path
from config import IMAGE_SIZE, GOOD_TRAIN, GOOD_VAL, DEFECT_TEST

def centre_crop_resize(img: "np.ndarray", size: int) -> "np.ndarray":
    """Resize so the short side == size, then centre-crop a size×size square."""
    import numpy as np
    h, w = img.shape[:2]
    # Scale so short side = size
    scale = size / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Centre crop
    y0 = (new_h - size) // 2
    x0 = (new_w - size) // 2
    return img[y0:y0+size, x0:x0+size]


def process_folder(folder: Path, prefix: str) -> int:
    """Process all images in folder in-place and rename sequentially."""
    if not folder.exists():
        print(f"  Skipping (not found): {folder}")
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    files = sorted([f for f in folder.iterdir() if f.suffix.lower() in exts])
    # Write to a tmp sub-folder to avoid name collisions during rename
    tmp = folder / "_tmp_prep"
    tmp.mkdir(exist_ok=True)
    for idx, fpath in enumerate(files):
        img = cv2.imread(str(fpath))
        if img is None:
            print(f"  Warning: cannot read {fpath.name}, skipping.")
            continue
        img = centre_crop_resize(img, IMAGE_SIZE)
        dst = tmp / f"{prefix}_{idx:04d}.png"
        cv2.imwrite(str(dst), img)
    # Remove originals and move processed files up
    for f in files:
        f.unlink()
    for f in tmp.iterdir():
        shutil.move(str(f), str(folder / f.name))
    tmp.rmdir()
    print(f"  ✓  {len(files)} images → {folder}")
    return len(files)


if __name__ == "__main__":
    total = 0
    total += process_folder(GOOD_TRAIN, "good_train")
    total += process_folder(GOOD_VAL,   "good_val")
    if DEFECT_TEST.exists():
        for sub in sorted(DEFECT_TEST.iterdir()):
            if sub.is_dir():
                total += process_folder(sub, sub.name)
    print(f"\nPrepared {total} images at {IMAGE_SIZE}×{IMAGE_SIZE} px.")

