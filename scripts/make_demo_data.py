"""
make_demo_data.py – Generate SYNTHETIC placeholder images for end-to-end testing.

⚠️  THIS DATA IS FAKE.  Results on this data are NOT real performance metrics.
    Delete the contents of data/ and replace with real photos before final demo.

Good images  : grey circle on white background, with a subtle noise texture.
Defect images: same circle, then one of three damage patterns applied:
  scratch       – a jagged dark line across the cap
  dot           – one or more dark blobs (simulating surface pits / contaminants)
  missing_chunk – a white wedge cut out of the cap edge
"""

import sys, os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cv2
from pathlib import Path
from config import (IMAGE_SIZE, GOOD_TRAIN, GOOD_VAL, DEFECT_TEST,
                    N_GOOD_TRAIN, N_GOOD_VAL, N_PER_DEFECT, DEFECT_TYPES)

RNG = np.random.default_rng(42)

# ── Helpers ───────────────────────────────────────────────────────────────

def _base_cap(size: int = IMAGE_SIZE) -> np.ndarray:
    """Return an (H, W, 3) uint8 image of a grey circle on a white background."""
    img = np.full((size, size, 3), 240, dtype=np.uint8)   # white BG
    cx, cy, r = size // 2, size // 2, int(size * 0.42)
    cv2.circle(img, (cx, cy), r, (160, 160, 160), -1)     # grey cap face
    # Subtle radial gradient to mimic lighting
    for dr in range(0, r, 4):
        alpha = int(10 * dr / r)
        cv2.circle(img, (cx, cy), r - dr, (160 + alpha, 160 + alpha, 160 + alpha), 2)
    # Fine noise texture (simulates matte surface)
    noise = RNG.integers(-8, 9, img.shape, dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _add_scratch(img: np.ndarray) -> np.ndarray:
    """Draw a jagged dark line across the cap face."""
    img = img.copy()
    size = img.shape[0]
    cx, cy, r = size // 2, size // 2, int(size * 0.42)
    # Random start / end within the cap
    angle1 = RNG.uniform(0, 2 * np.pi)
    angle2 = angle1 + RNG.uniform(np.pi * 0.5, np.pi * 1.0)
    pts = []
    for a in np.linspace(angle1, angle2, 12):
        px = int(cx + (r - 10) * np.cos(a) * RNG.uniform(0.4, 0.95))
        py = int(cy + (r - 10) * np.sin(a) * RNG.uniform(0.4, 0.95))
        pts.append((px, py))
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts], False, (40, 40, 40),
                  thickness=RNG.integers(1, 4))
    return img


def _add_dot(img: np.ndarray) -> np.ndarray:
    """Add 1–4 dark blobs simulating pits or contamination."""
    img = img.copy()
    size = img.shape[0]
    cx, cy, r = size // 2, size // 2, int(size * 0.42)
    n_dots = RNG.integers(1, 5)
    for _ in range(n_dots):
        angle = RNG.uniform(0, 2 * np.pi)
        dist  = RNG.uniform(0, r * 0.75)
        px = int(cx + dist * np.cos(angle))
        py = int(cy + dist * np.sin(angle))
        rad = RNG.integers(4, 14)
        cv2.circle(img, (px, py), rad, (30, 30, 30), -1)
    return img


def _add_missing_chunk(img: np.ndarray) -> np.ndarray:
    """Cut a triangular wedge from the edge of the cap."""
    img = img.copy()
    size = img.shape[0]
    cx, cy, r = size // 2, size // 2, int(size * 0.42)
    angle_mid = RNG.uniform(0, 2 * np.pi)
    half_arc  = RNG.uniform(0.2, 0.5)        # radians
    depth     = RNG.uniform(r * 0.25, r * 0.6)
    # Build a filled polygon: outer arc edge inward
    pts = [(cx, cy)]                          # center
    for a in np.linspace(angle_mid - half_arc, angle_mid + half_arc, 8):
        pts.append((int(cx + r * np.cos(a)), int(cy + r * np.sin(a))))
    pts.append((int(cx + (r - depth) * np.cos(angle_mid)),
                int(cy + (r - depth) * np.sin(angle_mid))))
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(img, [pts], (240, 240, 240))  # fill with BG colour
    return img


DEFECT_FN = {
    "scratch":       _add_scratch,
    "dot":           _add_dot,
    "missing_chunk": _add_missing_chunk,
}

# ── Main generation ───────────────────────────────────────────────────────

def generate(folder: Path, n: int, defect_fn=None, label: str = "good") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = _base_cap()
        if defect_fn is not None:
            img = defect_fn(img)
        path = folder / f"{label}_{i:04d}.png"
        cv2.imwrite(str(path), img)
    print(f"  ✓  {n} images → {folder}")


if __name__ == "__main__":
    print("=" * 60)
    print("⚠️  SYNTHETIC DEMO DATA – NOT REAL INSPECTION IMAGES")
    print("=" * 60)

    print("\n[1/3] Good training images …")
    generate(GOOD_TRAIN, N_GOOD_TRAIN, defect_fn=None, label="good")

    print("[2/3] Good validation images …")
    generate(GOOD_VAL, N_GOOD_VAL, defect_fn=None, label="good")

    print("[3/3] Defect test images …")
    for dtype in DEFECT_TYPES:
        dest = DEFECT_TEST / dtype
        generate(dest, N_PER_DEFECT, DEFECT_FN[dtype], label=dtype)

    total = N_GOOD_TRAIN + N_GOOD_VAL + len(DEFECT_TYPES) * N_PER_DEFECT
    print(f"\nDone. {total} synthetic images created.")
    print("To use real photos: delete data/good_train, data/good_val, data/defect_test")
    print("and drop your images in those folders, then re-run scripts/prepare_images.py")

