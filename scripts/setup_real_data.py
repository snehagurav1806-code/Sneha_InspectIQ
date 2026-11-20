"""
setup_real_data.py – Fully automated dataset preparation for real images.

Steps:
  1. Remove all synthetic placeholder images.
  2. Split the 68 real camera photos into good_train (50) and good_val (18).
  3. Generate realistic synthetic defects (scratch, crack, hole, dent)
     FROM the real validation images to populate defect_test/.
  4. Centre-crop and resize everything to 256×256 px.

After this, run train.py → evaluate.py → thresholds.py.
"""

import sys, os, shutil, random
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from pathlib import Path
from config import (IMAGE_SIZE, GOOD_TRAIN, GOOD_VAL, DEFECT_TEST,
                    DATA_DIR, DEFECT_TYPES)

RNG = np.random.default_rng(42)
random.seed(42)

# ── Step 0: Backup info ──────────────────────────────────────────────────
print("=" * 60)
print("  REAL DATA SETUP – Preparing dataset from camera photos")
print("=" * 60)

# ── Step 1: Identify and separate real vs synthetic images ───────────────
print("\n[Step 1/5] Identifying real camera photos...")

all_files = sorted(GOOD_TRAIN.glob("*.*"))
real_files = [f for f in all_files if f.name.startswith("IMG_") and f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
synthetic_files = [f for f in all_files if not f.name.startswith("IMG_") and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

print(f"  Found {len(real_files)} real camera photos")
print(f"  Found {len(synthetic_files)} synthetic placeholder images")

if len(real_files) == 0:
    print("❌ No real camera photos found in good_train/. Add IMG_* files first.")
    sys.exit(1)

# ── Step 2: Remove synthetic data from ALL folders ───────────────────────
print("\n[Step 2/5] Removing synthetic placeholder data...")

# Remove synthetic from good_train
removed = 0
for f in synthetic_files:
    f.unlink()
    removed += 1
print(f"  Removed {removed} synthetic files from good_train/")

# Remove synthetic from good_val
val_removed = 0
for f in GOOD_VAL.glob("*.*"):
    if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
        f.unlink()
        val_removed += 1
print(f"  Removed {val_removed} synthetic files from good_val/")

# Remove synthetic from defect_test (keep folder structure)
defect_removed = 0
if DEFECT_TEST.exists():
    for sub in DEFECT_TEST.iterdir():
        if sub.is_dir():
            for f in sub.glob("*.*"):
                f.unlink()
                defect_removed += 1
print(f"  Removed {defect_removed} synthetic files from defect_test/")

# ── Step 3: Split real photos into train / val ───────────────────────────
print("\n[Step 3/5] Splitting real photos into train/val...")

# Shuffle and split: 50 train, rest val (or 75/25 split)
real_files = sorted(GOOD_TRAIN.glob("IMG_*"))
real_list = list(real_files)
random.shuffle(real_list)

n_train = max(50, int(len(real_list) * 0.75))
n_val = len(real_list) - n_train

train_files = real_list[:n_train]
val_files = real_list[n_train:]

print(f"  Training set: {len(train_files)} images")
print(f"  Validation set: {len(val_files)} images")

# Move val files
GOOD_VAL.mkdir(parents=True, exist_ok=True)
for f in val_files:
    shutil.move(str(f), str(GOOD_VAL / f.name))

print(f"  ✓ Moved {len(val_files)} images to good_val/")

# ── Step 4: Generate synthetic defects from real images ──────────────────
print("\n[Step 4/5] Generating synthetic defect test images...")

def centre_crop_resize(img, size):
    """Resize so short side == size, then centre-crop a size×size square."""
    h, w = img.shape[:2]
    scale = size / min(h, w)
    nw, nh = int(w * scale), int(h * scale)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    y0 = (nh - size) // 2
    x0 = (nw - size) // 2
    return img[y0:y0+size, x0:x0+size]

def add_scratch(img):
    """Draw realistic scratches across the image."""
    out = img.copy()
    h, w = out.shape[:2]
    n_scratches = RNG.integers(1, 4)
    for _ in range(n_scratches):
        # Random start and end points
        x1 = int(RNG.uniform(w * 0.1, w * 0.9))
        y1 = int(RNG.uniform(h * 0.1, h * 0.9))
        angle = RNG.uniform(0, 2 * np.pi)
        length = int(RNG.uniform(h * 0.3, h * 0.8))
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))

        # Jagged scratch path
        pts = []
        steps = RNG.integers(8, 20)
        for t in np.linspace(0, 1, steps):
            px = int(x1 + t * (x2 - x1) + RNG.normal(0, 3))
            py = int(y1 + t * (y2 - y1) + RNG.normal(0, 3))
            pts.append((px, py))
        pts = np.array(pts, dtype=np.int32)

        # Dark scratch line
        thickness = int(RNG.integers(1, 3))
        color = tuple(int(c) for c in RNG.integers(20, 80, 3))
        cv2.polylines(out, [pts], False, color, thickness=thickness, lineType=cv2.LINE_AA)

        # Optional lighter edge highlight (simulates depth)
        if RNG.random() > 0.5:
            pts_shifted = pts + np.array([1, 1])
            cv2.polylines(out, [pts_shifted], False, (200, 200, 200), thickness=1, lineType=cv2.LINE_AA)

    return out

def add_crack(img):
    """Draw branching crack pattern."""
    out = img.copy()
    h, w = out.shape[:2]

    # Start point
    cx = int(RNG.uniform(w * 0.2, w * 0.8))
    cy = int(RNG.uniform(h * 0.2, h * 0.8))

    def draw_branch(start_x, start_y, angle, length, depth, thickness):
        if depth <= 0 or length < 5:
            return
        pts = [(start_x, start_y)]
        x, y = float(start_x), float(start_y)
        step_len = max(3, length // RNG.integers(5, 12))
        for _ in range(int(length // step_len)):
            angle += RNG.normal(0, 0.3)
            x += step_len * np.cos(angle)
            y += step_len * np.sin(angle)
            pts.append((int(x), int(y)))
        pts_arr = np.array(pts, dtype=np.int32)
        color = tuple(int(c) for c in RNG.integers(10, 60, 3))
        cv2.polylines(out, [pts_arr], False, color, thickness=thickness, lineType=cv2.LINE_AA)

        # Branch
        if depth > 1 and RNG.random() > 0.3:
            br_angle = angle + RNG.choice([-1, 1]) * RNG.uniform(0.4, 1.2)
            draw_branch(int(x), int(y), br_angle, length * 0.5, depth - 1, max(1, thickness - 1))

    n_cracks = RNG.integers(1, 3)
    for _ in range(n_cracks):
        angle = RNG.uniform(0, 2 * np.pi)
        length = RNG.uniform(h * 0.2, h * 0.6)
        draw_branch(cx, cy, angle, length, depth=RNG.integers(2, 4), thickness=RNG.integers(1, 3))

    return out

def add_hole(img):
    """Add dark hole/pit defects."""
    out = img.copy()
    h, w = out.shape[:2]
    n_holes = RNG.integers(1, 4)
    for _ in range(n_holes):
        cx = int(RNG.uniform(w * 0.15, w * 0.85))
        cy = int(RNG.uniform(h * 0.15, h * 0.85))
        radius = int(RNG.uniform(5, 20))
        # Dark center
        color = tuple(int(c) for c in RNG.integers(10, 50, 3))
        cv2.circle(out, (cx, cy), radius, color, -1)
        # Lighter rim
        rim_color = tuple(int(c) for c in RNG.integers(80, 140, 3))
        cv2.circle(out, (cx, cy), radius + 2, rim_color, 1)
    return out

def add_dent(img):
    """Add dent/deformation with brightness variation."""
    out = img.copy().astype(np.float32)
    h, w = out.shape[:2]
    n_dents = RNG.integers(1, 3)
    for _ in range(n_dents):
        cx = int(RNG.uniform(w * 0.2, w * 0.8))
        cy = int(RNG.uniform(h * 0.2, h * 0.8))
        rx = int(RNG.uniform(15, 40))
        ry = int(RNG.uniform(15, 40))
        # Create brightness variation mask
        Y, X = np.ogrid[:h, :w]
        dist = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
        mask = np.clip(1.0 - dist, 0, 1)
        # Darken or lighten
        factor = RNG.uniform(-50, -20)
        out += mask[:, :, np.newaxis] * factor

    return np.clip(out, 0, 255).astype(np.uint8)

def add_missing_chunk(img):
    """Cut out a chunk from the object."""
    out = img.copy()
    h, w = out.shape[:2]
    # Pick edge position
    angle = RNG.uniform(0, 2 * np.pi)
    cx, cy = w // 2, h // 2
    # Build a triangular chunk
    r = min(h, w) * 0.4
    half_arc = RNG.uniform(0.2, 0.5)
    pts = [(cx, cy)]
    for a in np.linspace(angle - half_arc, angle + half_arc, 8):
        px = int(cx + r * np.cos(a))
        py = int(cy + r * np.sin(a))
        pts.append((px, py))
    pts = np.array(pts, dtype=np.int32)
    # Fill with background-like color (sample from corner)
    bg_color = tuple(int(c) for c in out[5, 5].tolist())
    cv2.fillPoly(out, [pts], bg_color)
    return out


DEFECT_FNS = {
    "scratch": add_scratch,
    "crack": add_crack,
    "hole": add_hole,
    "dent": add_dent,
    "missing_chunk": add_missing_chunk,
}

# Update config defect types
defect_types = list(DEFECT_FNS.keys())

# Source images for defects: use val images + a few train images
all_source = list(GOOD_VAL.glob("IMG_*")) + list(GOOD_TRAIN.glob("IMG_*"))[:10]
N_PER_DEFECT = 15

for dtype, fn in DEFECT_FNS.items():
    dtype_dir = DEFECT_TEST / dtype
    dtype_dir.mkdir(parents=True, exist_ok=True)

    for i in range(N_PER_DEFECT):
        src_file = all_source[i % len(all_source)]
        img = cv2.imread(str(src_file))
        if img is None:
            continue
        # First crop/resize to standard size
        img = centre_crop_resize(img, IMAGE_SIZE)
        # Apply defect
        defective = fn(img)
        out_path = dtype_dir / f"{dtype}_{i:04d}.png"
        cv2.imwrite(str(out_path), defective)

    print(f"  ✓ Generated {N_PER_DEFECT} {dtype} defect images")

# ── Step 5: Preprocess all images to 256×256 ─────────────────────────────
print("\n[Step 5/5] Preprocessing all images to 256×256...")

def process_folder_in_place(folder, prefix=None):
    """Resize all images in folder to IMAGE_SIZE×IMAGE_SIZE in place."""
    if not folder.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    files = sorted([f for f in folder.iterdir() if f.suffix.lower() in exts])

    tmp = folder / "_tmp_prep"
    tmp.mkdir(exist_ok=True)

    for idx, fpath in enumerate(files):
        img = cv2.imread(str(fpath))
        if img is None:
            print(f"  Warning: cannot read {fpath.name}, skipping.")
            continue
        img = centre_crop_resize(img, IMAGE_SIZE)
        if prefix:
            dst = tmp / f"{prefix}_{idx:04d}.png"
        else:
            dst = tmp / fpath.with_suffix(".png").name
        cv2.imwrite(str(dst), img)

    # Remove originals and move processed
    for f in files:
        f.unlink()
    for f in tmp.iterdir():
        shutil.move(str(f), str(folder / f.name))
    tmp.rmdir()
    return len(files)

n1 = process_folder_in_place(GOOD_TRAIN, prefix="good_train")
print(f"  ✓ Processed {n1} images in good_train/")

n2 = process_folder_in_place(GOOD_VAL, prefix="good_val")
print(f"  ✓ Processed {n2} images in good_val/")

# Defect test images are already 256×256 from generation step
# but let's rename them cleanly
for sub in sorted(DEFECT_TEST.iterdir()):
    if sub.is_dir() and sub.name != "_tmp_prep":
        n = process_folder_in_place(sub, prefix=sub.name)
        # print(f"  ✓ {sub.name}/: {n} images")

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DATASET PREPARATION COMPLETE")
print("=" * 60)

train_count = len(list(GOOD_TRAIN.glob("*.png")))
val_count = len(list(GOOD_VAL.glob("*.png")))
defect_count = sum(1 for _ in DEFECT_TEST.rglob("*.png"))

print(f"  good_train  : {train_count} images (all real camera photos)")
print(f"  good_val    : {val_count} images (real camera photos)")
for sub in sorted(DEFECT_TEST.iterdir()):
    if sub.is_dir():
        n = len(list(sub.glob("*.png")))
        print(f"  defect_test/{sub.name}: {n} images")
print(f"  Total defect: {defect_count} images")
print(f"  Image size  : {IMAGE_SIZE}×{IMAGE_SIZE} px")
print(f"\nNext: Run python scripts/train.py")
