"""
recalibrate.py – Quick venue recalibration utility for Biorad Medisys demo.

When lighting or camera setup changes at the hackathon venue:
  1. Capture 20 fresh good (flawless) component images under venue lighting.
  2. Run: python scripts/recalibrate.py --new_good path/to/venue_good_photos/
  3. The script prepares images, refits PatchCore memory bank, rescores,
     and re-calculates operational thresholds in under 2 minutes.
"""

import sys, os, argparse, shutil, subprocess
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from pathlib import Path
import config

def main():
    parser = argparse.ArgumentParser(description="Venue Quick Recalibration Utility")
    parser.add_argument("--new_good", required=True,
                        help="Path to folder containing >= 20 photos of good parts under venue lighting")
    parser.add_argument("--keep_old", action="store_true",
                        help="Append to existing good_train instead of wiping")
    args = parser.parse_args()

    src = Path(args.new_good)
    if not src.exists():
        print(f"❌ Error: Source directory '{src}' does not exist.")
        sys.exit(1)

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    raw_files = [f for f in sorted(src.iterdir()) if f.suffix.lower() in valid_exts]

    if len(raw_files) < 10:
        print(f"❌ Error: Found only {len(raw_files)} images in '{src}'. Need at least 15-20 good parts.")
        sys.exit(1)

    print(f"Found {len(raw_files)} new good images for recalibration.")

    if not args.keep_old:
        print(f"Clearing previous good_train samples...")
        for old_f in config.GOOD_TRAIN.glob("*"):
            if old_f.is_file():
                old_f.unlink()

    # Process and copy into good_train
    config.GOOD_TRAIN.mkdir(parents=True, exist_ok=True)
    count = 0
    for idx, fpath in enumerate(raw_files):
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        # Resize & center crop
        h, w = img.shape[:2]
        scale = config.IMAGE_SIZE / min(h, w)
        nw, nh = int(w * scale), int(h * scale)
        img_res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        y0 = (nh - config.IMAGE_SIZE) // 2
        x0 = (nw - config.IMAGE_SIZE) // 2
        crop = img_res[y0:y0+config.IMAGE_SIZE, x0:x0+config.IMAGE_SIZE]

        dest = config.GOOD_TRAIN / f"recalib_{idx:04d}.png"
        cv2.imwrite(str(dest), crop)
        count += 1

    print(f"✓ Successfully staged {count} images in {config.GOOD_TRAIN}")

    py_exe = sys.executable
    root_scripts = config.ROOT / "scripts"

    print("\n[Step 1/3] Retraining PatchCore memory bank...")
    ret = subprocess.run([py_exe, str(root_scripts / "train.py")])
    if ret.returncode != 0:
        print("❌ Training failed.")
        sys.exit(1)

    print("\n[Step 2/3] Evaluating validation & defect sets...")
    ret = subprocess.run([py_exe, str(root_scripts / "evaluate.py")])
    if ret.returncode != 0:
        print("❌ Evaluation failed.")
        sys.exit(1)

    print("\n[Step 3/3] Re-calibrating PASS / REVIEW / REJECT thresholds...")
    ret = subprocess.run([py_exe, str(root_scripts / "thresholds.py")])
    if ret.returncode != 0:
        print("❌ Threshold calculation failed.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ RECALIBRATION COMPLETE!")
    print("New thresholds and memory bank are active. Reload Streamlit app.")
    print("=" * 60)

if __name__ == "__main__":
    main()
