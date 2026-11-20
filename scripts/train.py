"""
train.py – Train a PatchCore anomaly detection model on good parts only.

Uses the verified anomalib 2.7.0 API (Engine + Folder datamodule).
Training images  : data/good_train/
Validation images: data/good_val/      (scored, NOT used to train)
Saves model checkpoint to: models/patchcore/

PatchCore does not iterate over epochs in the traditional sense —
it builds a coreset (compressed memory bank of patch features) in ONE pass.
On CPU with 256×256 images this takes ~1-3 minutes for 50 images.
"""

import sys, os, shutil, warnings
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore", category=FutureWarning)   # suppress timm/anomalib deprecation noise

from pathlib import Path
from config import (GOOD_TRAIN, GOOD_VAL, DEFECT_TEST, MODELS_DIR,
                    IMAGE_SIZE, CORESET_RATIO, BACKBONE, DATA_DIR)

# ── Anomalib 2.7.0 verified API ──────────────────────────────────────────
from anomalib.data   import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine


def flatten_defects(tmp_dir: Path) -> Path:
    """
    Folder datamodule expects a single abnormal_dir flat folder.
    Copy all defect sub-folders into one temporary flat directory.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for sub in sorted(DEFECT_TEST.iterdir()):
        if sub.is_dir():
            for img in sub.iterdir():
                dst = tmp_dir / f"{sub.name}_{img.name}"
                if not dst.exists():
                    shutil.copy2(str(img), str(dst))
    return tmp_dir


def build_datamodule(abnormal_flat: Path) -> Folder:
    """
    Folder API (anomalib 2.7.0):
      name, normal_dir, root, abnormal_dir, normal_test_dir,
      train_batch_size, eval_batch_size, num_workers, seed
    Image resizing is handled upstream by prepare_images.py / make_demo_data.py.
    """
    return Folder(
        name            = "implant_caps",
        root            = str(DATA_DIR),
        normal_dir      = "good_train",
        abnormal_dir    = str(abnormal_flat.relative_to(DATA_DIR)),
        normal_test_dir = "good_val",
        train_batch_size= 8,     # small batch, CPU-friendly
        eval_batch_size = 8,
        num_workers     = 0,     # 0 on Windows to avoid multiprocessing issues
        seed            = 42,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  PatchCore Training  (anomalib 2.7.0 · CPU-only)")
    print(f"  Backbone: {BACKBONE}")
    print(f"  Coreset ratio: {CORESET_RATIO}")
    print("=" * 60)

    # Flatten defect folders for the datamodule
    abnormal_tmp = DATA_DIR / "_abnormal_flat"
    flatten_defects(abnormal_tmp)

    dm = build_datamodule(abnormal_tmp)

    model = Patchcore(
        backbone               = BACKBONE,
        layers                 = ["layer2", "layer3"],
        pre_trained            = True,
        coreset_sampling_ratio = CORESET_RATIO,
        num_neighbors          = 9,
        precision              = "float32",
    )

    out_dir = str(MODELS_DIR / "patchcore")
    # Engine passes **kwargs directly to the Lightning Trainer
    engine  = Engine(
        default_root_dir = out_dir,
        max_epochs       = 1,          # single-pass coreset build
        accelerator      = "cpu",
        devices          = 1,
        enable_progress_bar = True,
    )

    print("\n[1/2] Building memory bank (single-pass training) …")
    engine.fit(model=model, datamodule=dm)

    print("\n[2/2] Quick test on val+defect set …")
    engine.test(model=model, datamodule=dm)

    # Clean up the temporary flat abnormal dir
    shutil.rmtree(str(abnormal_tmp), ignore_errors=True)

    print(f"\n✓ Model checkpoint saved → {out_dir}")
    print("Training complete. Run evaluate.py next.")
