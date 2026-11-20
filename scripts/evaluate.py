"""
evaluate.py – Score every image in good_val and defect_test with the trained PatchCore.

For each image:
  - Runs model.model(tensor) inference to get raw continuous PatchCore distance.
  - Normalises scores to [0, 1] across the evaluation dataset.
  - Generates heatmap overlays and saves them to results/heatmaps/<split>/<filename>.png.
Outputs results/scores.csv with columns:
  image, split, defect_type, label, score, heatmap_path
"""

import sys, os, warnings
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from config import (GOOD_VAL, DEFECT_TEST, MODELS_DIR, RESULTS_DIR,
                    HEATMAPS_DIR, SCORES_CSV, IMAGE_SIZE)

from anomalib.models import Patchcore


def find_checkpoint() -> Path:
    """Find the trained PatchCore checkpoint under models/."""
    ckpt_dir = MODELS_DIR / "patchcore"
    ckpts = list(ckpt_dir.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}. Run train.py first.")
    # Pick the most recently modified checkpoint
    return sorted(ckpts, key=lambda p: p.stat().st_mtime)[-1]


def load_model(ckpt_path: Path) -> Patchcore:
    """Load a trained PatchCore model from a Lightning checkpoint."""
    model = Patchcore.load_from_checkpoint(str(ckpt_path))
    model.eval()
    return model


def preprocess_image(img_path: str | Path) -> torch.Tensor:
    """Load an image, resize to IMAGE_SIZE, apply ImageNet norm, return [1, 3, H, W] tensor."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = (img - mean) / std
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


@torch.no_grad()
def infer_one(model: Patchcore, img_path: str | Path) -> tuple[float, np.ndarray]:
    """
    Run core PatchCore model inference.
    Returns:
      raw_score: float (nearest-neighbour patch anomaly distance)
      amap: (H, W) float32 2D spatial heatmap
    """
    tensor = preprocess_image(img_path)
    output = model.model(tensor)
    
    score = float(output.pred_score.squeeze().item())
    amap = output.anomaly_map.squeeze().cpu().numpy().astype(np.float32)
    amap = cv2.resize(amap, (IMAGE_SIZE, IMAGE_SIZE))
    return score, amap


def save_heatmap(orig_path: Path, amap: np.ndarray, out_path: Path, score: float) -> None:
    """Overlay the anomaly map on the original image and save."""
    orig = cv2.imread(str(orig_path))
    orig = cv2.resize(orig, (IMAGE_SIZE, IMAGE_SIZE))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if amap.max() > amap.min():
        hm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        hm = (hm * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig, 0.55, hm_color, 0.45, 0)
    else:
        overlay = orig.copy()

    # Add text banner
    cv2.putText(overlay, f"Score: {score:.3f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(out_path), overlay)


def collect_images(split: str, folder: Path, label: int, defect_type: str) -> list[dict]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return [
        {"image": f.name, "split": split, "defect_type": defect_type,
         "label": label, "path": f}
        for f in sorted(folder.iterdir())
        if f.is_file() and f.suffix.lower() in exts
    ]


if __name__ == "__main__":
    print("=" * 60)
    print("  PatchCore Evaluation  (anomalib 2.7.0)")
    print("=" * 60)

    ckpt = find_checkpoint()
    print(f"Loading checkpoint: {ckpt}")
    model = load_model(ckpt)
    print("Model loaded ✓")

    items: list[dict] = []
    items += collect_images("good_val", GOOD_VAL, label=0, defect_type="none")
    for dtype_dir in sorted(DEFECT_TEST.iterdir()):
        if dtype_dir.is_dir():
            items += collect_images("defect_test", dtype_dir, label=1,
                                    defect_type=dtype_dir.name)

    print(f"\nScoring {len(items)} images …")
    raw_scores = []
    amaps = []

    for i, item in enumerate(items):
        s, a = infer_one(model, item["path"])
        raw_scores.append(s)
        amaps.append(a)
        if (i + 1) % 15 == 0 or (i + 1) == len(items):
            print(f"  {i+1}/{len(items)} scored …")

    # Min-max scale scores to [0, 1] range for intuitive thresholds
    arr = np.array(raw_scores, dtype=np.float64)
    s_min, s_max = float(arr.min()), float(arr.max())
    norm_scores = (arr - s_min) / (s_max - s_min + 1e-8)

    records = []
    for i, item in enumerate(items):
        ns = float(norm_scores[i])
        split_dir = HEATMAPS_DIR / item["split"]
        heatmap_path = split_dir / item["image"].replace(".jpg", ".png")
        save_heatmap(item["path"], amaps[i], heatmap_path, ns)
        records.append({
            "image":        item["image"],
            "split":        item["split"],
            "defect_type":  item["defect_type"],
            "label":        item["label"],
            "score":        round(ns, 4),
            "raw_score":    round(float(raw_scores[i]), 4),
            "heatmap_path": str(heatmap_path),
        })

    df = pd.DataFrame(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCORES_CSV, index=False)
    print(f"\n✓ Scores saved  → {SCORES_CSV}")
    print(f"✓ Heatmaps saved→ {HEATMAPS_DIR}")

    good_s   = df[df["label"] == 0]["score"]
    defect_s = df[df["label"] == 1]["score"]
    print(f"\nEvaluation summary (normalised score [0, 1]):")
    print(f"  Good parts   (n={len(good_s)}): mean={good_s.mean():.4f}, std={good_s.std():.4f}, min={good_s.min():.4f}, max={good_s.max():.4f}")
    print(f"  Defect parts (n={len(defect_s)}): mean={defect_s.mean():.4f}, std={defect_s.std():.4f}, min={defect_s.min():.4f}, max={defect_s.max():.4f}")
    print("\nNext: Run scripts/thresholds.py to compute PASS / REVIEW / REJECT zones.")
