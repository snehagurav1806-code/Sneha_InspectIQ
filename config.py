"""
config.py – Single source of truth for all tuneable settings.
Change OBJECT_NAME here and the rest of the codebase updates automatically.
"""

from pathlib import Path

# ── Object identity ───────────────────────────────────────────────────────
# Change this to match whatever you are physically inspecting.
OBJECT_NAME = "Plastic Bottle Cap"          # human-readable label
OBJECT_CODE = "cap"                          # used in filenames / folders

# ── Paths (all relative to the project root) ─────────────────────────────
ROOT         = Path(__file__).parent
DATA_DIR     = ROOT / "data"
GOOD_TRAIN   = DATA_DIR / "good_train"
GOOD_VAL     = DATA_DIR / "good_val"
DEFECT_TEST  = DATA_DIR / "defect_test"     # sub-dirs = defect types
MODELS_DIR   = ROOT / "models"
LOGS_DIR     = ROOT / "logs"
RESULTS_DIR  = ROOT / "results"
HEATMAPS_DIR = RESULTS_DIR / "heatmaps"

SCORES_CSV        = RESULTS_DIR / "scores.csv"
THRESHOLDS_JSON   = RESULTS_DIR / "thresholds.json"
INSPECTIONS_CSV   = LOGS_DIR   / "inspections.csv"
BASELINE_CSV      = RESULTS_DIR / "baseline_scores.csv"

# ── Image settings (CPU-friendly) ────────────────────────────────────────
IMAGE_SIZE   = 256          # resize & centre-crop to this square (px)

# ── PatchCore settings ───────────────────────────────────────────────────
# coreset_sampling_ratio: smaller = faster on CPU, slight accuracy trade-off
CORESET_RATIO    = 0.1      # 10 % of patches kept in memory bank
BACKBONE         = "wide_resnet50_2"
ANOMALY_MAP_SIZE = IMAGE_SIZE   # output heatmap resolution

# ── Default thresholds (overwritten by thresholds.py output) ─────────────
THRESHOLD_LOW  = 0.40       # below → PASS
THRESHOLD_HIGH = 0.60       # above → REJECT  (between = REVIEW)

# ── OpenCV baseline settings ─────────────────────────────────────────────
BASELINE_DIFF_THRESHOLD = 30   # pixel-level absolute diff threshold

# ── Demo-data settings ───────────────────────────────────────────────────
N_GOOD_TRAIN   = 50
N_GOOD_VAL     = 20
N_PER_DEFECT   = 15          # images per defect type
DEFECT_TYPES   = ["scratch", "dot", "missing_chunk"]

