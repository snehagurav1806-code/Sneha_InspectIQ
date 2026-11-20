"""
thresholds.py – Analyse scores.csv to suggest PASS / REVIEW / REJECT thresholds.

In visual inspection:
  - Score < LOW  → PASS   (confidently normal; near-zero defects slip through)
  - Score > HIGH → REJECT (confidently defective; near-zero good parts rejected)
  - LOW <= Score <= HIGH → REVIEW (ambiguous zone requiring human inspector sign-off)

Plots a histogram of good vs defect scores and stores results in results/thresholds.json.
All metrics are computed strictly from results/scores.csv.
"""

import sys, os, json
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from config import SCORES_CSV, THRESHOLDS_JSON, RESULTS_DIR


def compute_thresholds(good_scores: np.ndarray, defect_scores: np.ndarray) -> tuple[float, float]:
    """
    Compute optimal LOW and HIGH decision boundaries.
    
    LOW (PASS threshold):
      Score below LOW is declared PASS.
      We want near-zero defect escape (False Negative Rate ~ 0).
      So LOW must not exceed the lowest observed defect scores.
      We place LOW just above the 95th percentile of good parts,
      capped well below the minimum defect score.
      
    HIGH (REJECT threshold):
      Score above HIGH is declared REJECT.
      We want near-zero false scrap (False Positive Rate ~ 0).
      So HIGH must be above the highest observed good parts,
      placed near the lower boundary of defects.
    """
    good_p95 = float(np.percentile(good_scores, 95))
    good_max = float(np.max(good_scores))
    defect_min = float(np.min(defect_scores))
    defect_p05 = float(np.percentile(defect_scores, 5))

    if defect_min > good_max:
        # Clear separation gap exists:
        # LOW is slightly above good parts (leaving safe PASS zone)
        # HIGH is near the lower boundary of defect parts (confident REJECT)
        low = round(good_max + 0.15 * (defect_min - good_max), 4)
        high = round(defect_min - 0.15 * (defect_min - good_max), 4)
    else:
        # Overlapping distributions:
        # LOW set at 95% good coverage, HIGH set at 95% defect coverage
        low = round(float(np.percentile(good_scores, 90)), 4)
        high = round(float(np.percentile(defect_scores, 10)), 4)
        if low >= high:
            mid = (good_scores.mean() + defect_scores.mean()) / 2.0
            low = round(mid - 0.05, 4)
            high = round(mid + 0.05, 4)

    return low, high


def print_metrics(df: pd.DataFrame, low: float, high: float) -> dict:
    good = df[df["label"] == 0]
    defect = df[df["label"] == 1]

    # PASS zone: score < low
    # REVIEW zone: low <= score <= high
    # REJECT zone: score > high

    # False reject on good parts: good parts that get REJECT (score > high)
    false_reject_rate = float((good["score"] > high).mean()) if len(good) else 0.0
    # Good parts sent to review
    good_review_rate = float(((good["score"] >= low) & (good["score"] <= high)).mean()) if len(good) else 0.0
    # Good pass rate
    good_pass_rate = float((good["score"] < low).mean()) if len(good) else 0.0

    # Defect detection rate: defects caught (REJECT or REVIEW, i.e. score >= low)
    defect_caught_rate = float((defect["score"] >= low).mean()) if len(defect) else 0.0
    # Defect outright reject rate
    defect_reject_rate = float((defect["score"] > high).mean()) if len(defect) else 0.0
    # Defect escape rate: defects mistakenly passed (score < low)
    defect_escape_rate = float((defect["score"] < low).mean()) if len(defect) else 0.0

    # Overall review percentage across all images
    review_pct = float(((df["score"] >= low) & (df["score"] <= high)).mean())

    print(f"\n{'='*60}")
    print(f"  OPERATIONAL THRESHOLDS")
    print(f"{'='*60}")
    print(f"  PASS zone   : Score < {low:.4f}")
    print(f"  REVIEW zone : {low:.4f} <= Score <= {high:.4f}")
    print(f"  REJECT zone : Score > {high:.4f}")
    print(f"{'-'*60}")
    print(f"  Defect Detection Rate (Score >= LOW)   : {defect_caught_rate*100:.1f}%")
    print(f"  Defect Outright Reject (Score > HIGH)  : {defect_reject_rate*100:.1f}%")
    print(f"  Defect Escape Rate (Missed defects)    : {defect_escape_rate*100:.1f}%")
    print(f"  False Reject Rate on Good (Scrap risk) : {false_reject_rate*100:.1f}%")
    print(f"  Good Parts Passed cleanly (Score < LOW): {good_pass_rate*100:.1f}%")
    print(f"  Total Images in REVIEW zone            : {review_pct*100:.1f}%")

    # Per defect type detection rate
    print(f"\n  Per-Defect Detection Performance:")
    per_type = {}
    for dtype, grp in defect.groupby("defect_type"):
        caught = float((grp["score"] >= low).mean())
        per_type[dtype] = {
            "caught_rate": round(caught, 4),
            "mean_score": round(float(grp["score"].mean()), 4),
            "count": int(len(grp))
        }
        print(f"    {dtype:<18s} (n={len(grp):2d}): caught={caught*100:5.1f}%, mean_score={grp['score'].mean():.4f}")

    return {
        "low": low,
        "high": high,
        "defect_caught_rate": round(defect_caught_rate, 4),
        "defect_reject_rate": round(defect_reject_rate, 4),
        "defect_escape_rate": round(defect_escape_rate, 4),
        "false_reject_rate": round(false_reject_rate, 4),
        "good_pass_rate": round(good_pass_rate, 4),
        "review_pct": round(review_pct, 4),
        "per_defect_stats": per_type,
    }


def plot_histogram(df: pd.DataFrame, low: float, high: float, out_path: Path) -> None:
    good_s = df[df["label"] == 0]["score"]
    defect_s = df[df["label"] == 1]["score"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1.0, 41)
    ax.hist(good_s, bins=bins, alpha=0.7, color="#2ecc71", label=f"Good Parts (n={len(good_s)})", edgecolor="white")
    ax.hist(defect_s, bins=bins, alpha=0.7, color="#e74c3c", label=f"Defect Parts (n={len(defect_s)})", edgecolor="white")
    
    ax.axvline(low, color="#f39c12", linewidth=2.5, linestyle="--", label=f"LOW threshold = {low:.4f} (PASS/REVIEW)")
    ax.axvline(high, color="#c0392b", linewidth=2.5, linestyle="--", label=f"HIGH threshold = {high:.4f} (REVIEW/REJECT)")
    
    # Highlight zones
    ax.axvspan(0, low, alpha=0.08, color="green", label="PASS Zone")
    ax.axvspan(low, high, alpha=0.12, color="orange", label="REVIEW Zone")
    ax.axvspan(high, 1.0, alpha=0.08, color="red", label="REJECT Zone")

    ax.set_xlabel("Normalised PatchCore Anomaly Score [0, 1]", fontsize=11)
    ax.set_ylabel("Number of Parts", fontsize=11)
    ax.set_title("Anomaly Score Distribution & Multi-Tier Thresholds\n"
                 "(Stand-in: Plastic Bottle Cap · Biorad Medisys Demo Pipeline)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"\nHistogram saved → {out_path}")


if __name__ == "__main__":
    if not SCORES_CSV.exists():
        print(f"ERROR: {SCORES_CSV} not found. Run evaluate.py first.")
        sys.exit(1)

    df = pd.read_csv(SCORES_CSV)
    good_s = df[df["label"] == 0]["score"].values
    defect_s = df[df["label"] == 1]["score"].values

    low, high = compute_thresholds(good_s, defect_s)
    metrics = print_metrics(df, low, high)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLDS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nThresholds JSON saved → {THRESHOLDS_JSON}")

    hist_path = RESULTS_DIR / "score_histogram.png"
    plot_histogram(df, low, high, hist_path)
