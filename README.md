# Implant Visual Inspection Station 🔬
### Automated Surface Defect Quality Assurance System
**Organised for Biorad Medisys Hackathon — Srijan 2k26**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg)](https://pytorch.org/)
[![Anomalib](https://img.shields.io/badge/Anomalib-2.7.0-green.svg)](https://github.com/openvinotoolkit/anomalib)
[![Flask](https://img.shields.io/badge/Flask-Web_UI-black.svg)](https://flask.palletsprojects.com/)

---

## 📌 Problem Statement & Clinical Motivation
In orthopedic and arthroplasty manufacturing (e.g., knee implants, femoral/tibial trays, neuro and endoscopic components), surface integrity is safety-critical. A microscopic scratch, burr, crack, wrong dimension, or edge void can cause accelerated wear, tissue irritation, or catastrophic failure inside a patient.

Today, much of this inspection is done manually under magnification by human inspectors checking hundreds of parts per shift. This creates three critical bottlenecks:
- **Slow & Repetitive:** Checking hundreds of reflective parts by eye causes inspector fatigue and slow cycle times.
- **Inconsistent:** Two inspectors frequently disagree about marginal defects, introducing subjective variance.
- **High Financial & Clinical Risk:** A missed defect that reaches an operating theater is a life-threatening incident, while rejecting good parts wastes expensive biocompatible alloys and medical-grade polymers.

### The Solution: Unsupervised One-Class Anomaly Detection
Traditional AI classifiers fail in medical manufacturing because you cannot collect or predict every possible defect that will ever occur. 

Our system uses **PatchCore (State of the Art One-Class Anomaly Detection)** trained **strictly on flawless reference components (`good_train`)**. It builds an internal geometric memory of normal surface micro-patterns. Any physical deviation—whether a scratch, crack, missing chunk, or never-before-seen foreign defect—triggers an elevated anomaly score and pinpoints the spatial location of every defect on the part.

---

## 🔬 How PatchCore Works (Judge Pitch Guide)
When judges ask: *"How does your station detect defects it has never seen before during training?"*

1. **Feature Extraction:** High-resolution optical frames are passed through a pre-trained WideResNet-50 backbone. Dense, multiscale feature vectors are extracted from intermediate layers representing local surface geometry.
2. **Coreset Memory Bank:** A compact dictionary of normal surface texture is built via greedy minimax coreset subsampling, retaining 10% of patch descriptors while maintaining near-100% geometric surface coverage.
3. **Nearest-Neighbour Anomaly Distance:** When an unseen part is inspected, each local patch is compared to the closest normal patch in the memory bank:
   $$\text{Anomaly Score} = \max_{p \in P_{\text{test}}} \min_{m \in M_{\text{normal}}} \|p - m\|_2$$
   - Matches normal memory $\rightarrow$ Low Distance $\rightarrow$ **PASS**
   - Discrepancy (crack, burr, scratch) $\rightarrow$ High Distance $\rightarrow$ **Heatmap Hotspot** $\rightarrow$ **REJECT**
4. **Morphological Top-Hat Spatial Defect Isolation:**
   - Evaluates the spatial anomaly map and isolates localized defect elevation peaks from the base object color.
   - Merges contiguous crack fragments into bounding boxes with high-contrast badge labels (`DEFECT-1`, `DEFECT-2`, etc.), allowing the station to identify **multiple distinct defects on a single component**.
5. **Three-Tier Operational Gating:**
   - 🟢 **PASS** (`Score < Low Threshold`): Flawless component; released directly to packaging.
   - 🟡 **REVIEW** (`Low <= Score <= High`): Ambiguous tolerance; routed to human inspector for manual validation.
   - 🔴 **REJECT** (`Score > High Threshold`): Confirmed defect; quarantined with annotated defect coordinates.

---

## 🏛️ Medical Compliance & Industry Features
- **FDA 21 CFR Part 820 Audit Trail:** Every single optical scan logs timestamp, component serial ID, optical latency (ms), normalized anomaly score, station disposition, inspector ID, human override actions, and clinical sign-off notes into an immutable CSV audit trail.
- **Dual-Surface Visualizer:** Side-by-side comparison of the raw optical surface against the spatial JET heatmap overlay.
- **Live Optical Acquisition:** Supports both direct high-res file upload and continuous live camera feed.
- **One-Click CSV Export:** Production supervisors can download compliance reports for the entire shift.

---

## 🏗️ Project Structure

```
defect-inspector/
├── config.py                 # Central configuration (object name, image size, paths)
├── requirements.txt          # Pinned dependencies (anomalib 2.7.0, PyTorch, Flask, OpenCV)
├── README.md                 # Complete system documentation
├── DEMO_CHECKLIST.md         # Demo day pitch script & judge Q&A playbook
├── .gitignore                # Ignores model binaries (>100MB) & temp artifacts
├── data/
│   ├── good_train/           # Flawless training parts (normal reference only)
│   ├── good_val/             # Flawless validation parts (evaluates false-reject rate)
│   └── defect_test/          # Defective components (cracks, scratches, voids, chips)
├── scripts/
│   ├── make_demo_data.py     # Generates baseline demo data
│   ├── prepare_images.py     # Center-crops and normalizes raw photos to 256x256
│   ├── train.py              # Builds PatchCore memory bank on CPU
│   ├── evaluate.py           # Evaluates test parts, generates heatmaps & scores.csv
│   ├── thresholds.py         # Computes optimal PASS/REVIEW/REJECT boundaries
│   ├── baseline_opencv.py    # Naïve mean-image diff baseline for comparison
│   └── recalibrate.py        # 2-minute rapid re-training for venue lighting shifts
├── app/
│   ├── flask_app.py          # Industrial Flask inspection station backend
│   ├── templates/
│   │   └── index.html        # Medical-grade dark navy & gold inspection interface
│   └── static/               # Static assets & demo test samples
├── logs/
│   └── inspections.csv       # FDA 21 CFR Part 820 compliance audit log
└── results/
    ├── thresholds.json       # Calibrated decision thresholds
    └── score_histogram.png   # Class separation distribution
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install the dependencies:
```powershell
git clone https://github.com/SnehaGurav-create/Srijan_2k26.git
cd Srijan_2k26
pip install -r requirements.txt
```

### 2. Train the PatchCore Model
Train the memory bank strictly on defect-free parts:
```powershell
python scripts/train.py
```
*Trained on CPU in ~90 seconds.*

### 3. Calibrate Thresholds
Compute optimal three-tier decision boundaries:
```powershell
python scripts/evaluate.py
python scripts/thresholds.py
```

### 4. Launch the Inspection Station
Start the industrial inspection dashboard:
```powershell
python app/flask_app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser.

---

## 🎯 Demo Day Pitch & Cheat-Sheet for Judges

| Question | Winning Answer |
| :--- | :--- |
| **"Why not train YOLO or ResNet on defect photos?"** | In manufacturing, implants fail in unpredictable ways. A supervised classifier only catches defects you trained it on. PatchCore is trained strictly on good parts, so it catches *any* deviation—even defects never seen before. |
| **"Why is there a REVIEW tier instead of just PASS / FAIL?"** | Medical device manufacturing operates under zero-defect tolerance. Ambiguous edge cases (scores near the threshold) are flagged for human sign-off rather than risking an automated false pass. |
| **"How fast is inspection?"** | Under 800 ms per part on a standard laptop CPU (no GPU required). Fully deployable at standard conveyor speeds. |
| **"What standard does the audit log follow?"** | Formatted to satisfy FDA 21 CFR Part 820 quality management system requirements, logging inspector ID, automated decision, overrides, and timestamps. |

---

## 👥 Authors & Acknowledgments
- **Developer:** Sneha Gurav ([@SnehaGurav-create](https://github.com/SnehaGurav-create))
- **Event:** Srijan 2k26 Hackathon
- **Problem Sponsor:** Biorad Medisys
