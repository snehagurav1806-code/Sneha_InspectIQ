# 📋 Hackathon Demo Day Checklist
### Biorad Medisys Implant Visual Inspection Station

---

## 🛠️ Phase 1: 30 Minutes Before Presentation

- [ ] **Hardware & Camera Check**:
  - Connect your webcam or phone camera mount directly perpendicular to the inspection stage.
  - Position lighting at 45° angles to minimize specular glare on the component.
- [ ] **Run Venue Recalibration**:
  ```powershell
  python scripts/recalibrate.py --new_good path/to/20_fresh_good_photos/
  ```
- [ ] **Start the Inspection Station**:
  ```powershell
  streamlit run app/app.py
  ```
  - Verify UI loads on `http://localhost:8501`.
  - Check sidebar: Model indicator shows **"✓ PatchCore Model Loaded"**.
- [ ] **Verify Audio & Display**:
  - Maximize browser window (F11) for full-screen industrial station aesthetic.
  - Have sample physical parts ready on the table:
    1. Clean Flawless Part (Stand-in).
    2. Scratch Part.
    3. Pitted / Dotted Part.
    4. Edge Voids / Chipped Part.
    5. Clean Part reserved for the judge to deface!

---

## 🎭 Phase 2: Live 3-Minute Demo Pitch Flow

### Minute 1: The High-Stakes Problem
- *"Judges, Biorad Medisys manufactures life-critical orthopedic implants. If a human inspector misses a 50-micron scratch or porous void on a knee prosthesis, it causes accelerated osteolysis and revision surgery. Conversely, rejecting clean parts wastes expensive medical-grade alloys."*
- *"We cannot use traditional supervised classifiers because in production, you can't foresee every possible defect. Today, you might see a hairline stress crack or tooling gouge that never existed in training data."*

### Minute 2: The Solution (Show Tab 1: Live Inspection)
- Place the **Good Part** under the camera (or select from sample dropdown).
  - Point to **PASS badge** (Green).
  - Point to anomaly score (~0.02) and low latency (~200ms).
- Place the **Defective Part** under the camera.
  - Point to **REJECT badge** (Red).
  - Point out the **spatial heatmap** and red **defect bounding box** locating exactly where the flaw is.
- Demonstrate **Inspector Override**:
  - Type inspector notes and show human-in-the-loop audit sign-off.

### Minute 3: The Killer Judge Challenge & Scientific Validation
- **Invite the Judge**: *"Judges, take this clean part and add ANY defect you want — poke it with a pen, scratch it with a key, or put a smudge on it."*
- Put the judge's surprise defect part under the camera:
  - System immediately triggers **REJECT** and boxes the judge's mark!
- Switch to **Tab 4 (PatchCore vs OpenCV Baseline)**:
  - Show the side-by-side bimodal separation chart. Explain that naïve image subtraction fails due to slight lighting shifts, while deep PatchCore memory banks remain robust.
- Switch to **Tab 3 (Batch Reports & PDF Export)**:
  - Click **Download Official QA Certificate (PDF)** and show the clean medical certification report generated on the spot.

---

## ❓ Tough Judge Questions & Winning Answers

| Question | Winning Response |
| :--- | :--- |
| **"Why PatchCore instead of YOLO or a classifier?"** | *"Supervised models like YOLO only recognize defect classes they were trained on. If a foreign contaminant or new tooling burr appears, YOLO passes it. PatchCore models what 'healthy' looks like; any deviation is instantly flagged as an anomaly."* |
| **"How fast is it on production lines?"** | *"On an un-accelerated CPU it runs in ~200-300 ms per part (~150-200 parts per minute). On an edge GPU or Intel OpenVINO runtime, this drops below 30 ms, matching continuous conveyor speeds."* |
| **"What if the factory lighting changes?"** | *"We built an automated 2-minute recalibration script (`scripts/recalibrate.py`). 20 photos of clean parts under new lighting re-indexes the memory bank without needing data science intervention."* |
| **"What if the part is rotated?"** | *"PatchCore uses invariant deep convolutional features from intermediate layers of WideResNet-50. During data collection, rotating clean parts 360 degrees populates the memory bank with all valid orientations."* |

