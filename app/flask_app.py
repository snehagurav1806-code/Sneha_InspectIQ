"""
app/flask_app.py – Implant Visual Inspection Station (Flask backend)
Serves a professional medical-grade web UI with real-time anomaly detection.
Run: python app/flask_app.py
"""

import sys, os, io, base64, json, datetime, time, warnings
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
import torch

import config
from scripts.evaluate import find_checkpoint, load_model

app = Flask(__name__)

# ── Load model once at startup ────────────────────────────────────────────
print("Loading PatchCore model...")
_model = None
_ckpt_path = None
try:
    _ckpt_path = find_checkpoint()
    _model = load_model(_ckpt_path)
    print(f"✓ Model loaded from: {_ckpt_path}")
except Exception as e:
    print(f"✗ Model load failed: {e}")

# ── Load thresholds ───────────────────────────────────────────────────────
def load_thresholds():
    if config.THRESHOLDS_JSON.exists():
        with open(config.THRESHOLDS_JSON) as f:
            return json.load(f)
    return {"low": 0.11, "high": 0.47}

# ── Inference helpers ─────────────────────────────────────────────────────
def preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.resize(img_bgr, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std  = np.array([0.229, 0.224, 0.225], np.float32)
    img  = (img - mean) / std
    return torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)

@torch.no_grad()
def run_inference(img_bgr: np.ndarray):
    tensor = preprocess(img_bgr)
    out    = _model.model(tensor)
    raw    = float(out.pred_score.squeeze().item())
    amap   = out.anomaly_map.squeeze().cpu().numpy().astype(np.float32)
    amap   = cv2.resize(amap, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    # Normalise to [0,1] using empirically observed range
    norm   = max(0.0, min(1.0, (raw - 18.0) / 80.0))
    return raw, norm, amap

def merge_boxes(boxes, dist_thresh=10):
    """Merge bounding boxes that are overlapping or close to each other."""
    merged = []
    used = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]:
            continue
        x1, y1, w1, h1 = boxes[i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]:
                    continue
                x2, y2, w2, h2 = boxes[j]
                if not (x1 + w1 + dist_thresh < x2 or x2 + w2 + dist_thresh < x1 or
                        y1 + h1 + dist_thresh < y2 or y2 + h2 + dist_thresh < y1):
                    nx1 = min(x1, x2)
                    ny1 = min(y1, y2)
                    nx2 = max(x1 + w1, x2 + w2)
                    ny2 = max(y1 + h1, y2 + h2)
                    x1, y1, w1, h1 = nx1, ny1, nx2 - nx1, ny2 - ny1
                    used[j] = True
                    changed = True
        merged.append((x1, y1, w1, h1))
    return merged

def make_heatmap_overlay(img_bgr: np.ndarray, amap: np.ndarray, score: float, is_defective: bool = True):
    """Return (overlay_image, anomaly_count).
    Extracts all distinct defect regions (cracks, chips, voids) using morphological
    top-hat peak isolation and merges close contours so every defect gets a bounding box."""
    sz = config.IMAGE_SIZE
    img = cv2.resize(img_bgr, (sz, sz))
    anomaly_count = 0

    if amap.max() > amap.min() + 1e-6:
        norm_map = ((amap - amap.min()) / (amap.max() - amap.min()) * 255).astype(np.uint8)
        hm_col = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.5, hm_col, 0.5, 0)
    else:
        overlay = img.copy()
        norm_map = np.zeros((sz, sz), dtype=np.uint8)

    if is_defective and (amap.max() > amap.min() + 1e-6):
        # Mask strictly within the cap face & rim to eliminate table/drop-shadow false positives
        mask = np.zeros((sz, sz), dtype=np.uint8)
        cv2.circle(mask, (sz // 2, sz // 2), int(sz * 0.44), 255, -1)

        # Top-Hat isolates local defect peaks/cracks from varying cap baseline
        blurred = cv2.GaussianBlur(norm_map, (15, 15), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
        tophat_masked = cv2.bitwise_and(tophat, tophat, mask=mask)

        # Threshold local anomaly peaks (sharp cracks, chips, voids)
        _, bm = cv2.threshold(tophat_masked, 29, 255, cv2.THRESH_BINARY)
        k_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        bm_dil = cv2.dilate(bm, k_merge)

        cnts, _ = cv2.findContours(bm_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_boxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 90]

        # Merge nearby fragments that belong to the same crack
        boxes = merge_boxes(raw_boxes, dist_thresh=12)
        # Sort defects by area descending
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

        colours = [(255, 255, 255), (0, 255, 255), (255, 255, 0), (0, 200, 255), (255, 150, 100)]
        for i, (x, y, w, h) in enumerate(boxes[:5]):
            pad = 5
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(sz - x, w + 2 * pad)
            h = min(sz - y, h + 2 * pad)
            col = colours[i % len(colours)]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), col, 2)

            # Draw clean dark pill badge for label so text is 100% crisp and readable
            label = f"DEFECT-{i+1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.40
            thick = 1
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
            ty = max(th + 5, y - 5)
            cv2.rectangle(overlay, (x, ty - th - 3), (x + tw + 6, ty + baseline), (10, 15, 26), -1)
            cv2.rectangle(overlay, (x, ty - th - 3), (x + tw + 6, ty + baseline), col, 1)
            cv2.putText(overlay, label, (x + 3, ty - 1), font, scale, col, thick, cv2.LINE_AA)

        anomaly_count = len(boxes)

    return overlay, anomaly_count

def img_to_b64(img_bgr: np.ndarray) -> str:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")

def log_inspection(part_id, filename, score, decision, latency, override="None",
                   inspector="Automated", notes=""):
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "Part ID": part_id, "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": filename, "Anomaly Score": round(score, 4),
        "Decision": decision, "Inspector Override": override,
        "Inspector Name": inspector, "Latency (s)": round(latency, 3), "Notes": notes
    }
    import csv
    write_header = not config.INSPECTIONS_CSV.exists()
    with open(config.INSPECTIONS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)
    return row

def read_logs():
    if not config.INSPECTIONS_CSV.exists():
        return []
    import csv
    with open(config.INSPECTIONS_CSV, newline="") as f:
        return list(csv.DictReader(f))

# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    thresh = load_thresholds()
    return render_template("index.html",
                           object_name=config.OBJECT_NAME,
                           model_loaded=_model is not None,
                           low=thresh.get("low", 0.11),
                           high=thresh.get("high", 0.47))

@app.route("/inspect", methods=["POST"])
def inspect():
    if _model is None:
        return jsonify({"error": "Model not loaded. Run python scripts/train.py first."}), 500

    t0 = time.perf_counter()
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400

    # Decode image
    np_arr  = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Cannot decode image"}), 400

    raw_score, norm_score, amap = run_inference(img_bgr)
    latency = time.perf_counter() - t0

    # Decision using current thresholds
    thresh = load_thresholds()
    low  = float(request.form.get("low",  thresh.get("low",  0.11)))
    high = float(request.form.get("high", thresh.get("high", 0.47)))

    if norm_score < low:
        decision = "PASS"
    elif norm_score > high:
        decision = "REJECT"
    else:
        decision = "REVIEW"

    # Build overlaid heatmap image
    is_defective = (decision != "PASS")
    orig_resized = cv2.resize(img_bgr, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    heatmap_img, anomaly_regions = make_heatmap_overlay(img_bgr, amap, norm_score, is_defective)

    return jsonify({
        "score":           round(norm_score, 4),
        "raw_score":       round(raw_score, 2),
        "decision":        decision,
        "latency_ms":      round(latency * 1000, 1),
        "anomaly_regions": anomaly_regions,
        "orig_b64":        img_to_b64(orig_resized),
        "heat_b64":        img_to_b64(heatmap_img),
    })

@app.route("/log", methods=["POST"])
def log():
    data = request.get_json()
    row  = log_inspection(
        part_id=data.get("part_id", "BM-IMP-UNKNOWN"),
        filename=data.get("filename", "upload.jpg"),
        score=float(data.get("score", 0)),
        decision=data.get("decision", "UNKNOWN"),
        latency=float(data.get("latency_ms", 0)) / 1000,
        override=data.get("override", "None"),
        inspector=data.get("inspector", "Automated"),
        notes=data.get("notes", "")
    )
    return jsonify({"status": "logged", "row": row})

@app.route("/logs")
def logs():
    rows = read_logs()
    return jsonify(rows)

@app.route("/export/csv")
def export_csv():
    if not config.INSPECTIONS_CSV.exists():
        return "No log data", 404
    return send_file(str(config.INSPECTIONS_CSV),
                     mimetype="text/csv",
                     as_attachment=True,
                     download_name=f"biorad_inspection_{datetime.date.today()}.csv")

@app.route("/status")
def status():
    thresh = load_thresholds()
    return jsonify({
        "model_loaded": _model is not None,
        "checkpoint": str(_ckpt_path) if _ckpt_path else None,
        "object_name": config.OBJECT_NAME,
        "thresholds": thresh
    })

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  BIORAD MEDISYS – Implant Visual Inspection Station")
    print("  Open in browser: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
