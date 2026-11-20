import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import cv2, numpy as np
import app.flask_app as fa

img = cv2.imread('cap_3defects_input.png')
raw, norm, amap = fa.run_inference(img)
sz = 256
img_resized = cv2.resize(img, (sz, sz))
amap_resized = cv2.resize(amap, (sz, sz))
norm_map = ((amap_resized - amap_resized.min()) / (amap_resized.max() - amap_resized.min() + 1e-8) * 255).astype(np.uint8)

# 1. Mask border and corners
mask = np.zeros((sz, sz), dtype=np.uint8)
cv2.circle(mask, (sz//2, sz//2), int(sz * 0.44), 255, -1)
m = cv2.bitwise_and(norm_map, norm_map, mask=mask)

# 2. Find local peaks
k_peak = 15
dil = cv2.dilate(m, np.ones((k_peak, k_peak), np.uint8))
# Peak points: strictly local maxima with anomaly > 150
peak_mask = (m == dil) & (m >= 150)
peak_pts = np.argwhere(peak_mask) # (y, x)

print(f'Raw peak points: {len(peak_pts)}')

# 3. Cluster peak points that are close to each other (dist < 25)
clusters = []
for py, px in peak_pts:
    val = int(m[py, px])
    merged = False
    for c in clusters:
        cy, cx, cval, pts = c
        if (py - cy)**2 + (px - cx)**2 < 25**2:
            pts.append((py, px, val))
            if val > cval:
                c[0], c[1], c[2] = py, px, val
            merged = True
            break
    if not merged:
        clusters.append([py, px, val, [(py, px, val)]])

print(f'Distinct defect clusters: {len(clusters)}')
for i, c in enumerate(clusters):
    print(f'  Cluster {i+1}: peak at (x={c[1]}, y={c[0]}), value={c[2]}')

# 4. For each cluster peak, extract its local bounding box
# We threshold the local neighborhood around the peak
boxes = []
for cy, cx, cval, _ in clusters:
    # Local threshold around this peak
    local_th = max(140, int(cval * 0.78))
    # Threshold binary map
    _, bin_map = cv2.threshold(m, local_th, 255, cv2.THRESH_BINARY)
    # Find contour containing the peak
    cnts, _ = cv2.findContours(bin_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = False
    for cnt in cnts:
        if cv2.pointPolygonTest(cnt, (float(cx), float(cy)), False) >= 0:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # If the contour swallowed the whole cap, shrink to local window around peak
            if bw > sz * 0.50 or bh > sz * 0.50:
                pad = 25
                bx = max(0, cx - pad); by = max(0, cy - pad)
                bw = min(sz - bx, pad * 2); bh = min(sz - by, pad * 2)
            boxes.append((bx, by, bw, bh))
            found = True
            break
    if not found:
        pad = 18
        bx = max(0, cx - pad); by = max(0, cy - pad)
        bw = min(sz - bx, pad * 2); bh = min(sz - by, pad * 2)
        boxes.append((bx, by, bw, bh))

print(f'Generated {len(boxes)} boxes:')
for i, b in enumerate(boxes):
    print(f'  Box {i+1}: {b}')

# Draw overlay
vis = img_resized.copy()
hm = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(vis, 0.5, hm, 0.5, 0)
colours = [(255, 255, 255), (0, 255, 255), (255, 255, 0), (0, 200, 255), (255, 150, 100)]
for i, (bx, by, bw, bh) in enumerate(boxes):
    pad = 4
    x = max(0, bx - pad); y = max(0, by - pad)
    w = min(sz - x, bw + 2*pad); h = min(sz - y, bh + 2*pad)
    col = colours[i % len(colours)]
    cv2.rectangle(overlay, (x, y), (x+w, y+h), col, 2)
    label = f'DEFECT-{i+1}'
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    ty = max(th + 4, y - 4)
    cv2.rectangle(overlay, (x, ty - th - 2), (x + tw + 4, ty + bl), (10, 15, 26), -1)
    cv2.rectangle(overlay, (x, ty - th - 2), (x + tw + 4, ty + bl), col, 1)
    cv2.putText(overlay, label, (x + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

cv2.imwrite('cap_3defects_multi_result.png', overlay)
import shutil
shutil.copy('cap_3defects_multi_result.png', r'C:/Users/SNEHA/.gemini/antigravity/brain/39d19121-08f9-4b5d-aad0-db5952313916/cap_3defects_multi_result.png')
print('Saved cap_3defects_multi_result.png!')
