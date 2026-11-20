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

# Mask out outer margin
margin = int(sz * 0.10)
m = norm_map.copy()
m[:margin, :] = 0; m[-margin:, :] = 0; m[:, :margin] = 0; m[:, -margin:] = 0

# Also mask object using grabcut or color/luminance threshold to keep only the cap
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
# The cap is darker than the bright white background
_, cap_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
cap_mask = cv2.morphologyEx(cap_mask, cv2.MORPH_CLOSE, k)
# Erode slightly so border shadow isn't counted
cap_mask = cv2.erode(cap_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11)))
m = cv2.bitwise_and(m, m, mask=cap_mask)

# Save mask and masked map
cv2.imwrite('test_cap_mask.png', cap_mask)
cv2.imwrite('test_masked_map.png', m)

# Let's inspect local peaks in m:
# Maximum filter to find local maxima
kernel_size = 21
local_max = cv2.dilate(m, np.ones((kernel_size, kernel_size), np.uint8))
# Peaks are where m == local_max and m > threshold
# What is the anomaly distribution inside cap?
cap_vals = m[cap_mask > 0]
print(f'Inside cap: mean={cap_vals.mean():.1f}, std={cap_vals.std():.1f}, p80={np.percentile(cap_vals,80):.1f}, p90={np.percentile(cap_vals,90):.1f}, max={cap_vals.max()}')

# Find distinct peaks
peaks = (m == local_max) & (m >= np.percentile(cap_vals, 85)) & (m > 120)
peak_pts = np.argwhere(peaks) # y, x
print(f'Found {len(peak_pts)} peak points: {peak_pts}')

# Around each peak point, threshold at 80% of that peak value to get the defect contour!
boxes = []
for py, px in peak_pts:
    pval = m[py, px]
    # Local flood or local threshold
    local_thresh = int(pval * 0.82)
    # Check if this peak is already covered by an existing box
    already = False
    for bx, by, bw, bh in boxes:
        if bx <= px <= bx+bw and by <= py <= by+bh:
            already = True
            break
    if already:
        continue
    # Grow region around peak
    mask_peak = (m >= local_thresh).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask_peak, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.pointPolygonTest(c, (float(px), float(py)), False) >= 0:
            bx, by, bw, bh = cv2.boundingRect(c)
            # Filter huge boxes
            if bw < sz * 0.65 and bh < sz * 0.65 and bw * bh >= 25:
                boxes.append((bx, by, bw, bh))
            break

print(f'Extracted {len(boxes)} boxes from peaks:')
for i, b in enumerate(boxes):
    print(f'  Box {i+1}: {b}')

# Draw overlay
vis = img_resized.copy()
hm = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(vis, 0.5, hm, 0.5, 0)
for i, (bx, by, bw, bh) in enumerate(boxes):
    pad = 5
    x = max(0, bx - pad); y = max(0, by - pad)
    w = min(sz - x, bw + 2*pad); h = min(sz - y, bh + 2*pad)
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 255, 255), 2)
    cv2.putText(overlay, f'DEFECT-{i+1}', (x, max(12, y-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

cv2.imwrite('cap_3defects_peak_result.png', overlay)
import shutil
shutil.copy('cap_3defects_peak_result.png', r'C:/Users/SNEHA/.gemini/antigravity/brain/39d19121-08f9-4b5d-aad0-db5952313916/cap_3defects_peak_result.png')
print('Saved cap_3defects_peak_result.png')
