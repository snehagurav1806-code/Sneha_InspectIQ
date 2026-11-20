import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import cv2, numpy as np, importlib
import app.flask_app as fa
importlib.reload(fa)

images = {
    'nut_scratches': r'C:/Users/SNEHA/.gemini/antigravity/brain/39d19121-08f9-4b5d-aad0-db5952313916/.user_uploaded/media_1795207217845_4a2cd22b.jpg',
    'cap_2anomalies': r'C:/Users/SNEHA/.gemini/antigravity/brain/39d19121-08f9-4b5d-aad0-db5952313916/.user_uploaded/media_1795204777927_eb68f495.jpg',
    'cap_1scratch': r'C:/Users/SNEHA/.gemini/antigravity/brain/39d19121-08f9-4b5d-aad0-db5952313916/.user_uploaded/media_1795203624782_643b1575.jpg',
}

for name, path in images.items():
    img = cv2.imread(path)
    if img is None:
        print(f'{name}: CANNOT READ')
        continue
    raw, norm, amap = fa.run_inference(img)
    nm = ((amap-amap.min())/(amap.max()-amap.min()+1e-8)*255).astype(np.uint8)
    # Also print amap stats for diagnostics
    obj_pix = nm[nm > 30]
    if len(obj_pix) > 0:
        obj_mean = obj_pix.mean()
        obj_std  = obj_pix.std()
        adaptive  = int(min(240, max(160, obj_mean + 1.8 * obj_std)))
    else:
        adaptive = int(nm.max() * 0.80)
    overlay, cnt = fa.make_heatmap_overlay(img, amap, norm, is_defective=True)
    out = f'test_{name}.png'
    cv2.imwrite(out, overlay)
    print(f'{name}: score={norm:.4f}  boxes={cnt}  obj_mean={obj_mean:.1f}  obj_std={obj_std:.1f}  adaptive_thresh={adaptive}  saved={out}')
