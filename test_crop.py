import sys, warnings, shutil
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import cv2, numpy as np, importlib
import app.flask_app as fa
importlib.reload(fa)

art = r'C:/Users/SNEHA/.gemini/antigravity/brain/39d19121-08f9-4b5d-aad0-db5952313916/'

# The uploaded image is a side-by-side screenshot - crop just the left (original) half
nut_full = cv2.imread(art + '.user_uploaded/media_1795207217845_4a2cd22b.jpg')
h, w = nut_full.shape[:2]
# Left half = original acquired surface
nut_only = nut_full[:, :w//2]
cv2.imwrite('nut_only_input.png', nut_only)

raw, norm, amap = fa.run_inference(nut_only)
nm = ((amap-amap.min())/(amap.max()-amap.min()+1e-8)*255).astype(np.uint8)
overlay, cnt = fa.make_heatmap_overlay(nut_only, amap, norm, is_defective=True)
cv2.imwrite('nut_only_result.png', overlay)
shutil.copy('nut_only_result.png', art + 'nut_only_result.png')

obj_pix = nm[nm > 30]
print(f'Nut only: score={norm:.4f} boxes={cnt}')
print(f'  obj_mean={obj_pix.mean():.1f} obj_std={obj_pix.std():.1f}')
print(f'  adaptive_thresh={int(min(240,max(160,obj_pix.mean()+1.8*obj_pix.std())))}')
print(f'  saved: nut_only_result.png')

# Also test the cap images cropped to the left half too
for tag, path in [
    ('cap2', art+'.user_uploaded/media_1795204777927_eb68f495.jpg'),
    ('cap1', art+'.user_uploaded/media_1795203624782_643b1575.jpg'),
]:
    img = cv2.imread(path)
    h2, w2 = img.shape[:2]
    img_crop = img[:, :w2//2]
    raw, norm, amap = fa.run_inference(img_crop)
    overlay, cnt = fa.make_heatmap_overlay(img_crop, amap, norm, is_defective=True)
    out = f'{tag}_crop_result.png'
    cv2.imwrite(out, overlay)
    shutil.copy(out, art + out)
    print(f'{tag}: score={norm:.4f} boxes={cnt} -> {out}')
