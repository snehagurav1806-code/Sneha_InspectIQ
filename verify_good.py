import sys, warnings, shutil, pathlib
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import cv2, importlib
import app.flask_app as fa
importlib.reload(fa)

# Copy result images to artifact dir for viewing
art_dir = r'C:/Users/SNEHA/.gemini/antigravity/brain/39d19121-08f9-4b5d-aad0-db5952313916/'
for f in ['test_nut_scratches.png', 'test_cap_2anomalies.png', 'test_cap_1scratch.png']:
    shutil.copy(f, art_dir + f)
print('Copied to artifacts dir')

# Verify good images still score PASS
good_imgs = list(pathlib.Path('data/good_train').glob('*.png'))[:3]
for p in good_imgs:
    img = cv2.imread(str(p))
    raw, norm, amap = fa.run_inference(img)
    _, cnt = fa.make_heatmap_overlay(img, amap, norm, is_defective=False)
    decision = 'PASS' if norm < 0.3964 else 'REJECT'
    print(f'{p.name}: score={norm:.4f}  decision={decision}  boxes={cnt}')
