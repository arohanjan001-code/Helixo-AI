import cv2, os, random
from pathlib import Path
import matplotlib.pyplot as plt

images_dir = Path("dataset/images")
labels_dir = Path("dataset/labels")
out_path = "/Users/arohanjan/.gemini/antigravity/brain/d283d963-293a-45a0-90b1-0462845e932f/dataset_debug.jpg"

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for row, cls_dir in enumerate(['hel', 'nohel']):
    img_files = list((images_dir / cls_dir).glob("*.jpg"))
    random.seed(42)
    sample = random.sample(img_files, 5)
    
    for col, img_path in enumerate(sample):
        img = cv2.imread(str(img_path))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c, cx, cy, bw, bh = [float(x) for x in parts[:5]]
                        x1 = int((cx - bw/2) * w)
                        y1 = int((cy - bh/2) * h)
                        x2 = int((cx + bw/2) * w)
                        y2 = int((cy + bh/2) * h)
                        color = (0, 255, 0) if c == 0 else (255, 0, 0)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(img, str(int(c)), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        ax = axes[row, col]
        ax.imshow(img)
        ax.set_title(f"{cls_dir}/{img_path.name[:10]}")
        ax.axis('off')

plt.tight_layout()
plt.savefig(out_path)
print(f"Saved: {out_path}")
