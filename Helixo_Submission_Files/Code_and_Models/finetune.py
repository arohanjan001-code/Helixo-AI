import os, shutil, glob
from pathlib import Path
from ultralytics import YOLO

base_dir = Path("/Users/arohanjan/Desktop/pythonc_language copy 3/helixo_v2_k210")
personal_ds = base_dir / "dataset_personal"
images_dir = personal_ds / "images"
labels_dir = personal_ds / "labels"

if personal_ds.exists():
    shutil.rmtree(personal_ds)
images_dir.mkdir(parents=True)
labels_dir.mkdir(parents=True)

whatsapp_hel = glob.glob(str(base_dir / "dataset/images/hel/WhatsApp*"))
whatsapp_nohel = glob.glob(str(base_dir / "dataset/images/nohel/WhatsApp*"))
all_personal = whatsapp_hel + whatsapp_nohel
print(f"Found {len(all_personal)} personal images.")

count = 0
for img_path in all_personal:
    img_name = os.path.basename(img_path)
    stem = os.path.splitext(img_name)[0]
    lbl_path = base_dir / f"dataset/labels/{stem}.txt"
    if os.path.exists(lbl_path):
        shutil.copy(img_path, images_dir / img_name)
        shutil.copy(lbl_path, labels_dir / f"{stem}.txt")
        count += 1
print(f"Copied {count} matched images+labels.")

yaml_content = f"""path: {personal_ds}
train: images
val: images

names:
  0: helmet
  1: no_helmet
"""
config_path = base_dir / "finetune_config.yaml"
with open(config_path, "w") as f:
    f.write(yaml_content)

print(f"Loading heavy model...")
pt_path = base_dir / "training/weights/helixo_v2_96epoch_backup.pt"
model = YOLO(pt_path)

print("Starting Fine-tuning (15 epochs)...")
model.train(
    data=str(config_path),
    epochs=15,
    imgsz=224,
    batch=16,
    lr0=0.001,
    project="training/runs",
    name="finetune_k210",
    exist_ok=True
)

best_finetuned = base_dir / "training/runs/finetune_k210/weights/best.pt"
print(f"Exporting new ONNX from {best_finetuned}...")
model = YOLO(best_finetuned)
model.export(format="onnx", imgsz=224, opset=12)

onnx_src = base_dir / "training/runs/finetune_k210/weights/best.onnx"
onnx_dst = base_dir / "website_static/model/best.onnx"
shutil.copy(onnx_src, onnx_dst)
print("✅ ALL DONE! Updated ONNX in website_static/")
