import torch
from ultralytics import YOLO

pt_path = 'training/weights/helixo_v2_96epoch_backup.pt'

# 1. Direct checkpoint patching
ckpt = torch.load(pt_path, weights_only=False)
print("Old names:", ckpt['model'].names)
ckpt['model'].names = {0: 'no_helmet', 1: 'helmet'}
torch.save(ckpt, pt_path)
print("Saved new names into .pt checkpoint.")

# 2. Verify with YOLO and re-export
model = YOLO(pt_path)
print("Verified YOLO names:", model.names)
model.export(format='onnx', imgsz=224, opset=12)
