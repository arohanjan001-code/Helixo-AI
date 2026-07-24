from ultralytics import YOLO
import random, os

model = YOLO('training/weights/helixo_v2_96epoch_backup.pt')

print("--- HEL MET TEST ---")
for f in os.listdir('dataset/images/hel')[:3]:
    if f.endswith('.jpg'):
        res = model.predict(f'dataset/images/hel/{f}', verbose=False)[0]
        print(f"{f}: conf={[round(c.item(), 2) for c in res.boxes.conf]}, cls={[model.names[int(c.item())] for c in res.boxes.cls]}")
