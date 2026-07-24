## helixo_custom_trained.pt and helixo_custom_trained.onnx

Both files are the same model in different formats.

The model was trained on a custom dataset I built specifically for Indian road conditions — 
helmets of different types (full face, half face, visor up/down), bare heads, caps, turbans, 
all kinds of lighting. Training ran for 15 epochs on this custom data after an initial 
base training run. Total training time was around 15+ hours on my MacBook.

Architecture used: YOLOv8-nano (from Ultralytics), chosen because it's small enough to 
run on ESP32-S3 and K210 without needing a GPU.

Classes: helmet, no_helmet

Final numbers on validation set:
- Precision: 99.87%
- Recall: 97.96%  
- mAP@50: 97.50%

To load and test the .pt model in Python:

    from ultralytics import YOLO
    model = YOLO('helixo_custom_trained.pt')
    results = model('your_image.jpg')
    results[0].show()

The .onnx file is what runs in the browser demo:
https://imaginative-kitsune-6e83b9.netlify.app/
