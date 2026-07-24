"""
Helixo V2 K210 — Model Testing Website
Flask backend that loads YOLOv8-nano best.pt and runs inference.

Usage:
    cd helixo_v2_k210/website
    pip install flask
    python app.py

Then open http://localhost:5050 in browser.
"""

import io
import os
import base64
import json
from pathlib import Path

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Load model once at startup
MODEL_PATH = Path(__file__).parent.parent / "training" / "weights" / "best.pt"
model = None


def load_model():
    global model
    from ultralytics import YOLO
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print(f"✅ Model loaded: {MODEL_PATH}")
    print(f"   Classes: {model.names}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept image upload, run YOLOv8 inference, return results.
    Returns JSON with:
      - annotated_image: base64 encoded image with bounding boxes drawn
      - detections: list of {class, confidence, bbox}
      - summary: overall result (helmet/no_helmet/no_detection)
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Read image bytes
    img_bytes = file.read()

    # Run inference
    from PIL import Image
    import numpy as np

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    results = model(img, conf=0.25, iou=0.45, imgsz=224)

    # Parse detections
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class": model.names[cls_id],
                "class_id": cls_id,
                "confidence": round(conf * 100, 1),
                "bbox": {
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                }
            })

    # Generate annotated image
    annotated = results[0].plot()  # numpy array with boxes drawn
    annotated_pil = Image.fromarray(annotated)

    # Convert to base64
    buffer = io.BytesIO()
    annotated_pil.save(buffer, format="JPEG", quality=90)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Determine overall summary
    if len(detections) == 0:
        summary = "no_detection"
        summary_text = "⚠️ No Detection — Camera blocked or no person in frame"
        summary_color = "yellow"
    elif any(d["class"] == "helmet" for d in detections):
        if any(d["class"] == "no_helmet" for d in detections):
            summary = "mixed"
            summary_text = "⚠️ Mixed — Both helmet and no-helmet detected"
            summary_color = "orange"
        else:
            summary = "helmet"
            summary_text = "✅ Helmet Detected — Ignition ENABLED"
            summary_color = "green"
    else:
        summary = "no_helmet"
        summary_text = "❌ No Helmet — Ignition BLOCKED"
        summary_color = "red"

    return jsonify({
        "annotated_image": img_base64,
        "detections": detections,
        "detection_count": len(detections),
        "summary": summary,
        "summary_text": summary_text,
        "summary_color": summary_color,
    })


if __name__ == "__main__":
    load_model()
    print("\n🌐 Helixo V2 K210 — Test Website")
    print("   Open: http://localhost:5050")
    print("   Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
