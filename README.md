# Helixo AI

Helixo AI is a helmet detection system I built for two-wheelers. The idea is simple — if you don't wear a helmet, the bike won't start. The detection happens on-device (ESP32-S3 / K210) using a custom trained YOLOv8-nano model, so no internet or cloud needed.

This project got selected and received a seed grant of Rs. 16,000 from SSIP Gujarat.

**Live demo (runs in browser, no install needed):**
https://imaginative-kitsune-6e83b9.netlify.app/

---

## What's in this repo

**AI_Models/**
The trained model files. Two formats — .pt for Python/PyTorch and .onnx for the browser demo. Both are the same model, just different formats.

**Code_and_Models/**
All the code — the web frontend, Flask backend, ESP32/K210 firmware in C, training scripts, and export scripts.

**Training_Results/**
Actual graphs generated during training — loss curves, confusion matrix, F1 curve, PR curve, and sample predictions. Training took 15+ hours on my laptop.

**Documentation/**
Project presentation slides.

**Try_Live_Demo/**
A PDF with the live demo link and project overview, in case you prefer that.

---

## Results

Precision: 99.87%
Recall: 97.96%
mAP@50: 97.50%
Training: 115 epochs total on custom dataset of 12,198 images
Inference: under 300ms in browser

---

## How it works

Camera captures frames at 5-15 FPS. The model checks each frame for helmet/no_helmet. Instead of acting on a single frame (which could give false results due to vibration or occlusion), I built a 50-frame sliding window — it aggregates 10 seconds of predictions before deciding. If helmet ratio drops below 70%, a progressive protocol kicks in: first a buzzer warning, then gradual engine slowdown, then full cut-off.

---

## Team

Arohanjan — Lead (AI model + hardware + web)
Heerkumar Patel — Co-developer
Aadesh Tiwari — Co-developer
Mentor: Prof. Ankit Dhimmar, Prof. Nilesh Patel
Dr. S. & S. S. Ghandhy Government Engineering College, Surat
