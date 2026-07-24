"""
Helixo V2 K210 — Model Export Pipeline
Converts YOLOv8-nano best.pt → ONNX → K210 .kmodel

Usage:
    python convert_kmodel.py

Prerequisites:
    pip install ultralytics onnx onnxruntime
    
    For .kmodel conversion, you also need NNCase:
    pip install nncase nncase-kpu
    
    OR download NNCase from: https://github.com/kendryte/nncase/releases
"""

import os
import sys
import subprocess
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────
WEIGHTS_PATH = Path(__file__).parent.parent / "training" / "weights" / "best.pt"
OUTPUT_DIR = Path(__file__).parent / "models"
IMG_SIZE = 224               # Must match training input size
QUANTIZE = True              # INT8 quantization for K210 KPU
# ─────────────────────────────────────────────────────────────────


def step1_export_onnx():
    """Export YOLOv8 best.pt → ONNX format."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    if not WEIGHTS_PATH.exists():
        print(f"❌ Weights not found: {WEIGHTS_PATH}")
        print("   Run training first (python training/train.py)")
        sys.exit(1)
    
    print("=" * 60)
    print("📦 Step 1: Exporting YOLOv8 → ONNX")
    print("=" * 60)
    
    model = YOLO(str(WEIGHTS_PATH))
    onnx_path = model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        simplify=True,       # Simplify ONNX graph
        opset=11,            # ONNX opset version (K210 compatible)
        dynamic=False,       # Static shapes required for K210
    )
    
    # Move to output dir
    onnx_file = Path(onnx_path)
    output_onnx = OUTPUT_DIR / "helixo_v2_k210.onnx"
    onnx_file.rename(output_onnx)
    
    print(f"✅ ONNX exported: {output_onnx}")
    print(f"   Size: {output_onnx.stat().st_size / 1024:.1f} KB")
    return output_onnx


def step2_export_tflite():
    """Export YOLOv8 best.pt → TFLite (INT8 quantized) as intermediate."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("📦 Step 2: Exporting YOLOv8 → TFLite (INT8)")
    print("=" * 60)
    
    model = YOLO(str(WEIGHTS_PATH))
    tflite_path = model.export(
        format="tflite",
        imgsz=IMG_SIZE,
        int8=QUANTIZE,       # INT8 quantization
    )
    
    tflite_file = Path(tflite_path)
    output_tflite = OUTPUT_DIR / "helixo_v2_k210.tflite"
    if tflite_file.exists():
        tflite_file.rename(output_tflite)
        print(f"✅ TFLite exported: {output_tflite}")
        print(f"   Size: {output_tflite.stat().st_size / 1024:.1f} KB")
    
    return output_tflite


def step3_convert_kmodel(onnx_path):
    """
    Convert ONNX → .kmodel using NNCase compiler.
    
    NNCase is Kendryte's official neural network compiler for K210.
    It converts ONNX models to the K210's KPU-native .kmodel format.
    
    If NNCase is not installed as a Python package, it will try the CLI tool.
    """
    print("\n" + "=" * 60)
    print("📦 Step 3: Converting ONNX → K210 .kmodel")
    print("=" * 60)
    
    output_kmodel = OUTPUT_DIR / "helixo_v2_k210.kmodel"
    
    # Try Python API first
    try:
        import nncase
        
        print("   Using NNCase Python API...")
        
        # Read ONNX model
        with open(onnx_path, "rb") as f:
            onnx_data = f.read()
        
        # Configure compiler
        compiler = nncase.Compiler()
        
        import_options = nncase.ImportOptions()
        compile_options = nncase.CompileOptions()
        compile_options.target = "k210"            # Target K210 KPU
        compile_options.input_type = "uint8"        # Input pixel format
        compile_options.input_range = [0, 255]      # Pixel value range
        compile_options.input_shape = [1, 3, IMG_SIZE, IMG_SIZE]  # NCHW
        compile_options.input_layout = "NCHW"
        compile_options.output_layout = "NCHW"
        compile_options.dump_ir = False
        
        if QUANTIZE:
            compile_options.quant_type = "uint8"
            # Use calibration images for better quantization
            calib_dir = Path(__file__).parent.parent / "dataset" / "images"
            calib_images = list(calib_dir.glob("*.jpg"))[:50]  # Use 50 images
            
            if calib_images:
                print(f"   Using {len(calib_images)} calibration images...")
                # NNCase expects calibration data as numpy arrays
                import numpy as np
                from PIL import Image
                
                calib_data = []
                for img_path in calib_images:
                    img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
                    img_array = np.array(img).astype(np.float32)
                    img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW
                    img_array = np.expand_dims(img_array, 0)         # Add batch
                    calib_data.append(img_array)
                
                compiler.set_calibration_data(calib_data)
        
        compiler.compile(compile_options, import_options, onnx_data)
        
        # Write .kmodel
        kmodel_data = compiler.gencode()
        with open(output_kmodel, "wb") as f:
            f.write(kmodel_data)
        
        print(f"✅ .kmodel exported: {output_kmodel}")
        print(f"   Size: {output_kmodel.stat().st_size / 1024:.1f} KB")
        return output_kmodel
        
    except ImportError:
        print("⚠️  NNCase Python package not found.")
        print("")
        print("   To install NNCase:")
        print("     pip install nncase nncase-kpu")
        print("")
        print("   OR use NNCase CLI:")
        print(f"     ncc compile {onnx_path} {output_kmodel} \\")
        print(f"       --target k210 --input-type uint8 \\")
        print(f"       --input-shape 1,3,{IMG_SIZE},{IMG_SIZE}")
        print("")
        print("   OR use the online converter:")
        print("     https://maixhub.com/model_convert")
        print("")
        print("   ✅ ONNX file is ready for manual conversion.")
        return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🔧 HELIXO V2 K210 — Model Export Pipeline")
    print(f"   Source:  {WEIGHTS_PATH}")
    print(f"   Output:  {OUTPUT_DIR}")
    print(f"   Size:    {IMG_SIZE}x{IMG_SIZE}")
    print(f"   Quant:   {'INT8' if QUANTIZE else 'FP32'}")
    print("")
    
    # Step 1: ONNX
    onnx_path = step1_export_onnx()
    
    # Step 2: TFLite (useful backup / testing)
    step2_export_tflite()
    
    # Step 3: K210 .kmodel
    step3_convert_kmodel(onnx_path)
    
    print("\n" + "=" * 60)
    print("🏁 EXPORT COMPLETE")
    print("=" * 60)
    print(f"   Files in: {OUTPUT_DIR}")
    for f in OUTPUT_DIR.iterdir():
        if f.is_file():
            print(f"     → {f.name} ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
