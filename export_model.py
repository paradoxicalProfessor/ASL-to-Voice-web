"""
Model Export Script for YOLOv8 ASL Detection

Export trained model to various formats for deployment:
- ONNX (cross-platform, CPU/GPU)
- TFLite (mobile - Android/iOS)
- TorchScript (PyTorch deployment)
- CoreML (iOS)
- OpenVINO (Intel hardware acceleration)
"""

from ultralytics import YOLO
import torch
from pathlib import Path

def export_model(
    model_path='runs/train/asl_detection/weights/best.pt',
    formats=['onnx', 'tflite'],  # Can include: onnx, tflite, torchscript, coreml, openvino
    imgsz=640,
    optimize=True,
    simplify=True,  # Simplify ONNX model
    int8=False,  # INT8 quantization for TFLite (smaller but may reduce accuracy)
    half=False,  # FP16 quantization (faster on supported GPUs)
):
    """
    Export YOLOv8 model to multiple formats
    
    Args:
        model_path: Path to trained model weights (.pt file)
        formats: List of export formats
        imgsz: Input image size
        optimize: Optimize exported model
        simplify: Simplify ONNX model (reduces size)
        int8: Use INT8 quantization for TFLite (mobile-friendly)
        half: Use FP16 precision (faster on compatible hardware)
    """
    
    print("=" * 60)
    print("📦 YOLOv8 Model Export")
    print("=" * 60)
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("   Please train a model first using train_yolov8.py")
        return
    
    # Load model
    print(f"\n📥 Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Get model info
    print(f"   Model type: {model.type}")
    print(f"   Task: {model.task}")
    
    export_paths = {}
    
    for format_type in formats:
        print(f"\n🔄 Exporting to {format_type.upper()}...")
        
        try:
            if format_type.lower() == 'onnx':
                # Export to ONNX
                export_path = model.export(
                    format='onnx',
                    imgsz=imgsz,
                    optimize=optimize,
                    simplify=simplify,
                    dynamic=False,  # Fixed input size for better optimization
                    opset=12,  # ONNX opset version
                )
                print(f"   ✓ ONNX export successful: {export_path}")
                print(f"   ℹ️  Use with ONNX Runtime for CPU/GPU inference")
                export_paths['onnx'] = export_path
                
            elif format_type.lower() == 'tflite':
                # Export to TFLite (for mobile deployment)
                export_path = model.export(
                    format='tflite',
                    imgsz=imgsz,
                    int8=int8,  # INT8 quantization
                    half=False,  # TFLite doesn't support FP16
                )
                print(f"   ✓ TFLite export successful: {export_path}")
                print(f"   ℹ️  Use with TensorFlow Lite on Android/iOS")
                if int8:
                    print(f"   ℹ️  INT8 quantization enabled (smaller size, faster inference)")
                export_paths['tflite'] = export_path
                
            elif format_type.lower() == 'torchscript':
                # Export to TorchScript
                export_path = model.export(
                    format='torchscript',
                    imgsz=imgsz,
                    optimize=optimize,
                )
                print(f"   ✓ TorchScript export successful: {export_path}")
                print(f"   ℹ️  Use for PyTorch deployment (C++/Python)")
                export_paths['torchscript'] = export_path
                
            elif format_type.lower() == 'coreml':
                # Export to CoreML (iOS)
                export_path = model.export(
                    format='coreml',
                    imgsz=imgsz,
                    int8=int8,
                    half=half,
                    nms=True,  # Include NMS in CoreML model
                )
                print(f"   ✓ CoreML export successful: {export_path}")
                print(f"   ℹ️  Use with Core ML on iOS/macOS")
                export_paths['coreml'] = export_path
                
            elif format_type.lower() == 'openvino':
                # Export to OpenVINO (Intel hardware)
                export_path = model.export(
                    format='openvino',
                    imgsz=imgsz,
                    half=half,
                )
                print(f"   ✓ OpenVINO export successful: {export_path}")
                print(f"   ℹ️  Use with OpenVINO for Intel hardware acceleration")
                export_paths['openvino'] = export_path
                
            else:
                print(f"   ⚠️  Unknown format: {format_type}")
                
        except Exception as e:
            print(f"   ❌ Export to {format_type} failed: {str(e)}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Export Summary")
    print("=" * 60)
    
    for format_name, path in export_paths.items():
        file_size = Path(path).stat().st_size / (1024 * 1024)  # Size in MB
        print(f"   {format_name.upper():<15}: {path}")
        print(f"   {'Size':<15}: {file_size:.2f} MB")
        print()
    
    return export_paths

def export_for_mobile(model_path='runs/train/asl_detection/weights/best.pt', imgsz=640):
    """
    Optimized export for mobile deployment (Android/iOS)
    Creates both TFLite and CoreML versions
    """
    
    print("=" * 60)
    print("📱 Mobile Deployment Export")
    print("=" * 60)
    
    # Standard TFLite (better accuracy)
    print("\n1️⃣  Exporting TFLite (FP32 - Best Accuracy)")
    export_model(
        model_path=model_path,
        formats=['tflite'],
        imgsz=imgsz,
        int8=False,
        optimize=True,
    )
    
    # Quantized TFLite (smaller size, faster)
    print("\n2️⃣  Exporting TFLite (INT8 - Optimized for Mobile)")
    model = YOLO(model_path)
    quantized_path = model.export(
        format='tflite',
        imgsz=imgsz,
        int8=True,
    )
    print(f"   ✓ Quantized TFLite: {quantized_path}")
    
    # CoreML for iOS
    try:
        print("\n3️⃣  Exporting CoreML (iOS)")
        export_model(
            model_path=model_path,
            formats=['coreml'],
            imgsz=imgsz,
            half=False,
        )
    except Exception as e:
        print(f"   ⚠️  CoreML export requires macOS or coremltools")
        print(f"   Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ Mobile export complete!")
    print("=" * 60)
    print("\n📝 Deployment Notes:")
    print("   Android:")
    print("      - Use TFLite model with TensorFlow Lite")
    print("      - INT8 model recommended for better performance")
    print("      - Add TFLite dependency to build.gradle")
    print("\n   iOS:")
    print("      - Use CoreML model with Core ML framework")
    print("      - Drag model into Xcode project")
    print("      - Use Vision framework for preprocessing")

def export_for_edge_devices(model_path='runs/train/asl_detection/weights/best.pt', imgsz=640):
    """
    Export optimized models for edge devices (Raspberry Pi, Jetson Nano, etc.)
    """
    
    print("=" * 60)
    print("🖥️  Edge Device Export")
    print("=" * 60)
    
    formats = []
    
    # ONNX for general edge devices
    formats.append('onnx')
    
    # TensorRT for Jetson devices
    if torch.cuda.is_available():
        print("\n✓ CUDA available - including TensorRT-friendly ONNX")
    
    # OpenVINO for Intel devices (NUC, Movidius)
    formats.append('openvino')
    
    export_model(
        model_path=model_path,
        formats=formats,
        imgsz=imgsz,
        optimize=True,
        simplify=True,
        half=False,  # Keep FP32 for better compatibility
    )
    
    print("\n" + "=" * 60)
    print("📝 Edge Device Recommendations:")
    print("=" * 60)
    print("   Raspberry Pi 4/5:")
    print("      - Use ONNX with ONNX Runtime")
    print("      - Consider INT8 quantization")
    print("      - Reduce imgsz to 416 for better FPS")
    print("\n   NVIDIA Jetson (Nano/Xavier/Orin):")
    print("      - Use ONNX and convert to TensorRT")
    print("      - Enable FP16 for 2x speedup")
    print("      - Can achieve 30+ FPS at 640x640")
    print("\n   Intel NUC/Movidius:")
    print("      - Use OpenVINO format")
    print("      - Optimized for Intel hardware")

def benchmark_exported_models(model_path='runs/train/asl_detection/weights/best.pt', imgsz=640):
    """
    Benchmark different exported formats for speed comparison
    """
    import time
    import cv2
    import numpy as np
    
    print("=" * 60)
    print("⚡ Model Benchmark")
    print("=" * 60)
    
    # Create dummy image
    dummy_img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    
    results = {}
    
    # Benchmark PyTorch model
    print("\n📊 Benchmarking PyTorch (.pt)...")
    model_pt = YOLO(model_path)
    
    # Warmup
    for _ in range(10):
        _ = model_pt(dummy_img, verbose=False)
    
    # Benchmark
    start = time.time()
    n_runs = 100
    for _ in range(n_runs):
        _ = model_pt(dummy_img, verbose=False)
    end = time.time()
    
    pt_time = (end - start) / n_runs * 1000  # ms per inference
    results['PyTorch'] = pt_time
    print(f"   Average inference time: {pt_time:.2f} ms")
    print(f"   FPS: {1000/pt_time:.1f}")
    
    # Export and benchmark ONNX
    print("\n📊 Benchmarking ONNX...")
    try:
        onnx_path = model_pt.export(format='onnx', imgsz=imgsz, simplify=True)
        
        # Load ONNX model
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        
        # Warmup
        input_name = session.get_inputs()[0].name
        dummy_input = dummy_img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        for _ in range(10):
            _ = session.run(None, {input_name: dummy_input})
        
        # Benchmark
        start = time.time()
        for _ in range(n_runs):
            _ = session.run(None, {input_name: dummy_input})
        end = time.time()
        
        onnx_time = (end - start) / n_runs * 1000
        results['ONNX'] = onnx_time
        print(f"   Average inference time: {onnx_time:.2f} ms")
        print(f"   FPS: {1000/onnx_time:.1f}")
        print(f"   Speedup vs PyTorch: {pt_time/onnx_time:.2f}x")
        
    except Exception as e:
        print(f"   ⚠️  ONNX benchmark failed: {str(e)}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Benchmark Summary")
    print("=" * 60)
    for format_name, inference_time in sorted(results.items(), key=lambda x: x[1]):
        fps = 1000 / inference_time
        print(f"   {format_name:<15}: {inference_time:>7.2f} ms  ({fps:>5.1f} FPS)")
    print("=" * 60)

if __name__ == "__main__":
    # Example 1: Export to ONNX and TFLite (recommended)
    print("🚀 Starting model export...\n")
    
    export_paths = export_model(
        model_path='runs/train/asl_detection/weights/best.pt',
        formats=['onnx', 'tflite'],
        imgsz=640,
        optimize=True,
        simplify=True,
        int8=False,  # Set to True for smaller TFLite model
    )
    
    # Example 2: Export for mobile deployment
    # export_for_mobile(
    #     model_path='runs/train/asl_detection/weights/best.pt',
    #     imgsz=640,
    # )
    
    # Example 3: Export for edge devices
    # export_for_edge_devices(
    #     model_path='runs/train/asl_detection/weights/best.pt',
    #     imgsz=640,
    # )
    
    # Example 4: Export all formats
    # export_paths = export_model(
    #     model_path='runs/train/asl_detection/weights/best.pt',
    #     formats=['onnx', 'tflite', 'torchscript', 'coreml', 'openvino'],
    #     imgsz=640,
    # )
    
    # Example 5: Benchmark exported models
    # benchmark_exported_models(
    #     model_path='runs/train/asl_detection/weights/best.pt',
    #     imgsz=640,
    # )
