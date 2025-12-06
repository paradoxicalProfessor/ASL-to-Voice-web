"""
Quick Start Script for ASL Detection System

This script provides an interactive menu to:
1. Check system requirements
2. Train the model
3. Evaluate the model
4. Export the model
5. Run live inference

Usage:
    python quick_start.py
"""

import sys
import os
from pathlib import Path

def print_banner():
    print("=" * 70)
    print("  🤟 ASL to Voice - YOLOv8 Detection System")
    print("=" * 70)
    print()

def check_requirements():
    """Check if all required packages are installed"""
    print("🔍 Checking system requirements...\n")
    
    missing_packages = []
    required_packages = [
        ('torch', 'PyTorch'),
        ('ultralytics', 'Ultralytics YOLOv8'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('pyttsx3', 'Text-to-Speech'),
        ('yaml', 'PyYAML'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('sklearn', 'Scikit-learn'),
    ]
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            missing_packages.append(name)
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✓ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
        else:
            print("\n⚠️  GPU not available - training will use CPU (slower)")
    except:
        pass
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("\n💡 Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All requirements satisfied!")
        return True

def check_dataset():
    """Check if dataset is present"""
    print("\n🔍 Checking dataset...\n")
    
    data_yaml = Path('data.yaml')
    train_dir = Path('train/images')
    val_dir = Path('valid/images')
    
    if not data_yaml.exists():
        print("❌ data.yaml not found")
        return False
    
    if not train_dir.exists() or not val_dir.exists():
        print("❌ Dataset directories not found")
        return False
    
    train_images = len(list(train_dir.glob('*.jpg'))) + len(list(train_dir.glob('*.png')))
    val_images = len(list(val_dir.glob('*.jpg'))) + len(list(val_dir.glob('*.png')))
    
    print(f"✓ Training images: {train_images}")
    print(f"✓ Validation images: {val_images}")
    
    if train_images == 0 or val_images == 0:
        print("❌ No images found in dataset")
        return False
    
    print("✅ Dataset ready!")
    return True

def train_model():
    """Train YOLOv8 model"""
    print("\n" + "=" * 70)
    print("🚀 Training YOLOv8 Model")
    print("=" * 70)
    
    print("\nSelect model size:")
    print("1. Nano (fastest, smallest)")
    print("2. Small (recommended) ⭐")
    print("3. Medium (best accuracy)")
    
    choice = input("\nChoice (1-3) [2]: ").strip() or '2'
    
    model_sizes = {'1': 'n', '2': 's', '3': 'm'}
    model_size = model_sizes.get(choice, 's')
    
    epochs = input("\nNumber of epochs [150]: ").strip() or '150'
    epochs = int(epochs)
    
    batch_size = input("Batch size [16]: ").strip() or '16'
    batch_size = int(batch_size)
    
    print(f"\n📋 Configuration:")
    print(f"   Model: YOLOv8-{model_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    
    confirm = input("\nStart training? (y/n) [y]: ").strip().lower() or 'y'
    
    if confirm == 'y':
        print("\n🎯 Starting training...\n")
        from train_yolov8 import train_yolov8_asl
        
        try:
            results = train_yolov8_asl(
                model_size=model_size,
                epochs=epochs,
                batch_size=batch_size,
            )
            print("\n✅ Training completed successfully!")
            return True
        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            return False
    else:
        print("Training cancelled")
        return False

def evaluate_model():
    """Evaluate trained model"""
    print("\n" + "=" * 70)
    print("📊 Model Evaluation")
    print("=" * 70)
    
    model_path = input("\nModel path [runs/train/asl_detection/weights/best.pt]: ").strip()
    if not model_path:
        model_path = 'runs/train/asl_detection/weights/best.pt'
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("💡 Train a model first using option 2")
        return False
    
    split = input("Evaluate on (val/test) [val]: ").strip() or 'val'
    
    print(f"\n🔍 Evaluating on {split} set...\n")
    
    from evaluate_model import evaluate_model as eval_fn
    
    try:
        results, eval_results = eval_fn(
            model_path=model_path,
            split=split,
        )
        
        print("\n✅ Evaluation completed!")
        
        if eval_results['target_achieved']:
            print("🎉 Target achieved: mAP@0.5 ≥ 96%!")
        else:
            print(f"⚠️  Current mAP@0.5: {eval_results['overall_metrics']['mAP@0.5']*100:.2f}%")
        
        return True
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        return False

def export_model():
    """Export model for deployment"""
    print("\n" + "=" * 70)
    print("📦 Model Export")
    print("=" * 70)
    
    model_path = input("\nModel path [runs/train/asl_detection/weights/best.pt]: ").strip()
    if not model_path:
        model_path = 'runs/train/asl_detection/weights/best.pt'
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    print("\nSelect export format:")
    print("1. ONNX (desktop/web)")
    print("2. TFLite (mobile)")
    print("3. Both ONNX and TFLite ⭐")
    print("4. All formats (ONNX, TFLite, TorchScript)")
    
    choice = input("\nChoice (1-4) [3]: ").strip() or '3'
    
    formats_map = {
        '1': ['onnx'],
        '2': ['tflite'],
        '3': ['onnx', 'tflite'],
        '4': ['onnx', 'tflite', 'torchscript'],
    }
    
    formats = formats_map.get(choice, ['onnx', 'tflite'])
    
    print(f"\n🔄 Exporting to: {', '.join(formats).upper()}...\n")
    
    from export_model import export_model as export_fn
    
    try:
        export_paths = export_fn(
            model_path=model_path,
            formats=formats,
        )
        
        print("\n✅ Export completed!")
        return True
    except Exception as e:
        print(f"\n❌ Export failed: {str(e)}")
        return False

def run_live_inference():
    """Run live inference with webcam"""
    print("\n" + "=" * 70)
    print("📹 Live Inference")
    print("=" * 70)
    
    model_path = input("\nModel path [runs/train/asl_detection/weights/best.pt]: ").strip()
    if not model_path:
        model_path = 'runs/train/asl_detection/weights/best.pt'
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    camera_index = input("Camera index [0]: ").strip() or '0'
    camera_index = int(camera_index)
    
    print("\n🎥 Starting webcam detection...")
    print("\nControls:")
    print("  SPACE     : Add detected letter")
    print("  ENTER     : Finalize word and speak")
    print("  BACKSPACE : Delete character")
    print("  C         : Clear word")
    print("  R         : Reset sentence")
    print("  S         : Speak sentence")
    print("  Q         : Quit")
    print("\nPress any key to continue...")
    input()
    
    from live_inference import ASLDetector
    
    try:
        detector = ASLDetector(model_path=model_path)
        detector.run_webcam(camera_index=camera_index)
        return True
    except Exception as e:
        print(f"\n❌ Inference failed: {str(e)}")
        return False

def main_menu():
    """Display main menu and handle user input"""
    
    while True:
        print_banner()
        print("Main Menu:")
        print("1. Check System Requirements")
        print("2. Train Model")
        print("3. Evaluate Model")
        print("4. Export Model")
        print("5. Run Live Inference")
        print("6. Quick Setup Guide")
        print("0. Exit")
        print()
        
        choice = input("Select option (0-6): ").strip()
        
        if choice == '0':
            print("\n👋 Goodbye!")
            sys.exit(0)
        
        elif choice == '1':
            check_requirements()
            check_dataset()
        
        elif choice == '2':
            if check_requirements() and check_dataset():
                train_model()
            else:
                print("\n❌ Please fix requirements/dataset issues first")
        
        elif choice == '3':
            if check_requirements():
                evaluate_model()
            else:
                print("\n❌ Please install requirements first")
        
        elif choice == '4':
            if check_requirements():
                export_model()
            else:
                print("\n❌ Please install requirements first")
        
        elif choice == '5':
            if check_requirements():
                run_live_inference()
            else:
                print("\n❌ Please install requirements first")
        
        elif choice == '6':
            print("\n" + "=" * 70)
            print("📚 Quick Setup Guide")
            print("=" * 70)
            print("\n1. Install requirements:")
            print("   pip install -r requirements.txt")
            print("\n2. Install PyTorch with CUDA (for GPU):")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            print("\n3. Run option 1 to check system")
            print("\n4. Run option 2 to train model")
            print("\n5. Run option 5 for live detection")
            print("\n📖 For detailed instructions, see README.md")
        
        else:
            print("Invalid option. Please try again.")
        
        print("\n" + "=" * 70)
        input("\nPress ENTER to continue...")
        print("\n" * 2)

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
        sys.exit(0)
