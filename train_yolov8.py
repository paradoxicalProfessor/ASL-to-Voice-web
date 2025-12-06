"""
YOLOv8 Training Script for ASL Alphabet Detection
Goal: Achieve ≥96% validation accuracy with GPU acceleration

This script includes:
- Optimized hyperparameters for high accuracy
- Advanced augmentation techniques
- GPU utilization
- Class imbalance handling
- Automatic best model saving
"""

import torch
from ultralytics import YOLO
import yaml
import os
from pathlib import Path

def check_gpu():
    """Check if GPU is available and print device info"""
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("⚠ WARNING: GPU not available. Training will use CPU (much slower)")
        return False

def verify_dataset(data_yaml_path):
    """Verify dataset structure and count samples per class"""
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    print("\n📊 Dataset Verification:")
    print(f"Number of classes: {data['nc']}")
    print(f"Classes: {data['names']}")
    
    # Check if paths exist
    base_path = Path(data_yaml_path).parent
    for split in ['train', 'val', 'test']:
        img_path = base_path / data[split]
        if img_path.exists():
            num_images = len(list(img_path.glob('*.jpg'))) + len(list(img_path.glob('*.png')))
            print(f"✓ {split}: {num_images} images")
        else:
            print(f"✗ {split} path not found: {img_path}")
    
    return data

def train_yolov8_asl(
    data_yaml='data.yaml',
    model_size='n',  # Options: n (nano), s (small), m (medium), l (large), x (xlarge)
    epochs=150,
    imgsz=640,
    batch_size=16,
    device='0',  # '0' for GPU, 'cpu' for CPU
    patience=30,  # Early stopping patience
    project='runs/train',
    name='asl_detection',
):
    """
    Train YOLOv8 model for ASL alphabet detection with optimized hyperparameters
    
    Args:
        data_yaml: Path to dataset YAML file
        model_size: YOLOv8 model size (n/s/m/l/x) - start with 'n' or 's'
        epochs: Maximum training epochs
        imgsz: Input image size
        batch_size: Batch size (reduce if GPU memory issues)
        device: Device to use ('0' for GPU, 'cpu' for CPU)
        patience: Early stopping patience
        project: Project directory
        name: Experiment name
    """
    
    print("=" * 60)
    print("🚀 YOLOv8 ASL Alphabet Detection Training")
    print("=" * 60)
    
    # Check GPU availability
    gpu_available = check_gpu()
    if not gpu_available and device == '0':
        print("Switching to CPU mode...")
        device = 'cpu'
    
    # Verify dataset
    verify_dataset(data_yaml)
    
    # Load YOLOv8 model
    model_name = f'yolov8{model_size}.pt'
    print(f"\n📦 Loading {model_name}...")
    model = YOLO(model_name)
    
    print("\n⚙️ Training Configuration:")
    print(f"  Model: YOLOv8-{model_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Image Size: {imgsz}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Device: {device}")
    print(f"  Early Stopping Patience: {patience}")
    
    # Optimized hyperparameters for high accuracy (≥96%)
    print("\n🎯 Optimized Hyperparameters for High Accuracy:")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        
        # Optimizer settings
        optimizer='AdamW',  # AdamW often performs better than SGD for small datasets
        lr0=0.001,  # Initial learning rate (slightly lower for stability)
        lrf=0.01,  # Final learning rate (as fraction of lr0)
        momentum=0.937,
        weight_decay=0.0005,
        
        # Advanced training settings
        warmup_epochs=5.0,  # Warmup epochs for learning rate
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Data augmentation (crucial for generalization)
        hsv_h=0.015,  # HSV-Hue augmentation (0-1)
        hsv_s=0.7,    # HSV-Saturation augmentation (0-1)
        hsv_v=0.4,    # HSV-Value augmentation (0-1)
        degrees=10.0,  # Rotation (+/- deg)
        translate=0.1,  # Translation (+/- fraction)
        scale=0.5,  # Scaling (+/- gain)
        shear=0.0,  # Shear (+/- deg)
        perspective=0.0,  # Perspective (+/- fraction)
        flipud=0.0,  # Vertical flip probability
        fliplr=0.5,  # Horizontal flip probability
        mosaic=1.0,  # Mosaic augmentation probability
        mixup=0.1,  # Mixup augmentation probability
        copy_paste=0.0,  # Copy-paste augmentation probability
        
        # Regularization
        dropout=0.0,  # Dropout regularization (0-1)
        
        # Loss function weights
        box=7.5,  # Box loss weight
        cls=0.5,  # Class loss weight
        dfl=1.5,  # DFL loss weight
        
        # Class imbalance handling (if needed)
        # The model will automatically handle class weights based on frequency
        
        # Performance settings
        workers=0,  # Number of worker threads for data loading (0 to avoid multiprocessing issues on Windows)
        cache='disk',  # Cache images for faster training (use 'disk' to avoid memory issues)
        
        # Validation and saving
        val=True,  # Validate during training
        patience=patience,  # Early stopping patience
        save=True,  # Save checkpoints
        save_period=-1,  # Save checkpoint every x epochs (-1 to disable)
        
        # Monitoring
        plots=True,  # Generate training plots
        verbose=True,  # Verbose output
        
        # Multi-scale training (improves robustness)
        rect=False,  # Rectangular training (faster but less accurate)
        
        # Additional settings
        exist_ok=True,  # Allow overwriting existing project
        pretrained=True,  # Use pretrained weights
        resume=False,  # Resume training from last checkpoint
        amp=True,  # Automatic Mixed Precision training (faster on modern GPUs)
        fraction=1.0,  # Use fraction of dataset (1.0 = use all data)
        profile=False,  # Profile ONNX and TensorRT speeds during training
    )
    
    print("\n✅ Training Complete!")
    print(f"\n📁 Results saved to: {results.save_dir}")
    print(f"\n🏆 Best model: {results.save_dir}/weights/best.pt")
    print(f"📊 Last model: {results.save_dir}/weights/last.pt")
    
    # Print final metrics
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print("\n📈 Final Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    return results

def train_with_class_weights(data_yaml='data.yaml', **kwargs):
    """
    Advanced training with explicit class imbalance handling
    
    For severe class imbalance, you can compute class weights and modify the loss.
    YOLOv8 handles this internally, but this function demonstrates how to check balance.
    """
    import glob
    from collections import Counter
    
    # Count labels per class
    base_path = Path(data_yaml).parent
    train_labels = base_path / 'train' / 'labels'
    
    class_counts = Counter()
    for label_file in train_labels.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1
    
    print("\n📊 Class Distribution Analysis:")
    total_samples = sum(class_counts.values())
    
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    class_names = data['names']
    
    print(f"Total annotations: {total_samples}")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_samples) * 100
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Check imbalance
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f"\n⚖️ Imbalance Ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 3.0:
        print("⚠️ Significant class imbalance detected!")
        print("   Recommendations:")
        print("   - Consider data augmentation for minority classes")
        print("   - Use focal loss (YOLOv8 uses focal loss by default)")
        print("   - Increase training epochs")
    else:
        print("✓ Dataset appears relatively balanced")
    
    # Proceed with normal training
    return train_yolov8_asl(data_yaml=data_yaml, **kwargs)

if __name__ == "__main__":
    # Training configurations
    
    # Configuration 1: Fast training (testing setup)
    # Uncomment to use:
    # results = train_yolov8_asl(
    #     model_size='n',  # Nano model (fastest)
    #     epochs=50,
    #     batch_size=16,
    #     imgsz=640,
    # )
    
    # Configuration 2: Balanced training (recommended for 96%+ accuracy)
    results = train_yolov8_asl(
        model_size='s',  # Small model (good balance)
        epochs=150,
        batch_size=16,
        imgsz=640,
        patience=30,
    )
    
    # Configuration 3: Maximum accuracy (slower but best performance)
    # Uncomment to use:
    # results = train_yolov8_asl(
    #     model_size='m',  # Medium model
    #     epochs=200,
    #     batch_size=8,  # Smaller batch for stability
    #     imgsz=640,
    #     patience=40,
    # )
    
    # Optional: Analyze class distribution and train with explicit handling
    # results = train_with_class_weights(
    #     model_size='s',
    #     epochs=150,
    #     batch_size=16,
    # )
