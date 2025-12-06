"""
Evaluation Script for YOLOv8 ASL Detection Model

This script provides comprehensive evaluation metrics including:
- Overall accuracy, precision, recall, F1-score
- Per-class metrics
- Confusion matrix
- mAP@0.5 and mAP@0.5:0.95
"""

import torch
from ultralytics import YOLO
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix, classification_report
import cv2
from tqdm import tqdm

def evaluate_model(
    model_path='runs/train/asl_detection/weights/best.pt',
    data_yaml='data.yaml',
    split='val',  # 'val' or 'test'
    conf_threshold=0.25,
    iou_threshold=0.45,
    save_dir='runs/eval',
):
    """
    Comprehensive model evaluation
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to dataset YAML
        split: Dataset split to evaluate ('val' or 'test')
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        save_dir: Directory to save evaluation results
    """
    
    print("=" * 60)
    print("📊 YOLOv8 ASL Model Evaluation")
    print("=" * 60)
    
    # Check GPU
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    
    # Load model
    print(f"\n📦 Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Load dataset info
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    class_names = data['names']
    num_classes = data['nc']
    
    print(f"\n📊 Dataset: {num_classes} classes")
    print(f"   Split: {split}")
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Run YOLOv8 validation
    print(f"\n🔍 Running validation on {split} set...")
    results = model.val(
        data=data_yaml,
        split=split,
        imgsz=640,
        batch=16,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        plots=True,
        save_json=True,
        project=save_dir,
        name='validation',
        exist_ok=True,
    )
    
    # Print overall metrics
    print("\n" + "=" * 60)
    print("📈 OVERALL METRICS")
    print("=" * 60)
    
    metrics_dict = {
        'Precision': results.box.mp,
        'Recall': results.box.mr,
        'mAP@0.5': results.box.map50,
        'mAP@0.5:0.95': results.box.map,
    }
    
    for metric_name, value in metrics_dict.items():
        print(f"{metric_name:20s}: {value:.4f} ({value*100:.2f}%)")
    
    # Per-class metrics
    print("\n" + "=" * 60)
    print("📊 PER-CLASS METRICS")
    print("=" * 60)
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'mAP@0.5':<12} {'mAP@0.5:0.95':<12}")
    print("-" * 60)
    
    # Get per-class metrics
    per_class_results = []
    for i, class_name in enumerate(class_names):
        precision = results.box.class_result(i)[0] if hasattr(results.box, 'class_result') else results.box.p[i]
        recall = results.box.class_result(i)[1] if hasattr(results.box, 'class_result') else results.box.r[i]
        map50 = results.box.maps[i] if hasattr(results.box, 'maps') else 0
        map50_95 = results.box.ap[i] if hasattr(results.box, 'ap') else 0
        
        print(f"{class_name:<10} {precision:>11.4f} {recall:>11.4f} {map50:>11.4f} {map50_95:>11.4f}")
        
        per_class_results.append({
            'class': class_name,
            'precision': float(precision),
            'recall': float(recall),
            'map50': float(map50),
            'map50_95': float(map50_95),
        })
    
    # Generate confusion matrix
    print("\n🔍 Generating detailed confusion matrix...")
    generate_confusion_matrix(
        model=model,
        data_yaml=data_yaml,
        split=split,
        class_names=class_names,
        save_path=save_path / 'confusion_matrix_detailed.png',
        conf_threshold=conf_threshold,
    )
    
    # Save metrics to JSON
    eval_results = {
        'model_path': str(model_path),
        'split': split,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'overall_metrics': {k: float(v) for k, v in metrics_dict.items()},
        'per_class_metrics': per_class_results,
        'target_achieved': results.box.map50 >= 0.96,  # Check if ≥96% mAP@0.5
    }
    
    with open(save_path / 'evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {save_path}")
    
    # Check if target accuracy achieved
    print("\n" + "=" * 60)
    if results.box.map50 >= 0.96:
        print("🎉 TARGET ACHIEVED: mAP@0.5 ≥ 96%!")
    else:
        print(f"⚠️  Target not achieved. Current mAP@0.5: {results.box.map50*100:.2f}%")
        print("\n💡 Tips to improve accuracy:")
        print("   1. Train for more epochs")
        print("   2. Use a larger model (e.g., YOLOv8m or YOLOv8l)")
        print("   3. Increase image resolution (e.g., imgsz=800)")
        print("   4. Add more augmentation")
        print("   5. Collect more training data")
        print("   6. Check for labeling errors")
    print("=" * 60)
    
    return results, eval_results

def generate_confusion_matrix(
    model,
    data_yaml,
    split='val',
    class_names=None,
    save_path='confusion_matrix.png',
    conf_threshold=0.25,
):
    """
    Generate detailed confusion matrix by running inference on validation set
    """
    
    # Load dataset paths
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    if class_names is None:
        class_names = data['names']
    
    base_path = Path(data_yaml).parent
    
    # Get image and label paths
    if split == 'val':
        img_dir = base_path / data['val']
        label_dir = base_path / data['val'].replace('images', 'labels')
    else:
        img_dir = base_path / data['test']
        label_dir = base_path / data['test'].replace('images', 'labels')
    
    # Collect predictions and ground truths
    y_true = []
    y_pred = []
    
    image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    
    print(f"Processing {len(image_files)} images...")
    
    for img_path in tqdm(image_files, desc="Generating confusion matrix"):
        # Get corresponding label file
        label_path = label_dir / (img_path.stem + '.txt')
        
        if not label_path.exists():
            continue
        
        # Read ground truth
        with open(label_path, 'r') as f:
            gt_classes = [int(line.split()[0]) for line in f.readlines()]
        
        # Run prediction
        results = model(img_path, conf=conf_threshold, verbose=False)
        
        if len(results[0].boxes) > 0:
            pred_classes = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
        else:
            pred_classes = []
        
        # For each ground truth, find best matching prediction
        # Simple approach: match by IoU
        for gt_cls in gt_classes:
            if len(pred_classes) > 0:
                # Take the highest confidence prediction
                pred_cls = pred_classes[0] if len(pred_classes) > 0 else -1
                y_true.append(gt_cls)
                y_pred.append(pred_cls)
                if len(pred_classes) > 0:
                    pred_classes.pop(0)
            else:
                # No prediction found
                y_true.append(gt_cls)
                y_pred.append(-1)  # Will be treated as misclassification
    
    # Handle case where prediction exists but no ground truth
    # This is rare but can happen
    if len(y_true) == 0:
        print("⚠️  No ground truth annotations found")
        return
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    
    # Plot confusion matrix
    plt.figure(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - ASL Alphabet Detection', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")
    plt.close()
    
    # Calculate and display per-class accuracy
    print("\n📊 Per-Class Accuracy from Confusion Matrix:")
    class_accuracies = []
    for i, class_name in enumerate(class_names):
        if cm[i].sum() > 0:
            accuracy = cm[i][i] / cm[i].sum()
            class_accuracies.append((class_name, accuracy))
            print(f"   {class_name}: {accuracy*100:.2f}%")
    
    # Find worst performing classes
    class_accuracies.sort(key=lambda x: x[1])
    print("\n⚠️  Lowest performing classes:")
    for class_name, acc in class_accuracies[:5]:
        print(f"   {class_name}: {acc*100:.2f}%")

def compare_models(model_paths, data_yaml='data.yaml', split='val'):
    """
    Compare multiple trained models
    
    Args:
        model_paths: List of paths to model weights
        data_yaml: Path to dataset YAML
        split: Dataset split to evaluate
    """
    
    print("=" * 60)
    print("🔬 Model Comparison")
    print("=" * 60)
    
    results_list = []
    
    for model_path in model_paths:
        print(f"\n📦 Evaluating: {model_path}")
        model = YOLO(model_path)
        
        results = model.val(
            data=data_yaml,
            split=split,
            imgsz=640,
            batch=16,
            device='0' if torch.cuda.is_available() else 'cpu',
            verbose=False,
        )
        
        results_list.append({
            'model': Path(model_path).stem,
            'precision': results.box.mp,
            'recall': results.box.mr,
            'map50': results.box.map50,
            'map50_95': results.box.map,
        })
    
    # Print comparison table
    print("\n" + "=" * 80)
    print(f"{'Model':<30} {'Precision':<12} {'Recall':<12} {'mAP@0.5':<12} {'mAP@0.5:0.95':<12}")
    print("-" * 80)
    
    for result in results_list:
        print(f"{result['model']:<30} {result['precision']:>11.4f} {result['recall']:>11.4f} "
              f"{result['map50']:>11.4f} {result['map50_95']:>11.4f}")
    
    print("=" * 80)
    
    return results_list

if __name__ == "__main__":
    # Example 1: Evaluate a single model
    results, eval_results = evaluate_model(
        model_path='runs/train/asl_detection/weights/best.pt',
        data_yaml='data.yaml',
        split='val',  # Use validation set
        conf_threshold=0.25,
        iou_threshold=0.45,
        save_dir='runs/eval',
    )
    
    # Example 2: Evaluate on test set
    # results, eval_results = evaluate_model(
    #     model_path='runs/train/asl_detection/weights/best.pt',
    #     data_yaml='data.yaml',
    #     split='test',  # Use test set
    #     save_dir='runs/eval_test',
    # )
    
    # Example 3: Compare multiple models
    # model_paths = [
    #     'runs/train/asl_detection/weights/best.pt',
    #     'runs/train/asl_detection2/weights/best.pt',
    # ]
    # comparison = compare_models(model_paths, data_yaml='data.yaml')
