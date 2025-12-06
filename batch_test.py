"""
Batch Testing Script for ASL Detection

Test the model on multiple images or videos and generate a report
Useful for analyzing model performance on custom test cases
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter

class BatchTester:
    def __init__(self, model_path='runs/train/asl_detection/weights/best.pt', conf_threshold=0.5):
        """
        Initialize batch tester
        
        Args:
            model_path: Path to trained model
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.results = []
        
    def test_images(self, image_dir, save_dir='runs/batch_test', visualize=True):
        """
        Test model on a directory of images
        
        Args:
            image_dir: Directory containing test images
            save_dir: Directory to save results
            visualize: Whether to save annotated images
        """
        
        image_dir = Path(image_dir)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if visualize:
            vis_dir = save_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
        
        # Get all image files
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        print(f"\n🔍 Testing on {len(image_files)} images...")
        
        test_results = []
        detection_counts = Counter()
        
        for img_path in image_files:
            # Run detection
            results = self.model(img_path, conf=self.conf_threshold, verbose=False)
            
            detections = []
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    class_name = self.model.names[int(cls)]
                    detections.append({
                        'class': class_name,
                        'confidence': float(conf),
                        'bbox': box.tolist(),
                    })
                    detection_counts[class_name] += 1
            
            test_results.append({
                'image': str(img_path.name),
                'detections': detections,
                'num_detections': len(detections),
            })
            
            # Visualize
            if visualize and len(detections) > 0:
                img = cv2.imread(str(img_path))
                
                for det in detections:
                    box = det['bbox']
                    label = f"{det['class']}: {det['confidence']:.2f}"
                    
                    # Draw box
                    cv2.rectangle(img, 
                                (int(box[0]), int(box[1])), 
                                (int(box[2]), int(box[3])), 
                                (0, 255, 0), 2)
                    
                    # Draw label
                    cv2.putText(img, label, 
                              (int(box[0]), int(box[1])-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Save annotated image
                save_path = vis_dir / img_path.name
                cv2.imwrite(str(save_path), img)
        
        # Generate report
        self._generate_report(test_results, detection_counts, save_dir)
        
        return test_results
    
    def test_video(self, video_path, save_dir='runs/batch_test', save_video=True):
        """
        Test model on a video file
        
        Args:
            video_path: Path to video file
            save_dir: Directory to save results
            save_video: Whether to save annotated video
        """
        
        video_path = Path(video_path)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎥 Testing on video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        
        # Video writer
        if save_video:
            output_path = save_dir / f'annotated_{video_path.name}'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_results = []
        detection_counts = Counter()
        frame_num = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Run detection
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            detections = []
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    class_name = self.model.names[int(cls)]
                    detections.append({
                        'class': class_name,
                        'confidence': float(conf),
                    })
                    detection_counts[class_name] += 1
                    
                    # Draw on frame
                    cv2.rectangle(frame,
                                (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])),
                                (0, 255, 0), 2)
                    
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(frame, label,
                              (int(box[0]), int(box[1])-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            frame_results.append({
                'frame': frame_num,
                'detections': detections,
            })
            
            # Add frame info
            cv2.putText(frame, f"Frame: {frame_num}/{total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if save_video:
                out.write(frame)
            
            # Progress
            if frame_num % 30 == 0:
                print(f"   Processed {frame_num}/{total_frames} frames", end='\r')
        
        print(f"\n✅ Processed {frame_num} frames")
        
        cap.release()
        if save_video:
            out.release()
            print(f"   Saved annotated video: {output_path}")
        
        # Generate report
        video_report = {
            'video': str(video_path.name),
            'total_frames': frame_num,
            'fps': fps,
            'detections_by_class': dict(detection_counts),
            'avg_detections_per_frame': sum(len(fr['detections']) for fr in frame_results) / frame_num,
        }
        
        # Save report
        report_path = save_dir / 'video_report.json'
        with open(report_path, 'w') as f:
            json.dump(video_report, f, indent=2)
        
        print(f"   Saved report: {report_path}")
        
        # Print summary
        print(f"\n📊 Video Test Summary:")
        print(f"   Total detections: {sum(detection_counts.values())}")
        print(f"   Detections by class:")
        for class_name, count in detection_counts.most_common():
            print(f"      {class_name}: {count}")
        
        return video_report
    
    def _generate_report(self, test_results, detection_counts, save_dir):
        """Generate HTML and JSON report"""
        
        # JSON report
        report = {
            'timestamp': datetime.now().isoformat(),
            'model': str(self.model.ckpt_path),
            'confidence_threshold': self.conf_threshold,
            'total_images': len(test_results),
            'total_detections': sum(detection_counts.values()),
            'images_with_detections': sum(1 for r in test_results if r['num_detections'] > 0),
            'detection_counts': dict(detection_counts),
            'results': test_results,
        }
        
        report_path = save_dir / 'batch_test_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Test complete!")
        print(f"   Total images: {len(test_results)}")
        print(f"   Images with detections: {report['images_with_detections']}")
        print(f"   Total detections: {sum(detection_counts.values())}")
        print(f"\n   Report saved: {report_path}")
        
        # Print detection distribution
        if detection_counts:
            print(f"\n📊 Detection Distribution:")
            for class_name, count in detection_counts.most_common():
                percentage = (count / sum(detection_counts.values())) * 100
                print(f"   {class_name}: {count} ({percentage:.1f}%)")
            
            # Create visualization
            self._plot_distribution(detection_counts, save_dir / 'detection_distribution.png')
    
    def _plot_distribution(self, detection_counts, save_path):
        """Plot detection class distribution"""
        
        classes = list(detection_counts.keys())
        counts = list(detection_counts.values())
        
        plt.figure(figsize=(12, 6))
        plt.bar(classes, counts, color='steelblue')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Detection Class Distribution', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"   Distribution plot saved: {save_path}")

def main():
    """Main function with examples"""
    
    print("=" * 70)
    print("🧪 ASL Detection - Batch Testing")
    print("=" * 70)
    
    # Initialize tester
    tester = BatchTester(
        model_path='runs/train/asl_detection/weights/best.pt',
        conf_threshold=0.5,
    )
    
    # Example 1: Test on directory of images
    # tester.test_images(
    #     image_dir='test/images',
    #     save_dir='runs/batch_test',
    #     visualize=True,
    # )
    
    # Example 2: Test on video file
    # tester.test_video(
    #     video_path='test_video.mp4',
    #     save_dir='runs/batch_test',
    #     save_video=True,
    # )
    
    print("\n💡 Usage:")
    print("   1. Uncomment examples in main() function")
    print("   2. Provide path to test images or video")
    print("   3. Run: python batch_test.py")
    print("\n📖 See batch_test.py for more options")

if __name__ == "__main__":
    main()
