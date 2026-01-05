"""
ASL Dataset Capture Tool
========================
Capture images from webcam for ASL sign language dataset.

Instructions:
1. Run this script
2. Choose a letter (A-Z) to capture
3. Position your hand and press 'S' to save image
4. Press 'N' to switch to next letter
5. Press 'Q' to quit

Images will be saved to: custom_dataset_new/images/
Ready for YOLOv8 training after labeling!
"""

import cv2
import os
from datetime import datetime
import numpy as np

class ASLDatasetCapture:
    def __init__(self, output_dir="custom_dataset_new"):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.current_letter = 'A'
        self.letter_counts = {}
        
        # Create directories
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Load existing counts
        self._load_existing_counts()
        
        # Camera settings
        self.camera = None
        self.frame_width = 640
        self.frame_height = 480
        
    def _load_existing_counts(self):
        """Count existing images for each letter"""
        if os.path.exists(self.images_dir):
            for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                existing_files = [f for f in os.listdir(self.images_dir) 
                                if f.startswith(f"{letter}_") and f.endswith('.jpg')]
                self.letter_counts[letter] = len(existing_files)
        else:
            self.letter_counts = {letter: 0 for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
    
    def initialize_camera(self):
        """Initialize webcam"""
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not self.camera.isOpened():
            raise Exception("Could not open webcam")
        
        print("✓ Camera initialized")
        print(f"✓ Resolution: {self.frame_width}x{self.frame_height}")
    
    def preprocess_frame(self, frame):
        """Apply preprocessing to match training conditions"""
        # Resize to 416x416 (training size) to preview what model will see
        # Use padding instead of stretching to maintain aspect ratio
        height, width = frame.shape[:2]
        
        # Calculate padding
        if height > width:
            scale = 416 / height
            new_height = 416
            new_width = int(width * scale)
        else:
            scale = 416 / width
            new_width = 416
            new_height = int(height * scale)
        
        # Resize
        resized = cv2.resize(frame, (new_width, new_height))
        
        # Create padded image
        padded = np.zeros((416, 416, 3), dtype=np.uint8)
        y_offset = (416 - new_height) // 2
        x_offset = (416 - new_width) // 2
        padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return padded
    
    def draw_ui(self, frame):
        """Draw UI elements on frame"""
        # Create overlay
        overlay = frame.copy()
        
        # Header background
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "ASL Dataset Capture Tool", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Current letter (BIG)
        cv2.putText(frame, f"Letter: {self.current_letter}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 3)
        
        # Count
        count = self.letter_counts[self.current_letter]
        cv2.putText(frame, f"Captured: {count} images", (10, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Footer with instructions
        footer_y = frame.shape[0] - 60
        cv2.rectangle(overlay, (0, footer_y), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "S=Save | N=Next Letter | P=Previous | Q=Quit", 
                   (10, footer_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Tips
        cv2.putText(frame, "Tips: Good lighting, plain background, hold steady", 
                   (10, footer_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Center crosshair
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        cv2.line(frame, (center_x - 30, center_y), (center_x + 30, center_y), (0, 255, 0), 2)
        cv2.line(frame, (center_x, center_y - 30), (center_x, center_y + 30), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 100, (0, 255, 0), 2)
        
        return frame
    
    def save_image(self, frame):
        """Save captured image"""
        # Get next filename
        count = self.letter_counts[self.current_letter]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.current_letter}_{count+1:04d}_{timestamp}.jpg"
        filepath = os.path.join(self.images_dir, filename)
        
        # Save preprocessed image (416x416 with padding)
        processed = self.preprocess_frame(frame)
        
        # Save with high quality
        cv2.imwrite(filepath, processed, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Update count
        self.letter_counts[self.current_letter] += 1
        
        print(f"✓ Saved: {filename} (Total: {self.letter_counts[self.current_letter]})")
        
        return True
    
    def next_letter(self):
        """Switch to next letter"""
        current_idx = ord(self.current_letter) - ord('A')
        next_idx = (current_idx + 1) % 26
        self.current_letter = chr(ord('A') + next_idx)
        print(f"→ Switched to letter: {self.current_letter}")
    
    def previous_letter(self):
        """Switch to previous letter"""
        current_idx = ord(self.current_letter) - ord('A')
        prev_idx = (current_idx - 1) % 26
        self.current_letter = chr(ord('A') + prev_idx)
        print(f"← Switched to letter: {self.current_letter}")
    
    def run(self):
        """Main capture loop"""
        print("\n" + "="*60)
        print("ASL Dataset Capture Tool Started!")
        print("="*60)
        print(f"Output directory: {os.path.abspath(self.images_dir)}")
        print(f"Starting letter: {self.current_letter}")
        print("\nControls:")
        print("  S - Save current frame")
        print("  N - Next letter")
        print("  P - Previous letter")
        print("  Q - Quit")
        print("="*60 + "\n")
        
        self.initialize_camera()
        
        saved_flash = 0  # For visual feedback
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Draw UI
                display_frame = self.draw_ui(frame)
                
                # Show flash effect when saved
                if saved_flash > 0:
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], display_frame.shape[0]), 
                                (255, 255, 255), -1)
                    cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
                    saved_flash -= 1
                
                # Display
                cv2.imshow('ASL Dataset Capture', display_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\n✓ Quitting...")
                    break
                
                elif key == ord('s') or key == ord('S'):
                    self.save_image(frame)
                    saved_flash = 5  # Flash for 5 frames
                
                elif key == ord('n') or key == ord('N'):
                    self.next_letter()
                
                elif key == ord('p') or key == ord('P'):
                    self.previous_letter()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print("\n" + "="*60)
        print("Capture Session Summary:")
        print("="*60)
        total = 0
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            count = self.letter_counts[letter]
            if count > 0:
                print(f"  {letter}: {count} images")
                total += count
        print(f"\nTotal images captured: {total}")
        print(f"Images saved to: {os.path.abspath(self.images_dir)}")
        print("="*60)
        print("\n✓ Next step: Label images using Roboflow or LabelImg")
        print("  Then train with: python scripts/train_model.py")


if __name__ == "__main__":
    try:
        capture = ASLDatasetCapture()
        capture.run()
    except KeyboardInterrupt:
        print("\n\n✓ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
