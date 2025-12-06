"""
Real-Time ASL Detection with Text-to-Speech

This script provides:
1. Real-time detection from webcam or phone camera
2. Temporal filtering/smoothing for stable letter detection
3. Word and sentence assembly
4. Text-to-Speech (TTS) conversion
5. User-friendly GUI overlay

Controls:
- Press SPACE to add detected letter to word
- Press ENTER to finalize word and speak sentence
- Press BACKSPACE to delete last character
- Press 'c' to clear current word
- Press 'r' to reset entire sentence
- Press 's' to speak current sentence
- Press 'q' to quit
"""

import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import time
from collections import deque, Counter
import threading

class ASLDetector:
    def __init__(
        self,
        model_path='runs/train/asl_detection/weights/best.pt',
        conf_threshold=0.6,
        smoothing_window=15,  # Number of frames for temporal smoothing
        min_detection_frames=8,  # Minimum consecutive frames to register letter
    ):
        """
        Initialize ASL Detector with temporal smoothing
        
        Args:
            model_path: Path to trained YOLOv8 model
            conf_threshold: Confidence threshold for detections
            smoothing_window: Window size for temporal smoothing
            min_detection_frames: Minimum frames needed to register a letter
        """
        
        print("🚀 Initializing ASL to Voice System...")
        
        # Load model
        print(f"📦 Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Temporal smoothing
        self.smoothing_window = smoothing_window
        self.min_detection_frames = min_detection_frames
        self.detection_history = deque(maxlen=smoothing_window)
        
        # Text assembly
        self.current_word = ""
        self.sentence = []
        self.last_letter = None
        self.last_letter_time = 0
        self.letter_cooldown = 1.5  # Seconds before same letter can be added again
        
        # TTS settings (engine created fresh for each speech)
        print("🔊 Text-to-Speech ready...")
        self.tts_rate = 150  # Speed of speech
        self.tts_volume = 0.9  # Volume (0-1)
        
        # Threading for TTS (non-blocking)
        self.tts_thread = None
        self.is_speaking = False
        
        # Statistics
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        print("✅ System ready!")
    
    def get_smoothed_prediction(self):
        """
        Get most common prediction over recent frames (temporal smoothing)
        Returns the letter that appears most frequently in recent detections
        """
        if len(self.detection_history) < self.min_detection_frames:
            return None, 0.0
        
        # Count occurrences of each letter
        letter_counts = Counter(self.detection_history)
        
        # Get most common letter and its frequency
        if letter_counts:
            most_common_letter, count = letter_counts.most_common(1)[0]
            confidence = count / len(self.detection_history)
            
            # Only return if it appears in enough frames
            if count >= self.min_detection_frames:
                return most_common_letter, confidence
        
        return None, 0.0
    
    def add_letter_to_word(self, letter):
        """
        Add detected letter to current word with cooldown
        """
        current_time = time.time()
        
        # Check cooldown period
        if letter == self.last_letter and (current_time - self.last_letter_time) < self.letter_cooldown:
            return False
        
        self.current_word += letter
        self.last_letter = letter
        self.last_letter_time = current_time
        
        print(f"✏️  Added '{letter}' → Current word: {self.current_word}")
        return True
    
    def add_space(self):
        """
        Add space to current word (finalize current word and prepare for next)
        """
        if self.current_word:
            self.sentence.append(self.current_word)
            print(f"📝 Word added: {self.current_word}")
            self.current_word = ""
            self.last_letter = None
            print(f"␣ Space added → Current sentence: {' '.join(self.sentence)}")
            return True
        return False
    
    def add_period(self):
        """
        Add period to end sentence
        """
        if self.current_word:
            self.sentence.append(self.current_word)
            self.current_word = ""
        if self.sentence:
            # Add period to last word
            self.sentence[-1] += "."
            print(f"⏺️ Period added → Sentence: {' '.join(self.sentence)}")
            self.last_letter = None
            return True
        return False
    
    def add_comma(self):
        """
        Add comma to current word
        """
        if self.current_word:
            self.sentence.append(self.current_word)
            self.current_word = ""
        if self.sentence:
            # Add comma to last word
            self.sentence[-1] += ","
            print(f"📍 Comma added → Sentence: {' '.join(self.sentence)}")
            self.last_letter = None
            return True
        return False
    
    def finalize_word(self):
        """
        Add current word to sentence
        """
        if self.current_word:
            self.sentence.append(self.current_word)
            print(f"📝 Word added to sentence: {self.current_word}")
            print(f"📄 Current sentence: {' '.join(self.sentence)}")
            self.current_word = ""
            self.last_letter = None
            return True
        return False
    
    def speak_sentence(self):
        """
        Convert sentence to speech (non-blocking)
        """
        if not self.sentence:
            print("⚠️  No sentence to speak")
            return
        
        text = ' '.join(self.sentence)
        print(f"🔊 Speaking: {text}")
        
        # Use threading to avoid blocking
        if self.tts_thread and self.tts_thread.is_alive():
            print("⚠️  Already speaking...")
            return
        
        def speak():
            self.is_speaking = True
            try:
                # Create fresh engine instance (required for Windows pyttsx3)
                engine = pyttsx3.init()
                engine.setProperty('rate', self.tts_rate)
                engine.setProperty('volume', self.tts_volume)
                
                # Get and set voice
                voices = engine.getProperty('voices')
                if voices:
                    engine.setProperty('voice', voices[0].id)
                
                engine.say(text)
                engine.runAndWait()
                
                # Cleanup
                try:
                    engine.stop()
                except:
                    pass
            except Exception as e:
                print(f"❌ TTS Error: {e}")
            finally:
                self.is_speaking = False
        
        self.tts_thread = threading.Thread(target=speak)
        self.tts_thread.start()
    
    def delete_last_character(self):
        """
        Delete last character from current word
        """
        if self.current_word:
            self.current_word = self.current_word[:-1]
            print(f"⌫ Deleted character → Current word: {self.current_word}")
            return True
        return False
    
    def clear_word(self):
        """
        Clear current word
        """
        self.current_word = ""
        self.last_letter = None
        print("🗑️  Cleared current word")
    
    def reset_sentence(self):
        """
        Reset entire sentence
        """
        self.current_word = ""
        self.sentence = []
        self.last_letter = None
        print("🔄 Reset sentence")
    
    def draw_ui(self, frame, detected_letter, confidence, smoothed_letter, smoothed_conf):
        """
        Draw user interface overlay on frame
        """
        height, width = frame.shape[:2]
        
        # Semi-transparent overlay for text background
        overlay = frame.copy()
        
        # Top panel - Detection info
        cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Bottom panel - Word/Sentence
        cv2.rectangle(overlay, (0, height-150), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Title
        cv2.putText(frame, "ASL to Voice - Real-Time Detection", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Current detection (raw)
        if detected_letter:
            text = f"Detected: {detected_letter} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        else:
            cv2.putText(frame, "Detected: --", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        
        # Smoothed detection
        if smoothed_letter:
            text = f"Stable: {smoothed_letter} ({smoothed_conf:.2f})"
            color = (50, 255, 50) if smoothed_conf > 0.6 else (100, 200, 255)
            cv2.putText(frame, text, (10, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Large letter display
            cv2.putText(frame, smoothed_letter, (width-150, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 3.0, (50, 255, 50), 4)
        else:
            cv2.putText(frame, "Stable: --", (10, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (width-150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Current word
        word_text = f"Word: {self.current_word}_" if self.current_word else "Word: (empty)"
        cv2.putText(frame, word_text, (10, height-110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 100), 2)
        
        # Sentence
        sentence_text = ' '.join(self.sentence) if self.sentence else "(no sentence)"
        # Truncate if too long
        if len(sentence_text) > 60:
            sentence_text = sentence_text[-60:]
        cv2.putText(frame, f"Sentence: {sentence_text}", (10, height-70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 255, 150), 2)
        
        # Controls (line 1)
        controls1 = "SPACE:Add Letter | ENTER:Space | .:Period | ,:Comma | BKSP:Delete"
        cv2.putText(frame, controls1, (10, height-40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Controls (line 2)
        controls2 = "C:Clear Word | R:Reset | S:Speak | Q:Quit"
        cv2.putText(frame, controls2, (10, height-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Speaking indicator
        if self.is_speaking:
            cv2.putText(frame, "🔊 SPEAKING...", (width-200, height-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        
        return frame
    
    def run_webcam(self, camera_index=0, display_size=(1280, 720)):
        """
        Run real-time detection from webcam
        
        Args:
            camera_index: Camera device index (0 for default webcam)
            display_size: Window display size (width, height)
        """
        
        print("\n" + "=" * 60)
        print("📹 Starting Webcam Detection")
        print("=" * 60)
        print("\nControls:")
        print("  SPACE     : Add detected letter to word")
        print("  ENTER     : Add space (finalize word)")
        print("  .         : Add period (end sentence)")
        print("  ,         : Add comma")
        print("  BACKSPACE : Delete last character")
        print("  C         : Clear current word")
        print("  R         : Reset entire sentence")
        print("  S         : Speak current sentence")
        print("  Q         : Quit")
        print("=" * 60)
        
        # Open webcam
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_size[1])
        
        print(f"✅ Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        
        # Create window
        window_name = 'ASL to Voice'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_size[0], display_size[1])
        
        frame_times = deque(maxlen=30)
        
        try:
            while True:
                start_time = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Run detection
                results = self.model(frame, conf=self.conf_threshold, verbose=False)
                
                # Extract detections
                detected_letter = None
                confidence = 0.0
                
                if len(results[0].boxes) > 0:
                    # Get highest confidence detection
                    confidences = results[0].boxes.conf.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    
                    best_idx = np.argmax(confidences)
                    detected_letter = self.model.names[int(classes[best_idx])]
                    confidence = confidences[best_idx]
                    
                    # Draw bounding box
                    box = boxes[best_idx].astype(int)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    
                    label = f"{detected_letter}: {confidence:.2f}"
                    cv2.putText(frame, label, (box[0], box[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Add to detection history
                    self.detection_history.append(detected_letter)
                else:
                    # No detection
                    self.detection_history.append(None)
                
                # Get smoothed prediction
                smoothed_letter, smoothed_conf = self.get_smoothed_prediction()
                
                # Draw UI
                frame = self.draw_ui(frame, detected_letter, confidence, 
                                    smoothed_letter, smoothed_conf)
                
                # Display frame
                cv2.imshow(window_name, frame)
                
                # Calculate FPS
                frame_time = time.time() - start_time
                frame_times.append(frame_time)
                self.fps = 1.0 / (sum(frame_times) / len(frame_times))
                self.frame_count += 1
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                elif key == ord(' '):  # Space - add letter
                    if smoothed_letter:
                        self.add_letter_to_word(smoothed_letter)
                        self.detection_history.clear()  # Clear history after adding
                elif key == 13:  # Enter - add space (finalize word)
                    self.add_space()
                elif key == ord('.'):  # Period - end sentence
                    self.add_period()
                elif key == ord(','):  # Comma
                    self.add_comma()
                elif key == 8:  # Backspace - delete character
                    self.delete_last_character()
                elif key == ord('c'):  # Clear word
                    self.clear_word()
                elif key == ord('r'):  # Reset sentence
                    self.reset_sentence()
                elif key == ord('s'):  # Speak sentence
                    self.speak_sentence()
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            if self.tts_thread and self.tts_thread.is_alive():
                self.tts_thread.join(timeout=2)
            
            print("\n📊 Session Statistics:")
            print(f"   Total frames processed: {self.frame_count}")
            print(f"   Average FPS: {self.fps:.1f}")
            if self.sentence:
                print(f"   Final sentence: {' '.join(self.sentence)}")

def run_from_video_file(detector, video_path):
    """
    Run detection on a video file instead of webcam
    
    Args:
        detector: ASLDetector instance
        video_path: Path to video file
    """
    
    print(f"\n📹 Running detection on video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return
    
    # Similar to run_webcam but with video file
    # ... (implementation similar to run_webcam)
    
    cap.release()

def run_from_phone_camera(detector, phone_ip='192.168.1.100', port=8080):
    """
    Run detection from phone camera using IP Webcam app
    
    Args:
        detector: ASLDetector instance
        phone_ip: Phone IP address
        port: Port number (default 8080 for IP Webcam app)
    
    Note:
        Install "IP Webcam" app on Android phone
        Connect phone and computer to same WiFi network
        Start server in app and use the IP address shown
    """
    
    url = f"http://{phone_ip}:{port}/video"
    print(f"\n📱 Connecting to phone camera: {url}")
    print("   Make sure IP Webcam app is running on your phone")
    
    cap = cv2.VideoCapture(url)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not connect to phone camera")
        print(f"   Check that:")
        print(f"   1. Phone and computer are on same WiFi")
        print(f"   2. IP Webcam app is running")
        print(f"   3. IP address is correct: {phone_ip}")
        return
    
    print("✅ Connected to phone camera!")
    
    # Run detection (reuse webcam logic)
    detector.run_webcam(camera_index=url)
    
    cap.release()

if __name__ == "__main__":
    # Initialize detector
    detector = ASLDetector(
        model_path='runs/train/asl_detection/weights/best.pt',
        conf_threshold=0.6,
        smoothing_window=15,
        min_detection_frames=8,
    )
    
    # Run from webcam
    detector.run_webcam(camera_index=0, display_size=(1280, 720))
    
    # Alternative: Run from phone camera
    # Uncomment and update IP address:
    # run_from_phone_camera(
    #     detector,
    #     phone_ip='192.168.1.100',  # Change to your phone's IP
    #     port=8080,
    # )
    
    # Alternative: Run from video file
    # run_from_video_file(detector, 'test_video.mp4')
