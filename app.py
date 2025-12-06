"""
ASL to Voice - Optimized Gradio Interface
Snapshot-based detection for better performance
"""

import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, Counter
import time

# Global variables
model = None
detection_history = deque(maxlen=10)
current_word = ""
sentence = []
last_letter = None
last_letter_time = 0
letter_cooldown = 1.0

def load_model():
    """Load YOLOv8 model"""
    global model
    if model is None:
        print("Loading YOLOv8 model...")
        model = YOLO('runs/train/asl_detection/weights/best.pt')
        print("Model loaded successfully")
    return model

def detect_sign(image):
    """Detect ASL sign from image and return annotated image with detection info"""
    global detection_history
    
    try:
        if image is None:
            return None, "—", "—", current_word, " ".join(sentence)
        
        # Load model
        model = load_model()
        
        # Convert to BGR for OpenCV
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Run detection
        results = model(image_bgr, conf=0.5, verbose=False)
        
        # Extract detections
        detected_letter = None
        confidence = 0.0
        annotated_image = image.copy()
        
        if len(results[0].boxes) > 0:
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            best_idx = np.argmax(confidences)
            detected_letter = model.names[int(classes[best_idx])]
            confidence = float(confidences[best_idx])
            
            # Draw bounding box
            box = boxes[best_idx].astype(int)
            cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
            label = f"{detected_letter}: {confidence:.2f}"
            cv2.putText(annotated_image, label, (box[0], box[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            detection_history.append(detected_letter)
        else:
            detection_history.append(None)
        
        # Get smoothed prediction
        smoothed_letter = None
        smoothed_conf = 0.0
        if len(detection_history) >= 5:
            letter_counts = Counter([l for l in detection_history if l is not None])
            if letter_counts:
                smoothed_letter, count = letter_counts.most_common(1)[0]
                smoothed_conf = count / len([l for l in detection_history if l is not None])
        
        # Format output
        raw_text = f"{detected_letter} ({confidence:.0%})" if detected_letter else "—"
        stable_text = f"{smoothed_letter} ({smoothed_conf:.0%})" if smoothed_letter else "—"
        
        return annotated_image, raw_text, stable_text, current_word, " ".join(sentence)
    
    except Exception as e:
        print(f"Error in detect_sign: {str(e)}")
        return image if image is not None else None, f"Error: {str(e)}", "—", current_word, " ".join(sentence)

def add_letter():
    """Add stable detected letter to current word"""
    global current_word, last_letter, last_letter_time, detection_history
    
    if len(detection_history) < 5:
        return current_word, " ".join(sentence), "Need more detections - take more snapshots!"
    
    letter_counts = Counter([l for l in detection_history if l is not None])
    if not letter_counts:
        return current_word, " ".join(sentence), "No stable letter detected"
    
    smoothed_letter, count = letter_counts.most_common(1)[0]
    
    current_time = time.time()
    if smoothed_letter == last_letter and (current_time - last_letter_time) < letter_cooldown:
        return current_word, " ".join(sentence), "Cooldown active"
    
    current_word += smoothed_letter
    last_letter = smoothed_letter
    last_letter_time = current_time
    detection_history.clear()
    
    return current_word, " ".join(sentence), f"Added '{smoothed_letter}'"

def add_space():
    """Add space (finalize word)"""
    global current_word, sentence, last_letter
    
    if current_word:
        sentence.append(current_word)
        current_word = ""
        last_letter = None
        detection_history.clear()
        return current_word, " ".join(sentence), "Word added"
    return current_word, " ".join(sentence), "No word to add"

def add_punct(punct):
    """Add punctuation"""
    global current_word, sentence, last_letter
    
    if current_word:
        sentence.append(current_word)
        current_word = ""
    
    if sentence:
        sentence[-1] += punct
        last_letter = None
        return current_word, " ".join(sentence), f"Added '{punct}'"
    return current_word, " ".join(sentence), "No text to punctuate"

def delete_char():
    """Delete last character"""
    global current_word
    
    if current_word:
        current_word = current_word[:-1]
        return current_word, " ".join(sentence), "Deleted character"
    return current_word, " ".join(sentence), "Nothing to delete"

def clear_word():
    """Clear current word"""
    global current_word, last_letter, detection_history
    
    current_word = ""
    last_letter = None
    detection_history.clear()
    return current_word, " ".join(sentence), "Word cleared"

def reset_all():
    """Reset everything"""
    global current_word, sentence, last_letter, detection_history
    
    current_word = ""
    sentence = []
    last_letter = None
    detection_history.clear()
    return current_word, " ".join(sentence), "Reset complete"

# Create interface
with gr.Blocks(title="ASL to Voice") as demo:
    gr.Markdown("# ASL to Voice - Sign Language Detection")
    gr.Markdown("Take snapshots of ASL signs to build words and sentences!")
    
    with gr.Row():
        with gr.Column():
            webcam_input = gr.Image(label="Webcam", sources=["webcam"], type="numpy")
            annotated_output = gr.Image(label="Detection Result", type="numpy")
            detect_btn = gr.Button("Capture & Detect", variant="primary")
            
            with gr.Row():
                raw_out = gr.Textbox(label="Instant Detection", value="—")
                stable_out = gr.Textbox(label="Stable Letter", value="—")
        
        with gr.Column():
            word_out = gr.Textbox(label="Current Word", value="")
            sent_out = gr.Textbox(label="Sentence", value="", lines=3)
            status_out = gr.Textbox(label="Status", value="Ready")
            
            with gr.Row():
                add_btn = gr.Button("Add Letter", variant="primary")
                space_btn = gr.Button("Space")
            
            with gr.Row():
                period_btn = gr.Button(".")
                comma_btn = gr.Button(",")
                delete_btn = gr.Button("Delete")
            
            with gr.Row():
                clear_btn = gr.Button("Clear Word")
                reset_btn = gr.Button("Reset All", variant="stop")
    
    # Event handlers
    detect_btn.click(detect_sign, inputs=[webcam_input], outputs=[annotated_output, raw_out, stable_out, word_out, sent_out])
    add_btn.click(add_letter, outputs=[word_out, sent_out, status_out])
    space_btn.click(add_space, outputs=[word_out, sent_out, status_out])
    period_btn.click(lambda: add_punct("."), outputs=[word_out, sent_out, status_out])
    comma_btn.click(lambda: add_punct(","), outputs=[word_out, sent_out, status_out])
    delete_btn.click(delete_char, outputs=[word_out, sent_out, status_out])
    clear_btn.click(clear_word, outputs=[word_out, sent_out, status_out])
    reset_btn.click(reset_all, outputs=[word_out, sent_out, status_out])

if __name__ == "__main__":
    demo.launch()
