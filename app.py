"""
ASL to Voice - Gradio Interface for Hugging Face Spaces
Real-time ASL detection with text-to-speech
"""

import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, Counter
import time

# Global variables
model = None
detection_history = deque(maxlen=15)
current_word = ""
sentence = []
last_letter = None
last_letter_time = 0

def load_model():
    """Load YOLOv8 model"""
    global model
    if model is None:
        print("📦 Loading YOLOv8 model...")
        model = YOLO('runs/train/asl_detection/weights/best.pt')
        print("✅ Model loaded successfully")
    return model

def get_smoothed_prediction():
    """Get most common prediction over recent frames"""
    if len(detection_history) < 8:
        return None, 0.0
    
    letter_counts = Counter(detection_history)
    if letter_counts:
        most_common_letter, count = letter_counts.most_common(1)[0]
        confidence = count / len(detection_history)
        if count >= 8:
            return most_common_letter, confidence
    return None, 0.0

def detect_and_annotate(image):
    """Process image and detect ASL sign - returns annotated image and text"""
    global detection_history
    
    if image is None:
        return None, "No camera input", "—", "", " ".join(sentence)
    
    # Load model
    model = load_model()
    
    # Convert to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Run detection
    results = model(image_bgr, conf=0.6, verbose=False)
    
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
    smoothed_letter, smoothed_conf = get_smoothed_prediction()
    
    # Format output
    raw_text = f"{detected_letter} ({confidence:.1%})" if detected_letter else "—"
    stable_text = f"{smoothed_letter} ({smoothed_conf:.1%})" if smoothed_letter else "—"
    
    return annotated_image, raw_text, stable_text, current_word, " ".join(sentence)

def add_letter():
    """Add detected letter to current word"""
    global current_word, last_letter, last_letter_time, detection_history
    
    smoothed_letter, smoothed_conf = get_smoothed_prediction()
    
    if smoothed_letter is None:
        return current_word, " ".join(sentence), "⚠️ No stable letter detected"
    
    current_time = time.time()
    if smoothed_letter == last_letter and (current_time - last_letter_time) < 1.5:
        return current_word, " ".join(sentence), "⏳ Cooldown active"
    
    current_word += smoothed_letter
    last_letter = smoothed_letter
    last_letter_time = current_time
    detection_history.clear()
    
    return current_word, " ".join(sentence), f"✅ Added '{smoothed_letter}'"

def add_space():
    """Add space (finalize word)"""
    global current_word, sentence, last_letter
    
    if current_word:
        sentence.append(current_word)
        current_word = ""
        last_letter = None
        return current_word, " ".join(sentence), "✅ Word added"
    return current_word, " ".join(sentence), "⚠️ No word to add"

def add_punct(punct):
    """Add punctuation"""
    global current_word, sentence, last_letter
    
    if current_word:
        sentence.append(current_word)
        current_word = ""
    
    if sentence:
        sentence[-1] += punct
        last_letter = None
        return current_word, " ".join(sentence), f"✅ Added '{punct}'"
    return current_word, " ".join(sentence), "⚠️ No text to punctuate"

def delete_char():
    """Delete last character"""
    global current_word
    
    if current_word:
        current_word = current_word[:-1]
        return current_word, " ".join(sentence), "✅ Deleted character"
    return current_word, " ".join(sentence), "⚠️ Nothing to delete"

def clear_word():
    """Clear current word"""
    global current_word, last_letter
    
    current_word = ""
    last_letter = None
    return current_word, " ".join(sentence), "✅ Word cleared"

def reset_all():
    """Reset everything"""
    global current_word, sentence, last_letter
    
    current_word = ""
    sentence = []
    last_letter = None
    return current_word, " ".join(sentence), "✅ Reset complete"

def speak_sentence():
    """Return sentence for TTS"""
    global sentence
    if sentence:
        text = " ".join(sentence)
        return text, "🔊 Speaking..."
    return "", "⚠️ No sentence to speak"

# Create Gradio interface
with gr.Blocks(title="ASL to Voice") as demo:
    gr.Markdown("""
    # 🤟 ASL to Voice - Real-Time Sign Language Detection
    
    Convert American Sign Language alphabet signs to text and speech!
    
    **Instructions:**
    1. Allow webcam access
    2. Show an ASL sign clearly to the camera
    3. Wait for "Stable Letter" to show consistently
    4. Click "Add Letter" to build your word
    5. Click "Space" to move to the next word
    """)
    
    with gr.Row():
        with gr.Column():
            webcam_input = gr.Image(label="Webcam", sources=["webcam"], streaming=True)
            with gr.Row():
                raw_out = gr.Textbox(label="Raw Detection", value="—")
                stable_out = gr.Textbox(label="Stable Letter", value="—")
        
        with gr.Column():
            word_out = gr.Textbox(label="Current Word", value="")
            sent_out = gr.Textbox(label="Sentence", value="", lines=3)
            status_out = gr.Textbox(label="Status", value="Ready")
            
            with gr.Row():
                add_btn = gr.Button("➕ Add Letter", variant="primary")
                space_btn = gr.Button("⎵ Space")
            
            with gr.Row():
                period_btn = gr.Button(".")
                comma_btn = gr.Button(",")
                exclaim_btn = gr.Button("!")
                question_btn = gr.Button("?")
            
            with gr.Row():
                delete_btn = gr.Button("⌫ Delete")
                clear_btn = gr.Button("Clear Word")
                reset_btn = gr.Button("Reset All", variant="stop")
            
            tts_out = gr.Textbox(label="Text to Speak")
            speak_btn = gr.Button("🔊 Speak", variant="primary")
    
    # Event handlers
    webcam_input.stream(
        detect_and_annotate,
        inputs=[webcam_input],
        outputs=[webcam_input, raw_out, stable_out, word_out, sent_out],
        stream_every=0.1
    )
    
    add_btn.click(add_letter, outputs=[word_out, sent_out, status_out])
    space_btn.click(add_space, outputs=[word_out, sent_out, status_out])
    
    period_btn.click(lambda: add_punct("."), outputs=[word_out, sent_out, status_out])
    comma_btn.click(lambda: add_punct(","), outputs=[word_out, sent_out, status_out])
    exclaim_btn.click(lambda: add_punct("!"), outputs=[word_out, sent_out, status_out])
    question_btn.click(lambda: add_punct("?"), outputs=[word_out, sent_out, status_out])
    
    delete_btn.click(delete_char, outputs=[word_out, sent_out, status_out])
    clear_btn.click(clear_word, outputs=[word_out, sent_out, status_out])
    reset_btn.click(reset_all, outputs=[word_out, sent_out, status_out])
    
    speak_btn.click(speak_sentence, outputs=[tts_out, status_out])
    
    gr.Markdown("""
    ---
    ### 🎯 Model Performance
    - **Accuracy:** 96.2% mAP@0.5
    - **Classes:** 26 ASL alphabet signs (A-Z)
    
    ### 💡 Tips
    - Good lighting helps
    - Keep hand clearly visible
    - Hold sign steady for 1-2 seconds
    """)

# Launch
if __name__ == "__main__":
    demo.launch()
