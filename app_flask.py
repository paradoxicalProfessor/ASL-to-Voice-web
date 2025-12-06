"""
ASL to Voice - Flask Web Application
Real-time ASL detection with webcam streaming
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, Counter
import time
import base64
import json

app = Flask(__name__)

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

def process_frame(frame):
    """Process a single frame and return detection results"""
    global detection_history
    
    # Load model
    model = load_model()
    
    # Run detection
    results = model(frame, conf=0.5, verbose=False)
    
    # Extract detections
    detected_letter = None
    confidence = 0.0
    
    if len(results[0].boxes) > 0:
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        best_idx = np.argmax(confidences)
        detected_letter = model.names[int(classes[best_idx])]
        confidence = float(confidences[best_idx])
        
        # Draw bounding box
        box = boxes[best_idx].astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
        label = f"{detected_letter}: {confidence:.2f}"
        cv2.putText(frame, label, (box[0], box[1]-10),
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
    
    return frame, detected_letter, confidence, smoothed_letter, smoothed_conf

@app.route('/')
def index():
    """Render main page"""
    return render_template('index_flask.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        cap = cv2.VideoCapture(0)
        
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                
                # Process frame
                processed_frame, _, _, _, _ = process_frame(frame)
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            cap.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detection')
def get_detection():
    """Get current detection status"""
    # Get smoothed prediction
    smoothed_letter = None
    smoothed_conf = 0.0
    
    if len(detection_history) >= 5:
        letter_counts = Counter([l for l in detection_history if l is not None])
        if letter_counts:
            smoothed_letter, count = letter_counts.most_common(1)[0]
            smoothed_conf = count / len([l for l in detection_history if l is not None])
    
    # Get instant detection
    instant_letter = detection_history[-1] if detection_history else None
    
    return jsonify({
        'instant_letter': instant_letter if instant_letter else '—',
        'stable_letter': smoothed_letter if smoothed_letter else '—',
        'stable_confidence': f"{smoothed_conf:.0%}" if smoothed_letter else '—',
        'current_word': current_word,
        'sentence': ' '.join(sentence)
    })

@app.route('/add_letter', methods=['POST'])
def add_letter():
    """Add stable detected letter to current word"""
    global current_word, last_letter, last_letter_time, detection_history
    
    if len(detection_history) < 5:
        return jsonify({
            'success': False,
            'message': 'Need more detections',
            'current_word': current_word,
            'sentence': ' '.join(sentence)
        })
    
    letter_counts = Counter([l for l in detection_history if l is not None])
    if not letter_counts:
        return jsonify({
            'success': False,
            'message': 'No stable letter detected',
            'current_word': current_word,
            'sentence': ' '.join(sentence)
        })
    
    smoothed_letter, count = letter_counts.most_common(1)[0]
    
    current_time = time.time()
    if smoothed_letter == last_letter and (current_time - last_letter_time) < letter_cooldown:
        return jsonify({
            'success': False,
            'message': 'Cooldown active',
            'current_word': current_word,
            'sentence': ' '.join(sentence)
        })
    
    current_word += smoothed_letter
    last_letter = smoothed_letter
    last_letter_time = current_time
    detection_history.clear()
    
    return jsonify({
        'success': True,
        'message': f"Added '{smoothed_letter}'",
        'current_word': current_word,
        'sentence': ' '.join(sentence)
    })

@app.route('/add_space', methods=['POST'])
def add_space():
    """Add space (finalize word)"""
    global current_word, sentence, last_letter, detection_history
    
    if current_word:
        sentence.append(current_word)
        current_word = ""
        last_letter = None
        detection_history.clear()
        return jsonify({
            'success': True,
            'message': 'Word added',
            'current_word': current_word,
            'sentence': ' '.join(sentence)
        })
    
    return jsonify({
        'success': False,
        'message': 'No word to add',
        'current_word': current_word,
        'sentence': ' '.join(sentence)
    })

@app.route('/add_punct', methods=['POST'])
def add_punct():
    """Add punctuation"""
    global current_word, sentence, last_letter
    
    punct = request.json.get('punct', '.')
    
    if current_word:
        sentence.append(current_word)
        current_word = ""
    
    if sentence:
        sentence[-1] += punct
        last_letter = None
        return jsonify({
            'success': True,
            'message': f"Added '{punct}'",
            'current_word': current_word,
            'sentence': ' '.join(sentence)
        })
    
    return jsonify({
        'success': False,
        'message': 'No text to punctuate',
        'current_word': current_word,
        'sentence': ' '.join(sentence)
    })

@app.route('/delete_char', methods=['POST'])
def delete_char():
    """Delete last character"""
    global current_word
    
    if current_word:
        current_word = current_word[:-1]
        return jsonify({
            'success': True,
            'message': 'Deleted character',
            'current_word': current_word,
            'sentence': ' '.join(sentence)
        })
    
    return jsonify({
        'success': False,
        'message': 'Nothing to delete',
        'current_word': current_word,
        'sentence': ' '.join(sentence)
    })

@app.route('/clear_word', methods=['POST'])
def clear_word():
    """Clear current word"""
    global current_word, last_letter, detection_history
    
    current_word = ""
    last_letter = None
    detection_history.clear()
    
    return jsonify({
        'success': True,
        'message': 'Word cleared',
        'current_word': current_word,
        'sentence': ' '.join(sentence)
    })

@app.route('/reset_all', methods=['POST'])
def reset_all():
    """Reset everything"""
    global current_word, sentence, last_letter, detection_history
    
    current_word = ""
    sentence = []
    last_letter = None
    detection_history.clear()
    
    return jsonify({
        'success': True,
        'message': 'Reset complete',
        'current_word': current_word,
        'sentence': ' '.join(sentence)
    })

@app.route('/speak', methods=['POST'])
def speak():
    """Trigger text-to-speech"""
    text = ' '.join(sentence)
    if current_word:
        text += ' ' + current_word
    
    if not text.strip():
        return jsonify({
            'success': False,
            'message': 'No text to speak'
        })
    
    return jsonify({
        'success': True,
        'text': text.strip(),
        'message': 'Speaking...'
    })

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
