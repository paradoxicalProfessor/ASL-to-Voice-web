# ASL to Voice - Setup Instructions

## Prerequisites
- Python 3.8 or higher
- Webcam

## Installation Steps

### 1. Clone/Copy the Repository
```bash
git clone https://github.com/paradoxicalProfessor/ASL-to-Voice.git
cd ASL-to-Voice
```

Or simply copy the entire project folder to the new PC.

### 2. Install Required Packages

For the **GitHub version** (server-side processing):
```bash
pip install Flask opencv-python ultralytics torch torchvision Pillow numpy
```

For the **client-side version** (no lag), also install:
```bash
pip install onnx onnxruntime
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### 3. Run the Application

#### Option A: Server-Side Processing (Current GitHub Version - app_flask.py)
```bash
python app_flask.py
```
Open browser to: `http://localhost:5000`

**Features:**
- Processes frames on the server
- Uses PyTorch model (best.pt)
- Works immediately after cloning from GitHub
- May have lag on slower servers like Render

**Required Files:**
- `app_flask.py`
- `templates/index_flask.html`
- `runs/train/asl_detection/weights/best.pt`

#### Option B: Client-Side Processing (New Version - No Lag)
First, export the model to ONNX:
```bash
python export_to_onnx.py
```

Then run:
```bash
python app_client_side.py
```
Open browser to: `http://localhost:5000`

**Features:**
- Processes frames in the browser
- Uses ONNX model (best.onnx)
- No server lag, much faster
- Better for deployment on limited servers

**Required Files:**
- `app_client_side.py`
- `templates/index_client_side.html`
- `runs/train/asl_detection/weights/best.onnx`

## Troubleshooting

### "Model not found" error
- For server-side: Make sure `best.pt` exists in `runs/train/asl_detection/weights/`
- For client-side: Run `python export_to_onnx.py` to generate `best.onnx`

### Camera access denied
- Allow browser camera permissions when prompted

### Module not found errors
- Install missing packages: `pip install <package-name>`

### Port already in use
- Change port in the Python file or kill the process using that port

### Lag on Render/Heroku
- Use the client-side version (`app_client_side.py`) instead

## Model Performance
- Accuracy (mAP@0.5): 95.4%
- Precision: 92.5%
- Recall: 92.4%
- Supports A-Z ASL letters
