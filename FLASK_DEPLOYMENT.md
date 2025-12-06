# 🚀 Flask Version Deployment Guide

## ASL to Voice - Flask Web Application

---

## 📁 Files Overview

### Essential Files:
- `app_flask.py` - Main Flask application with video streaming
- `templates/index_flask.html` - Modern, responsive HTML interface
- `requirements_flask.txt` - Python dependencies for Flask version
- `runs/train/asl_detection/weights/best.pt` - Your trained model

---

## 🖥️ Local Development

### 1. Install Dependencies
```powershell
pip install -r requirements_flask.txt
```

### 2. Run the App
```powershell
python app_flask.py
```

### 3. Open in Browser
Navigate to: `http://localhost:5000`

---

## 🌐 Deployment Options

### Option 1: Render (Recommended - Free Tier Available)

#### Setup:
1. Create account at https://render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `asl-to-voice`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements_flask.txt`
   - **Start Command**: `gunicorn -w 1 -b 0.0.0.0:$PORT app_flask:app --timeout 120`
   - **Instance Type**: Free

#### Notes:
- Free tier has 750 hours/month
- App sleeps after inactivity (cold starts)
- Good for demos and testing

---

### Option 2: Railway (Easy Deployment)

#### Setup:
1. Create account at https://railway.app
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway auto-detects Flask app
5. Add environment variables if needed

#### Configuration:
Create `Procfile` in your project root:
```
web: gunicorn -w 1 -b 0.0.0.0:$PORT app_flask:app --timeout 120
```

#### Notes:
- $5 free credit/month
- Fast deployment
- Good performance

---

### Option 3: PythonAnywhere (Flask-Friendly)

#### Setup:
1. Create account at https://www.pythonanywhere.com
2. Upload files via dashboard or git clone
3. Create new web app:
   - Python 3.10
   - Manual configuration
4. Configure WSGI file:

```python
import sys
path = '/home/yourusername/ASL_to_Voice'
if path not in sys.path:
    sys.path.append(path)

from app_flask import app as application
```

5. Install requirements in virtual environment
6. Reload web app

#### Notes:
- Free tier available
- Limited CPU/bandwidth on free tier
- Good for persistent apps

---

### Option 4: Heroku (Classic Choice)

#### Setup:
1. Create account at https://heroku.com
2. Install Heroku CLI
3. Create `Procfile`:
```
web: gunicorn -w 1 -b 0.0.0.0:$PORT app_flask:app --timeout 120
```

4. Deploy:
```powershell
heroku login
heroku create asl-to-voice
git push heroku main
```

#### Notes:
- Free tier discontinued (paid plans only)
- Reliable and stable
- Good for production

---

### Option 5: AWS EC2 (Full Control)

#### Setup:
1. Launch Ubuntu EC2 instance
2. SSH into instance
3. Install dependencies:
```bash
sudo apt update
sudo apt install python3-pip python3-opencv
git clone <your-repo>
cd ASL_to_Voice
pip3 install -r requirements_flask.txt
```

4. Run with Gunicorn:
```bash
gunicorn -w 1 -b 0.0.0.0:5000 app_flask:app --timeout 120
```

5. Configure security group (allow port 80/443)
6. Optional: Use Nginx as reverse proxy

#### Notes:
- Free tier: 750 hours/month (1 year)
- Full control over server
- Requires more setup

---

### Option 6: Google Cloud Run (Serverless)

#### Setup:
1. Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_flask.txt .
RUN pip install --no-cache-dir -r requirements_flask.txt

COPY . .

CMD exec gunicorn --bind :$PORT --workers 1 --timeout 120 app_flask:app
```

2. Deploy:
```powershell
gcloud run deploy asl-to-voice --source . --platform managed --region us-central1 --allow-unauthenticated
```

#### Notes:
- Pay per use (generous free tier)
- Auto-scaling
- Cold starts possible

---

## 🔧 Production Configuration

### Gunicorn Settings (for all deployments):

```python
# gunicorn_config.py
bind = "0.0.0.0:5000"
workers = 1  # Important: Use 1 worker for webcam access
worker_class = "sync"
timeout = 120
keepalive = 5
```

Run with:
```powershell
gunicorn -c gunicorn_config.py app_flask:app
```

### Important Notes:
- **Use 1 worker** - Multiple workers cause camera access conflicts
- **Timeout 120s** - Model loading takes time
- **Thread mode** - Better for I/O operations

---

## 📊 Performance Optimization

### 1. Model Loading
- Model loads once on startup (global variable)
- Cached in memory for all requests

### 2. Video Streaming
- Uses MJPEG streaming (efficient for Flask)
- ~15-30 FPS depending on hardware

### 3. Detection Smoothing
- 10-frame buffer for stable predictions
- Cooldown prevents duplicate letters

---

## 🐛 Troubleshooting

### Problem: "Camera not found"
**Solution**: 
- Local: Check camera permissions
- Remote: Deploy on platform supporting webcam access (most cloud services don't allow direct camera access - users access via browser)

### Problem: "Model file not found"
**Solution**:
```powershell
# Verify model exists
Test-Path "runs/train/asl_detection/weights/best.pt"

# If missing, copy from your trained model location
```

### Problem: "Slow performance"
**Solution**:
- Use GPU instance (AWS, GCP)
- Reduce model inference frequency
- Lower video resolution

### Problem: "Connection timeout"
**Solution**:
- Increase Gunicorn timeout: `--timeout 180`
- Check firewall settings
- Ensure HTTPS for secure camera access

---

## 🔐 Security Considerations

### For Production:
1. **HTTPS Required** - Browsers block camera access on HTTP
2. **Environment Variables** - Don't hardcode sensitive data
3. **Rate Limiting** - Prevent abuse
4. **CORS Configuration** - If accessing from different domain

### Example with Flask-CORS:
```python
from flask_cors import CORS
CORS(app, resources={r"/*": {"origins": "https://yourdomain.com"}})
```

---

## 📱 Features

### Current Implementation:
- ✅ Real-time video streaming with detection
- ✅ Letter-by-letter word building
- ✅ Sentence construction
- ✅ Text-to-speech (browser-based)
- ✅ Punctuation support
- ✅ Delete/Clear/Reset functions
- ✅ Responsive design (mobile-friendly)

### Potential Enhancements:
- 📝 Save sentences to database
- 👥 Multi-user support
- 📊 Detection history/analytics
- 🎨 Theme customization
- 🌍 Multi-language support

---

## 🚀 Quick Start Commands

### Development:
```powershell
# Install dependencies
pip install -r requirements_flask.txt

# Run locally
python app_flask.py
```

### Production (example with Gunicorn):
```powershell
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn -w 1 -b 0.0.0.0:5000 app_flask:app --timeout 120
```

---

## 🔗 Useful Resources

- **Flask Documentation**: https://flask.palletsprojects.com/
- **Gunicorn Documentation**: https://docs.gunicorn.org/
- **Render Deployment**: https://render.com/docs
- **Railway Deployment**: https://docs.railway.app/
- **OpenCV Flask Streaming**: https://blog.miguelgrinberg.com/post/video-streaming-with-flask

---

## 💡 Best Practices

1. **Version Control**: Keep model file in Git LFS
2. **Environment Variables**: Use `.env` for configuration
3. **Logging**: Implement proper logging for debugging
4. **Monitoring**: Use platform monitoring tools
5. **Backups**: Regular backups of model and data

---

## 🎉 You're Ready!

Your Flask app is production-ready with:
- Real-time streaming
- Smooth detection
- Modern UI
- Mobile responsive
- Easy deployment options

Choose your deployment platform and go live! 🚀🤟

---

**Need Help?** Check platform-specific documentation or open an issue!
