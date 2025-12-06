# 🚀 Hugging Face Spaces Deployment Guide

## Complete Step-by-Step Instructions for Deploying ASL to Voice

---

## 📋 Prerequisites

Before you begin, make sure you have:

1. ✅ A Hugging Face account (free) - Sign up at https://huggingface.co/join
2. ✅ Git installed on your computer - Download from https://git-scm.com/
3. ✅ Your trained model file located at `runs/train/asl_detection/weights/best.pt`
4. ✅ All project files cleaned and ready (already done!)

---

## 🎯 Step 1: Create a Hugging Face Space

### 1.1 Log in to Hugging Face
- Go to https://huggingface.co/
- Click "Sign in" and enter your credentials

### 1.2 Create a New Space
1. Click your profile icon (top-right corner)
2. Select **"New Space"**
3. Fill in the details:
   - **Owner**: Your username (e.g., `your-username`)
   - **Space name**: `asl-to-voice` (or your preferred name)
   - **License**: MIT
   - **Select SDK**: Choose **Gradio**
   - **Space hardware**: Select **CPU basic** (free tier) or **GPU** if available
   - **Visibility**: Choose **Public** (so anyone can use it)
4. Click **"Create Space"**

---

## 🔧 Step 2: Prepare Your Local Repository

### 2.1 Open PowerShell/Terminal
Navigate to your project folder:
```powershell
cd C:\personal\Defense\ASL_to_Voice
```

### 2.2 Initialize Git (if not already done)
```powershell
git init
```

### 2.3 Add Hugging Face Remote
Replace `YOUR-USERNAME` and `SPACE-NAME` with your actual values:
```powershell
git remote add origin https://huggingface.co/spaces/researchpurpose/asl-to-voice
```

Example:
```powershell
git remote add origin https://huggingface.co/spaces/researchpurpose/asl-to-voice
```

---

## 📦 Step 3: Verify Your Files

Make sure your project has these essential files:

```
ASL_to_Voice/
│
├── app.py                          ← Main Gradio application
├── requirements.txt                ← Python dependencies
├── README.md                       ← Space description (with YAML frontmatter)
│
└── runs/
    └── train/
        └── asl_detection/
            └── weights/
                └── best.pt         ← Your trained model (CRITICAL!)
```

### Check if model exists:
```powershell
Test-Path "runs/train/asl_detection/weights/best.pt"
```
Should return `True`. If not, you need to copy your model file there!

---

## 🔑 Step 4: Authenticate with Hugging Face

### 4.1 Get Your Access Token
1. Go to https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name it: `git-push-token`
4. Select **"Write"** access
5. Click **"Generate"**
6. **COPY THE TOKEN** (you won't see it again!)

### 4.2 Configure Git Credentials
```powershell
git config --global credential.helper store
```

---

## 📤 Step 5: Push to Hugging Face Spaces

### 5.1 Stage All Files
```powershell
git add .
```

### 5.2 Commit Your Changes
```powershell
git commit -m "Initial deployment of ASL to Voice app"
```

### 5.3 Push to Hugging Face
```powershell
git push -u origin main
```

**When prompted:**
- **Username**: Your Hugging Face username
- **Password**: Paste your access token (the one you copied earlier)

### 5.4 If the branch is named "master" instead of "main":
```powershell
git branch -M main
git push -u origin main
```

---

## ⏳ Step 6: Wait for Build

### 6.1 Monitor Build Progress
1. Go to your Space URL: `https://huggingface.co/spaces/YOUR-USERNAME/SPACE-NAME`
2. You'll see a "Building" status with logs
3. The build process takes **5-10 minutes** (downloading dependencies + loading model)

### 6.2 Watch for These Steps:
- ✅ Cloning repository
- ✅ Installing requirements (Gradio, Ultralytics, PyTorch, etc.)
- ✅ Loading YOLOv8 model
- ✅ Starting Gradio server

---

## ✅ Step 7: Test Your Deployment

### 7.1 Once Build Completes:
Your Space will show the live Gradio interface!

### 7.2 Test the App:
1. Click **"Allow"** when prompted for webcam access
2. Show an ASL sign to your webcam
3. Wait for "Stable Letter" to appear
4. Click "Add Letter" to build words
5. Click "Speak" to test text-to-speech

---

## 🎨 Step 8: Customize Your Space (Optional)

### 8.1 Update Space Card
Edit `README.md` to change:
- Title
- Description
- Emoji
- Color scheme
- Instructions

### 8.2 Push Updates
```powershell
git add README.md
git commit -m "Update Space card"
git push
```

---

## 🔄 Step 9: Update Your App Later

### When you make changes to `app.py` or other files:

```powershell
# 1. Make your changes to files
# 2. Stage changes
git add .

# 3. Commit with a descriptive message
git commit -m "Improved detection accuracy"

# 4. Push to Hugging Face
git push
```

The Space will automatically rebuild with your changes!

---

## 🐛 Troubleshooting

### Problem: "Model file not found"
**Solution**: Ensure `runs/train/asl_detection/weights/best.pt` exists
```powershell
Test-Path "runs/train/asl_detection/weights/best.pt"
```

### Problem: "Permission denied"
**Solution**: Check your Hugging Face token has "Write" access

### Problem: "Out of memory"
**Solution**: 
- Reduce model size in app.py
- Request GPU hardware in Space settings
- Use a smaller YOLOv8 model (nano instead of small)

### Problem: "Build failed"
**Solution**: Check the build logs in your Space for specific errors
- Usually missing dependencies in `requirements.txt`
- Or Python version mismatch

### Problem: "Webcam not working"
**Solution**: 
- Make sure you're using HTTPS (not HTTP)
- Grant browser camera permissions
- Try a different browser (Chrome recommended)

---

## 📊 Monitoring Usage

### View Space Analytics:
1. Go to your Space page
2. Click **"Settings"** tab
3. View visitor stats, CPU/GPU usage, and uptime

---

## 🎉 You're Done!

Your ASL to Voice app is now live and publicly accessible at:
```
https://huggingface.co/spaces/YOUR-USERNAME/SPACE-NAME
```

### Share Your Space:
- Copy the URL and share it with friends
- Embed it in your website
- Share on social media with hashtags: #ASL #AI #HuggingFace

---

## 🔗 Useful Links

- **Your Hugging Face Spaces**: https://huggingface.co/spaces
- **Gradio Documentation**: https://gradio.app/docs/
- **Hugging Face Docs**: https://huggingface.co/docs/hub/spaces
- **Git Documentation**: https://git-scm.com/doc

---

## 💡 Pro Tips

1. **Use GPU for faster inference**: Upgrade to GPU hardware in Space settings (free tier available)
2. **Add examples**: Include demo images in your Space for users who don't have a webcam
3. **Monitor performance**: Check Space logs regularly for errors
4. **Version control**: Use meaningful commit messages for easy rollback
5. **Backup your model**: Keep a copy of `best.pt` somewhere safe!

---

## 🆘 Need Help?

- **Hugging Face Forum**: https://discuss.huggingface.co/
- **Gradio Discord**: https://discord.gg/gradio
- **GitHub Issues**: (if you have a public repo)

---

**Happy Deploying! 🚀🤟**
