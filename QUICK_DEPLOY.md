# 🚀 Quick Deployment Reference Card

## 📋 Pre-Deployment Checklist
- [x] Model file exists: `runs/train/asl_detection/weights/best.pt`
- [x] `app.py` is ready
- [x] `requirements.txt` is updated
- [x] `README.md` has Hugging Face YAML frontmatter
- [ ] Hugging Face account created
- [ ] Git installed on your computer

---

## ⚡ 5-Minute Deployment Commands

### Step 1: Create Space on Hugging Face
1. Go to https://huggingface.co/new-space
2. Choose: SDK = **Gradio**, Hardware = **CPU basic (free)**
3. Create the space

### Step 2: Get Your Space URL
Copy your space URL, it will look like:
```
https://huggingface.co/spaces/YOUR-USERNAME/SPACE-NAME
```

### Step 3: Run These Commands in PowerShell

```powershell
# Navigate to project
cd C:\personal\Defense\ASL_to_Voice

# Initialize git
git init

# Add Hugging Face remote (REPLACE with your actual URL)
git remote add origin https://huggingface.co/spaces/YOUR-USERNAME/SPACE-NAME

# Stage all files
git add .

# Commit
git commit -m "Deploy ASL to Voice app"

# Push (you'll be asked for username and token)
git push -u origin main
```

### Step 4: Get Access Token
1. Go to https://huggingface.co/settings/tokens
2. Click "New token" → Name: `git-push` → Access: **Write** → Generate
3. Copy the token

### Step 5: Enter Credentials When Prompted
- Username: `your-huggingface-username`
- Password: `paste-your-access-token-here`

---

## ⏳ Build Time
- Initial build: **5-10 minutes**
- Watch progress at your Space URL

---

## ✅ Done!
Your app will be live at:
```
https://huggingface.co/spaces/YOUR-USERNAME/SPACE-NAME
```

---

## 🔄 Update Your App Later

```powershell
# Make changes to your files
# Then:
git add .
git commit -m "Updated app"
git push
```

---

## 📚 Full Instructions
For detailed step-by-step guide with troubleshooting:
→ Open `HUGGINGFACE_DEPLOYMENT.md`

---

## 🆘 Common Issues

**Issue**: Can't push to git
**Fix**: Make sure you're using your access token as password, not your account password

**Issue**: Model not found error
**Fix**: Verify model exists with: `Test-Path "runs/train/asl_detection/weights/best.pt"`

**Issue**: Build fails
**Fix**: Check build logs in your Space for specific error messages

---

**Need help? Open HUGGINGFACE_DEPLOYMENT.md for complete guide!**
