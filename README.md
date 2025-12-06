---
title: ASL to Voice - Real-Time Sign Language Detection
emoji: 🤟
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# 🤟 ASL to Voice - Real-Time Sign Language Detection

Convert American Sign Language alphabet signs to text and speech in real-time!

## 🌟 Features

- **Real-time Detection**: Live webcam-based ASL alphabet recognition
- **High Accuracy**: 96.2% mAP@0.5 on 26 ASL letter classes (A-Z)
- **Temporal Smoothing**: Reduces jitter for stable letter detection
- **Text Assembly**: Build words and sentences from detected letters
- **Text-to-Speech**: Convert your signed messages to spoken words

## 🎯 How to Use

1. **Allow Webcam Access**: Click "Allow" when prompted for camera permissions
2. **Show ASL Sign**: Hold an ASL letter sign clearly in front of your webcam
3. **Wait for Stable Detection**: Watch the "Stable Letter" field for consistent detection (1-2 seconds)
4. **Add Letter**: Click "➕ Add Letter" button or press SPACE to add the detected letter to your word
5. **Build Words**: Continue signing letters to build complete words
6. **Add Space**: Click "⎵ Space" or press ENTER to finalize the current word and start a new one
7. **Add Punctuation**: Use punctuation buttons (. , ! ?) as needed
8. **Speak**: Click "🔊 Speak" to hear your sentence using text-to-speech

## 🎨 Interface

### Detection Panel
- **Raw Detection**: Shows instant detection with confidence
- **Stable Letter**: Shows temporally smoothed detection (more reliable)
- **Current Word**: Letter-by-letter word being built
- **Sentence**: Complete sentence with all words

### Control Buttons
- **➕ Add Letter**: Add the stable detected letter to current word
- **⎵ Space**: Complete word and add to sentence
- **. , ! ?**: Add punctuation marks
- **⌫ Delete**: Remove last character from current word
- **Clear Word**: Clear the entire current word
- **Reset All**: Start over with a fresh sentence
- **🔊 Speak**: Text-to-speech for your sentence

## 🎓 Model Information

- **Architecture**: YOLOv8 (You Only Look Once v8)
- **Classes**: 26 ASL alphabet letters (A-Z)
- **Training Dataset**: ASL Alphabet dataset from Roboflow
- **Performance**: 96.2% mAP@0.5 accuracy
- **Inference Speed**: Real-time (10 FPS processing)

## 💡 Tips for Best Results

1. **Lighting**: Ensure good lighting on your hands
2. **Background**: Plain, contrasting background works best
3. **Hand Position**: Keep your hand clearly visible and centered
4. **Steady Signs**: Hold each sign steady for 1-2 seconds
5. **Distance**: Keep hand at arm's length from camera

## 🔧 Technical Details

- **Framework**: Gradio for web interface
- **Model**: YOLOv8 for object detection
- **Smoothing**: 15-frame temporal window with 8-frame minimum threshold
- **Cooldown**: 1.5-second cooldown prevents duplicate letter additions

## 📝 ASL Alphabet Reference

The system recognizes all 26 letters of the ASL fingerspelling alphabet. For signs that involve motion (J, Z), use the starting position of the sign.

## 🚀 Deployment

This app is deployed on Hugging Face Spaces and runs entirely in your browser. Your webcam feed is processed locally and never stored or transmitted.

## 📄 License

MIT License - Feel free to use and modify!

## 🙏 Acknowledgments

- ASL Alphabet dataset from Roboflow
- YOLOv8 by Ultralytics
- Gradio for the web interface

---

**Enjoy communicating with ASL! 🤟✨**
