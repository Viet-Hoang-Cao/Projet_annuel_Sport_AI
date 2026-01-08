# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### 1. Install Dependencies
```bash
cd backend
pip install -r app/requirements.txt
```

### 2. Start the Server
```bash
python run_server.py
```

### 3. Open Your Browser
Navigate to: **http://localhost:8000**

## ✨ What's New

### Web Interface Features:
- 📹 **Drag & Drop Upload**: Simply drag your video file onto the upload area
- 🎥 **Video Preview**: See your video before analysis
- ⚡ **Real-time Analysis**: Get instant feedback on exercise form
- 🎨 **Beautiful UI**: Modern, responsive design
- 📊 **Detailed Results**: See confidence scores and exercise type

### Backend Improvements:
- ✅ **Better Error Handling**: Comprehensive error messages
- ✅ **CORS Support**: Works with any frontend
- ✅ **File Validation**: Checks file types and sizes
- ✅ **Robust Path Handling**: Works from any directory
- ✅ **Health Check Endpoint**: Monitor API status

## 📝 Usage

1. **Upload a Video**: Click the upload area or drag & drop a video file
2. **Preview**: Your video will appear in the preview area
3. **Analyze**: Click "Analyze Exercise Form" button
4. **View Results**: See if your form is GOOD or BAD with confidence score

## 🎯 Supported Exercises

- **Plank**: Detects proper alignment and form
- **Squat**: Analyzes depth and posture
- **Push-up**: Coming soon
- **Lunge**: Coming soon

## 🔧 Troubleshooting

**Model files not found?**
- Ensure model files are in `app/models/` and `app/scalers/`
- Check file names match exactly

**Port already in use?**
```bash
PORT=8080 python run_server.py
```

**Video upload fails?**
- Check video format (MP4, AVI, MOV supported)
- Ensure video shows a person performing an exercise
- Try a shorter video first

## 📚 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🎉 Enjoy!

Your Exercise AI web application is ready to use!