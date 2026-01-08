# Exercise AI - Web Application

A modern web application for analyzing exercise form using AI and computer vision.

## Features

- 🎥 **Video Upload**: Drag & drop or click to upload exercise videos
- 🤖 **AI Analysis**: Automatic exercise detection and form quality assessment
- 📊 **Real-time Results**: Instant feedback on exercise form (GOOD/BAD)
- 🎨 **Modern UI**: Beautiful, responsive web interface
- 🏋️ **Multiple Exercises**: Supports Plank, Squat, Push-up, and Lunge

## Installation

1. **Install dependencies:**
```bash
cd backend
pip install -r app/requirements.txt
```

2. **Ensure model files are present:**
   - `app/models/plank_mlp_full.pt`
   - `app/models/squat_mlp_full.pt`
   - `app/scalers/scaler_full_plank.pkl`
   - `app/scalers/scaler_full_squat.pkl`

## Running the Application

### Option 1: Using the startup script
```bash
python run_server.py
```

### Option 2: Using uvicorn directly
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Option 3: Using environment variables
```bash
PORT=8080 HOST=0.0.0.0 python run_server.py
```

## Accessing the Application

Once the server is running, open your browser and navigate to:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Endpoints

### POST /analyze
Upload a video file for exercise form analysis.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (video file)

**Response:**
```json
{
  "success": true,
  "filename": "exercise_video.mp4",
  "exercise": "plank",
  "result": "GOOD FORM",
  "confidence": 0.234
}
```

### GET /health
Check API health status.

**Response:**
```json
{
  "status": "ok",
  "message": "Exercise AI API is running"
}
```

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI application
│   ├── detectors/           # Exercise detection modules
│   │   ├── plank.py
│   │   ├── squat.py
│   │   └── router.py
│   ├── pose/                # Pose extraction utilities
│   │   └── extract.py
│   ├── models/              # ML model files
│   └── scalers/             # Feature scalers
├── static/                  # Frontend files
│   └── index.html
├── run_server.py            # Startup script
└── requirements.txt         # Python dependencies
```

## Supported Video Formats

- MP4
- AVI
- MOV
- Other formats supported by OpenCV

## Troubleshooting

### Model files not found
Ensure all model and scaler files are in the correct directories:
- `app/models/`
- `app/scalers/`

### Port already in use
Change the port using environment variables:
```bash
PORT=8080 python run_server.py
```

### Video upload fails
- Check file size (large files may timeout)
- Ensure video format is supported
- Verify video contains visible person performing exercise

## Development

The application uses:
- **FastAPI** for the backend API
- **MediaPipe** for pose detection
- **PyTorch** for ML inference
- **OpenCV** for video processing

## License

See main project README for license information.