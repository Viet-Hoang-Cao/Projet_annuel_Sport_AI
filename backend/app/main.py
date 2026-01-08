from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile
from pathlib import Path

from app.detectors.router import route
from app.training_plans import generate_training_plan, TrainingPlanRequest

# Create static directory if it doesn't exist
static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(exist_ok=True)

app = FastAPI(
    title="Exercise AI API",
    description="Real-time fitness form analyzer using MediaPipe and ML models",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    html_path = static_dir / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head><title>Exercise AI - Video Upload</title></head>
    <body>
        <h1>Exercise AI API</h1>
        <p>API is running. Please ensure index.html is in the static directory.</p>
    </body>
    </html>
    """)

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok", "message": "Exercise AI API is running"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Analyze uploaded video for exercise form detection.
    Supports: plank, squat, push-up, and other exercises.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=400,
            detail="File must be a video. Supported formats: mp4, avi, mov, etc."
        )
    
    # Create temporary file with proper extension
    file_ext = os.path.splitext(file.filename)[1] or '.mp4'
    temp_fd, temp_path = tempfile.mkstemp(suffix=file_ext)
    
    try:
        # Save uploaded file
        with os.fdopen(temp_fd, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        # Validate file exists and has content
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Run inference
        try:
            result = route(temp_path)
            return {
                "success": True,
                "filename": file.filename,
                **result
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error during analysis: {str(e)}"
            )
    
    finally:
        # Cleanup temporary file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            print(f"Warning: Could not remove temp file {temp_path}: {e}")

@app.post("/training-plan")
async def create_training_plan(request: TrainingPlanRequest):
    """
    Generate a personalized training plan based on user criteria.
    
    - **age**: User's age (12-100)
    - **objective**: Training goal (weight_loss, muscle_gain, fitness)
    - **level**: Fitness level (beginner, intermediate, advanced)
    - **frequency**: Number of training sessions per week (2-7)
    """
    try:
        # Validate inputs
        if request.age < 12 or request.age > 100:
            raise HTTPException(status_code=400, detail="Age must be between 12 and 100")
        
        if request.frequency < 2 or request.frequency > 7:
            raise HTTPException(status_code=400, detail="Frequency must be between 2 and 7 sessions per week")
        
        if request.objective not in ["weight_loss", "muscle_gain", "fitness"]:
            raise HTTPException(status_code=400, detail="Objective must be: weight_loss, muscle_gain, or fitness")
        
        if request.level not in ["beginner", "intermediate", "advanced"]:
            raise HTTPException(status_code=400, detail="Level must be: beginner, intermediate, or advanced")
        
        # Generate plan
        result = generate_training_plan(request)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating training plan: {str(e)}")
