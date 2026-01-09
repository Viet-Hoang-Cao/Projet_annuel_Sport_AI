from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
import shutil, os
from app.pose.annotate import analyze_and_annotate
from app.pose.annotate_squat import analyze_and_annotate_squat
from app.pose.extract import detect_motion_type
from app.pose.stream import stream_video
import uuid
import subprocess
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Exercise AI")

@app.get("/")
def root():
    return {"status": "ok"}

def normalize_video(path):
    fixed = path.replace(".MOV", "_fixed.mp4").replace(".mov", "_fixed.mp4")
    cmd = ["ffmpeg", "-y", "-i", path, "-vf", "scale=1280:-2",
           "-vcodec", "libx264", "-acodec", "aac", fixed]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return fixed

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    temp_path = f"temp_{uuid.uuid4().hex}.mp4"
    output_path = f"annotated_{uuid.uuid4().hex}.mp4"

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    temp_path = normalize_video(temp_path)

    # AUTO DETECT exercise
    ex = detect_motion_type(temp_path)

    if ex == "plank":
        out_video, verdict = analyze_and_annotate(temp_path, output_path)
    elif ex == "squat":
        out_video, verdict = analyze_and_annotate_squat(temp_path, output_path)
    else:
        return {"error": "Cannot detect exercise"}

    try:
        os.remove(temp_path)
    except:
        pass

    return FileResponse(
        out_video,
        media_type="video/mp4",
        headers={"X-Verdict": verdict, "X-Exercise": ex}
    )

@app.post("/stream")
async def stream(file: UploadFile = File(...)):
    temp_path = f"temp_{uuid.uuid4().hex}.mp4"
    with open(temp_path,"wb") as f:
        shutil.copyfileobj(file.file,f)

    return StreamingResponse(
        stream_video(temp_path),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )