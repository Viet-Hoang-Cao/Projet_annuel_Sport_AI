from fastapi import FastAPI, UploadFile, File
import shutil
import os

from app.detectors.router import route

app = FastAPI(title="Exercise AI API")

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # 1) sauvegarde temporaire
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2) inference
    result = route(temp_path)

    # 3) nettoyage
    try:
        os.remove(temp_path)
    except:
        pass

    return result
