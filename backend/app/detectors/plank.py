import torch
import torch.nn as nn
import numpy as np
import pickle
import mediapipe as mp
import cv2
import tempfile
import os
from pathlib import Path

# PATHS - resolve relative to this file's location
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH  = BASE_DIR / "models" / "plank_mlp_full.pt"
SCALER_PATH = BASE_DIR / "scalers" / "scaler_full_plank.pkl"

#LOAD SCALER
if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

#MODEL DEFINITION (EXACT SAME AS TRAINING)
class MLP_Full(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


INPUT_DIM = 132  # 33 landmarks × 4
MODEL = MLP_Full(INPUT_DIM)
MODEL.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu"))
MODEL.eval()

mp_pose = mp.solutions.pose


def extract_full_landmarks(results):
    row = []
    for lm in results.pose_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(row).reshape(1, -1)


def predict(video_bytes):
    """
    Predict plank form quality from video bytes.
    Returns GOOD FORM or BAD FORM with confidence score.
    """
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            path = tmp.name

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return {
                "exercise": "plank",
                "result": "Could not open video file",
                "error": "Video file is corrupted or unsupported format"
            }

        pose = mp_pose.Pose()
        probs = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            
            if not results.pose_landmarks:
                continue

            try:
                row = extract_full_landmarks(results)
                row = scaler.transform(row)
                x = torch.tensor(row, dtype=torch.float32)

                with torch.no_grad():
                    p = torch.sigmoid(MODEL(x)).item()
                    probs.append(p)
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                continue

        cap.release()

        if len(probs) == 0:
            return {
                "exercise": "plank",
                "result": "NO POSE DETECTED",
                "error": "Could not detect pose landmarks in video"
            }

        mean_prob = float(np.mean(probs))
        result = "BAD FORM" if mean_prob > 0.5 else "GOOD FORM"

        return {
            "exercise": "plank",
            "confidence": round(mean_prob, 3),
            "result": result,
            "frames_analyzed": frame_count,
            "frames_with_pose": len(probs)
        }
    
    except Exception as e:
        return {
            "exercise": "plank",
            "result": f"Error during prediction: {str(e)}",
            "error": str(e)
        }
    
    finally:
        # Cleanup temporary file
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"Warning: Could not remove temp file {path}: {e}")
