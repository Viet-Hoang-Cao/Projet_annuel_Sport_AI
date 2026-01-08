import torch
import torch.nn as nn
import numpy as np
import pickle
import mediapipe as mp
import cv2
import tempfile
import os

# PATHS
MODEL_PATH  = "app/models/plank_mlp_full.pt"
SCALER_PATH = "app/scalers/scaler_full_plank.pkl"

#LOAD SCALER
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
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
MODEL.eval()

mp_pose = mp.solutions.pose


def extract_full_landmarks(results):
    row = []
    for lm in results.pose_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(row).reshape(1, -1)


def predict(video_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        path = tmp.name

    cap = cv2.VideoCapture(path)
    pose = mp_pose.Pose()
    probs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if not results.pose_landmarks:
            continue

        row = extract_full_landmarks(results)
        row = scaler.transform(row)
        x = torch.tensor(row, dtype=torch.float32)

        with torch.no_grad():
            p = torch.sigmoid(MODEL(x)).item()
            probs.append(p)

    cap.release()
    os.remove(path)

    if len(probs) == 0:
        return {"exercise": "plank", "result": "NO POSE DETECTED"}

    mean_prob = float(np.mean(probs))
    result = "BAD FORM" if mean_prob > 0.5 else "GOOD FORM"

    return {
        "exercise": "plank",
        "confidence": round(mean_prob, 3),
        "result": result
    }
