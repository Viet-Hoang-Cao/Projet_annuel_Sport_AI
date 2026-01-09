import torch
import torch.nn as nn
import numpy as np
import pickle
import mediapipe as mp
import cv2
import tempfile
import os

MODEL_PATH  = "app/models/squat_mlp_full.pt"
SCALER_PATH = "app/scalers/scaler_full_squat.pkl"

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

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

MODEL = MLP_Full(132)
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

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            continue

        X = scaler.transform(extract_full_landmarks(res))
        p = torch.sigmoid(MODEL(torch.tensor(X, dtype=torch.float32))).item()
        probs.append(p)

    cap.release()
    os.remove(path)

    if len(probs) == 0:
        return {"exercise":"squat","result":"NO POSE DETECTED"}

    probs = np.array(probs)
    bad_ratio = np.mean(probs > 0.5)

    return {
        "exercise":"squat",
        "bad_ratio": round(float(bad_ratio),3),
        "result":"BAD FORM" if bad_ratio > 0.30 else "GOOD FORM"
    }
