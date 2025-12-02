import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
import pickle

# ---------------------------------------------------------
# 1) Charger le label encoder
# ---------------------------------------------------------
with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ---------------------------------------------------------
# 2) Architecture du modèle (identique à ton notebook !)
# ---------------------------------------------------------
class ExerciseLSTM(nn.Module):
    def __init__(self, input_size=132, hidden_size=128, num_layers=2, num_classes=20):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]          # Dernière frame
        out = self.fc(out)
        return out

# IMPORTANT → mettre num_classes = nombre de tes labels
num_classes = len(le.classes_)
model = ExerciseLSTM(input_size=132, hidden_size=128,
                     num_layers=2, num_classes=num_classes)

# Charger le modèle entraîné
model.load_state_dict(torch.load("models/exercise_model.pth", map_location="cpu"))
model.eval()

# ---------------------------------------------------------
# 3) Normalisation (identique à ton notebook !)
# ---------------------------------------------------------
def normalize_landmarks(arr):
    # arr shape = (n_frames, 132)
    arr = arr.copy()

    # reshape to (n_frames, 33, 4)
    pts = arr.reshape(len(arr), 33, 4)

    # Centrer sur le bassin (landmark 23 = left hip)
    center = pts[:, 23:24, :3]
    pts[:, :, :3] -= center

    # Normalisation taille du corps (distance hanche-épaules)
    ref = np.linalg.norm(
        pts[:, 11, :3] - pts[:, 23, :3], axis=1
    ).reshape(-1, 1, 1)

    pts[:, :, :3] /= (ref + 1e-6)

    return pts.reshape(len(arr), -1)

# ---------------------------------------------------------
# 4) Extraction Mediapipe
# ---------------------------------------------------------
def extract_landmarks_from_video(video_path):
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(frame_rgb)

        if results.pose_landmarks:
            row = []
            for lm in results.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])
            frames.append(row)

    cap.release()
    return np.array(frames)

# ---------------------------------------------------------
# 5) Prédiction
# ---------------------------------------------------------
def predict_exercise(video_path, seq_len=50):

    landmarks = extract_landmarks_from_video(video_path)

    if len(landmarks) < seq_len:
        raise ValueError("Vidéo trop courte pour la prédiction.")

    # Normalisation
    landmarks = normalize_landmarks(landmarks)

    # prendre la séquence du milieu
    mid = len(landmarks) // 2
    seq = landmarks[mid - seq_len//2 : mid + seq_len//2]

    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        out = model(seq)
        pred = torch.argmax(out, dim=1).item()

    return le.inverse_transform([pred])[0]

# ---------------------------------------------------------
# 6) EXEMPLE D'UTILISATION
# ---------------------------------------------------------
if __name__ == "__main__":
    video = "test_videos/squat.mp4"
    print("Exercice détecté :", predict_exercise(video))
