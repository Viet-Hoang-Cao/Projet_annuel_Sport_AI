import torch
import numpy as np
import cv2
import mediapipe as mp
import pickle
from collections import Counter
from model_definition import ExerciseLSTM  # ton modèle défini dans model_definition.py

# ---------------------------
# 1) Charger le LabelEncoder
# ---------------------------
with open(r"C:/Users/caovi/OneDrive/Desktop/projet annuel/models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ---------------------------
# 2) Charger le modèle
# ---------------------------
num_classes = len(le.classes_)
model = ExerciseLSTM(input_size=132, hidden_size=128, num_layers=2, num_classes=num_classes)
model.load_state_dict(torch.load(r"C:/Users/caovi/OneDrive/Desktop/projet annuel/models/exercise_model.pth", map_location="cpu"))
model.eval()

# ---------------------------
# 3) Initialiser Mediapipe
# ---------------------------
mp_pose = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------------
# 4) Extraire les landmarks
# ---------------------------
def extract_landmarks(frame):
    results = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None
    row = []
    for lm in results.pose_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])
    return row

# ---------------------------
# 5) Paramètres pour séquence et vote
# ---------------------------
SEQ_LEN = 50
STEP = 10

# ---------------------------
# 6) Ouvrir la vidéo
# ---------------------------
video_path = r"C:\Users\caovi\OneDrive\Desktop\projet annuel\notebooks\verified_data\data_crawl_10s\deadlift\2e22d77e-94ce-47f6-912d-a28f1c4f70cf.mp4"  # <-- changer le chemin
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError(f"Impossible d'ouvrir la vidéo : {video_path}")

landmarks = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pts = extract_landmarks(frame)
    if pts is not None:
        landmarks.append(pts)

cap.release()

if len(landmarks) < SEQ_LEN:
    raise ValueError("📉 Vidéo trop courte pour extraire une séquence complète.")

# ---------------------------
# 7) Prédiction par séquences glissantes
# ---------------------------
all_predictions = []

for i in range(0, len(landmarks) - SEQ_LEN + 1, STEP):
    seq = torch.tensor([landmarks[i:i+SEQ_LEN]], dtype=torch.float32)
    with torch.no_grad():
        pred = model(seq)
        label_id = torch.argmax(pred, 1).item()
        all_predictions.append(label_id)

# ---------------------------
# 8) Vote majoritaire
# ---------------------------
final_label_id = Counter(all_predictions).most_common(1)[0][0]
final_label = le.inverse_transform([final_label_id])[0]

print("✅ Exercice principal de la vidéo :", final_label)
