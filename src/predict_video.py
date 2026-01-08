import torch
import numpy as np
import cv2
import mediapipe as mp
import pickle
from collections import Counter
from model_definition import ExerciseLSTM
from sklearn.preprocessing import LabelEncoder
import sys
import os

# ---------------------------
# 1) Definir les classes (FIX for 22 classes model)
# ---------------------------
CLASSES_LIST = [
    'barbell biceps curl',
    'bench press',
    'chest fly machine',
    'deadlift',
    'decline bench press',
    'hammer curl',
    'hip thrust',
    'incline bench press',
    'lat pulldown',
    'lateral raise',
    'leg extension',
    'leg raises',
    'plank',
    'pull Up',
    'push-up',
    'romanian deadlift',
    'russian twist',
    'shoulder press',
    'squat',
    't bar row',
    'tricep Pushdown',
    'tricep dips'
]
CLASSES_LIST.sort()
le = LabelEncoder()
le.classes_ = np.array(CLASSES_LIST)

# ---------------------------
# 2) Charger le modèle
# ---------------------------
num_classes = len(CLASSES_LIST)
model = ExerciseLSTM(input_size=132, hidden_size=128, num_layers=2, num_classes=num_classes)

model_path = "notebooks/exercise_model.pth"
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found.")
    exit(1)

model.load_state_dict(torch.load(model_path, map_location="cpu"))
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
if len(sys.argv) > 1:
    video_path = sys.argv[1]
else:
    video_path = "test_video.mp4" # Placeholder

print(f"Processing video: {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Impossible d'ouvrir la vidéo : {video_path}")
    print("Usage: python src/predict_video.py <path_to_video>")
    exit(1)

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
    print("📉 Vidéo trop courte pour extraire une séquence complète.")
    exit(1)

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
if all_predictions:
    final_label_id = Counter(all_predictions).most_common(1)[0][0]
    final_label = le.inverse_transform([final_label_id])[0]
    print("✅ Exercice principal de la vidéo :", final_label)
else:
    print("No predictions made.")
