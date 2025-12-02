import torch
import cv2
import pickle
from collections import deque, Counter
from model_definition import ExerciseLSTM
import mediapipe as mp

# -------------------------------
# Charger le modèle et LabelEncoder
# -------------------------------
with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

num_classes = len(le.classes_)
model = ExerciseLSTM(input_size=132, hidden_size=128, num_layers=2, num_classes=num_classes)
model.load_state_dict(torch.load("models/exercise_model.pth", map_location="cpu"))
model.eval()

# -------------------------------
# Mediapipe
# -------------------------------
mp_pose = mp.solutions.pose.Pose(
    static_image_mode=False, model_complexity=1,
    enable_segmentation=False, min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_landmarks(frame):
    results = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None
    row = []
    for lm in results.pose_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])
    return row

# -------------------------------
# Live webcam
# -------------------------------
cap = cv2.VideoCapture(0)
SEQ_LEN = 50
buffer = []
history = deque(maxlen=10)  # 10 dernières prédictions pour vote

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pts = extract_landmarks(frame)
    if pts is not None:
        buffer.append(pts)

    if len(buffer) > SEQ_LEN:
        buffer.pop(0)

    if len(buffer) == SEQ_LEN:
        seq = torch.tensor([buffer], dtype=torch.float32)
        with torch.no_grad():
            pred = model(seq)
            label_id = torch.argmax(pred, 1).item()
            history.append(label_id)

        # Vote majoritaire
        final_label_id = Counter(history).most_common(1)[0][0]
        final_label = le.inverse_transform([final_label_id])[0]

        cv2.putText(frame, f"Prediction: {final_label}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Exercise Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
