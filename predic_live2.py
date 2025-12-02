import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp

# -----------------------------
# 1️⃣ Classes / mapping label ↔ id
# -----------------------------
classes = [
    "squat", "push-up", "bench press", "pull up", "plank",
    "tricep dips", "shoulder press", "deadlift", "bicep curl"
]

id_to_label = {i: label for i, label in enumerate(classes)}
num_classes = len(classes)

# -----------------------------
# 2️⃣ Modèle LSTM
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

input_size = 132
hidden_size = 128
num_layers = 2
model = LSTMModel(input_size, hidden_size, num_layers, num_classes)

# -----------------------------
# 3️⃣ Charger les poids
# -----------------------------
state_dict = torch.load("notebooks/exercise_model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# -----------------------------
# 4️⃣ Fonction extraction landmarks
# -----------------------------
def extract_landmarks_from_frame(frame, pose_model):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb)
    if results.pose_landmarks:
        return [lm_coord for lm in results.pose_landmarks.landmark for lm_coord in (lm.x, lm.y, lm.z, lm.visibility)]
    return None

# -----------------------------
# 5️⃣ Live webcam
# -----------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    seq_len = 50
    buffer = []

    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extract_landmarks_from_frame(frame, mp_pose)
        if landmarks:
            buffer.append(landmarks)

        if len(buffer) > seq_len:
            buffer.pop(0)

        if len(buffer) == seq_len:
            seq = torch.tensor([buffer], dtype=torch.float32)
            with torch.no_grad():
                out = model(seq)
                label_id = torch.argmax(out, dim=1).item()
                prediction = id_to_label[label_id]

            cv2.putText(frame, f"Prediction: {prediction}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Live Exercise Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
