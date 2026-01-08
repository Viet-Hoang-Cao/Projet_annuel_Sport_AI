import torch
import cv2
import pickle
from collections import deque, Counter
from model_definition import ExerciseLSTM
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
import numpy as np

# -------------------------------
# Charger le modèle et LabelEncoder
# -------------------------------

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

# Ensure they are sorted alphabetically as LabelEncoder would do
CLASSES_LIST.sort()

le = LabelEncoder()
# Manually set the classes
le.classes_ = np.array(CLASSES_LIST)

num_classes = len(CLASSES_LIST)
print(f"Model configured for {num_classes} classes.")

model = ExerciseLSTM(input_size=132, hidden_size=128, num_layers=2, num_classes=num_classes)

try:
    model.load_state_dict(torch.load("notebooks/exercise_model.pth", map_location="cpu"))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

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
droidcam_url = "http://192.168.0.25:4747/video"
print(f"Connecting to DroidCam at {droidcam_url}...")
cap = cv2.VideoCapture(droidcam_url)

if not cap.isOpened():
    print(f"Error: Could not connect to DroidCam at {droidcam_url}. Trying default webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open default webcam.")
        exit(1)

SEQ_LEN = 50
buffer = []
history = deque(maxlen=10)  # 10 dernières prédictions pour vote

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
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
        if history:
            final_label_id = Counter(history).most_common(1)[0][0]
            if 0 <= final_label_id < num_classes:
                final_label = le.inverse_transform([final_label_id])[0]
            else:
                final_label = "Unknown"

            cv2.putText(frame, f"Prediction: {final_label}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Exercise Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
