import torch
import cv2
import mediapipe as mp
import pickle

# Charger le modèle
model = torch.load("notebooks/exercise_model.pth", map_location="cpu")
model.eval()

# Charger les labels
with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Webcam

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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(rgb)

        if results.pose_landmarks:
            row = []
            for lm in results.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])

            buffer.append(row)

        # On garde un buffer fixe de 50 frames
        if len(buffer) > seq_len:
            buffer.pop(0)

        # Quand on a 50 frames → prédiction
        if len(buffer) == seq_len:
            seq = torch.tensor([buffer], dtype=torch.float32)

            with torch.no_grad():
                out = model(seq)
                label_id = torch.argmax(out, dim=1).item()
                prediction = le.inverse_transform([label_id])[0]

            cv2.putText(frame, f"Prediction: {prediction}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        cv2.imshow("Live Exercise Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
