import cv2, mediapipe as mp, torch, numpy as np
from app.detectors.plank import MODEL, scaler, extract_full_landmarks
from app.pose.extract import detect_motion_type

def stream_video(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp.solutions.pose.Pose()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            row = extract_full_landmarks(res)
            X = scaler.transform(row)
            p = torch.sigmoid(MODEL(torch.tensor(X,dtype=torch.float32))).item()

            label = "BAD FORM" if p>0.5 else "GOOD FORM"
            color = (0,0,255) if p>0.5 else (0,255,0)

            cv2.putText(frame,label,(40,50),cv2.FONT_HERSHEY_SIMPLEX,1,color,3)
            mp.solutions.drawing_utils.draw_landmarks(frame,res.pose_landmarks,mp.solutions.pose.POSE_CONNECTIONS)

        ret,jpg = cv2.imencode('.jpg',frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')