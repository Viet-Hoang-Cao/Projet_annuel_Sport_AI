import cv2, mediapipe as mp, torch, numpy as np
from app.detectors.plank import MODEL, scaler, extract_full_landmarks

def analyze_and_annotate(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    w,h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    pose = mp.solutions.pose.Pose()
    probs=[]

    while True:
        ret, frame = cap.read()
        if not ret: break

        res = pose.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            X = scaler.transform(extract_full_landmarks(res))
            p = torch.sigmoid(MODEL(torch.tensor(X,dtype=torch.float32))).item()
            probs.append(p)

            label = "BAD FORM" if p>0.5 else "GOOD FORM"
            color = (0,0,255) if p>0.5 else (0,255,0)
            cv2.putText(frame,label,(40,50),cv2.FONT_HERSHEY_SIMPLEX,1,color,3)
            mp.solutions.drawing_utils.draw_landmarks(frame,res.pose_landmarks,mp.solutions.pose.POSE_CONNECTIONS)

        out.write(frame)

    cap.release(); out.release()

    probs = np.array(probs)
    bad_ratio = np.mean(probs > 0.5)
    verdict = "BAD FORM" if bad_ratio > 0.30 else "GOOD FORM"

    return output_path, verdict
