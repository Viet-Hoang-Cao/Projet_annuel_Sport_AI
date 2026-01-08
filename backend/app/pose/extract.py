import mediapipe as mp, cv2, numpy as np

IMPORTANT_LMS = [
    "NOSE","LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW",
    "LEFT_WRIST","RIGHT_WRIST","LEFT_HIP","RIGHT_HIP",
    "LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE",
    "LEFT_HEEL","RIGHT_HEEL","LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX"
]

mp_pose = mp.solutions.pose

def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()
    rows = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks: continue

        row=[]
        for name in IMPORTANT_LMS:
            p = res.pose_landmarks.landmark[mp_pose.PoseLandmark[name].value]
            row += [p.x,p.y,p.z,p.visibility]
        rows.append(row)

    cap.release()
    return np.array(rows)

def detect_motion_type(video):
    cap=cv2.VideoCapture(video)
    ys=[]
    mp_pose=mp.solutions.pose.Pose()

    for _ in range(40):
        ret,f=cap.read()
        if not ret: break
        r=mp_pose.process(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
        if r.pose_landmarks:
            hip=r.pose_landmarks.landmark[24].y
            ys.append(hip)
    cap.release()

    d=np.ptp(ys)
    return "squat" if d>0.08 else "plank"
