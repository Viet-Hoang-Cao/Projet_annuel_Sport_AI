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
    """
    Detect exercise type by analyzing hip movement.
    Returns 'squat' if significant vertical movement, 'plank' otherwise.
    """
    try:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            return "unknown"
        
        ys = []
        mp_pose = mp.solutions.pose.Pose()
        frames_checked = 0
        max_frames = 40

        while frames_checked < max_frames:
            ret, f = cap.read()
            if not ret:
                break
            
            frames_checked += 1
            r = mp_pose.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            
            if r.pose_landmarks:
                # Use right hip landmark (index 24)
                hip = r.pose_landmarks.landmark[24].y
                ys.append(hip)
        
        cap.release()

        if len(ys) < 5:  # Need at least 5 frames with pose detection
            return "unknown"
        
        d = np.ptp(ys)  # Peak-to-peak (range) of hip y-coordinates
        return "squat" if d > 0.08 else "plank"
    
    except Exception as e:
        print(f"Error detecting motion type: {e}")
        return "unknown"
