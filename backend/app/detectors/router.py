from app.pose.extract import detect_motion_type
from app.pose.annotate import analyze_and_annotate as plank_annotate
from app.pose.annotate_squat import analyze_and_annotate_squat as squat_annotate

def route(video_path):
    exercise = detect_motion_type(video_path)

    if exercise == "plank":
        return plank_annotate(video_path)
    elif exercise == "squat":
        return squat_annotate(video_path)
    else:
        return plank_annotate(video_path)   # fallback sécurité
