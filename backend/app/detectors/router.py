from app.detectors.plank import predict as plank_predict
from app.detectors.squat import predict as squat_predict
from app.pose.extract import detect_motion_type

def route(video_path):
    ex = detect_motion_type(video_path)

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    if ex == "plank":
        return plank_predict(video_bytes)
    elif ex == "squat":
        return squat_predict(video_bytes)
    else:
        return {"exercise":"unknown","result":"Cannot detect"}