from app.detectors.plank import predict as plank_predict
from app.detectors.squat import predict as squat_predict
from app.pose.extract import detect_motion_type
import os

def route(video_path):
    """
    Route video to appropriate exercise detector.
    Returns analysis results with exercise type and form quality.
    """
    if not os.path.exists(video_path):
        return {
            "exercise": "unknown",
            "result": "Video file not found",
            "error": "File does not exist"
        }
    
    try:
        # Detect exercise type
        ex = detect_motion_type(video_path)
        
        # Read video bytes
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        
        if not video_bytes:
            return {
                "exercise": "unknown",
                "result": "Empty video file",
                "error": "File is empty"
            }
        
        # Route to appropriate detector
        if ex == "plank":
            return plank_predict(video_bytes)
        elif ex == "squat":
            return squat_predict(video_bytes)
        else:
            return {
                "exercise": "unknown",
                "result": f"Cannot detect exercise type. Detected: {ex}",
                "detected_type": ex
            }
    
    except Exception as e:
        return {
            "exercise": "unknown",
            "result": f"Error during analysis: {str(e)}",
            "error": str(e)
        }