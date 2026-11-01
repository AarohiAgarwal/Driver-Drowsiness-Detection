import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.face_detection import FaceDetector
from main import draw_master_hud

# Test image
sample_path = config.DDD_DATASET_DIR / "Drowsy" / "A0001.png"
raw = cv2.imread(str(sample_path))
frame = cv2.resize(raw, (640, 480))
bbox = (175, 130, 465, 420)

face_detector = FaceDetector()
faces = [{"bbox": bbox, "center": (320, 275), "area": (465-175)*(420-130)}]
primary_face = faces[0]

# Simulate active drowsy alert with wake-up strobe
annotated = face_detector.draw_face_annotations(
    frame, primary_face, num_total_faces=1,
    is_drowsy=True,
    current_time=1000.0  # Peak strobe phase
)

eye_results = {
    "detected": True,
    "left_state": "CLOSED",
    "right_state": "CLOSED",
    "left_conf": 99.2,
    "right_conf": 98.7,
    "left_ear": 0.125,
    "right_ear": 0.130,
    "avg_ear": 0.128
}

mouth_results = {
    "state": "YAWN",
    "confidence": 96.5,
    "mar": 0.540,
    "is_yawning": True
}

decision = {
    "state": "DROWSY",
    "drowsiness_score": 0.94,
    "temporal_metrics": {
        "closure_duration_sec": 2.8,
        "total_blinks": 7,
        "perclos": 0.48
    }
}

hud_final = draw_master_hud(
    annotated, faces, primary_face, eye_results, mouth_results,
    decision, ("SOUNDING [ALL]", (0, 0, 255)), fps=28.5,
    language_mode="all"
)

out_file = config.PREDICTIONS_DIR / "wake_up_strobe_demonstration.png"
cv2.imwrite(str(out_file), hud_final)
print(f"Saved wake-up strobe demonstration to: {out_file}")
