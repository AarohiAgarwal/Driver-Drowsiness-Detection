import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from main import draw_master_hud
from scratch.test_red_flash import apply_face_red_light_flash

sample_path = config.DDD_DATASET_DIR / "Drowsy" / "A0001.png"
raw = cv2.imread(str(sample_path))
frame = cv2.resize(raw, (640, 480))
bbox = (180, 140, 460, 420)
faces = [{"bbox": bbox}]
primary_face = faces[0]

eye_results = {
    "detected": True,
    "left_state": "CLOSED", "right_state": "CLOSED",
    "left_conf": 98.0, "right_conf": 97.0,
    "left_ear": 0.14, "right_ear": 0.15, "avg_ear": 0.145
}
mouth_results = {
    "state": "YAWN", "confidence": 92.0, "mar": 0.52, "is_yawning": True
}
decision_drowsy = {
    "state": "DROWSY",
    "drowsiness_score": 0.92,
    "temporal_metrics": {
        "closure_duration_sec": 2.5,
        "total_blinks": 5,
        "perclos": 0.40
    }
}

# 1. Apply face red strobe
flashed = apply_face_red_light_flash(frame, bbox, intensity=0.9)

# 2. Apply master HUD on top
hud = draw_master_hud(
    flashed, faces, primary_face, eye_results, mouth_results,
    decision_drowsy, ("SOUNDING [ALL]", (0, 0, 255)), fps=29.0,
    language_mode="all"
)

out_file = config.PREDICTIONS_DIR / "red_flash_tests" / "full_hud_with_red_face_flash.png"
cv2.imwrite(str(out_file), hud)
print(f"Saved full composition: {out_file.name}")
