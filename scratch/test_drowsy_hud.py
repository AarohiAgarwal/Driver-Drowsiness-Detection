import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from main import draw_master_hud

# Standard webcam frame (640x480)
sample_path = config.DDD_DATASET_DIR / "Drowsy" / "A0001.png"
raw = cv2.imread(str(sample_path))
# Resize to standard webcam resolution 640x480 for realistic dashboard testing
frame = cv2.resize(raw, (640, 480))

faces = [{"bbox": (180, 140, 460, 420)}]
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
    "drowsiness_score": 0.88,
    "temporal_metrics": {
        "closure_duration_sec": 2.1,
        "total_blinks": 4,
        "perclos": 0.35
    }
}

out_dir = config.PREDICTIONS_DIR / "multilingual_tests"
out_dir.mkdir(parents=True, exist_ok=True)

for mode in ["all", "hi", "pa", "en"]:
    hud = draw_master_hud(
        frame.copy(), faces, primary_face, eye_results, mouth_results,
        decision_drowsy, (f"SOUNDING [{mode.upper()}]", (0, 0, 255)), fps=28.5,
        language_mode=mode
    )
    out_file = out_dir / f"webcam_drowsy_{mode}.png"
    cv2.imwrite(str(out_file), hud)
    print(f"Saved: {out_file.name}")

print("Webcam-sized previews rendered!")
