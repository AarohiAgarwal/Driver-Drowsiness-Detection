import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from main import draw_master_hud

def apply_emergency_wake_up_strobe(frame: np.ndarray, bbox: tuple, intensity: float = 1.0) -> np.ndarray:
    """
    High-power emergency red wake-up strobe light:
    1. Full-screen red illumination burst (blasts maximum red lumens from monitor).
    2. Intense face-targeted floodlight wash.
    3. High-contrast thick perimeter strobe beacon.
    4. Emergency warning beacon tag: [⚡ WAKE UP! RED LIGHT FLASH ⚡]
    """
    if intensity <= 0.02:
        return frame

    h, w = frame.shape[:2]
    result = frame.copy().astype(np.float32)

    # 1. Full-frame red strobe burst (emits real-world light from screen to driver)
    # Drastically increases red channel while reducing blue/green for pure vivid emergency red
    result[:, :, 2] += 160.0 * intensity             # Red blast
    result[:, :, 1] = result[:, :, 1] * (1.0 - 0.45 * intensity)  # Suppress green
    result[:, :, 0] = result[:, :, 0] * (1.0 - 0.55 * intensity)  # Suppress blue
    np.clip(result, 0, 255, out=result)
    result = result.astype(np.uint8)

    # 2. Targeted high-power face spotlight flood
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        if x2 > x1 and y2 > y1:
            fw, fh = x2 - x1, y2 - y1
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Elliptical spotlight mask on face
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(mask, (center_x, center_y), (int(fw * 0.70), int(fh * 0.85)), 0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (35, 35), 0)

            # Super-saturated face spotlight
            face_flood = result.copy().astype(np.float32)
            face_flood[:, :, 2] = np.clip(face_flood[:, :, 2] + 90.0 * intensity, 0, 255)
            alpha_face = np.expand_dims(mask * (intensity * 0.75), axis=2)
            result = (result.astype(np.float32) * (1.0 - alpha_face) + face_flood * alpha_face).astype(np.uint8)

            # 3. Flashing face beacon box & header tag
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 4)
            tag_h = 32
            tag_w = min(w - x1 - 10, 290)
            cv2.rectangle(result, (x1, max(0, y1 - tag_h)), (x1 + tag_w, y1), (0, 0, 255), -1)
            cv2.putText(result, "WAKE UP! RED LIGHT FLASH", (x1 + 8, max(12, y1 - 9)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # 4. Massive full-screen emergency strobe border
    border_px = int(14 * intensity)
    if border_px > 0:
        cv2.rectangle(result, (0, 0), (w, h), (0, 0, 255), border_px)
        cv2.rectangle(result, (border_px, border_px), (w - border_px, h - border_px), (255, 255, 255), max(1, border_px // 4))

    return result

sample_path = config.DDD_DATASET_DIR / "Drowsy" / "A0001.png"
raw = cv2.imread(str(sample_path))
frame = cv2.resize(raw, (640, 480))
bbox = (180, 140, 460, 420)

out_dir = config.PREDICTIONS_DIR / "wake_up_light_tests"
out_dir.mkdir(parents=True, exist_ok=True)

# Test peak strobe
peak_strobe = apply_emergency_wake_up_strobe(frame, bbox, intensity=1.0)
cv2.imwrite(str(out_dir / "wake_up_strobe_peak.png"), peak_strobe)

# Full HUD composition with multilingual banner
faces = [{"bbox": bbox}]
primary_face = faces[0]
eye_results = {"detected": True, "left_state": "CLOSED", "right_state": "CLOSED", "left_conf": 99.0, "right_conf": 98.0, "left_ear": 0.12, "right_ear": 0.13, "avg_ear": 0.125}
mouth_results = {"state": "YAWN", "confidence": 95.0, "mar": 0.55, "is_yawning": True}
decision_drowsy = {"state": "DROWSY", "drowsiness_score": 0.95, "temporal_metrics": {"closure_duration_sec": 3.0, "total_blinks": 6, "perclos": 0.45}}

full_comp = draw_master_hud(
    peak_strobe, faces, primary_face, eye_results, mouth_results,
    decision_drowsy, ("SOUNDING [ALL]", (0, 0, 255)), fps=29.0,
    language_mode="all"
)
cv2.imwrite(str(out_dir / "full_hud_with_wake_up_strobe.png"), full_comp)

print("Wake-up strobe images generated successfully!")
