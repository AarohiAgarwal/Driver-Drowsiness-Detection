import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

def apply_face_red_light_flash(frame: np.ndarray, bbox: tuple, intensity: float = 0.75) -> np.ndarray:
    """
    Applies an intense emergency red strobe flash directly onto the driver's face:
    - Elliptical radial red wash simulating cabin emergency beacon light.
    - Boosts red channel while preserving facial features and eye visibility.
    - Pulsating outer boundary beacon ring.
    """
    if bbox is None or intensity <= 0.01:
        return frame

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame

    result = frame.copy()
    fw = x2 - x1
    fh = y2 - y1

    # 1. Create elliptical radial gradient mask over the face
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    axes = (int(fw * 0.65), int(fh * 0.75))

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, 1.0, -1)
    # Gaussian blur for soft natural spotlight/strobe falloff
    ksize = max(15, (int(min(fw, fh) * 0.25) // 2) * 2 + 1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    # 2. Compute red strobe illumination
    # BGR format: boost Red (channel 2), add slight warm boost, clip safely
    red_layer = result.copy().astype(np.float32)
    red_layer[:, :, 2] += 160.0 * intensity   # Intense Red Channel
    red_layer[:, :, 1] -= 30.0 * intensity    # Reduce Green
    red_layer[:, :, 0] -= 40.0 * intensity    # Reduce Blue
    np.clip(red_layer, 0, 255, out=red_layer)

    # 3. Blend red layer onto face using radial mask * intensity
    alpha = np.expand_dims(mask * (intensity * 0.85), axis=2)
    result = (result.astype(np.float32) * (1.0 - alpha) + red_layer * alpha).astype(np.uint8)

    # 4. Emergency pulsating face beacon box
    box_alpha = min(1.0, intensity * 1.2)
    overlay = result.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 4)
    # Flashing tag
    tag_h = 28
    cv2.rectangle(overlay, (x1, max(0, y1 - tag_h)), (x1 + 240, y1), (0, 0, 255), -1)
    cv2.putText(overlay, "[!] RED FLASH ALERT", (x1 + 6, max(12, y1 - 8)),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.addWeighted(overlay, box_alpha, result, 1.0 - box_alpha, 0, result)

    # 5. Screen peripheral red strobe vignette border
    border_thick = int(8 * intensity)
    if border_thick > 0:
        cv2.rectangle(result, (0, 0), (w, h), (0, 0, 255), border_thick)

    return result

# Test on sample
sample_path = config.DDD_DATASET_DIR / "Drowsy" / "A0001.png"
raw = cv2.imread(str(sample_path))
frame = cv2.resize(raw, (640, 480))
bbox = (180, 140, 460, 420)

out_dir = config.PREDICTIONS_DIR / "red_flash_tests"
out_dir.mkdir(parents=True, exist_ok=True)

# Generate frames at various pulse intensities (peak, mid, low)
for intensity, name in [(1.0, "peak"), (0.5, "mid"), (0.0, "off")]:
    flashed = apply_face_red_light_flash(frame, bbox, intensity=intensity)
    out_file = out_dir / f"face_red_flash_{name}.png"
    cv2.imwrite(str(out_file), flashed)
    print(f"Saved: {out_file.name}")

print("Red flash tests generated!")
