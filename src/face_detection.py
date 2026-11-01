"""
Face Detection and Facial Landmark Extraction Module.

Uses MediaPipe FaceLandmarker (0.10+ / 1.x Tasks API) to detect faces,
extract 478 3D facial landmarks, and calculate bounding boxes.

Handles:
- No face detected (graceful fallback)
- Multiple faces (identifies and tracks primary driver face)
- Face temporarily leaving frame (robust tracking state)
"""

import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"


def ensure_face_landmarker_model() -> Path:
    """
    Ensures that the MediaPipe face_landmarker.task model file is available locally.
    Downloads it from Google's official repository if missing.
    """
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        print(f"Downloading MediaPipe FaceLandmarker task asset to {MODEL_PATH}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Download complete.")
        except Exception as e:
            raise RuntimeError(f"Failed to download FaceLandmarker model: {e}")
    return MODEL_PATH


class FaceDetector:
    """
    Real-time face detection and 478-point landmark extractor using MediaPipe.
    """
    def __init__(self,
                 model_path: Optional[Path] = None,
                 max_num_faces: int = 3,
                 min_detection_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.model_path = model_path if model_path else ensure_face_landmarker_model()
        self.max_num_faces = max_num_faces

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=self.max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect_faces(self, frame_bgr: np.ndarray) -> List[Dict]:
        """
        Detects faces in a BGR frame and returns a list of detected face dictionaries.
        
        Each dictionary contains:
        - 'landmarks': 478 normalized landmarks (x, y, z)
        - 'landmarks_px': (478, 2) numpy array of pixel coordinates (x, y)
        - 'bbox': (x1, y1, x2, y2) bounding box enclosing the face
        - 'area': Bounding box pixel area (for primary face sorting)
        - 'center': (cx, cy) face center in pixel coordinates
        
        Returns empty list [] if no face is detected.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            result = self.detector.detect(mp_image)
        except Exception as e:
            # Prevent crashes on corrupted/dropped frames
            return []

        if not result.face_landmarks:
            return []

        detected_faces = []
        for face_lms in result.face_landmarks:
            # Convert normalized landmarks to pixel coordinates
            coords_px = np.array([[lm.x * w, lm.y * h] for lm in face_lms], dtype=np.float32)

            # Compute bounding box enclosing the face landmarks
            min_x = int(max(0, np.min(coords_px[:, 0])))
            min_y = int(max(0, np.min(coords_px[:, 1])))
            max_x = int(min(w - 1, np.max(coords_px[:, 0])))
            max_y = int(min(h - 1, np.max(coords_px[:, 1])))

            # Add 10% breathing room to bounding box
            bw = max_x - min_x
            bh = max_y - min_y
            x1 = int(max(0, min_x - bw * 0.05))
            y1 = int(max(0, min_y - bh * 0.15)) # extra headroom
            x2 = int(min(w - 1, max_x + bw * 0.05))
            y2 = int(min(h - 1, max_y + bh * 0.05))

            area = (x2 - x1) * (y2 - y1)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            detected_faces.append({
                "landmarks": face_lms,
                "landmarks_px": coords_px,
                "bbox": (x1, y1, x2, y2),
                "area": area,
                "center": center,
            })

        return detected_faces

    def get_primary_face(self, faces: List[Dict], frame_shape: Tuple[int, int]) -> Optional[Dict]:
        """
        Selects the primary driver face from multiple detected faces.
        Strategy: Prioritizes the largest face closest to the center of the frame.
        Handles:
        - 0 faces: returns None
        - 1 face: returns the face directly
        - >1 faces: returns the largest, centrally positioned driver face
        """
        if not faces:
            return None
        if len(faces) == 1:
            return faces[0]

        h, w = frame_shape[:2]
        frame_center = np.array([w / 2.0, h / 2.0])

        best_score = -float("inf")
        best_face = faces[0]

        for face in faces:
            # Area normalized
            norm_area = face["area"] / float(w * h)
            # Distance from center normalized
            dist = np.linalg.norm(np.array(face["center"]) - frame_center) / np.linalg.norm(frame_center)
            # Score: maximize area while penalizing extreme peripheral offset
            score = norm_area * 2.0 - dist * 0.5

            if score > best_score:
                best_score = score
                best_face = face

        return best_face

    def apply_face_red_light_flash(self,
                                   frame: np.ndarray,
                                   bbox: Tuple[int, int, int, int],
                                   current_time: Optional[float] = None,
                                   max_intensity: float = 1.0,
                                   frequency_hz: float = 3.5,
                                   fullscreen_burst: bool = True) -> np.ndarray:
        """
        Applies a high-intensity pulsating emergency red light strobe directly to wake up a sleeping driver:
        - Cycles at ~3.5 Hz alerting strobe rate to physiologically stimulate photoreceptors through closed eyelids.
        - Fullscreen Red Illumination Burst: Rapidly pumps maximum crimson red lumens from the monitor into the vehicle cabin.
        - Facial Spotlight Flood: Projects a wide, hyper-saturated red light wash centered on the driver's face.
        - Dual-Band Perimeter Strobe: 14px outer red and white flashing borders stimulate peripheral vision.
        - Emergency Beacon Tag: [!] WAKE UP! RED LIGHT FLASH.
        """
        if bbox is None:
            return frame

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return frame

        # Compute dynamic strobe pulse: sharp alert flash (0.0 to 1.0)
        t = current_time if current_time is not None else time.time()
        pulse = (np.sin(t * 2.0 * np.pi * frequency_hz) + 1.0) / 2.0
        intensity = float(np.power(pulse, 1.3)) * max_intensity

        if intensity < 0.04:
            return frame

        result = frame.copy().astype(np.float32)

        # 1. Fullscreen Red Illumination Burst (Screen acts as physical strobe lamp)
        if fullscreen_burst:
            result[:, :, 2] += 160.0 * intensity * 0.75             # Red blast
            result[:, :, 1] = result[:, :, 1] * (1.0 - 0.40 * intensity)  # Suppress green
            result[:, :, 0] = result[:, :, 0] * (1.0 - 0.50 * intensity)  # Suppress blue
            np.clip(result, 0, 255, out=result)

        result = result.astype(np.uint8)

        # 2. Targeted High-Power Face Spotlight Flood
        fw = x2 - x1
        fh = y2 - y1
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Elliptical spotlight mask on face
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (center_x, center_y), (int(fw * 0.70), int(fh * 0.85)), 0, 0, 360, 1.0, -1)
        ksize = max(15, (int(min(fw, fh) * 0.25) // 2) * 2 + 1)
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        # Super-saturated face spotlight
        face_flood = result.copy().astype(np.float32)
        face_flood[:, :, 2] = np.clip(face_flood[:, :, 2] + 90.0 * intensity, 0, 255)
        alpha_face = np.expand_dims(mask * (intensity * 0.80), axis=2)
        result = (result.astype(np.float32) * (1.0 - alpha_face) + face_flood * alpha_face).astype(np.uint8)

        # 3. Flashing face beacon box & header tag
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 4)
        tag_h = 32
        tag_w = min(w - x1 - 10, 290)
        cv2.rectangle(result, (x1, max(0, y1 - tag_h)), (x1 + tag_w, y1), (0, 0, 255), -1)
        cv2.putText(result, "[!] WAKE UP! RED LIGHT FLASH", (x1 + 8, max(12, y1 - 9)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

        # 4. Massive full-screen emergency perimeter strobe (dual-band red + white)
        border_px = int(14 * intensity)
        if border_px > 0:
            cv2.rectangle(result, (0, 0), (w, h), (0, 0, 255), border_px)
            inner_w = max(1, border_px // 4)
            cv2.rectangle(result, (border_px, border_px), (w - border_px, h - border_px), (255, 255, 255), inner_w)

        return result

    def draw_face_annotations(self,
                              frame: np.ndarray,
                              face_info: Optional[Dict],
                              num_total_faces: int = 1,
                              is_drowsy: bool = False,
                              current_time: Optional[float] = None) -> np.ndarray:
        """
        Draws visual annotations (bounding box, driver tracking tag, face count) on the frame.
        When is_drowsy is True, projects the high-intensity emergency wake-up red light strobe onto the frame and driver's face.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        if face_info is None:
            # Visual indicator when face is lost / leaving frame
            cv2.rectangle(annotated, (20, 20), (320, 65), (0, 0, 0), -1)
            cv2.putText(annotated, "NO FACE DETECTED", (30, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            return annotated

        x1, y1, x2, y2 = face_info["bbox"]

        # When drowsy alert is triggered: apply high-power emergency red light wake-up strobe!
        if is_drowsy:
            enabled = getattr(config, "ENABLE_FACE_RED_LIGHT_FLASH", True)
            if enabled:
                max_int = getattr(config, "RED_LIGHT_INTENSITY", 1.0)
                freq = getattr(config, "RED_LIGHT_FLASH_FREQ_HZ", 3.5)
                full_burst = getattr(config, "RED_LIGHT_FULLSCREEN_BURST", True)
                annotated = self.apply_face_red_light_flash(
                    annotated, (x1, y1, x2, y2),
                    current_time=current_time,
                    max_intensity=max_int,
                    frequency_hz=freq,
                    fullscreen_burst=full_burst
                )
            return annotated

        box_color = (0, 220, 0)

        # Draw smooth rounded bounding corners
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # Tag banner for normal state
        tag_text = "DRIVER (TRACKED)" if num_total_faces > 1 else "DRIVER"
        cv2.rectangle(annotated, (x1, max(0, y1 - 25)), (x1 + 180, y1), box_color, -1)
        cv2.putText(annotated, tag_text, (x1 + 8, max(12, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        if num_total_faces > 1:
            multi_text = f"Multiple Faces ({num_total_faces}) - Tracking Driver"
            cv2.putText(annotated, multi_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        return annotated
