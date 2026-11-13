"""
Real-Time Yawn Detection, Mouth Aspect Ratio (MAR) Calculation, and CNN Yawn Classification Module.

Webcam Pipeline:
Face -> Facial Landmarks -> Mouth Region Extraction -> Crop & Resize -> Yawn CNN Prediction -> Real-Time HUD

Features:
- Extracts Mouth Region of Interest (ROI) from MediaPipe facial landmarks with dynamic padding
- Computes Mouth Aspect Ratio (MAR) as a supporting biometric indicator
- Integrates the trained Yawn CNN (best_yawn_model.keras)
- Fully optional: gracefully falls back if yawn model is missing or disabled
- Renders mouth bounding box, MAR, and Yawn/No-Yawn state
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional, List

# Set Keras backend to torch
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import numpy as np
import keras

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# MediaPipe Mouth Landmarks
# Lip corners
MOUTH_CORNER_LEFT = 61
MOUTH_CORNER_RIGHT = 291

# Inner lip vertical points (for MAR)
MOUTH_INNER_TOP = 13
MOUTH_INNER_BOTTOM = 14

# Outer lip vertical points
MOUTH_OUTER_TOP = 0
MOUTH_OUTER_BOTTOM = 17

# Full outer mouth contour
MOUTH_CONTOUR = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
MOUTH_MAR_POINTS = [MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT, MOUTH_INNER_TOP, MOUTH_INNER_BOTTOM, MOUTH_OUTER_TOP, MOUTH_OUTER_BOTTOM]


def calculate_mar(landmarks_px: np.ndarray) -> float:
    """
    Calculates Mouth Aspect Ratio (MAR) based on inner and outer lip distances.
    
    Formula:
    MAR = (||p_inner_top - p_inner_bottom|| + 0.5 * ||p_outer_top - p_outer_bottom||) / (1.5 * ||p_left - p_right||)
    """
    if landmarks_px is None or len(landmarks_px) < 300:
        return 0.0

    p_left = landmarks_px[MOUTH_CORNER_LEFT]
    p_right = landmarks_px[MOUTH_CORNER_RIGHT]
    p_in_top = landmarks_px[MOUTH_INNER_TOP]
    p_in_bottom = landmarks_px[MOUTH_INNER_BOTTOM]
    p_out_top = landmarks_px[MOUTH_OUTER_TOP]
    p_out_bottom = landmarks_px[MOUTH_OUTER_BOTTOM]

    dist_h = np.linalg.norm(p_left - p_right)
    if dist_h <= 1e-6:
        return 0.0

    dist_v_in = np.linalg.norm(p_in_top - p_in_bottom)
    dist_v_out = np.linalg.norm(p_out_top - p_out_bottom)

    # Weighted MAR considering both inner aperture and outer jaw drop
    mar = (dist_v_in + 0.5 * dist_v_out) / (1.5 * dist_h)
    return float(mar)


class YawnDetector:
    """
    Mouth Region Extractor, MAR Evaluator, and Deep Learning Yawn Classifier.
    """
    def __init__(self,
                 model_path: Optional[Path] = None,
                 target_size: Tuple[int, int] = (64, 64),
                 padding_ratio: float = 0.35):
        self.target_size = target_size
        self.padding_ratio = padding_ratio
        self.is_available = False
        self.model = None

        if model_path is None:
            model_path = config.MODELS_DIR / "yawn_model" / "best_yawn_model.keras"
            if not model_path.exists():
                alt_path = config.MODELS_DIR / "yawn_model" / "yawn_model.keras"
                if alt_path.exists():
                    model_path = alt_path
        else:
            model_path = Path(model_path)

        if model_path.exists():
            try:
                print(f"Loading trained yawn classification model from: {model_path}")
                self.model = keras.models.load_model(model_path)
                self.is_available = True
                print("Yawn classification model loaded successfully.")
            except Exception as e:
                print(f"Warning: Could not load yawn model from {model_path}: {e}")
                print("Yawn module will operate in MAR biometric fallback mode.")
                self.is_available = False
        else:
            print(f"Notice: Yawn model not found at {model_path}. Yawn module disabled or running in MAR-only mode.")
            self.is_available = False

    def extract_mouth_roi(self,
                          frame_bgr: np.ndarray,
                          landmarks_px: np.ndarray) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
        """
        Extracts a padded, square mouth region of interest from the frame.
        Safely clamps coordinates to image dimensions.
        Returns: (cropped_bgr, (x1, y1, x2, y2))
        """
        if frame_bgr is None or landmarks_px is None:
            return None, (0, 0, 0, 0)

        h, w = frame_bgr.shape[:2]
        pts = landmarks_px[MOUTH_CONTOUR]

        min_x = np.min(pts[:, 0])
        max_x = np.max(pts[:, 0])
        min_y = np.min(pts[:, 1])
        max_y = np.max(pts[:, 1])

        bw = max_x - min_x
        bh = max_y - min_y
        side = max(bw, bh) * (1.0 + self.padding_ratio)

        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0

        x1 = int(max(0, cx - side / 2.0))
        y1 = int(max(0, cy - side / 2.0))
        x2 = int(min(w, cx + side / 2.0))
        y2 = int(min(h, cy + side / 2.0))

        if (x2 - x1) < 15 or (y2 - y1) < 15:
            return None, (0, 0, 0, 0)

        crop_bgr = frame_bgr[y1:y2, x1:x2].copy()
        return crop_bgr, (x1, y1, x2, y2)

    def classify_mouth_crop(self, crop_bgr: np.ndarray) -> Tuple[str, float, float]:
        """
        Preprocesses mouth crop and predicts yawn state using the trained CNN.
        Returns:
        - state: 'YAWN' or 'NO_YAWN'
        - confidence: Percentage (0.0 to 100.0)
        - prob_yawn: Raw probability P(Yawn)
        """
        if not self.is_available or self.model is None or crop_bgr is None or crop_bgr.size == 0:
            return "UNKNOWN", 0.0, 0.0

        # Convert to RGB, resize, normalize to [0, 1]
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        tensor_in = np.expand_dims(normalized, axis=0)

        # Run inference
        pred_tensor = self.model(tensor_in, training=False)
        if hasattr(pred_tensor, "detach"):
            prob_yawn = float(pred_tensor.detach().cpu().numpy()[0, 0])
        elif hasattr(pred_tensor, "numpy"):
            prob_yawn = float(pred_tensor.numpy()[0, 0])
        else:
            prob_yawn = float(pred_tensor[0, 0])

        if prob_yawn >= 0.5:
            state = "YAWN"
            confidence = prob_yawn * 100.0
        else:
            state = "NO_YAWN"
            confidence = (1.0 - prob_yawn) * 100.0

        return state, confidence, prob_yawn

    def process_face_mouth(self,
                           frame_bgr: np.ndarray,
                           face_info: Optional[Dict]) -> Dict:
        """
        Processes mouth region on a detected face:
        1. Computes Mouth Aspect Ratio (MAR)
        2. Extracts mouth ROI
        3. Predicts YAWN / NO_YAWN using trained CNN
        """
        if face_info is None or not face_info.get("landmarks_px", None) is not None:
            return {
                "detected": False,
                "state": "NONE",
                "confidence": 0.0,
                "prob_yawn": 0.0,
                "mar": 0.0,
                "mouth_bbox": None,
                "is_yawning": False,
                "is_available": self.is_available,
            }

        landmarks_px = face_info["landmarks_px"]

        # 1. MAR Biometric Calculation
        mar = calculate_mar(landmarks_px)

        # 2. Extract Mouth ROI
        mouth_crop, mouth_bbox = self.extract_mouth_roi(frame_bgr, landmarks_px)

        # 3. Yawn CNN Classification & MAR Multi-Signal Fusion
        if self.is_available and mouth_crop is not None:
            cnn_state, cnn_conf, prob_yawn = self.classify_mouth_crop(mouth_crop)
        else:
            cnn_state = "NO_YAWN"
            cnn_conf = 0.0
            prob_yawn = 0.0

        mar_thresh = getattr(config, "MAR_YAWN_THRESHOLD", 0.40)
        mar_score = float(np.clip((mar - 0.20) / (0.45 - 0.20), 0.0, 1.0))
        fused_yawn_prob = max(prob_yawn, mar_score)

        is_yawning = (cnn_state == "YAWN") or (mar >= mar_thresh) or (fused_yawn_prob >= 0.50)
        state = "YAWN" if is_yawning else "NO_YAWN"
        conf = (fused_yawn_prob * 100.0) if is_yawning else ((1.0 - fused_yawn_prob) * 100.0)

        return {
            "detected": True,
            "state": state,
            "confidence": conf,
            "prob_yawn": fused_yawn_prob,
            "mar": mar,
            "mouth_bbox": mouth_bbox,
            "is_yawning": is_yawning,
            "is_available": self.is_available,
            "mar_pts": landmarks_px[MOUTH_MAR_POINTS] if len(landmarks_px) > 300 else None,
        }

    def draw_mouth_annotations(self,
                              frame: np.ndarray,
                              mouth_results: Dict) -> np.ndarray:
        """
        Draws visual annotations (mouth bounding box, state label, MAR points) on the frame.
        """
        if not mouth_results.get("detected", False) or mouth_results.get("mouth_bbox") is None:
            return frame

        annotated = frame.copy()
        x1, y1, x2, y2 = mouth_results["mouth_bbox"]
        is_yawn = mouth_results["is_yawning"]
        state = mouth_results["state"]
        conf = mouth_results["confidence"]
        mar = mouth_results["mar"]

        color = (0, 140, 255) if is_yawn else (0, 200, 0) # Orange for Yawn, Green for Normal

        # Draw mouth bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label tag above mouth box
        tag_text = f"Mouth: {state} ({conf:.0f}%)"
        cv2.putText(annotated, tag_text, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Draw MAR keypoints
        if mouth_results.get("mar_pts") is not None:
            for pt in mouth_results["mar_pts"]:
                cv2.circle(annotated, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

        return annotated
