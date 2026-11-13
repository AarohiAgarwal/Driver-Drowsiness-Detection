"""
Real-Time Eye Detection, EAR Calculation, and CNN Eye State Classification Module.

Webcam Pipeline:
Webcam -> Frame -> Face Detection -> Facial Landmarks -> Eye Regions (L/R)
-> Crop & Resize -> CNN Eye State Inference + EAR Calculation -> Real-Time HUD

Features:
- Extracts Left & Right Eye Regions of Interest (ROI) with aspect-ratio padding
- Computes Eye Aspect Ratio (EAR) as a supporting biometric feature
- Feeds cropped eye ROIs into the trained CNN model (best_eye_model.keras)
- Displays HUD: Face BBox, Eye States, Confidence, Left/Right/Avg EAR, FPS
- Handles: No face, Multiple faces, Face temporarily leaving frame
- Supports live webcam streaming as well as video/image verification modes
"""

import os
import sys
import time
import argparse
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
from src.face_detection import FaceDetector
from src.yawn_detection import YawnDetector

# Landmark Indices from MediaPipe FaceMesh
# Left Eye (from driver's perspective / viewer's right)
LEFT_EYE_LANDMARKS_6 = [33, 160, 158, 133, 153, 144] # p1, p2, p3, p4, p5, p6
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Right Eye (from driver's perspective / viewer's left)
RIGHT_EYE_LANDMARKS_6 = [362, 385, 387, 263, 373, 380] # p1, p2, p3, p4, p5, p6
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


def calculate_ear(eye_points: np.ndarray) -> float:
    """
    Calculates the Eye Aspect Ratio (EAR) based on 6 landmark coordinates:
    p1: Outer corner, p4: Inner corner
    p2, p6: Upper and lower vertical points
    p3, p5: Upper and lower vertical points
    
    Formula:
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2.0 * ||p1 - p4||)
    """
    if len(eye_points) < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_points[:6]

    # Vertical distances
    dist_v1 = np.linalg.norm(p2 - p6)
    dist_v2 = np.linalg.norm(p3 - p5)

    # Horizontal distance
    dist_h = np.linalg.norm(p1 - p4)

    if dist_h <= 1e-6:
        return 0.0

    ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return float(ear)


class EyeDetector:
    """
    Eye Region Extractor, EAR Evaluator, and Deep Learning Eye State Classifier.
    """
    def __init__(self,
                 model_path: Optional[Path] = None,
                 target_size: Tuple[int, int] = (64, 64),
                 padding_ratio: float = 0.30):
        self.target_size = target_size
        self.padding_ratio = padding_ratio

        if model_path is None:
            # Default to primary trained eye CNN
            model_path = config.MODELS_DIR / "eye_model" / "best_eye_model.keras"

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Eye classification model not found at {model_path}. "
                                    f"Run src/train_eye_model.py first!")

        print(f"Loading trained eye classification model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        print("Eye classification model loaded successfully.")

    def extract_eye_roi(self,
                        frame_bgr: np.ndarray,
                        landmarks_px: np.ndarray,
                        contour_indices: List[int]) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
        """
        Extracts a padded, square eye region of interest from the frame.
        Safely clamps bounding box to image boundaries.
        Returns: (cropped_bgr, (x1, y1, x2, y2))
        """
        h, w = frame_bgr.shape[:2]
        pts = landmarks_px[contour_indices]

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

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None, (0, 0, 0, 0)

        crop_bgr = frame_bgr[y1:y2, x1:x2].copy()
        return crop_bgr, (x1, y1, x2, y2)

    def classify_eye_crop(self, crop_bgr: np.ndarray) -> Tuple[str, float, float]:
        """
        Preprocesses eye crop and predicts eye state using the trained CNN.
        Returns:
        - state: 'OPEN' or 'CLOSED'
        - confidence: Percentage (0.0 to 100.0)
        - prob_closed: Raw sigmoid probability P(Closed)
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return "UNKNOWN", 0.0, 0.0

        # Convert to RGB, resize, normalize to [0, 1]
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        tensor_in = np.expand_dims(normalized, axis=0)

        # Run inference
        pred_tensor = self.model(tensor_in, training=False)
        if hasattr(pred_tensor, "detach"):
            prob_closed = float(pred_tensor.detach().cpu().numpy()[0, 0])
        elif hasattr(pred_tensor, "numpy"):
            prob_closed = float(pred_tensor.numpy()[0, 0])
        else:
            prob_closed = float(pred_tensor[0, 0])

        if prob_closed >= 0.5:
            state = "CLOSED"
            confidence = prob_closed * 100.0
        else:
            state = "OPEN"
            confidence = (1.0 - prob_closed) * 100.0

        return state, confidence, prob_closed

    def process_face_eyes(self,
                          frame_bgr: np.ndarray,
                          face_info: Optional[Dict]) -> Dict:
        """
        Complete processing on a detected face:
        1. Calculates Left EAR, Right EAR, and Average EAR.
        2. Crops left and right eye ROIs.
        3. Predicts eye state using the CNN.
        """
        if face_info is None:
            return {
                "detected": False,
                "left_state": "NONE",
                "left_conf": 0.0,
                "right_state": "NONE",
                "right_conf": 0.0,
                "left_ear": 0.0,
                "right_ear": 0.0,
                "avg_ear": 0.0,
                "left_bbox": None,
                "right_bbox": None,
                "both_closed": False,
            }

        landmarks_px = face_info["landmarks_px"]

        # 1. EAR Calculation
        left_ear_pts = landmarks_px[LEFT_EYE_LANDMARKS_6]
        right_ear_pts = landmarks_px[RIGHT_EYE_LANDMARKS_6]

        left_ear = calculate_ear(left_ear_pts)
        right_ear = calculate_ear(right_ear_pts)
        avg_ear = (left_ear + right_ear) / 2.0

        # 2. Extract Eye ROIs
        left_crop, left_box = self.extract_eye_roi(frame_bgr, landmarks_px, LEFT_EYE_CONTOUR)
        right_crop, right_box = self.extract_eye_roi(frame_bgr, landmarks_px, RIGHT_EYE_CONTOUR)

        # 3. CNN Classification
        l_state_cnn, l_conf_cnn, l_prob_cnn = self.classify_eye_crop(left_crop)
        r_state_cnn, r_conf_cnn, r_prob_cnn = self.classify_eye_crop(right_crop)

        # 4. Multi-Signal Fusion (Biometric EAR + Deep Learning Eye CNN)
        ear_thresh = getattr(config, "EAR_CLOSED_THRESHOLD", 0.19)

        # Normalized EAR closure metrics: 0.0 (wide open) to 1.0 (fully closed)
        left_ear_closure = float(np.clip((0.26 - left_ear) / (0.26 - 0.14), 0.0, 1.0))
        right_ear_closure = float(np.clip((0.26 - right_ear) / (0.26 - 0.14), 0.0, 1.0))

        # Fused closure probability (combines CNN prediction with physical eye aperture)
        left_prob = float(np.clip(0.50 * l_prob_cnn + 0.50 * left_ear_closure, 0.0, 1.0))
        right_prob = float(np.clip(0.50 * r_prob_cnn + 0.50 * right_ear_closure, 0.0, 1.0))

        # Individual eye state decision (closed if EAR < threshold OR CNN says closed OR fused prob >= 0.5)
        left_closed = (left_ear < ear_thresh) or (l_prob_cnn >= 0.5) or (left_prob >= 0.5)
        right_closed = (right_ear < ear_thresh) or (r_prob_cnn >= 0.5) or (right_prob >= 0.5)

        l_state = "CLOSED" if left_closed else "OPEN"
        r_state = "CLOSED" if right_closed else "OPEN"

        # Fused confidence calibration
        l_conf = (left_prob * 100.0) if left_closed else ((1.0 - left_prob) * 100.0)
        r_conf = (right_prob * 100.0) if right_closed else ((1.0 - right_prob) * 100.0)

        # Both eyes closed: both individual eyes closed OR average EAR below threshold
        both_closed = (left_closed and right_closed) or (avg_ear < ear_thresh)

        return {
            "detected": True,
            "left_state": l_state,
            "left_conf": l_conf,
            "left_prob": left_prob,
            "right_state": r_state,
            "right_conf": r_conf,
            "right_prob": right_prob,
            "left_ear": left_ear,
            "right_ear": right_ear,
            "avg_ear": avg_ear,
            "left_bbox": left_box,
            "right_bbox": right_box,
            "left_ear_pts": left_ear_pts,
            "right_ear_pts": right_ear_pts,
            "both_closed": both_closed,
        }

    def render_hud(self,
                   frame: np.ndarray,
                   eye_results: Dict,
                   mouth_results: Optional[Dict] = None,
                   fps: float = 0.0) -> np.ndarray:
        """
        Renders HUD telemetry on the video frame:
        - Eye bounding boxes & states (OPEN / CLOSED)
        - Mouth bounding box & state (YAWN / NO_YAWN)
        - EAR values (Left, Right, Avg) & MAR value
        - Eye closure and yawn alerts
        - FPS counter
        """
        hud_frame = frame.copy()
        h, w = hud_frame.shape[:2]

        # Semi-transparent top dashboard banner (expanded to 115px for mouth/yawn telemetry)
        banner_h = 115
        overlay = hud_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, hud_frame, 0.25, 0, hud_frame)

        # Title and FPS
        cv2.putText(hud_frame, "DRIVER MONITORING SYSTEM | REAL-TIME VISION", (15, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hud_frame, f"FPS: {fps:.1f}", (w - 110, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        if not eye_results.get("detected", False):
            cv2.putText(hud_frame, "STATUS: SEARCHING FOR DRIVER FACE...", (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
            return hud_frame

        # Draw Eye Bounding Boxes
        l_color = (0, 0, 255) if eye_results["left_state"] == "CLOSED" else (0, 255, 0)
        r_color = (0, 0, 255) if eye_results["right_state"] == "CLOSED" else (0, 255, 0)

        if eye_results["left_bbox"]:
            lx1, ly1, lx2, ly2 = eye_results["left_bbox"]
            cv2.rectangle(hud_frame, (lx1, ly1), (lx2, ly2), l_color, 2)
            cv2.putText(hud_frame, f"L: {eye_results['left_state']}", (lx1, max(15, ly1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, l_color, 1, cv2.LINE_AA)

        if eye_results["right_bbox"]:
            rx1, ry1, rx2, ry2 = eye_results["right_bbox"]
            cv2.rectangle(hud_frame, (rx1, ry1), (rx2, ry2), r_color, 2)
            cv2.putText(hud_frame, f"R: {eye_results['right_state']}", (rx1, max(15, ry1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, r_color, 1, cv2.LINE_AA)

        # Draw EAR Landmark points
        if "left_ear_pts" in eye_results and eye_results["left_ear_pts"] is not None:
            for pt in eye_results["left_ear_pts"]:
                cv2.circle(hud_frame, (int(pt[0]), int(pt[1])), 2, (255, 255, 0), -1)
        if "right_ear_pts" in eye_results and eye_results["right_ear_pts"] is not None:
            for pt in eye_results["right_ear_pts"]:
                cv2.circle(hud_frame, (int(pt[0]), int(pt[1])), 2, (255, 255, 0), -1)

        # Telemetry: CNN Eye States & Confidence (Left Column)
        l_text = f"LEFT EYE:  {eye_results['left_state']} ({eye_results['left_conf']:.1f}%)"
        r_text = f"RIGHT EYE: {eye_results['right_state']} ({eye_results['right_conf']:.1f}%)"
        cv2.putText(hud_frame, l_text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, l_color, 2, cv2.LINE_AA)
        cv2.putText(hud_frame, r_text, (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.52, r_color, 2, cv2.LINE_AA)

        # Telemetry: Yawn State & Confidence (Right Column)
        if mouth_results and mouth_results.get("detected", False):
            m_state = mouth_results["state"]
            m_conf = mouth_results["confidence"]
            m_mar = mouth_results["mar"]
            m_color = (0, 140, 255) if mouth_results["is_yawning"] else (0, 220, 0)
            m_text = f"MOUTH: {m_state} ({m_conf:.1f}%)"
            mar_text = f"MAR:   {m_mar:.3f}"
            cv2.putText(hud_frame, m_text, (w // 2 + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, m_color, 2, cv2.LINE_AA)
            cv2.putText(hud_frame, mar_text, (w // 2 + 10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

        # Telemetry: EAR Values (Bottom Row of Banner)
        ear_text = (f"Left EAR: {eye_results['left_ear']:.3f}  |  "
                    f"Right EAR: {eye_results['right_ear']:.3f}  |  "
                    f"Average EAR: {eye_results['avg_ear']:.3f}")
        cv2.putText(hud_frame, ear_text, (15, 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        # Drowsiness Warnings
        is_eyes_closed = eye_results.get("both_closed", False)
        is_yawning = mouth_results.get("is_yawning", False) if mouth_results else False

        if is_eyes_closed or is_yawning:
            warn_overlay = hud_frame.copy()
            cv2.rectangle(warn_overlay, (w // 2 - 210, h - 60), (w // 2 + 210, h - 15), (0, 0, 190), -1)
            cv2.addWeighted(warn_overlay, 0.8, hud_frame, 0.2, 0, hud_frame)

            if is_eyes_closed and is_yawning:
                warn_msg = "WARNING: EYES CLOSED & YAWNING!"
            elif is_eyes_closed:
                warn_msg = "WARNING: EYES CLOSED!"
            else:
                warn_msg = "WARNING: YAWN DETECTED!"

            cv2.putText(hud_frame, warn_msg, (w // 2 - 185, h - 28),
                        cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

        return hud_frame


# ==============================================================================
# PIPELINE RUNNER & VERIFICATION
# ==============================================================================

def run_webcam_pipeline(camera_index: int = 0,
                        model_path: Optional[Path] = None,
                        yawn_model_path: Optional[Path] = None,
                        test_image_mode: Optional[Path] = None):
    """
    Executes real-time webcam facial, eye, and mouth/yawn detection pipeline.
    If test_image_mode is provided, verifies the pipeline on a static image/video sequence.
    """
    face_detector = FaceDetector()
    eye_detector = EyeDetector(model_path=model_path, target_size=(64, 64))
    yawn_detector = YawnDetector(model_path=yawn_model_path, target_size=(64, 64))

    print("\n" + "=" * 65)
    print("  LAUNCHING REAL-TIME FACE, EYE & YAWN DETECTION PIPELINE")
    print("=" * 65)
    print("Pipeline Components Active:")
    print("  1. MediaPipe FaceLandmarker (478 3D Landmarks)")
    print("  2. Driver Tracking (Multi-face prioritization & edge-case recovery)")
    print("  3. Dual Eye ROI Cropping & Dynamic Clamping")
    print("  4. Eye Aspect Ratio (EAR) Supporting Biometrics")
    print("  5. CNN Deep Learning Eye State Classifier (best_eye_model.keras)")
    print(f"  6. Yawn Detection Module (Enabled: {yawn_detector.is_available})")
    print("=" * 65 + "\n")

    if test_image_mode:
        print(f"Running verification mode on test image: {test_image_mode}")
        frame = cv2.imread(str(test_image_mode))
        if frame is None:
            raise ValueError(f"Unable to read test image at: {test_image_mode}")

        faces = face_detector.detect_faces(frame)
        primary_face = face_detector.get_primary_face(faces, frame.shape)
        annotated_frame = face_detector.draw_face_annotations(frame, primary_face, len(faces))
        eye_results = eye_detector.process_face_eyes(frame, primary_face)
        mouth_results = yawn_detector.process_face_mouth(frame, primary_face)

        annotated_frame = yawn_detector.draw_mouth_annotations(annotated_frame, mouth_results)
        hud_frame = eye_detector.render_hud(annotated_frame, eye_results, mouth_results=mouth_results, fps=30.0)

        out_path = config.PREDICTIONS_DIR / "webcam_pipeline_verification.png"
        cv2.imwrite(str(out_path), hud_frame)
        print(f"Pipeline verification successful! Verified output saved to: {out_path}")
        print("\nVerification Telemetry:")
        print(f"  • Faces Detected:   {len(faces)}")
        print(f"  • Left Eye State:    {eye_results['left_state']} ({eye_results['left_conf']:.1f}%)")
        print(f"  • Right Eye State:   {eye_results['right_state']} ({eye_results['right_conf']:.1f}%)")
        print(f"  • Left EAR:          {eye_results['left_ear']:.4f}")
        print(f"  • Right EAR:         {eye_results['right_ear']:.4f}")
        print(f"  • Average EAR:       {eye_results['avg_ear']:.4f}")
        print(f"  • Both Eyes Closed:  {eye_results['both_closed']}")
        print(f"  • Mouth State:       {mouth_results['state']} ({mouth_results['confidence']:.1f}%)")
        print(f"  • Mouth Aspect Ratio:{mouth_results['mar']:.4f}")
        print(f"  • Is Yawning:        {mouth_results['is_yawning']}")
        return

    # Real Webcam Loop
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Warning: Could not open webcam at index {camera_index}.")
        print("Falling back to dataset verification sample...")
        sample_img = config.DDD_DATASET_DIR / "Drowsy" / "A0001.png"
        run_webcam_pipeline(test_image_mode=sample_img, model_path=model_path, yawn_model_path=yawn_model_path)
        return

    print("Webcam stream started. Press 'q' or ESC in the video window to exit.")
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            curr_time = time.time()
            fps = 1.0 / max(1e-4, curr_time - prev_time)
            prev_time = curr_time

            # 1. Face Detection & Tracking
            faces = face_detector.detect_faces(frame)
            primary_face = face_detector.get_primary_face(faces, frame.shape)

            # 2. Eye Extraction, EAR & CNN Classification
            eye_results = eye_detector.process_face_eyes(frame, primary_face)

            # 3. Mouth Extraction, MAR & Yawn CNN Classification
            mouth_results = yawn_detector.process_face_mouth(frame, primary_face)

            # 4. Annotations & HUD Rendering
            is_drowsy = eye_results["both_closed"] or mouth_results["is_yawning"]
            annotated_frame = face_detector.draw_face_annotations(
                frame, primary_face, len(faces), is_drowsy=is_drowsy
            )
            annotated_frame = yawn_detector.draw_mouth_annotations(annotated_frame, mouth_results)
            hud_frame = eye_detector.render_hud(annotated_frame, eye_results, mouth_results=mouth_results, fps=fps)

            cv2.imshow("Driver Drowsiness Detection - Real-Time Eye & Yawn Pipeline", hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam pipeline terminated cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Face, Eye & Yawn Detection Pipeline")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Webcam device index")
    parser.add_argument("--eye-model", type=str, default=str(config.MODELS_DIR / "eye_model" / "best_eye_model.keras"),
                        help="Path to trained eye CNN model")
    parser.add_argument("--yawn-model", type=str, default=str(config.MODELS_DIR / "yawn_model" / "best_yawn_model.keras"),
                        help="Path to trained yawn CNN model")
    parser.add_argument("--verify", action="store_true", help="Run verification mode on a sample dataset image")
    parser.add_argument("--image", type=str, default=None, help="Path to image file for verification")
    args = parser.parse_args()

    test_image = None
    if args.verify:
        test_image = Path(args.image) if args.image else (config.DDD_DATASET_DIR / "Drowsy" / "A0001.png")

    run_webcam_pipeline(
        camera_index=args.camera,
        model_path=Path(args.eye_model),
        yawn_model_path=Path(args.yawn_model),
        test_image_mode=test_image,
    )


if __name__ == "__main__":
    main()
