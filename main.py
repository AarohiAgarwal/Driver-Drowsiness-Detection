"""
Real-Time Driver Drowsiness Detection and Alert System.

Primary Application Entry Point:
Integrates all deep learning, computer vision, and temporal analysis components
into a unified real-time webcam driver monitoring system:

Pipeline:
WEBCAM -> OpenCV -> Face Detection (MediaPipe) -> Facial Landmarks
-> Eye Extraction -> Eye CNN -> EAR -> Mouth Extraction -> Yawn CNN & MAR
-> Temporal Analysis -> Drowsiness Score -> State Machine (ALERT / WARNING / DROWSY)
-> Audible & Voice Alarm

Features:
- Live camera HUD telemetry: Eye status, Eye confidence, Left/Right/Avg EAR,
  Yawn status, MAR, Drowsiness score, Current state badge, FPS, Alarm status.
- Resilient edge-case handling: No face, Face disappearing, Multiple faces (tracks driver),
  Low confidence, Camera failure, and Missing models.
- Pre-trained model loading: Runs inference directly without retraining on launch.
- Non-blocking asynchronous alarm: Audio beeps and voice warning with automatic recovery.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set Keras backend to torch (fully compatible with Python 3.14)
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows console encoding for Hindi/Punjabi Unicode characters
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config
from src.face_detection import FaceDetector
from src.eye_detection import EyeDetector
from src.yawn_detection import YawnDetector
from src.drowsiness_detector import DrowsinessDetector, DrowsinessState
from src.alarm import DrowsinessAlarm


# ==============================================================================
# MODEL VERIFICATION & STARTUP CHECKS
# ==============================================================================

def verify_system_models() -> Tuple[Path, Optional[Path]]:
    """
    Checks that trained deep learning models exist locally before launching.
    Does NOT train models; loads existing checkpoints.
    """
    # 1. Primary Eye Model
    eye_model_path = config.MODELS_DIR / "eye_model" / "best_eye_model.keras"
    if not eye_model_path.exists():
        fallback_path = config.MODELS_DIR / "eye_model" / "eye_model.keras"
        if fallback_path.exists():
            eye_model_path = fallback_path
        else:
            raise FileNotFoundError(
                f"\n[ERROR] Primary Eye CNN model not found at:\n  {eye_model_path}\n"
                "Please run 'python src/train_eye_model.py' to train the eye model first!"
            )

    # 2. Optional Yawn Model
    yawn_model_path = config.MODELS_DIR / "yawn_model" / "best_yawn_model.keras"
    if not yawn_model_path.exists():
        fallback_yawn = config.MODELS_DIR / "yawn_model" / "yawn_model.keras"
        if fallback_yawn.exists():
            yawn_model_path = fallback_yawn
        else:
            print("\n[NOTICE] Yawn CNN model not found. Yawn module will operate in MAR biometric mode.")
            yawn_model_path = None

    return eye_model_path, yawn_model_path


# ==============================================================================
# MULTILINGUAL HUD FONT & BANNER RENDERING (English, Hindi, Punjabi)
# ==============================================================================

_CACHED_HUD_FONTS = {}

def get_hud_fonts():
    """
    Returns cached PIL ImageFont instances with Indic script support (Nirmala UI).
    Caches font objects in memory for ~0.5ms ultra-low per-frame latency.
    """
    global _CACHED_HUD_FONTS
    if not _CACHED_HUD_FONTS:
        font_candidates = [
            "C:/Windows/Fonts/Nirmala.ttc",
            "C:/Windows/Fonts/nirmala.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        font_path = None
        for cand in font_candidates:
            if Path(cand).exists():
                font_path = cand
                break
        try:
            if font_path:
                _CACHED_HUD_FONTS["bold"] = ImageFont.truetype(font_path, 17)
                _CACHED_HUD_FONTS["indic"] = ImageFont.truetype(font_path, 15)
                _CACHED_HUD_FONTS["small"] = ImageFont.truetype(font_path, 13)
            else:
                _CACHED_HUD_FONTS["bold"] = ImageFont.load_default()
                _CACHED_HUD_FONTS["indic"] = ImageFont.load_default()
                _CACHED_HUD_FONTS["small"] = ImageFont.load_default()
        except Exception:
            _CACHED_HUD_FONTS["bold"] = ImageFont.load_default()
            _CACHED_HUD_FONTS["indic"] = ImageFont.load_default()
            _CACHED_HUD_FONTS["small"] = ImageFont.load_default()
    return _CACHED_HUD_FONTS


def draw_multilingual_banner(frame: np.ndarray, state: str, language_mode: str = "all") -> np.ndarray:
    """
    Renders an emergency multilingual visual banner on the video viewport:
    - English (EN)
    - Hindi - हिन्दी (HI)
    - Punjabi - ਪੰਜਾਬੀ (PA)
    """
    h, w = frame.shape[:2]
    fonts = get_hud_fonts()

    # Dynamic scaling for various resolutions
    is_multi = (language_mode == "all")
    scale = min(1.0, max(0.4, w / 640.0))
    box_h = int((95 if is_multi else 60) * scale)
    box_y1 = max(h - box_h - int(12 * scale), 10)
    box_y2 = box_y1 + box_h
    box_w = min(w - 20, int(660 * scale))
    box_x1 = max(10, (w - box_w) // 2)
    box_x2 = box_x1 + box_w

    # Semi-transparent colored background
    overlay = frame.copy()
    if state == "DROWSY":
        bg_color = (0, 0, 180)     # Crimson Red
        border_color = (0, 0, 255)
    else:
        bg_color = (0, 110, 200)   # Warning Amber Gold
        border_color = (0, 160, 255)

    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), border_color, max(1, int(2 * scale)))

    # Render authentic Unicode text with PIL
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    center_x = (box_x1 + box_x2) // 2

    # Choose scaled font or standard font
    f_bold = fonts["bold"]
    f_indic = fonts["indic"]

    if state == "DROWSY":
        if is_multi:
            # 3-Language Warning: English + Hindi + Punjabi
            text_en = "[!] CRITICAL: DROWSINESS DETECTED! PLEASE TAKE A BREAK"
            text_hi = "सावधान! चालक को नींद आ रही है! कृपया गाड़ी रोकें और आराम करें।"
            text_pa = "ਖ਼ਬਰਦਾਰ! ਡਰਾਈਵਰ ਨੂੰ ਨੀਂਦ ਆ ਰਹੀ ਹੈ! ਕਿਰਪਾ ਕਰਕੇ ਗੱਡੀ ਰੋਕੋ ਅਤੇ ਆਰਾਮ ਕਰੋ।"

            draw.text((center_x, box_y1 + int(18 * scale)), text_en, font=f_bold, fill=(255, 255, 255), anchor="mm")
            draw.text((center_x, box_y1 + int(46 * scale)), text_hi, font=f_indic, fill=(255, 235, 80), anchor="mm")
            draw.text((center_x, box_y1 + int(74 * scale)), text_pa, font=f_indic, fill=(130, 255, 255), anchor="mm")
        elif language_mode == "hi":
            text_en = "[!] CRITICAL: DROWSINESS DETECTED!"
            text_hi = "सावधान! चालक को नींद आ रही है! कृपया गाड़ी रोकें और आराम करें।"
            draw.text((center_x, box_y1 + int(18 * scale)), text_en, font=f_bold, fill=(255, 255, 255), anchor="mm")
            draw.text((center_x, box_y1 + int(42 * scale)), text_hi, font=f_indic, fill=(255, 235, 80), anchor="mm")
        elif language_mode == "pa":
            text_en = "[!] CRITICAL: DROWSINESS DETECTED!"
            text_pa = "ਖ਼ਬਰਦਾਰ! ਡਰਾਈਵਰ ਨੂੰ ਨੀਂਦ ਆ ਰਹੀ ਹੈ! ਕਿਰਪਾ ਕਰਕੇ ਗੱਡੀ ਰੋਕੋ ਅਤੇ ਆਰਾਮ ਕਰੋ।"
            draw.text((center_x, box_y1 + int(18 * scale)), text_en, font=f_bold, fill=(255, 255, 255), anchor="mm")
            draw.text((center_x, box_y1 + int(42 * scale)), text_pa, font=f_indic, fill=(130, 255, 255), anchor="mm")
        else:  # en
            text_en = "[!] CRITICAL: DROWSINESS DETECTED! PLEASE TAKE A BREAK"
            draw.text((center_x, box_y1 + int(30 * scale)), text_en, font=f_bold, fill=(255, 255, 255), anchor="mm")

    else:  # WARNING
        if is_multi:
            text_en = "[!] WARNING: DRIVER FATIGUE DETECTED"
            text_hi = "चेतावनी: चालक को थकान महसूस हो रही है"
            text_pa = "ਚੇਤਾਵਨੀ: ਡਰਾਈਵਰ ਨੂੰ ਥਕਾਵਟ ਮਹਿਸੂਸ ਹੋ ਰਹੀ ਹੈ"
            draw.text((center_x, box_y1 + int(18 * scale)), text_en, font=f_bold, fill=(255, 255, 255), anchor="mm")
            draw.text((center_x, box_y1 + int(46 * scale)), text_hi, font=f_indic, fill=(255, 255, 255), anchor="mm")
            draw.text((center_x, box_y1 + int(74 * scale)), text_pa, font=f_indic, fill=(255, 255, 255), anchor="mm")
        elif language_mode == "hi":
            draw.text((center_x, box_y1 + int(18 * scale)), "[!] WARNING: DRIVER FATIGUE DETECTED", font=f_bold, fill=(255, 255, 255), anchor="mm")
            draw.text((center_x, box_y1 + int(42 * scale)), "चेतावनी: चालक को थकान महसूस हो रही है", font=f_indic, fill=(255, 255, 255), anchor="mm")
        elif language_mode == "pa":
            draw.text((center_x, box_y1 + int(18 * scale)), "[!] WARNING: DRIVER FATIGUE DETECTED", font=f_bold, fill=(255, 255, 255), anchor="mm")
            draw.text((center_x, box_y1 + int(42 * scale)), "ਚੇਤਾਵਨੀ: ਡਰਾਈਵਰ ਨੂੰ ਥਕਾਵਟ ਮਹਿਸੂਸ ਹੋ ਰਹੀ ਹੈ", font=f_indic, fill=(255, 255, 255), anchor="mm")
        else:
            draw.text((center_x, box_y1 + int(30 * scale)), "[!] WARNING: DRIVER FATIGUE DETECTED", font=f_bold, fill=(255, 255, 255), anchor="mm")

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ==============================================================================
# MASTER HUD RENDERING
# ==============================================================================

def draw_master_hud(frame: np.ndarray,
                    faces: List[Dict],
                    primary_face: Optional[Dict],
                    eye_results: Dict,
                    mouth_results: Optional[Dict],
                    decision_results: Dict,
                    alarm_status: Tuple[str, Tuple[int, int, int]],
                    fps: float,
                    language_mode: str = "all") -> np.ndarray:
    """
    Renders high-definition HUD telemetry on the video frame:
    - Eye Status (Left & Right discrete state)
    - Eye Confidence (Left & Right %)
    - Left EAR & Right EAR (& Avg EAR)
    - Yawn Status (discrete state, confidence, MAR)
    - Drowsiness Score (percentage & graphical meter bar)
    - Current State ([ ALERT ] / [ WARNING ] / [ DROWSY ])
    - FPS Metric
    - Alarm Status (STANDBY / SOUNDING / COOLDOWN / MUTED with Language Tag)
    - Multilingual Alert Overlays (English, Hindi, Punjabi)
    """
    hud_frame = frame.copy()
    h, w = hud_frame.shape[:2]

    # 1. Annotate Secondary Passenger Faces (if multiple faces detected)
    if len(faces) > 1 and primary_face is not None:
        for face in faces:
            if face is not primary_face:
                px1, py1, px2, py2 = face["bbox"]
                cv2.rectangle(hud_frame, (px1, py1), (px2, py2), (100, 100, 100), 1)
                cv2.putText(hud_frame, "Passenger (Ignored)", (px1, max(12, py1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1, cv2.LINE_AA)

    # 2. Top Dashboard Header Panel (140px height)
    banner_h = 140
    overlay = hud_frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.82, hud_frame, 0.18, 0, hud_frame)

    # Title & FPS & Controls
    cv2.putText(hud_frame, "DRIVER DROWSINESS DETECTION SYSTEM", (15, 23),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(hud_frame, f"FPS: {fps:.1f}", (w - 110, 23),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    # 3. State & Alarm Badges
    current_state_str = decision_results.get("state", "ALERT")
    state_colors = {
        "ALERT": ((0, 220, 0), (0, 60, 0)),        # Green
        "WARNING": ((0, 180, 255), (0, 50, 80)),    # Amber Gold
        "DROWSY": ((0, 0, 255), (0, 0, 90)),       # Crimson Red
    }
    fg_color, bg_color = state_colors.get(current_state_str, ((0, 220, 0), (0, 60, 0)))

    # Driver State Badge (Top Right)
    badge_w = 160
    badge_h = 32
    badge_x = w - badge_w - 15
    badge_y = 35
    cv2.rectangle(hud_frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), bg_color, -1)
    cv2.rectangle(hud_frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), fg_color, 2)
    cv2.putText(hud_frame, f"STATE: {current_state_str}", (badge_x + 10, badge_y + 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, fg_color, 2, cv2.LINE_AA)

    # Alarm Status Text (Below State Badge)
    alarm_text, alarm_color = alarm_status
    cv2.putText(hud_frame, f"Alarm: {alarm_text}", (badge_x, badge_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, alarm_color, 1, cv2.LINE_AA)

    # Handle No Face Detected gracefully
    if not eye_results.get("detected", False):
        cv2.putText(hud_frame, "STATUS: SEARCHING FOR DRIVER FACE...", (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(hud_frame, "Position face in camera view", (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)
        return hud_frame

    # 4. Telemetry Display (3 Columns)

    # --- Column 1: Eye Status & EAR ---
    l_state = eye_results.get("left_state", "OPEN")
    r_state = eye_results.get("right_state", "OPEN")
    l_conf = eye_results.get("left_conf", 0.0)
    r_conf = eye_results.get("right_conf", 0.0)
    l_ear = eye_results.get("left_ear", 0.0)
    r_ear = eye_results.get("right_ear", 0.0)
    avg_ear = eye_results.get("avg_ear", 0.0)

    # Row 1: Eye States and Confidence
    eye_status_txt = f"Left Eye: {l_state} ({l_conf:.0f}%)  |  Right Eye: {r_state} ({r_conf:.0f}%)"
    cv2.putText(hud_frame, eye_status_txt, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)

    # Row 2: EAR Values
    ear_txt = f"Left EAR: {l_ear:.3f}  |  Right EAR: {r_ear:.3f}  |  Avg EAR: {avg_ear:.3f}"
    cv2.putText(hud_frame, ear_txt, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    # --- Column 2: Mouth & Yawn Status ---
    m_state = mouth_results.get("state", "NO_YAWN") if mouth_results else "NONE"
    m_conf = mouth_results.get("confidence", 0.0) if mouth_results else 0.0
    mar = mouth_results.get("mar", 0.0) if mouth_results else 0.0
    is_yawn = mouth_results.get("is_yawning", False) if mouth_results else False
    m_color = (0, 140, 255) if is_yawn else (0, 220, 0)

    yawn_txt = f"Mouth/Yawn: {m_state} ({m_conf:.0f}%)  |  MAR: {mar:.3f}"
    cv2.putText(hud_frame, yawn_txt, (15, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.48, m_color, 1, cv2.LINE_AA)

    # --- Row 3: Temporal Stats & Closure Duration ---
    temp_metrics = decision_results.get("temporal_metrics", {})
    closure_sec = temp_metrics.get("closure_duration_sec", 0.0)
    blinks = temp_metrics.get("total_blinks", 0)
    perclos = temp_metrics.get("perclos", 0.0) * 100.0

    temp_txt = f"Closure: {closure_sec:.1f}s  |  PERCLOS: {perclos:.1f}%  |  Blinks: {blinks}"
    cv2.putText(hud_frame, temp_txt, (15, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1, cv2.LINE_AA)

    # --- Row 4: Drowsiness Fusion Score Meter Bar ---
    score = decision_results.get("drowsiness_score", 0.0)
    bar_x = 360
    bar_y = 100
    bar_w = 200
    bar_h = 16
    score_fill = int(bar_w * score)

    cv2.rectangle(hud_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    if score_fill > 0:
        cv2.rectangle(hud_frame, (bar_x, bar_y), (bar_x + score_fill, bar_y + bar_h), fg_color, -1)
    cv2.rectangle(hud_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)
    cv2.putText(hud_frame, f"Score: {score * 100:.1f}%", (bar_x + bar_w + 10, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, fg_color, 1, cv2.LINE_AA)

    # 5. Prominent Critical Multilingual Alert Overlays (English + Hindi + Punjabi)
    if current_state_str in ("DROWSY", "WARNING"):
        hud_frame = draw_multilingual_banner(hud_frame, current_state_str, language_mode)

    return hud_frame


# ==============================================================================
# MAIN SYSTEM EXECUTION
# ==============================================================================

def run_system(camera_index: int = 0,
               video_path: Optional[Path] = None,
               verify_sample: bool = False,
               mute_alarm: bool = False,
               language_mode: str = getattr(config, "ALERT_LANGUAGE_MODE", "all")):
    """
    Launches and coordinates the complete real-time driver drowsiness detection system.
    Supports English, Hindi, and Punjabi visual & audible alerts.
    """
    print("\n" + "=" * 70)
    print("  STARTING DRIVER DROWSINESS DETECTION & ALERT SYSTEM")
    print("=" * 70)

    # 1. Verify and Load Saved Models (Zero re-training on launch)
    eye_model_path, yawn_model_path = verify_system_models()
    print(f"Loading Eye CNN Checkpoint:   {eye_model_path}")
    if yawn_model_path:
        print(f"Loading Yawn CNN Checkpoint:  {yawn_model_path}")

    # 2. Instantiate Detection & Decision Engines
    face_detector = FaceDetector()
    eye_detector = EyeDetector(model_path=eye_model_path, target_size=(64, 64))
    yawn_detector = YawnDetector(model_path=yawn_model_path, target_size=(64, 64))
    drowsiness_detector = DrowsinessDetector()
    alarm = DrowsinessAlarm(
        cooldown_sec=config.ALARM_COOLDOWN,
        speech_message=config.ALARM_SPEECH_MESSAGE,
        enabled=(not mute_alarm and config.ENABLE_AUDIO_ALARM),
        language_mode=language_mode
    )

    lang_desc = {
        "all": "Multilingual (English + Hindi + Punjabi)",
        "hi":  "Hindi - हिन्दी",
        "pa":  "Punjabi - ਪੰਜਾਬੀ",
        "en":  "English"
    }.get(language_mode.lower(), language_mode)

    print("\nSystem Components Ready:")
    print("  • Face Detector:          MediaPipe FaceLandmarker (478 3D landmarks)")
    print("  • Eye Classifier:         Deep Learning Eye CNN (64x64, 100k params)")
    print("  • Biometrics:             Eye Aspect Ratio (EAR) + Mouth Aspect Ratio (MAR)")
    print(f"  • Yawn Classifier:        Yawn CNN (Enabled: {yawn_detector.is_available})")
    print("  • Temporal Engine:        Multi-Signal Fusion (ALERT / WARNING / DROWSY)")
    print(f"  • Alarm System:           Audio Siren + Voice Alerts (Muted: {mute_alarm})")
    print(f"  • Voice & HUD Language:   {lang_desc}")
    print("=" * 70 + "\n")

    # -------------------------------------------------------------
    # Verification Mode on Dataset Sample (Headless / Self-test)
    # -------------------------------------------------------------
    if verify_sample:
        sample_path = config.DDD_DATASET_DIR / "Drowsy" / "A0001.png"
        print(f"Running self-test verification on sample image: {sample_path}")
        frame = cv2.imread(str(sample_path))
        if frame is None:
            raise ValueError(f"Could not load test image from {sample_path}")

        faces = face_detector.detect_faces(frame)
        primary_face = face_detector.get_primary_face(faces, frame.shape)
        eye_results = eye_detector.process_face_eyes(frame, primary_face)
        mouth_results = yawn_detector.process_face_mouth(frame, primary_face)

        decision_results = drowsiness_detector.update(eye_results, mouth_results)
        alarm.trigger(decision_results["state"])

        alert_active = (decision_results["state"] == "DROWSY") or alarm.is_alert_active()
        annotated = face_detector.draw_face_annotations(
            frame, primary_face, len(faces),
            is_drowsy=alert_active
        )
        annotated = yawn_detector.draw_mouth_annotations(annotated, mouth_results)
        decision_for_hud = dict(decision_results)
        if alert_active:
            decision_for_hud["state"] = "DROWSY"

        hud_frame = draw_master_hud(
            annotated, faces, primary_face, eye_results, mouth_results,
            decision_for_hud, alarm.get_status(), fps=30.0,
            language_mode=alarm.get_language_mode()
        )

        out_path = config.PREDICTIONS_DIR / "main_system_verification.png"
        cv2.imwrite(str(out_path), hud_frame)
        print(f"Verification successful! Master HUD sample saved to: {out_path}")
        print("\nVerification Telemetry:")
        print(f"  • Driver State:      {decision_results['state']}")
        print(f"  • Drowsiness Score:  {decision_results['drowsiness_score']*100:.1f}%")
        print(f"  • Left Eye State:    {eye_results['left_state']} ({eye_results['left_conf']:.1f}%)")
        print(f"  • Right Eye State:   {eye_results['right_state']} ({eye_results['right_conf']:.1f}%)")
        print(f"  • Left / Right EAR:  {eye_results['left_ear']:.3f} / {eye_results['right_ear']:.3f}")
        print(f"  • Mouth / MAR:       {mouth_results['state']} | {mouth_results['mar']:.3f}")
        print(f"  • Alarm Status:      {alarm.get_status()[0]}")
        alarm.stop()
        return

    # -------------------------------------------------------------
    # Real-Time Video / Webcam Capture Loop
    # -------------------------------------------------------------
    if video_path:
        print(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
    else:
        print(f"Connecting to webcam at device index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)

    # Handle Camera Open Failure Gracefully
    if not cap.isOpened():
        print(f"\n[CAMERA WARNING] Could not access camera index {camera_index} (device busy or unavailable).")
        print("Falling back to self-test verification sample mode...")
        run_system(verify_sample=True, mute_alarm=mute_alarm, language_mode=language_mode)
        return

    print("Camera stream established successfully.")
    print("\nKeyboard Controls:")
    print("  • 'q' or ESC: Quit application")
    print("  • 'm':        Toggle audible alarm mute")
    print("  • 'l':        Cycle Alert Language (ALL -> HI -> PA -> EN)")
    print("  • 's':        Capture screenshot to outputs/predictions/")
    print("-" * 50 + "\n")

    prev_time = time.time()
    dropped_frames_count = 0

    try:
        while True:
            ret, frame = cap.read()

            # Handle Camera Disconnect or Dropped Frames Gracefully
            if not ret or frame is None:
                dropped_frames_count += 1
                if dropped_frames_count > 60:
                    print("\n[WARNING] Camera signal lost. Reconnecting or terminating cleanly...")
                    break
                time.sleep(0.01)
                continue

            dropped_frames_count = 0
            curr_time = time.time()
            fps = 1.0 / max(1e-4, curr_time - prev_time)
            prev_time = curr_time

            # Step 1: Detect Faces & Isolate Primary Driver Face
            faces = face_detector.detect_faces(frame)
            primary_face = face_detector.get_primary_face(faces, frame.shape)

            # Step 2: Eye ROI Extraction & Deep Learning Eye State Inference
            eye_results = eye_detector.process_face_eyes(frame, primary_face)

            # Step 3: Mouth ROI Extraction & Yawn CNN Inference
            mouth_results = yawn_detector.process_face_mouth(frame, primary_face)

            # Step 4: Temporal Drowsiness Analysis & Multi-Signal Fusion
            decision_results = drowsiness_detector.update(
                eye_results, mouth_results, current_time=curr_time
            )

            # Step 5: Audible Alarm Trigger (Starts on DROWSY, completes warning cycle)
            alarm.trigger(decision_results["state"], current_time=curr_time)

            # Step 6: Render Visual Overlays & HUD Dashboard (Multilingual + High-Intensity Wake-Up Strobe)
            # Alert mode stays active if decision is DROWSY OR if alarm is actively sounding warning cycle
            alert_active = (decision_results["state"] == "DROWSY") or alarm.is_alert_active()

            annotated_frame = face_detector.draw_face_annotations(
                frame, primary_face, len(faces),
                is_drowsy=alert_active,
                current_time=curr_time
            )
            annotated_frame = yawn_detector.draw_mouth_annotations(annotated_frame, mouth_results)

            # If alert cycle is currently playing, persist DROWSY HUD alert state so driver is fully woken
            decision_for_hud = dict(decision_results)
            if alert_active:
                decision_for_hud["state"] = "DROWSY"

            hud_frame = draw_master_hud(
                annotated_frame, faces, primary_face, eye_results, mouth_results,
                decision_for_hud, alarm.get_status(curr_time), fps=fps,
                language_mode=alarm.get_language_mode()
            )

            # Display window
            cv2.imshow("Driver Drowsiness Detection System", hud_frame)

            # Key Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                break
            elif key == ord("m"):  # Toggle Mute
                alarm.enabled = not alarm.enabled
                status = "MUTED" if not alarm.enabled else "ENABLED"
                print(f"[AUDIO] Alarm sound {status}")
            elif key == ord("l"):  # Cycle Alert Language
                next_modes = {"all": "hi", "hi": "pa", "pa": "en", "en": "all"}
                new_mode = next_modes.get(alarm.get_language_mode(), "all")
                alarm.set_language_mode(new_mode)
                label = {"all": "MULTILINGUAL (EN+HI+PA)", "hi": "HINDI (हिन्दी)", "pa": "PUNJABI (ਪੰਜਾਬੀ)", "en": "ENGLISH"}.get(new_mode, new_mode.upper())
                print(f"[LANGUAGE] Alert Language switched to: {label}")
            elif key == ord("s"):  # Save screenshot
                snap_path = config.PREDICTIONS_DIR / f"snapshot_{int(time.time())}.png"
                cv2.imwrite(str(snap_path), hud_frame)
                print(f"[SNAPSHOT] Saved HUD screenshot to: {snap_path}")

    finally:
        alarm.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("\nSystem closed cleanly. Session summary:")
        print(f"  • Total Blinks:       {drowsiness_detector.total_blinks}")
        print(f"  • Total Yawn Events:  {drowsiness_detector.total_yawn_events}")
        print(f"  • Total Alarms Fired: {alarm.total_alarms_triggered}")
        print("Goodbye.\n")


# ==============================================================================
# CLI ARGUMENTS
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Driver Drowsiness Detection & Multilingual Alert System"
    )
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                        help="Webcam device index (default: 0)")
    parser.add_argument("--video", type=str, default=None,
                        help="Optional path to video file for evaluation")
    parser.add_argument("--verify", action="store_true",
                        help="Run automated self-test verification on dataset sample image")
    parser.add_argument("--mute", action="store_true",
                        help="Mute audible alarms (visual warnings remain active)")
    parser.add_argument("--lang", "--language", type=str, default=getattr(config, "ALERT_LANGUAGE_MODE", "all"),
                        choices=["all", "en", "hi", "pa"],
                        help="Alert voice & HUD language: all (English+Hindi+Punjabi), hi (Hindi), pa (Punjabi), en (English)")
    args = parser.parse_args()

    run_system(
        camera_index=args.camera,
        video_path=Path(args.video) if args.video else None,
        verify_sample=args.verify,
        mute_alarm=args.mute,
        language_mode=args.lang,
    )


if __name__ == "__main__":
    main()
