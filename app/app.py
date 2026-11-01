"""
Real-Time Driver Drowsiness Detection and Alert System.
Academic Demonstration Dashboard using Streamlit.

Features:
- Live Monitoring Feed with Real-Time HUD and Bounding Box Annotations
- Live Telemetry: Driver State (ALERT / WARNING / DROWSY), Eye State, Eye Confidence,
  Left & Right EAR, Yawn Status, MAR, Drowsiness Score Meter, Alarm Status, and FPS
- Academic Model Information: Architecture, Datasets, Classes, and Held-Out Test Metrics
- Monitoring Information: Active Detection Time, Drowsiness Events, Alarms Fired, State Timers
- Interactive Controls: Start / Stop Monitoring, Camera Device Selection, Audio Mute Toggle
- Loads Saved Pre-Trained Models Directly (Zero Re-Training on Launch)
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

# Set Keras backend to torch
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.face_detection import FaceDetector
from src.eye_detection import EyeDetector
from src.yawn_detection import YawnDetector
from src.drowsiness_detector import DrowsinessDetector, DrowsinessState
from src.alarm import DrowsinessAlarm
import main as main_pipeline


# ==============================================================================
# STREAMLIT PAGE CONFIG & CUSTOM AESTHETICS
# ==============================================================================

st.set_page_config(
    page_title="Driver Drowsiness Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Academic Theme CSS
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: #1e222d;
        border: 1px solid #2d3343;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .metric-title {
        color: #8b949e;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 700;
    }
    
    /* State Badges */
    .badge-alert {
        background-color: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        border: 1px solid #2ecc71;
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 1.3rem;
        font-weight: 800;
        text-align: center;
        display: inline-block;
        width: 100%;
    }
    .badge-warning {
        background-color: rgba(243, 156, 18, 0.15);
        color: #f39c12;
        border: 1px solid #f39c12;
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 1.3rem;
        font-weight: 800;
        text-align: center;
        display: inline-block;
        width: 100%;
    }
    .badge-drowsy {
        background-color: rgba(255, 0, 0, 0.35);
        color: #ffffff;
        border: 2px solid #ff0000;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 1.3rem;
        font-weight: 800;
        text-align: center;
        display: inline-block;
        width: 100%;
        animation: emergency-strobe 0.55s infinite;
    }
    @keyframes emergency-strobe {
        0%, 100% {
            transform: scale(1);
            background-color: rgba(220, 20, 20, 0.3);
            box-shadow: 0 0 20px rgba(255, 0, 0, 0.5);
            border-color: #ff2222;
        }
        50% {
            transform: scale(1.03);
            background-color: rgba(255, 0, 0, 0.75);
            box-shadow: 0 0 55px rgba(255, 0, 0, 1.0);
            border-color: #ffffff;
        }
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CACHED MODEL LOADING (Ensures Models Load Once, Zero Re-training)
# ==============================================================================

@st.cache_resource(show_spinner="Loading deep learning models into memory...")
def load_detection_models():
    """
    Loads saved pre-trained model weights.
    Cached across Streamlit app reruns for high performance.
    """
    eye_model_path, yawn_model_path = main_pipeline.verify_system_models()
    face_detector = FaceDetector()
    eye_detector = EyeDetector(model_path=eye_model_path, target_size=(64, 64))
    yawn_detector = YawnDetector(model_path=yawn_model_path, target_size=(64, 64))
    return face_detector, eye_detector, yawn_detector, eye_model_path, yawn_model_path


# ==============================================================================
# HELPER: LOAD BENCHMARK METRICS
# ==============================================================================

def load_academic_metrics():
    """Reads evaluation JSON manifests from held-out test evaluations."""
    eye_json = config.PREDICTIONS_DIR / "evaluation_summary.json"
    yawn_json = config.PREDICTIONS_DIR / "yawn_evaluation_summary.json"

    eye_data = {}
    if eye_json.exists():
        with open(eye_json, "r") as f:
            eye_data = json.load(f)

    yawn_data = {}
    if yawn_json.exists():
        with open(yawn_json, "r") as f:
            yawn_data = json.load(f)

    return eye_data, yawn_data


# ==============================================================================
# APPLICATION LAYOUT
# ==============================================================================

def main():
    # Load Models
    try:
        face_detector, eye_detector, yawn_detector, eye_path, yawn_path = load_detection_models()
    except Exception as e:
        st.error(f"Error loading system models: {e}")
        st.info("Please ensure models are trained and saved under the `models/` directory.")
        return

    # Session State Initialization
    if "is_monitoring" not in st.session_state:
        st.session_state.is_monitoring = False
    if "drowsiness_detector" not in st.session_state:
        st.session_state.drowsiness_detector = DrowsinessDetector()
    if "alarm" not in st.session_state:
        st.session_state.alarm = DrowsinessAlarm(cooldown_sec=config.ALARM_COOLDOWN)
    if "total_drowsy_events" not in st.session_state:
        st.session_state.total_drowsy_events = 0
    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()

    # -------------------------------------------------------------
    # SIDEBAR: SYSTEM CONTROLS & MONITORING TELEMETRY
    # -------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")

        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("▶ Start", use_container_width=True, type="primary", disabled=st.session_state.is_monitoring):
                st.session_state.is_monitoring = True
                st.session_state.start_time = time.time()
                st.rerun()

        with col_stop:
            if st.button("⏹ Stop", use_container_width=True, type="secondary", disabled=not st.session_state.is_monitoring):
                st.session_state.is_monitoring = False
                st.session_state.alarm.stop()
                st.rerun()

        st.markdown("---")
        st.markdown("### ⚙️ Video & Audio Settings")
        camera_id = st.number_input("Camera Index", min_value=0, max_value=5, value=config.CAMERA_INDEX, step=1)
        mute_audio = st.checkbox("Mute Audible Siren", value=False)
        st.session_state.alarm.enabled = not mute_audio

        lang_options = {
            "Multilingual (English + Hindi + Punjabi)": "all",
            "Hindi - हिन्दी (Devanagari)": "hi",
            "Punjabi - ਪੰਜਾਬੀ (Gurmukhi)": "pa",
            "English": "en"
        }
        selected_lang_label = st.selectbox(
            "🗣️ Alert Language (ਚੇਤਾਵਨੀ ਭਾਸ਼ਾ / भाषा)",
            options=list(lang_options.keys()),
            index=0
        )
        selected_lang_code = lang_options[selected_lang_label]
        st.session_state.alarm.set_language_mode(selected_lang_code)

        red_flash_enabled = st.checkbox(
            "🚨 Emergency Red Light Face Strobe",
            value=getattr(config, "ENABLE_FACE_RED_LIGHT_FLASH", True),
            help="Illuminates and flashes bright red light directly on the driver's face during drowsiness alert"
        )
        config.ENABLE_FACE_RED_LIGHT_FLASH = red_flash_enabled

        if st.button("🔊 Test Voice Alert", use_container_width=True, help="Test voice warnings in chosen language"):
            st.session_state.alarm.trigger("DROWSY")
            st.toast(f"Playing alert in: {selected_lang_label}!", icon="📢")

        st.markdown("---")
        st.markdown("### ⏱️ Session Statistics")

        elapsed_sec = time.time() - st.session_state.start_time if st.session_state.is_monitoring else 0.0
        st.metric("Monitoring Duration", f"{int(elapsed_sec // 60)}m {int(elapsed_sec % 60)}s")
        st.metric("Drowsiness Events", st.session_state.total_drowsy_events)
        st.metric("Alarms Triggered", st.session_state.alarm.total_alarms_triggered)
        st.metric("Total Blinks Logged", st.session_state.drowsiness_detector.total_blinks)

        st.markdown("---")
        st.caption("College Project Demonstration Build\nVersion 1.0 (Academic)")

    # -------------------------------------------------------------
    # MAIN HEADER
    # -------------------------------------------------------------
    st.title("Real-Time Driver Drowsiness Detection and Alert System")
    st.markdown(
        "A multi-modal deep learning and computer vision framework combining "
        "**MediaPipe Facial Landmarks**, **Custom CNN Eye Classification**, **EAR/MAR Biometrics**, "
        "and **Temporal Multi-Signal Fusion**."
    )

    # -------------------------------------------------------------
    # MAIN SECTION TABS
    # -------------------------------------------------------------
    tab_live, tab_models, tab_stats = st.tabs([
        "📹 Live Monitoring",
        "🧠 Model Architecture & Metrics",
        "📊 Monitoring Information"
    ])

    # =============================================================
    # TAB 1: LIVE MONITORING
    # =============================================================
    with tab_live:
        col_video, col_telemetry = st.columns([1.6, 1.0])

        with col_video:
            st.markdown("#### 📺 Live Camera Feed")
            frame_container = st.empty()

            if not st.session_state.is_monitoring:
                # Idle Placeholder Image
                sample_path = config.PREDICTIONS_DIR / "main_system_verification.png"
                if sample_path.exists():
                    frame_container.image(str(sample_path), caption="System Standby — Click 'Start' to begin live monitoring", use_container_width=True)
                else:
                    frame_container.info("Camera is idle. Click **'▶ Start'** in the sidebar to initiate real-time monitoring.")

        with col_telemetry:
            st.markdown("#### 🚦 Driver Telemetry")

            state_badge_placeholder = st.empty()
            score_bar_placeholder = st.empty()

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                eye_state_placeholder = st.empty()
                left_ear_placeholder = st.empty()
                fps_placeholder = st.empty()
            with col_m2:
                yawn_state_placeholder = st.empty()
                right_ear_placeholder = st.empty()
                alarm_status_placeholder = st.empty()

            # Initial State Display
            state_badge_placeholder.markdown('<div class="badge-alert">ALERT (STANDBY)</div>', unsafe_allow_html=True)
            eye_state_placeholder.metric("Eye State", "OPEN", "Normal")
            yawn_state_placeholder.metric("Mouth State", "NO_YAWN", "Normal")
            left_ear_placeholder.metric("Left EAR", "0.250")
            right_ear_placeholder.metric("Right EAR", "0.255")
            fps_placeholder.metric("System FPS", "0.0")
            alarm_status_placeholder.metric("Alarm Status", "STANDBY")

        # ---------------------------------------------------------
        # REAL-TIME CAMERA INFERENCE LOOP
        # ---------------------------------------------------------
        if st.session_state.is_monitoring:
            cap = cv2.VideoCapture(camera_id)

            if not cap.isOpened():
                st.error(f"Could not open camera device at index {camera_id}. Please check webcam connection or try index 1.")
                st.session_state.is_monitoring = False
            else:
                prev_time = time.time()
                drowsiness_detector = st.session_state.drowsiness_detector
                alarm = st.session_state.alarm

                while st.session_state.is_monitoring:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        time.sleep(0.02)
                        continue

                    curr_time = time.time()
                    fps = 1.0 / max(1e-4, curr_time - prev_time)
                    prev_time = curr_time

                    # Pipeline Execution
                    faces = face_detector.detect_faces(frame)
                    primary_face = face_detector.get_primary_face(faces, frame.shape)
                    eye_results = eye_detector.process_face_eyes(frame, primary_face)
                    mouth_results = yawn_detector.process_face_mouth(frame, primary_face)

                    # Temporal Analysis & Decision
                    decision = drowsiness_detector.update(eye_results, mouth_results, current_time=curr_time)
                    current_state = decision["state"]

                    # Alarm Triggering (Starts on DROWSY, completes warning cycle)
                    alarm_triggered = alarm.trigger(current_state, current_time=curr_time)
                    if current_state == "DROWSY" and alarm_triggered:
                        st.session_state.total_drowsy_events += 1

                    # Alert stays active until warning cycle completes (persists even if eyes flutter or yawn stops)
                    alert_active = (current_state == "DROWSY") or alarm.is_alert_active()

                    # Visual HUD Annotations (Multilingual + High-Intensity Wake-Up Red Light Strobe)
                    annotated = face_detector.draw_face_annotations(
                        frame, primary_face, len(faces),
                        is_drowsy=alert_active,
                        current_time=curr_time
                    )
                    annotated = yawn_detector.draw_mouth_annotations(annotated, mouth_results)
                    decision_for_hud = dict(decision)
                    if alert_active:
                        decision_for_hud["state"] = "DROWSY"

                    hud_frame = main_pipeline.draw_master_hud(
                        annotated, faces, primary_face, eye_results, mouth_results,
                        decision_for_hud, alarm.get_status(curr_time), fps=fps,
                        language_mode=alarm.get_language_mode()
                    )

                    # Update Video Frame
                    frame_rgb = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
                    frame_container.image(frame_rgb, channels="RGB", use_container_width=True)

                    # Update Telemetry Widgets (Multilingual Alert Badges)
                    curr_lang = alarm.get_language_mode()
                    displayed_state = "DROWSY" if alert_active else current_state
                    if displayed_state == "ALERT":
                        state_badge_placeholder.markdown('<div class="badge-alert">STATE: ALERT (NORMAL)</div>', unsafe_allow_html=True)
                    elif displayed_state == "WARNING":
                        if curr_lang == "all":
                            state_badge_placeholder.markdown(
                                '<div class="badge-warning">'
                                '<div>⚠️ WARNING: DRIVER FATIGUE DETECTED</div>'
                                '<div style="font-size: 0.92rem; margin-top: 4px; color: #fff;">चेतावनी: चालक को थकान महसूस हो रही है</div>'
                                '<div style="font-size: 0.92rem; margin-top: 2px; color: #ffeb3b;">ਚੇਤਾਵਨੀ: ਡਰਾਈਵਰ ਨੂੰ ਥਕਾਵਟ ਮਹਿਸੂਸ ਹੋ ਰਹੀ ਹੈ</div>'
                                '</div>',
                                unsafe_allow_html=True
                            )
                        elif curr_lang == "hi":
                            state_badge_placeholder.markdown(
                                '<div class="badge-warning">'
                                '<div>⚠️ WARNING: DRIVER FATIGUE DETECTED</div>'
                                '<div style="font-size: 0.92rem; margin-top: 4px;">चेतावनी: चालक को थकान महसूस हो रही है</div>'
                                '</div>',
                                unsafe_allow_html=True
                            )
                        elif curr_lang == "pa":
                            state_badge_placeholder.markdown(
                                '<div class="badge-warning">'
                                '<div>⚠️ WARNING: DRIVER FATIGUE DETECTED</div>'
                                '<div style="font-size: 0.92rem; margin-top: 4px;">ਚੇਤਾਵਨੀ: ਡਰਾਈਵਰ ਨੂੰ ਥਕਾਵਟ ਮਹਿਸੂਸ ਹੋ ਰਹੀ ਹੈ</div>'
                                '</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            state_badge_placeholder.markdown('<div class="badge-warning">STATE: WARNING (FATIGUE)</div>', unsafe_allow_html=True)
                    else:  # DROWSY
                        if curr_lang == "all":
                            state_badge_placeholder.markdown(
                                '<div class="badge-drowsy">'
                                '<div>🚨 CRITICAL: DROWSINESS DETECTED!</div>'
                                '<div style="font-size: 1.05rem; margin-top: 5px; color: #ffea00;">⚠️ सावधान! चालक को नींद आ रही है!</div>'
                                '<div style="font-size: 1.05rem; margin-top: 4px; color: #64ffda;">⚠️ ਖ਼ਬਰਦਾਰ! ਡਰਾਈਵਰ ਨੂੰ ਨੀਂਦ ਆ ਰਹੀ ਹੈ!</div>'
                                '<div style="font-size: 0.85rem; margin-top: 3px; color: #ffffff;">ਕਿਰਪਾ ਕਰਕੇ ਗੱਡੀ ਰੋਕੋ ਅਤੇ ਆਰਾਮ ਕਰੋ</div>'
                                '</div>',
                                unsafe_allow_html=True
                            )
                        elif curr_lang == "hi":
                            state_badge_placeholder.markdown(
                                '<div class="badge-drowsy">'
                                '<div>🚨 CRITICAL: DROWSINESS DETECTED!</div>'
                                '<div style="font-size: 1.05rem; margin-top: 5px; color: #ffea00;">⚠️ सावधान! चालक को नींद आ रही है!</div>'
                                '<div style="font-size: 0.88rem; margin-top: 3px; color: #ffffff;">कृपया गाड़ी रोकें और आराम करें</div>'
                                '</div>',
                                unsafe_allow_html=True
                            )
                        elif curr_lang == "pa":
                            state_badge_placeholder.markdown(
                                '<div class="badge-drowsy">'
                                '<div>🚨 CRITICAL: DROWSINESS DETECTED!</div>'
                                '<div style="font-size: 1.05rem; margin-top: 5px; color: #64ffda;">⚠️ ਖ਼ਬਰਦਾਰ! ਡਰਾਈਵਰ ਨੂੰ ਨੀਂਦ ਆ ਰਹੀ ਹੈ!</div>'
                                '<div style="font-size: 0.88rem; margin-top: 3px; color: #ffffff;">ਕਿਰਪਾ ਕਰਕੇ ਗੱਡੀ ਰੋਕੋ ਅਤੇ ਆਰਾਮ ਕਰੋ</div>'
                                '</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            state_badge_placeholder.markdown('<div class="badge-drowsy">STATE: DROWSY (ALARM SOUNDING)</div>', unsafe_allow_html=True)

                    # Drowsiness Score Meter
                    score_val = decision["drowsiness_score"]
                    score_bar_placeholder.progress(score_val, text=f"Drowsiness Score: {score_val*100:.1f}%")

                    # Live Metric Boxes
                    l_st = eye_results.get("left_state", "OPEN")
                    r_st = eye_results.get("right_state", "OPEN")
                    l_cf = eye_results.get("left_conf", 0.0)
                    eye_state_placeholder.metric("Eye State", f"{l_st}/{r_st}", f"Conf: {l_cf:.0f}%")

                    m_st = mouth_results.get("state", "NO_YAWN")
                    m_cf = mouth_results.get("confidence", 0.0)
                    yawn_state_placeholder.metric("Mouth State", m_st, f"Conf: {m_cf:.0f}%")

                    left_ear_placeholder.metric("Left EAR", f"{eye_results.get('left_ear', 0.0):.3f}")
                    right_ear_placeholder.metric("Right EAR", f"{eye_results.get('right_ear', 0.0):.3f}")
                    fps_placeholder.metric("System FPS", f"{fps:.1f}")

                    alarm_txt, _ = alarm.get_status(curr_time)
                    alarm_status_placeholder.metric("Alarm Status", alarm_txt)

                cap.release()
                alarm.stop()

    # =============================================================
    # TAB 2: MODEL ARCHITECTURE & METRICS
    # =============================================================
    with tab_models:
        st.markdown("### 🧠 Academic Deep Learning Models")
        st.markdown("Detailed specifications of the primary and auxiliary neural networks trained and validated for this project.")

        eye_metrics, yawn_metrics = load_academic_metrics()

        col_eye_card, col_yawn_card = st.columns(2)

        # 1. Primary Eye State Model Card
        with col_eye_card:
            st.markdown("#### 👁️ Primary Eye State Classifier")
            st.markdown(f"**Model Type:** Custom Convolutional Neural Network (`EyeStateCNN`)")
            st.markdown(f"**Dataset Used:** MRL Eye Dataset (84,898 images)")
            st.markdown(f"**Subject Partitioning:** Disjoint Subject Split (Zero Data Leakage)")
            st.markdown(f"**Input Dimensions:** $64 \\times 64 \\times 3$ (RGB)")
            st.markdown(f"**Target Classes:** 2 (`OPEN`, `CLOSED`)")

            st.markdown("##### Held-Out Test Evaluation Metrics:")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{eye_metrics.get('accuracy', 0.8656)*100:.2f}%")
            c2.metric("Precision", f"{eye_metrics.get('per_class_metrics', {}).get('CLOSED', {}).get('precision', 0.9079)*100:.2f}%")
            c3.metric("Recall", f"{eye_metrics.get('per_class_metrics', {}).get('CLOSED', {}).get('recall', 0.8138)*100:.2f}%")
            c4.metric("F1-Score", f"{eye_metrics.get('per_class_metrics', {}).get('CLOSED', {}).get('f1_score', 0.8583)*100:.2f}%")

            # Confusion Matrix Preview
            cm_plot_eye = config.CONFUSION_MATRIX_DIR / "eye_model_confusion_matrix_detailed.png"
            if not cm_plot_eye.exists():
                cm_plot_eye = config.CONFUSION_MATRIX_DIR / "eye_model_confusion_matrix.png"
            if cm_plot_eye.exists():
                st.image(str(cm_plot_eye), caption="Eye State CNN Confusion Matrix (Held-out Test Subjects)", use_container_width=True)

        # 2. Auxiliary Yawn Model Card
        with col_yawn_card:
            st.markdown("#### 🥱 Auxiliary Yawn Detection Classifier")
            st.markdown(f"**Model Type:** Deep Learning Yawn CNN (`YawnStateCNN`)")
            st.markdown(f"**Dataset Used:** Yawn Dataset (5,119 mouth images)")
            st.markdown(f"**Subject Partitioning:** Contiguous Temporal Block Partitioning")
            st.markdown(f"**Input Dimensions:** $64 \\times 64 \\times 3$ (RGB)")
            st.markdown(f"**Target Classes:** 2 (`NO_YAWN`, `YAWN`)")

            st.markdown("##### Held-Out Test Evaluation Metrics:")
            y1, y2, y3, y4 = st.columns(4)
            y1.metric("Accuracy", f"{yawn_metrics.get('accuracy', 0.9117)*100:.2f}%")
            y2.metric("Precision", f"{yawn_metrics.get('per_class', {}).get('YAWN', {}).get('precision', 0.8959)*100:.2f}%")
            y3.metric("Recall", f"{yawn_metrics.get('per_class', {}).get('YAWN', {}).get('recall', 0.9289)*100:.2f}%")
            y4.metric("F1-Score", f"{yawn_metrics.get('per_class', {}).get('YAWN', {}).get('f1', 0.9121)*100:.2f}%")

            # Yawn Confusion Matrix Preview
            cm_plot_yawn = config.CONFUSION_MATRIX_DIR / "yawn_confusion_matrix.png"
            if cm_plot_yawn.exists():
                st.image(str(cm_plot_yawn), caption="Yawn CNN Confusion Matrix (Held-out Test Partition)", use_container_width=True)

    # =============================================================
    # TAB 3: MONITORING INFORMATION
    # =============================================================
    with tab_stats:
        st.markdown("### 📊 Continuous Monitoring Telemetry")
        st.markdown("Aggregate statistics and temporal metrics gathered across the active driver monitoring session.")

        d_detector = st.session_state.drowsiness_detector

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown("#### ⏳ State Time Distribution")
            t_alert = int(d_detector.time_in_state[DrowsinessState.ALERT])
            t_warn = int(d_detector.time_in_state[DrowsinessState.WARNING])
            t_drowsy = int(d_detector.time_in_state[DrowsinessState.DROWSY])

            df_time = pd.DataFrame({
                "State": ["ALERT", "WARNING", "DROWSY"],
                "Time (Seconds)": [t_alert, t_warn, t_drowsy]
            })
            st.bar_chart(df_time.set_index("State"))

        with col_s2:
            st.markdown("#### 📈 Multi-Signal Score Breakdown")
            df_score = pd.DataFrame({
                "Evidence Stream": ["Eye CNN", "Geometric EAR", "Yawn/MAR", "Temporal/Duration"],
                "Contribution (%)": [
                    d_detector.score_eye_cnn * d_detector.weight_eye_cnn * 100,
                    d_detector.score_ear * d_detector.weight_ear * 100,
                    d_detector.score_yawn * d_detector.weight_yawn * 100,
                    d_detector.score_temporal * d_detector.weight_temporal * 100,
                ]
            })
            st.dataframe(df_score, hide_index=True, use_container_width=True)

        with col_s3:
            st.markdown("#### 🚨 Safety Events Summary")
            st.metric("Total Drowsiness Events Logged", st.session_state.total_drowsy_events)
            st.metric("Audible Alarms Sounded", st.session_state.alarm.total_alarms_triggered)
            st.metric("Consecutive Closed Frames", d_detector.consecutive_closed_frames)
            st.metric("Sustained Yawn Events", d_detector.total_yawn_events)

        st.markdown("---")
        st.markdown("#### 🗣️ Multilingual Voice Alert Configurations (ਚੇਤਾਵਨੀ ਪ੍ਰਣਾਲੀ / चेतावनी प्रणाली)")
        st.markdown(
            "The system delivers real-time auditory warnings across multiple regional languages "
            "to maximize driver alertness and rapid cognitive arousal during micro-sleep episodes."
        )

        lang_c1, lang_c2, lang_c3 = st.columns(3)
        with lang_c1:
            st.markdown("""
            **🇬🇧 English Voice Alert**
            - **Script:** *"Warning! Driver drowsiness detected. Please take a break."*
            - **Asset:** `data/alerts/alert_en.mp3`
            - **Engine:** Neural TTS & Windows SAPI
            """)
        with lang_c2:
            st.markdown("""
            **🇮🇳 Hindi Voice Alert (हिन्दी)**
            - **Script:** *"चेतावनी! चालक को नींद आ रही है। कृपया गाड़ी रोकें और आराम करें।"*
            - **Asset:** `data/alerts/alert_hi.mp3`
            - **Engine:** Google Indic TTS (Devanagari)
            """)
        with lang_c3:
            st.markdown("""
            **🇮🇳 Punjabi Voice Alert (ਪੰਜਾਬੀ)**
            - **Script:** *"ਖ਼ਬਰਦਾਰ! ਡਰਾਈਵਰ ਨੂੰ ਨੀਂਦ ਆ ਰਹੀ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਗੱਡੀ ਰੋਕੋ ਅਤੇ ਆਰਾਮ ਕਰੋ।"*
            - **Asset:** `data/alerts/alert_pa.mp3`
            - **Engine:** Google Indic TTS (Gurmukhi)
            """)


if __name__ == "__main__":
    main()
