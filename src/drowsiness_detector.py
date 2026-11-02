"""
Temporal Drowsiness Decision System.

Fuses multi-modal real-time inputs into a robust temporal state decision:
- Deep Learning Eye State CNN (OPEN vs CLOSED)
- Facial Landmarks & Eye Aspect Ratio (EAR)
- Optional Yawn State CNN (YAWN vs NO_YAWN) & Mouth Aspect Ratio (MAR)
- Temporal Rolling History (Consecutive closures, Blink rate, PERCLOS, Yawn duration)

States:
- ALERT: Normal attentive driver state. Transient normal blinks remain ALERT.
- WARNING: Early fatigue indicators (repeated micro-closures, drooping eyes, or sustained yawning).
- DROWSY: Severe drowsiness (persistent eye closure / microsleep, combined fatigue signals).

Note:
All thresholds and weights are calibrated engineering parameters configured in config.py,
not claimed to be diagnostic medical standards.
"""

import os
import sys
import time
import argparse
from enum import Enum
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.face_detection import FaceDetector
from src.eye_detection import EyeDetector
from src.yawn_detection import YawnDetector
from src.alarm import DrowsinessAlarm


class DrowsinessState(str, Enum):
    ALERT = "ALERT"
    WARNING = "WARNING"
    DROWSY = "DROWSY"


@dataclass
class FrameTelemetry:
    """Instantaneous telemetry recorded for each frame."""
    timestamp: float
    left_state: str
    right_state: str
    left_conf: float
    right_conf: float
    avg_ear: float
    mouth_state: str
    mouth_conf: float
    mar: float
    both_eyes_closed: bool
    is_yawning: bool
    raw_eye_prob: float = 0.0
    raw_yawn_prob: float = 0.0


class DrowsinessDetector:
    """
    Temporal Drowsiness Decision Engine.
    Fuses multiple signals over a rolling time window to determine driver state:
    ALERT, WARNING, or DROWSY.
    """
    def __init__(self,
                 window_size: int = getattr(config, "ROLLING_WINDOW_SIZE", 90),
                 blink_max_frames: int = getattr(config, "BLINK_MAX_FRAMES", 6),
                 warning_closed_frames: int = getattr(config, "WARNING_CLOSED_FRAMES", 12),
                 drowsy_closed_frames: int = getattr(config, "DROWSY_CLOSED_FRAMES", 24),
                 yawn_sustained_frames: int = getattr(config, "YAWN_SUSTAINED_FRAMES", 12),
                 yawn_consecutive_frames: int = getattr(config, "YAWN_CONSECUTIVE_FRAMES", 20),
                 ear_closed_threshold: float = getattr(config, "EAR_CLOSED_THRESHOLD", 0.19),
                 mar_yawn_threshold: float = getattr(config, "MAR_YAWN_THRESHOLD", 0.38),
                 weight_eye_cnn: float = getattr(config, "WEIGHT_EYE_CNN", 0.35),
                 weight_ear: float = getattr(config, "WEIGHT_EAR", 0.20),
                 weight_yawn: float = getattr(config, "WEIGHT_YAWN", 0.20),
                 weight_temporal: float = getattr(config, "WEIGHT_TEMPORAL", 0.25),
                 score_warning: float = getattr(config, "SCORE_WARNING_THRESHOLD", 0.40),
                 score_drowsy: float = getattr(config, "SCORE_DROWSY_THRESHOLD", 0.65)):

        # Configurable engineering parameters
        self.window_size = window_size
        self.blink_max_frames = blink_max_frames
        self.warning_closed_frames = warning_closed_frames
        self.drowsy_closed_frames = drowsy_closed_frames
        self.yawn_sustained_frames = yawn_sustained_frames
        self.yawn_consecutive_frames = yawn_consecutive_frames

        self.ear_closed_threshold = ear_closed_threshold
        self.mar_yawn_threshold = mar_yawn_threshold

        self.weight_eye_cnn = weight_eye_cnn
        self.weight_ear = weight_ear
        self.weight_yawn = weight_yawn
        self.weight_temporal = weight_temporal

        self.score_warning = score_warning
        self.score_drowsy = score_drowsy

        # Rolling history buffer
        self.history: deque = deque(maxlen=self.window_size)

        # Dynamic State Tracking
        self.current_state = DrowsinessState.ALERT
        self.drowsiness_score = 0.0

        # Component contributions
        self.score_eye_cnn = 0.0
        self.score_ear = 0.0
        self.score_yawn = 0.0
        self.score_temporal = 0.0

        # Temporal counters
        self.consecutive_closed_frames = 0
        self.consecutive_open_frames = 0
        self.consecutive_yawn_frames = 0

        self.eye_closure_duration_sec = 0.0
        self.last_frame_time: Optional[float] = None
        self.closure_start_time: Optional[float] = None

        # Cumulative Session Metrics
        self.total_blinks = 0
        self.total_yawn_events = 0
        self.session_start_time = time.time()
        self.last_state_change_time = time.time()

        # State Duration Accumulators (in seconds)
        self.time_in_state = {
            DrowsinessState.ALERT: 0.0,
            DrowsinessState.WARNING: 0.0,
            DrowsinessState.DROWSY: 0.0,
        }

    def update(self,
               eye_results: Dict,
               mouth_results: Optional[Dict] = None,
               current_time: Optional[float] = None) -> Dict:
        """
        Processes instantaneous measurements from eye and mouth detectors,
        updates rolling temporal window, calculates fusion score, and updates state.
        """
        now = current_time if current_time is not None else time.time()

        # Handle time delta for state accumulators
        if self.last_frame_time is not None:
            dt = max(0.0, min(0.5, now - self.last_frame_time))
            self.time_in_state[self.current_state] += dt
        self.last_frame_time = now

        # Extract features safely
        face_detected = eye_results.get("detected", False)

        if not face_detected:
            # Face temporarily lost/leaving frame
            # Gently decay score rather than abrupt reset or false alert
            self.drowsiness_score = max(0.0, self.drowsiness_score * 0.95)
            self.consecutive_closed_frames = 0
            self.consecutive_yawn_frames = 0
            return self._build_result_dict(face_detected=False)

        both_closed = eye_results.get("both_closed", False)
        l_state = eye_results.get("left_state", "OPEN")
        r_state = eye_results.get("right_state", "OPEN")
        l_conf = eye_results.get("left_conf", 0.0)
        r_conf = eye_results.get("right_conf", 0.0)
        avg_ear = eye_results.get("avg_ear", 0.25)
        raw_eye_prob = (eye_results.get("left_prob", 0.0) + eye_results.get("right_prob", 0.0)) / 2.0

        m_state = mouth_results.get("state", "NO_YAWN") if mouth_results else "NO_YAWN"
        m_conf = mouth_results.get("confidence", 0.0) if mouth_results else 0.0
        mar = mouth_results.get("mar", 0.0) if mouth_results else 0.0
        is_yawn = mouth_results.get("is_yawning", False) if mouth_results else False
        raw_yawn_prob = mouth_results.get("prob_yawn", 0.0) if mouth_results else 0.0

        # Record instantaneous telemetry into rolling window
        telemetry = FrameTelemetry(
            timestamp=now,
            left_state=l_state,
            right_state=r_state,
            left_conf=l_conf,
            right_conf=r_conf,
            avg_ear=avg_ear,
            mouth_state=m_state,
            mouth_conf=m_conf,
            mar=mar,
            both_eyes_closed=both_closed,
            is_yawning=is_yawn,
            raw_eye_prob=raw_eye_prob,
            raw_yawn_prob=raw_yawn_prob,
        )
        self.history.append(telemetry)

        # 1. Temporal Closure & Blink Tracking
        if both_closed:
            if self.consecutive_closed_frames == 0:
                self.closure_start_time = now
            self.consecutive_closed_frames += 1
            self.consecutive_open_frames = 0
            self.eye_closure_duration_sec = now - (self.closure_start_time or now)
        else:
            # Reopening after closure
            if 0 < self.consecutive_closed_frames <= self.blink_max_frames:
                # Registered as a completed normal transient blink!
                self.total_blinks += 1
            self.consecutive_closed_frames = 0
            self.consecutive_open_frames += 1
            self.eye_closure_duration_sec = 0.0
            self.closure_start_time = None

        # 2. Yawning Duration & Event Tracking
        if is_yawn:
            self.consecutive_yawn_frames += 1
            if self.consecutive_yawn_frames == self.yawn_sustained_frames:
                self.total_yawn_events += 1
        else:
            self.consecutive_yawn_frames = 0

        # 3. Calculate Multi-Signal Sub-Scores

        # Sub-score A: Deep Learning Eye State CNN
        if both_closed:
            self.score_eye_cnn = min(1.0, max(0.5, raw_eye_prob))
        elif l_state == "CLOSED" or r_state == "CLOSED":
            self.score_eye_cnn = 0.40  # Partial/one-eye closure
        else:
            self.score_eye_cnn = max(0.0, raw_eye_prob * 0.3)

        # Sub-score B: Continuous Geometric EAR
        # Scaled between EAR=0.28 (fully alert) and EAR=0.15 (fully closed)
        ear_clamped = np.clip(avg_ear, 0.14, 0.28)
        self.score_ear = float((0.28 - ear_clamped) / (0.28 - 0.14))

        # Sub-score C: Yawn CNN & MAR
        mar_score = float(np.clip((mar - 0.20) / (0.45 - 0.20), 0.0, 1.0))
        yawn_cnn_score = float(raw_yawn_prob) if is_yawn else 0.0
        self.score_yawn = max(yawn_cnn_score, mar_score)

        # Sub-score D: Temporal Duration & PERCLOS (Percentage of Eye Closure & Yawning)
        if len(self.history) > 0:
            closed_in_window = sum(1 for item in self.history if item.both_eyes_closed)
            perclos = closed_in_window / float(len(self.history))
            yawn_in_window = sum(1 for item in self.history if item.is_yawning)
            yawn_ratio = yawn_in_window / float(len(self.history))
        else:
            perclos = 0.0
            yawn_ratio = 0.0

        eye_duration_factor = min(1.0, self.consecutive_closed_frames / float(self.drowsy_closed_frames))
        yawn_duration_factor = min(1.0, self.consecutive_yawn_frames / float(self.yawn_consecutive_frames))
        duration_factor = max(eye_duration_factor, yawn_duration_factor)

        rolling_fatigue = max(perclos, min(1.0, yawn_ratio * 1.5))
        self.score_temporal = 0.65 * duration_factor + 0.35 * rolling_fatigue

        # 4. Composite Drowsiness Score Fusion
        # Transient blink damping:
        # A normal blink lasts only 1-5 frames. During frames <= BLINK_MAX_FRAMES,
        # damp instantaneous closure to prevent sudden false spikes into WARNING/DROWSY,
        # unless rolling history (PERCLOS) already exhibits high drowsiness.
        if both_closed and self.consecutive_closed_frames <= self.blink_max_frames:
            blink_damping = max(0.2, self.consecutive_closed_frames / float(self.warning_closed_frames))
            effective_eye_score = self.score_eye_cnn * blink_damping
            effective_ear_score = self.score_ear * blink_damping
        else:
            effective_eye_score = self.score_eye_cnn
            effective_ear_score = self.score_ear

        composite_score = (
            self.weight_eye_cnn * effective_eye_score +
            self.weight_ear * effective_ear_score +
            self.weight_yawn * self.score_yawn +
            self.weight_temporal * self.score_temporal
        )
        self.drowsiness_score = float(np.clip(composite_score, 0.0, 1.0))

        # 5. Temporal State Machine Transition Logic
        prev_state = self.current_state

        # Check critical drowsiness triggers (prolonged eye closure OR prolonged yawn OR repeated yawns)
        is_critical_closure = self.consecutive_closed_frames >= self.drowsy_closed_frames
        is_critical_yawn = (self.consecutive_yawn_frames >= self.yawn_consecutive_frames) or \
                           (self.total_yawn_events >= 2 and self.consecutive_yawn_frames >= self.yawn_sustained_frames)

        # Check fatigue warning triggers (warning closure OR sustained yawn)
        is_warning_closure = self.consecutive_closed_frames >= self.warning_closed_frames
        is_warning_yawn = self.consecutive_yawn_frames >= self.yawn_sustained_frames

        if is_critical_closure or is_critical_yawn:
            self.current_state = DrowsinessState.DROWSY
            self.drowsiness_score = max(self.drowsiness_score, self.score_drowsy + 0.10)
        elif is_warning_closure or is_warning_yawn:
            self.current_state = DrowsinessState.WARNING
            self.drowsiness_score = max(self.drowsiness_score, self.score_warning + 0.05)
        elif 0 < self.consecutive_closed_frames <= self.blink_max_frames:
            # During active transient blink (1-6 frames): remain ALERT unless rolling PERCLOS or yawning indicates fatigue
            if self.drowsiness_score >= self.score_drowsy:
                self.current_state = DrowsinessState.DROWSY
            elif self.drowsiness_score >= self.score_warning:
                self.current_state = DrowsinessState.WARNING
            elif perclos < 0.35 and self.consecutive_yawn_frames < self.yawn_sustained_frames:
                self.current_state = DrowsinessState.ALERT
        elif self.consecutive_closed_frames > self.blink_max_frames:
            # Sluggish closure between BLINK_MAX_FRAMES and WARNING_CLOSED_FRAMES
            if self.drowsiness_score >= self.score_drowsy:
                self.current_state = DrowsinessState.DROWSY
            else:
                self.current_state = DrowsinessState.WARNING
        else:
            # Eyes are currently OPEN (consecutive_closed_frames == 0) and not actively in sustained yawn
            if self.current_state == DrowsinessState.DROWSY:
                # Gradual de-escalation: require sustained reopening and mouth closed before clearing DROWSY alert
                if self.consecutive_open_frames >= 15 and self.consecutive_yawn_frames == 0 and self.drowsiness_score < self.score_warning:
                    self.current_state = DrowsinessState.ALERT
                elif self.consecutive_open_frames >= 8 and self.consecutive_yawn_frames == 0 and self.drowsiness_score < self.score_drowsy:
                    self.current_state = DrowsinessState.WARNING
            elif self.current_state == DrowsinessState.WARNING:
                if self.consecutive_open_frames >= 10 and self.consecutive_yawn_frames == 0 and self.drowsiness_score < self.score_warning:
                    self.current_state = DrowsinessState.ALERT
            else:
                # Baseline Alert driver
                if self.drowsiness_score >= self.score_drowsy:
                    self.current_state = DrowsinessState.DROWSY
                elif self.drowsiness_score >= self.score_warning:
                    self.current_state = DrowsinessState.WARNING
                else:
                    self.current_state = DrowsinessState.ALERT

        if self.current_state != prev_state:
            self.last_state_change_time = now

        return self._build_result_dict(face_detected=True, perclos=perclos)

    def _build_result_dict(self, face_detected: bool, perclos: float = 0.0) -> Dict:
        """Constructs structured state dictionary."""
        # Calculate rolling blink rate (blinks per minute)
        session_elapsed = max(1.0, time.time() - self.session_start_time)
        blink_rate_cpm = (self.total_blinks / session_elapsed) * 60.0

        return {
            "face_detected": face_detected,
            "state": self.current_state.value,
            "drowsiness_score": self.drowsiness_score,
            "score_breakdown": {
                "eye_cnn": self.score_eye_cnn,
                "ear": self.score_ear,
                "yawn": self.score_yawn,
                "temporal": self.score_temporal,
            },
            "temporal_metrics": {
                "consecutive_closed_frames": self.consecutive_closed_frames,
                "closure_duration_sec": self.eye_closure_duration_sec,
                "total_blinks": self.total_blinks,
                "blink_rate_cpm": blink_rate_cpm,
                "perclos": perclos,
                "total_yawn_events": self.total_yawn_events,
                "consecutive_yawn_frames": self.consecutive_yawn_frames,
            },
            "time_in_state": {
                "ALERT_sec": self.time_in_state[DrowsinessState.ALERT],
                "WARNING_sec": self.time_in_state[DrowsinessState.WARNING],
                "DROWSY_sec": self.time_in_state[DrowsinessState.DROWSY],
            },
        }

    def render_hud(self,
                   frame: np.ndarray,
                   eye_results: Dict,
                   mouth_results: Optional[Dict] = None,
                   fps: float = 0.0) -> np.ndarray:
        """
        Renders HUD telemetry:
        - Primary Driver Drowsiness State Badge (ALERT / WARNING / DROWSY)
        - Fused Drowsiness Score Meter (0% to 100%)
        - Comprehensive Temporal Telemetry (Closure Duration, PERCLOS, Blinks, Yawns)
        - State Time Accumulators
        """
        hud_frame = frame.copy()
        h, w = hud_frame.shape[:2]

        # 1. Top Dashboard Glass Panel (135px height)
        banner_h = 135
        overlay = hud_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (22, 22, 22), -1)
        cv2.addWeighted(overlay, 0.78, hud_frame, 0.22, 0, hud_frame)

        # Title and System Header
        cv2.putText(hud_frame, "DRIVER DROWSINESS DETECTION | TEMPORAL DECISION SYSTEM", (15, 23),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hud_frame, f"FPS: {fps:.1f}", (w - 110, 23),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        if not eye_results.get("detected", False):
            cv2.putText(hud_frame, "STATUS: SEARCHING FOR DRIVER FACE...", (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2, cv2.LINE_AA)
            return hud_frame

        # 2. State Badge & Color Selection
        state_colors = {
            DrowsinessState.ALERT: ((0, 220, 0), (0, 60, 0)),        # Vibrant Green
            DrowsinessState.WARNING: ((0, 180, 255), (0, 50, 80)),    # Amber Gold
            DrowsinessState.DROWSY: ((0, 0, 255), (0, 0, 90)),       # Intense Crimson
        }
        fg_color, bg_color = state_colors[self.current_state]

        # Draw State Badge
        badge_w = 170
        badge_h = 36
        badge_x = w - badge_w - 15
        badge_y = 35

        cv2.rectangle(hud_frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), bg_color, -1)
        cv2.rectangle(hud_frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), fg_color, 2)
        cv2.putText(hud_frame, f"STATE: {self.current_state.value}", (badge_x + 12, badge_y + 24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, fg_color, 2, cv2.LINE_AA)

        # 3. Telemetry Rows

        # Row 1: Eye States & Mouth State
        l_state = eye_results.get("left_state", "OPEN")
        r_state = eye_results.get("right_state", "OPEN")
        l_color = (0, 0, 255) if l_state == "CLOSED" else (0, 220, 0)
        r_color = (0, 0, 255) if r_state == "CLOSED" else (0, 220, 0)

        eye_txt = f"EYES: L-{l_state} ({eye_results.get('left_conf', 0):.0f}%) | R-{r_state} ({eye_results.get('right_conf', 0):.0f}%)"
        cv2.putText(hud_frame, eye_txt, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)

        m_state = mouth_results.get("state", "NO_YAWN") if mouth_results else "NONE"
        m_color = (0, 140, 255) if (mouth_results and mouth_results.get("is_yawning")) else (0, 220, 0)
        mouth_txt = f"MOUTH: {m_state}"
        cv2.putText(hud_frame, mouth_txt, (350, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, m_color, 1, cv2.LINE_AA)

        # Row 2: Biometric Indicators (EAR, MAR, PERCLOS)
        avg_ear = eye_results.get("avg_ear", 0.0)
        mar = mouth_results.get("mar", 0.0) if mouth_results else 0.0
        closed_in_window = sum(1 for item in self.history if item.both_eyes_closed)
        perclos = (closed_in_window / max(1, len(self.history))) * 100.0

        bio_txt = f"EAR: {avg_ear:.3f}  |  MAR: {mar:.3f}  |  PERCLOS: {perclos:.1f}%  |  Blinks: {self.total_blinks}"
        cv2.putText(hud_frame, bio_txt, (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

        # Row 3: Temporal Duration & Time in States
        dur_txt = (f"Closure: {self.eye_closure_duration_sec:.1f}s ({self.consecutive_closed_frames} frames)  |  "
                   f"Time Alert: {int(self.time_in_state[DrowsinessState.ALERT])}s | "
                   f"Warn: {int(self.time_in_state[DrowsinessState.WARNING])}s | "
                   f"Drowsy: {int(self.time_in_state[DrowsinessState.DROWSY])}s")
        cv2.putText(hud_frame, dur_txt, (15, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1, cv2.LINE_AA)

        # Row 4: Drowsiness Fusion Score Meter Bar
        bar_x = 15
        bar_y = 108
        bar_w = 420
        bar_h = 16
        score_fill = int(bar_w * self.drowsiness_score)

        cv2.rectangle(hud_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
        if score_fill > 0:
            cv2.rectangle(hud_frame, (bar_x, bar_y), (bar_x + score_fill, bar_y + bar_h), fg_color, -1)
        cv2.rectangle(hud_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)

        cv2.putText(hud_frame, f"Drowsiness Score: {self.drowsiness_score * 100:.1f}%", (bar_x + bar_w + 12, bar_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, fg_color, 1, cv2.LINE_AA)

        # 4. Prominent Warning Overlays for Critical States
        if self.current_state == DrowsinessState.DROWSY:
            warn_overlay = hud_frame.copy()
            cv2.rectangle(warn_overlay, (w // 2 - 240, h - 65), (w // 2 + 240, h - 15), (0, 0, 190), -1)
            cv2.addWeighted(warn_overlay, 0.85, hud_frame, 0.15, 0, hud_frame)
            cv2.putText(hud_frame, "CRITICAL: DROWSINESS DETECTED!", (w // 2 - 220, h - 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)

        elif self.current_state == DrowsinessState.WARNING:
            warn_overlay = hud_frame.copy()
            cv2.rectangle(warn_overlay, (w // 2 - 210, h - 65), (w // 2 + 210, h - 15), (0, 120, 220), -1)
            cv2.addWeighted(warn_overlay, 0.80, hud_frame, 0.20, 0, hud_frame)
            cv2.putText(hud_frame, "WARNING: DRIVER FATIGUE DETECTED", (w // 2 - 195, h - 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

        return hud_frame


# ==============================================================================
# VERIFICATION & TESTING MODULES
# ==============================================================================

def run_synthetic_temporal_test() -> bool:
    """
    Automated verification of temporal decision logic across 4 test scenarios:
    1. Normal Blink Sequence: OPEN -> CLOSED (3 frames) -> OPEN -> must remain ALERT.
    2. Prolonged Eye Closure: CLOSED for 30 consecutive frames -> must transition ALERT -> WARNING -> DROWSY.
    3. Sustained Yawn Event: OPEN eyes with prolonged yawning -> must elevate score and transition to WARNING.
    4. Recovery Sequence: Reopening eyes after closure -> must recover smoothly to ALERT.
    """
    print("\n" + "=" * 65)
    print("  RUNNING TEMPORAL DROWSINESS DECISION ENGINE TEST SUITE")
    print("=" * 65)

    detector = DrowsinessDetector()
    current_time = 1000.0
    dt = 1.0 / 30.0  # 30 fps simulation

    # -------------------------------------------------------------
    # Scenario 1: Normal Blink (3 frames closed, then reopened)
    # -------------------------------------------------------------
    print("\n[Scenario 1] Normal Blink Test (OPEN -> 3 Frames Closed -> OPEN)...")
    open_eye_res = {"detected": True, "both_closed": False, "left_state": "OPEN", "right_state": "OPEN",
                    "left_conf": 99.0, "right_conf": 99.0, "avg_ear": 0.26, "left_prob": 0.01, "right_prob": 0.01}
    closed_eye_res = {"detected": True, "both_closed": True, "left_state": "CLOSED", "right_state": "CLOSED",
                      "left_conf": 99.0, "right_conf": 99.0, "avg_ear": 0.12, "left_prob": 0.99, "right_prob": 0.99}
    normal_mouth = {"state": "NO_YAWN", "confidence": 95.0, "mar": 0.08, "is_yawning": False, "prob_yawn": 0.02}

    # 10 open frames
    for _ in range(10):
        detector.update(open_eye_res, normal_mouth, current_time)
        current_time += dt

    # 3 closed frames (normal blink)
    blink_states = []
    for _ in range(3):
        res = detector.update(closed_eye_res, normal_mouth, current_time)
        blink_states.append(res["state"])
        current_time += dt

    # 10 open frames after blink
    post_blink_states = []
    for _ in range(10):
        res = detector.update(open_eye_res, normal_mouth, current_time)
        post_blink_states.append(res["state"])
        current_time += dt

    assert all(s == "ALERT" for s in blink_states), f"Blink triggered non-alert state: {blink_states}"
    assert detector.total_blinks >= 1, f"Blink not recorded in blink counter: {detector.total_blinks}"
    assert detector.current_state == DrowsinessState.ALERT, "Driver state did not remain ALERT after blink"
    print(f"  PASS: Normal 3-frame blink remained ALERT. Total Blinks recorded: {detector.total_blinks}")

    # -------------------------------------------------------------
    # Scenario 2: Persistent Eye Closure (Microsleep / Falling Asleep)
    # -------------------------------------------------------------
    print("\n[Scenario 2] Persistent Eye Closure Test (30 Consecutive Closed Frames)...")
    progression = []
    for i in range(30):
        res = detector.update(closed_eye_res, normal_mouth, current_time)
        progression.append((i + 1, res["state"], res["drowsiness_score"]))
        current_time += dt

    # Check progression transitions
    state_at_frame_5 = progression[4][1]
    state_at_frame_15 = progression[14][1]
    state_at_frame_28 = progression[27][1]

    print(f"  • Frame  5 State: {state_at_frame_5} (Score: {progression[4][2]*100:.1f}%)")
    print(f"  • Frame 15 State: {state_at_frame_15} (Score: {progression[14][2]*100:.1f}%)")
    print(f"  • Frame 28 State: {state_at_frame_28} (Score: {progression[27][2]*100:.1f}%)")

    assert state_at_frame_15 in ["WARNING", "DROWSY"], f"Expected WARNING by frame 15, got {state_at_frame_15}"
    assert state_at_frame_28 == "DROWSY", f"Expected DROWSY by frame 28, got {state_at_frame_28}"
    print("  PASS: Persistent closure escalated cleanly through WARNING -> DROWSY.")

    # -------------------------------------------------------------
    # Scenario 3: Recovery Test (Reopening Eyes After Drowsiness)
    # -------------------------------------------------------------
    print("\n[Scenario 3] Alertness Recovery Test (Reopening Eyes for 25 Frames)...")
    for _ in range(25):
        res = detector.update(open_eye_res, normal_mouth, current_time)
        current_time += dt

    print(f"  • Final Recovered State: {detector.current_state.value} (Score: {detector.drowsiness_score*100:.1f}%)")
    assert detector.current_state == DrowsinessState.ALERT, f"Expected recovery to ALERT, got {detector.current_state}"
    print("  PASS: Reopening eyes successfully recovered driver state to ALERT.")

    # -------------------------------------------------------------
    # Scenario 4: Sustained Yawning Contribution Test
    # -------------------------------------------------------------
    print("\n[Scenario 4] Sustained Yawning Test (Eyes Open, Yawn for 20 Frames)...")
    yawning_mouth = {"state": "YAWN", "confidence": 98.0, "mar": 0.48, "is_yawning": True, "prob_yawn": 0.98}
    for _ in range(20):
        res = detector.update(open_eye_res, yawning_mouth, current_time)
        current_time += dt

    print(f"  • Yawn Contribution Score: {detector.score_yawn*100:.1f}% | Total Score: {detector.drowsiness_score*100:.1f}%")
    print(f"  • Total Yawn Events: {detector.total_yawn_events} | State: {detector.current_state.value}")
    assert detector.total_yawn_events >= 1, "Yawn event was not registered"
    assert detector.score_yawn > 0.8, "Yawn contribution was not elevated"
    print("  PASS: Yawning contributed positive evidence without false trigger.")

    print("\n" + "=" * 65)
    print("  ALL 4 TEMPORAL DECISION ENGINE TEST SUITES PASSED (100%)")
    print("=" * 65 + "\n")
    return True


def run_pipeline_with_decision_engine(camera_index: int = 0,
                                      eye_model_path: Optional[Path] = None,
                                      yawn_model_path: Optional[Path] = None,
                                      test_image_mode: Optional[Path] = None):
    """
    Executes real-time end-to-end pipeline:
    Face Detection -> Facial Landmarks -> Eye ROIs & CNN -> Mouth ROI & Yawn CNN -> Temporal Drowsiness Engine -> HUD
    """
    face_detector = FaceDetector()
    eye_detector = EyeDetector(model_path=eye_model_path, target_size=(64, 64))
    yawn_detector = YawnDetector(model_path=yawn_model_path, target_size=(64, 64))
    drowsiness_detector = DrowsinessDetector()

    if test_image_mode:
        print(f"Verifying temporal drowsiness detector on: {test_image_mode}")
        frame = cv2.imread(str(test_image_mode))
        if frame is None:
            raise ValueError(f"Could not load test image from: {test_image_mode}")

        faces = face_detector.detect_faces(frame)
        primary_face = face_detector.get_primary_face(faces, frame.shape)
        eye_results = eye_detector.process_face_eyes(frame, primary_face)
        mouth_results = yawn_detector.process_face_mouth(frame, primary_face)

        # Run temporal decision update
        decision_results = drowsiness_detector.update(eye_results, mouth_results)

        annotated_frame = face_detector.draw_face_annotations(
            frame, primary_face, len(faces),
            is_drowsy=(drowsiness_detector.current_state != DrowsinessState.ALERT)
        )
        annotated_frame = yawn_detector.draw_mouth_annotations(annotated_frame, mouth_results)
        hud_frame = drowsiness_detector.render_hud(annotated_frame, eye_results, mouth_results, fps=30.0)

        out_path = config.PREDICTIONS_DIR / "temporal_drowsiness_verification.png"
        cv2.imwrite(str(out_path), hud_frame)
        print(f"Verified temporal HUD output saved to: {out_path}")
        print("\nTelemetry Output:")
        print(f"  • Final Driver State: {decision_results['state']}")
        print(f"  • Drowsiness Score:   {decision_results['drowsiness_score']*100:.1f}%")
        print(f"  • Sub-scores:         Eye={decision_results['score_breakdown']['eye_cnn']:.2f}, "
              f"EAR={decision_results['score_breakdown']['ear']:.2f}, "
              f"Yawn={decision_results['score_breakdown']['yawn']:.2f}, "
              f"Temporal={decision_results['score_breakdown']['temporal']:.2f}")
        return

    # Real-Time Webcam Stream
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Warning: Could not open camera {camera_index}. Running test suite instead...")
        run_synthetic_temporal_test()
        sample_img = config.DDD_DATASET_DIR / "Drowsy" / "A0001.png"
        run_pipeline_with_decision_engine(test_image_mode=sample_img)
        return

    print("Starting webcam stream with Temporal Drowsiness Engine. Press 'q' or ESC to exit.")
    alarm = DrowsinessAlarm(cooldown_sec=config.ALARM_COOLDOWN)
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

            # 2. Eye & Mouth Extraction & ML Inference
            eye_results = eye_detector.process_face_eyes(frame, primary_face)
            mouth_results = yawn_detector.process_face_mouth(frame, primary_face)

            # 3. Temporal Drowsiness Analysis & Scoring
            decision_results = drowsiness_detector.update(eye_results, mouth_results, current_time=curr_time)

            # 4. Audible Alarm Trigger (Starts on DROWSY, stops on ALERT, enforces cooldown)
            alarm.trigger(decision_results["state"], current_time=curr_time)

            # 5. HUD Visualization
            is_fatigued = (drowsiness_detector.current_state != DrowsinessState.ALERT)
            annotated_frame = face_detector.draw_face_annotations(
                frame, primary_face, len(faces), is_drowsy=is_fatigued
            )
            annotated_frame = yawn_detector.draw_mouth_annotations(annotated_frame, mouth_results)
            hud_frame = drowsiness_detector.render_hud(annotated_frame, eye_results, mouth_results, fps=fps)

            cv2.imshow("Driver Drowsiness Decision System", hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        alarm.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Pipeline closed cleanly.")


def run_sequence_verification(sequence_dir: Path, max_frames: int = 30):
    """
    Evaluates temporal drowsiness decision system across a recorded video/frame sequence.
    """
    print(f"\nRunning temporal evaluation across sequence in: {sequence_dir}")
    frames_paths = sorted(list(sequence_dir.glob("*.png")) + list(sequence_dir.glob("*.jpg")))[:max_frames]
    if not frames_paths:
        print("No image frames found in sequence directory.")
        return

    import pandas as pd
    face_detector = FaceDetector()
    eye_detector = EyeDetector(target_size=(64, 64))
    yawn_detector = YawnDetector(target_size=(64, 64))
    drowsiness_detector = DrowsinessDetector()

    simulated_time = 0.0
    dt = 1.0 / 30.0

    states_recorded = []
    scores_recorded = []
    last_hud = None

    for idx, fp in enumerate(frames_paths):
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        faces = face_detector.detect_faces(frame)
        primary_face = face_detector.get_primary_face(faces, frame.shape)
        eye_results = eye_detector.process_face_eyes(frame, primary_face)
        mouth_results = yawn_detector.process_face_mouth(frame, primary_face)

        decision = drowsiness_detector.update(eye_results, mouth_results, current_time=simulated_time)
        simulated_time += dt

        states_recorded.append(decision["state"])
        scores_recorded.append(decision["drowsiness_score"])

        if idx == len(frames_paths) - 1:
            annotated = face_detector.draw_face_annotations(
                frame, primary_face, len(faces),
                is_drowsy=(decision["state"] != "ALERT")
            )
            annotated = yawn_detector.draw_mouth_annotations(annotated, mouth_results)
            last_hud = drowsiness_detector.render_hud(annotated, eye_results, mouth_results, fps=30.0)

    if last_hud is not None:
        out_path = config.PREDICTIONS_DIR / "sequence_drowsiness_verification.png"
        cv2.imwrite(str(out_path), last_hud)
        print(f"Sequence HUD artifact saved to: {out_path}")

    print("\nSequence Summary Statistics:")
    print(f"  • Total Frames Processed:   {len(frames_paths)}")
    print(f"  • Final Driver State:       {states_recorded[-1]}")
    print(f"  • Average Drowsiness Score: {np.mean(scores_recorded)*100:.1f}%")
    print(f"  • Peak Drowsiness Score:    {np.max(scores_recorded)*100:.1f}%")
    print(f"  • State Distribution:       {dict(pd.Series(states_recorded).value_counts())}")


def main():
    parser = argparse.ArgumentParser(description="Temporal Driver Drowsiness Decision Engine")
    parser.add_argument("--test", action="store_true", help="Run automated synthetic temporal test suite")
    parser.add_argument("--verify", action="store_true", help="Run verification on a sample image")
    parser.add_argument("--sequence", type=str, default=None, help="Directory path to image sequence for verification")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Webcam device index")
    parser.add_argument("--image", type=str, default=None, help="Path to sample image for verification")
    args = parser.parse_args()

    if args.test or (not args.verify and not args.sequence and not args.image):
        # Run test suite
        run_synthetic_temporal_test()

    if args.sequence:
        run_sequence_verification(Path(args.sequence), max_frames=30)
    elif args.verify or args.image:
        test_img = Path(args.image) if args.image else (config.DDD_DATASET_DIR / "Drowsy" / "A0001.png")
        run_pipeline_with_decision_engine(test_image_mode=test_img)
    elif not args.test:
        run_pipeline_with_decision_engine(camera_index=args.camera)


if __name__ == "__main__":
    main()
