import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from src.drowsiness_detector import DrowsinessDetector, DrowsinessState
from src.alarm import DrowsinessAlarm

detector = DrowsinessDetector()
alarm = DrowsinessAlarm(enabled=True)

eye_open = {
    'detected': True,
    'both_closed': False,
    'left_state': 'OPEN',
    'right_state': 'OPEN',
    'left_conf': 95.0,
    'right_conf': 95.0,
    'avg_ear': 0.28,
    'left_prob': 0.05,
    'right_prob': 0.05
}

mouth_yawn = {
    'detected': True,
    'state': 'YAWN',
    'confidence': 98.0,
    'prob_yawn': 0.98,
    'mar': 0.55,
    'is_yawning': True
}

mouth_normal = {
    'detected': True,
    'state': 'NO_YAWN',
    'confidence': 95.0,
    'prob_yawn': 0.05,
    'mar': 0.15,
    'is_yawning': False
}

print("1. Simulating 25 frames of yawning:")
for i in range(1, 26):
    res = detector.update(eye_open, mouth_yawn)
    if i in [1, 10, 15, 20, 25]:
        print(f"Yawn Frame {i:2d}: State={res['state']}, Score={res['drowsiness_score']*100:.1f}%, YawnFrames={res['temporal_metrics']['consecutive_yawn_frames']}")

print("\n2. Simulating 35 frames after yawning stops:")
for i in range(1, 36):
    res = detector.update(eye_open, mouth_normal)
    if i in [1, 5, 10, 15, 20, 25, 30, 35]:
        print(f"Post-Yawn Frame {i:2d}: State={res['state']}, Score={res['drowsiness_score']*100:.1f}%, OpenFrames={detector.consecutive_open_frames}")
