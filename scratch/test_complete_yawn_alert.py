import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drowsiness_detector import DrowsinessDetector
from src.alarm import DrowsinessAlarm

print("=" * 70)
print("  COMPREHENSIVE END-TO-END YAWN ALERT VERIFICATION TEST")
print("=" * 70)

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

mouth_normal = {
    'detected': True,
    'state': 'NO_YAWN',
    'confidence': 95.0,
    'prob_yawn': 0.05,
    'mar': 0.15,
    'is_yawning': False
}

mouth_talking = {
    'detected': True,
    'state': 'NO_YAWN',
    'confidence': 70.0,
    'prob_yawn': 0.25,
    'mar': 0.26,
    'is_yawning': False
}

mouth_yawn = {
    'detected': True,
    'state': 'YAWN',
    'confidence': 98.0,
    'prob_yawn': 0.98,
    'mar': 0.55,
    'is_yawning': True
}

sim_time = time.time()
dt = 1.0 / 30.0

# 1. Normal Driving Test (30 frames)
print("\n[Phase 1] Normal Driving (30 frames):")
for _ in range(30):
    res = detector.update(eye_open, mouth_normal, current_time=sim_time)
    alarm.trigger(res["state"], current_time=sim_time)
    sim_time += dt

print(f"  State: {res['state']}, Score: {res['drowsiness_score']*100:.1f}%, Alarm: {alarm.is_playing}")
assert res['state'] == "ALERT", f"Expected ALERT, got {res['state']}"
assert not alarm.is_playing, "Alarm should not play during normal driving"
print("  >>> PASS: Normal driving remains ALERT without false alarms.")

# 2. Brief Talking (5 frames mouth open, MAR=0.26)
print("\n[Phase 2] Brief Mouth Opening / Speech (5 frames):")
for _ in range(5):
    res = detector.update(eye_open, mouth_talking, current_time=sim_time)
    alarm.trigger(res["state"], current_time=sim_time)
    sim_time += dt

print(f"  State: {res['state']}, Score: {res['drowsiness_score']*100:.1f}%, Alarm: {alarm.is_playing}")
assert res['state'] == "ALERT", f"Expected ALERT, got {res['state']}"
assert not alarm.is_playing, "Speech should not trigger alarm"
print("  >>> PASS: Natural speech phonemes do not trigger warning or alarm.")

# 3. Sustained Yawn Test (15 frames)
print("\n[Phase 3] Sustained Yawn Progression (1 to 25 frames):")
warning_at_frame = None
drowsy_at_frame = None

for f in range(1, 26):
    res = detector.update(eye_open, mouth_yawn, current_time=sim_time)
    alarm_fired = alarm.trigger(res["state"], current_time=sim_time)
    sim_time += dt

    if res['state'] == "WARNING" and warning_at_frame is None:
        warning_at_frame = f
    if res['state'] == "DROWSY" and drowsy_at_frame is None:
        drowsy_at_frame = f

    if f in [1, 5, 10, 12, 15, 19, 20, 25]:
        print(f"  Frame {f:2d}: State={res['state']}, Score={res['drowsiness_score']*100:.1f}%, YawnFrames={res['temporal_metrics']['consecutive_yawn_frames']}, AlarmPlaying={alarm.is_playing}")

print(f"\n  Transition Milestones:")
print(f"  - WARNING Reached: Frame {warning_at_frame} ({warning_at_frame/30.0:.2f}s)")
print(f"  - DROWSY Reached:  Frame {drowsy_at_frame} ({drowsy_at_frame/30.0:.2f}s)")
print(f"  - Total Yawns Registered: {res['temporal_metrics']['total_yawn_events']}")
print(f"  - Alarm Sounding: {alarm.is_playing}")

assert warning_at_frame == 12, f"Expected WARNING at frame 12, got {warning_at_frame}"
assert drowsy_at_frame == 20, f"Expected DROWSY at frame 20, got {drowsy_at_frame}"
assert alarm.is_playing, "Audible Alarm MUST be sounding at frame 20"
print("  >>> PASS: Sustained yawn escalates smoothly: ALERT -> WARNING -> DROWSY -> ALARM FIRED!")

# 4. Post-Yawn Recovery Test
print("\n[Phase 4] Post-Yawn Recovery (30 frames after mouth closed):")
alarm.stop() # Reset alarm for test
for f in range(1, 31):
    res = detector.update(eye_open, mouth_normal, current_time=sim_time)
    sim_time += dt
    if f in [5, 10, 15, 20, 30]:
        print(f"  Recovery Frame {f:2d}: State={res['state']}, Score={res['drowsiness_score']*100:.1f}%, OpenFrames={detector.consecutive_open_frames}")

print(f"  Final State after recovery: {res['state']}")
assert res['state'] == "ALERT", f"Expected ALERT after recovery, got {res['state']}"
print("  >>> PASS: Clean, smooth recovery to ALERT state.")

# 5. Repeated Yawn Test (2nd Yawn)
print("\n[Phase 5] Repeated Yawn Test (2nd Yawn after earlier yawn):")
second_yawn_drowsy_frame = None
for f in range(1, 16):
    res = detector.update(eye_open, mouth_yawn, current_time=sim_time)
    alarm.trigger(res["state"], current_time=sim_time)
    sim_time += dt
    if res['state'] == "DROWSY" and second_yawn_drowsy_frame is None:
        second_yawn_drowsy_frame = f
    if f in [5, 10, 12, 15]:
        print(f"  2nd Yawn Frame {f:2d}: State={res['state']}, Score={res['drowsiness_score']*100:.1f}%, TotalEvents={res['temporal_metrics']['total_yawn_events']}")

print(f"  - Second Yawn reached DROWSY at Frame {second_yawn_drowsy_frame} (immediate escalation due to cumulative fatigue!)")
assert second_yawn_drowsy_frame == 12, f"Expected 2nd yawn to escalate at frame 12, got {second_yawn_drowsy_frame}"
print("  >>> PASS: Repeated yawning correctly triggers rapid fatigue escalation!")

print("\n" + "=" * 70)
print("  ALL YAWN ALERT TESTS PASSED WITH 100% RELIABILITY!")
print("=" * 70)

