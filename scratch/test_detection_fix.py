import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import cv2
import config
from src.face_detection import FaceDetector
from src.eye_detection import EyeDetector
from src.yawn_detection import YawnDetector
from src.drowsiness_detector import DrowsinessDetector
from src.alarm import DrowsinessAlarm

fd = FaceDetector()
ed = EyeDetector()
yd = YawnDetector()
dd = DrowsinessDetector()
alarm = DrowsinessAlarm(cooldown_sec=1.0)

img_open = cv2.imread(str(config.DDD_DATASET_DIR / "Drowsy" / "A0001.png"))
faces_open = fd.detect_faces(img_open)
pf_open = fd.get_primary_face(faces_open, img_open.shape)
eye_open = ed.process_face_eyes(img_open, pf_open)
mouth_open = yd.process_face_mouth(img_open, pf_open)
dd_open = DrowsinessDetector()
res_open = dd_open.update(eye_open, mouth_open)

print("--- Immediate Eye Detection Results for A0001.png (Open Eyes) ---")
print("Left Eye State: ", eye_open["left_state"], f"({eye_open['left_conf']:.1f}%)", "EAR:", f"{eye_open['left_ear']:.3f}")
print("Right Eye State:", eye_open["right_state"], f"({eye_open['right_conf']:.1f}%)", "EAR:", f"{eye_open['right_ear']:.3f}")
print("Both Closed:    ", eye_open["both_closed"])
print("State:          ", res_open["state"], f"(Score={res_open['drowsiness_score']*100:.1f}%)")

img = cv2.imread(str(config.DDD_DATASET_DIR / "Drowsy" / "A0059.png"))
faces = fd.detect_faces(img)
pf = fd.get_primary_face(faces, img.shape)
eye_res = ed.process_face_eyes(img, pf)
mouth_res = yd.process_face_mouth(img, pf)

print("\n--- Immediate Eye Detection Results for A0059.png (Closed Eyes) ---")
print("Left Eye State: ", eye_res["left_state"], f"({eye_res['left_conf']:.1f}%)", "EAR:", f"{eye_res['left_ear']:.3f}")
print("Right Eye State:", eye_res["right_state"], f"({eye_res['right_conf']:.1f}%)", "EAR:", f"{eye_res['right_ear']:.3f}")
print("Both Closed:    ", eye_res["both_closed"])
print("Avg EAR:        ", f"{eye_res['avg_ear']:.3f}")

print("\n--- Simulating 26 Consecutive Frames of Closed Eyes ---")
for f_idx in range(1, 27):
    res = dd.update(eye_res, mouth_res)
    triggered = alarm.trigger(res["state"])
    st = res["state"]
    sc = res["drowsiness_score"] * 100.0
    if st != "ALERT":
        print(f"Frame {f_idx:2d}: State={st:7s} | Score={sc:5.1f}% | AlarmTriggered={triggered}")

print("\nTotal Alarms Triggered:", alarm.total_alarms_triggered)
print("Alarm Status:          ", alarm.get_status()[0])
alarm.stop()
