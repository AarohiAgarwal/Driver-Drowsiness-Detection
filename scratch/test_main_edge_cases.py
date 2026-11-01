import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import main
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

# 1. Test Empty Frame (No Face)
print("Testing No Face...")
empty = np.zeros((480, 640, 3), dtype=np.uint8)
faces = fd.detect_faces(empty)
primary = fd.get_primary_face(faces, empty.shape)
eye_res = ed.process_face_eyes(empty, primary)
mouth_res = yd.process_face_mouth(empty, primary)
decision = dd.update(eye_res, mouth_res)
alarm.trigger(decision["state"])
hud = main.draw_master_hud(empty, faces, primary, eye_res, mouth_res, decision, alarm.get_status(), 30.0)
assert hud is not None
print("PASS: No face handled cleanly.")

# 2. Test Multiple Faces
print("Testing Multiple Faces...")
sample = cv2.imread("dataset/DDD Dataset/Drowsy/A0001.png")
multi = np.hstack([sample, sample])
faces_m = fd.detect_faces(multi)
primary_m = fd.get_primary_face(faces_m, multi.shape)
eye_res_m = ed.process_face_eyes(multi, primary_m)
mouth_res_m = yd.process_face_mouth(multi, primary_m)
decision_m = dd.update(eye_res_m, mouth_res_m)
hud_m = main.draw_master_hud(multi, faces_m, primary_m, eye_res_m, mouth_res_m, decision_m, alarm.get_status(), 30.0)
assert hud_m is not None
print(f"PASS: Multiple faces ({len(faces_m)}) handled cleanly.")

# 3. Test Face Disappearing (Transition)
print("Testing Face Disappearing...")
decision_none = dd.update(eye_res, mouth_res)
assert decision_none["state"] in ["ALERT", "WARNING", "DROWSY"]
print("PASS: Face disappearing handled cleanly.")

print("All main.py edge case tests passed successfully!")
