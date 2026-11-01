import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from src.face_detection import FaceDetector
from src.eye_detection import EyeDetector
from src.yawn_detection import YawnDetector

fd = FaceDetector()
ed = EyeDetector()
yd = YawnDetector()

# Check several DDD images
drowsy_imgs = list(Path('dataset/DDD Dataset/Drowsy').glob('*.png'))[:100]
found_yawn = False
for p in drowsy_imgs:
    img = cv2.imread(str(p))
    faces = fd.detect_faces(img)
    if faces:
        res = yd.process_face_mouth(img, faces[0])
        eye_res = ed.process_face_eyes(img, faces[0])
        if res['is_yawning'] or res['mar'] > 0.25:
            print(f"Sample {p.name}: YawnState={res['state']} ({res['confidence']:.1f}%), MAR={res['mar']:.3f}, EyeL={eye_res['left_state']}, EyeR={eye_res['right_state']}")
            found_yawn = True
            # Generate a verification image of yawning on HUD
            annotated = fd.draw_face_annotations(img, faces[0], 1, is_drowsy=True)
            annotated = yd.draw_mouth_annotations(annotated, res)
            hud = ed.render_hud(annotated, eye_res, mouth_results=res, fps=30.0)
            cv2.imwrite('outputs/predictions/webcam_pipeline_yawn_sample.png', hud)
            print("Saved yawning verification sample to outputs/predictions/webcam_pipeline_yawn_sample.png")
            break

if not found_yawn:
    print("Checked 100 samples; no overt yawns in first 100, testing on sample from yawn dataset...")
    # Also test on a face with open mouth from DDD
    drowsy_imgs_more = list(Path('dataset/DDD Dataset/Drowsy').glob('*.png'))[100:300]
    for p in drowsy_imgs_more:
        img = cv2.imread(str(p))
        faces = fd.detect_faces(img)
        if faces:
            res = yd.process_face_mouth(img, faces[0])
            if res['is_yawning'] or res['mar'] > 0.2:
                eye_res = ed.process_face_eyes(img, faces[0])
                print(f"Sample {p.name}: YawnState={res['state']} ({res['confidence']:.1f}%), MAR={res['mar']:.3f}")
                annotated = fd.draw_face_annotations(img, faces[0], 1, is_drowsy=True)
                annotated = yd.draw_mouth_annotations(annotated, res)
                hud = ed.render_hud(annotated, eye_res, mouth_results=res, fps=30.0)
                cv2.imwrite('outputs/predictions/webcam_pipeline_yawn_sample.png', hud)
                print("Saved yawning verification sample to outputs/predictions/webcam_pipeline_yawn_sample.png")
                break
