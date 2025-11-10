import cv2
import mediapipe as mp
import numpy as np
from overlay_utils import add_filter

# ------- Settings -------
ASSET_PATHS = ["assets/visor.png", "visor.png"]
MAX_FACES = 1
DETECTION_CONF = 0.6
TRACKING_CONF = 0.6
REFINE = True  # use iris landmarks for more stable eye positions
TARGET_CAM_W, TARGET_CAM_H = 1280, 720
EYE_LM_LEFT, EYE_LM_RIGHT = 33, 263  # outer corners
NOSE_TIP = 1
DEBUG = True  # can toggle at runtime with 'd'

def load_asset(paths):
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img, p
    return None, None

def landmark_to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def main():
    global DEBUG
    visor_img, used_path = load_asset(ASSET_PATHS)
    if visor_img is None:
        raise FileNotFoundError(f"Could not load visor image. Tried: {ASSET_PATHS}")
    if DEBUG:
        print(f"[info] Loaded visor from: {used_path} shape={visor_img.shape}")

    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_CAM_H)

    with mp_face_mesh.FaceMesh(
        max_num_faces=MAX_FACES,
        refine_landmarks=REFINE,
        min_detection_confidence=DETECTION_CONF,
        min_tracking_confidence=TRACKING_CONF
    ) as face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror for selfie view and more stable detection on macOS
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    lms = face_landmarks.landmark

                    # Get keypoints
                    lx, ly = landmark_to_px(lms[EYE_LM_LEFT], w, h)
                    rx, ry = landmark_to_px(lms[EYE_LM_RIGHT], w, h)
                    nx, ny = landmark_to_px(lms[NOSE_TIP], w, h)

                    # Eye center & distance
                    eye_cx = (lx + rx) // 2
                    eye_cy = (ly + ry) // 2
                    eye_dist = max(1, int(np.hypot(rx - lx, ry - ly)))

                    # Maintain visor aspect ratio; scale by eye distance
                    vh, vw = visor_img.shape[:2]
                    target_w = int(eye_dist * 2.2)  # adjust spread across eyes
                    scale = target_w / max(1, vw)
                    target_h = int(vh * scale)

                    # Position: center around eye center, slightly above
                    x = int(eye_cx - target_w // 2)
                    y = int(eye_cy - target_h * 0.55)  # move up a bit

                    # Debug guides
                    if DEBUG:
                        cv2.circle(frame, (lx, ly), 3, (0, 255, 0), -1)
                        cv2.circle(frame, (rx, ry), 3, (0, 255, 0), -1)
                        cv2.circle(frame, (eye_cx, eye_cy), 3, (255, 0, 0), -1)
                        cv2.line(frame, (lx, ly), (rx, ry), (0, 255, 0), 1)
                        cv2.rectangle(frame, (x, y), (x + target_w, y + target_h), (255, 0, 255), 1)

                    # Overlay
                    frame = add_filter(frame, visor_img, (x, y), (target_w, target_h), debug=DEBUG)
            else:
                if DEBUG:
                    cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(frame, "Press 'q' to quit | 'd' debug on/off", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.imshow("AR Visor Filter", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                DEBUG = not DEBUG

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

