"""
Final AR App: Futuristic Sci-Fi Filter
- Face-wide overlay (visor/mask/helmet)
- Real-time face tracking
- Optional glowing frame overlay
- Press 's' to save your sci-fi selfie!
"""
'''import cv2
import mediapipe as mp
import numpy as np
from overlay_utils import add_filter

# Load assets
visor = cv2.imread("assets/visor-1.png", cv2.IMREAD_UNCHANGED)
frame_glow = cv2.imread("assets/glowing_frame.png", cv2.IMREAD_UNCHANGED)  # Optional

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

cap = cv2.VideoCapture(0)
screenshot_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        # Landmarks for left/right eyes, chin, and forehead
        left_eye = face_landmarks[33]
        right_eye = face_landmarks[263]
        chin = face_landmarks[152]
        forehead = face_landmarks[10]

        # Convert to pixel positions
        lx, ly = int(left_eye.x * w), int(left_eye.y * h)
        rx, ry = int(right_eye.x * w), int(right_eye.y * h)
        chin_x, chin_y = int(chin.x * w), int(chin.y * h)
        fore_x, fore_y = int(forehead.x * w), int(forehead.y * h)

        # Face center
        cx, cy = (lx + rx) // 2, (ly + ry) // 2

        # Face size metrics
        face_height = abs(chin_y - fore_y)
        face_width = abs(rx - lx) * 2

        # Final overlay scaling
        overlay_w = int(face_width * 1.5)
        overlay_h = int(face_height * 1.8)

        # Overlay position (centered, slightly above forehead)
        overlay_x = cx - overlay_w // 2
        overlay_y = fore_y - int(overlay_h * 0.25)

        # Add face-wide visor/mask
        frame = add_filter(frame, visor, (overlay_x, overlay_y), (overlay_w, overlay_h))

        # Optional glowing frame
        if frame_glow is not None:
            frame = add_filter(frame, frame_glow, (0, 0), (w, h))

    cv2.imshow("Sci-Fi Filter App - Press 's' to Save, 'q' to Quit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        filename = f"sci_fi_selfie_{screenshot_count}.png"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        screenshot_count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''

"""
#----------------------------------------------------------#
#For Multiple Faces

#----------------------------------------------------------#

Final AR App: Futuristic Sci-Fi Filter (Multi-Face Support)
- Face-wide overlay (visor/mask/helmet)
- Real-time face tracking for multiple faces
- Optional glowing frame overlay
- Press 's' to save your sci-fi selfie!
"""
import cv2
import mediapipe as mp
import numpy as np
from overlay_utils import add_filter

# Load assets
visor = cv2.imread("assets/visor-1.png", cv2.IMREAD_UNCHANGED)
frame_glow = cv2.imread("assets/glowing_frame.png", cv2.IMREAD_UNCHANGED)  # Optional

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=5,  # Support multiple faces
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

cap = cv2.VideoCapture(0)
screenshot_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Landmarks for left/right eyes, chin, and forehead
            landmarks = face_landmarks.landmark
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            chin = landmarks[152]
            forehead = landmarks[10]

            # Convert to pixel positions
            lx, ly = int(left_eye.x * w), int(left_eye.y * h)
            rx, ry = int(right_eye.x * w), int(right_eye.y * h)
            chin_x, chin_y = int(chin.x * w), int(chin.y * h)
            fore_x, fore_y = int(forehead.x * w), int(forehead.y * h)

            # Face center
            cx, cy = (lx + rx) // 2, (ly + ry) // 2

            # Face size metrics
            face_height = abs(chin_y - fore_y)
            face_width = abs(rx - lx) * 2

            # Final overlay scaling
            overlay_w = int(face_width * 1.5)
            overlay_h = int(face_height * 1.8)

            # Overlay position (centered, slightly above forehead)
            overlay_x = cx - overlay_w // 2
            overlay_y = fore_y - int(overlay_h * 0.25)

            # Add face-wide visor/mask
            frame = add_filter(frame, visor, (overlay_x, overlay_y), (overlay_w, overlay_h))

    # Optional glowing frame (applied once per frame)
    if frame_glow is not None:
        frame = add_filter(frame, frame_glow, (0, 0), (w, h))

    cv2.imshow("Sci-Fi Filter App - Press 's' to Save, 'q' to Quit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        filename = f"sci_fi_selfie_{screenshot_count}.png"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        screenshot_count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

