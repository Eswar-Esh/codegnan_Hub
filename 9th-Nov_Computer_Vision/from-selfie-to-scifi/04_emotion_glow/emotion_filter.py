"""
Change glow color based on mouth opening (smile or surprise).
Students can tweak logic and build expressions.
"""
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

def is_mouth_open(landmarks, img_shape):
    # Get upper and lower lips
    upper_lip = landmarks[13]  # Upper lip center
    lower_lip = landmarks[14]  # Lower lip center

    h, w, _ = img_shape
    dist = abs((lower_lip.y - upper_lip.y) * h)
    return dist > 18  # Threshold for mouth open

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
    while True:
        ret, frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                if is_mouth_open(landmarks, frame.shape):
                    glow_color = (0, 0, 255)  # Red glow
                else:
                    glow_color = (255, 255, 0)  # Yellow glow

                # Draw glow around face
                overlay = frame.copy()
                cv2.circle(
                    overlay,
                    (int(landmarks[1].x * frame.shape[1]), int(landmarks[1].y * frame.shape[0])),
                    60,
                    glow_color,
                    -1,
                )
                alpha = 0.4  # Transparency
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.imshow("Emotion Glow Filter - Press 'q' to Quit", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
