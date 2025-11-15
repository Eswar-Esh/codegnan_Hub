import cv2
import mediapipe as mp
import pyautogui

pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def is_pinch(lm):
    thumb_tip = lm[4]
    index_tip = lm[8]
    dist = abs(thumb_tip.x - index_tip.x) + abs(thumb_tip.y - index_tip.y)
    return dist < 0.05  # pinch threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            lm = handLms.landmark

            # Cursor Position → Use index finger tip (landmark 8)
            index_x = int(lm[8].x * screen_w)
            index_y = int(lm[8].y * screen_h)

            pyautogui.moveTo(index_x, index_y)

            # Click if pinch detected
            if is_pinch(lm):
                pyautogui.click()

    cv2.imshow("Cursor Control with Hand", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
