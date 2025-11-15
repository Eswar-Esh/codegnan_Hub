import cv2
import mediapipe as mp
import pyautogui

pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

jump_line = screen_h * 0.4   # Above this → jump
cooldown = 0

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

            # Use index fingertip for cursor
            x = int(lm[8].x * screen_w)
            y = int(lm[8].y * screen_h)

            pyautogui.moveTo(x, y)

            # Jump if cursor goes above line
            if y < jump_line and cooldown == 0:
                pyautogui.press("space")
                cooldown = 5  # small delay

    if cooldown > 0:
        cooldown -= 1

    # Show jump threshold
    cv2.line(frame, (0, int(jump_line)), (800, int(jump_line)), (0, 255, 0), 2)

    cv2.imshow("Dino Cursor Control", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
