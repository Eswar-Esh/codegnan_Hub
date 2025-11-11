"""
gesture_dino_ui.py
Chrome-Dino style simple runner with hand-gesture (open palm => jump).
Requires:
  - assets/background.png
  - assets/dinos.png

Install:
  pip install pygame opencv-python mediapipe numpy
Run:
  python gesture_dino_ui.py
"""

import pygame
import cv2
import mediapipe as mp
import numpy as np
import random
import sys
import time

# -------- CONFIG --------
ASSET_BG = "assets/back.png"
ASSET_DINO = "assets/dino.png"

SCREEN_W, SCREEN_H = 900, 400
DINO_WIDTH, DINO_HEIGHT = 80, 80
DINO_X = 100
AUTO_DETECT_GROUND = True
GROUND_Y_OVERRIDE = None

FPS = 30
INITIAL_SPEED = 8          # 🚀 increased starting speed
SPEED_INCREASE_EVERY = 4   # more frequent speed-ups
SCORE_INCREASE_RATE = 1.5  # how fast the score goes up
# ------------------------

# -------- Gesture helper (MediaPipe) --------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(min_detection_confidence=0.65, min_tracking_confidence=0.65)

def detect_open_hand_and_draw(frame):
    """Return (open_hand_bool, annotated_frame). Open means 4 or more fingers up."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)
    open_hand = False

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            lm = handLms.landmark
            tip_ids = [4, 8, 12, 16, 20]
            fingers = 0
            for i in range(1, 5):
                if lm[tip_ids[i]].y < lm[tip_ids[i] - 2].y:
                    fingers += 1
            if fingers >= 4:
                open_hand = True
    return open_hand, frame

# -------- Utility: detect ground y from background image --------
def detect_ground_y_from_surface(surf, samples=30, thresh=15):
    arr = pygame.surfarray.array3d(surf)
    arr = np.transpose(arr, (1, 0, 2))
    h, w, _ = arr.shape
    cx0 = max(0, w // 4)
    cx1 = min(w, 3 * w // 4)
    column = arr[:, cx0:cx1, :].mean(axis=1)
    diffs = np.sqrt(((np.diff(column, axis=0)) ** 2).sum(axis=1))
    window = 5
    if len(diffs) > window:
        diffs_smooth = np.convolve(diffs, np.ones(window) / window, mode='same')
    else:
        diffs_smooth = diffs
    for i in range(h - 2, int(h * 0.25), -1):
        if diffs_smooth[i - 1] > thresh:
            return i
    return int(h * 0.75)

# -------- Pygame + Game setup --------
pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Gesture Dino (simple)")

bg_img = pygame.image.load(ASSET_BG).convert()
bg_w, bg_h = bg_img.get_width(), bg_img.get_height()
scale = SCREEN_H / bg_h
bg_w_scaled = int(bg_w * scale)
bg_img = pygame.transform.smoothscale(bg_img, (bg_w_scaled, SCREEN_H))

dino_img = pygame.image.load(ASSET_DINO).convert_alpha()
dino_img = pygame.transform.smoothscale(dino_img, (DINO_WIDTH, DINO_HEIGHT))

if GROUND_Y_OVERRIDE is not None:
    GROUND_Y = GROUND_Y_OVERRIDE
else:
    if AUTO_DETECT_GROUND:
        try:
            GROUND_Y = detect_ground_y_from_surface(bg_img)
        except Exception:
            GROUND_Y = int(SCREEN_H * 0.75)
    else:
        GROUND_Y = int(SCREEN_H * 0.75)

DINO_BOTTOM_TO_GROUND_OFFSET = 0
dino_y = GROUND_Y - DINO_HEIGHT - DINO_BOTTOM_TO_GROUND_OFFSET

bg_x = 0
bg_speed = INITIAL_SPEED

obstacles = []
OBST_MIN_W, OBST_MAX_W = 20, 40
OBST_MIN_H, OBST_MAX_H = 40,70
OBST_GAP_MIN, OBST_GAP_MAX = 250, 500

def spawn_obstacle(x_pos=None):
    x = x_pos if x_pos is not None else SCREEN_W + random.randint(50, 300)
    w = random.randint(OBST_MIN_W, OBST_MAX_W)
    h = random.randint(OBST_MIN_H, OBST_MAX_H)
    y = GROUND_Y - h
    obstacles.append({'x': x, 'w': w, 'h': h, 'y': y})

spawn_obstacle(SCREEN_W + 200)

# Dino physics
is_jumping = False
jump_vel = 0
JUMP_STRENGTH = -13
GRAVITY = 0.8

score = 0
font = pygame.font.SysFont(None, 30)

cap = cv2.VideoCapture(0)
prev_open = False
clock = pygame.time.Clock()
running = True
last_speed_inc_score = 0

# --- Main loop ---
while running:
    # --- webcam ---
    ret, frame = cap.read()
    if not ret:
        open_hand = False
    else:
        frame = cv2.flip(frame, 1)
        open_hand, annotated = detect_open_hand_and_draw(frame)
        cv2.putText(annotated, f"Open:{open_hand}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Hand (press ESC to close)", annotated)

    # --- events ---
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                running = False

    # --- game logic ---
    bg_x -= bg_speed
    if bg_x <= -bg_w_scaled:
        bg_x = 0

    for ob in obstacles:
        ob['x'] -= bg_speed

    obstacles = [ob for ob in obstacles if ob['x'] + ob['w'] > -50]

    if not obstacles or (obstacles[-1]['x'] < SCREEN_W - random.randint(OBST_GAP_MIN, OBST_GAP_MAX)):
        spawn_obstacle(SCREEN_W + random.randint(100, 200))

    # ✅ SCORE UPDATE FIX
    score += int(SCORE_INCREASE_RATE * bg_speed * 0.1)

    # ✅ SPEED INCREASE FIX
    if score - last_speed_inc_score >= SPEED_INCREASE_EVERY * 100:
        bg_speed += 0.8
        last_speed_inc_score = score

    # Jump logic
    if open_hand and not prev_open and not is_jumping:
        is_jumping = True
        jump_vel = JUMP_STRENGTH
    prev_open = open_hand

    if is_jumping:
        dino_y += jump_vel
        jump_vel += GRAVITY
        if dino_y >= GROUND_Y - DINO_HEIGHT - DINO_BOTTOM_TO_GROUND_OFFSET:
            dino_y = GROUND_Y - DINO_HEIGHT - DINO_BOTTOM_TO_GROUND_OFFSET
            is_jumping = False
            jump_vel = 0

    # Collision
    dino_rect = pygame.Rect(DINO_X + 10, int(dino_y + 10), DINO_WIDTH - 20, DINO_HEIGHT - 10)
    hit = False
    for ob in obstacles:
        ob_rect = pygame.Rect(int(ob['x']), int(ob['y']), ob['w'], ob['h'])
        if dino_rect.colliderect(ob_rect):
            hit = True
            break

    # --- draw ---
    screen.fill((255, 255, 255))
    screen.blit(bg_img, (bg_x, 0))
    screen.blit(bg_img, (bg_x + bg_w_scaled, 0))

    for ob in obstacles:
        ox, oy, ow, oh = int(ob['x']), int(ob['y']), int(ob['w']), int(ob['h'])
        pygame.draw.rect(screen, (30, 120, 30), (ox, oy, ow, oh), border_radius=6)
        arm_h = max(8, oh // 4)
        pygame.draw.rect(screen, (30, 120, 30), (ox - ow // 2, oy + oh // 4, ow // 2, arm_h), border_radius=6)
        pygame.draw.rect(screen, (30, 120, 30), (ox + ow, oy + oh // 3, ow // 2, arm_h), border_radius=6)

    screen.blit(dino_img, (DINO_X, int(dino_y)))

    score_surf = font.render(f"Score: {score}", True, (10, 10, 10))
    screen.blit(score_surf, (10, 10))
    speed_surf = font.render(f"Speed: {bg_speed:.1f}", True, (10, 10, 10))
    screen.blit(speed_surf, (10, 34))

    if hit:
        over_font = pygame.font.SysFont(None, 56)
        over_surf = over_font.render("GAME OVER - Restarting...", True, (200, 30, 30))
        screen.blit(over_surf, (SCREEN_W // 2 - over_surf.get_width() // 2, SCREEN_H // 2 - 20))
        pygame.display.update()
        pygame.time.wait(900)
        obstacles.clear()
        spawn_obstacle(SCREEN_W + 200)
        bg_speed = INITIAL_SPEED
        score = 0
        last_speed_inc_score = 0
        dino_y = GROUND_Y - DINO_HEIGHT - DINO_BOTTOM_TO_GROUND_OFFSET
        is_jumping = False
        prev_open = False
        continue

    pygame.display.update()
    clock.tick(FPS)

# cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
