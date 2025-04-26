import cv2
import time
import json
import argparse
import numpy as np
import pyautogui
import mediapipe as mp
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────
TEMPLATE_PATH       = Path("gestures.json")
SAMPLES_PER_GESTURE = 30
MATCH_THRESHOLD     = 0.15  # for calibrated matching
COOLDOWN            = 0.5   # seconds between repeats
CONTINUOUS          = {'vol_up','vol_down','next','prev'}

# Gestures we calibrate
CALIB_GESTURES = [
    'play','pause','vol_up','vol_down','next','prev','gun_left','gun_right'
]

# Action map (both built-in & calibrated use these same actions)
ACTIONS = {
    'play':       lambda: pyautogui.press('k'),
    'pause':      lambda: pyautogui.press('k'),
    'vol_up':     lambda: pyautogui.press('up'),
    'vol_down':   lambda: pyautogui.press('down'),
    'next':       lambda: pyautogui.press('l'),
    'prev':       lambda: pyautogui.press('j'),
    'gun_left':   lambda: pyautogui.hotkey('alt','left'),
    'gun_right':  lambda: pyautogui.hotkey('alt','right'),
}

# ─── MEDIAPIPE SETUP ───────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw  = mp.solutions.drawing_utils

conn_spec      = mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
landmark_spec  = mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=6)

last_time    = 0
last_gesture = None

# ─── UTILITIES ─────────────────────────────────────────────────────────
def landmarks_to_vector(lm_list):
    """Flatten landmarks into a 42-vector normalized to wrist (landmark 0)."""
    bx, by = lm_list[0].x, lm_list[0].y
    vec = []
    for lm in lm_list:
        vec.extend([lm.x - bx, lm.y - by])
    return np.array(vec, dtype=np.float32)

# ─── CALIBRATION MODE ───────────────────────────────────────────────────
def calibrate():
    templates = {}
    cap = cv2.VideoCapture(0)
    for gesture in CALIB_GESTURES:
        samples = []
        print(f"\n=== Calibrating '{gesture}' ({len(samples)}/{SAMPLES_PER_GESTURE})")
        while len(samples) < SAMPLES_PER_GESTURE:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame,1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = hands.process(rgb)
            if res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame,
                    res.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                    conn_spec, landmark_spec
                )
            cv2.putText(frame,
                f"Hold '{gesture}' and press 'c': {len(samples)}/{SAMPLES_PER_GESTURE}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2
            )
            cv2.imshow("Calibrate", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('c') and res.multi_hand_landmarks:
                vec = landmarks_to_vector(res.multi_hand_landmarks[0].landmark)
                samples.append(vec)
            elif k in (ord('q'),27):
                print("Calibration aborted.")
                cap.release()
                cv2.destroyAllWindows()
                return
        templates[gesture] = np.mean(samples, axis=0).tolist()
        print(f"→ Saved template for '{gesture}'")
    cap.release()
    cv2.destroyAllWindows()
    TEMPLATE_PATH.write_text(json.dumps(templates))
    print(f"\nAll gestures saved to {TEMPLATE_PATH}")

# ─── RUN WITH CALIBRATED TEMPLATES ──────────────────────────────────────
def run_calibrated():
    global last_time, last_gesture
    data = json.loads(TEMPLATE_PATH.read_text())
    templates = {g: np.array(v, dtype=np.float32) for g,v in data.items()}

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame,1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = hands.process(rgb)
        now   = time.time()
        gesture = None

        if res.multi_hand_landmarks:
            lm0 = res.multi_hand_landmarks[0].landmark
            vec = landmarks_to_vector(lm0)
            # find nearest template
            dists = {g: np.linalg.norm(vec - t) for g,t in templates.items()}
            g, dist = min(dists.items(), key=lambda x: x[1])
            if dist < MATCH_THRESHOLD:
                gesture = g
            # draw
            mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0],
                                   mp_hands.HAND_CONNECTIONS,
                                   conn_spec, landmark_spec)

        # dispatch
        if gesture in CONTINUOUS:
            if gesture != last_gesture or now - last_time > COOLDOWN:
                ACTIONS[gesture]()
                last_gesture = gesture
                last_time = now
        else:
            if gesture and gesture != last_gesture:
                ACTIONS[gesture]()
                last_gesture = gesture
                last_time = now

        if not res.multi_hand_landmarks:
            last_gesture = None

        cv2.putText(frame, f"Gesture: {gesture or '—'}",
                    (10,40), cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),3)
        cv2.imshow("Calibrated Mode", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'),27):
            break

    cap.release()
    cv2.destroyAllWindows()

# ─── RUN WITH BUILT-IN DETECTION ────────────────────────────────────────
def run_builtin():
    global last_time, last_gesture

    def detect_two_hand_gun(lms, handedness_label):
        idx_up    = lms[8].y  < lms[6].y
        thumb_up  = lms[4].y  < lms[3].y
        mid_down  = lms[12].y > lms[10].y
        ring_down = lms[16].y > lms[14].y
        pinky_down= lms[20].y > lms[18].y
        if thumb_up and idx_up and mid_down and ring_down and pinky_down:
            return 'gun_left' if handedness_label=='Right' else 'gun_right'
        return None

    def detect_single(lms):
        idx   = lms[8].y  < lms[6].y
        mid   = lms[12].y < lms[10].y
        ring  = lms[16].y < lms[14].y
        pinky = lms[20].y < lms[18].y
        if not (idx or mid or ring or pinky):            return 'pause'
        if idx and mid and ring and pinky:               return 'play'
        if idx and mid and not(ring or pinky):           return 'vol_up'
        if ring and pinky and not(idx or mid):           return 'vol_down'
        if pinky and not(idx or mid or ring):            return 'next'
        if idx and mid and ring and not pinky:           return 'prev'
        return None

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame,1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = hands.process(rgb)
        now   = time.time()
        gesture = None

        # draw
        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS,
                                       conn_spec, landmark_spec)

        # two-hand gun
        if res.multi_hand_landmarks and res.multi_handedness:
            for lm, hs in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = hs.classification[0].label
                g = detect_two_hand_gun(lm.landmark, label)
                if g:
                    gesture = g
                    break

        # single-hand fallback
        if gesture is None and res.multi_hand_landmarks:
            gesture = detect_single(res.multi_hand_landmarks[0].landmark)

        # dispatch
        if gesture in CONTINUOUS:
            if gesture!=last_gesture or now-last_time>COOLDOWN:
                ACTIONS[gesture]()
                last_gesture=gesture; last_time=now
        else:
            if gesture and gesture!=last_gesture:
                ACTIONS[gesture]()
                last_gesture=gesture; last_time=now

        if not res.multi_hand_landmarks:
            last_gesture=None

        cv2.putText(frame, f"Gesture: {last_gesture or '—'}",
                    (10,40), cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),3)
        cv2.imshow("Built-In Mode", frame)
        if cv2.waitKey(1)&0xFF in (ord('q'),27):
            break

    cap.release()
    cv2.destroyAllWindows()

# ─── ENTRY POINT ───────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--calibrate',       action='store_true',
                   help="Record your own gestures")
    p.add_argument('--use_calibrated', action='store_true',
                   help="Use calibrated gestures instead of built-in")
    args = p.parse_args()

    if args.calibrate:
        calibrate()
    elif args.use_calibrated:
        if not TEMPLATE_PATH.exists():
            print("No calibration data found. Run with --calibrate first.")
        else:
            run_calibrated()
    else:
        run_builtin()
