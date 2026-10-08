"""
Smart Study Monitor
--------------------
Watches you through the webcam while you study and warns you when it thinks
you are:
  1. Falling asleep / eyes closed too long   -> plays alarm.wav
  2. Not visible to the camera (book covering face, looked away, left desk)
                                              -> plays faudio.wav
  3. Using your phone (detected in frame)     -> plays paudio.wav

Controls:
  q  -> quit

Setup:
    pip install -r requirements.txt
    python generate_sounds.py      # only needed once, creates the .wav alerts
    python app.py
"""
import time
import math
import os

import cv2
import numpy as np
import mediapipe as mp
import pygame

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EAR_THRESHOLD = 0.22          # eyes counted as "closed" below this ratio
EAR_CONSEC_FRAMES = 20        # frames closed before triggering sleep alarm
NO_FACE_SECONDS = 3.0         # seconds with no face before triggering alert
PHONE_CONF_THRESHOLD = 0.45
PHONE_CHECK_EVERY_N_FRAMES = 5   # run phone detector every N frames (perf)
ALERT_COOLDOWN = 4.0          # seconds between repeat plays of the same alert

SOUND_DIR = os.path.join(os.path.dirname(__file__), "sounds")

# Mediapipe FaceMesh eye landmark indices (6-point EAR)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# ---------------------------------------------------------------------------
# Sound manager (handles cooldowns so alarms don't spam)
# ---------------------------------------------------------------------------
class SoundManager:
    def __init__(self):
        pygame.mixer.init()
        self.sounds = {}
        self.last_played = {}
        for name, filename in [
            ("sleep", "alarm.wav"),
            ("noface", "faudio.wav"),
            ("phone", "paudio.wav"),
        ]:
            path = os.path.join(SOUND_DIR, filename)
            if os.path.exists(path):
                self.sounds[name] = pygame.mixer.Sound(path)
            else:
                print(f"[WARN] Missing sound file: {path} "
                      f"(run generate_sounds.py first)")
                self.sounds[name] = None

    def play(self, key):
        snd = self.sounds.get(key)
        if snd is None:
            return
        now = time.time()
        if now - self.last_played.get(key, 0) >= ALERT_COOLDOWN:
            snd.play()
            self.last_played[key] = now


# ---------------------------------------------------------------------------
# Eye Aspect Ratio
# ---------------------------------------------------------------------------
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(landmarks, eye_points, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_points]
    p1, p2, p3, p4, p5, p6 = pts
    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)


# ---------------------------------------------------------------------------
# Optional phone detector (YOLOv8). App still works fully without it.
# ---------------------------------------------------------------------------
class PhoneDetector:
    """Wraps ultralytics YOLO if installed; otherwise phone detection is
    silently skipped so the rest of the monitor still runs."""

    COCO_PHONE_CLASS_ID = 67  # "cell phone" in the COCO dataset

    def __init__(self):
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")  # downloads once, then cached
            print("[INFO] Phone detection enabled (YOLOv8n).")
        except Exception as e:
            print(f"[INFO] Phone detection disabled ({e}). "
                  f"Install with: pip install ultralytics")

    def detect_phone(self, frame):
        if self.model is None:
            return False
        results = self.model.predict(frame, verbose=False, conf=PHONE_CONF_THRESHOLD)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == self.COCO_PHONE_CLASS_ID:
                    return True
        return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    sounds = SoundManager()
    phone_detector = PhoneDetector()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    closed_frames = 0
    no_face_since = None
    frame_count = 0
    phone_visible = False
    status_text = "Watching..."
    status_color = (0, 200, 0)

    print("Smart Study Monitor running. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        frame_count += 1

        if results.multi_face_landmarks:
            no_face_since = None
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0
            ear_display = int(ear * 100)

            cv2.rectangle(frame, (10, 10), (230, 55), (200, 0, 200), -1)
            cv2.putText(frame, f"Eye Ratio: {ear_display}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if ear < EAR_THRESHOLD:
                closed_frames += 1
            else:
                closed_frames = 0

            if closed_frames >= EAR_CONSEC_FRAMES:
                status_text = "WAKE UP! You look drowsy"
                status_color = (0, 0, 255)
                sounds.play("sleep")
            else:
                status_text = "Focused"
                status_color = (0, 200, 0)
        else:
            if no_face_since is None:
                no_face_since = time.time()
            elapsed = time.time() - no_face_since
            if elapsed >= NO_FACE_SECONDS:
                status_text = "Face not visible - come back to your book!"
                status_color = (0, 140, 255)
                sounds.play("noface")

        # Phone detection (throttled for performance)
        if frame_count % PHONE_CHECK_EVERY_N_FRAMES == 0:
            phone_visible = phone_detector.detect_phone(frame)

        if phone_visible:
            status_text = "Phone detected - put it away!"
            status_color = (0, 0, 255)
            sounds.play("phone")

        cv2.putText(frame, status_text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        cv2.imshow("Smart Study Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
