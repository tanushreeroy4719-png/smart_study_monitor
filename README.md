# Smart Study Monitor

Webcam study companion that alerts you when you:
1. **Fall asleep / eyes closed too long** — plays `alarm.wav`
2. **Aren't visible to the camera** (book covering your face, looked away, left the desk) — plays `faudio.wav`
3. **Pick up your phone** (detected in frame) — plays `paudio.wav`

Shows a live "Eye Ratio" number on screen (like the app in your screenshot) and a status line at the bottom.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the alert sounds (only needed once — no internet download required,
#    they're synthesized locally, so no missing-file / broken-link issues)
python generate_sounds.py

# 3. Run it
python app.py
```

Press **q** in the video window to quit.

## How each detector works

| Feature | How |
|---|---|
| Drowsiness | MediaPipe FaceMesh gives 468 face landmarks → Eye Aspect Ratio (EAR) computed from 6 points per eye → below threshold for ~20 frames = drowsy |
| No-face / book covering | If FaceMesh detects no face for 3+ seconds straight, triggers a gentle reminder |
| Phone detection | YOLOv8n (via `ultralytics`) looks for the COCO "cell phone" class in each frame. **Optional** — if `ultralytics` isn't installed, this feature is silently skipped and everything else still works |

## Tuning

All the key numbers are constants at the top of `app.py`:

```python
EAR_THRESHOLD = 0.22          # lower = stricter (harder to trigger "eyes closed")
EAR_CONSEC_FRAMES = 20        # how many consecutive drowsy frames before alarm
NO_FACE_SECONDS = 3.0         # how long you can be out of frame before alert
PHONE_CONF_THRESHOLD = 0.45   # YOLO confidence needed to count as "phone"
ALERT_COOLDOWN = 4.0          # seconds between repeat alarm sounds
```

If the alarm triggers too easily during normal blinking, raise `EAR_CONSEC_FRAMES`.
If it takes too long to notice drowsiness, lower it.

## Notes

- First run of the phone detector downloads the small `yolov8n.pt` model (~6MB) automatically.
- If you don't want phone detection at all, just skip `pip install ultralytics` — the app will print a notice and continue running without it.
- Works with any standard webcam (`cv2.VideoCapture(0)`). If you have multiple cameras, change the `0` in `app.py`'s `main()`.
