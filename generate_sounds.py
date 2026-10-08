"""
Generates the three alert sound effects used by app.py:
  sounds/alarm.wav   -> played when eyes are closed too long (drowsy / sleeping)
  sounds/faudio.wav  -> played when no face is visible (book covering face / looked away)
  sounds/paudio.wav  -> played when a phone is detected in frame

Run this once before running app.py:
    python generate_sounds.py
"""
import numpy as np
import wave
import struct
import os

SAMPLE_RATE = 44100
OUT_DIR = os.path.join(os.path.dirname(__file__), "sounds")
os.makedirs(OUT_DIR, exist_ok=True)


def tone(freq, duration, volume=0.5, fade=0.01):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave_data = np.sin(freq * t * 2 * np.pi)
    fade_len = int(SAMPLE_RATE * fade)
    if fade_len > 0 and fade_len * 2 < len(wave_data):
        env = np.ones_like(wave_data)
        env[:fade_len] = np.linspace(0, 1, fade_len)
        env[-fade_len:] = np.linspace(1, 0, fade_len)
        wave_data *= env
    return wave_data * volume


def save_wav(filename, samples):
    samples = np.clip(samples, -1, 1)
    samples_int16 = (samples * 32767).astype(np.int16)
    path = os.path.join(OUT_DIR, filename)
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(struct.pack("<%dh" % len(samples_int16), *samples_int16))
    print(f"Created {path}")


def build_alarm():
    # Urgent alternating two-tone siren (drowsiness alarm)
    beep_high = tone(1000, 0.18, volume=0.6)
    beep_low = tone(700, 0.18, volume=0.6)
    silence = np.zeros(int(SAMPLE_RATE * 0.05))
    pattern = np.concatenate([beep_high, silence, beep_low, silence] * 3)
    save_wav("alarm.wav", pattern)


def build_face_alert():
    # Soft double-chime (no face detected / book covering face)
    chime1 = tone(523, 0.15, volume=0.45)   # C5
    chime2 = tone(659, 0.2, volume=0.45)    # E5
    gap = np.zeros(int(SAMPLE_RATE * 0.08))
    pattern = np.concatenate([chime1, gap, chime2])
    save_wav("faudio.wav", pattern)


def build_phone_alert():
    # Short triple-beep (phone detected)
    beep = tone(880, 0.12, volume=0.55)
    gap = np.zeros(int(SAMPLE_RATE * 0.08))
    pattern = np.concatenate([beep, gap, beep, gap, beep])
    save_wav("paudio.wav", pattern)


if __name__ == "__main__":
    build_alarm()
    build_face_alert()
    build_phone_alert()
    print("\nAll sound effects generated in the 'sounds' folder.")
