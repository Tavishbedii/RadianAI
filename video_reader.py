"""Radian AI - read the claims inside a video.

Extracts two kinds of text from a video:
  1. Speech  -> English text with faster-whisper (Hindi and other languages are
                translated to English automatically)
  2. On-screen text -> OCR on frames taken every few seconds (headlines, captions)

Install once:  python -m pip install faster-whisper opencv-python-headless pytesseract
The first run downloads the Whisper speech model (about 150 MB).
Radian AI checks the CLAIMS made in a video. It does not detect deepfakes.
"""
import difflib
import os
import re
import tempfile

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
WHISPER_SIZE = os.getenv("WHISPER_MODEL", "base")   # tiny / base / small (bigger = slower, more accurate)
MAX_SECONDS = 300          # videos longer than 5 minutes are rejected
FRAME_EVERY = 3            # seconds between frames read with OCR
MAX_FRAMES = 40

_whisper = None


def _whisper_model():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    return _whisper


def video_duration(path):
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        return frames / fps if fps else None
    except Exception:
        return None


def transcribe(path):
    """Speech in the video as English text. Returns (text, detected_language)."""
    segments, info = _whisper_model().transcribe(path, task="translate", vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments)
    return re.sub(r"\s+", " ", text).strip(), info.language


def _is_new(line, seen):
    return all(difflib.SequenceMatcher(None, line.lower(), s.lower()).ratio() < 0.8 for s in seen)


def screen_text(path, use_ocr=True):
    """On-screen text from frames sampled every FRAME_EVERY seconds, without repeats."""
    if not use_ocr:
        return ""
    import cv2
    import pytesseract
    from PIL import Image
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(fps * FRAME_EVERY))
    seen, frame_no, read = [], 0, 0
    while read < MAX_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(gray)
        if img.width < 1000:
            img = img.resize((img.width * 2, img.height * 2))
        raw = pytesseract.image_to_string(img, lang="eng+hin")
        for line in raw.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if len(line.split()) >= 3 and sum(c.isalpha() for c in line) > 8 and _is_new(line, seen):
                seen.append(line)
        frame_no += step
        read += 1
    cap.release()
    return ". ".join(seen)


def read_video(video_bytes, filename="video.mp4", use_ocr=True):
    """Return a dict with transcript, on-screen text, combined text and notes."""
    suffix = os.path.splitext(filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_bytes)
        path = tmp.name
    notes = []
    try:
        duration = video_duration(path)
        if duration and duration > MAX_SECONDS:
            raise ValueError(f"The offline verifier can read videos up to {MAX_SECONDS // 60} minutes. "
                             f"This one is {duration / 60:.1f} minutes. Trim it and try again.")
        try:
            transcript, language = transcribe(path)
            if language and language != "en":
                notes.append(f"Speech detected in '{language}' and translated to English")
        except Exception as e:
            transcript = ""
            notes.append(f"Speech could not be read ({type(e).__name__})")
        try:
            on_screen = screen_text(path, use_ocr)
        except Exception as e:
            on_screen = ""
            notes.append(f"On-screen text could not be read ({type(e).__name__})")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    parts = []
    if on_screen:
        parts.append("On-screen text: " + on_screen)
    if transcript:
        parts.append("Speech: " + transcript)
    return {"transcript": transcript, "on_screen": on_screen, "combined": "\n".join(parts),
            "duration": duration, "notes": notes}