"""
Face detection utility — returns a FaceAnalysis object with bbox geometry.
Used by the orchestrator to pass crop geometry to step11_crop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class FaceAnalysis:
    x: int          # bbox top-left x
    y: int          # bbox top-left y
    w: int          # bbox width
    h: int          # bbox height
    tilt_deg: float = 0.0


def analyse(image_array: np.ndarray) -> Optional[FaceAnalysis]:
    """
    Detect the primary face in an RGB numpy array.
    Returns FaceAnalysis with bounding box and tilt, or None if no face found.
    """
    img = Image.fromarray(image_array) if not isinstance(image_array, Image.Image) else image_array

    # ── Try MediaPipe first ───────────────────────────────────────
    result = _mediapipe_analyse(img)
    if result is not None:
        return result

    # ── Fall back to Haar cascade ─────────────────────────────────
    return _haar_analyse(img)


def _mediapipe_analyse(image: Image.Image) -> Optional[FaceAnalysis]:
    import os
    from pathlib import Path
    try:
        import mediapipe as mp
    except ImportError:
        return None

    model_path = os.environ.get(
        "MEDIAPIPE_FACE_LANDMARKER_MODEL",
        str(Path(__file__).with_name("face_landmarker.task")),
    )
    if not Path(model_path).is_file():
        return None

    try:
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
        )
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
            result = detector.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image))
            )
    except Exception:
        return None

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]
    if len(landmarks) < 468:
        return None

    width, height = image.size
    xs = [lm.x * width  for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]

    x = int(min(xs));  y = int(min(ys))
    w = int(max(xs)) - x
    h = int(max(ys)) - y

    # Eye corners for tilt
    LEFT_CORNERS  = (33, 133)
    RIGHT_CORNERS = (362, 263)
    lx = sum(landmarks[i].x * width  for i in LEFT_CORNERS)  / 2
    ly = sum(landmarks[i].y * height for i in LEFT_CORNERS)  / 2
    rx = sum(landmarks[i].x * width  for i in RIGHT_CORNERS) / 2
    ry = sum(landmarks[i].y * height for i in RIGHT_CORNERS) / 2
    tilt = math.degrees(math.atan2(ry - ly, rx - lx))

    return FaceAnalysis(x=x, y=y, w=w, h=h, tilt_deg=tilt)


def _haar_analyse(image: Image.Image) -> Optional[FaceAnalysis]:
    try:
        import cv2
        from pathlib import Path
    except ImportError:
        return None

    haar_dir = Path(cv2.data.haarcascades)
    face_cascade = cv2.CascadeClassifier(str(haar_dir / "haarcascade_frontalface_default.xml"))
    if face_cascade.empty():
        return None

    rgb  = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

    # Try to get tilt from eye cascade
    tilt = 0.0
    eye_cascade = cv2.CascadeClassifier(str(haar_dir / "haarcascade_eye.xml"))
    if not eye_cascade.empty():
        roi  = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=5)
        candidates = [
            (x + ex + ew/2, y + ey + eh/2)
            for ex, ey, ew, eh in eyes
            if ey + eh/2 < h * 0.65
        ]
        if len(candidates) >= 2:
            le = min(candidates, key=lambda p: p[0])
            re = max(candidates, key=lambda p: p[0])
            if le != re:
                tilt = math.degrees(math.atan2(re[1]-le[1], re[0]-le[0]))

    return FaceAnalysis(x=int(x), y=int(y), w=int(w), h=int(h), tilt_deg=tilt)