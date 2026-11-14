"""Step 01: correct roll tilt in an RGB portrait without changing canvas size."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

from PIL import Image


_ROLL_THRESHOLD_DEGREES = 2.0
# MediaPipe Face Mesh 468-point topology: 33/133 and 362/263 are the paired
# corners of the two eye contours, so their midpoints approximate eye centres.
_LEFT_EYE_CORNERS = (33, 133)
_RIGHT_EYE_CORNERS = (362, 263)


def _landmark_face_center_and_eyes(
    image: Image.Image,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    """Return ``(face_center, left_eye_center, right_eye_center)`` in pixels.

    The Face Landmarker task bundle (``face_landmarker.task``) contains the
    landmark model needed for 468 points.  Keep its path outside source code so
    desktop packaging can ship the model next to this module or supply it via
    ``MEDIAPIPE_FACE_LANDMARKER_MODEL``.
    """
    try:
        import mediapipe as mp
        import numpy as np
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
        # Model/runtime errors deliberately fall through to the OpenCV fallback.
        return None

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]
    if len(landmarks) < 468:
        return None

    width, height = image.size

    def point(index: int) -> Tuple[float, float]:
        landmark = landmarks[index]
        return landmark.x * width, landmark.y * height

    def mean_point(indices: Iterable[int]) -> Tuple[float, float]:
        points = [point(index) for index in indices]
        return (
            sum(x for x, _ in points) / len(points),
            sum(y for _, y in points) / len(points),
        )

    # The centre of the landmark bounding box is a stable face-centre rotation
    # pivot.  The pivot changes placement, not the roll angle itself.
    face_points = [point(index) for index in range(468)]
    xs, ys = zip(*face_points)
    face_center = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
    return face_center, mean_point(_LEFT_EYE_CORNERS), mean_point(_RIGHT_EYE_CORNERS)


def _haar_face_center_and_eyes(
    image: Image.Image,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    """Fallback using OpenCV's bundled Haar face and eye cascade XML files."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    haar_dir = Path(cv2.data.haarcascades)
    face_cascade = cv2.CascadeClassifier(str(haar_dir / "haarcascade_frontalface_default.xml"))
    eye_cascade = cv2.CascadeClassifier(str(haar_dir / "haarcascade_eye.xml"))
    if face_cascade.empty() or eye_cascade.empty():
        return None

    rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None

    # The largest detected face is the intended passport-photo subject.
    x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
    roi = gray[y : y + height, x : x + width]
    eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=5)
    if len(eyes) < 2:
        return None

    # Restrict to the upper face to reduce false positives from nostrils/mouth,
    # then select the two eyes furthest apart horizontally.
    candidates = [
        (x + ex + ew / 2.0, y + ey + eh / 2.0)
        for ex, ey, ew, eh in eyes
        if ey + eh / 2.0 < height * 0.65
    ]
    if len(candidates) < 2:
        return None
    left_eye, right_eye = min(candidates, key=lambda p: p[0]), max(candidates, key=lambda p: p[0])
    if left_eye == right_eye:
        return None

    return (x + width / 2.0, y + height / 2.0), left_eye, right_eye


def _face_center_and_eyes(
    image: Image.Image,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    """Prefer 468 MediaPipe landmarks; use Haar only when that path fails."""
    return _landmark_face_center_and_eyes(image) or _haar_face_center_and_eyes(image)


def align(img: Image.Image) -> Image.Image:
    """Return an RGB image whose detected eye line has no roll tilt.

    If neither detector can locate two eyes, or the roll is smaller than two
    degrees, the original RGB image is returned without resampling.
    """
    if not isinstance(img, Image.Image):
        raise TypeError("img must be a PIL.Image.Image")

    # The public contract is RGB.  Conversion is intentionally the only change
    # made when input arrives in another PIL mode.
    image = img if img.mode == "RGB" else img.convert("RGB")
    detected = _face_center_and_eyes(image)
    if detected is None:
        return image

    face_center, left_eye, right_eye = detected
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    if dx == 0:
        return image

    # atan2 gives the signed eye-line angle in image coordinates (y grows down).
    # Pillow's positive rotation is counter-clockwise, which brings a
    # downward-sloping left-to-right eye line back to horizontal.
    roll_degrees = math.degrees(math.atan2(dy, dx))
    if abs(roll_degrees) < _ROLL_THRESHOLD_DEGREES:
        return image

    # ``expand=False`` retains the exact original canvas.  Pixels exposed at
    # corners are filled with white rather than Pillow's default black.
    return image.rotate(
        roll_degrees,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        center=face_center,
        fillcolor=(255, 255, 255),
    )
