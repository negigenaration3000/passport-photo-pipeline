"""Step 05: produce local BiSeNet semantic masks for the portrait."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


_WEIGHTS_FILE = "79999_iter.pth"
_WEIGHTS_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=154JgKpzCPW82qINcVieuPH3fZ2e0P812&export=download&confirm=t"
)
_WEIGHTS_PATH = Path.home() / ".cache" / "passport_pipeline" / "bisenet" / _WEIGHTS_FILE
_INPUT_SIZE = (512, 512)
_MODEL = None
_DEVICE = None

_REGION_LABELS = {
    "background": (0,),
    "skin": (1,),
    "left_eye": (4,),
    "right_eye": (5,),
    "nose": (10,),
    "mouth": (11, 12, 13),
    "hair": (17,),
    "neck": (14,),
    "clothing": (16,),
}


def _ensure_weights() -> Path:
    """Return the cached checkpoint, downloading the upstream file once if needed."""
    if _WEIGHTS_PATH.is_file():
        return _WEIGHTS_PATH

    temporary_path = _WEIGHTS_PATH.with_suffix(".part")
    try:
        _WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_WEIGHTS_URL, temporary_path)
        temporary_path.replace(_WEIGHTS_PATH)
    except Exception as error:
        if temporary_path.exists():
            temporary_path.unlink()
        raise RuntimeError(f"BiSeNet weights download failed: {error}") from error
    return _WEIGHTS_PATH


def _get_model():
    """Load the 19-class network once into the best available PyTorch device."""
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL, _DEVICE

    weights_path = _ensure_weights()
    try:
        import torch
        from model.bisenet import BiSeNet

        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BiSeNet(n_classes=19)
        model.load_state_dict(torch.load(weights_path, map_location=_DEVICE))
        _MODEL = model.to(_DEVICE).eval()
        return _MODEL, _DEVICE
    except Exception as error:
        raise RuntimeError(f"BiSeNet model loading failed: {error}") from error


def _infer_labels(image: Image.Image) -> np.ndarray:
    """Return 512px class labels, resized back to the source dimensions."""
    try:
        import torch

        model, device = _get_model()
        resized = image.resize(_INPUT_SIZE, Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        normalized = (pixels - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            labels = model(tensor)[0].squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
        return np.asarray(
            Image.fromarray(labels).resize(image.size, Image.Resampling.NEAREST), dtype=np.uint8
        )
    except Exception as error:
        raise RuntimeError(f"BiSeNet inference failed: {error}") from error


def parse(img: Image.Image) -> dict[str, np.ndarray]:
    """Return named boolean semantic masks and the protected combined face mask."""
    if not isinstance(img, Image.Image):
        raise TypeError("img must be a PIL.Image.Image")

    image = img if img.mode == "RGB" else img.convert("RGB")
    labels = _infer_labels(image)
    masks = {
        name: np.isin(labels, class_ids)
        for name, class_ids in _REGION_LABELS.items()
    }
    masks["face"] = (
        masks["skin"]
        | masks["left_eye"]
        | masks["right_eye"]
        | masks["nose"]
        | masks["mouth"]
    )
    return masks
