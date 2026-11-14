from pathlib import Path
import os

FAL_KEY        = os.getenv("FAL_KEY", "")
DPI            = 300
PASSPORT_W_PX  = 413
PASSPORT_H_PX  = 531
TILT_THRESHOLD_DEG = 8.0
BG_BLUE        = (24, 104, 183)
BG_WHITE       = (255, 255, 255)
BASE_DIR       = Path(__file__).parent
UPLOAD_DIR     = BASE_DIR / "uploads"
OUTPUT_DIR     = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# settings alias so orchestrator imports work
class _Settings:
    FAL_KEY            = FAL_KEY
    DPI                = DPI
    PASSPORT_W_PX      = PASSPORT_W_PX
    PASSPORT_H_PX      = PASSPORT_H_PX
    TILT_THRESHOLD_DEG = TILT_THRESHOLD_DEG
    BG_BLUE            = BG_BLUE
    BG_WHITE           = BG_WHITE
    UPLOAD_DIR         = UPLOAD_DIR
    OUTPUT_DIR         = OUTPUT_DIR

settings = _Settings()
