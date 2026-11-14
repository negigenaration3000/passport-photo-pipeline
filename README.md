# Passport Photo Pipeline

An automated 12-step hybrid AI and computer-vision pipeline that transforms raw portrait photos into **ICAO 9303 compliant** passport photos (35×45mm at 300 DPI) and tiles them onto printable A4 sheets with cut guidelines.

## Features

- **Hybrid Processing**: Combines local OpenCV/MediaPipe/PyTorch models with cloud generative enhancement ([fal.ai](https://fal.ai)).
- **Roll Tilt Deskewing**: Automatically levels eye lines if tilt exceeds 2°.
- **Selective Denoising**: Analyzes Laplacian variance and conditionally removes print/scan noise using FastNLM.
- **Studio Alpha Matting**: Deep portrait isolation via BiRefNet with guided edge refinement for natural hair boundaries.
- **Facial Parsing & Lighting Balance**: Uses a 19-class BiSeNet network to isolate skin pixels and conditionally balance asymmetric lighting via LAB CLAHE.
- **CodeFormer Facial Restoration**: Enhances clarity and facial features.
- **Authentic Texture Preservation**: Employs frequency separation pre-CodeFormer and re-blends authentic high-frequency skin pores post-CodeFormer to prevent plastic/artificial skin.
- **ICAO 9303 Compliant Framing**: Enforces head-to-frame ratios and eye-height constraints on standard studio blue or white backgrounds.
- **A4 Print Layout**: Generates 300 DPI A4 sheets (30 tiled photos) with 1px cut marks for guillotine/scissor trimming.

---

## Pipeline Overview

```
[01] Align (Deskew eye tilt)
 └─ [02] Denoise (Laplacian variance check + FastNLM)
     ├─ [03] Matte (fal.ai BiRefNet RGBA extraction)
     │   └─ [04] Refine (OpenCV guided filter on alpha edge)
     └─ [05] Parse (PyTorch BiSeNet 19-class semantic masks)
         └─ [06] Luminance Check (Skin L/R delta)
             └─ [07] Equalise (Conditional LAB CLAHE on skin)
                 ├─ [09] Frequency Separation (Save authentic high-freq texture)
                 ├─ [08] Restore (fal.ai CodeFormer face enhancement)
                 └─ [10] Texture Re-blend (Re-merge high-freq pore detail)
                     └─ [11] ICAO Crop & Background Composite
                         └─ [12] Tiling on A4 (300 DPI with cut lines)
```

---

## Installation

### 1. Prerequisites
- Python 3.10+
- [Git](https://git-scm.com/)

### 2. Clone Repository
```bash
git clone https://github.com/<your-username>/passport-photo-pipeline.git
cd passport-photo-pipeline
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your [fal.ai](https://fal.ai) API key to `.env`:
```env
FAL_KEY=your_fal_key_here
```

---

## Usage

### Run the Pipeline
```bash
python orchestrator.py <input_image_path> <output_image_path> <background_color>
```

**Parameters:**
- `input_image_path` (default: `uploads/test.jpg`): Path to input portrait photo.
- `output_image_path` (default: `outputs/passport_a4.jpg`): Path to save the resulting A4 sheet or photo.
- `background_color` (default: `blue`): `blue` (`RGB(24, 104, 183)`) or `white` (`RGB(255, 255, 255)`).

**Example:**
```bash
python orchestrator.py uploads/portrait.jpg outputs/passport_sheet.jpg blue
```

---

## Running Tests

Run the full automated test suite:
```bash
python -m unittest discover -s . -p "test_*.py"
```

---

## License

MIT License.
