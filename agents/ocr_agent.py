"""
ocr_agent.py — OCR pipeline for handwritten essays.

Supports:
  • Tesseract OCR (English + Arabic)
  • Microsoft TrOCR via HuggingFace transformers (optional)
"""
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Tesseract wrapper
# ──────────────────────────────────────────────

def _tesseract_ocr(image_path: str, lang: str = "english") -> str:
    """Run Tesseract on an image file and return extracted text."""
    try:
        import pytesseract
        from PIL import Image, ImageEnhance, ImageFilter
        import cv2
        import numpy as np

        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

        # Read with OpenCV for preprocessing
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        # --- preprocessing ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Adaptive thresholding for handwriting
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        # Slight denoise
        denoised = cv2.fastNlMeansDenoising(thresh, h=10)

        pil_img = Image.fromarray(denoised)

        tess_lang = "ara" if lang.lower() == "arabic" else "eng"
        text = pytesseract.image_to_string(pil_img, lang=tess_lang)
        return text.strip()

    except ImportError as e:
        log.error(f"pytesseract/OpenCV not installed: {e}")
        return ""
    except Exception as e:
        log.error(f"Tesseract OCR failed on {image_path}: {e}")
        return ""


# ──────────────────────────────────────────────
# TrOCR wrapper (optional, GPU-preferred)
# ──────────────────────────────────────────────

_trocr_processor = None
_trocr_model = None


def _load_trocr():
    global _trocr_processor, _trocr_model
    if _trocr_processor is not None:
        return True
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from PIL import Image
        import torch
        model_name = "microsoft/trocr-base-handwritten"
        log.info(f"Loading TrOCR model: {model_name}")
        _trocr_processor = TrOCRProcessor.from_pretrained(model_name)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
        _trocr_model.eval()
        log.info("TrOCR model loaded.")
        return True
    except Exception as e:
        log.warning(f"Could not load TrOCR: {e}. Falling back to Tesseract.")
        return False


def _trocr_ocr(image_path: str) -> str:
    """Run Microsoft TrOCR on an image."""
    if not _load_trocr():
        return ""
    try:
        from PIL import Image
        import torch
        img = Image.open(image_path).convert("RGB")
        pixel_values = _trocr_processor(images=img, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = _trocr_model.generate(pixel_values)
        text = _trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()
    except Exception as e:
        log.error(f"TrOCR inference failed: {e}")
        return ""


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def transcribe(image_path: str, lang: str = "english") -> str:
    """
    Convert a handwritten essay image to text.

    Args:
        image_path: Absolute path to image file (PNG/JPG/TIFF).
        lang: 'english' or 'arabic'.

    Returns:
        Extracted text string.
    """
    if config.USE_TROCR and lang.lower() == "english":
        text = _trocr_ocr(image_path)
        if text:
            return text
    # Default: Tesseract
    return _tesseract_ocr(image_path, lang=lang)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        lang = sys.argv[2] if len(sys.argv) > 2 else "english"
        result = transcribe(path, lang)
        print("Transcription:", result)
    else:
        print("Usage: python ocr_agent.py <image_path> [english|arabic]")
