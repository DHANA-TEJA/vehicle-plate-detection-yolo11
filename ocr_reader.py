import cv2
import easyocr
import numpy as np
import pytesseract
from utils import license_complies_format, format_license

# ---------- CONFIG ----------
reader = easyocr.Reader(['en'], gpu=False)
# Optional: manually specify tesseract path on Windows if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
# ----------------------------


def preprocess_plate(plate_img):
    """Preprocess cropped plate for better OCR reading."""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 75, 75)
    gray = cv2.equalizeHist(gray)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = th.shape
    if w < 200:  # scale small plates
        scale_factor = 200 / w
        th = cv2.resize(th, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    return th


def read_easyocr(processed):
    """Try to read text using EasyOCR."""
    detections = reader.readtext(processed, detail=1, paragraph=False)
    best_text, best_score = None, 0
    for detection in detections:
        _, text, score = detection
        text = text.upper().replace(' ', '').strip()
        if len(text) < 4:
            continue
        if license_complies_format(text):
            return format_license(text), score
        if score > best_score:
            best_text, best_score = text, score
    return best_text, best_score


def read_tesseract(processed):
    """Fallback OCR using Tesseract."""
    config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(processed, config=config)
    text = text.upper().replace(' ', '').strip()
    if len(text) == 0:
        config = '--psm 11'
        text = pytesseract.image_to_string(processed, config=config).upper().replace(' ', '').strip()
    if len(text) < 4:
        return None, 0
    if license_complies_format(text):
        text = format_license(text)
    return text, 0.5  # no real confidence value, assume mid-level


def read_license_plate(plate_img):
    """
    Hybrid OCR — uses EasyOCR, falls back to Tesseract if unclear.
    Returns (text, confidence).
    """
    processed = preprocess_plate(plate_img)

    # Try EasyOCR first
    text, score = read_easyocr(processed)
    if text and len(text) >= 4:
        return text, score

    # Fallback to Tesseract
    text2, score2 = read_tesseract(processed)
    if text2:
        return text2, score2

    return None, None
