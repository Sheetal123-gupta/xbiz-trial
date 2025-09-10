# ocr_flask_api_static.py
from flask import Flask, jsonify
import cv2
import numpy as np
import pytesseract
import easyocr
import base64
import os
import json
import time

# Initialize Flask app
app = Flask(__name__)

# Initialize EasyOCR
easyocr_reader = easyocr.Reader(['en'], gpu=False)

# ---------------------------
# Config: Image path & Output
# ---------------------------
IMAGE_PATH = "aadhar_s.png"   # put your image path here
OUTPUT_DIR = "output-day2"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ocr_output.json")

# ---------------------------
# Image Preprocessing
# ---------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)

    # Resize (max dimension 1024)
    max_dim = 1024
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=30)

    # Adaptive threshold (binarization)
    binarized = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )

    return binarized

# ---------------------------
# OCR Functions
# ---------------------------
def ocr_tesseract(image):
    start = time.time()
    text = pytesseract.image_to_string(image)
    elapsed = time.time() - start
    return text, elapsed

def ocr_easyocr(image_path):
    start = time.time()
    result = easyocr_reader.readtext(image_path)
    text = "\n".join([text[1] for text in result])
    elapsed = time.time() - start
    return text, elapsed

def ocr_paddleocr_placeholder(image):
    return "PaddleOCR integration placeholder", 0.0

# ---------------------------
# Helper: Image to Base64
# ---------------------------
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# ---------------------------
# Flask Route
# ---------------------------
@app.route('/ocr', methods=['GET'])
def ocr_api():
    # Preprocess
    processed_img = preprocess_image(IMAGE_PATH)

    # OCR + timing
    tesseract_text, tesseract_time = ocr_tesseract(processed_img)
    easyocr_text, easyocr_time = ocr_easyocr(IMAGE_PATH)
    paddleocr_text, paddleocr_time = ocr_paddleocr_placeholder(processed_img)

    # Build JSON Response
    response = {
        "Input_Image_Base64": image_to_base64(IMAGE_PATH),
        "Tesseract": {
            "ocr_response": tesseract_text,
            "execution_time_sec": round(tesseract_time, 3),
            "text_length": len(tesseract_text)
        },
        "EasyOCR": {
            "ocr_response": easyocr_text,
            "execution_time_sec": round(easyocr_time, 3),
            "text_length": len(easyocr_text)
        },
        "PaddleOCR": {
            "ocr_response": paddleocr_text,
            "execution_time_sec": paddleocr_time,
            "text_length": len(paddleocr_text)
        }
    }

    # Save results to file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=4, ensure_ascii=False)

    return jsonify(response)

# ---------------------------
# Run Flask App
# ---------------------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
