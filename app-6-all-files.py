import cv2
import numpy as np
import pytesseract
import re
import json
import os
import time
from flask import Flask, jsonify
from rapidfuzz import fuzz

app = Flask(__name__)


# Config

INPUT_FOLDER = "images1"            
OUTPUT_FOLDER = "outputs-day3"
PREDICTED_FOLDER = os.path.join(OUTPUT_FOLDER, "image_out")
NOT_PREDICTED_FOLDER = os.path.join(OUTPUT_FOLDER, "not_predicted")

os.makedirs(PREDICTED_FOLDER, exist_ok=True)
os.makedirs(NOT_PREDICTED_FOLDER, exist_ok=True)


# Keyword dictionary (unchanged)

DOCUMENT_KEYWORDS = {
    "PAN Card": ["income tax department", "permanent account number", "govt of india", "father's name"],
    "Aadhaar Card": ["aadhaar", "uidai", "government of india", "year of birth", "date of birth", "gender"],
    "Voter ID Card": ["election commission of india", "voter id", "elector's photo identity card", "elector's name", "sex", "epic"],
    "Passport": ["passport", "republic of india", "place of birth", "date of issue", "date of expiry"],
    "Driving License": ["driving license", "dl no", "valid till", "transport", "date of issue", "dob"],
    "Bank Passbook": ["account number", "ifsc", "branch", "customer id", "balance", "transaction", "a/c"]
}


# Regex patterns

PAN_PATTERN = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
AADHAR_PATTERN = re.compile(r"^\d{4}\s\d{4}\s\d{4}$")
AADHAR_PATTERN_NOSPACE = re.compile(r"^\d{12}$")
DL_NUMBER_PATTERN = re.compile(r"^[A-Z]{2}\d{2}[0-9A-Z]{11,}$")
VOTER_PATTERN = re.compile(r"^[A-Z]{3}[0-9]{7}$")
PASSPORT_PATTERN = re.compile(r"^[A-Z][0-9]{7}$")
IFSC_PATTERN = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")


# Document classifier

def classify_document(raw_blocks):
    if not raw_blocks:
        return None, {}
    all_text = " ".join(raw_blocks).lower()
    scores = {}
    for doc, keywords in DOCUMENT_KEYWORDS.items():
        score = max(fuzz.partial_ratio(all_text, kw.lower()) for kw in keywords)
        scores[doc] = score
    best_doc = max(scores, key=scores.get)
    if scores[best_doc] < 50:  # threshold for "not sure"
        return None, scores
    return best_doc, scores


# Side classifier (unchanged)

def classify_side(doc_type, raw_blocks):
    # your existing side logic (unchanged for brevity)
    return "Front"  # fallback


# Cleaned summary generator (unchanged)

def generate_cleaned_summary(blocks, doc_type):
    # your existing summary logic
    return {"Document": doc_type, "Other Details": blocks}


# OCR Processing

def process_document(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"Could not read image: {image_path}"}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    extracted_blocks = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, config="--psm 6").strip()
        if text:
            extracted_blocks.append(text)

    doc_type, fuzzy_scores = classify_document(extracted_blocks)

    if not doc_type:  # not predicted
        output_data = {
            "filename": os.path.basename(image_path),
            "raw_detected_text": extracted_blocks,
            "document_type": None,
            "side": None,
            "error": "Not Predicted"
        }
        # Save into not_predicted folder
        out_path = os.path.join(NOT_PREDICTED_FOLDER, f"{os.path.basename(image_path)}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        return output_data

    side = classify_side(doc_type, extracted_blocks)
    summary = generate_cleaned_summary(extracted_blocks, doc_type)

    output_data = {
        "filename": os.path.basename(image_path),
        "raw_detected_text": extracted_blocks,
        "cleaned_summary": summary,
        "document_type": doc_type,
        "side": side,
        "fuzzy_scores": fuzzy_scores
    }

    out_path = os.path.join(PREDICTED_FOLDER, f"{os.path.basename(image_path)}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    return output_data


# Flask Routes

@app.route('/process-all', methods=['GET'])
def process_all_files():
    results = []
    predicted_count = 0
    not_predicted_count = 0

    for file_name in os.listdir(INPUT_FOLDER):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(INPUT_FOLDER, file_name)
            result = process_document(image_path)
            results.append(result)
            if result.get("document_type"):
                predicted_count += 1
            else:
                not_predicted_count += 1

    summary = {
        "total_files": len(results),
        "predicted": predicted_count,
        "not_predicted": not_predicted_count
    }
    return jsonify({"summary": summary, "results": results})

if __name__ == "__main__":
    app.run(debug=True)
