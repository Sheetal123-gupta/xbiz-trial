import cv2
import numpy as np
import pytesseract
import re
import json
import os
from flask import Flask, jsonify

app = Flask(__name__)

# -----------------------------
# Directories
# -----------------------------
BASE_DIR = os.path.dirname(__file__)  # folder of app.py
IMAGE_NAME = "a2.jpg"  # hardcoded image
IMAGE_PATH = os.path.join(BASE_DIR, IMAGE_NAME)
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs-day3")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# OCR & Document Processing
# -----------------------------
def process_document(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    # Document flags
    is_pan = is_aadhar = is_bank_passbook = is_driving_license = is_voter_id = False

    # Regex patterns
    pan_pattern = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
    aadhar_pattern = re.compile(r"^\d{4}\s\d{4}\s\d{4}$")
    aadhar_pattern_nospace = re.compile(r"^\d{12}$")
    dl_number_pattern = re.compile(r"^[A-Z]{2}\d{2}[0-9A-Z]{11,}$")

    # Keywords
    bank_keywords = ["IFSC", "CIF", "ACCOUNT", "A/C", "SAVING", "SB A/C", "CURRENT"]
    dl_keywords = ["DRIVING LICENCE", "DRIVING LICENSE", "DL NO", "VALID TILL", "DATE OF ISSUE", "DOB", "AUTHORISATION TO DRIVE"]
    voter_keywords = ["ELECTION COMMISSION OF INDIA", "VOTER ID", "ELECTOR'S PHOTO IDENTITY CARD"]

    extracted_blocks = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, config="--psm 6").strip()
        text_clean = text.upper().replace(" ", "")
        extracted_blocks.append(text)

        # Document detection
        if "PERMANENTACCOUNTNUMBER" in text_clean or "INCOMETAXDEPARTMENT" in text_clean or pan_pattern.match(text_clean):
            is_pan = True
        if aadhar_pattern.match(text.strip()) or aadhar_pattern_nospace.match(text.strip()) or "GOVERNMENTOFINDIA" in text_clean:
            is_aadhar = True
        for keyword in bank_keywords:
            if keyword in text.upper():
                is_bank_passbook = True
        for keyword in dl_keywords:
            if keyword in text.upper() or dl_number_pattern.match(text_clean):
                is_driving_license = True
        for keyword in voter_keywords:
            if keyword in text.upper():
                is_voter_id = True

    # Determine document type
    doc_type = "Unknown Document"
    if is_pan:
        doc_type = "PAN Card"
    elif is_aadhar:
        doc_type = "Aadhaar Card"
    elif is_bank_passbook:
        doc_type = "Bank Passbook"
    elif is_driving_license:
        doc_type = "Driving License"
    elif is_voter_id:
        doc_type = "Voter ID Card"

    # -----------------------
    # Cleaned Summary
    # -----------------------
    def generate_cleaned_summary(blocks, doc_type):
        summary = {
            "Document": doc_type,
            "Name": None,
            "Father’s Name": None,
            "DOB": None,
            "Number": None,
            "Issuing Authority": None,
            "Other Details": []
        }
        for text in blocks:
            clean_text = text.strip()
            if "ELECTION COMMISSION OF INDIA" in clean_text.upper():
                summary["Issuing Authority"] = "Election Commission of India"
            if "GOVERNMENT OF INDIA" in clean_text.upper():
                summary["Issuing Authority"] = "Government of India"
            if "NAME" in clean_text.upper() and "FATHER" not in clean_text.upper():
                summary["Name"] = clean_text.split(":")[-1].strip()
            if "FATHER" in clean_text.upper():
                summary["Father’s Name"] = clean_text.split(":")[-1].strip()
            dob_match = re.search(r"\d{2}[-/]\d{2}[-/]\d{4}", clean_text)
            if dob_match:
                summary["DOB"] = dob_match.group()
            if re.match(r"^\d{4}\s\d{4}\s\d{4}$", clean_text):
                summary["Number"] = clean_text
            elif re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", clean_text):
                summary["Number"] = clean_text
            elif re.match(r"^[A-Z]{2}\d{2}[0-9A-Z]{11,}$", clean_text):
                summary["Number"] = clean_text
            if clean_text and len(clean_text) > 2:
                summary["Other Details"].append(clean_text)
        return summary

    summary = generate_cleaned_summary(extracted_blocks, doc_type)

    output_data = {
        "filename": os.path.basename(image_path),
        "raw_detected_text": extracted_blocks,
        "cleaned_summary": summary,
        "document_type": doc_type
    }

    # Save JSON and image
    json_filename = os.path.join(OUTPUT_FOLDER, os.path.splitext(os.path.basename(image_path))[0] + ".json")
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, os.path.splitext(os.path.basename(image_path))[0] + "_output.jpg"), image)
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    return output_data

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return "Flask OCR API is running! Hardcoded image processing."

@app.route('/process-manual', methods=['GET'])
def process_manual_file():
    try:
        result = process_document(IMAGE_PATH)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    return jsonify(result)

# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
