import cv2
import numpy as np
import pytesseract
import re
import json
import os
import time
from flask import Flask, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)  
IMAGE_NAME = "document_side_detection/pan5.jpg"  
IMAGE_PATH = os.path.join(BASE_DIR, IMAGE_NAME)
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

REGEX_PATTERNS = {
    "PAN": re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$"),
    "AADHAAR": re.compile(r"^\d{4}\s\d{4}\s\d{4}$"),
    "AADHAAR_NOSPACE": re.compile(r"^\d{12}$"),
    "DL": re.compile(r"^[A-Z]{2}\d{2}[0-9A-Z]{11,}$"),
    "VOTER": re.compile(r"^[A-Z]{3}[0-9]{7}$"),
    "PASSPORT": re.compile(r"^[A-Z][0-9]{7}$"),
    "IFSC": re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")
}

DOC_RULES = {
    "PAN Card": {
        "keywords": ["PERMANENT ACCOUNT NUMBER", "INCOME TAX DEPARTMENT", "PAN"],
        "front": ["INCOME TAX", "PAN", "GOVERNMENT OF INDIA", "NAME", "FATHER", "DATE OF BIRTH"],
        "back": ["QR CODE", "VERIFY AUTHENTICITY", "NSDL", "UTIITSL"],
        "patterns": ["PAN"]
    },
    "Aadhaar Card": {
        "keywords": ["AADHAAR", "GOVERNMENT OF INDIA", "UNIQUE IDENTIFICATION"],
        "front": ["GOVERNMENT OF INDIA", "AADHAAR", "DOB", "GENDER", "NAME"],
        "back": ["ADDRESS", "DISTRICT", "STATE", "PIN"],
        "patterns": ["AADHAAR", "AADHAAR_NOSPACE"]
    },
    "Voter ID Card": {
        "keywords": ["ELECTION COMMISSION OF INDIA", "VOTER ID"],
        "front": ["ELECTION COMMISSION OF INDIA", "NAME", "DOB", "FATHER"],
        "back": ["ADDRESS", "DISTRICT", "STATE", "PIN"],
        "patterns": ["VOTER"]
    },
    "Passport": {
        "keywords": ["PASSPORT", "REPUBLIC OF INDIA"],
        "front": ["PASSPORT", "NAME", "NATIONALITY", "DATE OF BIRTH"],
        "back": ["ADDRESS", "EMERGENCY CONTACT"],
        "patterns": ["PASSPORT"]
    },
    "Bank Passbook": {
        "keywords": ["ACCOUNT", "IFSC", "SAVINGS", "CURRENT"],
        "front": ["ACCOUNT NUMBER", "IFSC", "BRANCH", "CUSTOMER ID"],
        "back": ["DEPOSIT", "WITHDRAWAL", "BALANCE"],
        "patterns": ["IFSC"]
    }
}

def extract_text_blocks(image,draw_boxes=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dilated = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    extracted = []
    for contour in sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1]):
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, config="--psm 6").strip()
        if text:
            extracted.append(text)
            if draw_boxes:
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    return extracted,image

def detect_document_type(blocks):
    """Detect the document type based on regex and keywords."""
    text_all = " ".join(blocks).upper()
    for doc, rules in DOC_RULES.items():
        if any(kw in text_all for kw in rules["keywords"]):
            return doc
        for pattern_key in rules["patterns"]:
            if any(REGEX_PATTERNS[pattern_key].match(b.replace(" ", "")) for b in blocks):
                return doc
    return "Unknown Document"

def detect_document_side(doc_type, blocks):
    """Detect document side (Front/Back) using indicators."""
    if doc_type not in DOC_RULES:
        return "Unknown"
    text_all = " ".join(blocks).upper()
    rules = DOC_RULES[doc_type]

    # Regex checks
    if any(REGEX_PATTERNS.get(p) and REGEX_PATTERNS[p].match(b.replace(" ", "")) for p in rules["patterns"] for b in blocks):
        return "Front"

    if any(ind in text_all for ind in rules["front"]):
        return "Front"
    elif any(ind in text_all for ind in rules["back"]):
        return "Back"
    return "Unknown"

def generate_summary(blocks, doc_type):
    """Generate a structured summary from extracted text blocks."""
    summary = {"Document": doc_type, "Name": None, "Father’s Name": None,
               "DOB": None, "Number": None, "Issuing Authority": None, "Other Details": []}

    for text in blocks:
        clean = text.strip()
        upper = clean.upper()

        if "ELECTION COMMISSION OF INDIA" in upper:
            summary["Issuing Authority"] = "Election Commission of India"
        if "GOVERNMENT OF INDIA" in upper:
            summary["Issuing Authority"] = "Government of India"

        if "NAME" in upper and "FATHER" not in upper:
            summary["Name"] = clean.split(":")[-1].strip()
        if "FATHER" in upper:
            summary["Father’s Name"] = clean.split(":")[-1].strip()

        if dob_match := re.search(r"\d{2}[-/]\d{2}[-/]\d{4}", clean):
            summary["DOB"] = dob_match.group()

        # Match number by regex
        for key, pattern in REGEX_PATTERNS.items():
            if pattern.match(clean.replace(" ", "")):
                summary["Number"] = clean
                break

        if len(clean) > 2:
            summary["Other Details"].append(clean)

    return summary

def process_document(image_path):
    """Main processing pipeline."""
    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"Could not read image: {image_path}"}

    blocks,image = extract_text_blocks(image,draw_boxes=True)
    doc_type = detect_document_type(blocks)
    side = detect_document_side(doc_type, blocks)
    summary = generate_summary(blocks, doc_type)

    output_data = {
        "filename": os.path.basename(image_path),
        "raw_detected_text": blocks,
        "cleaned_summary": summary,
        "document_type": doc_type,
        "side": side
    }

    # Save JSON + image
    ts = time.strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    json_filename = os.path.join(OUTPUT_FOLDER, f"{base_name}_{ts}.json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    result_image_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_{ts}_output.jpg")
    cv2.imwrite(result_image_path, image)

    return output_data

@app.route('/')
def home():
    return "Flask OCR API is running! Use /process-manual to test."

@app.route('/process-manual', methods=['GET'])
def process_manual_file():
    image_name = "C:\\Users\\ASUS\\Music\\xbiz-trial\\document_type_detection\\a2.jpg"
    image_path = os.path.join(os.getcwd(), image_name)
    result = process_document(image_path)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
