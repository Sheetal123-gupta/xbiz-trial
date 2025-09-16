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

# -----------------------------
# Config
# -----------------------------
BASE_DIR = os.path.dirname(__file__)  # folder where app.py is
INPUT_FOLDER = os.path.join(BASE_DIR, "input_images")  # folder inside project
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs") 
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Keyword dictionary

DOCUMENT_KEYWORDS = {
    "PAN Card": [
        "income tax department", "permanent account number", "govt of india", "father's name"
    ],
    "Aadhaar Card": [
        "aadhaar", "uidai", "government of india", "year of birth", "date of birth", "gender"
    ],
    "Voter ID Card": [
        "election commission of india", "voter id", "elector's photo identity card",
        "elector's name", "sex", "epic"
    ],
    "Passport": [
        "passport", "republic of india", "place of birth", "date of issue", "date of expiry"
    ],
    "Driving License": [
        "driving license", "dl no", "valid till", "transport", "date of issue", "dob"
    ],
    "Bank Passbook": [
        "account number", "ifsc", "branch", "customer id", "balance", "transaction", "a/c"
    ]
}

# -----------------------------
# Regex patterns
# -----------------------------
PAN_PATTERN = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
AADHAR_PATTERN = re.compile(r"^\d{4}\s\d{4}\s\d{4}$")
AADHAR_PATTERN_NOSPACE = re.compile(r"^\d{12}$")
DL_NUMBER_PATTERN = re.compile(r"^[A-Z]{2}\d{2}[0-9A-Z]{11,}$")
VOTER_PATTERN = re.compile(r"^[A-Z]{3}[0-9]{7}$")
PASSPORT_PATTERN = re.compile(r"^[A-Z][0-9]{7}$")
IFSC_PATTERN = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")

# -----------------------------
# Document classifier
# -----------------------------
def classify_document(raw_blocks):
    all_text = " ".join(raw_blocks).lower()
    scores = {}
    for doc, keywords in DOCUMENT_KEYWORDS.items():
        score = max(fuzz.partial_ratio(all_text, kw.lower()) for kw in keywords)
        scores[doc] = score
    best_doc = max(scores, key=scores.get)
    return best_doc, scores

# -----------------------------
# Side classifier
# -----------------------------
def classify_side(doc_type, raw_blocks):
    all_text = " ".join(raw_blocks).upper()
    side = "Unknown"

    def fuzzy_in(text, indicators, threshold=70):
        for ind in indicators:
            if fuzz.partial_ratio(text, ind.upper()) >= threshold:
                return True
        return False

    if doc_type == "PAN Card":
        if fuzzy_in(all_text, ["INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER", "GOVT. OF INDIA"]) \
           or any(PAN_PATTERN.match(block.replace(" ", "")) for block in raw_blocks):
            side = "Front"
        elif fuzzy_in(all_text, ["QR CODE", "NSDL", "UTIITSL"]):
            side = "Back"

    elif doc_type == "Aadhaar Card":
        if fuzzy_in(all_text, ["GOVERNMENT OF INDIA", "AADHAAR", "UIDAI", "DOB", "GENDER"]) \
           or any(AADHAR_PATTERN.match(b) or AADHAR_PATTERN_NOSPACE.match(b) for b in raw_blocks):
            side = "Front"
        elif fuzzy_in(all_text, ["ADDRESS", "DISTRICT", "STATE", "PIN", "CARE OF"]):
            side = "Back"

    elif doc_type == "Voter ID Card":
        if fuzzy_in(all_text, ["ELECTION COMMISSION OF INDIA", "ELECTOR'S PHOTO IDENTITY CARD", "NAME", "FATHER", "DOB"]) \
           or any(VOTER_PATTERN.match(b) for b in raw_blocks):
            side = "Front"
        elif fuzzy_in(all_text, ["ADDRESS", "DISTRICT", "STATE", "PIN CODE", "ISSUE DATE"]):
            side = "Back"

    elif doc_type == "Passport":
        if fuzzy_in(all_text, ["PASSPORT", "REPUBLIC OF INDIA", "NATIONALITY", "DATE OF BIRTH"]) \
           or any(PASSPORT_PATTERN.match(b) for b in raw_blocks):
            side = "Front"
        elif fuzzy_in(all_text, ["ADDRESS", "EMERGENCY CONTACT", "PLACE OF ISSUE"]):
            side = "Back"

    elif doc_type == "Bank Passbook":
        if fuzzy_in(all_text, ["ACCOUNT NUMBER", "IFSC", "BRANCH", "CUSTOMER ID", "SAVINGS ACCOUNT"]) \
           or any(IFSC_PATTERN.match(b) for b in raw_blocks):
            side = "Front"
        elif fuzzy_in(all_text, ["DEPOSIT", "WITHDRAWAL", "BALANCE", "CHEQUE"]):
            side = "Back"

    return side

# -----------------------------
# Cleaned summary generator
# -----------------------------
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
        upper_text = clean_text.upper()

        if "ELECTION COMMISSION OF INDIA" in upper_text:
            summary["Issuing Authority"] = "Election Commission of India"
        if "GOVERNMENT OF INDIA" in upper_text:
            summary["Issuing Authority"] = "Government of India"

        if "NAME" in upper_text and "FATHER" not in upper_text:
            summary["Name"] = clean_text.split(":")[-1].strip()
        if "FATHER" in upper_text:
            summary["Father’s Name"] = clean_text.split(":")[-1].strip()

        dob_match = re.search(r"\d{2}[-/]\d{2}[-/]\d{4}", clean_text)
        if dob_match:
            summary["DOB"] = dob_match.group()

        if PAN_PATTERN.match(clean_text):
            summary["Number"] = clean_text
        elif AADHAR_PATTERN.match(clean_text) or AADHAR_PATTERN_NOSPACE.match(clean_text):
            summary["Number"] = clean_text
        elif DL_NUMBER_PATTERN.match(clean_text):
            summary["Number"] = clean_text
        elif VOTER_PATTERN.match(clean_text):
            summary["Number"] = clean_text
        elif PASSPORT_PATTERN.match(clean_text):
            summary["Number"] = clean_text

        if clean_text and len(clean_text) > 2:
            summary["Other Details"].append(clean_text)

    return summary

# -------------OCR PROCESSING ----------------

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
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        roi = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, config="--psm 6").strip()
        if text:
            extracted_blocks.append(text)

    # Document classification
    doc_type, fuzzy_scores = classify_document(extracted_blocks)

    # Side classification
    side = classify_side(doc_type, extracted_blocks)

    # Generate cleaned summary
    summary = generate_cleaned_summary(extracted_blocks, doc_type)

    # Output
    output_data = {
        "filename": os.path.basename(image_path),
        "raw_detected_text": extracted_blocks,
        "cleaned_summary": summary,
        "document_type": doc_type,
        "side": side,
        "fuzzy_scores": fuzzy_scores
    }

    # Save output JSON
    ts = time.strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    json_filename = os.path.join(OUTPUT_FOLDER, f"{base_name}_{ts}.json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    # Save processed image
    result_image_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_{ts}_output.jpg")
    cv2.imwrite(result_image_path, image)

    return output_data

# -----------------------------
# Flask Routes
# -----------------------------
@app.route('/')
def home():
    return "Flask OCR API with fuzzy classification is running!"

@app.route('/process-all', methods=['GET'])
def process_all_files():
    results = []
    for file_name in os.listdir(INPUT_FOLDER):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(INPUT_FOLDER, file_name)
            result = process_document(image_path)
            results.append(result)
    return jsonify(results)

@app.route('/process/<filename>', methods=['GET'])
def process_single_file(filename):
    image_path = os.path.join(INPUT_FOLDER, filename)
    if not os.path.exists(image_path):
        return jsonify({"error": f"File {filename} not found in {INPUT_FOLDER}"}), 404
    result = process_document(image_path)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
