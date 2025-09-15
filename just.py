#app-3 ka correction he 
import cv2
import numpy as np
import pytesseract
import re
import json
import os
import time
from flask import Flask, jsonify

app = Flask(__name__)

# -----------------------------
# Config
# -----------------------------
OUTPUT_FOLDER = "outputs-day3"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# OCR & Document Processing
# -----------------------------
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

    # Flags
    is_pan = False
    is_aadhar = False
    is_bank_passbook = False
    is_driving_license = False
    is_voter_id = False
    is_passport = False

    # Regex patterns
    pan_pattern = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
    aadhar_pattern = re.compile(r"^\d{4}\s\d{4}\s\d{4}$")
    aadhar_pattern_nospace = re.compile(r"^\d{12}$")
    dl_number_pattern = re.compile(r"^[A-Z]{2}\d{2}[0-9A-Z]{11,}$")
    voter_pattern = re.compile(r"^[A-Z]{3}[0-9]{7}$")
    passport_pattern = re.compile(r"^[A-Z][0-9]{7}$")
    ifsc_pattern = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")

    # Keywords
    bank_keywords = ["IFSC", "CIF", "ACCOUNT", "A/C", "SAVING", "SB A/C", "CURRENT"]
    dl_keywords = ["DRIVING LICENCE", "DRIVING LICENSE", "DL NO", "VALID TILL", "DATE OF ISSUE", "DOB"]
    voter_keywords = ["ELECTION COMMISSION OF INDIA", "VOTER ID", "ELECTOR'S PHOTO IDENTITY CARD"]
    passport_keywords = ["PASSPORT", "REPUBLIC OF INDIA"]

    extracted_blocks = []

    # OCR block by block
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, config="--psm 6").strip()
        text_clean = text.upper().replace(" ", "")
        extracted_blocks.append(text)

        # Detect PAN
        if "PERMANENTACCOUNTNUMBER" in text_clean or "INCOMETAXPANSERVICESUNIT" in text_clean or "INCOMETAXDEPARTMENT" in text_clean or pan_pattern.match(text_clean):
            is_pan = True
        # Detect Aadhaar
        if aadhar_pattern.match(text.strip()) or aadhar_pattern_nospace.match(text.strip()) or "GOVERNMENTOFINDIA" in text_clean or "UNIQUEIDENTIFICATIONAUTHORITYOFINDIA" in text_clean:
            is_aadhar = True
        # Detect Bank
        for keyword in bank_keywords:
            if keyword in text.upper():
                is_bank_passbook = True
        # Detect DL
        for keyword in dl_keywords:
            if keyword in text.upper() or dl_number_pattern.match(text_clean):
                is_driving_license = True
        # Detect Voter
        for keyword in voter_keywords:
            if keyword in text.upper() or voter_pattern.match(text.strip()):
                is_voter_id = True
        # Detect Passport
        for keyword in passport_keywords:
            if keyword in text.upper() or passport_pattern.match(text.strip()):
                is_passport = True

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
    elif is_passport:
        doc_type = "Passport"

    side = "Unknown"
    all_text = " ".join(extracted_blocks).upper()

    if doc_type == "PAN Card":
        front_indicators = ["PERMANENT ACCOUNT NUMBER",
        "INCOME TAX DEPARTMENT",
        "GOVT. OF INDIA",
        "GOVERNMENT OF INDIA",
        "NAME",
        "FATHER",
        "DATE OF BIRTH"]
        back_indicators = ["INCOME TAX PAN SERVICES UNIT",
        "CBD BELAPUR",
        "NSDL",
        "UTIITSL",
        "QR CODE",
        "SCAN THIS CODE",
        "VERIFY AUTHENTICITY"]
        has_pan_number = any(pan_pattern.match(block.replace(" ", "")) for block in extracted_blocks)
        if any(ind in all_text for ind in front_indicators) or has_pan_number:
            side = "Front"
        elif any(ind in all_text for ind in back_indicators):
            side = "Back"

    elif doc_type == "Aadhaar Card":
        front_indicators = ["GOVERNMENT OF INDIA", "AADHAAR", "NAME", "DOB", "GENDER"]
        back_indicators = ["ADDRESS","UNIQUE IDENTIFICATION","UNIQUE IDENTIFICATION AUTHORITY OF INDIA", "DISTRICT", "STATE", "PIN", "C/O", "CARE OF", "MOBILE"]
        has_aadhar_number = any(aadhar_pattern.match(block.strip()) or aadhar_pattern_nospace.match(block.strip()) for block in extracted_blocks)
        if any(ind in all_text for ind in front_indicators) or has_aadhar_number:
            side = "Front"
        elif any(ind in all_text for ind in back_indicators):
            side = "Back"

    elif doc_type == "Voter ID Card":
        front_indicators = ["ELECTION COMMISSION OF INDIA", "VOTER ID", "ELECTOR'S PHOTO IDENTITY CARD", "NAME", "FATHER", "DOB", "GENDER"]
        back_indicators = ["ADDRESS", "DISTRICT", "STATE", "PIN CODE", "C/O", "CARE OF", "ISSUE DATE"]
        has_voter_number = any(voter_pattern.match(block.strip()) for block in extracted_blocks)
        if any(ind in all_text for ind in front_indicators) or has_voter_number:
            side = "Front"
        elif any(ind in all_text for ind in back_indicators):
            side = "Back"

    elif doc_type == "Passport":
        front_indicators = ["REPUBLIC OF INDIA", "PASSPORT", "NAME", "NATIONALITY", "DATE OF BIRTH", "SEX"]
        back_indicators = ["ADDRESS", "EMERGENCY CONTACT", "PLACE OF ISSUE", "ISSUING AUTHORITY"]
        has_passport_number = any(passport_pattern.match(block.strip()) for block in extracted_blocks)
        if any(ind in all_text for ind in front_indicators) or has_passport_number:
            side = "Front"
        elif any(ind in all_text for ind in back_indicators):
            side = "Back"

    elif doc_type == "Bank Passbook":
        front_indicators = ["ACCOUNT NUMBER", "IFSC", "BRANCH", "MICR", "CUSTOMER ID", "SAVINGS ACCOUNT", "CURRENT ACCOUNT"]
        back_indicators = ["DEPOSIT", "WITHDRAWAL", "BALANCE", "CHEQUE NO", "NARRATION"]
        has_ifsc = any(ifsc_pattern.match(block.strip()) for block in extracted_blocks)
        if any(ind in all_text for ind in front_indicators) or has_ifsc:
            side = "Front"
        elif any(ind in all_text for ind in back_indicators):
            side = "Back"

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
            if pan_pattern.match(clean_text):
                summary["Number"] = clean_text
            elif aadhar_pattern.match(clean_text) or aadhar_pattern_nospace.match(clean_text):
                summary["Number"] = clean_text
            elif dl_number_pattern.match(clean_text):
                summary["Number"] = clean_text
            elif voter_pattern.match(clean_text):
                summary["Number"] = clean_text
            elif passport_pattern.match(clean_text):
                summary["Number"] = clean_text
            if clean_text and len(clean_text) > 2:
                summary["Other Details"].append(clean_text)
        return summary

    summary = generate_cleaned_summary(extracted_blocks, doc_type)

    output_data = {
        "filename": os.path.basename(image_path),
        "raw_detected_text": extracted_blocks,
        "cleaned_summary": summary,
        "document_type": doc_type,
        "side": side
    }

    # Save output JSON with timestamp to avoid overwrite
    ts = time.strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    json_filename = os.path.join(OUTPUT_FOLDER, f"{base_name}_{ts}.json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    # Save output image
    result_image_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_{ts}_output.jpg")
    cv2.imwrite(result_image_path, image)
    return output_data


@app.route('/')
def home():
    return "Flask OCR API is running! Use /process-manual to test."

@app.route('/process-manual', methods=['GET'])
def process_manual_file():
    # Replace with your test file
    image_name = "aadhar_back.png"
    image_path = os.path.join(os.getcwd(), image_name)
    result = process_document(image_path)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
