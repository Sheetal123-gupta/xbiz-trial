from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import pytesseract
import re
import json
import os
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# OCR & Document Processing Logic
def process_document(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    # Document detection flags
    is_pan = False
    is_aadhar = False
    is_bank_passbook = False
    is_driving_license = False
    is_voter_id = False
    is_passport_id = False

    # Regex patterns
    pan_pattern = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
    aadhar_pattern = re.compile(r"^\d{4}\s\d{4}\s\d{4}$")
    aadhar_pattern_nospace = re.compile(r"^\d{12}$")
    dl_number_pattern = re.compile(r"^[A-Z]{2}\d{2}[0-9A-Z]{11,}$")

    # Keywords for document type
    bank_keywords = ["IFSC", "CIF", "ACCOUNT", "A/C", "SAVING", "SB A/C", "CURRENT","BRANCH CODE","BRANCH"]
    dl_keywords = ["DRIVING LICENCE", "DRIVING LICENSE", "DL NO", "VALID TILL", "DATE OF ISSUE", "DOB", "AUTHORISATION TO DRIVE"]
    voter_keywords = ["ELECTION COMMISSION OF INDIA", "VOTER ID", "ELECTOR'S PHOTO IDENTITY CARD"]
    passport_keywords = ["REPUBLIC OF INDIA", "PASSPORT", "TYPE", "CODE", "DATE OF ISSUE", "DATE OF EXPIRY"]

    # Side detection keywords
    aadhaar_front_keywords = ["GOVERNMENT OF INDIA", "AADHAAR", "UIDAI"]
    aadhaar_back_keywords = ["VID", "ENROLMENT", "HELPLINE", "WWW.UIDAI.GOV.IN", "ADDRESS"]

    pan_front_keywords = ["INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER"]
    pan_back_keywords = ["INCOME TAX PAN SERVICES UNIT", "CBD BELAPUR", "NSDL", "UTIITSL"]

    dl_front_keywords = ["DRIVING LICENCE", "DL NO", "VALID TILL"]
    dl_back_keywords = ["AUTHORISED TO DRIVE", "COV", "TRANSPORT", "NON-TRANSPORT"]

    voter_front_keywords = ["ELECTION","ELECTION COMMISSION OF INDIA", "PHOTO IDENTITY CARD"]
    voter_back_keywords = ["ADDRESS", "EPIC NO"]

    passport_front_keywords = ["REPUBLIC OF INDIA", "PASSPORT", "TYPE", "CODE"]
    passport_back_keywords = ["PARENTS NAME", "ADDRESS", "PLACE OF ISSUE"]

    extracted_blocks = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, config="--psm 6").strip()
        text_clean = text.upper().replace(" ", "")
        extracted_blocks.append(text)

        # Document detection
        if "PERMANENTACCOUNTNUMBER" in text_clean or "INCOMETAXDEPARTMENT" in text_clean or "INCOMETAXPAN" in text_clean or pan_pattern.match(text_clean):
            is_pan = True
        if aadhar_pattern.match(text.strip()) or aadhar_pattern_nospace.match(text.strip()) or "GOVERNMENTOFINDIA" in text_clean or "UNIQUEIDENTIFICATIONAUTHORITYOFINDIA" in text_clean:
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
        for keyword in passport_keywords:
            if keyword in text.upper():
                is_passport_id = True

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
    elif is_passport_id:
        doc_type = "Passport"

    # Determine document side
    doc_side = "Unknown Side"
    blocks_upper = [b.upper() for b in extracted_blocks]

    if is_aadhar:
        if any(any(k in b for k in aadhaar_front_keywords) for b in blocks_upper):
            doc_side = "Front"
        elif any(any(k in b for k in aadhaar_back_keywords) for b in blocks_upper):
            doc_side = "Back"
    elif is_pan:
        if any(any(k in b for k in pan_front_keywords) for b in blocks_upper):
            doc_side = "Front"
        elif any(any(k in b for k in pan_back_keywords) for b in blocks_upper):
            doc_side = "Back"
    elif is_driving_license:
        if any(any(k in b for k in dl_front_keywords) for b in blocks_upper):
            doc_side = "Front"
        elif any(any(k in b for k in dl_back_keywords) for b in blocks_upper):
            doc_side = "Back"
    elif is_voter_id:
        if any(any(k in b for k in voter_front_keywords) for b in blocks_upper):
            doc_side = "Front"
        elif any(any(k in b for k in voter_back_keywords) for b in blocks_upper):
            doc_side = "Back"
    elif is_passport_id:
        if any(any(k in b for k in passport_front_keywords) for b in blocks_upper):
            doc_side = "Front"
        elif any(any(k in b for k in passport_back_keywords) for b in blocks_upper):
            doc_side = "Back"

    # Cleaned Summary Function
    def generate_cleaned_summary(blocks, doc_type):
        summary = {
            "Document": doc_type,
            "Name": None,
            "Father's Name": None,
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
                summary["Father's Name"] = clean_text.split(":")[-1].strip()
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

    # Generate unique filename for output
    unique_id = str(uuid.uuid4())[:8]
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    output_data = {
        "filename": os.path.basename(image_path),
        "raw_detected_text": extracted_blocks,
        "cleaned_summary": summary,
        "document_type": doc_type,
        "document_side": doc_side  # ✅ Added side info
    }

    # Save output JSON
    json_filename = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_name}_{unique_id}.json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    # Save output image with contours
    result_image_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_name}_{unique_id}_output.jpg")
    cv2.imwrite(result_image_path, image)

    # Add the processed image filename to the response
    output_data['processed_image'] = f"{base_name}_{unique_id}_output.jpg"

    return output_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename
        })
    
    return jsonify({'error': 'File type not allowed'})

@app.route('/extract/<filename>', methods=['GET'])
def extract_text(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'})
    
    try:
        result = process_document(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
