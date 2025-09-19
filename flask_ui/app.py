from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import pytesseract
import re
import json
import os
import uuid
import base64
from werkzeug.utils import secure_filename
from rapidfuzz import fuzz  # ✅ Added RapidFuzz

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

# -----------------------------
# Document classification keywords
# -----------------------------
DOCUMENT_KEYWORDS = {
    "Driving License": ["DL NO", "DRIVING LICENCE", "AUTHORISATION TO DRIVE", "MCWG", "TRANSPORT", "RTO", "VALID TILL"],
    "Aadhaar Card": ["AADHAAR", "UNIQUE IDENTIFICATION AUTHORITY", "VID", "GOVERNMENT OF INDIA", "UIDAI"],
    "PAN Card": ["INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER", "INCOME TAX"],
    "Bank Passbook": ["BANK OF", "ACCOUNT NO", "IFSC", "PASSBOOK", "BRANCH"],
    "Voter ID Card": ["ELECTION COMMISSION OF INDIA", "VOTER ID", "EPIC NO","ELECTION","ELECTORAL REGISTRATION OFFICER"],
    "Passport": ["REPUBLIC OF INDIA", "PASSPORT", "TYPE", "CODE", "DATE OF ISSUE"]
}

# -----------------------------
# OCR & Document Processing Logic
# -----------------------------
def process_document(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    extracted_blocks = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, config="--psm 6").strip()
        if text:
            extracted_blocks.append(text)
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw text above rectangle
            text_to_display = text.replace('\n', ' ')
            font_scale = 0.5
            thickness = 1
            text_size, _ = cv2.getTextSize(text_to_display, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x = x
            text_y = y - 5 if y - 5 > 10 else y + h + 15
            cv2.putText(image, text_to_display, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

    full_text = " ".join(extracted_blocks).upper()

    # -----------------------------
    # Document classification using RapidFuzz
    # -----------------------------
    best_match = ("Unknown Document", 0)
    for doc_type, keywords in DOCUMENT_KEYWORDS.items():
        score = 0
        for kw in keywords:
            ratio = fuzz.partial_ratio(kw, full_text)
            if ratio > 80:
                score += ratio
        if score > best_match[1]:
            best_match = (doc_type, score)
    doc_type = best_match[0]

    # -----------------------------
    # Side detection logic
    # -----------------------------
    doc_side = "Unknown Side"
    blocks_upper = [b.upper() for b in extracted_blocks]

    if doc_type == "Aadhaar Card":
        if any(fuzz.partial_ratio(k, b) > 80 for k in ["GOVERNMENT OF INDIA", "AADHAAR"] for b in blocks_upper):
            doc_side = "Front"
        elif any(fuzz.partial_ratio(k, b) > 80 for k in ["VID", "ENROLMENT", "ADDRESS"] for b in blocks_upper):
            doc_side = "Back"
    elif doc_type == "PAN Card":
        if any(fuzz.partial_ratio(k, b) > 80 for k in ["INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER"] for b in blocks_upper):
            doc_side = "Front"
        elif any(fuzz.partial_ratio(k, b) > 80 for k in ["PAN SERVICES UNIT", "NSDL", "UTIITSL"] for b in blocks_upper):
            doc_side = "Back"

    elif doc_type == "Voter ID Card":
        if any(fuzz.partial_ratio(k, b) > 70 for k in ["ELECTION COMMISSION OF INDIA",
            "VOTER ID",
            "EPIC NO",
            "PHOTO",
            "GENDER",
            "FATHER"] for b in blocks_upper):
            doc_side = "Front"
        elif any(fuzz.partial_ratio(k, b) > 60 for k in ["ELECTORAL REGISTRATION OFFICER",
            "CONSTITUENCY",
            "FACSIMILE SIGNATURE",
            "ADDRESS"] for b in blocks_upper):
            doc_side = "Back" 


    elif doc_type == "Driving License":
        if any(fuzz.partial_ratio(k, b) > 80 for k in ["DRIVING LICENCE", "DL NO"] for b in blocks_upper):
            doc_side = "Front"
        elif any(fuzz.partial_ratio(k, b) > 80 for k in ["AUTHORISATION TO DRIVE", "COV", "TRANSPORT"] for b in blocks_upper):
            doc_side = "Back"

    # -----------------------------
    # Cleaned Summary Function
    # -----------------------------
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
            if fuzz.partial_ratio("ELECTION COMMISSION OF INDIA", clean_text.upper()) > 85:
                summary["Issuing Authority"] = "Election Commission of India"
            if fuzz.partial_ratio("GOVERNMENT OF INDIA", clean_text.upper()) > 85:
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

    # Save result image to file
    result_image_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_name}_{unique_id}_output.jpg")
    cv2.imwrite(result_image_path, image)

    # ✅ Convert processed image to base64
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    output_data = {
        "filename": os.path.basename(image_path),
        "raw_detected_text": extracted_blocks,
        "cleaned_summary": summary,
        "document_type": doc_type,
        "document_side": doc_side,
        "processed_img_base64":img_base64
    }

    # Save output JSON
    json_filename = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_name}_{unique_id}.json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    # Save output image with contours
    result_image_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_name}_{unique_id}_output.jpg")
    cv2.imwrite(result_image_path, image)

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