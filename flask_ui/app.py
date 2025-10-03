from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2, numpy as np, pytesseract, re, json, os, uuid, base64
from werkzeug.utils import secure_filename
from rapidfuzz import fuzz

app = Flask(__name__)

# -----------------------------
# Configuration
# -----------------------------
app.config.update(
    SECRET_KEY="your-secret-key-here",
    UPLOAD_FOLDER="uploads",
    OUTPUT_FOLDER="outputs",
    ALLOWED_EXTENSIONS={"png", "jpg", "jpeg", "bmp", "tiff"}
)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# -----------------------------
# Helper Functions
# -----------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def save_json(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def img_to_base64(image) -> str:
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

# -----------------------------
# Document classification keywords
# -----------------------------
DOCUMENT_KEYWORDS = {
    "Driving License": ["DL NO", "DRIVING LICENCE", "AUTHORISATION TO DRIVE", "MCWG", "TRANSPORT", "RTO", "VALID TILL"],
    "Aadhaar Card": ["AADHAAR", "UNIQUE IDENTIFICATION AUTHORITY", "VID", "GOVERNMENT OF INDIA", "UIDAI"],
    "PAN Card": ["INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER", "INCOME TAX"],
    "Bank Passbook": ["BANK OF", "ACCOUNT NO", "IFSC", "PASSBOOK", "BRANCH"],
    "Voter ID Card": ["ELECTION COMMISSION OF INDIA", "VOTER ID", "EPIC NO", "ELECTORAL REGISTRATION OFFICER"],
    "Passport": ["REPUBLIC OF INDIA", "PASSPORT", "TYPE", "CODE", "DATE OF ISSUE"]
}

# -----------------------------
# OCR & Processing
# -----------------------------
def extract_text_blocks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dilated = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocks = []
    for x, y, w, h in [cv2.boundingRect(c) for c in sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])]:
        roi = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, config="--psm 6").strip()
        if text:
            blocks.append(text)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, text.replace("\n", " "), (x, y-5 if y > 20 else y+h+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return blocks

def classify_document(full_text):
    best_match = ("Unknown Document", 0)
    for doc_type, keywords in DOCUMENT_KEYWORDS.items():
        score = sum(fuzz.partial_ratio(kw, full_text) for kw in keywords if fuzz.partial_ratio(kw, full_text) > 80)
        if score > best_match[1]:
            best_match = (doc_type, score)
    return best_match[0]

def detect_side(doc_type, blocks):
    blocks_upper = [b.upper() for b in blocks]
    if doc_type == "Aadhaar Card":
        if any(fuzz.partial_ratio(k, b) > 80 for k in ["GOVERNMENT OF INDIA", "AADHAAR"] for b in blocks_upper):
            return "Front"
        if any(fuzz.partial_ratio(k, b) > 80 for k in ["VID", "ENROLMENT", "ADDRESS"] for b in blocks_upper):
            return "Back"
    if doc_type == "PAN Card":
        if any(fuzz.partial_ratio(k, b) > 80 for k in ["INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER"] for b in blocks_upper):
            return "Front"
        if any(fuzz.partial_ratio(k, b) > 80 for k in ["NSDL", "UTIITSL"] for b in blocks_upper):
            return "Back"
    if doc_type == "Voter ID Card":
        if any(fuzz.partial_ratio(k, b) > 70 for k in ["ELECTION COMMISSION OF INDIA","VOTER ID","EPIC NO","PHOTO","GENDER","FATHER"] for b in blocks_upper):
            return "Front"
        if any(fuzz.partial_ratio(k, b) > 60 for k in ["ELECTORAL REGISTRATION OFFICER","CONSTITUENCY","FACSIMILE SIGNATURE","ADDRESS"] for b in blocks_upper):
            return "Back"
    if doc_type == "Driving License":
        if any(fuzz.partial_ratio(k, b) > 80 for k in ["DRIVING LICENCE", "DL NO"] for b in blocks_upper):
            return "Front"
        if any(fuzz.partial_ratio(k, b) > 80 for k in ["AUTHORISATION TO DRIVE", "COV", "TRANSPORT"] for b in blocks_upper):
            return "Back"
    return "Unknown Side"

def generate_summary(blocks, doc_type):
    summary = {"Document": doc_type, "Name": None, "Father's Name": None,
               "DOB": None, "Number": None, "Issuing Authority": None, "Other Details": []}

    for text in blocks:
        t = text.strip()
        upper = t.upper()
        if fuzz.partial_ratio("ELECTION COMMISSION OF INDIA", upper) > 85:
            summary["Issuing Authority"] = "Election Commission of India"
        if fuzz.partial_ratio("GOVERNMENT OF INDIA", upper) > 85:
            summary["Issuing Authority"] = "Government of India"
        if "NAME" in upper and "FATHER" not in upper:
            summary["Name"] = t.split(":")[-1].strip()
        if "FATHER" in upper:
            summary["Father's Name"] = t.split(":")[-1].strip()
        if dob := re.search(r"\d{2}[-/]\d{2}[-/]\d{4}", t):
            summary["DOB"] = dob.group()
        if re.match(r"^\d{4}\s\d{4}\s\d{4}$", t) or \
           re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", t) or \
           re.match(r"^[A-Z]{2}\d{2}[0-9A-Z]{11,}$", t):
            summary["Number"] = t
        if len(t) > 2:
            summary["Other Details"].append(t)
    return summary

def process_document(image_path):
    image = cv2.imread(image_path)
    blocks = extract_text_blocks(image)
    full_text = " ".join(blocks).upper()
    doc_type = classify_document(full_text)
    doc_side = detect_side(doc_type, blocks)
    summary = generate_summary(blocks, doc_type)

    unique_id = str(uuid.uuid4())[:8]
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    result_img = f"{base_name}_{unique_id}_output.jpg"
    result_json = f"{base_name}_{unique_id}.json"

    cv2.imwrite(os.path.join(app.config["OUTPUT_FOLDER"], result_img), image)
    output_data = {
        "filename": os.path.basename(image_path),
        "raw_detected_text": blocks,
        "cleaned_summary": summary,
        "document_type": doc_type,
        "document_side": doc_side,
        "processed_img_base64": img_to_base64(image),
        "processed_image": result_img
    }
    save_json(output_data, os.path.join(app.config["OUTPUT_FOLDER"], result_json))
    return output_data

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index(): return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file provided"})
    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        return jsonify({"message": "File uploaded successfully", "filename": filename})
    return jsonify({"error": "File type not allowed"})

@app.route("/extract/<filename>")
def extract_text(filename):
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(path): return jsonify({"error": "File not found"})
    try: return jsonify(process_document(path))
    except Exception as e: return jsonify({"error": f"Processing failed: {e}"})

@app.route("/uploads/<filename>")
def uploaded_file(filename): return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/outputs/<filename>")
def output_file(filename): return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
