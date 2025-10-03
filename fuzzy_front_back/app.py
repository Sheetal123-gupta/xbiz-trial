import os, re, cv2, json, time
import numpy as np
import pytesseract
from flask import Flask, jsonify
from rapidfuzz import fuzz

# -----------------------------
# Flask & Folders Setup
# -----------------------------
app = Flask(__name__)
BASE_DIR = os.path.dirname(__file__)
INPUT_FOLDER = os.path.join(BASE_DIR, "input_images")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# Keyword Dictionary & Patterns
# -----------------------------
DOCUMENT_KEYWORDS = {
    "PAN Card": ["income tax department", "permanent account number", "govt of india", "father's name"],
    "Aadhaar Card": ["aadhaar", "uidai", "government of india", "year of birth", "date of birth", "gender"],
    "Voter ID Card": ["election commission of india", "voter id", "elector's photo identity card", "epic"],
    "Passport": ["passport", "republic of india", "place of birth", "date of issue", "date of expiry"],
    "Driving License": ["driving license", "dl no", "valid till", "transport", "date of issue", "dob"],
    "Bank Passbook": ["account number", "ifsc", "branch", "customer id", "balance", "transaction", "a/c"]
}

PATTERNS = {
    "PAN": re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$"),
    "AADHAAR": re.compile(r"^\d{4}\s\d{4}\s\d{4}$"),
    "AADHAAR_NOSPACE": re.compile(r"^\d{12}$"),
    "DL": re.compile(r"^[A-Z]{2}\d{2}[0-9A-Z]{11,}$"),
    "VOTER": re.compile(r"^[A-Z]{3}[0-9]{7}$"),
    "PASSPORT": re.compile(r"^[A-Z][0-9]{7}$"),
    "IFSC": re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")
}

# -----------------------------
# Core Functions
# -----------------------------
def classify_document(blocks):
    text = " ".join(blocks).lower()
    scores = {doc: max(fuzz.partial_ratio(text, kw.lower()) for kw in kws)
              for doc, kws in DOCUMENT_KEYWORDS.items()}
    return max(scores, key=scores.get), scores

def classify_side(doc_type, blocks):
    text = " ".join(blocks).upper()

    def match_any(indicators, threshold=70):
        return any(fuzz.partial_ratio(text, ind.upper()) >= threshold for ind in indicators)

    side_rules = {
        "PAN Card": (["INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER", "GOVT OF INDIA"], ["QR CODE", "NSDL", "UTIITSL"]),
        "Aadhaar Card": (["GOVERNMENT OF INDIA", "AADHAAR", "UIDAI", "DOB", "GENDER"], ["ADDRESS", "DISTRICT", "STATE", "PIN"]),
        "Voter ID Card": (["ELECTION COMMISSION", "PHOTO IDENTITY CARD", "NAME", "FATHER"], ["ADDRESS", "DISTRICT", "STATE", "PIN CODE"]),
        "Passport": (["PASSPORT", "REPUBLIC OF INDIA", "NATIONALITY"], ["ADDRESS", "EMERGENCY CONTACT", "PLACE OF ISSUE"]),
        "Bank Passbook": (["ACCOUNT NUMBER", "IFSC", "BRANCH"], ["DEPOSIT", "WITHDRAWAL", "BALANCE", "CHEQUE"])
    }

    if doc_type in side_rules:
        front, back = side_rules[doc_type]
        if match_any(front) or any(p.match(b.replace(" ", "")) for p in PATTERNS.values() for b in blocks):
            return "Front"
        if match_any(back): return "Back"
    return "Unknown"

def generate_summary(blocks, doc_type):
    summary = {"Document": doc_type, "Name": None, "Father’s Name": None,
               "DOB": None, "Number": None, "Issuing Authority": None, "Other Details": []}

    for text in blocks:
        clean, upper = text.strip(), text.strip().upper()
        if "ELECTION COMMISSION" in upper: summary["Issuing Authority"] = "Election Commission of India"
        if "GOVERNMENT OF INDIA" in upper: summary["Issuing Authority"] = "Government of India"
        if "NAME" in upper and "FATHER" not in upper: summary["Name"] = clean.split(":")[-1].strip()
        if "FATHER" in upper: summary["Father’s Name"] = clean.split(":")[-1].strip()
        if dob := re.search(r"\d{2}[-/]\d{2}[-/]\d{4}", clean): summary["DOB"] = dob.group()
        if any(p.match(clean.replace(" ", "")) for p in PATTERNS.values()): summary["Number"] = clean
        if len(clean) > 2: summary["Other Details"].append(clean)
    return summary

def process_document(image_path):
    image = cv2.imread(image_path)
    if image is None: return {"error": f"Could not read image: {image_path}"}

    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dilated = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract text blocks
    blocks = []
    for c in sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1]):
        x, y, w, h = cv2.boundingRect(c)
        roi = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, config="--psm 6").strip()
        if text: blocks.append(text)

    doc_type, scores = classify_document(blocks)
    side = classify_side(doc_type, blocks)
    summary = generate_summary(blocks, doc_type)

    # Save outputs
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(image_path))[0]
    result = {"filename": os.path.basename(image_path),
              "raw_detected_text": blocks, "cleaned_summary": summary,
              "document_type": doc_type, "side": side, "fuzzy_scores": scores}
    with open(os.path.join(OUTPUT_FOLDER, f"{base}_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{base}_{ts}_output.jpg"), image)
    return result

# -----------------------------
# Flask Routes
# -----------------------------
@app.route('/')
def home(): return "Flask OCR API with fuzzy classification is running!"

@app.route('/process-all')
def process_all_files():
    results = [process_document(os.path.join(INPUT_FOLDER, f))
               for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return jsonify(results)

@app.route('/process/<filename>')
def process_single_file(filename):
    path = os.path.join(INPUT_FOLDER, filename)
    return jsonify(process_document(path) if os.path.exists(path)
                   else {"error": f"{filename} not found in {INPUT_FOLDER}"})
    
if __name__ == "__main__":
    app.run(debug=True)
