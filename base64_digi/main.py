from flask import Flask, request, jsonify
import os
import uuid
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re
from rapidfuzz import fuzz
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Setup Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- Document Type Keywords ---
DOC_KEYWORDS = {
    "PAN Card": ["income tax department", "permanent account number", "आयकर विभाग", "भारत सरकार"],
    "Aadhaar Card": ["aadhaar", "आधार", "unique identification", "year of birth", "जन्म", "सत्यमेव जयते"],
    "Voter ID": ["election commission", "voter", "epic", "पहचान पत्र", "identity card"],
    "Bank Document": ["bank", "ifsc", "branch", "account"],
    "Driving License": ["driving licence", "transport department", "license number"]
}

FUZZY_THRESHOLD = 55
MIN_KEYWORD_MATCHES = 2

# --- Correct rotation ---
def correct_rotation(image):
    rotation_applied = False
    try:
        osd = pytesseract.image_to_osd(image)
        angle = int(re.search(r"Rotate: (\d+)", osd).group(1))
        if angle != 0:
            rotation_applied = True
            if angle == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception:
        pass
    return image, rotation_applied

# --- Correct skew ---
def correct_skew(image):
    skew_applied = False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(binary == 255) < np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)

    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)

    if lines is not None:
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            if -60 < angle < 60:
                angles.append(angle)

        if angles:
            median_angle = np.median(angles)
            if abs(median_angle) > 0.5:
                skew_applied = True
                (h, w) = image.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return image, skew_applied

# --- OCR preparation ---
def prepare_image_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(filtered, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 15)
    if np.sum(thresh == 255) > np.sum(thresh == 0):
        thresh = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return cleaned

# --- Preprocess ---
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found.")
    img, _ = correct_rotation(img)
    img, _ = correct_skew(img)
    return img

# --- OCR extraction ---
def extract_text_and_boxes(image):
    ocr_ready = prepare_image_for_ocr(image)
    data = pytesseract.image_to_data(
        ocr_ready,
        output_type=Output.DICT,
        lang="eng",
        config="--oem 3 --psm 6"
    )
    boxes = []
    lines = {}
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word or not word.isascii():
            continue
        block, par, line = data["block_num"][i], data["par_num"][i], data["line_num"][i]
        key = (block, par, line)
        lines.setdefault(key, []).append(word)
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        boxes.append({
            "text": word,
            "x_min": x, "y_min": y,
            "x_max": x + w, "y_max": y + h
        })
    full_text = "\n".join([" ".join(lines[k]) for k in sorted(lines.keys())])
    return full_text, boxes

# --- Document type detection ---
def detect_document_type(text):
    text = text.lower()
    scores = {}
    for doc_type, keywords in DOC_KEYWORDS.items():
        score = sum(1 for kw in keywords if fuzz.partial_ratio(kw.lower(), text) >= FUZZY_THRESHOLD)
        scores[doc_type] = score
    if re.search(r"\b\d{4} \d{4} \d{4}\b", text):
        return "Aadhaar Card"
    if re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text):
        return "PAN Card"
    if re.search(r"\b[A-Z]{2}/\d+/\d+/\d+\b", text):
        return "Voter ID"
    best_match = max(scores, key=scores.get)
    return best_match if scores[best_match] >= MIN_KEYWORD_MATCHES else "Unknown"

# --- Draw boxes ---
def draw_boxes(image, boxes, output_path):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img_rgb)
    for idx, b in enumerate(boxes, start=1):
        rect = Rectangle((b["x_min"], b["y_min"]),
                         b["x_max"] - b["x_min"],
                         b["y_max"] - b["y_min"],
                         linewidth=1.5, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(b["x_min"], b["y_min"] - 5, f"{idx}. {b['text']}",
                fontsize=6, color="yellow",
                bbox=dict(facecolor="black", alpha=0.4, pad=1),
                verticalalignment="bottom")
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

# --- Document processor ---
def process_document(image_path):
    image = preprocess_image(image_path)
    text, boxes = extract_text_and_boxes(image)
    doc_type = detect_document_type(text)

    base = os.path.splitext(os.path.basename(image_path))[0]

    # Save annotated image
    annotated_path = f"{base}_annotated.png"
    annotated_full = os.path.join(UPLOAD_FOLDER, annotated_path)
    draw_boxes(image, boxes, annotated_full)

    # Save extracted text
    text_filename = f"{base}_extracted.txt"
    text_fullpath = os.path.join(UPLOAD_FOLDER, text_filename)
    with open(text_fullpath, "w", encoding="utf-8") as f:
        f.write(f"Document Type: {doc_type}\n\n{text}")

    return {
        "document_type": doc_type,
        "extracted_text": text,
        "annotated_image": annotated_full,
        "text_file": text_fullpath
    }

# --- API Route ---
@app.route("/extract", methods=["POST"])
def extract_api():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file:
        return jsonify({"error": "Empty file"}), 400

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        result = process_document(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Run ---
if __name__ == "__main__":
    app.run(debug=True)
