import os, cv2, re, json, uuid
import numpy as np
import pytesseract
from pytesseract import Output
from rapidfuzz import fuzz
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime

# --- Constants ---
BASE_FOLDER = "transactions"
os.makedirs(BASE_FOLDER, exist_ok=True)

DOC_KEYWORDS = {
    "PAN Card": ["income tax department", "permanent account number", "आयकर विभाग", "भारत सरकार"],
    "Aadhaar Card": ["aadhaar", "आधार", "unique identification", "year of birth", "जन्म", "सत्यमेव जयते"],
    "Voter ID": ["election commission", "voter", "epic", "पहचान पत्र", "identity card"],
    "Bank Document": ["bank", "ifsc", "branch", "account"],
    "Driving License": ["driving licence", "transport department", "license number","Authorization","Authorisation to drive"]
}
FUZZY_THRESHOLD = 55
MIN_KEYWORD_MATCHES = 2


# --- OCR helpers ---
def correct_rotation(image):
    try:
        osd = pytesseract.image_to_osd(image)
        angle = int(re.search(r"Rotate: (\d+)", osd).group(1))
        if angle != 0:
            if angle == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception as e:
        print(f"[WARN] Rotation detection failed: {e}")
    return image


def correct_skew(image):
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
                (h, w) = image.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h),
                                       flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return image


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


def extract_text_and_boxes(image):
    ocr_ready = prepare_image_for_ocr(image)
    data = pytesseract.image_to_data(
        ocr_ready, output_type=Output.DICT, lang="eng", config='--oem 3 --psm 6'
    )
    boxes, lines = [], {}
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word:
            continue
        block, par, line = data["block_num"][i], data["par_num"][i], data["line_num"][i]
        key = (block, par, line)
        lines.setdefault(key, []).append(word)
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        boxes.append({"text": word, "x_min": x, "y_min": y, "x_max": x + w, "y_max": y + h})
    full_text_lines = [' '.join(lines[k]) for k in sorted(lines.keys())]
    return "\n".join(full_text_lines), boxes


def detect_document_type(text):
    text_low = text.lower()
    scores = {doc: sum(1 for kw in kws if fuzz.partial_ratio(kw.lower(), text_low) >= FUZZY_THRESHOLD)
              for doc, kws in DOC_KEYWORDS.items()}
    if re.search(r"\b\d{4} \d{4} \d{4}\b", text_low): return "Aadhaar Card"
    if re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text_low): return "PAN Card"
    if re.search(r"\b[A-Z]{2}/\d+/\d+/\d+\b", text_low): return "Voter ID"
    best = max(scores, key=scores.get)
    return best if scores[best] >= MIN_KEYWORD_MATCHES else "Unknown"


def draw_boxes(image, boxes, output_path):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img_rgb)
    for b in boxes:
        rect = Rectangle((b["x_min"], b["y_min"]),
                         b["x_max"] - b["x_min"], b["y_max"] - b["y_min"],
                         linewidth=1.5, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
    ax.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()


def log_error(txn_folder, msg):
    err_file = os.path.join(txn_folder, "errorlogs", "errors.log")
    with open(err_file, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} - {msg}\n")


def process_document(file_path, txn_id):
    filename = os.path.basename(file_path)
    txn_folder = os.path.join(BASE_FOLDER, f"{txn_id}_{os.path.splitext(filename)[0]}")

    try:
        # 1. Create all required folders
        os.makedirs(os.path.join(txn_folder, "inputs"), exist_ok=True)
        os.makedirs(os.path.join(txn_folder, "assets"), exist_ok=True)
        os.makedirs(os.path.join(txn_folder, "outputs", "text"), exist_ok=True)
        os.makedirs(os.path.join(txn_folder, "outputs", "annotated"), exist_ok=True)
        os.makedirs(os.path.join(txn_folder, "errorlogs"), exist_ok=True)

        # 2. Move uploaded file into inputs
        input_path = os.path.join(txn_folder, "inputs", filename)
        os.rename(file_path, input_path)

        # 3. Read + preprocess
        img = cv2.imread(input_path)
        img = correct_rotation(img)
        img = correct_skew(img)

        # 4. OCR
        text, boxes = extract_text_and_boxes(img)
        doc_type = detect_document_type(text)

        # 5. Save annotated image
        annotated_path = os.path.join(txn_folder, "outputs", "annotated", f"{filename}_annotated.png")
        draw_boxes(img, boxes, annotated_path)

        # 6. Save extracted text
        text_path = os.path.join(txn_folder, "outputs", "text", f"{filename}_extracted.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(f"Document Type: {doc_type}\n\n{text}")

        # 7. Save assets (request + response JSONs)
        req_data = {
            "txnId": txn_id,
            "docType": os.path.splitext(filename)[1],
            "source": "OCR_RAW",
            "documentName": filename,
            "caseNo": "case001"
        }
        with open(os.path.join(txn_folder, "assets", "request.json"), "w", encoding="utf-8") as f:
            json.dump(req_data, f, indent=2, ensure_ascii=False)

        res_data = {
            "documentName": filename,
            "docType": os.path.splitext(filename)[1],
            "extractedText": text,
            "documentTypeDetected": doc_type
        }
        with open(os.path.join(txn_folder, "assets", "response.json"), "w", encoding="utf-8") as f:
            json.dump(res_data, f, indent=2, ensure_ascii=False)

        return {
            "document_type": doc_type,
            "text": text,
            "annotated_image": annotated_path,
            "text_file": text_path
        }

    except Exception as e:
        # Log to errorlogs/errors.log
        os.makedirs(os.path.join(txn_folder, "errorlogs"), exist_ok=True)
        log_error(txn_folder, str(e))

        return {"error": str(e)}
