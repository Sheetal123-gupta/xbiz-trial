import os, re, json, cv2, uuid
import numpy as np
import pytesseract
from pytesseract import Output
from rapidfuzz import fuzz
from matplotlib import pyplot as plt
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


# --- OCR Helpers ---
def correct_rotation(image):
    """Correct image rotation using Tesseract OSD."""
    try:
        angle = int(re.search(r"Rotate: (\d+)", pytesseract.image_to_osd(image)).group(1))
        rotate_map = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        if angle in rotate_map:
            image = cv2.rotate(image, rotate_map[angle])
    except Exception:
        pass
    return image


def correct_skew(image):
    """Deskew image using Hough transform."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(binary == 255) < np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)

    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

    if lines is not None:
        angles = [(theta * 180 / np.pi) - 90 for rho, theta in lines[:, 0] if -60 < (theta * 180 / np.pi) - 90 < 60]
        if angles:
            median_angle = np.median(angles)
            if abs(median_angle) > 0.5:
                h, w = image.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1)
                image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return image


def prepare_image_for_ocr(image):
    """Apply denoising + thresholding for OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 15)
    if np.sum(thresh == 255) > np.sum(thresh == 0):
        thresh = cv2.bitwise_not(thresh)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((1, 1), np.uint8))


def extract_text_and_boxes(image):
    """Extract OCR text with bounding boxes."""
    data = pytesseract.image_to_data(prepare_image_for_ocr(image),
                                     output_type=Output.DICT,
                                     lang="eng", config="--oem 3 --psm 6")
    lines, boxes = {}, []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        boxes.append({
            "text": word,
            "x_min": data["left"][i],
            "y_min": data["top"][i],
            "x_max": data["left"][i] + data["width"][i],
            "y_max": data["top"][i] + data["height"][i]
        })
    return "\n".join(" ".join(lines[k]) for k in sorted(lines)), boxes


def detect_document_type(text):
    """Detect type of document by regex + fuzzy matching."""
    text_low = text.lower()
    if re.search(r"\b\d{4} \d{4} \d{4}\b", text_low): return "Aadhaar Card"
    if re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text_low): return "PAN Card"
    if re.search(r"\b[A-Z]{2}/\d+/\d+/\d+\b", text_low): return "Voter ID"

    scores = {doc: sum(fuzz.partial_ratio(kw.lower(), text_low) >= FUZZY_THRESHOLD for kw in kws)
              for doc, kws in DOC_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= MIN_KEYWORD_MATCHES else "Unknown"


def draw_boxes(image, boxes, output_path):
    """Save annotated image with bounding boxes."""
    fig, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for b in boxes:
        ax.add_patch(Rectangle((b["x_min"], b["y_min"]),
                               b["x_max"] - b["x_min"], b["y_max"] - b["y_min"],
                               linewidth=1.5, edgecolor="lime", facecolor="none"))
    ax.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()


def log_error(txn_folder, msg):
    with open(os.path.join(txn_folder, "errorlogs", "errors.log"), "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} - {msg}\n")


def process_document(file_path, txn_id):
    """Main pipeline: preprocess → OCR → classify → save outputs."""
    filename = os.path.basename(file_path)
    txn_folder = os.path.join(BASE_FOLDER, f"{txn_id}_{os.path.splitext(filename)[0]}")

    try:
        # Create folders
        for folder in ["inputs", "assets", "outputs/text", "outputs/annotated", "errorlogs"]:
            os.makedirs(os.path.join(txn_folder, folder), exist_ok=True)
        

        # Move uploaded file
        input_path = os.path.join(txn_folder, "inputs", filename)
        os.rename(file_path, input_path)

        results=[]
        if filename.lower().endswith(".pdf"):
            pages=convert_from_path(input_path)
            for page_num,page in enumerate(pages,start=1):
                page_path=os.path.join(txn_folder,"inputs",f"page{page_num}.png")
                page.save(page_path,"PNG")

                # Preprocess
                img = cv2.imread(input_path)
                img = correct_rotation(correct_skew(img))

                # OCR
                text, boxes = extract_text_and_boxes(img)
                doc_type = detect_document_type(text)

        # Save annotated image
                annotated_path = os.path.join(txn_folder, "outputs/annotated", f"{filename}_annotated.png")
                draw_boxes(img, boxes, annotated_path)

                # Save extracted text
                text_path = os.path.join(txn_folder, "outputs/text", f"{filename}_extracted.txt")
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(f"Document Type: {doc_type}\n\n{text}")
                
                results.append({
                    "page_num":page_num,
                    "document type":doc_type,
                    "text":text,
                    "annotated_image":annotated_path,
                    "tex_file":text_path
                })
        else:
            #single image
            img = cv2.imread(input_path)
            img = correct_rotation(correct_skew(img))

            text, boxes = extract_text_and_boxes(img)
            doc_type = detect_document_type(text)

            annotated_path = os.path.join(txn_folder, "outputs/annotated", f"{filename}_annotated.png")
            draw_boxes(img, boxes, annotated_path)

            text_path = os.path.join(txn_folder, "outputs/text", f"{filename}_extracted.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(f"Document Type: {doc_type}\n\n{text}")

            results.append({
                "page_num": 1,
                "document_type": doc_type,
                "text": text,
                "annotated_image": annotated_path,
                "text_file": text_path
            })


        # Save request/response JSON
        with open(os.path.join(txn_folder, "assets", "request.json"), "w", encoding="utf-8") as f:
            json.dump({"txnId": txn_id, "docType": os.path.splitext(filename)[1],
                       "source": "OCR_RAW", "documentName": filename, "caseNo": "case001"},
                      f, indent=2, ensure_ascii=False)

        # Save assets (request + aggregated response)
        with open(os.path.join(txn_folder, "assets", "request.json"), "w", encoding="utf-8") as f:
            json.dump({"txnId": txn_id, "documentName": filename, "source": "OCR_RAW", "caseNo": "case001"},
                      f, indent=2, ensure_ascii=False)

        with open(os.path.join(txn_folder, "assets", "response.json"), "w", encoding="utf-8") as f:
            json.dump({"documentName": filename, "pages": results}, f, indent=2, ensure_ascii=False)

        return {"txn_id": txn_id, "results": results, "txn_folder": txn_folder}

        

    except Exception as e:
        log_error(txn_folder, str(e))
        return {"error": str(e)}
